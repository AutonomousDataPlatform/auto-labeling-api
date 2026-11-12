# 사용법: python compare_data_hashes.py <출력 폴더> --eval 평가목록.txt --group 학습=학습목록.txt [--group 검증=검증목록.txt] [--cache 해시캐시.json]
#   목록 파일은 한 줄에 파일 경로 하나. 같은 파일을 여러 번 계산하지 않도록 --cache로 (경로 -> SHA-256)을 공유할 수 있다.
"""학습(·검증) 데이터와 평가 데이터의 파일 해시(SHA-256)를 비교해 서로 겹치지 않음을 보이는 증적을 만든다.

출력 폴더
  eval_sha256.txt            평가 데이터 파일별 SHA-256 ("<해시>  <경로>", sha256sum 형식)
  <그룹>_sha256.txt          학습·검증 데이터 파일별 SHA-256
  compare_result.txt         그룹별 파일 수, 전체 해시, 평가 데이터와 내용이 같은 파일 수와 목록
전체 해시: 파일별 SHA-256을 정렬해 한 줄에 하나씩 적은 텍스트의 SHA-256 (파일 순서·경로와 무관하게 데이터 묶음 하나를 나타낸다)
  셸에서 같은 값 계산: cut -d' ' -f1 <그룹>_sha256.txt | LC_ALL=C sort | sha256sum
내용이 같은 파일: 파일 이름이 달라도 해시가 같으면 같은 파일로 센다
"""
import argparse, hashlib, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--eval", required=True)
ap.add_argument("--group", action="append", required=True, help="이름=목록파일")
ap.add_argument("--cache")
args = ap.parse_args()
t0 = time.time()
cache = json.load(open(args.cache)) if args.cache and os.path.exists(args.cache) else {}


def sha256(path):
    if path in cache:
        return path, cache[path]
    for k in range(5):                         # 네트워크 드라이브의 일시적 읽기 오류는 다시 시도한다
        try:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            return path, h.hexdigest()
        except OSError:
            if k == 4:
                raise
            time.sleep(2 + k)


def hash_list(name, list_path):
    paths = sorted({l.strip() for l in open(list_path, encoding="utf-8") if l.strip()})
    out = {}
    with ThreadPoolExecutor(16) as ex:
        for i, (p, h) in enumerate(ex.map(sha256, paths), 1):
            out[p] = cache[p] = h
            if i % 5000 == 0:
                print(f"[{time.time() - t0:6.0f}s] {name} {i:,}/{len(paths):,}", flush=True)
    with open(os.path.join(args.out, f"{name}_sha256.txt"), "w", encoding="utf-8") as f:
        f.write("".join(f"{out[p]}  {p}\n" for p in paths))
    print(f"[{time.time() - t0:6.0f}s] {name} {len(paths):,}개 완료", flush=True)
    return out


def total_hash(hashes):
    return hashlib.sha256("".join(h + "\n" for h in sorted(hashes.values())).encode()).hexdigest()


os.makedirs(args.out, exist_ok=True)
ev = hash_list("eval", args.eval)
ev_by_hash = {}
for p, h in ev.items():
    ev_by_hash.setdefault(h, []).append(p)
lines = [f"{'·'.join(g.split('=', 1)[0] for g in args.group)} 데이터와 평가 데이터 해시 비교 (SHA-256)", "=" * 40,
         f"생성: {time.strftime('%Y-%m-%d %H:%M:%S')}",
         "전체 해시 = 파일별 SHA-256을 정렬한 목록의 SHA-256 (cut -d' ' -f1 <그룹>_sha256.txt | LC_ALL=C sort | sha256sum)", "",
         f"[평가] 파일 {len(ev):,}개 (고유 해시 {len(ev_by_hash):,}개)  전체 해시 {total_hash(ev)}", ""]
ok = True
for g in args.group:
    name, list_path = g.split("=", 1)
    hs = hash_list(name, list_path)
    common = sorted((h, p) for p, h in hs.items() if h in ev_by_hash)
    ok &= not common
    lines += [f"[{name}] 파일 {len(hs):,}개 (고유 해시 {len(set(hs.values())):,}개)  전체 해시 {total_hash(hs)}",
              f"  평가 데이터와 내용이 같은 파일: {len(common):,}개" + ("  → 서로 다름" if not common else "")]
    for h, p in common[:1000]:
        lines.append(f"    {h}  {p}  =  {', '.join(ev_by_hash[h])}")
    lines.append("")
lines.append("결과: 평가 데이터와 겹치는 파일 없음" if ok else "결과: 평가 데이터와 내용이 같은 파일 있음 (위 목록 참고)")
with open(os.path.join(args.out, "compare_result.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
if args.cache:
    json.dump(cache, open(args.cache, "w"))
print("\n".join(lines))
sys.exit(0 if ok else 1)
