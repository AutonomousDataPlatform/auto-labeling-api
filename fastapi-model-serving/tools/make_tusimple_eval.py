# 사용법: python make_tusimple_eval.py <TuSimple test 루트(clips/ 포함)> <test_label.json> <출력 폴더>
# 결과물: Eval/TuSimple/{images/, gt_lanes.json, test_label.json, README.txt, checksums.md5}
"""TuSimple 공식 test의 라벨된 프레임 2,782장으로 차선 인지 평가셋을 만든다.

- 이미지: test_label.json의 raw_file(clips/<날짜>/<클립ID>/20.jpg)을 복사하고 원본·복사본 MD5를 대조
  파일명이 모두 20.jpg라 <날짜>_<클립ID>.jpg로 바꾼다
- GT: test_label.json의 lanes·h_samples를 그대로 두고 raw_file만 바뀐 파일명으로 쓴 gt_lanes.json
"""
import hashlib, json, os, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor

ROOT, LABEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)


def flat(raw_file):
    """clips/0530/<클립ID>/20.jpg -> 0530_<클립ID>.jpg"""
    return "_".join(raw_file.split("/")[1:3]) + ".jpg"


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def copy_one(rec):
    """복사 후 원본·복사본 MD5를 각각 계산해 돌려준다. 이미 같은 파일이 있으면 다시 쓰지 않는다."""
    data = read_bytes(os.path.join(ROOT, rec["raw_file"]))
    src_md5 = hashlib.md5(data).hexdigest()
    dst = f"{OUT}/images/{flat(rec['raw_file'])}"
    if not (os.path.exists(dst) and hashlib.md5(read_bytes(dst)).hexdigest() == src_md5):
        with open(dst + ".part", "wb") as f:
            f.write(data)
        os.replace(dst + ".part", dst)
    return flat(rec["raw_file"]), src_md5, hashlib.md5(read_bytes(dst)).hexdigest()


labels = [json.loads(l) for l in open(LABEL, encoding="utf-8") if l.strip()]
missing = [r["raw_file"] for r in labels if not os.path.isfile(os.path.join(ROOT, r["raw_file"]))]
assert not missing, f"이미지 없는 라벨 {len(missing)}개: {missing[:3]}"
names = [flat(r["raw_file"]) for r in labels]
assert len(set(names)) == len(names), "바꾼 파일명이 겹침"
by_date = {}
for r in labels:
    by_date[r["raw_file"].split("/")[1]] = by_date.get(r["raw_file"].split("/")[1], 0) + 1
log(f"라벨 {len(labels)}개, 날짜별 {by_date}")

os.makedirs(f"{OUT}/images", exist_ok=True)
stale = sorted(set(os.listdir(f"{OUT}/images")) - set(names))
assert not stale, f"평가셋에 없는 이미지가 images/에 있음 {len(stale)}개: {stale[:3]}"

sums, bad = {}, []
with ThreadPoolExecutor(8) as ex:
    for i, (name, src_md5, dst_md5) in enumerate(ex.map(copy_one, labels), 1):
        sums[f"images/{name}"] = dst_md5
        if src_md5 != dst_md5:
            bad.append(name)
        if i % 500 == 0:
            log(f"복사 {i}/{len(labels)}")
assert not bad, f"MD5 불일치 {len(bad)}장: {bad[:5]}"

shutil.copyfile(LABEL, f"{OUT}/test_label.json")
with open(f"{OUT}/gt_lanes.json", "w", encoding="utf-8", newline="\n") as f:
    for r in labels:
        f.write(json.dumps({"raw_file": flat(r["raw_file"]), "orig_raw_file": r["raw_file"],
                            "lanes": r["lanes"], "h_samples": r["h_samples"]}) + "\n")
with open(f"{OUT}/README.txt", "w", encoding="utf-8", newline="\n") as f:
    f.write(f"""TuSimple lane detection 공식 test 평가셋
구성: test_label.json(원본 그대로)의 라벨된 프레임 {len(labels):,}장 ({', '.join(f'{k} {v:,}' for k, v in sorted(by_date.items()))})
images/          파일명 = <날짜>_<클립ID>.jpg  (원래 clips/<날짜>/<클립ID>/20.jpg, 모든 파일명이 20.jpg라 바꿈)
gt_lanes.json    TuSimple 형식 정답. raw_file은 바뀐 파일명, orig_raw_file은 원래 경로, lanes·h_samples는 원본 그대로
test_label.json  원본 정답 파일 (MD5 {hashlib.md5(read_bytes(LABEL)).hexdigest()})
checksums.md5    파일별 MD5 (재검증: md5sum -c checksums.md5)
평가 지표: TuSimple 공식 LaneEval (Accuracy, FP, FN)
생성 스크립트: auto-labeling-api 저장소 fastapi-model-serving/tools/make_tusimple_eval.py
""")
for rel in ("test_label.json", "gt_lanes.json", "README.txt"):
    sums[rel] = hashlib.md5(read_bytes(f"{OUT}/{rel}")).hexdigest()
with open(f"{OUT}/checksums.md5", "w", encoding="utf-8", newline="\n") as f:
    f.write("".join(f"{v}  {k}\n" for k, v in sorted(sums.items())))
log(f"완료: 이미지 {len(labels)}장 복사·MD5 대조 불일치 0, checksums.md5 {len(sums)}개")
