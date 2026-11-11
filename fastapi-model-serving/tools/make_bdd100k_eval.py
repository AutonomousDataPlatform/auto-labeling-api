# 사용법: python make_bdd100k_eval.py <BDD100K 루트(images/100k, labels/100k 포함)> <출력 폴더> [학습 목록.txt ...]
#                                      [--splits val,train] [--sample N] [--seed 42]
# 결과물: Eval/BDD100K/{images/, weather_time_gt.csv, val_list.txt, checksums.md5}
"""BDD100K로 날씨·시간 분류 평가셋을 만든다.

- 후보: --splits로 고른 BDD100K 공식 분할(기본 val)의 이미지
- 학습 목록(선택): 모델 학습에 쓴 분할 목록(각 줄 "<이미지 경로> <라벨>")을 주면 그 이미지를 후보에서 뺀다.
  배포 모델은 BDD100K train·val·test의 시내 도로(city street) 장면을 모두 학습에 썼으므로, 인증 평가셋은
  effnet/data/scene_as_split_no_foggy/train.txt(날씨)와 scene_as_split_time/train.txt(시간)를 줘서 만든다
- 표본(선택): --sample N이면 남은 후보에서 --seed로 N장을 무작위로 뽑는다
- 이미지: 고른 이미지를 그대로 복사하고, 원본과 복사본을 각각 읽어 MD5를 대조
- GT: labels/100k/<분할>/<이름>.json 의 attributes.weather / timeofday 를 모델 클래스명으로 바꿔 CSV로 저장
  날씨 clear·partly cloudy → Clear, overcast → Overcast, rainy → Rainy, snowy → Snowy
  시간 daytime → Daytime, night → Night
  그 밖의 값(foggy, undefined, dawn/dusk)은 칸을 비워 평가에서 제외하고 원본 값은 bdd_* 열에 남긴다
  bdd_scene·bdd_split 열에는 장면 유형(attributes.scene)과 공식 분할을 남겨 평가셋 구성을 확인할 수 있게 한다

인증 평가셋(2026-11-10):
  python make_bdd100k_eval.py <26_BDD-100K> <Eval/BDD100K> <data>/scene_as_split_no_foggy/train.txt \\
      <data>/scene_as_split_time/train.txt --splits val,train --sample 10000 --seed 42
"""
import argparse, csv, hashlib, json, os, random, time
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("out")
ap.add_argument("exclude_lists", nargs="*")
ap.add_argument("--splits", default="val", help="후보로 쓸 공식 분할, 쉼표로 구분 (예: val,train)")
ap.add_argument("--sample", type=int, default=0, help="0이면 후보 전체")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()
ROOT, OUT = args.root, args.out
WEATHER = {"clear": "Clear", "partly cloudy": "Clear", "overcast": "Overcast", "rainy": "Rainy", "snowy": "Snowy"}
TIME = {"daytime": "Daytime", "night": "Night"}
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)


def retry(fn, *a, tries=5):
    """sshfs의 일시적 ENOENT·EACCES를 몇 번 다시 시도한다."""
    for i in range(tries):
        try:
            return fn(*a)
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(1 + i)


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def copy_one(item):
    """복사 후 원본·복사본 MD5를 각각 계산해 돌려준다. 이미 같은 파일이 있으면 다시 쓰지 않는다."""
    split, name = item
    data = retry(read_bytes, f"{ROOT}/images/100k/{split}/{name}")
    src_md5 = hashlib.md5(data).hexdigest()
    dst = f"{OUT}/images/{name}"
    if not (os.path.exists(dst) and os.path.getsize(dst) == len(data)):
        with open(dst + ".part", "wb") as f:
            f.write(data)
        os.replace(dst + ".part", dst)
    return name, src_md5, hashlib.md5(read_bytes(dst)).hexdigest()


def attributes(item):
    split, name = item
    a = json.loads(retry(read_bytes, f"{ROOT}/labels/100k/{split}/{name[:-4]}.json"))["attributes"]
    return split, name, a.get("weather", ""), a.get("timeofday", ""), a.get("scene", "")


os.makedirs(f"{OUT}/images", exist_ok=True)
items = []                                       # (공식 분할, 파일명)
for split in args.splits.split(","):
    names = sorted(n for n in retry(os.listdir, f"{ROOT}/images/100k/{split}") if n.endswith(".jpg"))
    labels = set(n[:-5] for n in retry(os.listdir, f"{ROOT}/labels/100k/{split}") if n.endswith(".json"))
    missing = sorted(set(n[:-4] for n in names) - labels)
    assert not missing, f"{split}: 라벨 없는 이미지 {len(missing)}장: {missing[:5]}"
    items += [(split, n) for n in names]
    log(f"{split}: 이미지 {len(names)}장, 라벨 {len(labels)}개")
assert len({n for _, n in items}) == len(items), "공식 분할 사이에 같은 파일명이 있음"

trained = set()
for path in args.exclude_lists:
    with open(path, encoding="utf-8") as f:
        trained.update(os.path.basename(line.split()[0]) for line in f if line.strip())
if args.exclude_lists:
    items = [it for it in items if it[1] not in trained]
    log(f"학습 목록 {len(args.exclude_lists)}개({len(trained)}장)에 있는 이미지를 빼고 후보 {len(items)}장")
if args.sample:
    items = sorted(random.Random(args.seed).sample(items, args.sample), key=lambda it: it[1])
    log(f"시드 {args.seed}로 {len(items)}장 표본 추출")

with ThreadPoolExecutor(8) as ex:
    attrs = list(ex.map(attributes, items))
rows = [{"filename": n, "weather": WEATHER.get(w, ""), "time": TIME.get(t, ""),
         "bdd_weather": w, "bdd_timeofday": t, "bdd_scene": sc, "bdd_split": sp} for sp, n, w, t, sc in attrs]
with open(f"{OUT}/weather_time_gt.csv", "w", newline="", encoding="utf-8") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
with open(f"{OUT}/val_list.txt", "w", encoding="utf-8") as f:
    f.write("".join(f"{n}\n" for _, n in items))
n_w = sum(bool(r["weather"]) for r in rows)
n_t = sum(bool(r["time"]) for r in rows)
log(f"GT CSV 저장: 날씨 평가 대상 {n_w}장, 시간 평가 대상 {n_t}장")

sums, bad = {}, []
with ThreadPoolExecutor(8) as ex:
    for i, (name, src_md5, dst_md5) in enumerate(ex.map(copy_one, items), 1):
        sums[f"images/{name}"] = dst_md5
        if src_md5 != dst_md5:
            bad.append(name)
        if i % 1000 == 0:
            log(f"복사 {i}/{len(items)}")
assert not bad, f"MD5 불일치 {len(bad)}장: {bad[:5]}"
for rel in ("weather_time_gt.csv", "val_list.txt"):
    sums[rel] = hashlib.md5(read_bytes(f"{OUT}/{rel}")).hexdigest()
with open(f"{OUT}/checksums.md5", "w", encoding="utf-8") as f:
    f.write("".join(f"{v}  {k}\n" for k, v in sorted(sums.items())))
log(f"완료: 이미지 {len(items)}장 복사·MD5 대조 불일치 0, checksums.md5 {len(sums)}개")
