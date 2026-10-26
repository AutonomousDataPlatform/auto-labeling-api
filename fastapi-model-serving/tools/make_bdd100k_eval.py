# 사용법: python make_bdd100k_eval.py <BDD100K 루트(images/100k, labels/100k 포함)> <출력 폴더>
# 결과물: Eval/BDD100K/{images/, weather_time_gt.csv, val_list.txt, checksums.md5}
"""BDD100K 공식 val 10,000장으로 날씨·시간 분류 평가셋을 만든다.

- 이미지: images/100k/val 전체를 그대로 복사하고, 원본과 복사본을 각각 읽어 MD5를 대조
- GT: labels/100k/val/<이름>.json 의 attributes.weather / timeofday 를 모델 클래스명으로 바꿔 CSV로 저장
  날씨 clear·partly cloudy → Clear, overcast → Overcast, rainy → Rainy, snowy → Snowy
  시간 daytime → Daytime, night → Night
  그 밖의 값(foggy, undefined, dawn/dusk)은 칸을 비워 평가에서 제외하고 원본 값은 bdd_* 열에 남긴다
"""
import csv, hashlib, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor

ROOT, OUT = sys.argv[1], sys.argv[2]
IMG_SRC, LBL_SRC = f"{ROOT}/images/100k/val", f"{ROOT}/labels/100k/val"
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


def copy_one(name):
    """복사 후 원본·복사본 MD5를 각각 계산해 돌려준다. 이미 같은 파일이 있으면 다시 쓰지 않는다."""
    data = retry(read_bytes, f"{IMG_SRC}/{name}")
    src_md5 = hashlib.md5(data).hexdigest()
    dst = f"{OUT}/images/{name}"
    if not (os.path.exists(dst) and os.path.getsize(dst) == len(data)):
        with open(dst + ".part", "wb") as f:
            f.write(data)
        os.replace(dst + ".part", dst)
    return name, src_md5, hashlib.md5(read_bytes(dst)).hexdigest()


def attributes(stem):
    a = json.loads(retry(read_bytes, f"{LBL_SRC}/{stem}.json"))["attributes"]
    return stem, a.get("weather", ""), a.get("timeofday", "")


os.makedirs(f"{OUT}/images", exist_ok=True)
names = sorted(n for n in retry(os.listdir, IMG_SRC) if n.endswith(".jpg"))
stems = [n[:-4] for n in names]
labels = set(n[:-5] for n in retry(os.listdir, LBL_SRC) if n.endswith(".json"))
missing = sorted(set(stems) - labels)
assert not missing, f"라벨 없는 이미지 {len(missing)}장: {missing[:5]}"
log(f"이미지 {len(names)}장, 라벨 {len(labels)}개")

with ThreadPoolExecutor(8) as ex:
    attrs = list(ex.map(attributes, stems))
rows = [{"filename": f"{s}.jpg", "weather": WEATHER.get(w, ""), "time": TIME.get(t, ""),
         "bdd_weather": w, "bdd_timeofday": t} for s, w, t in attrs]
with open(f"{OUT}/weather_time_gt.csv", "w", newline="", encoding="utf-8") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
with open(f"{OUT}/val_list.txt", "w", encoding="utf-8") as f:
    f.write("".join(f"{n}\n" for n in names))
n_w = sum(bool(r["weather"]) for r in rows)
n_t = sum(bool(r["time"]) for r in rows)
log(f"GT CSV 저장: 날씨 평가 대상 {n_w}장, 시간 평가 대상 {n_t}장")

sums, bad = {}, []
with ThreadPoolExecutor(8) as ex:
    for i, (name, src_md5, dst_md5) in enumerate(ex.map(copy_one, names), 1):
        sums[f"images/{name}"] = dst_md5
        if src_md5 != dst_md5:
            bad.append(name)
        if i % 1000 == 0:
            log(f"복사 {i}/{len(names)}")
assert not bad, f"MD5 불일치 {len(bad)}장: {bad[:5]}"
for rel in ("weather_time_gt.csv", "val_list.txt"):
    sums[rel] = hashlib.md5(read_bytes(f"{OUT}/{rel}")).hexdigest()
with open(f"{OUT}/checksums.md5", "w", encoding="utf-8") as f:
    f.write("".join(f"{v}  {k}\n" for k, v in sorted(sums.items())))
log(f"완료: 이미지 {len(names)}장 복사·MD5 대조 불일치 0, checksums.md5 {len(sums)}개")
