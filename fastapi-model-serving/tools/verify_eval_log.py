# 사용법: python verify_eval_log.py <로그 폴더>   (웹 UI [정확도 측정] 때 저장된 Eval_log/<데이터셋>_<시각>)
"""건별 비교 로그(*_log.csv)만으로 정확도를 다시 계산해 report.json의 값과 대조한다.
추론 로그(inference_log.txt)가 있으면 추론 성공 줄 수와 줄별 판정의 합계도 대조한다.

정확도 = 정답으로 판정된 생성 라벨 수 / 판정한 생성 라벨 수
  detection_log.csv  kind=생성 줄 중 result=TP / (TP + FP)        (IGN은 무시 영역이라 판정에서 제외)
  weather_log.csv    result=1 / (result가 0 또는 1)               (빈칸은 GT가 없어 판정에서 제외)
  time_log.csv       위와 같음
  lanes_log.csv      kind=생성 줄 중 result=TP / kind=생성 줄 수
모든 항목이 일치하면 종료 코드 0, 하나라도 다르면 1.
"""
import csv, json, os, sys

LOG_DIR = sys.argv[1]
report = json.load(open(os.path.join(LOG_DIR, "report.json"), encoding="utf-8"))


def rows(name):
    path = os.path.join(LOG_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def generated_tp(log, judged_results):
    gen = [r for r in log if r["kind"] == "생성"]
    tp = sum(r["result"] == "TP" for r in gen)
    return tp, sum(r["result"] in judged_results for r in gen)


checks = []
for key, name, count in (("detection", "detection_log.csv", lambda log: generated_tp(log, ("TP", "FP"))),
                         ("weather", "weather_log.csv", lambda log: (sum(r["result"] == "1" for r in log),
                                                                    sum(r["result"] in ("0", "1") for r in log))),
                         ("time", "time_log.csv", lambda log: (sum(r["result"] == "1" for r in log),
                                                              sum(r["result"] in ("0", "1") for r in log))),
                         ("lanes", "lanes_log.csv", lambda log: generated_tp(log, ("TP", "FP")))):
    log = rows(name)
    if log is None or key not in report:
        continue
    correct, judged = count(log)
    recomputed = correct / judged if judged else 0.0
    checks.append((key, correct, judged, recomputed, report[key]["accuracy"]))

print(f"로그 폴더: {LOG_DIR}  (데이터셋 {report.get('dataset')}, 이미지 {report.get('n_images')}장, 측정 {report.get('created')})")
print(f"{'항목':10s} {'정답':>8s} {'판정':>8s} {'로그로 계산한 정확도':>18s} {'리포트 정확도':>12s}  일치")
ok = True
for key, correct, judged, recomputed, reported in checks:
    same = abs(recomputed - reported) < 1e-12
    ok &= same
    print(f"{key:10s} {correct:8,d} {judged:8,d} {recomputed * 100:17.4f}% {reported * 100:11.4f}%  {'O' if same else 'X'}")
# 추론 로그(inference_log.txt): 추론 한 건마다 한 줄. 줄 수와 줄별 판정의 합이 리포트와 같은지 본다
inf_path = os.path.join(LOG_DIR, "inference_log.txt")
if os.path.exists(inf_path):
    import re
    lines = open(inf_path, encoding="utf-8").read().splitlines()
    n_ok = sum(" 추론 성공 " in l for l in lines)
    n_fail = sum(" 추론 실패 " in l for l in lines)
    same_n = n_ok == report.get("n_images")
    ok &= same_n
    print(f"추론 로그: 추론 성공 {n_ok:,}줄, 실패 {n_fail:,}줄 (리포트 이미지 {report.get('n_images'):,}장)  {'O' if same_n else 'X'}")
    sums = {}
    for l in lines:
        if " 추론 성공 " not in l:
            continue
        m = re.search(r"정답\(TP\) (\d+), 오답\(FP\) (\d+)", l)
        if m:
            key = "lanes" if "출력 차선" in l else "detection"
            c, j = sums.get(key, (0, 0)); sums[key] = (c + int(m.group(1)), j + int(m.group(1)) + int(m.group(2)))
        for key, name in (("weather", "날씨"), ("time", "시간")):
            m = re.search(rf"판정 .*?{name} (정답|오답)", l)
            if m:
                c, j = sums.get(key, (0, 0)); sums[key] = (c + (m.group(1) == "정답"), j + 1)
    for key, correct, judged, _, _ in checks:
        c, j = sums.get(key, (0, 0))
        same = (c, j) == (correct, judged)
        ok &= same
        print(f"추론 로그 줄별 판정 합계 {key:10s} 정답 {c:,} / 판정 {j:,}  (건별 비교 로그 {correct:,} / {judged:,})  {'O' if same else 'X'}")
print("결과: 모두 일치" if ok and checks else "결과: 불일치 또는 확인할 로그 없음")
sys.exit(0 if ok and checks else 1)
