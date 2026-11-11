#!/usr/bin/env bash
# 인증 시험 증적(시험 PC 기준)을 한 폴더에 모은다: SW 정보, 모델 SHA-256, 평가 데이터 폴더·파일 수, 평가 데이터 SHA-256.
# 사용법: bash tools/collect_evidence.sh <EVAL_DIR> [출력 폴더]
#   출력 폴더 기본값: <EVAL_DIR>와 같은 위치의 Evidence_<날짜시각>
#   서버를 켜지 않아도 된다. 평가 데이터 해시 계산에 몇 분 걸린다.
set -u
SERVING="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"     # fastapi-model-serving
REPO="$(dirname "$SERVING")"
EVAL_DIR="$(cd "${1:?사용법: bash tools/collect_evidence.sh <EVAL_DIR> [출력 폴더]}" && pwd)"
OUT="${2:-$(dirname "$EVAL_DIR")/Evidence_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT/sw" "$OUT/eval_sha256"
DATASETS=(nuScenes KITTI BDD100K TuSimple)
declare -A IMAGES=([nuScenes]=CAM_FRONT [KITTI]=image_2 [BDD100K]=images [TuSimple]=images)
CONDA_ENVS=(bigdata bigdata_clrnet)

# 모델 경로는 사전조건 스크립트의 목록을 그대로 쓴다 (경로·해시를 한 곳에서만 관리)
mapfile -t MODELS < <(sed -n '/^MODELS=(/,/^)/p' "$SERVING/tools/check_preconditions.sh" | grep -o '"[^"]*"' | tr -d '"')

conda_bin="$(command -v conda || true)"
[ -z "$conda_bin" ] && conda_bin="${CONDA_EXE:-}"
for c in "$HOME"/anaconda3/bin/conda "$HOME"/miniconda3/bin/conda "$HOME"/miniforge3/bin/conda; do
  [ -z "$conda_bin" ] && [ -x "$c" ] && conda_bin="$c"
done

# 라이브러리 버전 확인 코드 (conda run은 표준 입력을 넘기지 않으므로 -c로 넘긴다)
VERSIONS_PY=$(cat <<'PY'
import importlib, platform
print(f"    Python {platform.python_version()}")
for mod, name in [("torch", "PyTorch"), ("torchvision", "torchvision"), ("ultralytics", "Ultralytics"),
                  ("efficientnet_pytorch", "efficientnet-pytorch"), ("mmcv", "mmcv"), ("clrnet", "CLRNet"),
                  ("fastapi", "FastAPI"), ("uvicorn", "Uvicorn"), ("streamlit", "Streamlit"),
                  ("numpy", "NumPy"), ("cv2", "OpenCV"), ("PIL", "Pillow")]:
    try:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", "설치됨")
        extra = f" (CUDA {m.version.cuda}, GPU 사용 {m.cuda.is_available()})" if mod == "torch" else ""
        print(f"    {name} {v}{extra}")
    except Exception:
        pass
PY
)

echo "증적 폴더: $OUT"

# ---------------------------------------------------------------------------
echo "1/4 SW·HW 정보"
{
  echo "수집 시각  $(date '+%Y-%m-%d %H:%M:%S')"
  echo "호스트     $(hostname)"
  echo "운영체제   $(. /etc/os-release && echo "$PRETTY_NAME") / 커널 $(uname -r)"
  echo "CPU        $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //') x $(nproc)"
  echo "메모리     $(free -h | awk '/^Mem/ {print $2}')"
  command -v nvidia-smi >/dev/null &&
    echo "GPU        $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
  echo "저장소     $REPO"
  echo "커밋       $(git -C "$REPO" log -1 --format='%H %ad %s' --date=iso 2>/dev/null)"
  [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ] && echo "           (커밋하지 않은 변경 있음)"
  echo "conda      ${conda_bin:-없음} $("${conda_bin:-false}" --version 2>/dev/null)"
  echo
  echo "[개발 언어·프레임워크·주요 라이브러리 버전]"
  for e in "${CONDA_ENVS[@]}"; do
    echo "- conda 환경 $e"
    "$conda_bin" run -n "$e" python -c "$VERSIONS_PY" 2>/dev/null
  done
} > "$OUT/sw/sw_info.txt" 2>&1
for e in "${CONDA_ENVS[@]}"; do
  "$conda_bin" env export -n "$e" > "$OUT/sw/conda_env_$e.yml" 2>/dev/null
  "$conda_bin" run -n "$e" python -m pip freeze > "$OUT/sw/pip_freeze_$e.txt" 2>/dev/null
done
cp "$SERVING"/fastapi/requirements.txt "$OUT/sw/requirements_fastapi.txt" 2>/dev/null
cp "$SERVING"/streamlit/requirements.txt "$OUT/sw/requirements_streamlit.txt" 2>/dev/null
cp "$SERVING"/LaneDetection/requirements.txt "$OUT/sw/requirements_lane.txt" 2>/dev/null
cat "$OUT/sw/sw_info.txt"

# ---------------------------------------------------------------------------
echo; echo "2/4 모델 SHA-256"
: > "$OUT/model_sha256.txt"
for m in "${MODELS[@]}"; do
  IFS='|' read -r name path want <<< "$m"
  got="$(sha256sum "$SERVING/$path" 2>/dev/null | cut -d' ' -f1)"
  printf '%s  %s  (%s, 기대값과 %s)\n' "${got:-파일없음}" "$path" "$name" "$([ "$got" = "$want" ] && echo 일치 || echo 불일치)" \
    | tee -a "$OUT/model_sha256.txt"
done

# ---------------------------------------------------------------------------
echo; echo "3/4 평가 데이터 폴더·파일 수"
{
  echo "평가 데이터 $EVAL_DIR"
  for d in "${DATASETS[@]}"; do
    echo "[$d]  전체 파일 $(find "$EVAL_DIR/$d" -type f | wc -l)개"
    find "$EVAL_DIR/$d" -mindepth 1 -maxdepth 1 | sort | while read -r p; do
      if [ -d "$p" ]; then echo "  $(basename "$p")/  파일 $(find "$p" -type f | wc -l)개"
      else echo "  $(basename "$p")  $(stat -c %s "$p") bytes"; fi
    done
  done
} | tee "$OUT/eval_counts.txt"

# ---------------------------------------------------------------------------
echo; echo "4/4 평가 데이터 SHA-256 (데이터셋마다 몇 분)"
{
  echo "평가 데이터 SHA-256 (파일별 목록: eval_sha256/<데이터셋>_sha256.txt, 경로는 데이터셋 폴더 기준)"
  echo "전체 해시 = 파일별 SHA-256을 정렬한 목록의 SHA-256 (cut -d' ' -f1 <목록> | LC_ALL=C sort | sha256sum)"
} > "$OUT/eval_sha256_total.txt"
for d in "${DATASETS[@]}"; do
  list="$OUT/eval_sha256/${d}_sha256.txt"
  (cd "$EVAL_DIR/$d" && find . -type f | sed 's#^\./##' | LC_ALL=C sort | xargs -d '\n' sha256sum) > "$list"
  all="$(cut -d' ' -f1 "$list" | LC_ALL=C sort | sha256sum | cut -d' ' -f1)"
  img="$(grep -F "  ${IMAGES[$d]}/" "$list" | cut -d' ' -f1 | LC_ALL=C sort | sha256sum | cut -d' ' -f1)"
  nimg="$(grep -cF "  ${IMAGES[$d]}/" "$list")"
  printf '[%s] 전체 파일 %s개  전체 해시 %s\n         이미지(%s/) %s개  전체 해시 %s\n' \
    "$d" "$(wc -l < "$list")" "$all" "${IMAGES[$d]}" "$nimg" "$img" | tee -a "$OUT/eval_sha256_total.txt"
done

echo; echo "완료: $OUT"
