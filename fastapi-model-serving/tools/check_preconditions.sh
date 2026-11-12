#!/usr/bin/env bash
# 인증 시험 사전조건(시험절차서의 사전조건 1~4)을 한 번에 확인한다. 서버를 켜기 전에 실행한다.
# 사용법: bash tools/check_preconditions.sh [EVAL_DIR]
#   EVAL_DIR 기본값은 웹 UI 기본값과 같은 /mnt/d/02_Dataset/04_Bigdata/Eval
#   모든 항목이 OK면 종료 코드 0, 하나라도 FAIL이면 1. 기록이 필요하면 뒤에 | tee precheck.log 를 붙인다.

SERVING="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"     # fastapi-model-serving
REPO="$(dirname "$SERVING")"
EVAL_DIR="${1:-${EVAL_DIR:-/mnt/d/02_Dataset/04_Bigdata/Eval}}"

# 모델 가중치: 이름|fastapi-model-serving 기준 경로|SHA-256
# 가중치 파일이나 경로를 바꾸면 서버 코드, 이 목록, 시험절차서의 SHA-256을 같이 고친다
MODELS=(
  "객체|fastapi/yolo11x_finetuned_kitti_nusc_1280_vis40_ep20.pt|4c0a3bf163da068ae1665c53ce793dca9da36c1b6033b191ebdc27a5d7b73d98"
  "날씨|fastapi/weights/best_model_weather_e100_lr00001_b64_260826_bdd100k_b3_class_4.pth|2ed8f817658414f701caf892233e7136f9d314903ca3e04875dbe6e1e17f061d"
  "시간|fastapi/weights/best_model_time_e20_lr0001_b64_260831_2_b3_class_2.pth|e59f84882f0c4cd768c6d66769d143cd49daace38e148e651c38b6c299d02d14"
  "차선|LaneDetection/weights/tusimple_r18.pth|f175ab318d6711e4a219d8d8239684f0b960b14bc3b224e16b696ff31a4c0f31"
)
CONDA_ENVS=(bigdata bigdata_clrnet)
PORTS=(8000 8001 8501)
DATASETS=(nuScenes KITTI BDD100K TuSimple)

fails=0
ok()   { printf '  [OK]   %s\n' "$*"; }
fail() { printf '  [FAIL] %s\n' "$*"; fails=$((fails + 1)); }
info() { printf '         %s\n' "$*"; }

echo "자동 라벨링 툴 인증 시험 사전조건 확인 ($(date '+%Y-%m-%d %H:%M:%S'), $(hostname))"
echo "  저장소      $REPO"
if git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
  dirty=""
  [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ] && dirty=" (커밋하지 않은 변경 있음)"
  echo "  커밋        $(git -C "$REPO" log -1 --format='%h %ad %s' --date=short)$dirty"
fi
echo "  평가 데이터 $EVAL_DIR"

# ---------------------------------------------------------------------------
echo; echo "1. 실행 환경"
missing=()
for f in fastapi/server.py LaneDetection/server.py streamlit/ui.py streamlit/evaluation.py; do
  [ -f "$SERVING/$f" ] || missing+=("$f")
done
if [ ${#missing[@]} -eq 0 ]; then ok "저장소 파일 (모델 서버, 차선 서버, 웹 UI, 평가 모듈)"; else fail "저장소 파일 없음: ${missing[*]}"; fi

# 스크립트에서는 conda가 PATH에 없을 수 있어 흔한 설치 위치도 찾아본다
conda_bin="$(command -v conda || true)"
[ -z "$conda_bin" ] && conda_bin="${CONDA_EXE:-}"
if [ -z "$conda_bin" ]; then
  for c in "$HOME"/anaconda3/bin/conda "$HOME"/miniconda3/bin/conda "$HOME"/miniforge3/bin/conda; do
    [ -x "$c" ] && conda_bin="$c" && break
  done
fi
if [ -z "$conda_bin" ]; then
  fail "conda를 찾을 수 없음"
else
  envs="$("$conda_bin" env list 2>/dev/null | awk '!/^#/ && NF {print $1}')"
  for e in "${CONDA_ENVS[@]}"; do
    if grep -qx "$e" <<< "$envs"; then ok "conda 환경 $e"; else fail "conda 환경 $e 없음"; fi
  done
fi

listening="$(ss -ltn 2>/dev/null | awk 'NR > 1 {print $4}')"
for p in "${PORTS[@]}"; do
  if grep -Eq "[:.]$p\$" <<< "$listening"; then
    fail "포트 $p 사용 중 (서버를 켜기 전에 실행했는지 확인)"
  else
    ok "포트 $p 비어 있음"
  fi
done

# ---------------------------------------------------------------------------
echo; echo "2. 모델 무결성 (SHA-256)"
for m in "${MODELS[@]}"; do
  IFS='|' read -r name path want <<< "$m"
  if [ ! -f "$SERVING/$path" ]; then fail "$name 가중치 없음: $path"; continue; fi
  # 서버 코드의 가중치 경로만 바꾸고 이 목록을 안 고치면, 서버가 읽지 않는 파일을 검사하게 되므로 함께 본다
  if ! grep -rqF --include='*.py' "$(basename "$path")" "$SERVING/fastapi" "$SERVING/LaneDetection"; then
    fail "$name 서버 코드가 $(basename "$path")를 쓰지 않음 (가중치 경로가 바뀌었다면 이 스크립트의 MODELS도 고칠 것)"
    continue
  fi
  got="$(sha256sum "$SERVING/$path" | cut -d' ' -f1)"
  if [ "$got" = "$want" ]; then ok "$name $got  $path"; else fail "$name SHA-256 불일치: $got (기대값 $want)  $path"; fi
done

# ---------------------------------------------------------------------------
echo; echo "3. 평가 데이터 무결성 (데이터셋마다 md5sum -c checksums.md5, 몇 분 걸림)"
for d in "${DATASETS[@]}"; do
  dir="$EVAL_DIR/$d"
  if [ ! -f "$dir/checksums.md5" ]; then fail "$d: $dir/checksums.md5 없음"; continue; fi
  n=$(wc -l < "$dir/checksums.md5")
  errs="$(cd "$dir" && md5sum -c --quiet checksums.md5 2>&1)"
  # 목록에 없는 파일이 이미지 폴더에 섞이면 평가 장수가 달라지므로 함께 본다
  extra="$(cd "$dir" && comm -23 <(find . -type f ! -name checksums.md5 | sed 's#^\./##' | sort) \
                                 <(awk '{print $2}' checksums.md5 | sort))"
  if [ -z "$errs" ] && [ -z "$extra" ]; then
    ok "$d 파일 $n개"
    continue
  fi
  if [ -n "$errs" ]; then
    fail "$d: $(grep -c ': FAILED' <<< "$errs")개 파일 불일치 또는 읽기 실패 (전체 $n개)"
    grep ': FAILED' <<< "$errs" | head -5 | while read -r l; do info "$l"; done
    grep -q 'Cannot allocate memory' <<< "$errs" &&
      info "→ 'Cannot allocate memory'는 WSL 메모리 부족으로 파일을 못 읽은 것. 다른 프로그램을 닫고 다시 실행"
  fi
  if [ -n "$extra" ]; then
    fail "$d: 체크섬 목록에 없는 파일 $(wc -l <<< "$extra")개"
    head -5 <<< "$extra" | while read -r l; do info "$l"; done
  fi
done

# ---------------------------------------------------------------------------
echo; echo "4. 부하 격리 (GPU를 쓰는 다른 프로세스)"
# WSL의 nvidia-smi는 프로세스 목록을 보여주지 않으므로 GPU 장치(/dev/dxg, 리눅스는 /dev/nvidia*)를 연 프로세스를 찾는다.
# 다른 사용자의 프로세스는 권한상 보이지 않는다.
# 화면 표시용 데스크톱 프로세스(Xorg, gnome-shell 등)는 웹 UI를 쓰려면 필요하므로 부하로 세지 않고 따로 표시한다
DESKTOP_PROCS='^(Xorg|Xwayland|gnome-shell|kwin_x11|kwin_wayland|plasmashell|gdm-x-session|compiz|cinnamon|xfwm4)$'
gpu_procs=(); desktop_procs=()
for p in /proc/[0-9]*; do
  [ "${p#/proc/}" = "$$" ] && continue
  if ls -l "$p/fd" 2>/dev/null | grep -qE '/dev/(dxg|nvidia)'; then
    entry="${p#/proc/} $(tr '\0' ' ' < "$p/cmdline" 2>/dev/null | cut -c1-100)"
    if grep -qE "$DESKTOP_PROCS" "$p/comm" 2>/dev/null; then desktop_procs+=("$entry"); else gpu_procs+=("$entry"); fi
  fi
done
if [ ${#gpu_procs[@]} -eq 0 ]; then
  ok "GPU를 쓰는 프로세스 없음"
else
  fail "GPU를 쓰는 프로세스 ${#gpu_procs[@]}개 (시험과 무관한 프로세스면 종료)"
  for g in "${gpu_procs[@]}"; do info "PID $g"; done
fi
for g in "${desktop_procs[@]}"; do info "화면 표시용 데스크톱 프로세스 (부하 아님) PID $g"; done
command -v nvidia-smi >/dev/null &&
  info "GPU $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null)"
info "CPU 부하(1분 평균) $(cut -d' ' -f1 /proc/loadavg), 코어 $(nproc)개"

# ---------------------------------------------------------------------------
echo
if [ "$fails" -eq 0 ]; then
  echo "결과: 사전조건 1~4 모두 OK"
  exit 0
fi
echo "결과: FAIL ${fails}개. 위 [FAIL] 항목을 해결한 뒤 다시 실행하세요"
exit 1
