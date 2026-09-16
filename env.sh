# source this before running anything here.
# 코드는 이 폴더, 가중치/데이터는 youngjae 경로에서 읽기만 한다.
export YJ=/mnt/ssd4/youngjae/cvpr2027_yj
export HF_HOME=$YJ/hf_cache          # 읽기 전용으로 씀 (권한 필요: 아래 참고)
export HF_HUB_OFFLINE=1              # 읽기 전용 캐시에 lock/write 시도 방지
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token   # 남의 토큰 안 건드림 (내 토큰 사용)
export TMPDIR=/mnt/ssd4/youngjae/tmp
export AGD=$YJ/agd_dataset
export PY=/mnt/ssd4/youngjae/envs/cvpr2027/bin/python
# NOTE: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 는 켜지 말 것 (CUDA init error)

# 읽기 전용 HF 캐시에서 lock 오류가 나면 스냅샷 경로를 직접 쓴다.
_snap=$(ls -d $HF_HOME/hub/models--nvidia--Cosmos-Predict2-2B-Video2World/snapshots/*/ 2>/dev/null | head -1)
[ -n "$_snap" ] && export COSMOS_MODEL="${_snap%/}"
