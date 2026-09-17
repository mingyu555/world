# upstream_youngjae — 이 폴더의 코드는 제가 쓴 것이 아닙니다

원본: `/mnt/ssd4/youngjae/cvpr2027_yj/cosmos_exp/` (youngjae 작성)
복사 시점: 2026-09-17

제 분석(`../METHOD.md`)이 내놓은 레이어 배치를 **학습시키고 평가하는** 데 쓴
하니스입니다. 레포를 자체 완결되게 하려고 그대로 복사해 두었습니다. 수정하지
않았고, 갱신은 원본 트리를 따릅니다.

## 무엇이 무엇인가

| 파일 | 역할 | 제 실험에서의 쓰임 |
|---|---|---|
| `train_lora.py` | factor 별 expert LoRA 학습 (rectified flow, `--layers` 로 배치 지정) | 제 배치와 대조 배치를 같은 예산(1.39 M)으로 학습 |
| `compare_lora.py` | **CSR** — 진짜 절 vs 반증 절을 같은 노이즈로 생성해 지표가 기대 방향으로 움직인 비율 | A·D 판정 (G 는 판정기가 없어 거부됨) |
| `eval_fidelity.py` | 실영상을 기준으로 체크포인트별 생성 + 지표 측정 | 품질표의 행 생성 |
| `report_table.py` | `eval_fidelity` 결과를 표로 | 최종 비교표 |
| `metrics_agd.py` | 판정기 본체 — `luma`(A), `flow_mean`(D, sparse LK), `lane centroid`(G) | 모든 factor 지표의 출처 |
| `quality_metrics.py` | FVD(S3D) · SSIM/PSNR · `delta_spike` 등 화질 | 품질표의 오른쪽 절반 |
| `validate_metrics.py` | **판정기를 실영상에서 먼저 검증** | 아래 표의 근거 |
| `score_quality.py` | 화질만 따로 채점 | |
| `run_same_noise.py` | 파이프라인 빌더 · `NEG` 프롬프트 · latent shape (다른 스크립트가 import) | |
| `trace_utils.py`, `span_utils.py` | 보조 | |

## 판정기의 사용 범위 (youngjae 가 실영상 122클립에서 검증한 결과)

이게 가장 중요합니다. 판정기가 실영상에서 GT 를 못 맞추면 생성물을 판정할
자격이 없고, 검증 결과가 각 지표의 쓸 수 있는 범위를 정합니다.

```
A  luma        night vs day  AUC 0.999    → 사용 가능
   saturation  rain vs dry   AUC 0.923    → 사용 가능
   blue_warm                 AUC 0.52     → 폐기

D  flow_mean   ego 속도와 r = +0.62       → 클립 단위 속도 대비만 가능
   flow_slope  가속 vs 감속 AUC 0.50      → 사용 금지
   (dense Farneback 은 r = +0.03 으로 실패 → sparse Lucas-Kanade 로 교체)

G  lane centroid  map 차로와 r = -0.06    → 작동 안 함
                                          → compare_lora.py 가 --factor G 를 거부
```

그래서 G 는 CSR 을 낼 수 없고 품질 지표로만 볼 수 있습니다.

## 실행 예

```bash
source ../env.sh

# 학습 (배치를 --layers 로 지정)
$PY train_lora.py --factor D --layers 12,13 --rank 20 --alpha 80 \
    --steps 800 --cache $YJ/cosmos_exp/cache_wm --out ../runs_train/mg_D

# CSR
$PY compare_lora.py --factor D --runs ../runs_train/mg_D \
    --dataset $YJ/wm_dataset --clips 8 --out ../runs_eval/D_csr

# 품질표
$PY eval_fidelity.py --runs ../runs_train/mg_D --dataset $YJ/wm_dataset \
    --clips 18 --out ../runs_eval/fid
$PY report_table.py --dirs ../runs_eval/fid --dataset $YJ/wm_dataset
```

`cache_wm`(사전 계산된 latent·텍스트 임베딩)과 `wm_dataset`은 용량 때문에
복사하지 않았습니다. youngjae 트리에서 읽습니다.
