# cvpr2027_world — factor × layer 측정

코드만 이 폴더에 있고, **가중치와 데이터셋은 `/mnt/ssd4/youngjae/cvpr2027_yj`에서 읽기만** 한다.
(`env.sh`가 `HF_HOME`을 그쪽 `hf_cache`로, `AGD`를 그쪽 `agd_dataset`으로 잡는다.)

## 무엇을 재는가

Cosmos-Predict2-2B의 **어느 layer / 어느 branch가 A·G·D 각 factor를 담당하는지**를
teacher-forced activation patching으로 잰다 (plan §26–28).

AGD 궤적을 한 번 돌리면서, 지정한 step의 reference latent `z_t`에서:

```
v_full   = f(z_t, AGD)
v_ref(g) = f(z_t, AGD\g)                caption에서 factor g 삭제
v_patch  = f(z_t, AGD) 인데, 한 (layer, branch)의 residual delta 만
           AGD\g 스트림이 그 자리에 쓴 값으로 교체

restore(g,l,b,t) = <v_patch − v_full, v_ref(g) − v_full> / ‖v_ref(g) − v_full‖²
```

`restore`는 factor g의 효과가 그 (layer, branch)를 통해 흐르는 정도다.
모든 tap을 교체하면 reference 스트림이 정확히 재현되므로 (순수 residual) 토이
모델에서는 단일 tap 합이 1.0005로 가산적이었다 (`test_block_patch.py`).

**그러나 실제 모델에서는 가산적이지 않다** — scene-0626_f24 에서 단일 tap 84개의
합이 **13.3** 이었다. 단일 branch 치환 하나의 평균 `mag`가 0.46, 즉 한 곳만 바꿔도
factor 전체 효과 크기의 46%가 재현된다. factor 효과가 (layer, branch)에 걸쳐 매우
중복돼 있다는 뜻이고, **`restore`를 "몇 %"로 읽으면 안 된다.**
`analyze_factor_layer.py`가 이 합을 가산성 진단으로 출력한다.

### 왜 이 방식인가

- **최종 영상/latent 비교는 안 된다.** step 0의 4e-4 차이가 step 34에 1000배로 증폭·포화해서,
  `cosmos_exp/causal_ablation.py`는 어느 layer를 막아도 align 0.25로 똑같이 나왔다.
  카오스 증폭을 재는 것이지 factor가 어디서 들어갔는지를 재는 게 아니다.
- **cross-attention span만 가리는 것으로는 부족하다.** 텍스트는 `attn2`로만 들어오지만,
  factor를 실제로 렌더링하는 건 `attn1`(시공간)과 `ff`다.
  `cosmos_exp/causal_probe.py`는 read-in만 본다. patching은 세 branch 전부를 본다.

### 공짜로 얻는 control 2개

| control | 기대값 | 의미 |
|---|---|---|
| `own` — AGD 자기 delta를 재주입 | restore = 0 | 수치 floor |
| `L0/attn1` — layer 0 self-attn | restore = 0 | 두 스트림이 같은 입력을 받으므로 구조적 0 |

## 실행

```bash
source env.sh
GPU=0 ./run_probe.sh                       # frames 29, group 4, step_every 7
GPU=0 GROUP=1 TAG=L1 ./run_probe.sh        # layer 1개 단위 (28 layer 전부)
SAMPLE=$AGD/scene-0629_f16 GPU=1 ./run_probe.sh
```

`run_probe.sh`가 probe → `analyze_factor_layer.py`까지 이어서 돈다.

### 비용 / 메모리

| 설정 | latent T | tap config | 예상 forward | 예상 시간 | GPU |
|---|---|---|---|---|---|
| `FRAMES=13 GROUP=14 STEP_EVERY=34` (smoke) | 4 | 6 | ~40 | ~2분 | ~12 GB |
| `FRAMES=29 GROUP=4 STEP_EVERY=7` (기본) | 8 | 21 | ~340 | ~10–15분 | ~18 GB |
| `FRAMES=29 GROUP=1 STEP_EVERY=7` (정밀) | 8 | 84 | ~1,300 | ~35분 | ~18 GB |
| `FRAMES=93` (Cosmos 기본) | 24 | — | — | — | ~40 GB |

`--frames`가 메모리 손잡이다. **해상도는 704×1280을 유지해야 한다** (480p는 깨진 영상).
captured delta는 CPU에 쌓인다 (T=8, tap 84개 = 약 9.7 GB).

## 산출물

```
runs/<sample_id>/
├── factor_layer_<tag>.pt      원시 기록 (restore/mag/align/cross, write_norm, controls)
├── analysis.json              layer profile, branch share, specificity, 전 curve
├── placement.json             ★ LoRA 배치 결정 (아래 RULE)
├── fig_layer_profile.png      factor × layer  (plan §49 맵)
├── fig_restore_heatmap.png    layer × branch, factor별 panel
├── fig_step_profile.png       timestep profile
└── fig_specificity.png        factor × factor cross-restore
```

`placement.json`의 선택 규칙은 **데이터를 보기 전에 고정**했다. plan §44의
random-placement baseline과 공정하게 비교하려면 사후 선택을 하면 안 된다.

```
restore 평균 > 0  AND  specificity > tau(기본 1.0)  →  factor별 restore 상위 topk(기본 4)
specificity(f) = |restore_f| / (Σ_{g≠f} |restore_g| + eps)          (plan §28)
```

## 파일

| 파일 | 내용 |
|---|---|
| `block_patch.py` | `BlockPatcher` — block forward를 계측 사본으로 교체, branch delta capture/inject |
| `cosmos_common.py` | 모델 id, negative prompt, `build_pipe`, `latent_shape` |
| `factor_layer_probe.py` | 측정 본체 |
| `analyze_factor_layer.py` | 집계 · 그림 · `placement.json` |
| `test_block_patch.py` | **가중치 없이** 도는 정확성 테스트 (실제 diffusers block, 작은 차원) |
| `test_analyze_smoke.py` | 스키마 + 분석기 + 배치 규칙 테스트 (ground truth 심어서 회수 확인) |

두 테스트는 지금 통과 상태다.

```
$PY test_block_patch.py     # pass-through 동일 / self-injection no-op / 전체 injection = 참조 스트림 / restore 합 ≈ 1
$PY test_analyze_smoke.py   # 심은 A→ff@L12, G→attn2@L21, D→attn1@L24 를 모두 회수
```

## 실행 기록 (2026-09-09, scene-0626_f24, 45 frame / latent T=12)

`hf_cache/hub` 에 ACL 로 읽기 권한을 받아 5개 축을 모두 돌렸다.

| # | 축 | 스크립트 | 시간 |
|---|---|---|---|
| 1 | restore (patching) | `factor_layer_probe.py` (28 layer x 3 branch x 3 factor x 5 step) | 35분 |
| 2 | write_f (attn2 span 분해) | `cosmos_exp/factor_attention.py` | 2분 |
| 3 | local_div + ref Gram | `local_div_probe.py` | 11분 |
| 4 | leverage (gradient) | `grad_probe.py` | 4분 |
| 5 | 순위상관 | `compare_axes.py` | 즉시 |

### 통제

- null (AGD 자기 delta 재주입): 5 step 전부 `0.00e+00`
- L0/attn1 구조적 0: A=G=D=`0.0000`
- reference Gram `|cos|` 0.11~0.58 → factor 대비 설계는 건강 (분리 상한 ≈0.58)

### 확정된 방법론

**분석은 attn2 에 한정하고, 그 branch 가 쓰는 양으로 정규화하고, factor 차분을 봐야 한다.**

이유 (측정값):

| branch | write `||delta||/||h||` | factor 로 인한 상대 변화 | factor 간 layer 상관 |
|---|---|---|---|
| attn1 | 0.343 | 0.062~0.085 | **+0.98** (factor-blind) |
| ff | 0.416 | 0.077~0.104 | **+0.98** (factor-blind) |
| **attn2** | **0.032** | **0.303~0.431** | **+0.31~0.73** (판별) |

attn2 는 작게 쓰지만 factor 에 따라 크게 바뀐다. 절대 크기로 재는 지표
(`D_h`, `restore`, `local_div`, `leverage` 의 raw 프로파일)는 전부 attn1/ff 에
압도되어 factor 를 구분하지 못한다. 또 patching 은 깊은 layer, gradient 는 얕은
layer 로 각각 편향되므로 (전파 사슬 길이) raw peak layer 는 해석하면 안 된다.

### 결과 (N=1, 가설 수준)

| factor | local_div | grad attn2 | grad norm2 | 수렴 |
|---|---|---|---|---|
| A | L3, L8~11 | L22, L18, L9 | L9, L13, L4, L11 | **L9~L11** |
| G | L20, L23, L24 | L24, L23, L2, L0 | L2, L23, L20 | **L20/23/24** |
| D | 없음 | L14, L13, L12, L11 | L14, L16, L8 | **L11~L16** (gradient 만) |

D 는 현재 거동에서는 국소화되지 않는데 학습 leverage 에서는 가장 강하게
국소화된다 (세 factor 중 최대). plan H1 의 전제와 일관.

### 학습 전 파라미터 간섭 `cos(g_f, g_g)`

| | A-G | A-D | G-D |
|---|---|---|---|
| attn2 | +0.313 | +0.022 | -0.236 |
| norm2 | +0.283 | +0.033 | -0.266 |
| attn1 / ff | +0.40 | +0.05 | -0.16 |

A-G 정렬(간섭), A-D 직교(공존 가능), G-D 음의 정렬(충돌).
attn2 에서 A-G 간섭은 L6~L9 에서 최소(+0.06), L18~L27 에서 최대(+0.47) —
A 의 자리(L9~11)가 간섭 최소 구간과 일치한다.

### 데이터 문제 (중요)

`agd_dataset` 100 샘플 중 **38개**에 factor 어구 문제가 있다.
`scene-0014_f16` 은 GD 캡션(=A 제거 참조)에 A 어구가 남아 있어 "+A" 대비가
appearance 가 아니라 **어순**을 잰다 — 기존 `cosmos_exp` 의 `GD->AGD` 결과도
같은 이유로 무효다. 완전히 깨끗한 샘플은 **63개**.
`factor_layer_probe.py` / `grad_probe.py` / `local_div_probe.py` 는 오염 샘플을
자동으로 거부한다 (`--allow_dirty` 로 우회 가능).

### 다음 (규칙 재고정 후)

placement 규칙을 **attn2 한정 / write 정규화 / factor 차분**으로 다시 고정한 뒤
N 확장. 앞 규칙(restore 순위)은 branch 혼합으로 신호가 묻히는 타당성 실패라
폐기한다 — 답이 마음에 들지 않아서가 아니다. 이 구분을 기록해 둔다: 규칙을
결과를 보고 바꾸면 plan 44 의 random-placement 비교가 무효가 된다.

- N=20: 깨끗한 63 샘플에서 층화 선정, `--step_every 3`
- C축(학습 후 ΔW): 위 예측(A->L9~11, G->L20~24, D->L11~16, A-G 간섭)의 검증.
  expert training set(plan 56C)이 선행 조건이고, **sample 당 caption 하나**로
  구성해야 한다 (8조합을 같은 영상에 학습시키면 factor 무감각을 학습한다).
