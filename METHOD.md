# 레이어 배치 분석 — 방법 인수인계

Cosmos-Predict2-2B-Video2World 에서 **A(외형) / G(기하) / D(동역학) 각각이 어느
DiT 블록에서 작동하는지**를 찾는 방법. 다른 서버의 에이전트가 그대로 재현할 수
있도록 절차와 함정을 적는다.

작업 트리: `/home/mingyu/cvpr2027_world` (코드만)
읽기 전용 참조: `/mnt/ssd4/youngjae/cvpr2027_yj` (가중치·데이터셋·학습/평가 하니스)

---

## 0. 한 줄 요약

캡션에서 factor 구절 **하나만 교체**해 특정 블록 윈도우에만 주입하고, **디코딩된
픽셀**이 얼마나 달라지는지 잰다. 같은 노이즈·같은 조건 이미지를 쓰므로 차이는
캡션에서만 온다.

B-LoRA (ECCV 2024) Sec.3 의 블록 분석을 비디오·A/G/D 에 맞춘 것이다.

---

## 1. 왜 이 방법인가 (먼저 실패한 것들)

| 시도한 방법 | 결과 | 이유 |
|---|---|---|
| 레이어별 인과 주입 → restore 사영 | ❌ | Σ_L restore = 2.73 인데 전체 동시 주입은 0.32. 중복이 심해 비가법 |
| ΔW 방향/크기 (학습 전후 가중치 차) | ❌ | 방향의 93% 가 seed 로 결정됨 (cos 0.074 across seed vs 0.741 across factor) |
| 학습 denoising loss | ❌ | 캡션이 loss 의 0.147%. youngjae 도 배치 간 차이 0.003 으로 구분 불가 확인 |
| **캡션 교체 주입 → 픽셀 지표** | ✅ | 유일하게 재현되는 국소화를 냄 |

레이어별 인과 주입이 실패하는 것은 구현 문제가 아니다. Basu et al. (ICLR 2024)
가 UNet 에서 같은 결론을 냈다 — "distributed, not isolated, but distinct per
attribute".

---

## 2. 절차

### 2.1 캡션 쌍 만들기 (가장 중요)

**길이를 맞춘 교체**여야 한다. 덧붙이기는 안 된다.

```
B-LoRA:   "a [bunny] sitting"  →  "a [tiger] sitting"
          길이 동일, 내용 정반대

처음에 틀린 것:
  "The vehicle continues straight."
  → "The vehicle continues straight on a clear day with dry roads and bright lighting."
  +10 단어이고, 추가된 구절이 base 와 모순되지 않는다.
  base 가 날씨를 언급하지 않으니 "clear day" 를 붙여도 모델이 바꿀 게 없다.
```

데이터셋의 구절 재고에서 상대를 고른다:

1. 각 구절에서 앞의 전치사(`on a `, `in the ` 등)를 분리 → core
2. core 끼리 **단어 수 ±3** 안인 후보만 남김
3. 그중 **SigLIP 텍스트 코사인이 가장 낮은 것**(= 가장 의미가 먼 것) 선택
4. 교체는 core 만, 전치사는 원문 것을 유지 (문법 보존)

```
"on a clear day with dry roads and bright lighting"
→ "on a rainy night with wet roads and dim streetlights"     8→8 단어, cos 0.60
```

쌍 배정은 **샘플의 clean 목록 내 위치**로 정한다. 샤딩 순서에 의존하면 다른
실행에서 잰 ceiling 으로 정규화할 수 없다.

### 2.2 생성 3종

```
V0  기준   : 28블록 전부 ref 캡션
V   주입   : window w 의 블록만 inj 캡션, 나머지 24개는 ref
ceiling    : 28블록 전부 inj 캡션   ← 정규화 기준
```

`z0`(초기 노이즈)와 조건 이미지를 고정한다.

### 2.3 주입 구현

`block_patch.py` 가 각 블록의 `forward` 를 교체해 `encoder_hidden_states` 를
갈아끼운다. 게이트가 둘 필요하다.

```python
e = kw.get("encoder_hidden_states")
if gate["pos"] is None: gate["pos"] = e     # 스텝0 첫 호출 = conditional pass
on = bool(gate["map"]) and e is gate["pos"] # ← CFG 의 negative pass 배제
if on and gate["band"] is not None:         # ← (선택) 스텝 구간 게이트
    lo, hi = gate["band"]; on = lo <= gate["step"] <= hi
```

**CFG 주의**: 파이프라인은 스텝마다 transformer 를 두 번 부른다(조건/무조건,
`pipeline_cosmos2_video2world.py` L709/L728). 무조건 pass 까지 바꾸면 negative
prompt 를 오염시킨다.

### 2.4 픽셀 지표

```
d_lum      |평균 휘도 차|                         A 고유
d_hist     채널별 색 히스토그램 거리 (32 bin)      A 고유
d_struct   1 − SSIM(z-정규화 휘도)                G 고유, 밝기 변화 제거됨
d_flow     |Farneback flow 크기 변화|             D 고유
d_flowdir  flow 방향 평균 각도 변화                D 고유
d_toward   [sim(V,inj)−sim(V,ref)] − [sim(V0,inj)−sim(V0,ref)]   SigLIP, B-LoRA 원 통계
d_lum_f0   0번 프레임만의 휘도 변화                ← control, 항상 0 이어야 함
```

`d_toward` 는 **모션을 못 읽는다** (SigLIP 은 이미지 모델). D 판정에 쓰면 안 된다.

### 2.5 정규화와 검정

샘플마다 효과 크기가 10배씩 다르므로 **자기 ceiling 으로 나눈다**:

```
recover(w, m) = metric(window w) / metric(ceiling)
```

최종 판정은 비율 크기가 아니라 **샘플별 argmax 의 일치도**:

```
샘플마다 7개 윈도우 중 어디가 최대인지 세고 이항검정 (chance = 1/7)
6개 지표에 BH-FDR 보정
```

---

## 3. 두 개의 하드 control

둘 다 통과해야 결과를 믿을 수 있다.

```
1. ref 캡션을 28블록 전부에 주입 → V0 와 정확히 동일 (max|dV| = 0.000e+00)
   게이팅이 정확하다는 증거

2. d_lum_f0 = 0.0000  (모든 행)
   파이프라인이 첫 latent 프레임을 조건 이미지로 매 스텝 덮어쓰기 때문.
   0 이 아니면 주입이 새고 있다는 뜻
```

스텝 구간 게이트를 쓸 때는 `0-19`(전 구간) 밴드가 게이트 없는 ceiling 과 **비트
단위로 같아야** 한다.

---

## 4. 절대 빠뜨리면 안 되는 두 가지 (여기서 두 번 틀렸다)

### 4.1 ceiling 스크리닝을 먼저 한다

윈도우 그리드는 **캡션 교체가 영상을 실제로 바꾸는 클립에서만** 의미가 있다.

```
agd_dataset: D 효과가 실재하는 클립 10/59 (17%)
             → 스크리닝 없이 7클립으로 그리드를 돌렸더니 5개가 효과 0 이라 희석됨
             → 48셀 중 1개만 q<0.10, 그것도 캡션 길이 artifact
```

ceiling(28블록 전부 주입)을 전 클립에 먼저 돌리고, 효과가 큰 상위 8~10 클립만
그리드에 넣는다. 스크리닝은 샘플당 2생성이라 그리드(9생성)의 1/4 비용이다.

### 4.2 도메인을 학습/평가와 맞춘다

**이게 가장 크게 틀렸던 부분이다.**

```
               agd_dataset          wm_dataset
프레임률        2 fps 키프레임        네이티브 ~12 Hz
생성 설정       45 프레임 @ fps16     29 프레임 @ fps12
```

파이프라인이 첫 프레임을 조건 이미지로 고정하므로 **외형·기하 변화는 프레임이
진행되며 누적되는 drift 로만 나타난다.** 프레임 수가 짧으면 누적될 여유가 없다:

```
같은 day→night 교체, 마지막 프레임 휘도 변화
frames=13:  0.0009     frames=45:  0.1392      155배
```

학습 설정(13프레임)에서 프레임 수를 가져왔다가 "A·G 는 텍스트로 픽셀에 도달하지
않는다"는 **틀린 결론**을 냈다. 그리고 45프레임/agd 에서 잰 배치로 학습했더니
A 가 졌다 — wm_dataset 에서 다시 재니 A 의 자리가 L4-5 → L8-11 로 이동했다.

**분석은 반드시 학습·평가와 같은 데이터셋·프레임 수·fps 로 한다.**

---

## 5. 실행

```bash
source ./env.sh

# 1) ceiling 스크리닝 (전 클립, 샘플당 2생성)
./wm_screen.sh                    # 122클립 × A/G/D, 6 GPU, ~2시간

# 2) 효과 강한 상위 10클립 선정 → runs/wm_screen/top10.json

# 3) 윈도우 그리드 (7윈도우 + ceiling, 샘플당 9생성)
./wm_grid.sh                      # 3 factor × 10클립, ~1시간

# 4) 판독
$PY analyze_swap_grid.py runs/wm_grid
```

핵심 파일:

```
swap_probe.py          교체 주입 + 생성 + 지표 (--pairs 로 손으로 쓴 쌍도 가능)
block_patch.py         블록별 텍스트 override
analyze_swap_grid.py   ceiling 정규화 + argmax 이항검정 + BH-FDR
analyze_stepband.py    스텝 구간 판독
merge_experts.py       disjoint expert 병합 (겹치면 거부)
```

---

## 6. 이 방법으로 나온 결과와 그 검증

### 6.1 분석이 지목한 배치 (wm_dataset, 29프레임 @12fps)

```
factor   배치        근거
A        L8~11      d_lum 5/8  q=0.014
G        L4~7       d_hist·d_flowdir 5/8  q=0.007
D        L12~15     d_struct·d_flow·d_flowdir 6/8 q=0.000
```

세 배치가 서로 겹치지 않아 병합이 정확하다 (state_dict 합집합).
D → L12 근방은 agd_dataset(45프레임)에서도 같은 윈도우가 나왔다 — 유일하게 도메인
간 재현된 결과다.

### 6.2 실제로 학습시켜 본 결과 — 부분적으로만 맞았다

각 배치에 LoRA 를 붙여 800 step 학습하고 (youngjae `train_lora.py`, `cache_wm`,
1.39 M 로 예산 통일) youngjae 의 판정기로 평가했다.

```
checkpoint   배치            r(flow,speed)  fvd_s3d   ssim   psnr  spike   CSR
real            —               0.742          —       —      —    1.33    —
base            —               0.276       116.4   0.622  19.70   3.65   0.62
mg_A         L4,5 (agd 분석)    0.194        64.2   0.662  20.88   3.44   0.88
mg_D         L12,13 (agd)       0.770        82.4   0.660  20.51   1.69   0.88
mg_G         L12,13 (agd)       0.679        88.2   0.653  19.93   1.85    —
wm_A         L8-11 (wm 분석)    0.211        65.8   0.665  21.17   2.05   1.00
wm_G         L4-7 (wm)          0.518        61.5   0.671  21.31   2.33    —
wm_D         L12-15 (wm)        0.673        62.3   0.666  20.86   1.89   1.00
A_analysis   L3,9,11,12,17(yj)  0.700        55.0   0.666     —    2.29    —
D_analysis   L0,2,3,5,6 (yj)    0.418        58.8   0.676     —    2.33   0.67
```

**맞은 것**

- D 는 이 방법이 정확히 짚는다. `mg_D`(L12,13)가 `r(flow,speed)` 0.770 으로 전
  체크포인트 1위이고 실영상 0.742 를 넘는다. youngjae 의 write-share 가 지목한
  L0,2,3,5,6 은 0.418 에 그친다.
- 교차 대조가 크게 갈린다. D 를 A 자리(L4,5)에 두면 CSR 0.88→0.50, G 를 A 자리에
  두면 `r(flow,speed)` 가 **−0.081 로 부호가 뒤집힌다.**
- 도메인을 맞춘 배치가 CSR 을 올린다 (A·D 모두 0.88 → 1.00) 그리고 화질도
  올린다 (G 의 FVD 88→62, D 의 FVD 82→62).

**틀린 것**

- **A 는 이 방법이 못 맞춘다.** agd 분석의 L4-5 도, wm 분석의 L8-11 도
  `r(flow,speed)` 가 0.19~0.21 로 youngjae 의 0.700 에 3배 이상 밀린다. wm 분석이
  지목한 L8-11 안에 youngjae 의 L9·L11·L12 가 들어 있는데도 성능이 재현되지
  않았다 — 레이어 집합이 겹쳐도 개수와 rank 가 다르면 결과가 달라진다는 뜻이다
  (내 4층×r10 vs youngjae 5층×r8).
- **CSR 과 `r(flow,speed)` 는 서로 다른 것을 잰다.** `wm_D` 는 CSR 1.00 인데
  `r(flow,speed)` 는 `mg_D` 보다 낮다. 전자는 "캡션을 뒤집으면 따라오는가",
  후자는 "속도를 실제와 얼마나 맞게 내는가"다. 한 지표로 배치를 고르면 안 된다.
- 같은 1.39 M 이라도 **좁고 두꺼운 쪽**(2층×r20)이 넓고 얇은 쪽(4층×r10)보다
  D 에서 좋았다. 배치와 별개로 rank·폭이 독립 변수다.

**따라서 이 방법의 현재 위치**: D 에 대해서는 write-share 보다 학습 결과를 잘
예측한다. A 에 대해서는 예측하지 못한다. G 는 판정기가 없어 판정 불가.

---

## 7. 부수적으로 확인된 구조적 사실

```
패딩 싱크
  attn2 에 bias 가 없고 T5 패딩이 정확히 0
  → to_v(0)=0, norm_k(to_k(0))=0 → 패딩 501개의 logit 이 전부 0
  → 실토큰 확률질량 = S/(S+501) = 0.74~4.89%
  attn2 가 ‖h‖ 의 3% 만 쓰면서 factor 민감도는 최고인 이유

텍스트 통로
  to_v 가 주 통로 (인과 복원 0.523 vs to_k 0.322, 27/28 레이어)
  후반 레이어는 to_k/to_v 동반 필수 (짝 깨면 분기 출력 9~15배 이탈)
  attn1 은 텍스트를 안 받아 28/28 에서 0 — 구조적으로 factor expert 불가

타임스텝
  세 factor 모두 초반 7/20 스텝이 전 구간 효과의 96~104%
  후반 6스텝은 0~8%  → expert 라우팅 축이 아니라 공통 게이트
```

---

## 8. 함정 목록

1. **CFG negative pass 오염** — 주입을 conditional pass 로 게이팅할 것
2. **첫 프레임 고정** — `d_lum_f0` 가 0 인지 항상 확인
3. **구절 길이** — 토큰 수가 다르면 길이 artifact 가 factor 효과로 보임
4. **쌍 배정 순서 의존** — 샘플 이름 기반 결정적 인덱스를 쓸 것
5. **SigLIP 은 모션을 못 읽음** — D 를 `d_toward` 로 판정하지 말 것
6. **ceiling 스크리닝 생략** — 효과 없는 클립이 평균을 희석
7. **GPU 6 (이 서버)** — CUDA 초기화 실패가 반복됨, 다른 카드를 쓸 것
8. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 금지** — 이 torch/드라이버
   조합에서 CUDA driver error 발생 (youngjae HANDOFF §3.2)
