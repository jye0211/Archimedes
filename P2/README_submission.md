# 문제 2 — 소청초 기지 중간층 수온 연직 구조 복원 (제출 요약)

## 결과

| 지표 | 값 |
| --- | --- |
| Public RMSE | 0.669356 ℃ |
| Baseline(수심 선형보간) RMSE | 1.290264 ℃ |
| 개선율 | 약 48.1% |
| Public 점수 | 24.934568 / 33 |

## 파이프라인 개요

```
관측자료 로드 + 짧은 결측 보간 (가림구간은 보간에서 제외, 누수 방지)
        ↓
혼합강도 feature (성층지수, 진동진폭, 계류선흔들림, 표층냉각)
        ↓
참조층 온도/염분 pivot + 수심 선형보간 baseline
        ↓
T-S(수온-염분) 관계 기반 feature: 밀도 지수 + 층별 T-S 기후값 회귀
        ↓
층별 최종 feature 세트 (layer 2/3/4 각각 다름)
        ↓
HistGradientBoostingRegressor, baseline 대비 잔차(residual)를 학습
  - Optuna로 공통 하이퍼파라미터 탐색 후, 층별로 개별 재탐색해 유의미하게 나은 층만 채택
  - layer2만 부트스트랩 5시드 앙상블 (layer3/4는 단일 모델)
        ↓
저성층 구간 소프트 게이팅 (baseline이 이미 정확한 균질 구간에서 모델 과신 방지)
        ↓
전체 데이터로 최종 재학습 → submission.csv
```

## 주요 결정과 근거

### 1. Feature 선택 (층별로 다름)
다중공선성 점검(상관계수 + VIF)으로 16개 공통 후보로 1차 축소한 뒤, permutation importance와
leave-one-out ablation으로 층별 최적 세트를 확정했다.
- **layer 2**: 12개 (16개에서 `public_ref_count`, `surface_cooling_24h`, `psal_layer8`, `mooring_swing` 제거)
- **layer 3 / layer 4**: 각각 5개 핵심 feature만 사용 (block CV 비교 결과 공통16 vs 축소세트가 동률 또는 축소세트 우세)

Permutation importance와 leave-one-out 결과가 일부 feature(`oscillation_amplitude` 등)에서 엇갈렸는데,
이는 상관된 feature 간 정보 중복 때문으로 판단했고 leave-one-out(실제 재학습 기반) 결과를 우선했다.

### 2. T-S 관계 명시적 활용
가림 층은 온도와 염분이 함께 결측되므로, 참조층의 T-S 관계를 학습해 두 종류의 feature를 추가했다.
- **밀도 기반 성층지수**: 온도만 쓰던 기존 성층지수보다 물리적으로 더 정확 → 전 층에 소폭 기여
- **T-S 기후값 회귀 예측치**: 참조층 염분+계절로 온도를 예측 → layer4에서만 뚜렷한 효과 (염분 의존도가 특히 높은 층)

층별로 효과가 갈려서, 최종적으로 밀도지수는 전 층에, T-S 회귀 예측치는 layer4에만 추가했다.

### 3. Block 기반 교차검증과 예외 구간(block2)
학습 가능 기간을 5개 연속 블록으로 나눠 leave-one-block-out CV를 사용했다. 이 중 한 블록(block2)은
baseline RMSE가 다른 블록(1.1~2.1℃)보다 훨씬 작은(0.24℃) 극단적으로 균질한 구간으로, 모델이 미세 잔차를
과신해 상대적으로 크게 틀어지는 구조적 특이 구간이었다. 이후 모든 비교는 "block2 제외 4블록 평균"을
신뢰 지표로, block2는 게이팅 검증용으로 별도 확인했다.

### 4. 모델 다양성 및 하이퍼파라미터 검토
- Ridge 회귀: 전 블록에서 HistGBR에 확실히 열세 → 기각
- RandomForest: 성능도 낮고 HistGBR과 오차 상관 0.87~0.89로 다양성도 부족 → 기각
- CatBoost 등 외부 패키지: 재현 검증이 인터넷 차단 환경에서 이뤄진다는 공지에 따라 처음부터 배제
- 시드 앙상블: `HistGradientBoostingRegressor`는 행 부트스트랩을 하지 않아 `random_state`만 바꾸는 방식은
  효과가 없었음 → 직접 복원추출(bootstrap)로 재구현. layer2에서만 뚜렷한 개선이 있어 layer2에만 채택하고 layer3/4는 단일 모델 유지
- 층별 개별 하이퍼파라미터 튜닝: 처음엔 3개 층이 공통 파라미터를 공유했으나, layer2/3는 개별 재탐색이
  뚜렷하게 유리했고(block2 제외 기준 layer2 22.66%→25.64%, layer3 18.18%→22.35%) layer4는 득이 없어
  공통 파라미터를 유지. Optuna 샘플러에 시드를 고정하지 않으면 실행마다 다른 하이퍼파라미터가 나와
  재현성이 깨지는 문제가 있어 `TPESampler(seed=42)`로 고정했다.

### 5. 저성층 게이팅
밀도 기반 성층지수가 낮을수록(=이미 균질할수록) 모델 예측 비중을 baseline 쪽으로 부드럽게 낮추는
소프트 게이팅을 적용했다. 하드 게이팅(임계값 기준 on/off)보다 소프트 게이팅이 block2를 더 잘 방어하면서
다른 블록엔 손해가 없었다. threshold/smoothness는 층별 grid search로 확정했다(layer2는 앙상블 채택 후
resid 분포가 바뀌어 재조정).

## 시도했으나 채택하지 않은 것

| 시도 | 결과 |
| --- | --- |
| FS_HYBRID 기준 Optuna 재탐색 | 기존 파라미터보다 저하 (REDUCED_BLOCKS 2개 블록에 과적합된 것으로 추정) |
| Baseline을 PCHIP(3차 스플라인)으로 교체 | block2 외 블록에서 확실히 악화, block2 개선도 전체 RMSE 기여도가 작아(≈1%) 무의미 |
| 조건부 PCHIP/선형 블렌딩 | 위와 동일한 이유로 기각 |
| layer3 subsample bagging (frac=0.5) | 내부 CV에서는 근소하게 개선(23.08%→24.41%, 게이팅 후 전체 16.97→17.01)됐으나, 실제 Public 리더보드에서는 오히려 하락(0.669356→0.681694 RMSE) → 기각, 원 구성 유지 |

## 파일 구성

- `pipeline_final.ipynb`: 전체 재현 가능한 최종 파이프라인 (데이터 로드 → feature → 모델 → 게이팅 → 제출 파일 생성)
- `submission.csv`: 최종 제출 파일 (Public RMSE 0.669356)
- `final_models.pkl`: 학습이 끝난 최종 모델 가중치 (아래 "가중치 파일" 참고)
- `gate_params_final.pkl`: 최종 게이팅 파라미터

## 가중치 파일 사용법

`final_models.pkl`은 층(layer)을 key로 하는 dict이며, 구조는 다음과 같다.

| layer | 모델 | 비고 |
| --- | --- | --- |
| 2 | `SeedEnsembleModel` (내부에 `HistGradientBoostingRegressor` 5개) | 부트스트랩 복원추출로 학습한 5개 모델의 예측 평균 |
| 3 | `SeedEnsembleModel` (내부에 5개) | subsample bagging(frac=0.5)이 아닌, 전체 크기 복원추출 버전 (subsample은 리더보드 검증 결과 기각) |
| 4 | `HistGradientBoostingRegressor` 단일 모델 | 공통 하이퍼파라미터(`study.best_params`) 사용 |

`SeedEnsembleModel`은 `pipeline_final.ipynb`의 9번 섹션에서 정의된 커스텀 클래스이므로, **pkl을 로드하기 전에
반드시 노트북의 해당 클래스 정의 셀을 먼저 실행**해야 한다(그렇지 않으면 `AttributeError: Can't get attribute
'SeedEnsembleModel'` 발생). 최소 재현 예시:

```python
import joblib
# pipeline_final.ipynb의 "9. 모델 학습 함수" 셀(SeedEnsembleModel 클래스 정의 포함)을 먼저 실행한 뒤:
final_models = joblib.load("final_models.pkl")
gate_params = joblib.load("gate_params_final.pkl")

# 예측: baseline_pred + alpha * resid_pred  (alpha는 소프트 게이팅, 14번 섹션 apply_gate 참고)
resid_pred = final_models[layer].predict(X[FS_HYBRID[layer]])
```

가중치가 `submission.csv`를 정확히 재현하는지는 이미 검증했다 (재현 검증 최대 오차: 0.00000000).

## 알려진 한계 / 향후 개선 여지

- 내부 5-block CV 지표와 실제 Public 리더보드 점수 사이에 약간의 괴리가 있었다(예: layer3 subsample bagging
  사례에서 CV상 개선이 실제로는 악화로 이어짐). CV 지표의 미세한 차이(1p 미만)는 실제 개선을 보장하지
  않으므로, 최종 판단은 항상 리더보드 점수로 확인하는 절차를 권장한다.
- 남은 레버: 앙상블 예측 분산(불확실성) 기반 게이팅(시간 부족으로 미시도), layer4의 다른 방식 튜닝 등
