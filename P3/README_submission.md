# 문제 3 — 유의파고 단기 예측 제출 파이프라인

본 문서는 `pipeline_final.ipynb`의 실행 방법과 최종 모델 구조, 생성 파일의 역할을 정리한 제출용 지침서입니다.

---

## 파일 구조

`pipeline_final.ipynb`는 저장소 루트에서 실행하는 것을 기준으로 합니다.

```text
P3/
├─ data/
│  ├─ train_wave.csv
│  ├─ train_atmos.csv
│  ├─ test_context.parquet
│  ├─ test_index.csv
│  └─ sample_submission.csv
│
├─ pipeline_final.ipynb
├─ final_models.pkl
├─ gate_params_final.pkl
├─ submission.csv
└─ README.md
```

### 입력 파일

| 파일 | 내용 |
|---|---|
| `data/train_wave.csv` | 3개 기지의 연속 파랑 관측자료, 20분 간격 |
| `data/train_atmos.csv` | 3개 기지의 연속 기상 관측자료, 10분 간격 |
| `data/test_context.parquet` | 200개 예측 사례의 기준시각 이전 48시간 context |
| `data/test_index.csv` | 6개 리드타임의 예측 대상 key |
| `data/sample_submission.csv` | 최종 제출 형식 및 행 순서 기준 |

`baseline_persistence.csv`와 `score.py`는 본 파이프라인 실행에 필요하지 않습니다.

### 생성 파일

| 파일 | 내용 |
|---|---|
| `final_models.pkl` | 최종 예측에 실제 사용되는 CatBoost 모델 60개와 모델 metadata |
| `gate_params_final.pkl` | 리드타임별 고정 feature branch routing, feature schema, seed 및 모델 설정 |
| `submission.csv` | 최종 1,200개 유의파고 예측값 |

`gate_params_final.pkl`은 별도의 학습형 gate classifier가 아니라 **리드타임별 고정 routing/configuration 정보**를 저장하는 파일입니다.

---

## 예측 문제

예측 대상은 다음 6개 리드타임의 유의파고 `hs`입니다.

```text
+3 h / +6 h / +9 h / +12 h / +18 h / +24 h
```

3개 관측기지를 하나의 전역 모델 구조에서 사용하며, 기지 정보는 다음 순서의 one-hot feature로 추가됩니다.

```text
G-ORS / I-ORS / S-ORS
```

모델은 미래 유의파고 자체를 직접 학습하지 않고 현재 유의파고 대비 변화량을 학습합니다.

\[
\Delta H_s(\tau)=H_s(t+\tau)-H_s(t)
\]

최종 예측은 다음과 같이 복원합니다.

\[
\widehat{H_s(t+\tau)}
=
H_s(t)+\widehat{\Delta H_s(\tau)}
\]

최종 `hs_pred`는 제출 규격에 맞추어 `0–30 m` 범위로 제한합니다.

---

## 전체 파이프라인

`pipeline_final.ipynb`는 중간 checkpoint나 사전 생성 prediction을 사용하지 않고 `data/`의 배포 원본에서 시작합니다.

```text
Raw Data
   ↓
시간 형식 및 기본 무결성 검사
   ↓
Train candidate / 미래 Hs target 생성
   ↓
Wave feature 생성
   ↓
Atmospheric feature 생성
   ↓
Wind–wave interaction feature 생성
   ↓
Core feature schema 구성
   ↓
Atmospheric shape feature 생성
   ↓
리드타임별 입력 branch 구성
   ↓
CatBoost × fixed 10 seeds 학습
   ↓
Seed prediction arithmetic mean
   ↓
현재 Hs + 예측 ΔHs
   ↓
sample_submission 순서 정렬
   ↓
submission.csv 생성
   ↓
final_models.pkl / gate_params_final.pkl 저장
   ↓
저장 모델 재로드 후 inference 동일성 검증
```

---

## 학습 데이터 구성

Train 파랑 자료에서 각 기지별 미래 `hs`를 shift하여 6개 target을 생성합니다.

Candidate는 다음 조건을 만족해야 합니다.

- 기준시각 `hs ≥ 1.5 m`
- +3 / +6 / +9 / +12 / +18 / +24 h target이 모두 유효
- 기준시각 이전 48시간의 context 확보
- +24시간 target까지 미래 자료 확보

정상 실행 시 Train candidate는 총 **24,360개**입니다.

| Station | Candidate 수 |
|---|---:|
| G-ORS | 9,893 |
| I-ORS | 7,312 |
| S-ORS | 7,155 |
| **Total** | **24,360** |

이 값이 다르면 raw data, 시간 정렬 또는 전처리 환경을 먼저 확인해야 합니다.

---

## Feature 구성

### Core feature set

기본 branch는 총 **130개 numeric feature**를 사용합니다.

| Feature group | 개수 | 주요 내용 |
|---|---:|---|
| Wave | 66 | 현재 Hs, lag, 변화량, rolling statistics, 파향의 순환형 표현 등 |
| Wind magnitude | 25 | 풍속·돌풍 current, lag, change, rolling mean, valid ratio |
| Wind direction | 11 | 풍향 sin/cos, lag, 방향 정렬도 |
| Wind–wave interaction | 12 | 바람-파랑 정렬, 횡방향 성분, 파향 평행 풍속·돌풍 |
| Pressure | 16 | 기압 current, lag, change, rolling mean/std, valid ratio |
| **Total** | **130** | |

Station one-hot 3개를 결합하므로 실제 Core 모델 입력 차원은 다음과 같습니다.

```text
130 numeric + 3 station = 133 features
```

### Atmospheric shape feature

+3시간 예측에는 Core feature에 **19개 atmospheric shape feature**를 추가합니다.

Shape feature는 최근 기상 trajectory의 형태를 표현합니다.

- 풍속 slope: 1 / 3 / 6 / 12 / 24 h
- 돌풍 slope: 3 / 6 / 12 h
- 기압 slope: 3 / 6 / 12 / 24 h
- 풍향 concentration: 3 / 6 / 12 / 24 h
- 풍속 EWM gap: 3 / 6 / 12 h

따라서 +3시간 branch는 다음과 같습니다.

```text
130 Core
+ 19 Atmospheric Shape
= 149 numeric

149 numeric + 3 station = 152 model inputs
```

---

## 리드타임별 모델 구조

| Lead time | Numeric feature branch | Station one-hot | Input 차원 | Seed model 수 |
|---:|---|---:|---:|---:|
| +3 h | Core130 + Atmospheric Shape19 | 3 | 152 | 10 |
| +6 h | Core130 | 3 | 133 | 10 |
| +9 h | Core130 | 3 | 133 | 10 |
| +12 h | Core130 | 3 | 133 | 10 |
| +18 h | Core130 | 3 | 133 | 10 |
| +24 h | Core130 | 3 | 133 | 10 |

총 학습 모델 수는 다음과 같습니다.

\[
1\times10 + 5\times10 = 60
\]

즉 `final_models.pkl`에는 최종 inference에 필요한 **CatBoost 60개**만 저장됩니다.

---

## CatBoost 설정

모든 seed model은 동일한 CatBoost 설정을 사용합니다.

```python
CATBOOST_BASE_PARAMS = {
    "iterations": 450,
    "learning_rate": 0.03,
    "depth": 4,
    "l2_leaf_reg": 5.0,
    "random_strength": 1.0,
    "loss_function": "RMSE",
    "verbose": False,
    "allow_writing_files": False,
}
```

고정 seed는 다음 10개입니다.

```python
[11, 29, 47, 71, 101, 137, 173, 211, 251, 307]
```

각 리드타임에서는 10개 seed model의 `ΔHs` prediction을 **단순 산술평균(arithmetic mean)** 합니다.

\[
\widehat{\Delta H_s}
=
\frac{1}{10}
\sum_{k=1}^{10}
\widehat{\Delta H_s}^{(k)}
\]

Leaderboard 결과를 이용한 연속 blending coefficient, prediction correction 또는 추가 threshold는 사용하지 않습니다.

---

## 실행 환경

필수 Python package는 다음과 같습니다.

```text
numpy
pandas
pyarrow
catboost
```

Exact reproduction은 다음 CatBoost 버전에서 확인되었습니다.

```text
catboost==1.2.10
```

권장 설치 예시는 다음과 같습니다.

```bash
pip install numpy pandas pyarrow catboost==1.2.10
```

CatBoost 버전이 다를 경우 동일한 모델 구조에서도 극미한 수치 차이가 발생할 수 있습니다.

---

## 실행 방법

### 1. 데이터 배치

저장소 루트에 `data/` 폴더를 만들고 다음 파일을 넣습니다.

```text
data/train_wave.csv
data/train_atmos.csv
data/test_context.parquet
data/test_index.csv
data/sample_submission.csv
```

### 2. Notebook 실행

`pipeline_final.ipynb`를 저장소 루트에서 열고 **첫 셀부터 마지막 셀까지 순서대로 실행**합니다.

Notebook은 현재 작업 경로를 프로젝트 루트로 사용합니다.

```python
PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"
```

따라서 Notebook을 다른 위치에서 실행하면 입력 파일을 찾지 못할 수 있습니다.

### 3. 생성 파일 확인

정상 실행이 완료되면 저장소 루트에 다음 3개 파일이 생성됩니다.

```text
final_models.pkl
gate_params_final.pkl
submission.csv
```

---

## 생성 파일 상세

### `final_models.pkl`

최종 prediction에 실제로 사용되는 CatBoost 모델과 metadata를 포함합니다.

주요 구조:

```text
models
├─ AtmosShape149
│  └─ 3h
│     └─ 10 seed models
└─ Core130
   ├─ 6h  ─ 10 seed models
   ├─ 9h  ─ 10 seed models
   ├─ 12h ─ 10 seed models
   ├─ 18h ─ 10 seed models
   └─ 24h ─ 10 seed models
```

metadata에는 다음 정보가 저장됩니다.

- lead time
- station order
- fixed seed 목록
- target 정의
- seed aggregation 방식
- prediction clipping 범위
- CatBoost version 및 parameter
- feature 개수

### `gate_params_final.pkl`

다음 고정 routing을 저장합니다.

```text
+3 h  → AtmosShape149
+6 h  → Core130
+9 h  → Core130
+12 h → Core130
+18 h → Core130
+24 h → Core130
```

추가로 다음 정보가 포함됩니다.

- Core feature 이름 및 순서
- Atmospheric shape feature 이름 및 순서
- +3h용 149개 feature schema
- station order
- fixed seed 목록
- CatBoost parameter
- target 및 aggregation 방식

### `submission.csv`

최종 형식:

```text
case_id,station,lead_h,hs_pred
```

총 **1,200행**이며 `sample_submission.csv`의 key 및 행 순서를 그대로 유지합니다.

Notebook 내부에서 다음 항목을 자동 검증합니다.

- 1,200 rows
- key 중복 없음
- `hs_pred` 결측 없음
- `hs_pred` Inf 없음
- `sample_submission.csv`와 key 순서 동일
- prediction 범위 `0–30 m`

---

## 재현성 및 무결성 검사

최종 파일을 저장한 뒤 Notebook은 `final_models.pkl`과 `gate_params_final.pkl`을 다시 불러와 별도 inference를 수행합니다.

재로드한 모델로 생성한 prediction과 최초 생성한 `submission.csv`의 prediction 차이를 확인하여 저장/로드 과정에서 결과가 변하지 않았는지 검증합니다.

또한 `submission.csv`의 SHA256을 계산하여 검증된 reference fingerprint와 비교합니다.

검증된 SHA256:

```text
59da21d052e08255473461dac00078b477c31a902afb7f63f1f79930dca2f851
```

동일한 배포 데이터와 검증된 실행환경에서 위 SHA256과 일치하면 최종 제출 prediction이 동일하게 재생성된 것입니다.

SHA256이 다를 경우 다음 항목을 우선 확인합니다.

1. `catboost` 버전
2. `pandas` 및 Parquet reader 환경
3. 입력 데이터 파일 변경 여부
4. Notebook 실행 위치
5. 모든 셀을 처음부터 순차 실행했는지 여부

---

## 제출 시 주의사항

- `submission.csv`를 Notebook 실행 후 수동으로 수정하지 않습니다.
- `final_models.pkl`과 `gate_params_final.pkl`도 실행 후 별도 편집하지 않습니다.
- `data/` 원본 파일의 column 이름, 시간 정보 또는 행 순서를 임의로 변경하지 않습니다.
- `gate_params_final.pkl`은 학습형 gate model이 아니라 고정 routing metadata입니다.
- 재현 검증이 필요한 경우 새 kernel에서 `pipeline_final.ipynb`를 처음부터 다시 실행하는 것을 권장합니다.

최종 제출 파일의 `hs_pred`는 모두 유한한 값이어야 하며 `0–30 m` 범위에 있어야 합니다.
