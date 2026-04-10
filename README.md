# 🚴🏻 calorie_prediction

> 생체 데이터 기반 칼로리 소모량 회귀 예측 프로젝트

---

## 🎯 Abstract

<img width="1980" height="1061" alt="image" src="https://github.com/user-attachments/assets/ae5d3ffa-76dc-4036-82ed-1dbde14c9e77" />

<img width="1071" height="238" alt="image" src="https://github.com/user-attachments/assets/06ee2c92-54c0-426b-a524-0ee182245f8a" />

[최종 4위 기록]

이 프로젝트는 **운동 시간, 심박수, 체온, 체중, 성별, 나이**와 같은 생체 데이터를 활용해 칼로리 소모량을 예측하는 회귀 모델을 개발한 해커톤 프로젝트다.  
프로젝트는 크게 두 개의 흐름으로 진행되었다.

**첫 번째**는 **생리학적 해석이 가능한 feature engineering 중심의 일반 모델링 흐름**이다.  
EDA를 통해 **데이터**가 **강한 비선형성**과 **긴 꼬리 분포**를 가진다는 점을 확인했고, 이를 바탕으로 `hr_ratio`, `activity_proxy`, `bsa_intensity_time`, `exercise_stress_index`, `age_gen_corr`, 구간 분리형 `hr_load` 등 다양한 생리 기반 파생변수를 설계했다.  
이 흐름에서는 **SVR이 가장 강한 단일 모델**로 확인되었고, CatBoost와 LightGBM이 그 뒤를 이었다.

**두 번째**는 프로젝트 후반의 **formula recovery 흐름**이다.  
상관관계 분석과 실험을 통해 `pre_cal`, `pre_cal_rounded`가 타깃 구조를 거의 직접적으로 재현한다는 점을 발견했고, 최종 보고서에서는 이 변수를 사용한 **Linear Regression**이 가장 낮은 오차를 기록했다.  
이 결과는 “**복잡한 모델이 항상 더 좋은 것은 아니며, 데이터 구조를 이해하는 것이 모델 복잡도보다 중요하다**”는 핵심 결론으로 이어졌다.

따라서 이 저장소는 단순히 최종 점수만 정리한 결과물이 아니라,

1. **도메인 기반 feature engineering 실험**,
2. **단일 모델 비교와 스태킹 실패 분석**,
3. **타깃 생성 구조 추적(formula recovery)**

까지 포함한 전체 연구 로그를 담고 있다.

---

## 📋 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 프로젝트명 | 생체 데이터를 활용한 칼로리 소모량 회귀 예측 모델 개발 |
| 주최 | 데이콘 x 오즈코딩스쿨 |
| 목표 | 칼로리 소모량(`calories_burned`) 회귀 예측 |
| 데이터 규모 | 학습 데이터 7,500개 샘플 |
| 원본 핵심 변수 | `Exercise_Duration`, `BPM`, `Body_Temperature(F)`, `Height(feet + inches)`, `Weight(lb)`, `Weight_Status`, `Gender`, `Age` |
| 타겟 변수 | `Calories_Burned` |
| 평가 지표 | RMSE |
| 초기 강한 단일 모델 | **SVR** |
| 최종 보고서 기준 모델 | **Linear Regression (`pre_cal_rounded` 기반)** |
| OOF / 5-Fold CV RMSE | **0.0472** *(최종 보고서 기준 5-Fold 평균)* |
| 리더보드 RMSE | **0.05776** |
| 해석 포인트 | 일반 모델링 흐름에서는 SVR이 가장 강했지만, 최종 단계에서는 `pre_cal_rounded`가 타깃 구조를 거의 직접 재현하면서 Linear Regression이 압도적으로 유리해짐 |

> **📌 중요한 해석 포인트**  
> 이 프로젝트에는 서로 다른 두 성격의 성능 축이 공존한다.
>
> - **일반화 중심 실험 축**: v2~v9, SVR / CatBoost / LightGBM / Stacking 비교
> - **타겟 생성식 복원(formula recovery) 축**: v10, `pre_cal_rounded` 기반 Linear Regression
>
> 따라서 성능 수치는 같은 목적의 실험끼리 비교하는 것이 정확하다.

---

## 핵심 인사이트

### 1) 운동 데이터는 단순 선형 물리량이 아니라 복합 생리 반응에 가깝다
EDA 결과, 대부분의 변수는 정규분포가 아니고 강한 비대칭성과 긴 오른쪽 꼬리를 보였다.  
특히 운동 강도·심박 부하·열 반응 관련 파생변수는 곱셈 구조 때문에 극단값이 커졌고, 이 때문에 로그 변환과 상호작용 변수가 필요했다.

### 2) 이상치는 제거보다 해석이 먼저였다
체온, 심박, 운동시간이 결합된 값들은 단순 오염치라기보다 **고강도 운동에서 발생하는 의미 있는 생리 반응**일 가능성이 높았다.  
그래서 프로젝트에서는 이상치를 무조건 제거하지 않고, 조건부 가중치 조정과 구간 분리형 변수로 대응했다.

### 3) 일반 모델링 축에서는 SVR이 가장 안정적이었다
단일 모델 비교에서 SVR이 가장 낮은 RMSE를 기록했고, CatBoost와 LightGBM은 그 뒤를 이었다.  
특히 체온 이상치 가중치 조정보다 **모델 자체와 feature set의 품질**이 성능에 더 큰 영향을 주었다.

### 4) 스태킹은 “좋은 모델을 많이 쌓는 것”만으로 좋아지지 않았다
v9 스태킹 실험에서는 베이스 모델들의 예측 상관이 지나치게 높아 학습 다양성이 부족했다.  
잔차 기반 메타 피처, KNN 밀도 기반 메타 피처까지 추가했지만 결국 **단일 SVR을 넘지 못했다**.

### 5) 후반부의 진짜 전환점은 `pre_cal_rounded`였다
히트맵과 SHAP 분석에서 `pre_cal_rounded`의 영향력이 압도적이었고, 최종적으로는 이 변수가 타깃 생성 구조를 거의 직접 복원하고 있음을 확인했다.  
결과적으로 최종 보고서에서는 **Linear Regression**이 가장 낮은 RMSE를 기록했다.

---

## 🤖 최종 모델 구성

### A. 일반 모델링 흐름의 대표 모델: SVR 🛟
- 입력: v2/v9 계열의 생리 기반 feature set
- 전처리: 수치형 스케일링(StandardScaler) + 범주형 인코딩(OneHotEncoder)
- 튜닝: Optuna 기반 `C`, `epsilon`, `gamma`
- 해석: 일반적인 ML 경쟁 관점에서 가장 강한 단일 모델

### B. 최종 보고서 기준 모델: Linear Regression 📈
최종 보고서에서 선택된 모델은 `pre_cal_rounded`를 핵심 입력으로 사용하는 **Linear Regression**이다.

#### 핵심 변수
- `pre_cal`
- `pre_cal_rounded`

#### 생성 아이디어
논문 기반 칼로리 수식 구조를 참고한 뒤, 데이터에 맞게 회귀 계수를 역추적하여 생성했다.

#### 남성 추정식
```python
pre_cal_male = ((age * 0.2017) + (weight_kg * 0.09036) + (bpm * 0.6309) - 55.0969) * ex_dura / 4.184
```

#### 여성 추정식
```python
pre_cal_female = ((age * 0.0740) - (weight_kg * 0.05741) + (bpm * 0.4472) - 20.4022) * ex_dura / 4.184
```

#### 반올림 변수
```python
pre_cal_rounded = np.floor(pre_cal + 0.5)
```

#### 최종 학습
```python
X = calories_data[["pre_cal_rounded"]]
y = calories_data["calories_burned"]
model = LinearRegression()
```

### 해석상 주의점
이 최종 모델은 “새로운 생리 패턴을 학습한 모델”이라기보다, **타깃을 생성한 구조를 매우 근접하게 복원한 모델**이라는 성격이 강하다.  
따라서 이 저장소에서는 **SVR 기반 일반 모델링 흐름**과 **Linear formula-recovery 흐름**을 구분해서 읽는 것이 중요하다.

---

## 📊 성능 추이

### 1) 단일 모델 스크리닝 (v2 중심)

<img width="437" height="261" alt="v2 모델별 성능 비교" src="https://github.com/user-attachments/assets/9216133d-bdc6-4bf1-af2b-c480df1d5086" />


| 모델 | CV / OOF RMSE | Holdout / Valid RMSE | 해석 |
|---|---:|---:|---|
| SVR (Optuna) | 0.3607 | 0.3362 | 일반 모델링 축 최강 단일 모델 |
| CatBoost | 1.0042 | 0.9484 | 트리 계열 중 상대적으로 우수 |
| LightGBM | 1.3093 | 1.2606 | v2 기준 성능 열세. but 속도 가장 빠름 |
| 무가중치 실험(SVR) | - | **0.3176** | 이상치 가중치보다 feature/model 조합 영향이 더 큼 |

### 2) 스태킹 실험 (v9)

<img width="507" height="302" alt="v9 모델별 성능 비교" src="https://github.com/user-attachments/assets/4ecbee3e-658f-433c-9f6d-97368f95fbe4" />
<img width="507" height="302" alt="v9 모델별 성능 비교(하위 성능 모델 제거)" src="https://github.com/user-attachments/assets/dde89317-8b91-4468-a2e7-0428335dff8d" />

| 실험 | OOF / CV RMSE | 해석 |
|---|---:|---|
| SVR base | **0.3406** | v9 기준 베이스라인 |
| 3-model stacking (Meta=Ridge) | 0.3411 | 단일 SVR보다 미세하게 열세 |
| 3-model stacking (Positive LR) | 0.3411 | 개선 없음 |
| Selected SVR stacking | 0.3416 | 효과 없음 |
| 성별 분리 SVR | Valid 0.3399 | 데이터 감소로 오히려 저하 |

### 3) formula recovery 단계 (v10)

<img width="561" height="324" alt="최종 모델 성능 비교" src="https://github.com/user-attachments/assets/181d3fe1-9fae-46c5-8d16-34ffc462c28b" />


| 모델 | Validation RMSE | 5-Fold CV RMSE | 해석 |
|---|---:|---:|---|
| Linear Regression | **0.0578** | **0.0472** | 최종 보고서 기준 최종 선택 |
| LightGBM Optuna | 0.9782 | 0.2586 | 같은 v10에서도 Linear가 압도적 |

### 성능 해석 요약
- **일반 ML 실험 축**: SVR > CatBoost > LightGBM
- **최종 구조 복원 축**: Linear Regression (`pre_cal_rounded`) >> LightGBM
- 결론적으로 **모델 복잡도보다 데이터 구조가 성능을 결정**했다.

---

## 🏗️ 프로젝트 구조

```text
calorie_prediction/
├── README.md                     # 프로젝트 개요, 실험 흐름, 성능, 실행 방법 정리
├── requirements.txt              # 실행에 필요한 Python 패키지 목록
├── .gitignore                    # GitHub에 올리지 않을 파일/폴더 설정
│
├── data/                         # 원본/가공 데이터 저장 폴더 (GitHub 업로드 제외)
│   ├── raw/                      # 대회 제공 원본 데이터
│   └── processed/                # feature engineering 및 버전별 가공 데이터
│
├── docs/                         # 프로젝트 보조 문서 및 정리 자료
│   ├── notebook_mapping.md       # 원본 노트북명 ↔ GitHub용 대표 노트북 매핑
│   ├── model_experiments.md      # 모델별 시도, 성능, 해석 정리
│   ├── experiment_comparison.csv # 실험 결과 비교표 (모델/변수셋/RMSE 등)
│   ├── feature_summary.csv       # 변수 설명 및 버전별 feature 요약
│   └── known_issues.md           # 한계점, 주의사항, 미해결 이슈 정리
│
├── figures/                      # README, 발표자료, 문서에 사용하는 시각화 산출물
│   ├── eda/                      # EDA 시각화 결과
│   ├── shap/                     # SHAP 결과 이미지
│   └── performance/              # 모델 성능 비교 그래프
│
├── notebooks/                    # 실험 과정을 단계별로 담은 핵심 노트북
│   ├── 00_eda_feature_engineering_master.ipynb
│   │                             # 전체 EDA와 파생변수 생성 로직을 정리한 마스터 노트북
│   ├── phase1_01_svr_v2.ipynb
│   │                             # v2 변수셋 기반 SVR 단일 모델 실험
│   ├── phase1_02_lightgbm_v2.ipynb
│   │                             # v2 변수셋 기반 LightGBM 실험
│   ├── phase1_03_catboost_v2.ipynb
│   │                             # v2 변수셋 기반 CatBoost 실험
│   ├── phase2_01_stacking_v9_residual_meta.ipynb
│   │                             # v9 변수셋 기반 residual/meta stacking 실험
│   ├── phase3_01_formula_recovery_linear.ipynb
│   │                             # v10의 formula-recovery 성격을 확인한 Linear Regression 실험
│   └── phase3_02_lightgbm_v10_feature_variant.ipynb
│                                 # v10 feature variant 기반 LightGBM 비교 실험
│
└── src/                          # 노트북 실험을 재사용 가능하게 정리한 모듈
    ├── __init__.py               # src 패키지 초기화 파일
    ├── config.py                 # 공통 상수, seed, 버전별 설정값 정의
    ├── feature_sets.py           # v1~v10 feature 목록 정리
    ├── features.py               # 파생변수 생성 및 feature engineering 함수
    ├── preprocessing.py          # 전처리 파이프라인 및 입력 데이터 준비 로직
    ├── model.py                  # 모델 생성, 학습, 예측 관련 함수
    ├── stacking.py               # stacking / residual meta 모델 관련 로직
    ├── formula_recovery.py       # formula-recovery 전용 실험 로직
    ├── plots.py                  # EDA 및 성능 비교용 시각화 함수
    └── utils.py                  # 공통 유틸 함수 (저장, 출력, 보조 계산 등)
```

---

## 💻 노트북 설명

### `00_eda_feature_engineering_master.ipynb`
프로젝트의 출발점.  
원본 데이터 탐색, 변수 분포 확인, 이상치 해석, 상관관계 분석, 3D 관계 시각화, 파생변수 설계, 버전별 데이터셋(v2~v10) 저장 로직이 모여 있는 **마스터 노트북**.

### `phase1_01_svr_v2.ipynb`
v2 feature set을 기반으로 SVR을 학습한 노트북.  
Optuna 튜닝, 이상치 가중치 실험, holdout 성능 검증이 포함된다.

### `phase1_02_lightgbm_v2.ipynb`
v2 feature set으로 LightGBM을 실험한 노트북.  
Optuna와 GridSearchCV를 통해 트리 계열 성능을 비교한다.

### `phase1_03_catboost_v2.ipynb`
v2 feature set 기반 CatBoost 실험 노트북.  
트리 기반 모델 중 상대적으로 가장 좋은 성능을 보였던 축이다.

### `phase2_01_stacking_v9_residual_meta.ipynb`
v9 feature set으로 스태킹을 실험한 노트북.  
Residual stacking, OOF prediction, KNN density meta-feature, meta model(Ridge/Linear) 비교가 핵심이다.

### `phase3_01_formula_recovery_linear.ipynb`
`pre_cal_rounded`를 중심으로 타깃 구조를 복원한 뒤 Linear Regression을 학습한 노트북.  
최종 보고서의 핵심 결과를 재현하는 노트북이다.

### `phase3_02_lightgbm_v10_feature_variant.ipynb`
v10 feature set에서 LightGBM을 다시 실험한 노트북.  
최종적으로는 Linear Regression보다 열세였지만, v10에서도 비선형 트리 모델이 어느 정도까지 따라오는지 확인하는 비교 축이다.

---

## 💿 v1~v10 변수 요약

> 아래 표는 업로드된 버전별 CSV(`final_train_adj*.csv`)를 비교해 정리한 요약이다.  
> `final_train_adj.csv`가 사실상 **v1(base)** 역할을 한다.

| 버전 | 컬럼 수 | 성격 | 핵심 변수 예시 |
|---|---:|---|---|
| v1 (base) | 12 | 초반 압축형 베이스셋 | `ex_dura`, `bpm`, `body_temp`, `hr_ratio`, `bmi`, `log_exercise_stress_index`, `log_bsa_intensity_time` |
| v2 | 14 | 일반 모델링 1차 핵심셋 | `bpm_per_kg`, `temp_per_kg`, `exercise_stress_index`, `bsa_intensity_time` |
| v3 | 16 | v2 + 온도/집단보정 확장 | `body_temp_c`, `sqrt_age_gen_corr` |
| v4 | 21 | 심박 부하 시간 구조 반영 | `hr_load_per_kg`, `hr_load_per_bsa`, `hr_load_short/mid/long` |
| v5 | 11 | 공격적 축소형 실험셋 | `hr_load_high`, `high_load_male`, `sqrt_age_gen_corr` |
| v6 | 58 | **feature engineering 마스터셋** | raw + 로그변환 + 상호작용 + 구간화 + 커스텀 변수 전체 |
| v7 | 26 | 상관/선택기반 축소셋 | `log_activity_proxy`, `log_metabolic_load`, `temp_diff_sq`, `hr_efficiency` |
| v8 | 27 | v7 + raw 재투입 | `ex_dura`, `gender` 재추가 |
| v9 | 16 | 스태킹용 compact 셋 | raw 유지 + `bsa_intensity_time`, `exercise_stress_index_raw`, `age_section`, `bpm_section`, `ex_section` |
| v10 | 17 | formula-recovery 최종셋 | v9 + `pre_cal_rounded` |

### v6를 왜 중요하게 봐야 하나?
`v6`는 단순히 컬럼 수가 많은 버전이 아니라, 프로젝트에서 설계한 거의 모든 파생변수를 보존한 **실험용 마스터 데이터셋**이다.  
이후 버전(v7~v10)은 이 v6를 기반으로 **선택, 축소, 구조 변경**을 거쳐 목적별 데이터셋으로 파생되었다.

---

## 전체 실험 흐름

### Step 1. 1차 EDA
- 원본 데이터 분포 확인
- 오른쪽 꼬리와 비선형성 확인
- 체온·심박·운동시간 조합에 대한 이상치 가설 수립
- 집단별(성별/연령/체중 상태) 분포 차이 확인

### Step 2. 파생변수 대량 생성
- 단위 통일: ft/in → cm, lb → kg, F → C
- 생리 지표: `bmi`, `bsa`, `max_hr`, `hr_ratio`
- 운동 부하 지표: `activity_proxy`, `bsa_intensity_time`, `bpm_per_kg`
- 스트레스/대사 지표: `exercise_stress_index`, `metabolic_load`, `cardio_load`
- 열 반응 지표: `temp_diff`, `temp_per_kg`, `heat_accumulation`
- 시간 구간 분리: `hr_load_short/mid/long`, `hr_load_high`
- 커스텀 지표: `age_gen_corr`, `sqrt_age_gen_corr`
- 상호작용 및 로그 변환 변수 생성

### Step 3. 이상치/가중치 실험
- `body_temp >= 41°C` & `bpm < 100` 조합을 생리적으로 불일치한 이상치 후보로 설정
- 층화 분리 + 학습 가중치 0.5 적용
- 결론: 성능 개선 효과 미미, 이후 실험에서는 제거

### Step 4. 단일 모델 스크리닝
- SVR, LightGBM, CatBoost, RandomForest, XGBoost, AdaBoost 등 비교
- 모든 모델을 RMSE 기준으로 비교
- 이 단계의 핵심 결론: **SVR이 가장 안정적**

### Step 5. 2차 EDA + feature refinement
- 잔차 vs BPM 분석
- 연령·성별 집단별 차이 재확인
- 다중공선성 제거
- 스위치형/조건형 파생변수 추가

### Step 6. 스태킹 실험 (v9)
- 다양한 base model 조합 생성
- OOF 예측, 잔차, KNN 밀도 기반 메타 피처 생성
- Ridge / Positive Linear / Tree meta 비교
- 결론: **학습 다양성 부족으로 SVR 단일 모델을 넘지 못함**

### Step 7. formula recovery (v10)
- 논문 기반 칼로리 소모 식에서 아이디어 획득
- `pre_cal`, `pre_cal_rounded` 생성
- 히트맵과 SHAP에서 타깃과의 구조적 일치성 확인
- Linear Regression이 최종 보고서 기준 최종 모델로 선정

---

## 사용 기술

### Language / Environment
- Python
- Google Colab
- Jupyter Notebook

### Data / Analysis
- pandas
- numpy

### Visualization
- matplotlib
- koreanize-matplotlib(한글 깨짐 방지)
- seaborn
- shap

### Machine Learning
- scikit-learn
- SVR
- Linear Regression / Ridge / Lasso / ElasticNet
- RandomForestRegressor
- ExtraTreesRegressor
- LightGBM
- CatBoost
- XGBoost
- StackingRegressor

### Tuning / Validation
- Optuna
- GridSearchCV
- KFold / StratifiedKFold
- OOF prediction

---

## 실행 방법

### 1. 환경 설치
```bash
pip install -r requirements.txt
```

### 2. 데이터 준비
이 저장소는 버전별 CSV를 중심으로 실험이 진행된다.  
대표적으로 아래 파일들을 준비한다.

- `final_train_adj_v2.csv`
- `final_train_adj_v9.csv`
- `final_train_adj_v10.csv`
- (선택) `final_train_adj_v6.csv`

### 3. 권장 실행 순서

#### (1) EDA 및 변수 생성 흐름 확인
```bash
notebooks/00_eda_feature_engineering_master.ipynb
```

#### (2) 일반 모델링 흐름 재현
```bash
notebooks/phase1_01_svr_v2.ipynb
notebooks/phase1_02_lightgbm_v2.ipynb
notebooks/phase1_03_catboost_v2.ipynb
```

#### (3) 스태킹 실험 재현
```bash
notebooks/phase2_01_stacking_v9_residual_meta.ipynb
```

#### (4) 최종 formula recovery 흐름 재현
```bash
notebooks/phase3_01_formula_recovery_linear.ipynb
notebooks/phase3_02_lightgbm_v10_feature_variant.ipynb
```

### 4. `src/` 사용 방식
이 프로젝트는 본질적으로 **노트북 중심 실험 저장소**다.  
따라서 재현의 기준은 `notebooks/`이며, `src/`는 이를 보조하는 공개 저장소용 모듈로 이해하는 것이 가장 정확하다.

---

## What I learned

### 1) 모델보다 데이터 구조가 더 중요할 수 있다
처음에는 복잡한 앙상블과 스태킹이 더 높은 성능을 낼 것이라고 기대했지만, 실제로는 데이터 구조를 더 잘 반영한 단순 모델이 더 강했다.  
특히 최종 단계에서는 Linear Regression이 가장 낮은 RMSE를 기록했다.

### 2) Feature engineering은 “많이 만드는 것”이 아니라 “왜 만드는지 설명할 수 있어야 하는 것”이다
이 프로젝트에서는 생리학적 해석이 가능한 지표를 만들기 위해 많은 파생변수를 설계했다.  
중요한 것은 숫자를 늘리는 것이 아니라, **체온·심박·시간·체격이 어떻게 칼로리 소모로 이어지는지 설명 가능한 구조**를 만드는 일이었다.

### 3) 이상치 처리는 규칙보다 검증이 우선이다
체온 41도 이상, 낮은 BPM 같은 조합을 이상치로 보고 가중치를 조정했지만 실제 성능 개선은 거의 없었다.  
직관적으로 맞는 가설이라도 **반드시 검증으로 확인해야 한다**는 점을 배웠다.

### 4) 스태킹의 핵심은 모델 수가 아니라 다양성이다
베이스 모델을 여러 개 쌓아도 예측 패턴이 거의 같다면 메타 모델은 배울 것이 없다.  
이 프로젝트에서 스태킹이 실패한 가장 큰 이유는 **모델 수 부족이 아니라 학습 다양성 부족**이었다.

### 5) 높은 성능이 항상 더 좋은 모델링을 의미하지는 않는다
`pre_cal_rounded`를 포함한 final phase는 매우 강력한 성능을 만들었지만, 동시에 “이 값이 실제로 무엇을 학습한 것인가?”라는 질문을 남겼다.  
프로젝트를 통해 **성능, 해석 가능성, 일반화 가능성**을 함께 봐야 한다는 점을 분명히 배웠다.

---

## References

- Keytel, L. R. et al. (2005). *Prediction of energy expenditure from heart rate monitoring during submaximal exercise.*
- Webb, P. (1981). *Energy expenditure and fat-free mass in men and women.*
- Taylor, K. M. et al. (2024). *Relation of body surface area-to-mass ratio to risk of exertional heat stroke.*
- Foster, C. et al. (2017). *Monitoring training loads: The past, the present, and the future.*
- Buller, M. J. et al. (2023). *Individualized monitoring of heat illness risk.*
