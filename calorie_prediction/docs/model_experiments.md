# 모델 실험 정리

## 1. 실험 단계 개요
여러 파생변수 조합(v1~v10)을 이용하여 실험을 수행하였으나 대표적인 내용만 정리하였습니다.

### 파일별 파생변수 정보
| 버전                           |   컬럼 수 | 해석                                         |
| ---------------------------- | -----: | ------------------------------------------ |
| base (`final_train_adj.csv`) |     12 | 초반 핵심 변수만 남긴 기본형                           |
| v2                           |     14 | 체중 정규화, stress/load 계열 일부 추가               |
| v3                           |     16 | v2 + `body_temp_c`, `sqrt_age_gen_corr` 추가     |
| v4                           |     21 | 심박부하 분해 변수(`hr_load_*`) 확장                  |
| v5                           |     11 | 타겟과 강한 상관관계를 가지는 변수만 남긴 축소형            |
| **v6**                       | **58** | **raw + 파생변수 전체 (feature engineering 수행용)** |
| v7                           |     26 | 로그/변환 중심 선택형                                |
| v8                           |     27 | v7 + 일부 raw 변수 재투입(단위 변환으로 인한 오차 증가 확인)|
| v9                           |     16 | formula-recovery 직전 단순화 버전                   |
| v10                          |     17 | 구간화 변수 + `pre_cal_rounded` 포함                |


### Phase 1 - v2 단일 모델 스크리닝
같은 전처리/파생변수 체계 위에서 여러 회귀 모델을 비교한 구간입니다.

| Model | CV RMSE | Holdout RMSE | 해석 |
|---|---:|---:|---|
| SVR | 0.3607 | 0.3318 | 가장 강한 단일 모델 |
| CatBoost | 1.0042 | 0.9484 | 트리 기반 중 가장 안정적 |
| LightGBM | 1.3093 | 1.2606 | CatBoost 다음 |

해석 포인트:
- 트리 계열보다 **SVR이 훨씬 강했다**는 점이 핵심.
- CatBoost는 트리 기반 대안으로 의미 있었지만, 최종 기준점은 SVR이었습니다.

### Phase 2 - v9 스태킹 / 잔차 보정
잔차 모델, density 메타 피처, Ridge/Positive Linear meta model을 이용해 스태킹을 시도.

| 실험 | OOF/CV RMSE | 해석 |
|---|---:|---|
| 단일 SVR | 0.34060 | 기준점 |
| 단일 LightGBM | 1.27225 | 보조 모델 |
| 단일 XGBoost | 1.50727 | 보조 모델 |
| Stacking (Meta=Ridge) | 0.34112 | 개선 없음 |
| Stacking (Meta=Positive Linear) | 0.34105 | 개선 없음 |

해석 포인트:
- base model diversity가 충분하지 않아 **stacking 이득이 거의 없었습니다**.
- 최대한 base model의 학습 다양성을 확보하고자 메타모델 학습 데이터로 각 베이스 모델의 잔차, KNN을 활용해 density 피처를 만들어 메타모델을 학습시켰지만, 단일 SVR 성능보다 나아지지는 못했습니다.

### Phase 3 - v10 formula recovery / feature variant
v10은 일반화 예측이라기보다, 논문/공식 기반 구조를 이용해 타겟 생성식을 역추정하는 성격이 강합니다.

#### 3-1. Linear Regression with `pre_cal_rounded`
| 실험 | Holdout RMSE | 해석 |
|---|---:|---|
| Linear Regression (`pre_cal_rounded`) | 0.06 | 사실상 formula-recovery benchmark |

#### 3-2. LightGBM feature variant notebook
대표 노트북: `phase3_02_lightgbm_v10_feature_variant.ipynb`

| 설정 | RMSE | 비고 |
|---|---:|---|
| Base LightGBM 3-Fold CV | 1.0555 | 기본 세팅 |
| Base Holdout | 0.98 | 기본 세팅 |
| Optuna-tuned 3-Fold CV | 0.3024 | 저장된 출력 기준 |

이 노트북에서 중요하게 등장하는 피처:
- `exercise_stress_index`
- `metabolic_load`
- `hr_ratio`
- `bpm_per_kg`
- `temp_per_kg`
- `temp_diff`

## 2. 해석 시 참고사항

1. **v2 / v9 / v10은 feature set이 다르므로 단순 점수 비교를 조심해야 한다.**
2. **SVR은 일반화 관점의 strongest single model이었다.**
3. **stacking은 실험적으로 시도했지만 예측 다양성 부족으로 큰 개선이 없었다.**
4. **v10 Linear는 formula-recovery benchmark로 인해 얻어진 성능이다.**
5. **v10 LightGBM feature variant notebook은 formula-derived feature 바깥에서의 트리 모델 실험을 보여주는 대표 노트북이다.**
