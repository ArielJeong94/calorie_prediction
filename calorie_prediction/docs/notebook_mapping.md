# 노트북 매핑 가이드

## 공개 저장소에서는 왜 리네이밍된 노트북만 남겼는가?
공개 GitHub에서는 **원본명보다 역할이 드러나는 이름**이 훨씬 좋습니다.

이 프로젝트는 원래 노트북 수가 많고 파일명이 길며 버전 표기가 혼재되어 있었습니다.
그래서 최종 공개본에서는:
- `phase1`, `phase2`, `phase3`로 실험 단계를 드러내고
- 모델명과 역할을 파일명에 직접 넣고
- 원본 파일명은 이 문서에서만 보존했습니다.

즉, **GitHub에는 리네이밍 파일만 남기고**, 원본 추적은 이 매핑 문서로 해결하는 방식이 가장 깔끔합니다.

## 대표 노트북 매핑

| 원본 경로 | 공개 저장소 파일명 | 유지 이유 |
|---|---|---|
| `main/01_calories_EDA_preprocessing_코드 취합.ipynb` | `notebooks/00_eda_feature_engineering_master.ipynb` | EDA + 파생변수 생성 총정리 |
| `main/v2_model_정리/02_01. SVR_옵튜나 적용O_v2.ipynb` | `notebooks/phase1_01_svr_v2.ipynb` | 일반화 기준 최강 단일 모델 |
| `main/v2_model_정리/07_LightGBM_Regression_v2.ipynb` | `notebooks/phase1_02_lightgbm_v2.ipynb` | 트리 기반 대표 |
| `main/v2_model_정리/08_Cat_Boost_Regression_v2.ipynb` | `notebooks/phase1_03_catboost_v2.ipynb` | 트리 기반 최고 성능 |
| `main/model_template/09_2. Stacking_Classification_data_v9사용_version4_잔차보정.ipynb` | `notebooks/phase2_01_stacking_v9_residual_meta.ipynb` | 잔차보정 스태킹 최종 대표 |
| `main/v10_model_정리/01. LinearRegression(최종 리더보드).ipynb` | `notebooks/phase3_01_formula_recovery_linear.ipynb` | formula-recovery benchmark |
| `main/v10_model_정리/07_03_피쳐변경시도_LightGBM_v10_Regression.ipynb` | `notebooks/phase3_02_lightgbm_v10_feature_variant.ipynb` | v10 피처 변경 LightGBM 실험 대표 |

## 제외한 파일 유형
- 동일 실험의 초안 / 사본 / 중간 버전
- 발표 제작용 노트북
- CSV 생성 전용 노트북
- SHAP 또는 스크린샷 중심 보조 노트북
