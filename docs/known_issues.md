# 알려둘 점

## 1. `src/`와 `notebooks/`의 관계
원본 프로젝트는 노트북 중심으로 진행되었습니다.
따라서 정확한 실험 흐름과 저장된 성능 수치는 `notebooks/`가 1차 기준입니다.
`src/`는 공개 저장소용으로 정리한 보조 모듈입니다.

## 2. 점수 비교 해석 주의
- v2, v9, v10은 feature set이 다릅니다.
- 특히 v10의 `pre_cal`, `pre_cal_rounded`는 타겟 생성 구조에 매우 가깝습니다.
- 따라서 v10 점수는 일반화 회귀 성능과 동일선상에서 해석하면 안 됩니다.

## 3. v10 LightGBM notebook의 출력
저장된 노트북 출력 기준으로는 다음이 확인됩니다.
- Base 3-Fold CV RMSE: 1.0555
- Base Holdout RMSE: 0.98
- Optuna 3-Fold CV RMSE: 0.3024

시간상 제약으로 인해 최종 모델은 LinearRegression으로 제출하여 LightGBM에 대해서는 Optuna 3-Fold CV RMSE까지만 구하고 Holdout은 진행하지 않았습니다.
이후 업데이트에서는 LightGBM을 활용한 일반화 모델을 구축할 예정입니다.
