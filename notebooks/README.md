# notebooks/

이 폴더에는 **GitHub 공개용 대표 노트북만 리네이밍해서** 남겼습니다.

원칙은 다음과 같습니다.
- 공개 저장소에는 **의미가 바로 보이는 이름**을 사용합니다.
- 원본 파일명과 실제 경로는 `docs/notebook_mapping.md`에 기록합니다.
- 중복 버전, 초안, 발표 제작용 노트북은 제외합니다.

## 포함된 대표 노트북
- `00_eda_feature_engineering_master.ipynb`
- `phase1_01_svr_v2.ipynb`
- `phase1_02_lightgbm_v2.ipynb`
- `phase1_03_catboost_v2.ipynb`
- `phase2_01_stacking_v9_residual_meta.ipynb`
- `phase3_01_formula_recovery_linear.ipynb`
- `phase3_02_lightgbm_v10_feature_variant.ipynb`

## 주의
이 프로젝트는 원래 **노트북 중심으로 실험이 진행**되었습니다.
최종 점수와 실험 흐름을 재현할 때는 `notebooks/`를 우선 기준으로 보고,
`src/`는 공개 저장소용 재사용 가능한 보조 모듈로 이해하는 것이 가장 정확합니다.
