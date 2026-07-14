# OmniAnomaly (PyTorch / Apple MPS)

[OmniAnomaly](https://github.com/haowen-xu/omni-anomaly) 공식 구현을 **PyTorch** 기반으로 포팅한 버전입니다.  
macOS Apple Silicon(MPS) GPU 가속을 지원합니다.

## 환경 설정

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch는 [공식 설치 가이드](https://pytorch.org/get-started/locally/)에 따라 MPS 지원 버전을 설치하세요.

## 데이터 준비

### SMD (Server Machine Dataset)

```bash
python download_smd.py
python data_preprocess.py SMD
```

데이터는 `ServerMachineDataset/`에 저장됩니다 (Git에는 포함되지 않음).

### SMAP / MSL

원본 S3 URL(`telemanom/data.zip`)은 현재 **403 Forbidden**으로 받을 수 없습니다.  
아래 방법 중 하나를 사용하세요.

**방법 1 — Hugging Face (권장)**

```bash
pip install huggingface_hub
python download_smap_msl.py   # → data/
python data_preprocess.py SMAP
python data_preprocess.py MSL
```

데이터는 `data/`에 저장됩니다 (Git에는 포함되지 않음).

**방법 2 — Kaggle (API 키 필요)**

```bash
pip install kaggle
# ~/.kaggle/kaggle.json 설정 후
kaggle datasets download -d patrickfleith/nasa-anomaly-detection-dataset-smap-msl
unzip nasa-anomaly-detection-dataset-smap-msl.zip -d data
```

다운로드 후 폴더 구조:

```
data/
├── labeled_anomalies.csv
├── train/    # A-1.npy, ...
└── test/
```

전처리 결과는 `processed/` 폴더에 저장됩니다.

## 실행

```bash
python main.py
```

훈련 로그는 기본적으로 `log/`에 저장됩니다.

| 파일 | 내용 |
|------|------|
| `{dataset}_{timestamp}_train.log` | 전체 실행 로그 (학습·점수 산출·평가·print 전부) |
| `{dataset}_{timestamp}_train_history.json` | step별 loss, valid_loss 등 |
| `{dataset}_{timestamp}_train_history.csv` | 위와 동일 (CSV) |

평가 결과(점수, metrics)는 `result/`에 저장됩니다.

로그 경로 변경:

```bash
python main.py --dataset SMAP --log_dir my_logs
```

설정 변경 예시:

```bash
# SMD
python main.py --dataset machine-1-1 --max_epoch 10 --level 0.005 --device mps

# SMAP / MSL (데이터셋별 level 권장값 사용)
python main.py --dataset SMAP --max_epoch 10 --level 0.07 --device mps
python main.py --dataset MSL --max_epoch 10 --level 0.01 --device mps
```

학습 중 validation loss가 개선될 때마다 **best 모델**이 자동 저장됩니다.

```
model/
├── SMAP/best_model.pt
├── MSL/best_model.pt
└── machine-1-1/best_model.pt
```

저장된 모델로 테스트만 실행:

```bash
python main.py --dataset SMAP --max_epoch 0 --restore_dir model/SMAP --device mps
```

## 평가 지표 해석

실행이 끝나면 `result/metrics.json`과 로그에 **두 종류**의 평가 결과가 함께 출력됩니다.

| 접두사 | 방식 | 설명 |
|--------|------|------|
| `pot-*` | **POT** (Peaks Over Threshold) | 논문·공식 코드에서 주로 사용. **이 지표를 기준으로 보세요.** |
| `best-f1`, `precision`, `recall` 등 | **best-F1 grid search** | 고정 구간 `[-400, 400]`에서 threshold를 탐색 |

### POT 결과를 메인으로 보기

OmniAnomaly는 학습 데이터의 점수 분포로 POT threshold를 정한 뒤 테스트에 적용합니다.  
정상적으로 학습·평가되었다면 아래와 비슷한 수준이 나옵니다.

```
pot-f1:        ~0.9 이상 (데이터셋·학습 상태에 따라 다름)
pot-precision: ~0.9
pot-recall:    ~0.9
pot-FP:        수천 ~ 수만 (전체 테스트 포인트 대비 소수)
```

### best-F1이 이상해 보일 때

SMAP·MSL처럼 **재구성 log-probability 점수 스케일이 매우 큰** 경우(예: `-10^9` 근처),  
기본 `bf_search` 범위 `[-400, 400]`과 맞지 않아 best-F1이 왜곡될 수 있습니다.

증상 예시:

- `recall ≈ 1.0`, `FN = 0` 인데 `precision`이 매우 낮음
- `FP`가 수십만으로 비정상적으로 많음
- `threshold`가 `-399`처럼 **탐색 범위 경계**에 걸림
- 반면 `pot-threshold`는 `-10^9` 근처로 정상적으로 잡힘

이 경우 **모델이 전부 이상으로 예측한 것이 아니라**, best-F1 탐색 범위가 점수 스케일과 맞지 않은 것입니다.  
**`pot-*` 지표를 신뢰**하고, `best-f1` / `precision` / `recall` / `FP`는 참고용으로만 보세요.

### 학습이 충분한지 확인

로그에서 아래를 확인하세요.

- `stopped_epoch`가 `max_epoch`보다 훨씬 작으면 **early stop으로 일찍 종료**된 것입니다.
- `valid_loss`가 `-10^19`처럼 비정상적으로 크면 **수렴 전**일 수 있습니다.

#### Early stop이 1 epoch 안에 걸리는 이유

기본 설정에서 early stop은 **epoch 단위가 아니라 validation 횟수**로 동작합니다.

- `valid_step_freq=100` → 100 step마다 validation 1회
- `early_stop_patience=30` → 30회 연속 개선 없으면 종료
- SMAP은 1 epoch ≈ 1,800 step → 예전 `patience=10`이면 **1 epoch도 채우기 전**에 종료될 수 있음

또한 **초기 step(100 근처)** 의 `valid_loss`가 비정상적으로 작게 나와 best로 고정되면, 이후 개선이 없다고 판단해 patience가 빨리 찹니다.

#### Early stop 관련 기본값 (수정됨)

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `early_stop_min_epochs` | 3 | 최소 이 epoch 수는 학습 |
| `early_stop_patience` | 30 | warmup 이후 validation 30회 동안 개선 없을 때 종료 |
| `early_stop_warmup_steps` | 300 | 초기 300 step은 best/patience 집계 안 함 |

```bash
# early stop 끄기
python main.py --dataset SMAP --no_early_stop --max_epoch 10

# patience / min epoch 조정
python main.py --dataset SMAP --early_stop_patience 50 --early_stop_min_epochs 5
```

권장:

- SMAP·MSL은 `--max_epoch 10` 이상으로 학습
- 데이터셋별 **POT `level` 권장값** 사용 (아래 표 참고)

## 주요 변경 사항 (공식 TensorFlow 버전 대비)

| 항목 | 공식 (TF 1.x) | 이 버전 (PyTorch) |
|------|---------------|-------------------|
| 프레임워크 | TensorFlow + tfsnippet | PyTorch 2.x |
| GPU | CUDA | **MPS** (Mac) / CUDA / CPU |
| 모델 저장 | VariableSaver | `model/{dataset}/best_model.pt` |
| 설정 | tfsnippet Config | `ExpConfig` + argparse |

모델 구조(GRU-VAE, RecurrentDistribution, Planar Flow, POT 평가)는 원본과 동일하게 유지했습니다.

## 디렉터리 구조

```
OmniAnomaly/
├── main.py
├── data_preprocess.py
├── requirements.txt
├── omni_anomaly/
│   ├── model.py
│   ├── vae.py
│   ├── training.py
│   ├── prediction.py
│   └── ...
├── ServerMachineDataset/   # SMD 데이터
├── processed/              # 전처리된 pkl
├── log/                    # 훈련 로그
├── model/                  # 학습된 체크포인트
└── result/                 # 점수 및 평가 결과
```

## POT level 권장값

POT 평가의 `level`은 데이터셋마다 다르게 설정하는 것이 좋습니다.  
기본값은 `0.01`이지만, **SMAP은 `0.07`을 사용하세요.**

| 데이터셋 | level | 실행 예시 |
|----------|-------|-----------|
| SMAP | **0.07** | `python main.py --dataset SMAP --level 0.07` |
| MSL | 0.01 | `python main.py --dataset MSL --level 0.01` |
| SMD group 1 | 0.005 | `python main.py --dataset machine-1-1 --level 0.005` |
| SMD group 2 | 0.0075 | `python main.py --dataset machine-2-1 --level 0.0075` |
| SMD group 3 | 0.0001 | `python main.py --dataset machine-3-1 --level 0.0001` |
