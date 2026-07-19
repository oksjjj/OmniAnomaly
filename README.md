# OmniAnomaly (PyTorch)

[NetManAIOps/OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly) (KDD 2019) 공식 TensorFlow 구현을 **알고리즘 수준에서 그대로** PyTorch로 포팅한 버전입니다.

## 원본과의 대응

| 항목 | 공식 (TF 1.12 + tfsnippet) | 이 버전 (PyTorch) |
|------|---------------------------|-------------------|
| 프레임워크 | TensorFlow + tfsnippet + TFP | PyTorch 2.x |
| 학습 손실 | `mean(log q − log p(x\|z))` (prior 제외) | **`mean(log q − log p(x\|z) − log p(z))`** (GSSM prior 포함) |
| Posterior | RecurrentDistribution + Planar NF (`u_hat`) | 동일 |
| Prior | LinearGaussianStateSpaceModel | Identity GSSM |
| Early stop | TrainLoop: best valid 가중치만 복원 | 동일 (patience로 epoch 중단 없음) |
| Grad clip | `tf.clip_by_norm` (텐서별) | 텐서별 clip |
| L2 | ExpConfig에만 있고 그래프에 미적용 | `weight_decay=0` |
| 디바이스 | CUDA | MPS / CUDA / CPU |

의도적인 수정:
1. 원본 `RecurrentDistribution.log_prob_step`은 `[z_t, input_q]`로 concat 하는데, sampling은 `[input_q, z_{t-1}]`입니다. 그대로 두면 SGVB가 붕괴합니다. 이 포팅은 **sampling과 동일한 conditioning**으로 density를 계산합니다.
2. 원본 `OmniAnomaly.get_training_loss`는 `log p(z)`(GSSM)를 SGVB에서 빼지만, 논문의 Linear Gaussian State Space connection을 살리기 위해 이 포팅은 **`log_joint = log p(x|z) + log p(z)`** 로 학습합니다.

## 환경 설정

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 데이터 준비

### SMD

```bash
python download_smd.py
python data_preprocess.py SMD
```

### SMAP / MSL

```bash
python download_smap_msl.py
python data_preprocess.py SMAP
python data_preprocess.py MSL
```

## 실행

```bash
python main.py
```

설정은 원본과 같은 `ExpConfig` 기본값을 사용합니다.

```bash
# SMD
python main.py --dataset machine-1-1 --max_epoch 10 --level 0.005

# SMAP / MSL
python main.py --dataset SMAP --max_epoch 10 --level 0.07
python main.py --dataset MSL --max_epoch 10 --level 0.01
```

결과·점수는 `result/`, 체크포인트는 `model/{dataset}/`에 저장됩니다.

CUDA에서 실행:

```bash
python main.py --dataset SMAP --max_epoch 10 --level 0.07 --device cuda
```

### TensorBoard

학습 중 `loss` / `valid_loss` / `lr` 등이 `log/tensorboard/{dataset}/{timestamp}/`에 기록됩니다.

```bash
tensorboard --logdir log/tensorboard
```

브라우저에서 `http://localhost:6006` 을 여세요. 끄려면 `--no_tensorboard`.

## POT level 권장값 (원본 README)

| 데이터셋 | level |
|----------|-------|
| SMAP | 0.07 |
| MSL | 0.01 |
| SMD group 1 | 0.005 |
| SMD group 2 | 0.0075 |
| SMD group 3 | 0.0001 |

## 디렉터리

```
OmniAnomaly/
├── main.py
├── data_preprocess.py
├── omni_anomaly/
│   ├── model.py
│   ├── vae.py
│   ├── recurrent_distribution.py
│   ├── flows.py
│   ├── training.py
│   ├── prediction.py
│   └── ...
├── ServerMachineDataset/
├── processed/
└── result/
```
