# Time-Series × LLM Numeric-Text Alignment Research

## Overview

본 연구는 **시계열 수치 정보(time-series values)를 자연어로 변환하여 LLM 임베딩 공간과 정렬(alignment)** 시키고, 이를 기반으로 **시계열 예측(time-series forecasting)** 성능을 향상시키는 모델 구조를 제안한다.

## Environment Setup
<pre>
Python >= 3.10
CUDA 11.8+ recommended

Dependencies are listed in requirements.txt.

Install all dependencies with:

pip install -r requirements.txt
</pre>

### Key Ideas

- 수치 → 텍스트 변환 후 GPT 임베딩 추출  
- 시계열 encoder와 Text encoder 간 cross-modal alignment  
- time flow + channel 정보를 모두 반영하는 alignment loss  
- forecasting 성능 향상을 통한 정렬 효과 실증

---

## Core Motivation

### Problem

기존 time-series–text alignment 기법들은 다음과 같은 한계를 가진다.

- 시계열 구간 단위의 channel-wise alignment 또는 attention score 학습에 머물러 거친 정렬만 수행  
- 정렬의 기준(reference)이 되는 명시적 의미 기준점 부재  
- 단순 패턴 매칭에 의존하여 의미적 prototype 정렬이 어려움  

→ 결과적으로 왜 특정 시계열 표현이 특정 의미와 연결되는지 해석하기 어렵다.

### Our Solution

1. 수치 값을 자연어 prompt로 변환하여 LLM의 언어 기반 수치 priors 활용  
2. GPT-2 임베딩을 semantic reference space로 사용 (파라미터 freeze)  
3. Time-series encoder만 학습하여 시계열 표현을 LLM 임베딩 공간으로 사상  
4. Time alignment + Channel alignment + Forecasting loss 공동 최적화

---

## System Architecture

### Embedding Pipeline

- `GenPromptEmb` : 숫자 + timestamp → 자연어 prompt 생성  
- GPT-2 tokenizer + model → 마지막 토큰 임베딩 추출  
- `.h5` 형태로 embedding 저장 (`anchor_avg`, `prompt_token_embeddings` 등)

### Prompt Generation Example

The following example illustrates how raw numeric time-series values are converted into natural-language prompts before being encoded by GPT-2.

Raw time-series values:
<pre>
[23, 4, 10.5]
</pre>
Corresponding generated prompt:
<pre>
From [t1] to [t2], the values were twenty three, four, and ten point five every hour.
The total trend value was minus twelve point five.
</pre>
This prompt explicitly expresses numeric values in natural language to leverage the LLM’s pretrained numerical and semantic understanding, providing a semantic reference space for numeric–text alignment.

---

## Project Structure

<pre>
├── data_provider/
│   ├── data_loader_emb.py
│   └── data_loader_save.py
├── layers/
├── models/
├── scripts/
├── storage/
│   ├── gen_prompt_emb.py
│   └── store_emb.py
├── utils/
└── train.py
</pre>

---

## Data Preparation

Download raw datasets from the official repositories:

ETTh1 : https://github.com/zhouhaoyi/ETDataset  
ILI   : https://github.com/thuml/Autoformer

Place the raw CSV files as:

datasets/ETTh1.csv  
datasets/ILI.csv

## Embedding Generation

Run the following command to convert raw time-series values into
numeric–text prompt embeddings stored in HDF5 format.

```bash
python storage/store_emb.py \
  --data_path ETTh1 \
  --divide 96 \
  --save_path embeddings/ETTh1_96.h5
```

## Data Structure (Embedding format)

Each HDF5 file contains numeric–text embeddings
generated from raw time-series values.

Example shape:

(32, 551, 768, 7)
(batch_size, num_tokens, model_dim, channel)

Each time-series sample is segmented into multiple sliding windows, and each window is converted into a natural-language prompt and encoded by GPT-2 at the token level. These multiple prompt-window embeddings are stored per sample in the HDF5 file.

---

## Key Losses

```text
loss = loss_pred 
     + λ_c * loss_align_channel 
     + λ_t * loss_align_time
```

- Forecasting Loss (MSE)  
- Time Alignment Loss   (loss_align_time)
- Channel Alignment Loss   (loss_align_channel)

These three losses are jointly optimized to align temporal structure,
channel-wise semantics, and forecasting accuracy.

---

## Codebase

| File | Description |
|------|-------------|
| train.py | training pipeline |
| data_loader_emb.py | h5-based dataloader |
| data_loader_save.py | embedding generator loader |
| gen_prompt_emb.py | prompt generation |
| store_emb.py | embedding storage |

---
```md
## Quick Start
The following commands reproduce the full pipeline from raw data to forecasting.
```
```bash
# Step 1. Generate numeric-text embeddings
python storage/store_emb.py --data_path ETTh1 --divide train --batch_size 32

# Step 2. Train forecasting model
python train.py --epochs 70 --data_path ETTh1 --use_contrastive --time_window 3 --batch_size 16 --stride 8 --align_weight 0.3 --align_weight_time 0.1
```

## How to Run

### 1. Generate Embeddings

```bash
python ./storage/store_emb.py \
  --data_path ETTh1 \
  --divide train \
  --batch_size 32
```
<pre>
--data_path    : Time-series dataset name
--divide       : Data split mode (train / valid / test)
</pre>

### 2. Train Model
```bash
python train.py \
  --data_path ETTh1 \
  --batch_size 16 \
  --epochs 70 \
  --stride 8 \
  --align_weight 1.0 \
  --align_weight_time 0.5 \
  --time_window 4
```
<pre>
--pred_len            : Prediction horizon length
--data_path           : Dataset name (e.g., ETTh1, ILI)
--batch_size          : Training batch size
--epochs              : Number of training epochs
--stride              : Sliding window shift 
                        (smaller value → more samples, higher computation)
--align_weight        : Weight for channel-wise alignment loss
--align_weight_time   : Weight for temporal alignment loss
--time_window         : Temporal positive range for alignment
                        (e.g., 5 → ±2 time steps considered positive pairs)
</pre>

---

## Baseline

We compare our model with TimeCMA using identical settings:
- batch_size = 16
- epochs = 70
- stride = 8

---

## Expected Result & Analysis

### ETTh1 Dataset (epoch = 70, align_channel = 0.1, align_time = 0.3, stride= 8)

<pre>
pred_len | Metric | TimeCMA | Ours(MY) | Delta
------------------------------------------------
96       | MSE    | 0.3985  | 0.3961   | -0.0024
96       | MAE    | 0.4194  | 0.4178   | -0.0016
192      | MSE    | 0.4529  | 0.4417   | -0.0112
192      | MAE    | 0.4480  | 0.4411   | -0.0069
336      | MSE    | 0.4882  | 0.4848   | -0.0034
336      | MAE    | 0.4636  | 0.4630   | -0.0006
720      | MSE    | 0.5198  | 0.4915   | -0.0283
720      | MAE    | 0.5038  | 0.4808   | -0.0230
</pre>

Analysis
	- MY outperforms TimeCMA across all prediction horizons.
	- Largest gains are observed at pred_len=192 and 720.
	- Maximum improvement at pred_len=720: MSE -0.0283, MAE -0.0230.
This shows that numeric–text alignment remains effective even for long-horizon forecasting on ETTh1.

### ILI (seq_len=36, epoch=100, stride=2)

<pre>
pred_len | Metric | TimeCMA | Ours(MY) | Delta
------------------------------------------------
24       | MSE    | 2.4568  | 1.6837   | -0.7731
24       | MAE    | 0.9561  | 0.8438   | -0.1123
36       | MSE    | 1.9894  | 1.8832   | -0.1062
36       | MAE    | 0.9472  | 0.8842   | -0.0630
48       | MSE    | 2.2464  | 2.3041   | +0.0577
48       | MAE    | 0.9951  | 0.9844   | -0.0107
60       | MSE    | 1.8181  | 2.0267   | +0.2086
60       | MAE    | 0.8754  | 0.9287   | +0.0533
</pre>

Analysis
	- Strong improvements for short horizons (pred_len=24, 36), including a 31.5% MSE reduction at pred_len=24.
	- For pred_len ≥ 48, performance degrades, indicating that alignment is most effective for short-term forecasting under distribution shift.

---

### Why This Matters
- LLM의 수치 이해 한계를 극복
- 시계열–언어 간 의미적 정렬 가능
- 최소 파라미터 학습으로 효율적 구조
  
-----

## Summary

“숫자를 언어 공간으로 끌어올리고, 언어적 의미를 시계열 예측에 투입한다.”

본 연구는 LLM의 언어적 표현 능력을 시계열 문제에 직접 연결하여
alignment 기반 forecasting 성능 향상을 달성하는 것을 목표로 한다.
