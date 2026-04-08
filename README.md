# Time-Series × LLM Numeric-Text Alignment Research

## Summary
A semantically grounded time-series forecasting framework that uses **numbers as the semantic bridge** between time-series and the language semantic space of LLMs.

## Overview
This project investigates a new time-series–LLM alignment framework in which **numbers serve as the core alignment criterion**. By converting numeric values into natural-language expressions, the framework aligns time-series representations with the language semantic space of LLMs to build more interpretable and semantically grounded forecasting models.## Overview

## Environment Setup
<pre>
Python >= 3.10
CUDA 11.8+ recommended

Dependencies are listed in requirements.txt.

Install all dependencies with:

pip install -r requirements.txt
</pre>

---

## Core Motivation

### Problem
Existing time-series–text alignment methods still suffer from several limitations:

- They often rely on **coarse, channel-wise alignment**, without fully capturing temporal semantics across time.
- **Paired time-series–text data is scarce**, making explicit cross-modal semantic grounding difficult.
- **Attention-based matching** with textual anchors or prototypes does not guarantee true semantic alignment.
- Prompt-based approaches remain limited: some rely on **soft prompts** that are not human-interpretable, while others still suffer from a substantial **modality gap** between time-series embeddings and text representations.

→ As a result, it remains unclear why a specific time-series pattern is linked to a particular textual meaning, limiting interpretability and trustworthiness.

### Key Insight
- **Numbers are the shared semantic unit between time-series and text.**
- Instead of forcing LLMs to directly process raw time-series values, a more principled approach is to **align time-series representations with the language semantic space of LLMs**.
- Converting numeric values into **human-interpretable natural-language expressions** provides a meaningful basis for semantic alignment.

### Our Solution
- Convert **numeric values and timestamps into natural-language prompts**.
- Use **frozen GPT-2 embeddings** as a text-derived semantic reference space.
- Train the **time-series encoder** to align with these semantic embeddings.
- Design an alignment objective that captures both **temporal flow and channel structure**.
- Jointly optimize alignment and forecasting objectives to improve both **interpretability and forecasting performance**.
  
## Key Ideas

- **Numbers as the core alignment criterion**
- **Numeric-to-text prompt conversion**
- **Frozen GPT-2 embeddings as semantic references**
- **Time-series encoder alignment in the LLM space**
- **Temporal flow + channel-aware alignment**
- **Forecasting-based validation of semantic alignment**

---
## System Architecture

### Embedding Pipeline

- `GenPromptEmb` : Numeric values + timestamps → natural-language prompt  
- GPT-2 tokenizer + model → extracts the last-token embedding  
- saves embeddings in `.h5` format (`anchor_avg`, `prompt_token_embeddings` etc.)

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
- Cross-modal alignment without pair supervision
- Semantically map the time-series modality into the LLM embedding space
- 
