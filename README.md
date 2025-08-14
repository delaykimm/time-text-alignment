# TimeCMA-TS-Text Alignment

**TimeCMA-TS-Text Alignment**는 시계열 데이터의 수치 값을 LLM(Text) 임베딩 공간에 정렬(alignment)하여 **시계열 ↔ 자연어 표현**의 공통 표현 공간을 학습하는 프레임워크입니다.  
이 프로젝트는 **시계열 예측 성능 향상**과 **수치-텍스트 의미 연결**을 동시에 달성하는 것을 목표로 합니다.

---

## 📌 연구 개요

### 1. 문제 정의
- 시계열 예측 모델은 수치 데이터에 강하지만, 자연어 기반 LLM은 수치 표현의 의미를 잘 학습하지 못하는 경우가 많음.
- 예: `"23"`이라는 텍스트 토큰이 실제 시계열 값 23.0과 의미적으로 가까워지도록 학습.

### 2. 방법론
1. **시계열 인코더** → 채널별(Time-series) 임베딩 생성  
2. **프롬프트 임베딩(LLM)** → 시계열 데이터를 자연어 문장으로 변환 후 LLM 임베딩 추출  
3. **Cross-Modal Alignment**  
   - **정렬 손실**:  
     - (a) **Cosine Alignment Loss** – 시계열 임베딩 ↔ 해당 시점의 텍스트 토큰 임베딩 평균  
     - (b) **Pairwise Distance MSE** – 두 임베딩 공간의 구조적 유사성 유지  
4. **공통 공간 학습**: 예측 손실 + 정렬 손실의 가중합

---

## 🛠 설치 방법
```bash
git clone https://github.com/username/TimeCMA-TS-Text.git
cd TimeCMA-TS-Text
conda create -n timecma python=3.10
conda activate timecma
pip install -r requirements.txt
```

## 🚀 실행 예시
```bash
python train.py \
  --data_path ./dataset/ETTm1.csv \
  --model_name TimeCMA \
  --seq_len 96 \
  --pred_len 24 \
  --batch_size 32 \
  --learning_rate 0.0005 \
  --align_weight 1.5
```

## 📊 실험 설정
- 데이터셋: ETT(Electricity Transformer Temperature)
- 모델 구조:
      Time-series Encoder (Transformer 기반)
      GPT-2 기반 LLM 임베딩
- 손실 구성:
      **1. Prediction Loss(MAE/MSE) - 시계열 예측**
      **2. Alignment Loss - 수치 <-> 텍스트 정렬 (숫자 토큰 기반)**
        Cosine Alignment
        Pairwise Distance MSE



