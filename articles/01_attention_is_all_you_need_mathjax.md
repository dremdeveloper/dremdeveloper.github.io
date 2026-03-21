---
title: "Attention Is All You Need"
math: true
---

<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    processEscapes: true,
    tags: 'ams'
  },
  options: {
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
  },
  svg: { fontCache: 'global' }
};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>

## 핵심 요약

- Transformer는 RNN이나 CNN 없이 **self-attention만으로 시퀀스 변환 모델을 구성할 수 있다**는 점을 보여 준 논문이다.
- **Query–Key–Value, multi-head attention, positional encoding, feed-forward network**를 하나의 표준 구조로 묶어 이후 LLM의 기본 뼈대를 만들었다.
- 기계번역 실험에서 성능과 학습 효율을 함께 개선했고, 특히 **병렬화가 쉽다**는 점이 큰 강점으로 드러났다.
- 다만 self-attention의 계산량이 시퀀스 길이에 대해 **\(O(n^2)\)** 으로 커지기 때문에, 긴 문맥에서는 여전히 비용 문제가 남는다.

# Attention Is All You Need (arXiv:1706.03762)

원문 논문: Vaswani et al., 2017. https://arxiv.org/pdf/1706.03762<br>생성일(Asia/Seoul): 2026-03-21<br>주의: 본 문서는 원문 내용을 기반으로 한 ‘확장 해설’이며, 원문에 없는 내용은 [추가 설명(일반 지식)] 또는 [추가 아이디어(원문 외)]로 명시했습니다.

※ 요청에 따라 **구현/재현 관점 메모** 및 **프로젝트 적용 지침**에 해당하는 항목은 본 문서에서 제외하였다.

# 목차

- 0. 논문 메타 정보, 사용 허가 문구, 저자/기여 요약
- Abstract
- 1. Introduction
- 2. Background
- 3. Model Architecture
- 4. Why Self-Attention
- 5. Training
- 6. Results
- 7. Conclusion + Acknowledgements
- References (원문 [1]–[40])
- Appendix: Attention Visualizations
- I. 용어집(Glossary)

## 0. 논문 메타 정보, 사용 허가 문구, 저자/기여 요약

### 0.1 제목/저자/소속

제목: Attention Is All You Need

저자: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin

발표 표기: NIPS 2017, Long Beach, CA, USA

### 0.2 표/그림 재사용 허가 문구

표지에 ‘적절한 출처 표기(proper attribution)가 있으면 저널리즘/학술 목적에 한해 표와 그림을 재현할 수 있도록 Google이 허가한다’는 문구가 명시되어 있습니다.

### 0.3 동등 기여/저자 기여(각주) 요약

- 별표(*)는 동등 기여이며, 저자 나열 순서는 랜덤.
- Jakob: RNN을 self-attention으로 대체하는 아이디어 제안 및 초기 평가 시작.
- Ashish & Illia: 최초 Transformer 모델 설계/구현, 전반 작업 관여.
- Noam: scaled dot-product attention, multi-head attention, 파라미터 없는(position) 표현 제안 및 세부 관여.
- Niki: tensor2tensor에서 다양한 변형 모델 설계/구현/튜닝/평가.
- Llion: 변형 실험, 초기 코드베이스, 효율적 추론/시각화 담당.
- Lukasz & Aidan: tensor2tensor 구성요소 설계/구현, 기존 코드베이스 교체로 결과 개선 및 연구 가속.
- †/‡는 Google Brain/Google Research 재직 중 수행을 뜻함.
[추가 설명(일반 지식)] 대형 시스템 논문에서 ‘저자 기여’는 구현/재현 관점에서 코드 출처와 핵심 아이디어 주도자를 파악하는 데 도움됩니다.

## Abstract

### A. 먼저 볼 것

- 질문: attention만으로, RNN/CNN 없이도 seq2seq(번역 등)에서 높은 품질과 병렬화/학습 효율을 얻을 수 있는가?
- 체크포인트: (i) ‘recurrence와 convolution을 배제한다’는 표현의 정확한 의미, (ii) 번역 벤치마크에서의 성능 보고 방식, (iii) 병렬화 및 학습 효율에 관한 주장
### B. 원문 요지

기존의 강력한 sequence transduction 모델은 복잡한 recurrent 또는 convolutional 네트워크 기반의 encoder-decoder 구조이며, 최고 성능 모델은 encoder와 decoder 사이에 attention을 사용합니다.

저자들은 Transformer라는 더 단순한 아키텍처를 제안하며, 이는 attention만을 사용하고 recurrence와 convolution을 전혀 사용하지 않습니다.

두 번역 과제에서 Transformer는 더 좋은 품질, 더 높은 병렬화 가능성, 더 적은 학습 시간을 보였습니다.

WMT14 EN-DE에서 BLEU 28.4를 달성해 기존 최고(ensemble 포함) 대비 2 BLEU 이상 개선했다고 주장합니다.

WMT14 EN-FR에서 single-model SOTA BLEU 41.8을 보고하며, 8 GPU에서 3.5일 학습으로 달성했다고 말합니다.

Transformer는 번역 외에도 영어 constituency parsing에 적용 가능함을 보였다고 합니다.

[추가 설명(일반 지식)] ‘recurrence/conv 배제’는 RNN/CNN 블록을 쓰지 않는다는 뜻이며, 대신 선형층/FFN/LayerNorm/잔차 연결은 존재합니다.

## 1. Introduction

### A. 먼저 볼 것

- 질문: RNN 기반 seq2seq의 병목(순차성)은 무엇이며, attention-only가 왜 필요/가능한가?
- 체크포인트: (i) RNN 기반 seq2seq의 순차성 병목, (ii) 기존 연구에서 attention이 결합되는 방식, (iii) 제안 모델이 주장하는 병렬화 및 학습 효율
### B. 원문 요지

LSTM/GRU 같은 RNN은 language modeling과 machine translation에서 확고한 SOTA였습니다.

RNN은 시간축 위치를 따라 계산이 분해되어, 길이가 길어질수록 병렬화가 어렵고 메모리 제약이 심해집니다.

여러 연구가 계산 효율을 개선했지만 ‘순차 계산’의 근본 제약은 남아 있습니다.

attention은 입력/출력 간 거리에 무관하게 의존성을 모델링할 수 있게 해주지만, 대부분의 경우 RNN과 결합되어 사용되었습니다.

Transformer는 recurrence를 배제하고 attention만으로 global dependency를 모델링하며, 더 큰 병렬화와 더 빠른 학습(8×P100에서 최소 12시간)으로 SOTA에 도달할 수 있다고 주장합니다.

### C. 친절한 직관/예시

RNN은 은닉 상태가 이전 시점의 상태에 의존하므로 시간축 방향의 순차 계산이 필연적으로 발생합니다. 반면 attention 기반 계산은 행렬 연산으로 위치 간 상호작용을 병렬적으로 처리할 수 있어 병렬화에 유리합니다.

주의: attention-only라 해도 decoder의 자기회귀 생성은 토큰 단위 생성이 남습니다(그래서 mask가 필요).

## 2. Background

### A. 먼저 볼 것

- 질문: 순차 계산을 줄이려는 CNN 계열 접근과 self-attention의 차이는?
- 체크포인트: ConvS2S/ByteNet의 거리-연산량 증가, self-attention(intra-attention) 정의, Transformer의 ‘entirely on self-attention’ 주장 범위
### B. 원문 요지

Extended Neural GPU, ByteNet, ConvS2S는 convolution을 기반으로 전체 위치에 대한 표현을 병렬 계산하여 순차성을 줄였습니다.

하지만 임의 두 위치 간 신호를 연결하는 데 필요한 연산 수가 거리와 함께 증가합니다(ConvS2S는 선형, ByteNet은 로그).

Transformer에서는 이 연산 수가 상수로 줄어들지만, attention의 평균화로 인한 해상도 저하 가능성이 있으며 multi-head로 이를 보완한다고 설명합니다.

self-attention(intra-attention)은 시퀀스 내 서로 다른 위치들을 연결해 표현을 계산하는 메커니즘이며, 여러 NLP 과제에서 성공적으로 사용되었습니다.

저자들은 Transformer가 sequence-aligned RNN/conv 없이, 입력과 출력 표현 계산을 entirely self-attention에 의존하는 최초의 transduction 모델이라고 주장합니다(‘우리 지식 범위에서’).

[추가 설명(일반 지식)] 여기서 ‘해상도(resolution)’는 평균화로 정보가 뭉개질 수 있다는 직관이며, 논문은 이를 multi-head로 상쇄한다고만 말합니다(정량 정의는 없음).

## 3. Model Architecture

### A. 먼저 볼 것

- 질문: encoder-decoder 스택 구조는? attention은 어떻게 정의되고 multi-head는 왜 필요한가? 순서 정보는 어떻게 주입하나?
- 체크포인트: Add & Norm(잔차+LayerNorm), decoder의 masked self-attention, 주요 차원 d_model, d_k, d_v, d_ff, h, N
### B. 원문 요지

#### 3.1 Encoder and Decoder Stacks

Transformer는 encoder-decoder 구조를 따릅니다. encoder는 입력 시퀀스를 연속 표현 z_1,...,z_n으로 매핑하고, decoder는 auto-regressive하게 출력 y_1,...,y_m을 생성합니다.

Encoder는 동일한 N=6 레이어 스택이며, 각 레이어는 (multi-head self-attention)과 (position-wise FFN) 두 sub-layer로 구성됩니다.

각 sub-layer는 residual connection과 layer normalization을 사용하며, 형태는 LayerNorm(x + Sublayer(x))입니다.

Residual을 위해 모든 sub-layer와 embedding의 출력 차원을 d_model=512로 맞춥니다.

Decoder도 N=6 레이어이며, masked self-attention + encoder-decoder attention + FFN의 3 sub-layer를 가집니다.

Decoder의 self-attention은 미래 토큰을 보지 못하도록 masking을 적용해 위치 i의 예측이 i보다 작은 출력에만 의존하도록 합니다.

#### 3.2 Attention

##### 3.2.1 Scaled Dot-Product Attention

Attention(Q,K,V) = softmax((QK^T)/sqrt(d_k)) V  (식 (1))

Dot-product attention은 최적화된 행렬곱으로 빠르고 메모리 효율적이며, d_k가 클 때 softmax 포화를 완화하기 위해 1/sqrt(d_k)로 스케일링합니다. 각주에서는 q·k의 분산이 d_k로 증가함을 통해 직관을 제공합니다.

##### 3.2.2 Multi-Head Attention

MultiHead(Q,K,V) = Concat(head_1,...,head_h) W^O, head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)

설정: h=8, d_k=d_v=d_model/h=64. 단일 head는 평균화로 인해 다양한 관계 포착이 억제될 수 있어 multi-head가 유리하다고 설명합니다.

##### 3.2.3 Applications of Attention

- Encoder-Decoder Attention: query는 decoder, key/value는 encoder 출력.
- Encoder Self-Attention: Q/K/V 모두 encoder 이전 레이어 출력.
- Decoder Self-Attention(마스킹): 불법 연결을 softmax 입력에서 -∞로 마스킹.
#### 3.3 Position-wise Feed-Forward Networks

FFN(x) = max(0, xW1 + b1) W2 + b2  (식 (2))

각 위치에 독립적으로 동일 FFN을 적용하며, d_model=512, d_ff=2048. 1×1 convolution 두 번으로도 볼 수 있다고 설명합니다.

#### 3.4 Embeddings and Softmax

입력/출력 토큰을 d_model 차원 embedding으로 변환하고, decoder 출력은 linear + softmax로 다음 토큰 확률을 출력합니다.

입력/출력 embedding과 pre-softmax linear transformation에서 같은 weight matrix를 공유합니다(weight tying).

Embedding에는 sqrt(d_model)을 곱합니다.

#### 3.5 Positional Encoding

recurrence/conv가 없으므로 순서 정보를 주입해야 합니다. 입력 embedding에 positional encoding을 더합니다(같은 차원 d_model).

고정(sinusoidal) PE: PE(pos,2i)=sin(pos/10000^{2i/d_model}), PE(pos,2i+1)=cos(pos/10000^{2i/d_model}).

learned positional embedding도 실험했고 결과는 거의 동일했습니다(Table 3 row E). sinusoids는 더 긴 길이에 extrapolation 가능성을 기대해 선택했다고 말합니다.

#### Table 1: 레이어 타입 비교(요약)

> Table 1 삽입

표의 기호: n=sequence length, d=representation dimension, k=kernel size, r=restricted neighborhood size. 이 표는 이론적 비교이며 실제 속도는 구현/하드웨어에 좌우될 수 있습니다.

#### 표기(Notation) 요약

| 기호 | 의미(원문 요지) | 쉬운 설명 |
| --- | --- | --- |
| x1..xn | 입력 심볼 시퀀스 | 입력 토큰들 |
| z1..zn | encoder 출력 표현 | 입력 위치별 hidden |
| y1..ym | decoder 출력 시퀀스 | 출력 토큰들 |
| N | 레이어 수(예: 6) | 블록 반복 수 |
| d_model | 모델 차원(예: 512) | 표현 폭 |
| h | head 수(예: 8) | 병렬 attention 개수 |
| d_k, d_v | key/query, value 차원(예: 64) | head 내부 차원 |
| d_ff | FFN 내부 차원(예: 2048) | MLP 확장 폭 |

## 4. Why Self-Attention

### A. 먼저 볼 것

- 비교 기준 3가지: (1) 레이어당 계산 복잡도, (2) 병렬화 가능성(최소 순차 연산 수), (3) 장거리 의존성 path length
- Table 1의 세 열이 위 기준과 대응되며, restricted attention은 긴 시퀀스의 미래 과제로 언급됨
### B. 원문 요지

self-attention, recurrent, convolutional layer를 (복잡도/순차성/경로 길이)로 비교합니다.

장거리 의존성 학습은 forward/backward 신호가 지나야 하는 경로가 짧을수록 쉬울 수 있다고 말하며, 최대 경로 길이를 비교합니다.

self-attention은 상수 번의 순차 연산으로 모든 위치를 연결하지만, recurrent는 O(n)의 순차 연산이 필요합니다.

n < d일 때 self-attention이 recurrent보다 빠르다고 주장하며, word-piece/BPE를 쓰는 번역에서 이런 조건이 흔하다고 언급합니다.

긴 시퀀스에서는 neighborhood r로 제한한 restricted attention을 제안하며, 이때 경로 길이는 O(n/r)로 증가한다고 합니다(미래 연구 과제).

해석 가능성 측면에서 attention distribution을 검사할 수 있고, appendix에 예시를 제시한다고 합니다.

## 5. Training

### 5.1 Training Data and Batching

WMT14 EN-DE: 약 4.5M 문장쌍. EN-FR: 36M 문장.

EN-DE는 BPE 사용, source-target 공유 vocab 약 37k. EN-FR은 32k word-piece vocab.

배치는 길이가 비슷한 문장쌍끼리 묶고, batch당 source 약 25k 토큰 + target 약 25k 토큰이 되도록 구성.

### 5.2 Hardware and Schedule

8×NVIDIA P100 GPU 한 대의 머신에서 학습.

base: step time 약 0.4s, 100k steps, 또는 12시간.

big: step time 약 1.0s, 300k steps, 또는 3.5일.

### 5.3 Optimizer

Adam 사용: β1=0.9, β2=0.98, ε=1e-9.

Learning rate schedule (식 (3)):<br>  lrate = d_model^{-0.5} * min(step^{-0.5}, step * warmup_steps^{-1.5}), warmup_steps=4000

### 5.4 Regularization

본문은 ‘세 가지 정규화’를 사용한다고 말합니다.

명시적으로 설명된 항목: (1) Residual Dropout: sub-layer 출력에 dropout 적용 후 residual add & layer norm, 그리고 embedding+positional encoding 합에도 dropout 적용. base dropout=0.1.

(2) Label Smoothing: ε_ls=0.1. perplexity는 나빠질 수 있으나 accuracy/BLEU가 개선됨.

‘세 번째 정규화’가 무엇인지 본문에서 명확히 항목화되어 설명되지는 않습니다(논문에 명시 없음).

## 6. Results

### 6.1 Machine Translation

WMT14 EN-DE: Transformer(big)이 BLEU 28.4로 기존 최고(ensemble 포함) 대비 2 BLEU 이상 개선했다고 주장. 8×P100에서 3.5일 학습.

WMT14 EN-FR: 본문 서술에 BLEU 41.0이 등장하지만, Table 2에는 Transformer(big) BLEU 41.8로 제시됨(원문 내 불일치; 명시적 해소 없음).

Decoding: base는 마지막 5개 체크포인트 평균, big은 마지막 20개 평균(각 10분 간격 저장). beam size=4, length penalty α=0.6. max output length는 input length + 50(가능하면 조기 종료).

#### Table 2: BLEU 및 Training Cost(FLOPs)

> Table 2 삽입

Training cost는 학습시간×GPU 수×GPU의 sustained FP32 TFLOPS 추정치로 계산합니다. (각주에서 K80=2.8, K40=3.7, M40=6.0, P100=9.5 TFLOPS를 사용했다고 명시)

### 6.2 Model Variations

base 모델을 여러 방식으로 변형하고 EN-DE dev(newstest2013)에서 PPL/BLEU/파라미터 수를 보고합니다.

Table 3의 perplexity는 wordpiece 기준이며 per-word perplexity와 비교하면 안 된다고 경고합니다.

Table 3의 variation 실험은 beam search를 사용하지만 checkpoint averaging은 하지 않습니다.

#### Table 3 핵심 요약(행별)

> Table 3 삽입

- Base: N=6, d_model=512, d_ff=2048, h=8, d_k=d_v=64, P_drop=0.1, ε_ls=0.1, 100K steps, PPL=4.92, BLEU=25.8, params=65M
- (A) head 수 변화: h=1은 24.9 BLEU로 best 대비 0.9 낮음. head 수가 과도하게 커질 경우 성능 저하가 관찰됩니다.
- (B) d_k 축소: d_k=16/32에서 BLEU가 감소. compatibility 결정이 쉽지 않다는 해석.
- (C) 모델 크기 변화: 깊이/폭/FFN을 늘리면 대체로 BLEU 향상.
- (D) dropout/label smoothing: dropout=0은 BLEU 저하; label smoothing 값 변화는 BLEU에 영향.
- (E) learned positional embedding은 sinusoidal과 거의 비슷한 성능.
- Big(참고): N=6, d_model=1024, d_ff=4096, h=16, P_drop=0.3, 300K steps, PPL=4.33, BLEU=26.4, params=213M
### 6.3 English Constituency Parsing

Transformer 일반화를 보기 위해 영어 constituency parsing에 적용합니다. 출력이 구조 제약이 강하고 입력보다 길어질 수 있는 과제입니다.

모델: 4-layer Transformer, d_model=1024. 데이터: WSJ 약 40K 문장, semi-supervised로 약 17M 문장 추가.

vocab: WSJ only 16K, semi-supervised 32K.

dev(Section 22)에서 dropout(‘attention & residual’), learning rate, beam size만 소수 실험으로 선택하고 나머지는 EN-DE base와 동일하게 유지.

추론: max output length = input+300, beam=21, length penalty α=0.3.

#### Table 4: WSJ Section 23 F1

> Table 4 삽입

## 7. Conclusion + Acknowledgements

Transformer는 self-attention만으로 구성된 sequence transduction 모델로서, recurrence를 multi-head self-attention으로 대체했다고 요약합니다.

WMT14 EN-DE 및 EN-FR에서 새로운 SOTA를 달성했고, 특히 EN-DE에서는 기존 모든 앙상블보다도 좋았다고 강조합니다.

미래 과제: 다른 과제/다른 modality(이미지/오디오/비디오)로 확장, 큰 입출력을 효율적으로 처리하기 위한 local/restricted attention, 그리고 생성(generation)을 덜 순차적으로 만드는 연구.

코드는 Tensor2Tensor에 있다고 밝힙니다: https://github.com/tensorflow/tensor2tensor

Acknowledgements: Nal Kalchbrenner, Stephan Gouws에게 감사.

## References (원문 [1]–[40])

1. [1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv:1607.06450, 2016.
1. [2] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. CoRR abs/1409.0473, 2014.
1. [3] Denny Britz, Anna Goldie, Minh-Thang Luong, and Quoc V. Le. Massive exploration of neural machine translation architectures. CoRR abs/1703.03906, 2017.
1. [4] Jianpeng Cheng, Li Dong, and Mirella Lapata. Long short-term memory-networks for machine reading. arXiv:1601.06733, 2016.
1. [5] Kyunghyun Cho et al. Learning phrase representations using rnn encoder-decoder for statistical machine translation. CoRR abs/1406.1078, 2014.
1. [6] Francois Chollet. Xception: Deep learning with depthwise separable convolutions. arXiv:1610.02357, 2016.
1. [7] Junyoung Chung et al. Empirical evaluation of gated recurrent neural networks on sequence modeling. CoRR abs/1412.3555, 2014.
1. [8] Chris Dyer et al. Recurrent neural network grammars. NAACL, 2016.
1. [9] Jonas Gehring et al. Convolutional sequence to sequence learning. arXiv:1705.03122v2, 2017.
1. [10] Alex Graves. Generating sequences with recurrent neural networks. arXiv:1308.0850, 2013.
1. [11] Kaiming He et al. Deep residual learning for image recognition. CVPR, 2016.
1. [12] Sepp Hochreiter et al. Gradient flow in recurrent nets: the difficulty of learning long-term dependencies, 2001.
1. [13] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural Computation, 1997.
1. [14] Zhongqiang Huang and Mary Harper. Self-training PCFG grammars with latent annotations across languages. EMNLP, 2009.
1. [15] Rafal Jozefowicz et al. Exploring the limits of language modeling. arXiv:1602.02410, 2016.
1. [16] Łukasz Kaiser and Samy Bengio. Can active memory replace attention? NeurIPS, 2016.
1. [17] Łukasz Kaiser and Ilya Sutskever. Neural GPUs learn algorithms. ICLR, 2016.
1. [18] Nal Kalchbrenner et al. Neural machine translation in linear time. arXiv:1610.10099v2, 2017.
1. [19] Yoon Kim et al. Structured attention networks. ICLR, 2017.
1. [20] Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. ICLR, 2015.
1. [21] Oleksii Kuchaiev and Boris Ginsburg. Factorization tricks for LSTM networks. arXiv:1703.10722, 2017.
1. [22] Zhouhan Lin et al. A structured self-attentive sentence embedding. arXiv:1703.03130, 2017.
1. [23] Minh-Thang Luong et al. Multi-task sequence to sequence learning. arXiv:1511.06114, 2015.
1. [24] Minh-Thang Luong et al. Effective approaches to attention-based neural machine translation. arXiv:1508.04025, 2015.
1. [25] Mitchell P Marcus et al. Building a large annotated corpus of English: The Penn Treebank. Computational Linguistics, 1993.
1. [26] David McClosky et al. Effective self-training for parsing. NAACL-HLT, 2006.
1. [27] Ankur Parikh et al. A decomposable attention model. EMNLP, 2016.
1. [28] Romain Paulus et al. A deep reinforced model for abstractive summarization. arXiv:1705.04304, 2017.
1. [29] Slav Petrov et al. Learning accurate, compact, and interpretable tree annotation. ACL/COLING, 2006.
1. [30] Ofir Press and Lior Wolf. Using the output embedding to improve language models. arXiv:1608.05859, 2016.
1. [31] Rico Sennrich et al. Neural machine translation of rare words with subword units. arXiv:1508.07909, 2015.
1. [32] Noam Shazeer et al. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv:1701.06538, 2017.
1. [33] Nitish Srivastava et al. Dropout: a simple way to prevent neural networks from overfitting. JMLR, 2014.
1. [34] Sainbayar Sukhbaatar et al. End-to-end memory networks. NeurIPS 28, 2015.
1. [35] Ilya Sutskever et al. Sequence to sequence learning with neural networks. NeurIPS, 2014.
1. [36] Christian Szegedy et al. Rethinking the inception architecture for computer vision. CoRR abs/1512.00567, 2015.
1. [37] Vinyals & Kaiser et al. Grammar as a foreign language. NeurIPS, 2015.
1. [38] Yonghui Wu et al. Google’s neural machine translation system: Bridging the gap between human and machine translation. arXiv:1609.08144, 2016.
1. [39] Jie Zhou et al. Deep recurrent models with fast-forward connections for neural machine translation. CoRR abs/1606.04199, 2016.
1. [40] Muhua Zhu et al. Fast and accurate shift-reduce constituent parsing. ACL, 2013.
## Appendix: Attention Visualizations

### Figure 3

> Figure 3 삽입

Encoder self-attention layer 5/6에서 장거리 의존을 따라가는 예시. ‘making … more difficult’와 같은 장거리 관계에 여러 head가 관여하는 모습을 보이며, ‘making’ 단어에 대해서만 attention을 표시했다고 설명합니다. 색은 head를 의미하며 컬러로 보는 것이 좋다고 언급합니다.

### Figure 4

> Figure 4 삽입

layer 5/6의 두 head가 anaphora resolution(대명사 지시 해소)에 관여하는 것으로 보인다고 설명. Top은 head 5 전체 attention, Bottom은 ‘its’에서 나온 attention을 head 5/6으로 분리해 보여주며, ‘its’의 attention이 매우 sharp하다고 코멘트합니다.

### Figure 5

> Figure 5 삽입

많은 head가 문장 구조와 관련된 행동을 보이며 서로 다른 head가 서로 다른 task를 수행하도록 학습되었다고 결론짓습니다.

[주의] 원문은 ‘apparently involved’ 등 조심스러운 표현을 사용하며, 이 시각화는 정량적 증명이라기보다 관찰/해석 도구로 보는 것이 안전합니다.
