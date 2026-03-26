---
title: "Transformer의 시작 — Attention Is All You Need"
math: true
---


# Attention Is All You Need

Ashish Vaswani\*, Noam Shazeer\*, Niki Parmar\*, Jakob Uszkoreit\*, Llion Jones\*, Aidan N. Gomez\*†, Łukasz Kaiser\*, Illia Polosukhin\*‡

Google Brain / Google Research / University of Toronto

31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA.

## Abstract

기존의 지배적인 시퀀스 변환 모델은 encoder와 decoder를 포함하는 복잡한 recurrent network나 convolutional network에 기대고 있었다. 가장 성능이 좋은 모델들도 대개 encoder와 decoder 사이를 attention mechanism으로 이어 주는 형태였다. 이 논문은 recurrence와 convolution을 완전히 걷어내고, attention mechanism만으로 동작하는 새로운 구조인 Transformer를 제안한다.

저자들이 내세우는 핵심 주장은 두 가지다. 첫째, attention만으로도 번역 같은 대표적인 sequence transduction 문제에서 충분히 강력한 모델을 만들 수 있다는 점이다. 둘째, 이렇게 하면 계산이 시간축을 따라 순차적으로 흘러가던 구조를 행렬 연산 중심의 병렬 구조로 바꿀 수 있기 때문에, 학습 속도와 확장성에서 큰 이점을 얻을 수 있다는 점이다.

실험에서는 두 가지 기계번역 과제에서 품질과 학습 효율이 모두 좋아졌다고 보고한다. WMT 2014 English-to-German에서는 BLEU 28.4를 기록해 당시 최고 성능보다 2 BLEU 이상 높았고, WMT 2014 English-to-French에서는 8개의 GPU에서 3.5일 학습 후 single-model 기준 41.8 BLEU를 달성했다고 적는다. 또한 Transformer가 대규모 번역 데이터뿐 아니라 영어 constituency parsing 같은 다른 구조적 과제에도 잘 일반화된다고 주장한다.

\* 동등 기여이며 저자 나열 순서는 랜덤이다. Jakob Uszkoreit은 RNN을 self-attention으로 대체하자는 아이디어를 제안하고 그 가능성을 평가하기 시작했다. Ashish Vaswani와 Illia Polosukhin은 최초 Transformer 모델을 설계하고 구현했다. Noam Shazeer는 scaled dot-product attention, multi-head attention, 그리고 파라미터 없는 위치 표현을 제안했다. Niki Parmar는 원래 코드베이스와 tensor2tensor에서 수많은 변형을 설계하고 구현하고 튜닝하고 평가했다. Llion Jones는 새로운 변형 실험, 초기 코드베이스, 효율적인 추론과 시각화를 담당했다. Łukasz Kaiser와 Aidan N. Gomez는 tensor2tensor의 여러 부분을 설계하고 구현하여 결과를 개선하고 연구 속도를 크게 끌어올렸다.  
† Google Brain 재직 중 수행.  
‡ Google Research 재직 중 수행.

## 1. Introduction

Recurrent neural network, 특히 long short-term memory와 gated recurrent neural network는 language modeling과 machine translation 같은 sequence modeling 및 transduction 문제에서 오랫동안 사실상의 표준 구조로 자리 잡아 왔다. 이후 연구들도 대부분 이 계열을 더 깊고 더 강하게 만들거나, encoder-decoder 구조 위에 attention을 얹어 성능 한계를 밀어붙이는 방향으로 전개되었다.

하지만 RNN 계열 구조에는 쉽게 사라지지 않는 병목이 있다. 입력과 출력의 각 위치에 대한 계산이 시간축을 따라 분해되기 때문에, 현재 시점의 hidden state를 만들려면 직전 시점의 hidden state가 먼저 준비되어 있어야 한다. 이런 순차성은 한 문장 내부 계산을 병렬화하기 어렵게 만들고, 문장이 길어질수록 메모리 사용량과 학습 시간이 함께 커진다. 배치 크기를 키우는 방식으로도 이 한계를 완전히 상쇄하기 어렵다.

Attention mechanism은 이 문제에 대한 중요한 실마리를 제공했다. Attention은 입력이나 출력에서 멀리 떨어진 위치 사이의 의존성을 직접 연결해 줄 수 있기 때문에, 장거리 의존성 문제를 다룰 때 매우 유용하다. 다만 당시 대부분의 모델에서 attention은 RNN 위에 얹힌 보조 장치였다. 다시 말해, attention이 중요하다고 해도 계산의 뼈대는 여전히 recurrence가 담당하고 있었다.

이 논문은 그 관점을 한 번 더 밀어붙인다. 정말로 필요한 것이 recurrence가 아니라면, sequence transduction 전체를 attention만으로 다시 세울 수 있지 않을까라는 질문에서 출발한다. 저자들은 Transformer가 recurrence 없이도 입력과 출력 사이의 global dependency를 다룰 수 있으며, 그 결과 훨씬 더 높은 병렬화와 더 빠른 학습을 얻을 수 있다고 주장한다. 논문이 말하는 새로움은 attention을 추가했다는 데 있지 않고, attention만으로도 encoder-decoder의 핵심 계산을 구성할 수 있다고 보여 준 데 있다.

## 2. Background

순차 계산을 줄이려는 시도 자체는 이 논문이 처음은 아니다. Extended Neural GPU, ByteNet, ConvS2S 같은 모델들도 입력과 출력의 모든 위치에 대한 hidden representation을 병렬로 계산하려는 목적을 갖고 있었고, 이를 위해 convolutional network를 기본 블록으로 사용했다. 이런 구조는 RNN보다 병렬화하기 쉽지만, 임의의 두 위치 사이의 정보를 연결하는 데 필요한 연산 수가 위치 간 거리와 함께 증가한다는 약점이 있다. ConvS2S에서는 그 증가가 선형적이고, ByteNet에서는 로그적으로 완화되지만, 어쨌든 멀리 떨어진 위치 사이의 신호는 여러 단계를 거쳐야 한다.

Transformer는 이 연결 길이를 더 줄이려 한다. Self-attention에서는 한 위치가 같은 시퀀스 안의 다른 모든 위치를 직접 바라볼 수 있기 때문에, 이론적으로 임의의 두 위치 사이를 연결하는 데 필요한 연산 수를 상수 수준으로 만들 수 있다. 다만 그 대가도 있다. 여러 위치의 value를 가중합하는 과정에서 정보가 평균화되므로, 세밀한 구분이 흐려질 수 있다. 저자들은 이 문제를 Multi-Head Attention으로 완화할 수 있다고 본다.

여기서 self-attention은 하나의 시퀀스 내부에서 서로 다른 위치를 서로 관련지어 새로운 표현을 만드는 메커니즘을 뜻한다. 논문은 이를 intra-attention이라고도 부른다. 이 방식은 reading comprehension, abstractive summarization, textual entailment, 문장 표현 학습 등 다양한 과제에서 이미 유용성이 확인되어 있었다. 즉 Transformer가 완전히 새로운 primitive를 만든 것은 아니고, 이미 효과가 알려진 self-attention을 sequence transduction의 중심 구조로 끌어올렸다고 보는 편이 정확하다.

저자들은 자신들의 지식 범위에서 Transformer가 sequence-aligned RNN이나 convolution을 사용하지 않고, 입력과 출력 표현을 entirely self-attention에 의존해 계산하는 최초의 transduction model이라고 말한다. 이어지는 절에서는 이 주장을 뒷받침하기 위해 구조를 설명하고, 왜 self-attention이 합리적인 선택인지 계산 복잡도와 경로 길이 관점에서 논증하고, 실제 실험으로 결과를 제시한다.

## 3. Model Architecture

당시의 강력한 neural sequence transduction 모델 대부분은 encoder-decoder 구조를 따른다. Encoder는 입력 시퀀스의 기호 표현 $(x_1, \ldots, x_n)$을 연속 표현 시퀀스 $z = (z_1, \ldots, z_n)$으로 바꾸고, decoder는 이 $z$를 바탕으로 출력 시퀀스 $(y_1, \ldots, y_m)$를 한 원소씩 생성한다. 각 단계에서 decoder는 auto-regressive하게 동작하므로, 다음 토큰을 만들 때 이전에 생성한 토큰들을 추가 입력으로 사용한다.

Transformer도 이 encoder-decoder의 큰 틀은 유지한다. 달라지는 것은 내부 연산 방식이다. Encoder와 decoder 모두를 stacked self-attention과 point-wise fully connected layer의 조합으로 구성하고, recurrence 없이도 입력과 출력을 연결하는 전체 경로를 만든다.

<figure>
  <img src="/assets/images/attention_is_all_you_need/attention_is_all_you_need_figure_01.png" alt="Figure 1. Transformer model architecture." />
  <figcaption><strong>Figure 1.</strong> The Transformer model architecture from the paper.</figcaption>
</figure>

이 구성도를 따라가면 왼쪽이 encoder stack이고 오른쪽이 decoder stack이다. 아래쪽에서는 입력 토큰과 출력 토큰을 각각 embedding으로 바꾸고 positional encoding을 더한다. Encoder 쪽 블록은 `Multi-Head Attention`과 `Feed Forward`가 한 층을 이루며 이것이 $N$번 반복된다. Decoder 쪽 블록은 여기에 `Masked Multi-Head Attention`이 먼저 들어가고, 그 위에서 encoder의 출력을 바라보는 또 하나의 `Multi-Head Attention`이 추가된다. 맨 위에서는 linear layer와 softmax를 통해 다음 토큰의 확률을 만든다. 구조도 한 장만 보아도, 이 모델이 attention과 position-wise feed-forward를 반복해 전체 계산을 쌓아 올리는 방식이라는 점이 분명하게 드러난다.

### 3.1 Encoder and Decoder Stacks

Encoder는 동일한 layer를 $N = 6$개 쌓은 구조다. 각 layer는 두 개의 sub-layer를 가진다. 첫 번째 sub-layer는 multi-head self-attention이고, 두 번째 sub-layer는 간단한 position-wise fully connected feed-forward network다. 두 sub-layer 각각 바깥에는 residual connection이 둘러져 있고, 그 다음에 layer normalization이 적용된다. 논문에서 이 형태를 직접 식으로 적으면 다음과 같다.

$$
\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))
$$

이 식의 의미는 단순하다. 먼저 어떤 sub-layer가 입력 $x$를 변환한 결과를 만든 다음, 그 결과를 원래 입력 $x$와 더하고, 그 합을 정규화한다. Residual connection은 깊은 네트워크에서 정보와 gradient가 더 안정적으로 흐르도록 돕고, layer normalization은 표현의 스케일을 조절해 학습을 더 수월하게 만든다. 이런 잔차 연결을 자연스럽게 적용하려면 더하는 두 벡터의 차원이 같아야 하므로, 논문은 모델 전체의 공통 표현 차원으로 $d_{model} = 512$를 둔다. Embedding layer와 모든 sub-layer가 이 차원을 공유하는 이유가 여기에 있다.

Decoder도 동일한 layer를 $N = 6$개 쌓는다. 다만 encoder와 달리 각 layer가 세 개의 sub-layer를 가진다. 첫 번째는 masked self-attention, 두 번째는 encoder-decoder attention, 세 번째는 position-wise feed-forward network다. Decoder에서도 각 sub-layer마다 residual connection과 layer normalization을 동일하게 사용한다.

여기서 가장 중요한 차이는 첫 번째 self-attention에 mask가 들어간다는 점이다. Decoder는 다음 토큰을 예측할 때 미래 정답을 미리 보면 안 되므로, 위치 $i$에서의 계산은 오직 $i$ 이하의 이미 알려진 출력에만 접근할 수 있어야 한다. 논문은 출력 embedding을 한 칸 오른쪽으로 민 입력과 self-attention mask를 함께 사용해 이 조건을 만족시킨다. 결과적으로 decoder는 학습 중에도 추론 시와 같은 정보 제약을 유지하면서 auto-regressive하게 동작한다.

### 3.2 Attention

Attention function은 query와 key-value pair들의 집합을 입력으로 받아 하나의 출력을 만드는 함수로 볼 수 있다. Query, key, value, 그리고 output은 모두 벡터다. 출력은 value들의 가중합으로 계산되며, 각 value에 붙는 가중치는 query와 해당 key가 얼마나 잘 맞는지를 나타내는 compatibility function으로 정해진다. 직관적으로 말하면 query는 지금 무엇을 찾고 있는지를, key는 각 위치가 어떤 종류의 정보를 제공할 수 있는지를, value는 실제로 가져올 내용을 담고 있다고 볼 수 있다.

<figure>
  <img src="/assets/images/attention_is_all_you_need/attention_is_all_you_need_figure_02.png" alt="Figure 2. Scaled dot-product attention and multi-head attention." />
  <figcaption><strong>Figure 2.</strong> Scaled dot-product attention and multi-head attention.</figcaption>
</figure>

왼쪽 도식은 Scaled Dot-Product Attention의 계산 순서를 보여 준다. Query와 key를 먼저 곱해 점수를 만든 뒤, 이 점수를 스케일링하고 필요하면 mask를 적용하고 softmax로 확률 분포를 만든 다음, 마지막에 value를 가중합한다. 오른쪽 도식은 이 과정을 여러 개의 head에서 병렬로 수행한 뒤 concat과 linear projection으로 다시 합치는 구조를 나타낸다. Transformer의 attention이 단일 분포 하나로 끝나는 것이 아니라, 여러 표현 공간에서 병렬로 관계를 읽어 오는 구조라는 점이 이 그림에서 가장 잘 드러난다.

#### 3.2.1 Scaled Dot-Product Attention

저자들은 자신들의 기본 attention을 Scaled Dot-Product Attention이라고 부른다. 입력은 차원 $d_k$의 query와 key, 그리고 차원 $d_v$의 value다. 각 query와 모든 key의 dot product를 계산하면, 그 query가 각각의 key와 얼마나 잘 맞는지에 대한 점수를 얻을 수 있다. 이 점수를 그대로 softmax에 넣지 않고 $\sqrt{d_k}$로 나누어 스케일을 줄인 뒤 softmax를 취하면, 모든 key에 대한 정규화된 가중치가 나온다. 마지막으로 이 가중치로 value를 가중합해 출력 벡터를 얻는다.

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\tag{1}
$$

이 식을 행렬 관점에서 보면 의미가 훨씬 분명해진다. $Q$의 각 행은 하나의 query이고, $K$의 각 행은 하나의 key다. 따라서 $QK^T$의 $(i, j)$ 원소는 $i$번째 query와 $j$번째 key의 유사도를 뜻한다. 여기에 row-wise softmax를 적용하면, 각 query가 모든 key를 어떻게 가중할지에 대한 분포가 생긴다. 마지막으로 이 분포를 $V$에 곱하면, 각 query마다 모든 value를 어떤 비율로 섞을지가 결정되어 새로운 출력 표현이 만들어진다. 한 문장 안의 각 위치가 다른 모든 위치에서 어떤 정보를 얼마나 가져올지를 한 번의 행렬 연산으로 계산하는 셈이다.

논문은 additive attention과 dot-product attention을 비교한다. Additive attention은 작은 feed-forward network로 compatibility를 계산하고, dot-product attention은 말 그대로 내적으로 계산한다. 이론적인 복잡도는 비슷하지만, dot-product attention은 고도로 최적화된 matrix multiplication으로 구현할 수 있어 실제로 훨씬 빠르고 공간 효율적이다. Transformer가 대규모 병렬 학습에 유리한 이유도 여기에서 나온다.

그렇다고 dot-product만으로 모든 문제가 해결되는 것은 아니다. $d_k$가 커지면 dot product의 절댓값도 커지는 경향이 있어서 softmax가 지나치게 뾰족해지고, 그 결과 gradient가 매우 작아지는 구간에 쉽게 들어갈 수 있다. 논문 각주에서는 $q$와 $k$의 각 성분이 평균 0, 분산 1인 독립 확률변수라고 가정하면 $q \cdot k$의 분산이 $d_k$가 된다고 설명한다. 즉 차원이 커질수록 점수의 스케일이 자연스럽게 커진다. $\sqrt{d_k}$로 나누는 이유는 바로 이 효과를 완화해서 학습을 안정시키기 위해서다.

#### 3.2.2 Multi-Head Attention

단일 attention 하나만 사용하면 모든 관계를 하나의 attention map에 압축해야 한다. 그러면 서로 다른 종류의 관계가 같은 분포 안에서 평균화되어 버릴 수 있다. 논문은 이 문제를 피하기 위해 query, key, value를 여러 개의 서로 다른 표현 공간으로 선형 투영한 다음, 각 공간에서 attention을 병렬로 수행하는 Multi-Head Attention을 사용한다.

$$
\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(head_1, \ldots, head_h)W^O
$$

$$
head_i = \mathrm{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

여기서 $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$, $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$, $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$는 $i$번째 head의 투영 행렬이고, $W^O \in \mathbb{R}^{hd_v \times d_{model}}$는 모든 head의 출력을 다시 합쳐 원래 차원으로 되돌리는 행렬이다.

이 식이 말하는 바는 분명하다. 먼저 하나의 입력 표현을 여러 head에 맞게 서로 다르게 읽는다. 각 head는 자기만의 투영 행렬을 통해 query, key, value를 만든 뒤 attention을 수행하므로, 결과적으로 서로 다른 관계 유형에 민감한 분포를 학습할 수 있다. 어떤 head는 장거리 의존성에, 어떤 head는 구문적 결합에, 또 다른 head는 정렬 정보에 더 민감해질 수 있다. 마지막에는 이 서로 다른 시각들을 이어 붙여 하나의 풍부한 표현으로 돌려놓는다.

논문에서는 $h = 8$개의 병렬 head를 사용한다. 각 head에 대해 $d_k = d_v = d_{model}/h = 64$로 둔다. $d_{model} = 512$이므로, 8개의 head를 이어 붙이면 다시 512차원이 된다. Head 하나하나의 차원을 줄였기 때문에 전체 계산량은 full dimension의 single-head attention과 비슷하게 유지되면서도, 표현력은 훨씬 좋아진다. 즉 multi-head는 단순히 attention을 여러 번 반복하는 것이 아니라, 비슷한 계산 비용으로 다양한 관계를 병렬로 읽어 오는 장치다.

#### 3.2.3 Applications of Attention in our Model

Transformer는 multi-head attention을 세 가지 방식으로 사용한다.

첫째, encoder-decoder attention에서는 query가 이전 decoder layer에서 오고, key와 value는 encoder 출력에서 온다. 따라서 decoder의 각 위치는 출력 문맥을 바탕으로 입력 전체를 다시 참조할 수 있다. 번역 관점에서 보면, 지금 생성하려는 target token이 source sentence의 어느 부분을 주로 바라봐야 하는지를 결정하는 단계다.

둘째, encoder 안의 self-attention에서는 query, key, value가 모두 같은 곳, 즉 이전 encoder layer의 출력에서 온다. 그래서 encoder의 각 위치는 현재 문장 안의 모든 위치를 직접 바라보며 자기 표현을 업데이트할 수 있다. 이 과정에서 각 토큰은 더 이상 고립된 단어 벡터가 아니라, 문장 전체 문맥을 반영한 contextualized representation이 된다.

셋째, decoder 안의 self-attention도 query, key, value를 같은 decoder 표현에서 가져오지만, auto-regressive property를 지키기 위해 미래 위치를 보지 못하도록 막는다. 논문은 scaled dot-product attention 내부에서 불법 연결에 해당하는 softmax 입력을 $-\infty$로 마스킹하는 방식으로 이를 구현한다. 이렇게 하면 softmax 이후 그 위치의 가중치는 사실상 0이 되므로, 미래 토큰은 현재 예측에 아무 영향도 주지 못한다.

### 3.3 Position-wise Feed-Forward Networks

Attention sub-layer 뒤에는 각 위치에 독립적으로 적용되는 fully connected feed-forward network가 붙는다. 논문은 이를 position-wise feed-forward network라고 부른다. 여기서 position-wise라는 말은 같은 가중치를 모든 위치에 똑같이 적용한다는 뜻이다. 즉 토큰들 사이의 상호작용은 attention에서 담당하고, FFN은 각 위치의 표현을 개별적으로 다시 가공해 비선형성을 부여한다.

$$
\mathrm{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
\tag{2}
$$

이 식은 두 개의 linear transformation 사이에 ReLU activation이 들어간 가장 단순한 형태의 MLP다. 입력 차원과 출력 차원은 모두 $d_{model} = 512$이고, 중간의 hidden 차원은 $d_{ff} = 2048$이다. 따라서 먼저 512차원을 2048차원으로 확장해 충분한 비선형 조합을 만들고, 다시 512차원으로 줄여 residual stream에 되돌려 보낸다.

논문은 이를 kernel size가 1인 convolution 두 번으로 볼 수도 있다고 말한다. 이 설명은 FFN이 토큰 간 정보를 섞지 않고, 각 위치에서만 동일한 변환을 수행한다는 점을 잘 보여 준다. Self-attention이 위치 간 정보를 섞어 주는 역할이라면, FFN은 그렇게 섞인 표현을 위치별로 더 정교하게 재구성하는 역할을 맡는다.

### 3.4 Embeddings and Softmax

Transformer는 입력 토큰과 출력 토큰을 $d_{model}$ 차원의 벡터로 바꾸기 위해 learned embedding을 사용한다. Decoder의 최종 출력은 learned linear transformation과 softmax를 거쳐 다음 토큰의 확률 분포로 바뀐다. 이는 당시의 다른 sequence transduction 모델과 같은 기본 틀이다.

다만 이 논문은 두 embedding layer와 pre-softmax linear transformation 사이에서 같은 weight matrix를 공유한다. 흔히 weight tying이라고 부르는 설정이다. 이렇게 하면 입력 측과 출력 측의 어휘 공간을 같은 파라미터로 다루게 되고, 전체 파라미터 수도 줄어든다. 논문은 embedding layer에서 이 weight에 $\sqrt{d_{model}}$을 곱한다고만 짧게 적어 두는데, 이 처리는 token embedding의 스케일을 조정하는 역할을 한다.

<figure>
  <img src="/assets/images/attention_is_all_you_need/attention_is_all_you_need_table_01.png" alt="Table 1. Maximum path lengths, per-layer complexity, and minimum sequential operations for different layer types." />
  <figcaption><strong>Table 1.</strong> Maximum path lengths, per-layer complexity, and minimum sequential operations for different layer types.</figcaption>
</figure>

이 비교표는 다음 절에서 이어질 self-attention의 계산적 정당화를 미리 압축해 놓은 형태다. 각 layer type을 레이어당 계산 복잡도, 순차 연산 수, 그리고 네트워크 안에서 임의의 두 위치를 연결하는 최대 경로 길이로 비교하고 있다. 왜 저자들이 recurrence를 버리는 것이 단순한 취향이 아니라 학습 효율과 장거리 의존성 학습의 측면에서 합리적이라고 보는지가 바로 이 표를 통해 전개된다.

### 3.5 Positional Encoding

Transformer에는 recurrence도 convolution도 없기 때문에, 토큰의 순서 정보가 구조 안에 자동으로 들어오지 않는다. 따라서 모델이 시퀀스의 순서를 활용하게 하려면 각 토큰이 문장 안에서 어느 위치에 있는지에 대한 정보를 별도로 주입해야 한다. 이를 위해 논문은 encoder와 decoder stack의 맨 아래에서 input embedding에 positional encoding을 더한다.

Positional encoding은 embedding과 같은 차원 $d_{model}$을 가져야 한다. 그래야 두 벡터를 단순히 더하는 것만으로 토큰 의미와 위치 정보를 한 표현 안에 겹쳐 놓을 수 있다. 논문은 위치 표현에 여러 선택지가 있을 수 있다고 인정하면서도, 이 작업에서는 learned positional embedding 대신 sine과 cosine으로 이루어진 고정형 positional encoding을 사용한다.

$$
PE_{(pos, 2i)} = \sin\left(pos / 10000^{2i/d_{model}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(pos / 10000^{2i/d_{model}}\right)
$$

여기서 $pos$는 위치이고 $i$는 차원 인덱스다. 짝수 차원에는 sine, 홀수 차원에는 cosine이 들어간다. 각 차원은 서로 다른 주파수의 sinusoid에 해당하고, 그 파장은 $2\pi$에서 $10000 \cdot 2\pi$까지 기하급수적으로 늘어난다. 따라서 어떤 위치든 하나의 단일 숫자가 아니라, 여러 주파수의 파동이 겹쳐진 고유한 패턴으로 표현된다.

저자들이 이 방식을 고른 이유는 상대적 위치를 쉽게 다룰 수 있을 것이라고 보았기 때문이다. 고정된 offset $k$에 대해 $PE_{pos+k}$를 $PE_{pos}$의 선형 함수로 나타낼 수 있으므로, 모델이 특정 거리만큼 떨어진 위치를 찾는 규칙을 배우기 유리할 것이라고 기대한다. 실제로 learned positional embedding도 실험했지만 Table 3의 row (E)에서 보듯 결과는 거의 비슷했다. 그럼에도 고정형 sinusoid를 택한 것은 학습에서 보지 못한 더 긴 시퀀스 길이로 외삽될 가능성을 기대했기 때문이다.

## 4. Why Self-Attention

이 절에서 저자들은 self-attention layer를, 길이가 가변적인 입력 시퀀스를 같은 길이의 다른 표현 시퀀스로 바꾸는 데 자주 쓰이던 recurrent layer와 convolutional layer와 비교한다. 비교의 초점은 세 가지다. 첫째는 레이어당 총 계산 복잡도이고, 둘째는 그중 얼마나 많은 계산을 병렬화할 수 있는지, 즉 최소 순차 연산 수이며, 셋째는 장거리 의존성을 연결할 때 네트워크 안에서 지나야 하는 최대 경로 길이다.

이 세 기준이 중요한 이유는 장거리 의존성 학습이 sequence transduction의 핵심 난제이기 때문이다. 입력과 출력의 임의의 두 위치 사이를 오가는 forward signal과 backward signal이 거쳐야 할 경로가 짧을수록, 멀리 떨어진 관계를 학습하기 쉬워진다. Self-attention이 단순히 빠르기만 한 것이 아니라 long-range dependency 학습에도 유리하다고 주장하려면, 바로 이 경로 길이 관점이 필요하다.

표 1을 기준으로 보면 self-attention layer는 레이어당 복잡도가 $O(n^2 \cdot d)$이고, 최소 순차 연산 수는 $O(1)$, 최대 경로 길이도 $O(1)$이다. 즉 길이 $n$의 시퀀스에서 모든 위치 쌍의 상호작용을 한 층 안에서 직접 계산하기 때문에, 한 위치에서 다른 위치로 정보가 전달되기 위해 여러 층을 거칠 필요가 없다. 반면 recurrent layer는 복잡도가 $O(n \cdot d^2)$이고 최소 순차 연산 수와 최대 경로 길이가 모두 $O(n)$이다. 문장이 길어질수록 시간축을 따라 순서대로 계산해야 하고, 멀리 떨어진 위치 사이의 정보 전달도 그만큼 길어진다.

Convolutional layer는 최소 순차 연산 수가 $O(1)$이라 병렬화 측면에서는 좋지만, kernel width $k$가 시퀀스 전체 길이보다 작으면 한 층만으로 모든 위치 쌍을 직접 연결하지 못한다. Contiguous convolution이라면 $O(n/k)$개의 층을 쌓아야 하고, dilated convolution이라도 $O(\log_k n)$층이 필요하다. 결국 최대 경로 길이가 길어질 수밖에 없다. 게다가 일반적인 convolutional layer의 레이어당 복잡도는 $O(k \cdot n \cdot d^2)$로 커져 recurrent layer보다도 비쌀 수 있다.

논문은 특히 문장 수준 번역에서 시퀀스 길이 $n$이 표현 차원 $d$보다 작은 경우가 많다고 지적한다. Word-piece나 byte-pair representation을 쓰는 경우가 여기에 해당한다. 이런 영역에서는 self-attention의 $O(n^2 \cdot d)$가 recurrent의 $O(n \cdot d^2)$보다 실제로 유리해질 수 있다. 다시 말해, self-attention의 제곱항이 항상 치명적인 것은 아니며, 당시 번역 환경에서는 오히려 더 나은 계산-병렬화 균형을 줄 수 있었다.

아주 긴 시퀀스에서는 모든 위치를 전부 보는 self-attention이 부담이 될 수 있다. 그래서 표 1에는 restricted self-attention도 함께 등장한다. Neighborhood 크기를 $r$로 제한하면 복잡도는 $O(r \cdot n \cdot d)$로 줄어든다. 대신 최대 경로 길이는 $O(n/r)$로 늘어난다. 저자들은 이 절충안을 향후 연구 과제로 남겨 둔다.

또 하나의 중요한 대목은 해석 가능성이다. 저자들은 self-attention이 비교적 해석 가능한 모델을 만들 수 있다고 본다. Attention distribution을 직접 들여다보면 어떤 위치가 어떤 위치를 강하게 참조하는지가 드러나기 때문이다. Appendix에 실린 Figure 3, Figure 4, Figure 5는 바로 그 예시다. 개별 head들이 서로 다른 일을 맡고 있고, 그중 일부는 문장의 구문 구조나 의미 구조와 관련된 행동을 보인다는 점이 이 절의 마지막 주장이다.

## 5. Training

이 절은 Transformer의 성능을 가능하게 한 학습 설정을 설명한다. 구조가 아무리 좋아도 적절한 데이터 전처리, 스케줄, 정규화가 없으면 결과가 나오지 않기 때문에, 논문은 실험 세부를 비교적 구체적으로 적어 둔다.

### 5.1 Training Data and Batching

English-to-German 실험에는 표준 WMT 2014 English-German 데이터셋을 사용하며, 약 450만 개의 문장쌍이 들어 있다. 문장들은 byte-pair encoding으로 인코딩하고, source와 target이 공유하는 vocabulary는 약 37,000개다. Subword 단위로 쪼개는 이유는 희귀 단어 문제를 줄이고, 어휘 수를 통제하면서도 다양한 단어 형태를 표현하기 위해서다.

English-to-French 실험은 훨씬 더 큰 WMT 2014 English-French 데이터셋을 사용한다. 여기에는 약 3,600만 개의 문장이 있고, 토큰은 32,000개의 word-piece vocabulary로 분할한다. 데이터 규모가 커질수록 모델이 학습할 수 있는 정렬 패턴과 번역 대응의 종류도 많아지므로, 이 과제는 모델의 확장성을 보여 주기에 적합하다.

문장쌍은 대략적인 길이가 비슷한 것끼리 같은 batch에 묶는다. 각 training batch에는 대략 25,000개의 source token과 25,000개의 target token이 들어가도록 맞춘다. 길이가 비슷한 문장을 묶으면 padding 낭비를 줄일 수 있고, GPU 메모리를 더 효율적으로 사용할 수 있다. Sequence model에서 batching이 단순한 구현상의 편의가 아니라 실제 학습 효율을 좌우하는 요소라는 점이 여기서 드러난다.

### 5.2 Hardware and Schedule

모델은 8개의 NVIDIA P100 GPU가 달린 한 대의 머신에서 학습한다. 논문 전체에서 설명한 하이퍼파라미터를 사용하는 base model은 training step당 약 0.4초가 걸리고, 총 100,000 step을 학습하므로 대략 12시간이 필요하다. 당시 번역 모델의 규모를 생각하면 매우 짧은 시간이다.

Big model은 표 3 마지막 줄에 정리되어 있는 더 큰 설정을 사용한다. 이 경우 step당 시간은 약 1.0초이고 총 300,000 step을 학습하므로 약 3.5일이 걸린다. 모델 크기를 키우면서도 학습 시간을 이 정도로 유지할 수 있었던 배경에는 self-attention 구조가 제공하는 높은 병렬화 가능성이 있다.

### 5.3 Optimizer

Optimizer로는 Adam을 사용한다. 하이퍼파라미터는 $\beta_1 = 0.9$, $\beta_2 = 0.98$, $\epsilon = 10^{-9}$다. 여기서 중요한 것은 단순히 Adam을 썼다는 사실보다 learning rate를 어떻게 스케줄링했느냐이다. 논문은 학습 전 구간에 걸쳐 learning rate를 고정하지 않고, 아래 식에 따라 변화시키는 방식을 사용한다.

$$
\mathrm{lrate} = d_{model}^{-0.5} \cdot \min\left(step\_num^{-0.5},\; step\_num \cdot warmup\_steps^{-1.5}\right)
\tag{3}
$$

이 식은 두 구간으로 나누어 이해하면 쉽다. 처음 $warmup\_steps$ 동안에는 $step\_num \cdot warmup\_steps^{-1.5}$ 항이 더 작기 때문에 learning rate가 step 수에 비례해 선형으로 증가한다. 즉 학습 초반에는 작은 learning rate로 시작해 점진적으로 예열한다. Warmup 이후에는 $step\_num^{-0.5}$ 항이 더 작아지므로, learning rate가 step의 inverse square root에 비례해 천천히 감소한다. 논문에서는 $warmup\_steps = 4000$을 사용한다.

여기서 앞의 $d_{model}^{-0.5}$는 모델 차원에 따른 스케일 보정 역할을 한다. 모델이 커질수록 적절한 learning rate의 절대 크기도 달라지기 때문에, 이 항을 통해 차원 변화에 따라 학습률을 조정한다. 이후 많은 Transformer 계열 구현에서 이 스케줄이 거의 표준처럼 인용될 정도로, 이 식은 논문 전체에서 매우 영향력 있는 설계 중 하나가 되었다.

### 5.4 Regularization

학습에서는 정규화 기법도 함께 사용한다. 논문은 residual 경로 주변의 dropout과 label smoothing을 명시적으로 설명한다.

<figure>
  <img src="/assets/images/attention_is_all_you_need/attention_is_all_you_need_table_02.png" alt="Table 2. Transformer achieves better BLEU scores with lower training cost." />
  <figcaption><strong>Table 2.</strong> Transformer achieves better BLEU scores with lower training cost.</figcaption>
</figure>

Residual dropout에서는 각 sub-layer의 출력에 dropout을 적용한 뒤, 그 결과를 sub-layer 입력에 더하고 normalization을 수행한다. 즉 residual path 바깥에서 무작위로 일부 활성값을 끄는 방식이다. 여기에 더해 encoder와 decoder 모두에서 embedding과 positional encoding의 합에도 dropout을 적용한다. Base model에서는 $P_{drop} = 0.1$을 사용한다.

이 설정은 attention이나 FFN이 지나치게 특정 활성 패턴에 의존하는 것을 막는 역할을 한다. 특히 Transformer처럼 폭이 넓고 병렬 head가 많은 구조에서는 각 head나 sub-layer가 특정 패턴에 과하게 맞춰질 수 있기 때문에, dropout은 일반화 성능을 확보하는 데 중요한 축이 된다.

Label smoothing에서는 정답 분포를 완전히 one-hot으로 두지 않고 약간 퍼뜨린다. 논문에서는 $\epsilon_{ls} = 0.1$을 사용한다. 이 방법은 모델이 특정 정답 토큰에 확신을 과도하게 몰아 주는 것을 막아 준다. 그 대가로 perplexity는 나빠질 수 있다. 실제로 논문도 label smoothing이 perplexity에는 손해를 주지만 accuracy와 BLEU score는 개선한다고 적는다. 번역에서는 정답이 하나의 문자 그대로만 존재하는 경우가 드물기 때문에, 확률 질량을 조금 더 부드럽게 분배하는 것이 오히려 더 나은 일반화로 이어질 수 있다.

## 6. Results

### 6.1 Machine Translation

WMT 2014 English-to-German 번역 과제에서 big Transformer는 표 2의 `Transformer (big)` 행처럼 BLEU 28.4를 기록한다. 당시 비교 대상 가운데 EN-DE 최고 수치는 `ConvS2S Ensemble`의 26.36이고, 그보다 더 낮은 26.30이 `GNMT + RL Ensemble`, 26.03이 `MoE`, 25.16이 `ConvS2S`, 24.6이 `GNMT + RL`, 23.75가 `ByteNet`이다. 따라서 28.4는 이전 최고 ensemble보다도 2 BLEU 이상 높은 수치다. 논문이 EN-DE에서 특히 강하게 자신감을 보이는 이유가 여기에 있다.

Base model만 보아도 BLEU 27.3으로 당시까지 보고된 모든 이전 모델과 ensemble을 넘어선다. 이 대목은 중요하다. 저자들의 주장이 단지 `큰 모델을 만들었더니 이겼다`가 아니라, 비교적 기본 설정의 Transformer조차 이전 SOTA를 제친다는 점을 강조하고 있기 때문이다. 즉 attention-only 구조 그 자체가 이미 경쟁력을 가진다는 메시지다.

학습 비용에서도 격차가 크다. 표 2에 따르면 EN-DE training cost는 `Transformer (base model)`이 $3.3 \cdot 10^{18}$ FLOPs, `Transformer (big)`이 $2.3 \cdot 10^{19}$ FLOPs다. 이에 비해 `ConvS2S Ensemble`은 $7.7 \cdot 10^{19}$ FLOPs, `GNMT + RL Ensemble`은 $1.8 \cdot 10^{20}$ FLOPs가 필요하다. 품질이 더 좋아졌는데 비용은 훨씬 낮아진 셈이다. Transformer가 단순히 정확도만 높은 모델이 아니라, 계산 효율 면에서도 새로운 기준을 제시했다는 것이 표 2의 핵심이다.

WMT 2014 English-to-French에서도 논문은 강한 결과를 제시한다. 표 2에서 `Transformer (big)`은 EN-FR 41.8 BLEU를 기록한다. 당시 single model들 가운데 `Deep-Att + PosUnk`는 39.2, `GNMT + RL`은 39.92, `ConvS2S`는 40.46, `MoE`는 40.56이었다. 따라서 41.8은 single model 기준으로 확실한 우위다. 더 나아가 당시 ensemble 결과인 `ConvS2S Ensemble` 41.29보다도 높다.

본문 설명에서는 EN-FR에 대해 41.0이라는 수치를 적고, 이 모델이 이전 single model을 모두 능가하며 이전 SOTA 모델 대비 4분의 1 미만의 training cost로 달성되었다고 말한다. English-to-French용 `Transformer (big)`은 dropout rate를 기본 big 설정의 0.3이 아니라 0.1로 두고 학습했다. 데이터 규모와 과제 특성이 달라지면 같은 구조라도 regularization 세기를 다르게 잡아야 한다는 점이 이 한 줄에 담겨 있다.

Decoding 설정도 함께 적어 두고 있다. Base model은 마지막 5개 checkpoint를 평균해 사용했고, big model은 마지막 20개 checkpoint를 평균했다. Beam search에서는 beam size 4와 length penalty $\alpha = 0.6$을 사용한다. 최대 출력 길이는 입력 길이 + 50으로 두되, 가능한 경우 조기 종료한다. 논문이 단순히 모델 구조만 발표한 것이 아니라, 실제로 성능을 안정적으로 끌어내는 추론 설정까지 함께 보고하고 있다는 점을 놓치면 안 된다.

표 2의 training cost는 논문이 직접 FLOPs를 재지 않고, 학습 시간, GPU 개수, 그리고 각 GPU의 sustained single-precision floating-point capacity 추정치를 곱해 계산한 값이다. 각주에서는 K80, K40, M40, P100에 대해 각각 2.8, 3.7, 6.0, 9.5 TFLOPS를 사용했다고 적는다. 절대적인 FLOPs 수치보다 중요한 것은 상대 비교다. 같은 시대의 강한 번역 모델들과 나란히 놓았을 때 Transformer가 비용 대비 성능에서 매우 우수했다는 사실이 여기서 드러난다.

### 6.2 Model Variations

이 절은 단순한 성능 보고를 넘어, Transformer의 어떤 요소가 실제로 중요한지 살펴보는 ablation 역할을 한다. 실험은 English-to-German development set인 newstest2013에서 수행하고, beam search는 앞 절과 같은 설정을 사용하지만 checkpoint averaging은 하지 않는다. 따라서 표 3은 최종 테스트 점수라기보다 구조 선택의 이유를 보여 주는 지도에 가깝다.

<figure>
  <img src="/assets/images/attention_is_all_you_need/attention_is_all_you_need_table_03.png" alt="Table 3. Variations on the Transformer architecture." />
  <figcaption><strong>Table 3.</strong> Variations on the Transformer architecture.</figcaption>
</figure>

Base row는 표 3의 기준점이다. $N = 6$, $d_{model} = 512$, $d_{ff} = 2048$, $h = 8$, $d_k = d_v = 64$, $P_{drop} = 0.1$, $\epsilon_{ls} = 0.1$, 100K step으로 학습하며, dev perplexity는 4.92, dev BLEU는 25.8, 파라미터 수는 65M이다. 이후 모든 variation은 특별히 적지 않은 항목은 이 base와 동일한 것으로 본다.

Row group (A)는 attention head 수와 head 내부 차원을 바꾼 실험이다. Head를 1개만 쓰고 $d_k = d_v = 512$로 키우면 dev PPL 5.29, BLEU 24.9로 떨어진다. Head 4개에 $d_k = d_v = 128$이면 BLEU 25.5, head 16개에 32차원이면 25.8, head 32개에 16차원이면 25.4다. 여기서 보이는 메시지는 분명하다. Head가 너무 적으면 하나의 attention map에 너무 많은 관계를 욱여넣어야 해서 손해를 보고, 반대로 head가 너무 많아져 각 head의 차원이 지나치게 작아져도 표현력이 떨어진다. 논문이 선택한 8-head 설정은 이 둘 사이의 균형점에 있다.

Row group (B)는 attention key size $d_k$를 줄였을 때 어떤 일이 생기는지 보여 준다. $d_k = 16$이면 dev PPL 5.16, BLEU 25.1, 파라미터 58M이고, $d_k = 32$이면 dev PPL 5.01, BLEU 25.4, 파라미터 60M이다. Base의 64보다 작은 값으로 줄일수록 품질이 내려간다. 저자들은 이를 두고 compatibility를 판정하는 일이 생각보다 쉽지 않으며, 단순한 dot product보다 더 정교한 compatibility function이 도움이 될 여지가 있다고 해석한다. Dot product attention이 빠르다고 해서 그 표현 능력이 무한한 것은 아니라는 점을 인정하는 대목이다.

Row group (C)는 모델 크기와 깊이를 바꾸는 실험이다. Layer 수를 2로 줄이면 BLEU 23.7, 4로 줄이면 25.3, 8로 늘리면 25.5가 된다. 즉 너무 얕아지면 성능이 확실히 떨어지고, 깊이를 늘리면 어느 정도 회복된다. 또 $d_{model}$을 256으로 줄이고 $d_k = d_v = 32$로 맞추면 BLEU 24.5, 파라미터 28M이고, $d_{model}$을 1024로 키우고 $d_k = d_v = 128$로 늘리면 BLEU 26.0, 파라미터 168M이다. $d_{ff}$를 1024로 줄이면 BLEU 25.4, 4096으로 늘리면 26.2가 된다. 전체적으로 보면 모델을 크게 만들수록 성능이 올라간다. Big row가 결국 26.4 BLEU까지 가는 것도 같은 흐름 위에 있다.

Row group (D)는 regularization의 효과를 보여 준다. 먼저 dropout을 0.0으로 두면 dev PPL 5.77, BLEU 24.6이고, 0.2로 올리면 dev PPL 4.95, BLEU 25.5다. 즉 dropout은 overfitting을 막는 데 매우 중요하다. 이어서 label smoothing을 0.0으로 두면 dev PPL 4.67, BLEU 25.3이고, 0.2로 올리면 dev PPL 5.47, BLEU 25.7이다. 여기서는 perplexity와 BLEU가 반드시 같은 방향으로 움직이지 않는다는 점이 분명히 드러난다. Label smoothing은 모델을 덜 확신하게 만들어 perplexity는 악화시킬 수 있지만, 실제 번역 품질 지표인 BLEU는 오히려 좋아질 수 있다.

Row group (E)는 sinusoidal positional encoding 대신 learned positional embedding을 넣은 실험이다. 결과는 dev PPL 4.92, BLEU 25.7로 base의 4.92, 25.8과 거의 같다. 따라서 positional encoding을 고정형으로 둘지 학습형으로 둘지가 이 논문에서 결정적인 차이를 만드는 요소는 아니었다. 저자들이 sinusoidal 방식을 택한 것은 성능 때문이라기보다, 더 긴 시퀀스에 대한 외삽 가능성을 기대했기 때문이다.

마지막 big row는 논문의 고성능 설정을 요약한다. $N = 6$, $d_{model} = 1024$, $d_{ff} = 4096$, head 수 16, dropout 0.3, 300K step 학습, dev PPL 4.33, dev BLEU 26.4, 파라미터 수 213M이다. Base 대비 파라미터가 65M에서 213M으로 커지면서 품질도 분명히 상승한다. 즉 Transformer는 작은 모델에서도 강하지만, 규모를 키웠을 때도 성능이 계속 좋아지는 구조라는 점을 이 행이 보여 준다.

### 6.3 English Constituency Parsing

Transformer가 번역 이외의 과제에도 일반화되는지 보기 위해, 논문은 English constituency parsing 실험을 수행한다. 이 과제는 출력이 트리 구조를 반영해야 하고 입력보다 더 길어질 수 있다는 점에서 번역과 성격이 다르다. 또한 작은 데이터 조건에서는 RNN sequence-to-sequence 모델이 SOTA에 오르지 못했다는 점도 함께 지적한다. 따라서 여기서 좋은 결과가 나오면 Transformer의 유효 범위가 훨씬 넓어진다.

논문은 $d_{model} = 1024$인 4-layer Transformer를 Penn Treebank의 Wall Street Journal portion, 약 40K training sentence에 대해 학습한다. 추가로, 약 17M 문장을 사용하는 semi-supervised setting도 실험한다. WSJ only 설정에는 16K vocabulary를, semi-supervised 설정에는 32K vocabulary를 사용한다. 개발셋인 Section 22에서는 dropout, attention과 residual 관련 설정, learning rate, beam size 정도만 소수의 실험으로 골랐고, 나머지 파라미터는 English-to-German base translation model에서 가져온다. 구조적 변형을 크게 손보지 않았다는 점이 중요하다.

<figure>
  <img src="/assets/images/attention_is_all_you_need/attention_is_all_you_need_table_04.png" alt="Table 4. English constituency parsing results." />
  <figcaption><strong>Table 4.</strong> English constituency parsing results.</figcaption>
</figure>

표 4를 보면 WSJ only, discriminative setting에서 `Transformer (4 layers)`는 F1 91.3을 기록한다. 이는 `Vinyals & Kaiser et al. (2014)`의 88.3, `Petrov et al. (2006)`의 90.4, `Zhu et al. (2013)`의 90.4보다 높지만, `Dyer et al. (2016)`의 91.7에는 조금 못 미친다. 즉 적은 데이터만으로도 매우 강력한 discriminative parser 수준까지 올라간다는 뜻이다.

Semi-supervised setting에서는 `Transformer (4 layers)`가 F1 92.7을 기록한다. 이는 `Zhu et al. (2013)` 91.3, `Huang & Harper (2009)` 91.3, `McClosky et al. (2006)` 92.1, `Vinyals & Kaiser et al. (2014)` 92.1보다 모두 높다. 다만 표 전체를 통틀어 보면 `Luong et al. (2015)`의 multi-task 93.0, `Dyer et al. (2016)`의 generative 93.3이 더 높다. 그래서 논문은 자신들의 모델이 RNNG를 제외한 기존 결과를 모두 앞섰다고 정리한다.

추론에서는 최대 출력 길이를 입력 길이 + 300으로 늘리고, WSJ only와 semi-supervised 설정 모두에 beam size 21과 $\alpha = 0.3$을 사용한다. Tree 형태의 긴 출력을 생성해야 하므로 번역보다 더 넉넉한 decoding budget을 둔 셈이다. 저자들은 이런 task-specific tuning이 거의 없었음에도 결과가 surprisingly well하다고 평가한다. 특히 WSJ 40K 문장만으로도 BerkeleyParser를 능가했다는 점을 RNN seq2seq와의 차별점으로 강조한다.

## 7. Conclusion

이 논문은 Transformer를 attention만으로 이루어진 최초의 sequence transduction model로 제시한다. Encoder-decoder 구조에서 그동안 핵심으로 여겨졌던 recurrent layer를 multi-headed self-attention으로 대체하고도, 번역 같은 대표 과제에서 오히려 더 좋은 결과를 얻을 수 있음을 보였다.

번역 과제에서는 recurrent 또는 convolutional architecture보다 훨씬 빠르게 학습할 수 있었고, WMT 2014 English-to-German과 English-to-French 모두에서 새로운 state of the art를 기록했다고 결론짓는다. 특히 English-to-German에서는 이전에 보고된 모든 ensemble보다 좋은 결과를 얻었다는 점을 별도로 강조한다.

저자들은 attention-based model의 미래에 큰 기대를 드러낸다. 앞으로는 text 이외의 modality, 예를 들어 image, audio, video로 Transformer를 확장하고, 큰 입력과 출력을 효율적으로 다루기 위해 local attention과 restricted attention 같은 변형을 탐구하겠다고 말한다. 또한 생성 과정을 덜 순차적으로 만드는 것도 중요한 향후 과제로 남겨 둔다.

논문은 마지막으로, 자신들이 사용한 학습 및 평가 코드가 Tensor2Tensor에 공개되어 있다고 밝힌다. Transformer가 단순한 아이디어 제안에 그치지 않고, 이후 연구가 바로 이어질 수 있는 구현 기반까지 함께 제공되었다는 의미다.

## Acknowledgements

Nal Kalchbrenner와 Stephan Gouws의 유익한 코멘트, 수정, 영감에 감사를 표한다.

## References

1. Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton. *Layer normalization*. arXiv preprint arXiv:1607.06450, 2016.
2. Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio. *Neural machine translation by jointly learning to align and translate*. CoRR, abs/1409.0473, 2014.
3. Denny Britz, Anna Goldie, Minh-Thang Luong, Quoc V. Le. *Massive exploration of neural machine translation architectures*. CoRR, abs/1703.03906, 2017.
4. Jianpeng Cheng, Li Dong, Mirella Lapata. *Long short-term memory-networks for machine reading*. arXiv preprint arXiv:1601.06733, 2016.
5. Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Fethi Bougares, Holger Schwenk, Yoshua Bengio. *Learning phrase representations using rnn encoder-decoder for statistical machine translation*. CoRR, abs/1406.1078, 2014.
6. Francois Chollet. *Xception: Deep learning with depthwise separable convolutions*. arXiv preprint arXiv:1610.02357, 2016.
7. Junyoung Chung, Çaglar Gülçehre, Kyunghyun Cho, Yoshua Bengio. *Empirical evaluation of gated recurrent neural networks on sequence modeling*. CoRR, abs/1412.3555, 2014.
8. Chris Dyer, Adhiguna Kuncoro, Miguel Ballesteros, Noah A. Smith. *Recurrent neural network grammars*. Proceedings of NAACL, 2016.
9. Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, Yann N. Dauphin. *Convolutional sequence to sequence learning*. arXiv preprint arXiv:1705.03122v2, 2017.
10. Alex Graves. *Generating sequences with recurrent neural networks*. arXiv preprint arXiv:1308.0850, 2013.
11. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. *Deep residual learning for image recognition*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016.
12. Sepp Hochreiter, Yoshua Bengio, Paolo Frasconi, Jürgen Schmidhuber. *Gradient flow in recurrent nets: the difficulty of learning long-term dependencies*. 2001.
13. Sepp Hochreiter, Jürgen Schmidhuber. *Long short-term memory*. Neural Computation, 9(8):1735–1780, 1997.
14. Zhongqiang Huang, Mary Harper. *Self-training PCFG grammars with latent annotations across languages*. Proceedings of the 2009 Conference on Empirical Methods in Natural Language Processing, 2009.
15. Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, Yonghui Wu. *Exploring the limits of language modeling*. arXiv preprint arXiv:1602.02410, 2016.
16. Łukasz Kaiser, Samy Bengio. *Can active memory replace attention?* Advances in Neural Information Processing Systems, 2016.
17. Łukasz Kaiser, Ilya Sutskever. *Neural GPUs learn algorithms*. International Conference on Learning Representations, 2016.
18. Nal Kalchbrenner, Lasse Espeholt, Karen Simonyan, Aaron van den Oord, Alex Graves, Koray Kavukcuoglu. *Neural machine translation in linear time*. arXiv preprint arXiv:1610.10099v2, 2017.
19. Yoon Kim, Carl Denton, Luong Hoang, Alexander M. Rush. *Structured attention networks*. International Conference on Learning Representations, 2017.
20. Diederik Kingma, Jimmy Ba. *Adam: A method for stochastic optimization*. ICLR, 2015.
21. Oleksii Kuchaiev, Boris Ginsburg. *Factorization tricks for LSTM networks*. arXiv preprint arXiv:1703.10722, 2017.
22. Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen Zhou, Yoshua Bengio. *A structured self-attentive sentence embedding*. arXiv preprint arXiv:1703.03130, 2017.
23. Minh-Thang Luong, Quoc V. Le, Ilya Sutskever, Oriol Vinyals, Lukasz Kaiser. *Multi-task sequence to sequence learning*. arXiv preprint arXiv:1511.06114, 2015.
24. Minh-Thang Luong, Hieu Pham, Christopher D. Manning. *Effective approaches to attention-based neural machine translation*. arXiv preprint arXiv:1508.04025, 2015.
25. Mitchell P. Marcus, Mary Ann Marcinkiewicz, Beatrice Santorini. *Building a large annotated corpus of english: The penn treebank*. Computational Linguistics, 19(2):313–330, 1993.
26. David McClosky, Eugene Charniak, Mark Johnson. *Effective self-training for parsing*. Proceedings of the Human Language Technology Conference of the NAACL, 2006.
27. Ankur Parikh, Oscar Täckström, Dipanjan Das, Jakob Uszkoreit. *A decomposable attention model*. Empirical Methods in Natural Language Processing, 2016.
28. Romain Paulus, Caiming Xiong, Richard Socher. *A deep reinforced model for abstractive summarization*. arXiv preprint arXiv:1705.04304, 2017.
29. Slav Petrov, Leon Barrett, Romain Thibaux, Dan Klein. *Learning accurate, compact, and interpretable tree annotation*. Proceedings of the 21st International Conference on Computational Linguistics and 44th Annual Meeting of the ACL, 2006.
30. Ofir Press, Lior Wolf. *Using the output embedding to improve language models*. arXiv preprint arXiv:1608.05859, 2016.
31. Rico Sennrich, Barry Haddow, Alexandra Birch. *Neural machine translation of rare words with subword units*. arXiv preprint arXiv:1508.07909, 2015.
32. Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean. *Outrageously large neural networks: The sparsely-gated mixture-of-experts layer*. arXiv preprint arXiv:1701.06538, 2017.
33. Nitish Srivastava, Geoffrey E. Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov. *Dropout: a simple way to prevent neural networks from overfitting*. Journal of Machine Learning Research, 15(1):1929–1958, 2014.
34. Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus. *End-to-end memory networks*. Advances in Neural Information Processing Systems 28, 2015.
35. Ilya Sutskever, Oriol Vinyals, Quoc V. Le. *Sequence to sequence learning with neural networks*. Advances in Neural Information Processing Systems, 2014.
36. Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna. *Rethinking the inception architecture for computer vision*. CoRR, abs/1512.00567, 2015.
37. Vinyals, Kaiser, Koo, Petrov, Sutskever, Hinton. *Grammar as a foreign language*. Advances in Neural Information Processing Systems, 2015.
38. Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. *Google’s neural machine translation system: Bridging the gap between human and machine translation*. arXiv preprint arXiv:1609.08144, 2016.
39. Jie Zhou, Ying Cao, Xuguang Wang, Peng Li, Wei Xu. *Deep recurrent models with fast-forward connections for neural machine translation*. CoRR, abs/1606.04199, 2016.
40. Muhua Zhu, Yue Zhang, Wenliang Chen, Min Zhang, Jingbo Zhu. *Fast and accurate shift-reduce constituent parsing*. Proceedings of the 51st Annual Meeting of the ACL (Volume 1: Long Papers), 2013.

## Attention Visualizations

<figure>
  <img src="/assets/images/attention_is_all_you_need/attention_is_all_you_need_figure_03.png" alt="Figure 3. An example of attention distributions in an encoder self-attention layer." />
  <figcaption><strong>Figure 3.</strong> Example attention distributions in an encoder self-attention layer.</figcaption>
</figure>

부록의 첫 번째 시각화는 encoder self-attention의 여섯 개 layer 중 다섯 번째 layer에서, 단어 `making`이 문장 안의 먼 위치들과 어떻게 연결되는지를 보여 준다. 핵심은 attention이 인접한 몇 개 토큰에만 머무르지 않는다는 점이다. 여러 head가 `making`에서 멀리 떨어진 `more difficult` 쪽으로 직접 점프하며 장거리 의존성을 포착한다. 논문이 self-attention의 path length가 짧다고 말할 때 그것이 실제로 어떤 모습인지 이 예시가 시각적으로 보여 준다. 서로 다른 색은 서로 다른 head를 뜻하며, head마다 다른 연결 패턴을 갖는다는 점도 함께 드러난다.

이 그림을 통해 볼 수 있는 것은 단순히 `멀리 본다`는 사실만이 아니다. 여러 head가 같은 단어에서 출발하더라도 서로 다른 위치를 참조하면서, 하나의 관계만 보는 것이 아니라 문장 전체 의미를 구성하는 여러 단서를 병렬로 읽고 있다는 점이 보인다. Self-attention이 long-range dependency를 처리할 수 있다는 논문의 주장과 Multi-Head Attention이 평균화를 완화한다는 주장이 이 그림에서 자연스럽게 만난다.

<figure>
  <img src="/assets/images/attention_is_all_you_need/attention_is_all_you_need_figure_04.png" alt="Figure 4. Two attention heads that appear to specialize in anaphora resolution." />
  <figcaption><strong>Figure 4.</strong> Two attention heads that appear to specialize in anaphora resolution.</figcaption>
</figure>

두 번째 시각화는 같은 다섯 번째 layer의 두 attention head가 anaphora resolution, 즉 대명사와 그 선행사를 연결하는 문제와 관련된 것처럼 보인다는 예를 제시한다. 위쪽에는 head 5의 전체 attention 분포가, 아래쪽에는 단어 `its`에서 출발한 attention만 따로 떼어 head 5와 head 6에 대해 보여 준다. 논문이 말하듯 이 단어에 대한 attention은 매우 sharp하다. 특정 단어가 문장 전체를 흐릿하게 평균하는 것이 아니라, 몇몇 후보 위치에 매우 강하게 집중하는 것이다.

이 예시는 attention head가 단순히 위치 가까운 토큰을 무차별적으로 섞는 장치가 아니라, 문장 안의 의미적 연결에 반응하는 선택적 메커니즘일 수 있음을 시사한다. 모든 head가 이런 역할을 한다는 뜻은 아니지만, 일부 head는 대명사 해석처럼 언어적으로 분명한 구조를 읽어 내는 방향으로 특화될 수 있다는 점을 잘 보여 준다.

<figure>
  <img src="/assets/images/attention_is_all_you_need/attention_is_all_you_need_figure_05.png" alt="Figure 5. Two attention heads that appear to learn syntactic structure." />
  <figcaption><strong>Figure 5.</strong> Two attention heads that appear to learn syntactic structure.</figcaption>
</figure>

세 번째 시각화는 문장 구조와 관련된 행동을 보이는 head의 예를 두 개 보여 준다. 두 그림 모두 encoder self-attention의 다섯 번째 layer에서 나온 것이지만, 연결 패턴은 분명히 다르다. 어떤 head는 특정 형태의 토큰 결합을 반복적으로 잡고, 다른 head는 또 다른 방식의 구조적 결합에 반응한다. 논문이 `the heads clearly learned to perform different tasks`라고 결론짓는 이유가 여기에 있다.

즉 attention head는 모두 같은 일을 중복해서 하는 것이 아니라, 서로 다른 종류의 관계를 나누어 맡는 경향을 보인다. 이것은 Multi-Head Attention의 설계가 단순한 계산량 분산이 아니라 표현 공간 분할이라는 점을 다시 확인시켜 준다. 문장의 구조를 한 개의 거대한 attention map으로 처리하는 대신, 여러 head가 각자의 규칙을 학습해 전체 구조를 분담하는 것이다.

## 추가 설명

### 1. Query, Key, Value를 문장 안의 역할로 다시 보면

Attention 식을 처음 볼 때 가장 헷갈리는 부분은 왜 같은 입력에서 굳이 query, key, value를 따로 만들 필요가 있느냐는 점이다. Self-attention에서는 같은 토큰 표현에서 세 벡터를 모두 뽑아 오기 때문에, 얼핏 보면 같은 정보를 세 번 복사하는 것처럼 보일 수 있다. 하지만 역할은 분명히 다르다. Query는 현재 위치가 지금 무엇을 찾고 있는지를 나타내고, key는 각 위치가 어떤 종류의 질의에 잘 반응할지를 나타내며, value는 실제로 가져갈 내용을 담는다.

예를 들어 번역 encoder에서 어떤 단어가 주어-서술어 관계를 파악하고 싶다고 하자. 그 단어의 query는 `내가 지금 문장 안에서 무엇을 참조해야 하는가`를 묻는 표현이 되고, 다른 단어들의 key는 `나는 이런 종류의 질의와 잘 맞는다`는 주소 역할을 한다. Query와 key의 내적은 주소를 찾는 단계이고, softmax로 얻은 가중치를 value에 곱하는 것은 실제 내용을 읽어 오는 단계다. 그래서 `QK^T`는 어디를 볼지를 정하고, `V`는 그곳에서 무엇을 가져올지를 정한다고 이해하면 된다.

### 2. Self-Attention만으로도 Seq2Seq가 성립하는 이유

Seq2seq 모델이 되려면 사실 세 가지 기능만 있으면 된다. 첫째, 입력 문장을 문맥화된 표현으로 바꾸는 encoder가 필요하다. 둘째, 출력 prefix를 바탕으로 다음 토큰을 예측하는 decoder가 필요하다. 셋째, decoder가 입력 문장을 참조할 수 있어야 한다. Transformer는 이 세 기능을 모두 attention으로 재구성한다.

Encoder self-attention은 입력 문장 안의 모든 위치를 한 번에 연결해 contextualized source representation을 만든다. Decoder self-attention은 지금까지 생성한 출력 prefix를 문맥화한다. 그리고 encoder-decoder attention은 출력 위치 하나하나가 입력 전체를 다시 참조하게 만든다. 결국 seq2seq에 꼭 필요한 것은 recurrence 그 자체가 아니라, 입력 내부 관계, 출력 내부 관계, 입력-출력 관계를 적절히 계산할 수 있는 장치였다는 것이 드러난다. Transformer는 그 세 장치를 모두 attention으로 통일한 구조다.

### 3. Decoder Mask가 없으면 왜 문제가 되는가

Transformer decoder는 학습할 때 정답 문장을 통째로 알고 있다. 만약 mask가 없다면, 위치 $i$의 출력 표현은 자기보다 뒤에 있는 정답 토큰들까지 자유롭게 볼 수 있게 된다. 그러면 모델은 사실상 미래 정답을 훔쳐본 상태로 현재 토큰을 예측하게 되고, 학습 시 손실은 인위적으로 낮아진다. 하지만 추론 시에는 미래 토큰을 모른 채 한 글자씩 생성해야 하므로, 학습과 추론의 조건이 완전히 어긋난다.

Mask는 이 문제를 원천적으로 막는다. Softmax 이전에 미래 위치의 점수를 $-\infty$로 바꾸면 그쪽 attention weight는 0이 된다. 이렇게 하면 학습 중에도 decoder는 오직 왼쪽 문맥만 보고 다음 토큰을 예측해야 한다. Transformer가 병렬 학습을 하면서도 auto-regressive generation을 유지할 수 있는 핵심이 바로 이 triangular mask다.

### 4. Multi-Head Attention은 왜 단일 Head보다 나은가

단일 head에서도 모든 토큰 쌍의 관계를 계산할 수는 있다. 문제는 그 모든 관계를 하나의 attention 분포와 하나의 value mixing 규칙으로 압축해야 한다는 데 있다. 문장 안에는 주어-동사 관계, 수식 관계, coreference, 위치 정렬, 형태적 단서 등 여러 종류의 관계가 동시에 존재한다. 단일 head는 이 다양한 관계를 한 장의 지도로 표현해야 하므로 쉽게 평균화된다.

Multi-head는 입력을 서로 다른 표현 공간으로 투영해 여러 장의 지도를 동시에 만든다. 각 head는 자기만의 query, key, value 투영을 배우므로, 어떤 head는 장거리 의존성에, 어떤 head는 구문 구조에, 어떤 head는 정렬 정보에 민감해질 수 있다. 부록 그림들이 보여 주듯 실제로 head마다 꽤 다른 패턴이 나타난다. 그래서 multi-head의 핵심 이점은 `더 많이 본다`가 아니라 `다르게 본다`에 가깝다.

### 5. Positional Encoding을 더하는 방식이 중요한 이유

위치 정보를 주입하는 방법은 여러 가지가 있을 수 있다. Concatenation을 할 수도 있고, 별도의 position network를 둘 수도 있다. Transformer는 가장 단순한 방법인 덧셈을 택한다. Embedding과 positional encoding이 같은 차원을 가지므로 두 벡터를 더하면, 하나의 residual stream 안에 토큰 의미와 위치 정보가 함께 들어간다.

이 방식의 장점은 구조를 복잡하게 만들지 않는다는 점이다. Self-attention과 FFN은 입력 차원이 고정되어 있어야 residual connection이 매끄럽게 동작하는데, 위치 정보를 더하기로 넣으면 기존 차원을 유지한 채로 위치 단서를 제공할 수 있다. 또 여러 layer를 거치면서 모델은 필요할 때 의미 정보와 위치 정보를 함께 활용할 수 있다. Sinusoidal encoding의 경우 서로 다른 주파수의 조합이기 때문에, 절대 위치뿐 아니라 상대적 거리 정보도 선형적으로 읽어 내기 쉬운 형태가 된다.

### 6. 표 1의 Maximum Path Length가 왜 중요한가

딥러닝 모델에서 두 위치 사이의 정보 전달은 보통 여러 layer를 통과하며 이루어진다. 이때 한 위치의 변화가 다른 위치의 표현에 영향을 주기까지 거쳐야 하는 단계 수가 길면, gradient가 약해지거나 관계 학습이 어려워질 수 있다. RNN에서 긴 문장의 장거리 의존성이 어려웠던 이유도 여기에 있다. 정보가 시간축을 따라 여러 step을 통과해야 하기 때문이다.

Self-attention에서는 한 layer 안에서 모든 위치가 서로 직접 연결된다. 그래서 최대 path length가 $O(1)$이다. 어떤 두 단어가 문장 양 끝에 있더라도 한 번의 attention으로 바로 영향을 주고받을 수 있다. 물론 실제 학습이 언제나 쉬운 것은 아니지만, 구조적으로는 장거리 의존성을 전달하는 통로가 짧다. Transformer가 긴 문맥 관계를 더 잘 다룰 수 있다고 기대할 수 있는 근거가 바로 여기에 있다.

### 7. 표 3이 말해 주는 것은 'attention이면 다 된다'가 아니라는 점이다

표 3은 Transformer의 성공이 단순한 슬로건 하나에서 나온 것이 아님을 보여 준다. Attention을 쓴다고 해서 아무 설정이나 잘 되는 것은 아니다. Head 수가 너무 적어도 나쁘고 너무 많아도 나쁘며, key 차원이 너무 작아도 품질이 떨어진다. 모델 폭과 FFN 차원을 키우면 분명히 좋아지고, dropout을 없애면 쉽게 overfit하며, label smoothing은 perplexity와 BLEU 사이의 상충 관계를 만든다.

즉 이 논문의 기여는 `attention is all you need`라는 선언에만 있지 않다. 실제로는 self-attention, multi-head, positional encoding, FFN, residual/layer norm, optimizer schedule, label smoothing, checkpoint averaging까지 포함한 전체 recipe를 함께 제시했다는 점이 중요하다. 이후 연구들이 Transformer를 재현할 때 논문 본문뿐 아니라 표 3 같은 ablation 결과를 매우 중요하게 보는 이유도 여기에 있다.

### 8. 이 논문이 이후 모델들에 남긴 구조적 유산

지금의 관점에서 보면 Transformer의 가장 큰 유산은 개별 과제 성능보다 아키텍처 표준을 바꾸었다는 데 있다. Encoder-only 모델은 BERT 계열로, decoder-only 모델은 GPT 계열로, encoder-decoder 모델은 T5 같은 계열로 뻗어 나갔다. 세부 구성은 많이 달라졌지만, 토큰 표현을 self-attention과 position-wise MLP로 반복 갱신한다는 큰 틀은 그대로 남아 있다.

동시에 이 논문이 남긴 한계도 분명하다. Self-attention의 기본 비용은 시퀀스 길이에 대해 $O(n^2)$이므로, 매우 긴 문맥에서는 계산량과 메모리 사용이 빠르게 커진다. 논문이 결론에서 local attention이나 restricted attention을 향후 과제로 적어 둔 이유도 이 때문이다. 이후 긴 문맥 Transformer 연구의 상당수는 바로 이 비용 문제를 해결하려는 방향으로 전개되었다. 다시 말해 Transformer는 완성형 구조라기보다, 이후 수많은 변형과 확장의 출발점이 된 기본 설계도라고 보는 편이 맞다.
