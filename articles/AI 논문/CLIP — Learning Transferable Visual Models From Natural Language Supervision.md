---
title: "CLIP — Learning Transferable Visual Models From Natural Language Supervision"
math: true
---

# Learning Transferable Visual Models From Natural Language Supervision

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever

OpenAI

Proceedings of the 38th International Conference on Machine Learning (ICML 2021), PMLR 139.

## Abstract

기존의 최상위권 컴퓨터 비전 시스템은 대체로 미리 정해 둔 고정된 객체 범주 집합을 예측하도록 학습된다. 이런 방식은 특정 벤치마크에서는 강력하지만, 그 범주 집합 바깥의 개념으로 모델을 확장하려면 다시 라벨 데이터를 모아 별도로 학습해야 한다는 한계가 있다. 이 논문은 이미지를 설명하는 자연어를 직접 감독 신호로 삼으면 훨씬 더 넓은 개념 공간을 다룰 수 있지 않겠느냐는 질문에서 출발한다.

저자들은 인터넷에서 수집한 4억 개의 이미지-텍스트 쌍 위에서, 주어진 이미지와 주어진 텍스트가 서로 짝이 맞는지를 맞히는 단순한 대조학습 목표만으로도 매우 강력한 시각 표현을 배울 수 있다고 주장한다. 이 접근은 나중에 별도의 데이터셋 전용 학습을 하지 않아도, 자연어로 클래스 이름이나 설명을 써 넣는 것만으로 새로운 분류기를 만들 수 있게 한다. 다시 말해 텍스트가 학습용 라벨의 역할뿐 아니라 추론 시점의 인터페이스 역할까지 함께 맡는다.

실험에서는 OCR, 행동 인식, 지리 위치 추정, 세밀한 시각 분류처럼 서로 성격이 다른 30개가 넘는 비전 데이터셋으로 전이 성능을 검증한다. CLIP은 많은 과제에서 추가 학습 없이도 강한 성능을 보이고, 어떤 경우에는 완전지도학습 기반의 강한 기준선과도 맞먹는다. 대표적으로 ImageNet에서는 128만 장의 라벨 데이터로 학습한 원래의 ResNet-50과 비슷한 정확도를 제로샷으로 달성한다. 논문은 코드와 사전학습 가중치도 함께 공개한다고 밝힌다.

## 1. Introduction and Motivating Work

NLP에서는 대규모 비정형 텍스트를 이용한 사전학습이 이미 판도를 바꾸고 있었다. 언어 모델은 원래 특정 과제에 맞춰 설계된 작은 모델들을 대신해, 거대한 텍스트 말뭉치에서 먼저 일반 능력을 학습한 뒤 프롬프트나 소량의 예시만으로 다양한 과제로 전이하는 방향으로 발전했다. GPT 계열과 text-to-text 계열의 성과는 웹 규모 텍스트 안에 이미 충분히 풍부한 감독 신호가 숨어 있음을 보여 주었다.

저자들은 같은 질문을 컴퓨터 비전으로 옮긴다. 비전에서는 여전히 ImageNet 같은 crowd-labeled 데이터셋에 의존하는 관행이 강하다. 하지만 인터넷에는 이미지와 함께 텍스트가 붙어 있는 자료가 엄청난 규모로 공개되어 있다. 만약 모델이 그 텍스트를 직접 읽으면서 시각 개념을 배운다면, 사람 손으로 정의한 좁은 라벨 공간을 넘어설 수 있을지도 모른다.

이 방향의 선행연구는 이미 존재했다. 이미지 캡션의 단어를 예측하도록 CNN을 학습한 연구, n-gram을 예측해 제로샷 전이를 시도한 연구, 그리고 최근의 VirTex, ICMLM, ConVIRT처럼 비전과 언어를 함께 학습하는 모델들이 그 예다. 다만 이들 방법은 여전히 최신 비전 모델들과 비교하면 성능이 모자랐다. 논문은 그 핵심 차이를 규모에서 찾는다. 약지도나 대규모 지도학습 모델은 수백만에서 수십억 장에 달하는 이미지로 매우 오래 학습되었지만, 기존의 이미지-텍스트 학습은 훨씬 작은 데이터와 적은 연산량에 머물러 있었다.

이 논문은 바로 그 규모의 격차를 메우려 한다. ConVIRT에서 영감을 받은 단순한 대조학습 구조를 아주 큰 데이터에 적용하고, 그것을 CLIP, 즉 Contrastive Language-Image Pre-training이라고 부른다. 핵심 주장은 두 가지다. 첫째, 자연어 감독만으로도 시각 모델이 매우 다양한 과제를 내부적으로 학습할 수 있다는 점이다. 둘째, 그렇게 얻은 능력을 자연어 프롬프트를 통해 별도 미세조정 없이 바로 꺼내 쓸 수 있다는 점이다.

[Figure 1 삽입 위치]

이 그림은 논문의 전체 아이디어를 한 장으로 요약한다. 학습 단계에서는 이미지 인코더와 텍스트 인코더를 함께 두고, 같은 배치 안에서 어떤 이미지와 어떤 텍스트가 실제 짝인지 맞히도록 만든다. 추론 단계에서는 데이터셋의 클래스 이름이나 설명 문장을 텍스트 인코더에 넣어 클래스별 텍스트 임베딩을 만든 뒤, 이미지 임베딩과 가장 잘 맞는 클래스를 선택한다. 기존의 이미지 분류기가 선형 분류기를 파라미터로 학습하는 데 비해, CLIP은 텍스트 자체로 분류기를 합성한다는 차이가 여기서 드러난다.

논문은 이 방식이 OCR, 행동 인식, 지리 위치 추정 같은 다양한 과제를 사전학습 중에 함께 흡수하며, ImageNet에서는 당시 공개된 최고 성능 모델과 견줄 만큼 강력한 제로샷 성능까지 낼 수 있다고 주장한다. 또한 동일한 ImageNet 정확도를 가진 일반적인 지도학습 모델들보다 분포 이동에 더 강한 모습도 보인다고 말한다.

## 2. Approach

논문의 중심 아이디어는 단순하다. 이미지와 자연어를 같은 의미 공간으로 끌어와, 서로 맞는 쌍은 가깝게, 맞지 않는 쌍은 멀어지게 학습하면 된다. 하지만 실제로 이 생각을 강력한 시스템으로 만들려면 충분히 큰 데이터, 적절한 사전학습 목표, 그리고 확장 가능한 모델 설계가 함께 필요하다. 이 절은 그 세 가지를 차례대로 설명한다.

### 2.1. Creating a Sufficiently Large Dataset

기존 연구가 주로 사용한 이미지-텍스트 데이터셋은 MS-COCO, Visual Genome, YFCC100M 같은 것들이었다. MS-COCO와 Visual Genome은 사람이 정성스럽게 붙인 설명이 있어 품질은 좋지만 규모가 현대적 기준에서는 작다. 반대로 YFCC100M은 1억 장 규모라는 장점이 있지만, 이미지에 붙은 제목이나 설명의 품질이 들쭉날쭉하고 자연어가 아닌 경우도 많다. 실제로 영어 자연어 제목이나 설명만 남기면 1억 장이 약 1,500만 장 수준까지 줄어든다.

저자들은 자연어 감독의 진짜 장점은 인터넷에 공개된 자료의 압도적인 양에 있다고 본다. 그래서 여러 공개 소스에서 4억 개의 이미지-텍스트 쌍을 수집해 새로운 데이터셋을 만든다. 이 데이터셋을 WIT, 즉 WebImageText라고 부른다. 여기서 중요한 것은 단순히 많이 모으는 것이 아니라 가능한 넓은 시각 개념을 덮도록 수집 전략을 짠 점이다.

이를 위해 저자들은 약 50만 개의 질의어 집합을 만들고, 그 질의어를 텍스트에 포함하는 이미지-텍스트 쌍을 찾는다. 각 질의어마다 최대 2만 개 정도만 포함해 대략적인 클래스 균형을 맞추려 한다. 기본 질의어는 영어 위키피디아에서 충분히 자주 등장하는 단어들이고, 여기에 높은 PMI를 가진 bi-gram, 검색량이 일정 수준 이상인 위키피디아 문서 제목, 그리고 WordNet synset을 추가한다. 결과적으로 WIT는 어휘량 면에서도 GPT-2를 학습한 WebText와 비슷한 수준의 텍스트 규모를 가진다.

이 데이터 구성 방식의 핵심은 라벨 공간을 사전에 강하게 닫아 두지 않는 데 있다. 사람이 분류 체계를 먼저 정하고 이미지를 그 체계에 끼워 넣는 대신, 인터넷에 실제로 함께 등장하는 시각 자료와 언어 표현을 최대한 폭넓게 수집한다. CLIP의 제로샷 성능은 이후 이 폭넓은 수집 전략이 얼마나 많은 시각 개념을 사전학습 속에 녹여 넣었는지를 보여 주는 지표가 된다.

### 2.2. Selecting an Efficient Pre-Training Method

처음부터 저자들이 곧바로 현재의 CLIP 목표를 쓴 것은 아니다. 초기 접근은 VirTex와 비슷하게 이미지 CNN과 텍스트 트랜스포머를 함께 두고, 이미지로부터 캡션 전체를 예측하도록 학습하는 방식이었다. 그러나 저자들은 이 방향이 대규모 확장에 비해 너무 비효율적이라고 본다. 고표현력의 언어 모델을 붙여 캡션을 생성하게 만들면 계산량은 커지는데, 제로샷 이미지 분류로 이어지는 효율은 충분히 좋지 않았다.

[Figure 2 삽입 위치]

이 그림은 그 비효율을 수치로 보여 준다. 동일한 이미지 처리량 기준으로 보면, 트랜스포머 언어 모델을 이용한 이미지 캡션 방식은 bag-of-words를 예측하는 더 단순한 기준선보다 ImageNet 제로샷 정확도가 약 세 배 느리게 올라간다. 그리고 같은 bag-of-words 기반에서도 예측 목표를 대조학습 목표로 바꾸면 효율이 다시 약 네 배 개선된다. 즉 CLIP의 성과는 단순히 데이터가 커서가 아니라, 데이터 크기에 잘 맞는 학습 목표를 골랐기 때문이기도 하다.

CLIP은 한 배치 안에 들어 있는 $N$개의 이미지-텍스트 쌍을 놓고, 가능한 $N \times N$개의 조합 중 어떤 것들이 진짜 짝인지 맞히도록 학습된다. 이 과정에서 모델은 이미지와 텍스트를 같은 다중모달 임베딩 공간으로 사상한다. 실제로 맞는 쌍의 코사인 유사도는 높이고, 배치 안의 나머지 잘못된 쌍은 낮추는 방식이다. 손실 함수는 이미지 방향과 텍스트 방향을 모두 고려한 대칭적인 cross-entropy다.

[Figure 3 삽입 위치]

그림의 의사코드를 수식으로 적으면 구조가 더 선명해진다. 우선 이미지 인코더와 텍스트 인코더는 각각 다음과 같이 특징 벡터를 만든다.

$$
I_f = f_{\text{image}}(I), \qquad T_f = f_{\text{text}}(T)
$$

여기서 $I_f \in \mathbb{R}^{N \times d_i}$, $T_f \in \mathbb{R}^{N \times d_t}$는 배치 안의 이미지와 텍스트 특징이다. 이어서 학습 가능한 선형 사영을 통해 두 모달리티를 같은 임베딩 차원으로 옮기고, L2 정규화를 적용한다.

$$
I_e = \frac{I_f W_i}{\lVert I_f W_i \rVert_2}, \qquad
T_e = \frac{T_f W_t}{\lVert T_f W_t \rVert_2}
$$

정규화를 했기 때문에 두 임베딩의 내적은 사실상 코사인 유사도가 된다. 이제 배치 전체에 대한 유사도 행렬은 다음과 같이 계산된다.

$$
S = I_e T_e^{\top} \cdot e^t
$$

여기서 $t$는 학습 가능한 스칼라 파라미터이며, softmax에 들어가기 전 로짓의 스케일을 조절한다. 일반적인 온도 파라미터 $\tau$로 생각하면 $e^t = 1/\tau$에 해당한다. $S$의 대각성분은 올바른 이미지-텍스트 쌍의 유사도이고, 비대각성분은 배치 안의 음성 샘플과의 유사도다.

이미지에서 텍스트를 맞히는 손실과, 텍스트에서 이미지를 맞히는 손실을 각각 쓰면 다음과 같다.

$$
L_{\text{image}} = \mathrm{CE}(S, [0,1,\dots,N-1])
$$

$$
L_{\text{text}} = \mathrm{CE}(S^{\top}, [0,1,\dots,N-1])
$$

최종 손실은 두 방향의 평균이다.

$$
L = \frac{L_{\text{image}} + L_{\text{text}}}{2}
$$

이 손실의 직관은 간단하다. 각 이미지는 배치 안의 여러 텍스트 가운데 자기 짝 텍스트를 골라야 하고, 각 텍스트도 배치 안의 여러 이미지 가운데 자기 짝 이미지를 골라야 한다. 두 모달리티가 서로를 검색하는 구조를 동시에 학습하는 셈이다. 이 때문에 CLIP은 생성 모델보다 훨씬 단순한 목표로도 강한 다중모달 정렬 능력을 얻게 된다.

저자들은 Zhang 등에서 사용한 ConVIRT보다 학습을 더 단순하게 만든다. 사전학습 가중치 초기화도 쓰지 않고, 표현 공간과 대조학습 임베딩 공간 사이의 비선형 투영도 없애며, 단순한 선형 사영만 남긴다. 텍스트 변환 함수에서 문장 하나를 샘플링하는 과정도 제거한다. 이미지 증강 역시 resized image에서 뽑는 random square crop 하나만 사용한다. 과적합이 큰 문제는 아니라고 판단했기 때문에, 복잡한 정규화 장치를 여러 개 붙이기보다 가장 핵심적인 구조만 남긴 것이다.

특히 온도 파라미터를 하이퍼파라미터로 두지 않고 직접 학습하게 한 점이 중요하다. 로짓이 너무 뾰족해지면 학습이 불안정해지고, 너무 평평해지면 양성과 음성을 구별하는 힘이 약해진다. CLIP은 그 균형점을 데이터와 모델이 스스로 찾게 한다.

### 2.3. Choosing and Scaling a Model

이미지 인코더로는 두 계열을 검토한다. 첫 번째는 널리 쓰이는 ResNet-50 계열이고, 두 번째는 비교적 최근에 등장한 Vision Transformer 계열이다. ResNet 쪽은 기본 구조를 그대로 쓰지 않고 몇 가지 개선을 더한다. He 등의 ResNetD 수정, Zhang의 anti-aliased rect-2 blur pooling, 그리고 마지막 global average pooling을 대체하는 attention pooling이 그 핵심이다. 이 attention pooling은 이미지의 전역 평균 표현을 query로 삼고, 공간 위치 특징들을 key와 value로 삼는 단일 층의 multi-head QKV attention으로 구현된다.

Vision Transformer 쪽은 Dosovitskiy 등의 구조를 거의 그대로 따르되, patch embedding과 position embedding을 합친 뒤 트랜스포머에 넣기 전에 layer normalization을 하나 더 추가하고 초기화 방식도 조금 바꾼다. 즉 CLIP은 CNN과 ViT를 모두 실험하지만, 텍스트 감독과 대조학습이라는 큰 틀은 동일하게 유지한다.

텍스트 인코더는 Radford 등의 트랜스포머 구조를 따른 12층, 너비 512, 8-head 모델이다. 텍스트는 lower-cased byte pair encoding으로 토큰화되며, 문장 앞뒤에 [SOS]와 [EOS] 토큰을 붙인다. 최종 표현으로는 가장 높은 층에서 [EOS] 위치의 활성값을 사용하고, 여기에 layer normalization을 적용한 뒤 다중모달 임베딩 공간으로 선형 사영한다. 텍스트 인코더에 masked self-attention을 유지한 이유는 나중에 언어 모델링 목적을 보조 목표로 붙일 가능성을 열어 두기 위해서라고 설명한다.

모델 스케일링 전략도 논문의 중요한 포인트다. 이전 비전 연구는 너비나 깊이 한 축만 크게 늘리는 경우가 많았지만, 저자들은 EfficientNet 계열의 아이디어를 따라 너비, 깊이, 해상도에 연산량을 고르게 배분하는 편이 더 낫다고 본다. 따라서 ResNet 계열은 추가 계산 자원을 세 축에 비교적 균등하게 분배한다. 반면 텍스트 인코더는 성능이 상대적으로 덜 민감하다고 보아, 깊이는 고정하고 너비만 늘린다.

### 2.4. Pre-training

실제 실험에서는 총 다섯 개의 ResNet과 세 개의 Vision Transformer를 학습한다. ResNet 쪽은 RN50, RN101, 그리고 RN50 대비 대략 4배, 16배, 64배의 계산량을 쓰는 RN50x4, RN50x16, RN50x64다. ViT 쪽은 ViT-B/32, ViT-B/16, ViT-L/14를 학습한다.

가장 큰 ResNet인 RN50x64는 V100 GPU 592개로 18일 동안 학습했고, 가장 큰 Vision Transformer는 V100 256개로 12일 동안 학습했다. ViT-L/14는 성능을 더 끌어올리기 위해 336픽셀 해상도에서 추가로 한 epoch 더 사전학습하고, 이 모델을 ViT-L/14@336px라고 표기한다. 논문에서 특별히 다른 언급이 없을 때 CLIP의 대표 결과라고 부르는 것은 이 ViT-L/14@336px다.

이 대목은 CLIP이 단순한 아이디어 실험이 아니라, 실제로 상당한 규모의 연산을 투입해 끝까지 스케일을 밀어 본 연구임을 보여 준다. 동시에 결과를 보면 같은 계산 예산 안에서도 ViT 계열이 ResNet보다 더 효율적으로 성장하는 경향이 나타나는데, 그 점은 뒤의 representation learning 분석에서 다시 확인된다.

### 2.5. Using CLIP

CLIP은 WIT에서 이미지와 텍스트의 짝을 맞히는 과제로 사전학습되었지만, 저자들은 이 능력을 다운스트림에서도 그대로 재사용하려 한다. 특정 데이터셋의 클래스 집합이 주어지면, 각 클래스 이름이나 설명 문장을 텍스트 인코더에 넣어 텍스트 임베딩을 만들고, 테스트 이미지와 가장 잘 맞는 텍스트를 고르면 된다. 논문 그림에서는 `A photo of a {object}.` 같은 프롬프트 템플릿을 사용해 클래스 이름만 넣은 텍스트보다 더 자연스러운 문장을 만들고 있음을 보여 준다.

개념적으로 쓰면, 각 클래스 $c$에 대해 프롬프트 함수 $p(c)$를 만들고, 텍스트 인코더로부터 클래스 벡터 $w_c$를 얻는다.

$$
w_c = f_{\text{text}}(p(c))
$$

테스트 이미지 $x$의 예측은 이미지 임베딩과 클래스 텍스트 임베딩의 유사도를 비교하는 방식이 된다.

$$
\hat{y} = \arg\max_{c \in \mathcal{C}} \cos\big(f_{\text{image}}(x), w_c\big)
$$

논문은 여기에 여러 개의 템플릿을 써서 프롬프트 앙상블을 하는 실험도 수행한다. 예를 들어 단순히 `dog`라는 단어 하나를 넣는 대신, `a photo of a dog`, `a blurry photo of a dog`, `a cropped photo of a dog`처럼 여러 문장을 만들어 그 평균 임베딩을 사용하는 방식이다. 이 단계에서 자연어가 라벨 이름 이상의 역할을 하게 된다. 텍스트는 모델에게 지금 어떤 과제를 하라는지를 더 정확하게 지정하는 인터페이스가 된다.

한편 당시 비지도 및 자기지도학습 컴퓨터 비전 연구는 주로 representation learning을 평가했기 때문에, 저자들도 CLIP의 표현 자체가 얼마나 좋은지 보기 위해 linear probe 평가도 함께 수행한다. 따라서 논문의 후반부는 제로샷 성능과 선형 분류 성능을 나란히 비교하는 형태로 전개된다.

[Table 1 삽입 위치]

## 3. Analysis

### 3.1. Initial Comparison to Visual N-Grams

저자들이 알기로, 기존에 표준 이미지 분류 데이터셋에 대해 CLIP과 유사한 방식의 제로샷 전이를 체계적으로 살핀 대표적인 연구는 Visual N-Grams였다. 이 방법 역시 이미지와 텍스트를 연결해 학습했지만, 성능은 아직 proof-of-concept 단계에 가까웠다. 표 1은 CLIP이 이 선행연구 대비 얼마나 큰 폭으로 향상되었는지를 압축해 보여 준다.

aYahoo, ImageNet, SUN 세 데이터셋 모두에서 CLIP은 Visual N-Grams를 크게 앞선다. 특히 ImageNet에서는 11.5%에서 76.2%로 뛰어오른다. 이는 단순한 점수 상승 이상이다. 사람이 붙인 128만 장의 ImageNet 학습 데이터를 전혀 쓰지 않고도, 원래의 ResNet-50과 맞먹는 수준까지 올라왔다는 뜻이기 때문이다. 논문은 top-5 정확도도 매우 높아서 95% 수준에 이르며, 이는 Inception-V4와 비슷하다고 설명한다.

저자들은 물론 이 비교가 완전한 apples-to-apples는 아니라고 인정한다. 데이터셋, 모델 구조, 학습 목표, 연산 규모가 모두 달라졌기 때문이다. 그럼에도 이 표는 자연어 감독 기반 제로샷 비전 분류기가 더 이상 개념 증명 수준이 아니라, 실제로 강한 실용성을 갖기 시작했음을 보여 주는 상징적인 결과로 읽힌다.

[Figure 4 삽입 위치]

### 3.2. Zero-Shot Performance

컴퓨터 비전에서 제로샷 학습은 전통적으로 보지 못한 객체 클래스로의 일반화를 가리키는 경우가 많다. 하지만 이 논문은 더 넓은 의미를 택한다. 여기서 제로샷은 보지 못한 클래스뿐 아니라, 아예 보지 못한 데이터셋과 과제로의 전이를 의미한다. 저자들은 이를 task learning의 관점으로 해석한다. 즉 좋은 모델은 단순히 표현을 잘 만드는 것을 넘어, 새로운 과제 자체를 얼마나 빠르게 읽어 내는가로도 평가받아야 한다는 입장이다.

이를 위해 논문은 Visual N-Grams가 다뤘던 세 개 데이터셋을 넘어, 30개가 넘는 데이터셋과 50개가 넘는 기존 비전 시스템을 비교 대상으로 삼는다. 우선 가장 기본적인 비교로, ResNet-50 특징 위에 정규화된 logistic regression 분류기를 완전지도학습으로 학습한 기준선과 CLIP의 제로샷 분류기를 맞붙인다. 그림 4는 27개 데이터셋에서 이 비교를 시각화한 것이다.

전체적으로 보면 CLIP의 제로샷 분류기는 이 강한 지도학습 기준선을 약간 앞서며, 27개 중 16개 데이터셋에서 승리한다. 가장 인상적인 사례는 STL10이다. 이 데이터셋은 원래 적은 라벨 수 때문에 비지도학습을 장려하려는 의도로 만들어졌는데, CLIP은 아예 라벨을 쓰지 않고도 99.3%를 기록해 새로운 최고 성능으로 보인다고 논문은 말한다.

세밀한 분류 과제에서는 결과가 꽤 갈린다. Stanford Cars와 Food101에서는 ResNet-50 기반 지도학습 기준선보다 20% 이상 높지만, Flowers102와 FGVCAircraft에서는 오히려 10% 이상 낮다. 저자들은 이를 WIT와 ImageNet이 각 과제에 대해 제공하는 감독 신호의 양 차이로 해석한다. 인터넷 텍스트 안에서 자동차나 음식은 풍부하게 언급될 수 있지만, 꽃 품종이나 항공기 세부 기종은 그렇지 않을 수 있다는 뜻이다.

일반 객체 분류 데이터셋인 ImageNet, CIFAR10, PascalVOC2007에서는 CLIP이 대체로 근소한 우위를 보인다. 특히 흥미로운 것은 행동 인식 비디오 데이터셋에서의 성능이다. Kinetics700에서는 ResNet-50 특징 기반 지도학습 기준선보다 14.5% 높고, UCF101에서도 7.7% 높다. 논문은 이를 ImageNet이 주로 명사 중심의 객체 감독을 주는 반면, 자연어는 동사나 사건 개념까지 더 폭넓게 포함하기 때문일 수 있다고 추측한다.

반대로 CLIP이 눈에 띄게 약한 분야도 있다. 위성 이미지 분류인 EuroSAT와 RESISC45, 병리 이미지에서의 림프절 종양 탐지인 PatchCamelyon, 합성 장면에서의 객체 수 세기인 CLEVRCounts, 교통 표지 인식인 GTSRB, 차량까지의 거리 추정인 KITTI Distance 같은 과제들이 그렇다. 저자들은 이런 결과가 CLIP의 제로샷 능력이 아직 복잡하고 전문적인 과제에서는 부족하다는 점을 보여 준다고 인정한다.

다만 논문은 여기서 한 가지 중요한 단서를 붙인다. 인간도 전혀 경험이 없는 특수 과제를 순수 제로샷으로 잘 하기는 어렵다는 점이다. 예를 들어 림프절 종양 판별은 대부분의 사람에게도 사전 지식이 없으면 불가능에 가깝다. 따라서 어려운 과제에서 제로샷이 낮다고 해서 그것만으로 학습 능력을 단정하기는 조심스럽다는 태도를 취한다.

제로샷을 완전지도학습 기준선과 비교하는 것이 task learning을 보여 준다면, few-shot과의 비교는 제로샷의 직접적인 한계를 더 잘 드러낸다. 저자들은 바로 이 지점을 그림 5에서 분석한다.

[Figure 5 삽입 위치]

그림 5는 CLIP의 제로샷 분류가 동일한 특징 공간 위에서 학습한 few-shot logistic regression보다 의외로 강하다는 점을 보여 준다. 구체적으로는 CLIP의 제로샷 분류가 같은 CLIP 특징 공간 위에서 4-shot logistic regression을 한 것과 비슷한 평균 성능을 보인다. 더 나아가 당시 공개된 여러 모델 가운데 가장 잘 나온 16-shot 선형 분류기 결과와도 거의 맞먹는다.

논문은 이것이 왜 가능한지 직관도 함께 제시한다. 제로샷 CLIP은 자연어를 통해 시각 개념을 직접 전달받는다. 예를 들어 `dog`라는 개념은 텍스트 표현 안에서 비교적 직접적으로 지정된다. 반면 one-shot이나 few-shot의 예시 기반 지도학습은 몇 장의 이미지로부터 그 이미지가 강조하려는 개념이 무엇인지 역으로 추론해야 한다. 한 장의 이미지 안에는 다양한 시각 개념이 들어 있을 수 있고, 학습자는 그중 무엇을 핵심 개념으로 잡아야 하는지 명시적 지시를 받지 못한다. 그래서 예시가 아주 적을 때는 오히려 자연어 설명이 더 강력한 감독이 될 수 있다는 것이다.

또한 다른 모델들의 특징 위에서 few-shot logistic regression을 한 결과와 비교해도, 제로샷 CLIP은 대체로 최고 수준의 16-shot 결과에 근접한다. 이는 CLIP의 제로샷 인터페이스가 단순한 편의 기능이 아니라, 실제로 매우 경쟁력 있는 과제 지정 방식임을 보여 준다.

[Figure 6 삽입 위치]

### 3.3. Representation Learning

논문은 제로샷 성능에 초점을 맞추지만, 표현 자체의 품질도 놓치지 않는다. 이를 위해 가장 널리 쓰이는 linear probe 프로토콜을 적용한다. 핵심은 사전학습된 특징 추출기는 고정한 채, 그 위에 선형 분류기 하나만 학습해 표현의 일반성을 비교하는 것이다. 하이퍼파라미터 개입이 적고 여러 모델을 공정하게 비교하기 좋다는 장점이 있다.

그림 6의 왼쪽은 Kornblith 등의 12개 데이터셋 평가 모음에서의 평균 선형 분류 성능을 보여 준다. 여기서 CLIP 모델은 연산량이 늘어날수록 매우 잘 확장되고, 가장 큰 모델은 당시 최고 수준이던 Noisy Student EfficientNet-L2를 전체 점수와 계산 효율 양쪽에서 약간 앞선다. 특히 Vision Transformer 기반 CLIP은 ResNet 기반 CLIP보다 같은 연산량에서 약 세 배 정도 더 효율적이라고 분석한다. 이는 충분히 큰 데이터에서 ViT가 CNN보다 계산 효율 면에서 유리하다는 기존 관찰을 다시 확인해 준다.

논문의 최고 모델인 ViT-L/14@336px는 이 12개 데이터셋 모음에서 기존 최고 시스템보다 평균 2.6% 높다. 하지만 저자들은 이 평가 모음이 ImageNet과 겹치는 성격의 과제에 다소 편향되어 있을 수 있다고 본다. CLIP은 지리 위치 추정, OCR, 얼굴 표정 분류, 행동 인식처럼 기존 단일 비전 모델이 끝에서 끝까지 랜덤 초기화로 함께 학습한 적이 거의 없던 폭넓은 과제를 다루기 때문이다.

그래서 논문은 더 넓은 27개 데이터셋 평가 모음을 추가로 사용한다. 여기에 GTSRB 같은 교통 표지 인식, 여러 VTAB 계열 데이터셋, 그리고 앞서 언급한 다양한 비전 과제들이 포함된다. 이 더 넓은 평가 모음에서는 CLIP의 장점이 더 뚜렷하게 나타난다. 모델 크기와 상관없이 모든 CLIP 모델이 계산 효율 측면에서 다른 평가 대상들을 앞서고, 최고 모델의 평균 점수 개선 폭도 2.6%에서 5%로 커진다.

즉 CLIP의 표현은 단순히 ImageNet 비슷한 과제에만 강한 것이 아니라, 자연어 감독이 풍부하게 제공할 수 있는 넓은 개념 범위 위에서 더 일반적인 표현을 만들고 있음을 시사한다.

### 3.4. Robustness to Natural Distribution Shift

ImageNet 정확도가 높아질수록 모델이 더 강해졌다고 말하기 쉽지만, 이후 연구들은 이 모델들이 조금만 다른 자연 이미지 분포로 가도 성능이 크게 흔들린다는 사실을 계속 보고해 왔다. Taori 등의 연구는 이를 체계적으로 분석하며, ImageNet 정확도와 분포 이동 데이터셋에서의 정확도 사이에는 대체로 logit 공간에서 선형적인 관계가 있다고 설명한다. 여기서 중요한 개념이 effective robustness와 relative robustness다.

Relative robustness는 바깥 분포 성능 자체가 얼마나 높은가를 뜻하고, effective robustness는 같은 in-distribution 성능을 가진 모델들끼리 비교했을 때 예상보다 얼마나 덜 무너지는가를 뜻한다. 논문은 기존 연구에 등장한 대부분의 모델이 결국 ImageNet에 맞춰 학습되거나 미세조정되었기 때문에, 그 자체가 특정 분포의 우연한 상관관계를 과하게 학습했을 가능성이 있다고 본다.

제로샷 모델은 그런 의미에서 흥미롭다. 특정 데이터셋의 훈련 분포를 직접 보고 최적화한 것이 아니므로, 그 분포에만 통하는 지름길을 이용할 여지가 상대적으로 적을 수 있다. CLIP은 바로 이 가설을 시험한다. 그림 7은 7개의 자연 분포 이동 데이터셋에서 CLIP의 위치를 비교한 결과다.

[Figure 7 삽입 위치]

결과는 분명하다. 모든 제로샷 CLIP 모델은 effective robustness를 큰 폭으로 끌어올리고, ImageNet 정확도와 분포 이동 정확도 사이의 이른바 robustness gap을 최대 75%까지 줄인다. 논문은 CLIP이 Taori 등이 분석한 204개 기존 모델과는 완전히 다른 견고성 전선을 그린다고 말한다. 그림 오른쪽은 바나나 클래스처럼 여러 분포 이동 데이터셋에 공통으로 등장하는 사례를 통해, 같은 ImageNet 정확도를 가진 일반 모델과 제로샷 CLIP이 실제로 어떻게 다른지를 시각적으로 보여 준다.

이 대목은 CLIP의 성능이 단순히 평균 점수만 높은 것이 아니라, 데이터셋 특유의 편향에 덜 묶인 표현을 가질 가능성을 보여 준다는 점에서 중요하다. 논문이 큰 규모의 task-agnostic pre-training과 zero-shot 평가를 함께 강조하는 이유가 여기 있다.

## 4. Data Overlap Analysis

대규모 인터넷 데이터로 사전학습할 때 늘 따라오는 질문은 다운스트림 평가 데이터와의 중복이다. WIT처럼 4억 쌍 규모의 데이터라면, 일부 테스트 예제가 우연히 학습 데이터에 섞여 있을 가능성을 완전히 배제하기 어렵다. 저자들은 이 문제를 중복 제거 분석으로 점검하고, 자세한 내용은 보충자료에 실었다고 말한다.

35개 데이터셋 가운데 9개는 아예 중복이 검출되지 않았다. 전체적으로는 중간값 2.2%, 평균 3.2% 수준의 중복률이 관찰된다. 하지만 이 정도 중복은 전체 정확도를 거의 흔들지 못한다. 실제로 정확도 변화가 0.1%를 넘는 데이터셋은 7개뿐이었고, 그중에서도 Bonferroni 보정 후 통계적으로 유의한 것은 2개뿐이다. 가장 큰 개선도 Birdsnap에서 0.6% 정도에 그친다.

즉 CLIP의 성능을 몇몇 중복 예제로 설명하기는 어렵다는 것이 논문의 결론이다. 저자들은 Mahajan 등의 대규모 약지도학습과 Kolesnikov 등의 대규모 사전학습에서도 비슷한 중복률과 미미한 성능 변화가 보고되었다고 덧붙인다.

## 5. Broader Impacts

논문은 CLIP의 장점을 이야기한 뒤 곧바로 위험도 다룬다. CLIP은 별도의 task-specific training data 없이도 사용자가 원하는 클래스를 직접 설계할 수 있다. 하지만 어떤 클래스 집합을 설계하느냐가 모델의 성능과 편향을 동시에 강하게 좌우한다. 즉 인터페이스가 유연해졌다는 것은 동시에 잘못 설계된 클래스 체계가 해로운 결과를 만들 가능성도 커졌다는 뜻이다.

저자들은 Fairface의 인종 라벨과 함께 `criminal`, `animal` 같은 극단적으로 부적절한 라벨을 후보 집합에 넣는 실험을 예로 든다. 이 경우 CLIP은 0세에서 20세 사이 사람의 이미지를 32.3% 비율로 그런 모욕적 범주에 분류하는 경향을 보인다. 그런데 후보 클래스에 `child`를 추가하면 이 비율이 8.7%로 떨어진다. 이는 모델 편향이 단순히 내부 표현의 문제만이 아니라, 사용자 측 클래스 설계와 상호작용하면서 나타난다는 점을 보여 준다.

논문은 또한 성별과 인종에 따라 `crime`, `non-human` 같은 범주로 분류되는 비율에 차이가 나타난다고 보고한다. 극단적인 주의를 기울여도 차별적 영향의 가능성이 남는다는 뜻이다. 즉 CLIP은 강력하지만, 책임 있는 사용을 위해서는 프롬프트 설계와 평가 체계가 함께 고민되어야 한다.

또 다른 위험은 niche task의 접근 장벽이 낮아진다는 점이다. 별도의 데이터셋 구축 없이도 특정 인물 식별이나 감시 관련 작업을 시도할 수 있기 때문이다. 저자들은 이를 확인하기 위해 CelebA를 사용한 유명인 식별 성능도 측정한다. CLIP은 후보가 100명일 때 top-1 정확도 59.2%, 후보가 1000명일 때 43.3%를 기록한다. 논문은 이것이 상용 수준의 얼굴 인식 시스템에 비해 경쟁력 있는 성능은 아니라고 말하지만, task-agnostic pre-training만으로 이런 수준까지 도달했다는 사실 자체가 프라이버시와 감시 문제를 환기한다고 본다.

## 6. Limitations

논문은 CLIP의 성과를 크게 내세우지만, 한계도 명확히 적는다. 첫째, 제로샷 CLIP의 성능은 자주 ResNet-50 특징 위에 얹은 선형 분류기 정도와 경쟁하는 수준에 머문다. 이 기준선은 이미 전체 SOTA와는 거리가 있다. 다시 말해 CLIP은 인상적인 제로샷 능력을 보여 주지만, 그 자체가 모든 과제에서 절대적인 최고 성능이라는 뜻은 아니다.

저자들은 제로샷 CLIP이 전체 평가 모음에서 전반적 SOTA에 도달하려면 대략 1000배의 추가 연산이 필요하다고 추정한다. 현재 하드웨어로는 비현실적인 규모다. 따라서 앞으로는 단순히 더 큰 데이터와 더 많은 GPU만이 아니라, 계산 효율과 데이터 효율을 함께 높이는 연구가 필요하다는 결론으로 이어진다.

둘째, 논문은 스스로를 엄격한 의미의 완전한 제로샷 연구라고 부르지 않는다. 개발 과정에서 저자들은 반복적으로 validation set 성능을 확인해 모델과 프롬프트를 조정했다. 이는 semi-supervised learning에서 제기된 것과 비슷한 문제를 안고 있다. 실제 배포 환경의 진짜 제로샷은 미리 검증 세트를 들여다볼 수 없는 경우가 많기 때문이다.

셋째, 평가 데이터셋 구성 문제도 있다. Kornblith 등의 12개 데이터셋 모음은 표준화되어 있지만, 논문의 주 분석에 쓰인 27개 데이터셋 묶음은 CLIP의 능력과 어느 정도 공진화한 집합이라고 볼 수 있다. 저자들은 넓은 제로샷 전이 능력을 제대로 재는 새 벤치마크가 필요하다고 인정한다.

마지막으로 자연어가 분류기 인터페이스라는 점 자체에도 한계가 있다. 텍스트만으로 지정하기 어려운 복잡한 과제가 있고, 실제 예시 기반 학습이 주는 정보는 여전히 중요하다. 그런데 CLIP은 few-shot 성능을 직접 최적화하지 않았기 때문에, 오히려 제로샷에서 few-shot으로 넘어갈 때 직관에 어긋나는 성능 하락이 나타나기도 한다. 이는 제로샷 인터페이스와 예시 기반 지도학습이 동일한 것이 아님을 보여 준다.

## 7. Related Work

자연어 감독으로 비전 과제를 학습하려는 아이디어는 갑자기 등장한 것이 아니다. 오래전에는 이미지와 함께 붙은 명사, 형용사를 예측해 내용 기반 이미지 검색을 개선하려는 연구가 있었다. 이후에는 이미지 캡션의 단어를 맞히는 분류기들의 가중치 공간을 학습해 더 데이터 효율적인 표현을 얻으려는 시도도 있었고, 이미지 특징과 텍스트 태그 특징을 함께 다루는 심층 볼츠만 머신 연구도 이어졌다.

웹에서 모은 이미지를 감독으로 삼는 webly supervised learning도 CLIP의 가까운 배경이다. 검색 엔진 결과를 라벨처럼 취급해 시각 개념 분류기를 학습하는 계열의 연구들은, 정제된 데이터셋이 없더라도 인터넷 자체가 거대한 약지도 데이터 원천이 될 수 있음을 보여 주었다. 논문은 특히 `Learning Everything about Anything: Webly-Supervised Visual Concept Learning`을 이름까지 언급하며 CLIP과 목적이 유사하다고 말한다.

제로샷 비전 연구도 CLIP의 직접적인 토대다. 보지 못한 클래스로의 일반화를 다룬 초기 연구들, 이미지와 언어 표현을 연결해 CIFAR10과 ImageNet에서 제로샷 전이를 시도한 연구들, 자연어로부터 분류기를 생성하는 아이디어를 다룬 연구들이 이미 존재했다. CLIP의 새로움은 이 아이디어들을 훨씬 큰 규모의 웹 데이터와 현대적 구조, 그리고 단순한 대조학습으로 밀어붙였다는 데 있다.

또한 비전과 언어의 결합은 이미지 분류를 넘어서 비디오 이해, 강화학습, 시각질문응답 등으로 확장되어 왔다. 논문은 자신들의 연구가 이런 흐름과 연결되어 있지만, 목표는 복합 추론보다는 넓은 범위의 시각 과제를 자연어 인터페이스로 제로샷 전이시키는 데 있다고 정리한다.

## 8. Conclusion

논문은 NLP에서 웹 규모 사전학습이 보여 준 성공 공식을 컴퓨터 비전으로 옮길 수 있는지를 묻고, 그 답이 상당 부분 예라는 쪽으로 기울어 있음을 보여 준다. 자연어와 이미지의 짝맞추기라는 단순한 사전학습 목표만으로도, 모델은 내부적으로 매우 다양한 시각 과제를 학습한다. 이후 자연어 프롬프트를 통해 그 능력을 데이터셋별 추가 학습 없이 곧바로 꺼내 쓸 수 있다.

충분한 규모에 도달하면 이 방식은 과제별 지도학습 모델과 경쟁할 수 있고, 어떤 경우에는 견고성 면에서 더 나은 특성까지 보인다. 동시에 계산 효율, 편향, few-shot 전환, 평가 벤치마크 설계 같은 문제가 여전히 남아 있다. 따라서 이 논문의 결론은 `문제가 끝났다`가 아니라, 자연어 감독 기반 비전 학습이 실제로 통한다는 사실을 대규모 실험으로 입증했고 이제 그 가능성과 위험을 함께 다뤄야 한다는 쪽에 가깝다.

## Acknowledgments

저자들은 먼저 CLIP의 학습 데이터가 될 수 있는 자료를 만들어 낸 수많은 사람들에게 감사를 전한다. 또한 OpenAI 내부에서 이미지 조건부 언어 모델 작업을 했던 Susan Zhang, 의사코드의 오류를 잡아 준 Ishaan Gulrajani, broader impacts 절에 대해 피드백을 준 Irene Solaiman, Miles Brundage, Gillian Hadfield에게 감사를 표한다.

아울러 이 프로젝트에 필요한 소프트웨어와 하드웨어 인프라를 구축한 Acceleration 팀과 Supercomputing 팀의 기여를 강조한다. 마지막으로 NumPy, SciPy, ftfy, TensorFlow, PyTorch, pandas 등 연구 전반에 사용된 오픈소스 소프트웨어 생태계에도 감사한다.

## References

1. Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., Dean, J., Devin, M., Ghemawat, S., Irving, G., Isard, M., et al. Tensorflow: A system for large-scale machine learning. In 12th {USENIX} symposium on operating systems design and implementation ({OSDI} 16), pp. 265–283, 2016.
2. Alayrac, J.-B., Recasens, A., Schneider, R., Arandjelovic, R., Ramapuram, J., De Fauw, J., Smaira, L., Dieleman, S., and Zisserman, A. Self-supervised multimodal versatile networks. arXiv preprint arXiv:2006.16228, 2020.
3. Alcorn, M. A., Li, Q., Gong, Z., Wang, C., Mai, L., Ku, W.- S., and Nguyen, A. Strike (with) a pose: Neural networks are easily fooled by strange poses of familiar objects. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 4845–4854, 2019.
4. Assiri, Y. Stochastic optimization of plain convolutional neural networks with simple methods. arXiv preprint arXiv:2001.08856, 2020.
5. Barbu, A., Mayo, D., Alverio, J., Luo, W., Wang, C., Gutfreund, D., Tenenbaum, J., and Katz, B. Objectnet: A large-scale bias-controlled dataset for pushing the limits of object recognition models. In Advances in Neural Information Processing Systems, pp. 9453–9463, 2019.
6. Bechmann, A. and Bowker, G. C. Unsupervised by any other name: Hidden layers of knowledge production in artificial intelligence on social media. Big Data & Society, 6(1):205395171881956, January 2019. doi: 10.1177/ 2053951718819569. URL https://doi.org/10. 1177/2053951718819569.
7. Blaise Aguera y Arcas, M. M. and Todorov, A. Physiognomy's new clothes. 2017. URL https://medium.com/@blaisea/ physiognomys-new-clothes-f2d4b59fdd6a.
8. Bolukbasi, T., Chang, K.-W., Zou, J. Y., Saligrama, V., and Kalai, A. T. Man is to computer programmer as woman is to homemaker? debiasing word embeddings. Advances in neural information processing systems, 29:4349–4357, 2016.
9. Bowker, G. C. and Star, S. L. Sorting things out: Classification and its consequences. MIT press, 2000.
10. Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 2020.
11. Browne, S. Dark Matters: Surveillance of Blackness. Duke University Press, 2015.
12. Bulent Sariyildiz, M., Perez, J., and Larlus, D. Learning visual representations with caption annotations. arXiv e-prints, pp. arXiv–2008, 2020.
13. Buolamwini, J. and Gebru, T. Gender shades: Intersectional accuracy disparities in commercial gender classification. In Conference on fairness, accountability and transparency, pp. 77–91, 2018.
14. Carreira, J., Noland, E., Hillier, C., and Zisserman, A. A short note on the kinetics-700 human action dataset. arXiv preprint arXiv:1907.06987, 2019.
15. Chen, T., Kornblith, S., Swersky, K., Norouzi, M., and Hinton, G. Big self-supervised models are strong semisupervised learners. arXiv preprint arXiv:2006.10029, 2020a.
16. Chen, X., Fan, H., Girshick, R., and He, K. Improved baselines with momentum contrastive learning. arXiv preprint arXiv:2003.04297, 2020b.
17. Chen, Y.-C., Li, L., Yu, L., Kholy, A. E., Ahmed, F., Gan, Z., Cheng, Y., and Liu, J. Uniter: Learning universal imagetext representations. arXiv preprint arXiv:1909.11740, 2019.
18. Cheng, G., Han, J., and Lu, X. Remote sensing image scene classification: Benchmark and state of the art. Proceedings of the IEEE, 105(10):1865–1883, 2017.
19. Church, K. W. and Hanks, P. Word association norms, mutual information, and lexicography. Computational Linguistics, 16(1):22–29, 1990. URL https://www. aclweb.org/anthology/J90-1003.
20. Coates, A., Ng, A., and Lee, H. An analysis of singlelayer networks in unsupervised feature learning. In Proceedings of the fourteenth international conference on artificial intelligence and statistics, pp. 215–223, 2011.
21. Crawford, K. The trouble with bias. NIPS 2017 Keynote, 2017. URL https://www.youtube.com/ watch?v=fMym_BKWQzk.
22. Dai, A. M. and Le, Q. V. Semi-supervised sequence learning. In Advances in neural information processing systems, pp. 3079–3087, 2015. D'Amour, A., Heller, K., Moldovan, D., Adlam, B., Alipanahi, B., Beutel, A., Chen, C., Deaton, J., Eisenstein, J., Hoffman, M. D., et al. Underspecification presents challenges for credibility in modern machine learning. arXiv preprint arXiv:2011.03395, 2020.
23. Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei- Fei, L. ImageNet: A Large-Scale Hierarchical Image Database. In CVPR09, 2009.
24. Deng, J., Berg, A. C., Satheesh, S., Su, H., Khosla, A., and Fei-Fei, L. Ilsvrc 2012, 2012. URL http://www. image-net.org/challenges/LSVRC/2012/.
25. Desai, K. and Johnson, J. Virtex: Learning visual representations from textual annotations. arXiv preprint arXiv:2006.06666, 2020.
26. Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.
27. Divvala, S. K., Farhadi, A., and Guestrin, C. Learning everything about anything: Webly-supervised visual concept learning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 3270– 3277, 2014.
28. Dodge, S. and Karam, L. A study and comparison of human and deep learning recognition performance under visual distortions. In 2017 26th international conference on computer communication and networks (ICCCN), pp. 1– 7. IEEE, 2017.
29. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.
30. Elhoseiny, M., Saleh, B., and Elgammal, A. Write a classifier: Zero-shot learning using purely textual descriptions. In Proceedings of the IEEE International Conference on Computer Vision, pp. 2584–2591, 2013.
31. Fergus, R., Fei-Fei, L., Perona, P., and Zisserman, A. Learning object categories from google's image search. In Tenth IEEE International Conference on Computer Vision (ICCV'05) Volume 1, volume 2, pp. 1816–1823. IEEE, 2005.
32. Frome, A., Corrado, G. S., Shlens, J., Bengio, S., Dean, J., Ranzato, M., and Mikolov, T. Devise: A deep visualsemantic embedding model. In Advances in neural information processing systems, pp. 2121–2129, 2013.
33. Gan, Z., Chen, Y.-C., Li, L., Zhu, C., Cheng, Y., and Liu, J. Large-scale adversarial training for vision-and-language representation learning. arXiv preprint arXiv:2006.06195, 2020.
34. Gao, T., Fisch, A., and Chen, D. Making pre-trained language models better few-shot learners. arXiv preprint arXiv:2012.15723, 2020.
35. Garvie, C., May 2019. URL https://www. flawedfacedata.com/.
36. Geiger, A., Lenz, P., and Urtasun, R. Are we ready for autonomous driving? the kitti vision benchmark suite. In Conference on Computer Vision and Pattern Recognition (CVPR), 2012.
37. Geirhos, R., Rubisch, P., Michaelis, C., Bethge, M., Wichmann, F. A., and Brendel, W. Imagenet-trained cnns are biased towards texture; increasing shape bias improves accuracy and robustness. arXiv preprint arXiv:1811.12231, 2018.
38. Goodfellow, I. J., Erhan, D., Carrier, P. L., Courville, A., Mirza, M., Hamner, B., Cukierski, W., Tang, Y., Thaler, D., Lee, D.-H., et al. Challenges in representation learning: A report on three machine learning contests. Neural Networks, 64:59–63, 2015.
39. Google. Google cloud api: Celebrity recognition. URL https://cloud.google.com/vision/docs/ celebrity-recognition.
40. Grill, J.-B., Strub, F., Altche, F., Tallec, C., Richemond, P. H., Buchatskaya, E., Doersch, C., Pires, B. A., Guo, Z. D., Azar, M. G., et al. Bootstrap your own latent: A new approach to self-supervised learning. arXiv preprint arXiv:2006.07733, 2020.
41. Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A., Fernandez del Rıo, J., Wiebe, M., Peterson, P., Gerard-Marchant, P., Sheppard, K., Reddy, T., Weckesser, W., Abbasi, H., Gohlke, C., and Oliphant, T. E. Array programming with NumPy. Nature, 585:357–362, 2020. doi: 10.1038/ s41586-020-2649-2.
42. Hays, J. and Efros, A. A. Im2gps: estimating geographic information from a single image. In 2008 ieee conference on computer vision and pattern recognition, pp. 1–8. IEEE, 2008.
43. He, K., Zhang, X., Ren, S., and Sun, J. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE international conference on computer vision, pp. 1026–1034, 2015.
44. He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016a.
45. He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016b.
46. He, K., Fan, H., Wu, Y., Xie, S., and Girshick, R. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9729– 9738, 2020.
47. He, T., Zhang, Z., Zhang, H., Zhang, Z., Xie, J., and Li, M. Bag of tricks for image classification with convolutional neural networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 558– 567, 2019.
48. Helber, P., Bischke, B., Dengel, A., and Borth, D. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12(7):2217–2226, 2019.
49. Henaff, O. Data-efficient image recognition with contrastive predictive coding. In International Conference on Machine Learning, pp. 4182–4192. PMLR, 2020.
50. Hendrycks, D. and Gimpel, K. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016.
51. Hendrycks, D., Liu, X., Wallace, E., Dziedzic, A., Krishnan, R., and Song, D. Pretrained transformers improve out-ofdistribution robustness. arXiv preprint arXiv:2004.06100, 2020.
52. Hermann, K. M., Hill, F., Green, S., Wang, F., Faulkner, R., Soyer, H., Szepesvari, D., Czarnecki, W. M., Jaderberg, M., Teplyashin, D., et al. Grounded language learning in a simulated 3d world. arXiv preprint arXiv:1706.06551, 2017.
53. Hestness, J., Narang, S., Ardalani, N., Diamos, G., Jun, H., Kianinejad, H., Patwary, M., Ali, M., Yang, Y., and Zhou, Y. Deep learning scaling is predictable, empirically. arXiv preprint arXiv:1712.00409, 2017.
54. Hongsuck Seo, P., Weyand, T., Sim, J., and Han, B. Cplanet: Enhancing image geolocalization by combinatorial partitioning of maps. In Proceedings of the European Conference on Computer Vision (ECCV), pp. 536–551, 2018.
55. Howard, J. and Ruder, S. Universal language model fine-tuning for text classification. arXiv preprint arXiv:1801.06146, 2018.
56. Ioffe, S. and Szegedy, C. Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167, 2015.
57. Jaderberg, M., Simonyan, K., Vedaldi, A., and Zisserman, A. Deep structured output learning for unconstrained text recognition. arXiv preprint arXiv:1412.5903, 2014.
58. Jaderberg, M., Simonyan, K., Zisserman, A., et al. Spatial transformer networks. Advances in neural information processing systems, 28:2017–2025, 2015.
59. Johnson, J., Hariharan, B., van der Maaten, L., Fei-Fei, L., Lawrence Zitnick, C., and Girshick, R. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 2901–2910, 2017.
60. Joulin, A., Van Der Maaten, L., Jabri, A., and Vasilache, N. Learning visual features from large weakly supervised data. In European Conference on Computer Vision, pp. 67–84. Springer, 2016.
61. Kalfaoglu, M., Kalkan, S., and Alatan, A. A. Late temporal modeling in 3d cnn architectures with bert for action recognition. arXiv preprint arXiv:2008.01232, 2020.
62. Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.
63. Keyes, O. The misgendering machines: Trans/hci implications of automatic gender recognition. Proceedings of the ACM on Human-Computer Interaction, 2(CSCW):1–22, 2018.
64. Kiela, D., Firooz, H., Mohan, A., Goswami, V., Singh, A., Ringshia, P., and Testuggine, D. The hateful memes challenge: Detecting hate speech in multimodal memes. arXiv preprint arXiv:2005.04790, 2020.
65. Kolesnikov, A., Beyer, L., Zhai, X., Puigcerver, J., Yung, J., Gelly, S., and Houlsby, N. Large scale learning of general visual representations for transfer. arXiv preprint arXiv:1912.11370, 2019.
66. Kornblith, S., Shlens, J., and Le, Q. V. Do better imagenet models transfer better? In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2661–2671, 2019.
67. Krishna, R., Zhu, Y., Groth, O., Johnson, J., Hata, K., Kravitz, J., Chen, S., Kalantidis, Y., Li, L.-J., Shamma, D. A., et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision, 123(1):32–73, 2017. Karkkainen, K. and Joo, J. Fairface: Face attribute dataset for balanced race, gender, and age, 2019.
68. Lake, B. M., Ullman, T. D., Tenenbaum, J. B., and Gershman, S. J. Building machines that learn and think like people, 2016.
69. Lampert, C. H., Nickisch, H., and Harmeling, S. Learning to detect unseen object classes by between-class attribute transfer. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pp. 951–958. IEEE, 2009.
70. Larochelle, H., Erhan, D., and Bengio, Y. Zero-data learning of new tasks. 2008.
71. LeCun, Y. The mnist database of handwritten digits. http://yann. lecun. com/exdb/mnist/.
72. Lei Ba, J., Swersky, K., Fidler, S., et al. Predicting deep zero-shot convolutional neural networks using textual descriptions. In Proceedings of the IEEE International Conference on Computer Vision, pp. 4247–4255, 2015.
73. Li, A., Jabri, A., Joulin, A., and van der Maaten, L. Learning visual n-grams from web data. In Proceedings of the IEEE International Conference on Computer Vision, pp. 4183–4192, 2017.
74. Li, G., Duan, N., Fang, Y., Gong, M., and Jiang, D. Unicoder-vl: A universal encoder for vision and language by cross-modal pre-training. 2020a.
75. Li, X., Yin, X., Li, C., Hu, X., Zhang, P., Zhang, L., Wang, L., Hu, H., Dong, L., Wei, F., et al. Oscar: Objectsemantics aligned pre-training for vision-language tasks. arXiv preprint arXiv:2004.06165, 2020b.
76. Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollar, P., and Zitnick, C. L. Microsoft coco: Common objects in context. In European conference on computer vision, pp. 740–755. Springer, 2014.
77. Linzen, T. How can we accelerate progress towards human-like linguistic generalization? arXiv preprint arXiv:2005.00955, 2020.
78. Lippe, P., Holla, N., Chandra, S., Rajamanickam, S., Antoniou, G., Shutova, E., and Yannakoudakis, H. A multimodal framework for the detection of hateful memes. arXiv preprint arXiv:2012.12871, 2020.
79. Liu, Z., Luo, P., Wang, X., and Tang, X. Large-scale celebfaces attributes (celeba) dataset. Retrieved August, 15 (2018):11, 2018.
80. Lu, J., Batra, D., Parikh, D., and Lee, S. Vilbert: Pretraining task-agnostic visiolinguistic representations for visionand-language tasks. In Advances in Neural Information Processing Systems, pp. 13–23, 2019.
81. Lu, Z., Xiong, X., Li, Y., Stroud, J., and Ross, D. Leveraging weakly supervised data and pose representation for action recognition, 2020. URL https://www.youtube. com/watch?v=KOQFxbPPLOE&t=1390s.
82. Mahajan, D., Girshick, R., Ramanathan, V., He, K., Paluri, M., Li, Y., Bharambe, A., and van der Maaten, L. Exploring the limits of weakly supervised pretraining. In Proceedings of the European Conference on Computer Vision (ECCV), pp. 181–196, 2018.
83. McCann, B., Keskar, N. S., Xiong, C., and Socher, R. The natural language decathlon: Multitask learning as question answering. arXiv preprint arXiv:1806.08730, 2018.
84. Miech, A., Zhukov, D., Alayrac, J.-B., Tapaswi, M., Laptev, I., and Sivic, J. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In Proceedings of the IEEE international conference on computer vision, pp. 2630–2640, 2019.
85. Miech, A., Alayrac, J.-B., Laptev, I., Sivic, J., and Zisserman, A. Rareact: A video dataset of unusual interactions. arXiv preprint arXiv:2008.01018, 2020a.
86. Miech, A., Alayrac, J.-B., Smaira, L., Laptev, I., Sivic, J., and Zisserman, A. End-to-end learning of visual representations from uncurated instructional videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9879–9889, 2020b.
87. Miller, G. A. Wordnet: a lexical database for english. Communications of the ACM, 38(11):39–41, 1995.
88. Miller, J., Krauth, K., Recht, B., and Schmidt, L. The effect of natural distribution shift on question answering models. arXiv preprint arXiv:2004.14444, 2020.
89. Mishra, A., Alahari, K., and Jawahar, C. Scene text recognition using higher order language priors. 2012.
90. Mori, Y., Takahashi, H., and Oka, R. Image-to-word transformation based on dividing and vector quantizing images with words. Citeseer, 1999.
91. Muller-Budack, E., Pustu-Iren, K., and Ewerth, R. Geolocation estimation of photos using a hierarchical model and scene classification. In Proceedings of the European Conference on Computer Vision (ECCV), pp. 563–579, 2018.
92. Netzer, Y., Wang, T., Coates, A., Bissacco, A., Wu, B., and Ng, A. Y. Reading digits in natural images with unsupervised feature learning. 2011.
93. Noble, S. U. Algorithms of oppression: How search engines reinforce racism. 2018.
94. Nosek, B. A., Banaji, M. R., and Greenwald, A. G. Harvesting implicit group attitudes and beliefs from a demonstration web site. Group Dynamics: Theory, Research, and Practice, 6(1):101, 2002.
95. Oh, S., Hoogs, A., Perera, A., Cuntoor, N., Chen, C.-C., Lee, J. T., Mukherjee, S., Aggarwal, J., Lee, H., Davis, L., et al. A large-scale benchmark dataset for event recognition in surveillance video. In CVPR 2011, pp. 3153–3160. IEEE, 2011.
96. Oliver, A., Odena, A., Raffel, C. A., Cubuk, E. D., and Goodfellow, I. Realistic evaluation of deep semi-supervised learning algorithms. Advances in neural information processing systems, 31:3235–3246, 2018.
97. pandas development team, T. pandas-dev/pandas: Pandas, February 2020. URL https://doi.org/10. 5281/zenodo.3509134.
98. Parkhi, O. M., Vedaldi, A., Zisserman, A., and Jawahar, C. V. Cats and dogs. In IEEE Conference on Computer Vision and Pattern Recognition, 2012.
99. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., and Chintala, S. Pytorch: An imperative style, high-performance deep learning library. In Wallach, H., Larochelle, H., Beygelzimer, A., d'Alche-Buc, F., Fox, E., and Garnett, R. (eds.), Advances in Neural Information Processing Systems 32, pp. 8024–8035, 2019.
100. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., and Duchesnay, E. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12:2825–2830, 2011.
101. Pennington, J., Socher, R., and Manning, C. D. Glove: Global vectors for word representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pp. 1532–1543, 2014.
102. Peters, M. E., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K., and Zettlemoyer, L. Deep contextualized word representations. arXiv preprint arXiv:1802.05365, 2018.
103. Qi, D., Su, L., Song, J., Cui, E., Bharti, T., and Sacheti, A. Imagebert: Cross-modal pre-training with largescale weak-supervised image-text data. arXiv preprint arXiv:2001.07966, 2020.
104. Quattoni, A., Collins, M., and Darrell, T. Learning visual representations using images with captions. In 2007 IEEE Conference on Computer Vision and Pattern Recognition, pp. 1–8. IEEE, 2007.
105. Radford, A., Narasimhan, K., Salimans, T., and Sutskever, I. Improving language understanding by generative pretraining, 2018.
106. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., and Sutskever, I. Language models are unsupervised multitask learners. 2019.
107. Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. Exploring the limits of transfer learning with a unified text-to-text transformer. arXiv preprint arXiv:1910.10683, 2019.
108. Raji, I. D., Gebru, T., Mitchell, M., Buolamwini, J., Lee, J., and Denton, E. Saving face: Investigating the ethical concerns of facial recognition auditing, 2020.
109. Ramanathan, V., Liang, P., and Fei-Fei, L. Video event understanding using natural language descriptions. In Proceedings of the IEEE International Conference on Computer Vision, pp. 905–912, 2013.
110. Recht, B., Roelofs, R., Schmidt, L., and Shankar, V. Do imagenet classifiers generalize to imagenet? arXiv preprint arXiv:1902.10811, 2019.
111. Salimans, T. and Kingma, D. P. Weight normalization: A simple reparameterization to accelerate training of deep neural networks. In Advances in neural information processing systems, pp. 901–909, 2016.
112. Scheuerman, M. K., Paul, J. M., and Brubaker, J. R. How computers see gender: An evaluation of gender classification in commercial facial analysis services. Proceedings of the ACM on Human-Computer Interaction, 3(CSCW): 1–33, 2019.
113. Schwemmer, C., Knight, C., Bello-Pardo, E. D., Oklobdzija, S., Schoonvelde, M., and Lockhart, J. W. Diagnosing gender bias in image recognition systems. Socius, 6: 2378023120967171, 2020.
114. Sennrich, R., Haddow, B., and Birch, A. Neural machine translation of rare words with subword units. arXiv preprint arXiv:1508.07909, 2015.
115. Singh, A., Natarajan, V., Shah, M., Jiang, Y., Chen, X., Batra, D., Parikh, D., and Rohrbach, M. Towards vqa models that can read. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 8317–8326, 2019.
116. Socher, R., Ganjoo, M., Manning, C. D., and Ng, A. Zeroshot learning through cross-modal transfer. In Advances in neural information processing systems, pp. 935–943, 2013a.
117. Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A. Y., and Potts, C. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 conference on empirical methods in natural language processing, pp. 1631–1642, 2013b.
118. Sohn, K. Improved deep metric learning with multi-class n-pair loss objective. In Advances in neural information processing systems, pp. 1857–1865, 2016.
119. Solaiman, I., Brundage, M., Clark, J., Askell, A., Herbert- Voss, A., Wu, J., Radford, A., Krueger, G., Kim, J. W., Kreps, S., McCain, M., Newhouse, A., Blazakis, J., McGuffie, K., and Wang, J. Release strategies and the social impacts of language models, 2019.
120. Soomro, K., Zamir, A. R., and Shah, M. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402, 2012.
121. Speer, R. ftfy. Zenodo, 2019. URL https://doi.org/ 10.5281/zenodo.2591652. Version 5.5.
122. Srivastava, N. and Salakhutdinov, R. Multimodal learning with deep boltzmann machines. In NIPS, 2012.
123. Stallkamp, J., Schlipsing, M., Salmen, J., and Igel, C. The German Traffic Sign Recognition Benchmark: A multiclass classification competition. In IEEE International Joint Conference on Neural Networks, pp. 1453–1460, 2011.
124. Szegedy, C., Ioffe, S., Vanhoucke, V., and Alemi, A. Inception-v4, inception-resnet and the impact of residual connections on learning. arXiv preprint arXiv:1602.07261, 2016.
125. Tan, H. and Bansal, M. Lxmert: Learning cross-modality encoder representations from transformers. arXiv preprint arXiv:1908.07490, 2019.
126. Tan, M. and Le, Q. V. Efficientnet: Rethinking model scaling for convolutional neural networks. arXiv preprint arXiv:1905.11946, 2019.
127. Taori, R., Dave, A., Shankar, V., Carlini, N., Recht, B., and Schmidt, L. Measuring robustness to natural distribution shifts in image classification. arXiv preprint arXiv:2007.00644, 2020.
128. Thomee, B., Shamma, D. A., Friedland, G., Elizalde, B., Ni, K., Poland, D., Borth, D., and Li, L.-J. Yfcc100m: The new data in multimedia research. Communications of the ACM, 59(2):64–73, 2016.
129. Tian, Y., Krishnan, D., and Isola, P. Contrastive multiview coding. arXiv preprint arXiv:1906.05849, 2019.
130. Tian, Y., Wang, Y., Krishnan, D., Tenenbaum, J. B., and Isola, P. Rethinking few-shot image classification: a good embedding is all you need? arXiv preprint arXiv:2003.11539, 2020.
131. Touvron, H., Vedaldi, A., Douze, M., and Jegou, H. Fixing the train-test resolution discrepancy. In Advances in neural information processing systems, pp. 8252–8262, 2019.
132. Varadarajan, J. and Odobez, J.-M. Topic models for scene analysis and abnormality detection. In 2009 IEEE 12th International Conference on Computer Vision Workshops, ICCV Workshops, pp. 1338–1345. IEEE, 2009.
133. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. In Advances in neural information processing systems, pp. 5998–6008, 2017.
134. Veeling, B. S., Linmans, J., Winkens, J., Cohen, T., and Welling, M. Rotation equivariant CNNs for digital pathology. June 2018.
135. Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., van der Walt, S. J., Brett, M., Wilson, J., Millman, K. J., Mayorov, N., Nelson, A. R. J., Jones, E., Kern, R., Larson, E., Carey, C. J., Polat, ˙I., Feng, Y., Moore, E. W., VanderPlas, J., Laxalde, D., Perktold, J., Cimrman, R., Henriksen, I., Quintero, E. A., Harris, C. R., Archibald, A. M., Ribeiro, A. H., Pedregosa, F., van Mulbregt, P., and SciPy 1.0 Contributors. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17:261–272, 2020. doi: 10.1038/s41592-019-0686-2.
136. Vo, N., Jacobs, N., and Hays, J. Revisiting im2gps in the deep learning era. In Proceedings of the IEEE International Conference on Computer Vision, pp. 2621–2630, 2017.
137. Wang, A., Singh, A., Michael, J., Hill, F., Levy, O., and Bowman, S. R. Glue: A multi-task benchmark and analysis platform for natural language understanding. arXiv preprint arXiv:1804.07461, 2018.
138. Wang, H., Lu, P., Zhang, H., Yang, M., Bai, X., Xu, Y., He, M., Wang, Y., and Liu, W. All you need is boundary: Toward arbitrary-shaped text spotting. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pp. 12160–12167, 2020.
139. Weyand, T., Kostrikov, I., and Philbin, J. Planet-photo geolocation with convolutional neural networks. In European Conference on Computer Vision, pp. 37–55. Springer, 2016.
140. Wu, Y., Kirillov, A., Massa, F., Lo, W.-Y., and Girshick, R. Detectron2. https://github.com/ facebookresearch/detectron2, 2019.
141. Xie, Q., Luong, M.-T., Hovy, E., and Le, Q. V. Self-training with noisy student improves imagenet classification. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10687–10698, 2020.
142. Yang, Z., Lu, Y., Wang, J., Yin, X., Florencio, D., Wang, L., Zhang, C., Zhang, L., and Luo, J. Tap: Text-aware pre-training for text-vqa and text-caption. arXiv preprint arXiv:2012.04638, 2020.
143. Yogatama, D., d'Autume, C. d. M., Connor, J., Kocisky, T., Chrzanowski, M., Kong, L., Lazaridou, A., Ling, W., Yu, L., Dyer, C., et al. Learning and evaluating general linguistic intelligence. arXiv preprint arXiv:1901.11373, 2019.
144. Yu, F., Tang, J., Yin, W., Sun, Y., Tian, H., Wu, H., and Wang, H. Ernie-vil: Knowledge enhanced visionlanguage representations through scene graph. arXiv preprint arXiv:2006.16934, 2020.
145. Zeiler, M. D. and Fergus, R. Visualizing and understanding convolutional networks. In European conference on computer vision, pp. 818–833. Springer, 2014.
146. Zhai, X., Puigcerver, J., Kolesnikov, A., Ruyssen, P., Riquelme, C., Lucic, M., Djolonga, J., Pinto, A. S., Neumann, M., Dosovitskiy, A., et al. A large-scale study of representation learning with the visual task adaptation benchmark. arXiv preprint arXiv:1910.04867, 2019.
147. Zhang, R. Making convolutional networks shift-invariant again. arXiv preprint arXiv:1904.11486, 2019.
148. Zhang, Y., Jiang, H., Miura, Y., Manning, C. D., and Langlotz, C. P. Contrastive learning of medical visual representations from paired images and text. arXiv preprint arXiv:2010.00747, 2020.
149. Zuboff, S. Big other: surveillance capitalism and the prospects of an information civilization. Journal of Information Technology, 30(1):75–89, 2015.

## 추가 설명

### 1. 왜 캡션 생성보다 대조학습이 더 효율적인가

캡션 생성은 이미지 하나를 보고 문장의 모든 토큰을 차례대로 맞혀야 한다. 이때 모델은 시각 정보뿐 아니라 언어적 유창성까지 함께 책임져야 하므로, 제로샷 분류에 꼭 필요하지 않은 계산도 많이 하게 된다. 반면 CLIP의 목표는 훨씬 직접적이다. 이 이미지와 이 문장이 서로 맞는가만 판단하면 된다. 그래서 모델의 용량과 연산이 이미지-텍스트 정렬 자체에 집중된다.

또한 배치 안의 다른 샘플들이 자연스럽게 음성 예시로 작동하기 때문에, 별도의 복잡한 negative sampling 없이도 표현을 날카롭게 만들 수 있다. 같은 계산량을 썼을 때 제로샷 전이 속도가 더 빠른 이유가 여기에 있다.

### 2. CLIP의 손실 함수는 실제로 무엇을 배우게 만드는가

CLIP의 유사도 행렬 $S$는 행 기준으로 보면 한 이미지가 여러 텍스트 가운데 무엇과 가장 가까운지를 나타내고, 열 기준으로 보면 한 텍스트가 여러 이미지 가운데 무엇과 가장 가까운지를 나타낸다. 이미지 방향 손실은 각 이미지가 자기 짝 텍스트를 고르도록 만들고, 텍스트 방향 손실은 각 텍스트가 자기 짝 이미지를 고르도록 만든다.

두 방향을 모두 최적화한다는 점이 중요하다. 이미지에서 텍스트만 찾게 하면 텍스트 임베딩 공간이 느슨해질 수 있고, 텍스트에서 이미지만 찾게 해도 반대 문제가 생긴다. 대칭 손실은 두 모달리티가 같은 의미 공간을 공유하도록 강하게 압박한다. 이 구조 덕분에 나중에 텍스트만 넣어도 분류기처럼 쓸 수 있고, 이미지-텍스트 검색에도 바로 활용할 수 있다.

### 3. 왜 자연어 프롬프트가 분류기 역할을 할 수 있는가

전통적인 분류기는 학습 데이터로부터 각 클래스의 결정경계를 직접 파라미터화한다. 하지만 CLIP에서는 텍스트 인코더가 이미 방대한 웹 데이터 속에서 `dog`, `car`, `a photo of a dog` 같은 표현이 어떤 시각 개념과 연결되는지를 학습했다. 따라서 클래스 이름을 문장으로 만들어 임베딩하면, 그것이 곧 해당 클래스 방향을 가리키는 벡터가 된다.

이 관점에서 보면 프롬프트 엔지니어링이 중요한 이유도 분명해진다. 단어 하나만 던지는 것보다, 실제 이미지-텍스트 데이터에서 더 자주 등장했을 법한 문장 형식으로 바꿔 주면 텍스트 인코더가 학습 중 보았던 분포와 더 가까워진다. 그래서 `a photo of a dog` 같은 템플릿이 성능을 끌어올린다.

### 4. 왜 제로샷 CLIP이 때로 few-shot 선형 분류기보다 강한가

예시가 아주 적을 때의 지도학습은 생각보다 불안정하다. 한 장의 이미지가 보여 주는 정보는 너무 많기 때문이다. 예를 들어 자동차 사진 한 장을 주면, 학습자는 차종을 배워야 하는지 색을 배워야 하는지 배경을 배워야 하는지 명시적으로 알 수 없다. 이런 모호성은 샘플 수가 적을수록 커진다.

반면 자연어는 개념을 직접 지정한다. `A photo of a dog`라는 표현은 지금 배워야 할 것이 개라는 점을 바로 알려 준다. CLIP은 이미 대규모 사전학습으로 시각-언어 정렬을 익혔기 때문에, 이 자연어 지시를 곧바로 분류 방향으로 해석할 수 있다. 그래서 1-shot이나 2-shot처럼 정보가 부족한 상황에서는 제로샷이 오히려 더 안정적일 수 있다.

### 5. CLIP이 강한 과제와 약한 과제가 갈리는 이유

CLIP은 웹 텍스트가 풍부하게 제공하는 개념일수록 강하다. 일반 객체, 음식, 자동차, 행동, OCR처럼 인터넷에 이미지와 설명이 많이 붙는 개념은 자연어 감독만으로도 잘 학습된다. 반대로 위성영상, 의료영상, 정밀 계수, 거리 추정처럼 전문적이거나 텍스트로 드러나기 어려운 과제는 약해지기 쉽다.

즉 CLIP의 한계는 단순히 모델 구조의 한계만이 아니라, 웹에서 관찰 가능한 감독 신호의 성격과도 깊게 연결되어 있다. 이 점을 이해하면, 왜 어떤 벤치마크에서는 압도적이고 어떤 벤치마크에서는 크게 밀리는지가 더 분명해진다.

### 6. 왜 ViT 기반 CLIP이 더 계산 효율적으로 보였는가

논문에서 ViT 기반 CLIP은 ResNet 기반 CLIP보다 같은 GFLOPs당 더 높은 선형 분류 성능을 낸다. 이는 충분히 큰 데이터와 적절한 학습 설정이 주어질 때, ViT가 전역적 상호작용을 더 직접적으로 처리하기 쉬운 구조라는 점과 연결된다. CLIP의 사전학습은 이미지 전체와 텍스트 전체의 의미 정렬이 핵심이므로, 지역적 패턴을 단계적으로 쌓아 가는 CNN보다 패치 단위 토큰들을 전역적으로 섞는 ViT가 더 잘 맞아떨어졌을 가능성이 크다.

물론 이것이 언제나 ViT가 절대적으로 우월하다는 뜻은 아니다. 데이터 규모와 연산 예산이 줄어들면 상황이 달라질 수 있다. 논문의 메시지는 특정 구조의 승패라기보다, 자연어 감독 대규모 사전학습이라는 환경에서는 ViT가 특히 좋은 계산 효율을 보였다는 쪽에 가깝다.

### 7. 제로샷 CLIP의 견고성이 더 좋아 보이는 이유

기존 ImageNet 모델은 결국 ImageNet 분포 위에서 직접 파라미터를 조정한다. 이 과정에서 그 데이터셋에 특화된 배경, 촬영 방식, 구도 같은 우연한 상관관계를 함께 학습할 수 있다. 반면 제로샷 CLIP은 특정 다운스트림 데이터셋의 훈련 샘플을 보지 않고, 훨씬 넓은 웹 분포에서 학습한 뒤 텍스트 프롬프트만으로 과제를 지정한다.

그래서 ImageNet에만 통하는 지름길에 덜 기대게 되고, 자연 분포 이동이 생겼을 때도 상대적으로 덜 무너질 수 있다. 논문이 말하는 effective robustness의 향상은 바로 이런 차이를 수량화한 결과라고 볼 수 있다.

### 8. 이 논문의 역사적 의미

이 논문의 진짜 의미는 `비전에도 거대한 웹 사전학습이 통한다`는 점을 설득력 있게 보여 주었다는 데 있다. 이후 등장한 ALIGN, LiT, Florence, Flamingo, BLIP 계열, 최근의 대규모 vision-language model 흐름은 모두 어느 정도 이 논문이 열어 놓은 길 위에서 확장되었다. 특히 텍스트를 단순한 보조 정보가 아니라, 학습 목표이자 추론 인터페이스로 동시에 사용하는 발상은 이후 멀티모달 연구 전반의 기본 틀이 되었다.

동시에 이 논문은 편향과 감시 가능성, 평가 절차의 허점, 데이터 중복 문제를 본문 안에서 함께 다뤘다는 점에서도 중요하다. 성능만이 아니라 사용 맥락까지 논문 구조 안에 넣었다는 점이, 이후 대규모 멀티모달 연구를 읽는 기준점 역할을 한다.
