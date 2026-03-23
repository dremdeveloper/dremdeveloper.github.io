# OCR-free Document Understanding Transformer

Geewook Kim, Teakgyu Hong, Moonbin Yim, Jeongyeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, Seunghyun Park

NAVER CLOVA, NAVER Search, NAVER AI Lab, Upstage, Tmax, Google, LBox

## Abstract

이 논문은 문서 이미지 이해를 위해 OCR에 의존하지 않는 end-to-end Transformer 모델을 제안한다. 기존의 Visual Document Understanding(VDU) 방법은 보통 OCR 엔진으로 먼저 글자를 읽고, 그 OCR 결과를 다시 문서 분류·문서 정보 추출·문서 시각 질의응답 같은 다운스트림 모델에 넘긴다. 이 방식은 이미 강력한 기준선이지만, OCR 자체의 계산 비용이 높고, 언어나 도메인이 바뀌면 일반화가 흔들리며, OCR 오류가 뒤 단계로 그대로 전파된다는 문제가 있다.

저자들은 이 문제를 해결하기 위해 Donut(Document Understanding Transformer)라는 OCR-free 모델을 제안한다. Donut은 Transformer 기반의 시각 인코더와 텍스트 디코더만으로 구성되며, 문서 이미지를 입력받아 원하는 구조화 출력(JSON 등)을 직접 생성한다. 학습은 먼저 문서 이미지에서 텍스트를 읽도록 하는 pre-training 단계와, 이후 특정 다운스트림 태스크에 맞게 구조화 출력을 생성하도록 하는 fine-tuning 단계로 나뉜다.

논문은 Donut이 개념적으로 단순함에도 불구하고, 다양한 문서 이해 과제에서 속도와 정확도 양쪽 모두에서 강력한 결과를 낸다고 주장한다. 또한 실제 대규모 문서 이미지가 부족한 언어와 도메인에 대비하기 위해 SynthDoG라는 합성 문서 생성기를 함께 제안하며, 이를 통해 영어뿐 아니라 중국어·일본어·한국어를 포함한 다국어 사전학습이 가능함을 보인다.

## Keywords

Visual Document Understanding, Document Information Extraction, Optical Character Recognition, End-to-End Transformer

## 1. Introduction

문서 이미지는 송장, 영수증, 명함, 양식 문서처럼 실제 업무 환경에서 매우 흔하게 등장한다. 이런 이미지를 자동으로 처리하려면 단순히 픽셀 패턴을 보는 것만으로는 충분하지 않다. 문서 안의 텍스트를 읽어야 하고, 각 텍스트 조각이 어떤 의미를 가지는지 파악해야 하며, 서로 어떤 구조와 관계를 이루는지도 이해해야 한다. 그래서 문서 분류, 정보 추출, 문서 VQA는 모두 시각 정보와 언어 정보를 동시에 다루는 문제로 이해된다.

기존 VDU 시스템은 대체로 두 단계로 나뉜다. 먼저 OCR이 문서 안의 글자를 읽고 위치 정보를 추출한다. 그다음 다운스트림 모델이 이 텍스트와 좌표를 받아 문서를 이해한다. 문서 파싱을 예로 들면, 텍스트 탐지, 텍스트 인식, 그리고 파싱이라는 세 모듈이 순차적으로 이어진다. 이 파이프라인은 직관적이지만, 앞단의 OCR이 시스템 전체의 병목이 되기 쉽다.

Figure 1은 전통적인 문서 정보 추출 파이프라인을 보여 준다. 원시 문서 이미지에서 먼저 텍스트 영역을 검출하고, 각 영역을 문자 인식기로 보내 문자열을 얻은 뒤, 마지막으로 이 문자열과 위치 정보를 이용해 구조화된 정보를 복원하는 흐름이다. 즉 문서 이해의 핵심 입력이 실제 이미지가 아니라 OCR 산출물이라는 점이 기존 접근의 특징이다.

【Figure 1 삽입 위치 - OCR_free_Document_Understanding_Transformer_Figure_01.png】

저자들은 OCR 의존 방식에 세 가지 큰 한계가 있다고 본다. 첫째, 고품질 OCR을 쓰려면 추론 비용이 크다. 둘째, 오프더셸프 OCR 엔진은 언어 변화나 도메인 변화에 유연하지 않다. 셋째, OCR이 한 글자라도 잘못 읽으면 그 오류가 문서 이해 단계 전체를 흔든다. 특히 한자권 언어처럼 문자 집합이 복잡한 환경에서는 이 문제가 더 심각해질 수 있다. 그래서 실제 시스템에서는 후처리용 OCR 보정 모듈까지 추가되는 경우가 많지만, 이는 시스템 복잡성과 유지비를 더 올린다.

이 논문은 이 전통적 흐름을 뒤집는다. 목표는 문서 이미지를 바로 원하는 출력으로 매핑하는 것이다. OCR 결과를 중간 표현으로 두지 않고, 이미지에서 구조화 정보로 직접 가는 end-to-end 경로를 구성함으로써 OCR 비용, OCR 오류 전파, OCR 엔진 교체 비용을 동시에 줄이려는 것이다. Donut은 이런 목표를 Transformer-only 구조로 구현한다.

Figure 2는 제안 방식의 전체 개요와 시스템 수준 비교를 보여 준다. 기존 방식은 이미지 → OCR → 다운스트림 모델의 경로를 거치지만, Donut은 이미지 → E2E 모델 → 구조화 출력이라는 더 짧은 경로를 사용한다. 이 그림에서 중요한 메시지는 세 가지다. 메모리 측면에서 Donut은 OCR 모델 파라미터를 따로 안고 가지 않는다. 속도 측면에서도 전통적인 OCR 기반 파이프라인보다 빠르다. 그리고 문서 IE 정확도 역시 더 높다. 논문은 이 그림을 통해 “단순한 구조가 더 싸고 더 빠르면서도 더 정확할 수 있다”는 주장을 서두에서 분명히 제시한다.

【Figure 2 삽입 위치 - OCR_free_Document_Understanding_Transformer_Figure_02.png】

Donut의 학습 방식은 pre-train-and-fine-tune 체계다. 먼저 pre-training 단계에서는 문서 이미지 안의 텍스트를 읽는 법을 배운다. 이때 학습 목표는 이미지와 이전 텍스트 문맥을 함께 보고 다음 토큰을 맞히는 것이다. 이후 fine-tuning 단계에서는 특정 과제에 맞게 문서 전체를 이해하도록 학습한다. 분류 문제라면 클래스 토큰을, 정보 추출 문제라면 JSON 구조를, 문서 VQA라면 질문에 대한 답변 문자열을 생성한다.

논문의 기여는 네 가지로 요약된다. 첫째, OCR 없이 end-to-end로 학습되는 Transformer 기반 VDU를 제안한다. 둘째, 텍스트 읽기라는 단순한 사전학습 목표와 합성 문서 생성기 SynthDoG를 결합해 다국어·다도메인 확장을 가능하게 한다. 셋째, 공개 벤치마크와 실제 서비스 데이터 모두에서 높은 정확도와 비용 효율성을 보인다. 넷째, 코드, 사전학습 모델, 합성 데이터 생성기를 함께 공개한다.

## 2. Method

### 2.1 Preliminary: background

문서 이해 연구는 오래전부터 진행되어 왔지만, 초기 접근은 주로 이미지 분류 백본에 의존하는 OCR-independent 시각 모델이 중심이었다. 이런 방법은 문서의 전반적 유형을 구별하는 데는 어느 정도 효과가 있었지만, 실제로 텍스트를 읽고 구조를 복원해야 하는 정보 추출 문제에서는 한계가 컸다. 이후 OCR과 BERT 계열 모델이 발전하면서, OCR 결과를 텍스트 시퀀스로 직렬화한 뒤 언어모델에 넣는 방식이 주류가 되었다.

최근의 강력한 VDU 시스템은 OCR 엔진과 대규모 실제 문서 이미지 데이터 둘 다에 크게 기대고 있다. 즉 좋은 OCR을 확보해야 하고, 그 OCR 결과를 잘 활용할 수 있는 문서 이해 백본도 필요하며, 다시 그 전체 체계를 사전학습할 만큼 큰 문서 코퍼스도 요구된다. 저자들은 이 점이 실제 산업 환경에서 매우 비싸고 관리하기 어려운 선택이라고 본다.

### 2.2 Document Understanding Transformer

Donut은 문서 이미지를 직접 이해하기 위한 self-contained 모델이다. 구조는 단순하다. 앞단에는 시각 인코더가 있어 이미지를 잠재 임베딩으로 바꾸고, 뒷단에는 텍스트 디코더가 있어 그 임베딩을 토큰 시퀀스로 풀어낸다. 그리고 이 토큰 시퀀스는 최종적으로 JSON 같은 구조화 표현으로 되돌릴 수 있다.

Figure 3은 Donut의 전반적 동작 방식을 보여 준다. 같은 모델이 입력 이미지와 프롬프트에 따라 문서 분류, 문서 파싱, 문서 VQA를 모두 수행할 수 있다는 점이 핵심이다. 즉 출력 헤드가 태스크마다 따로 달린 것이 아니라, 디코더가 생성해야 할 출력 형식만 달라진다. 이 덕분에 분류·추출·질의응답을 하나의 생성 문제로 통일할 수 있다.

【Figure 3 삽입 위치 - OCR_free_Document_Understanding_Transformer_Figure_03.png】

#### 2.2.1 Encoder

시각 인코더는 입력 문서 이미지 \(x \in \mathbb{R}^{H \times W \times C}\)를 임베딩 집합으로 바꾼다.

$$
\{ z_i \mid z_i \in \mathbb{R}^{d}, \; 1 \le i \le n \}
$$

여기서 \(n\)은 특징 맵 크기, 즉 패치 수에 해당하고, \(d\)는 인코더 잠재 벡터 차원이다. 논문은 CNN 기반 인코더도 가능하다고 설명하지만, 실제 실험에서는 Swin Transformer를 사용한다. 이유는 문서 파싱 예비 실험에서 가장 좋은 성능을 보였기 때문이다.

Swin Transformer는 먼저 이미지를 겹치지 않는 패치로 나눈다. 이후 shifted window 기반 multi-head self-attention과 2층 MLP가 반복 적용되고, 단계마다 patch merging이 이루어진다. 이렇게 얻어진 마지막 블록의 출력 \(\{z\}\)가 디코더 입력으로 전달된다. 문서 이미지는 일반 자연 이미지보다 글자 밀도와 공간 구조가 중요하므로, 지역적 윈도우와 계층적 병합을 함께 쓰는 Swin이 잘 맞는다는 것이 저자들의 판단이다.

#### 2.2.2 Decoder

디코더는 인코더 출력 \(\{z\}\)를 받아 길이 \(m\)의 토큰 시퀀스 \((y_i)_{i=1}^{m}\)를 생성한다.

$$
y_i \in \mathbb{R}^{v}
$$

여기서 \(y_i\)는 \(i\)번째 토큰의 one-hot 벡터이고, \(v\)는 어휘 크기다. 논문은 디코더 구조로 BART를 사용하며, 공개된 다국어 BART 가중치로 초기화한다. 이 선택은 문서 이미지를 직접 문장 또는 구조화 토큰열로 바꾸는 생성 문제에 자연스럽다. 분류라면 짧은 클래스 시퀀스를, 정보 추출이라면 길고 구조적인 JSON 시퀀스를, VQA라면 질문에 대한 답을 동일한 생성 방식으로 처리할 수 있기 때문이다.

#### 2.2.3 Model Input

학습 시에는 teacher forcing을 사용한다. 즉 디코더가 이전 시점의 모델 예측을 입력으로 받는 것이 아니라, 정답 토큰을 입력으로 받는다. 따라서 학습 중에는 전체 정답 시퀀스를 한 번에 안정적으로 지도할 수 있다. 반면 추론 시에는 GPT 계열 생성과 비슷하게 프롬프트를 주고, 그다음 토큰부터 자동회귀적으로 생성한다.

태스크별 프롬프트는 아주 중요하다. 문서 분류에서는 `<classification>` 같은 시작 신호가 들어가고, 파싱에서는 `<parsing>`, 문서 VQA에서는 `<vqa><question> ... </question><answer>` 같은 형태가 들어간다. 이 프롬프트가 디코더에게 “이번에는 어떤 형식의 출력을 생성해야 하는가”를 알려 주는 셈이다.

#### 2.2.4 Output Conversion

Donut의 출력은 단순한 자유 텍스트가 아니라 구조화 표현으로 복원될 수 있어야 한다. 논문은 JSON을 채택한다. 이유는 표현력이 높고, 문서 정보 추출에서 흔히 필요한 계층 구조·리스트·중첩된 필드를 자연스럽게 담을 수 있기 때문이다.

구현 방식은 단순하다. 각 필드를 `[START_*]`와 `[END_*]` 사이에 두어 토큰열을 JSON으로 되돌린다. 예를 들어 이름 필드라면 `[START_name] ... [END_name]`처럼 감싼다. 만약 구조가 잘못 생성되어 시작 토큰만 있고 끝 토큰이 없다면, 해당 필드는 추출 실패로 간주한다. 즉 Donut은 생성 모델이지만, 출력 파싱 규칙은 비교적 엄격하다.

### 2.3 Pre-training

#### 2.3.1 Task

사전학습의 목표는 문서 이미지 속 텍스트를 읽는 것이다. 보다 정확히 말하면, 이미지와 이전 텍스트 문맥을 함께 조건으로 두고 다음 토큰을 예측하는 시각 언어 모델링을 수행한다. 저자들은 이를 pseudo-OCR task로 볼 수 있다고 설명한다. 별도의 OCR 모듈을 두는 대신, 모델이 이미지로부터 직접 문자열을 생성하게 만들기 때문이다.

이 단계의 핵심은 문서 전체를 왼쪽 위에서 오른쪽 아래로 읽는 순서로 정렬된 토큰열을 맞히도록 만드는 것이다. 즉 Donut은 먼저 “문서를 읽는 법”을 배우고, 그 뒤에 “문서를 이해하는 법”을 배운다. 이 순서가 fine-tuning 안정성과 일반화에 중요하다는 것이 논문의 관점이다.

#### 2.3.2 Visual Corpora

논문은 실제 문서 이미지 코퍼스로 IIT-CDIP를 사용한다. 이는 약 11M장의 영어 스캔 문서 이미지로 구성된다. 여기에는 상용 CLOVA OCR API를 적용해 pseudo text label을 얻는다. 다만 저자들은 이 방식이 영어처럼 대규모 실문서 코퍼스가 있는 경우에만 쉽다고 지적한다. 다른 언어나 산업 도메인에서는 같은 규모의 데이터와 OCR 라벨을 마련하기 어렵다.

그래서 논문은 SynthDoG라는 합성 문서 생성기를 함께 제안한다. 영어·중국어·일본어·한국어 Wikipedia를 기반으로 언어별 0.5M 샘플씩, 총 2M개의 다국어 합성 문서를 만든다. 이 합성 데이터는 Donut이 언어와 레이아웃 다양성에 강해지도록 만드는 핵심 재료다.

Figure 4는 SynthDoG가 생성한 영어·중국어·일본어·한국어 예시를 보여 준다. 단순히 텍스트만 합성한 것이 아니라, 배경, 종이 질감, 레이아웃, 왜곡, 노이즈를 함께 넣어 실제 문서 사진처럼 보이도록 만든 것이 특징이다. 이 그림은 Donut의 다국어 사전학습이 왜 가능한지를 매우 직관적으로 보여 준다.

【Figure 4 삽입 위치 - OCR_free_Document_Understanding_Transformer_Figure_04.png】

#### 2.3.3 Synthetic Document Generator

SynthDoG의 생성 파이프라인은 배경, 문서 질감, 텍스트 내용, 레이아웃, 후처리의 다섯 층으로 이해할 수 있다. 배경 이미지는 ImageNet에서 샘플링하고, 문서 표면 질감은 종이 사진에서 샘플링한다. 텍스트 내용은 Wikipedia에서 가져오며, 레이아웃은 격자를 무작위로 쌓는 간단한 규칙 기반 알고리즘으로 생성한다.

여기에 다양한 렌더링 기법이 더해진다. 색상 조정, 밝기·대비 변화, 그림자, 움직임 블러, 가우시안 블러, JPEG 압축 같은 후처리를 적용해 실제 촬영 문서처럼 보이게 만든다. 핵심은 완벽하게 현실적인 문서를 만드는 것이 아니라, 사전학습에 필요한 시각적 변동성을 충분히 제공하는 것이다.

### 2.4 Fine-tuning

사전학습이 “문서를 읽는 법”을 배우는 단계라면, fine-tuning은 “문서를 이해하는 법”을 배우는 단계다. 논문은 모든 다운스트림 태스크를 JSON 예측 문제로 통일한다. 이 설계는 매우 중요하다. 분류, 정보 추출, 질의응답을 각각 다른 출력 헤드로 처리하는 대신, 모두 “주어진 프롬프트에 맞는 토큰 시퀀스를 생성하라”는 하나의 문제로 바꾼다.

예를 들어 문서 분류에서는 `[START_class][memo][END_class]` 같은 토큰열을 생성하고, 이것을 `{"class": "memo"}`라는 JSON으로 바꾼다. 문서 파싱에서는 필드와 중첩 구조를 포함한 더 긴 JSON을 생성하고, 문서 VQA에서는 질문-답변 쌍을 생성한다. 이렇게 하면 모델 구조를 태스크마다 바꾸지 않고도 다양한 문서 이해 문제를 처리할 수 있다.

## 3. Experiments and Analyses

이 절에서는 Donut을 세 가지 응용 과제, 즉 문서 분류, 문서 정보 추출, 문서 시각 질의응답에 대해 평가한다. 데이터셋은 공개 벤치마크와 실제 서비스용 비공개 산업 데이터셋을 모두 포함한다. 논문의 중요한 장점은 “연구용 벤치마크에서만 잘 되는 모델”이 아니라 “실제 서비스 데이터에서도 유의미하게 작동하는 모델”을 보여 준다는 데 있다.

Figure 5는 논문이 사용하는 다운스트림 데이터셋의 예시를 보여 준다. 분류용 문서, 영수증·티켓·명함 같은 IE 문서, 그리고 질문과 함께 주어지는 문서 VQA 샘플을 한눈에 볼 수 있다. Donut이 다루는 입력 분포가 매우 넓다는 점을 이 그림이 잘 보여 준다.

【Figure 5 삽입 위치 - OCR_free_Document_Understanding_Transformer_Figure_05.png】

### 3.1 Downstream Tasks and Datasets

#### 3.1.1 Document Classification

문서 분류는 입력 문서가 어떤 유형인지 판별하는 문제다. 많은 기존 모델이 인코더 출력 위에 softmax 분류기를 얹는 방식을 쓰지만, Donut은 여기서도 JSON 생성이라는 동일한 인터페이스를 유지한다. 즉 클래스 정보를 담은 짧은 시퀀스를 생성한다. 평가는 테스트셋 전체 accuracy로 수행한다.

**RVL-CDIP.** RVL-CDIP는 16개 클래스, 총 400K 이미지로 구성된 대규모 문서 분류 데이터셋이다. letter, memo, email 등 문서 유형이 고르게 포함되어 있다. 학습 320K, 검증 40K, 테스트 40K로 나뉜다.

#### 3.1.2 Document Information Extraction

문서 정보 추출은 문서 이미지에서 구조화된 정보를 복원하는 문제다. 단순히 글자를 읽는 것만으로는 충분하지 않고, 어떤 문자열이 어떤 필드에 속하는지, 여러 필드가 어떤 계층 구조를 이루는지까지 이해해야 한다. 영수증에서는 품목명·수량·가격이 item 단위로 묶여야 하고, 합계 정보는 전체 total 구조로 묶여야 한다. 이 때문에 문서 IE는 문서 이해 능력을 가장 직접적으로 시험하는 과제라고 할 수 있다.

논문은 두 가지 평가지표를 사용한다. 첫 번째는 field-level F1이다. 이는 추출된 필드 값이 정답과 완전히 일치하는지를 본다. 한 글자라도 틀리면 그 필드 전체가 실패로 처리되므로, 엄격한 지표다. 두 번째는 Tree Edit Distance(TED) 기반 accuracy다. 문서 정보를 트리 구조로 보고, 예측 트리와 정답 트리 사이의 편집 거리를 이용해 전체 구조까지 평가한다. 논문이 쓰는 식은 다음과 같다.

$$
\max \left( 0,\; 1 - \frac{TED(pr, gt)}{TED(\varnothing, gt)} \right)
$$

여기서 \(gt\)는 정답 트리, \(pr\)는 예측 트리, \(\varnothing\)는 빈 트리다. 이 지표는 단순 문자열 일치가 아니라, 그룹 구조와 중첩 관계까지 반영한다는 점에서 문서 IE에 특히 중요하다.

논문은 네 개의 IE 데이터셋을 쓴다.

**CORD.** CORD는 0.8K train, 0.1K valid, 0.1K test로 구성된 영수증 데이터셋이다. 30개의 고유 필드가 있으며, items > item > {name, count, price}처럼 중첩 구조가 존재한다.

**Ticket.** 중국어 기차표 데이터셋으로, 1.5K train과 0.4K test가 있다. 논문은 train의 10%를 validation으로 분리한다. 구조는 단순하고, 각 키는 한 번만 나오며 위치도 거의 고정된다.

**Business Card (In-Service Data).** 실제 서비스에서 수집된 일본어 명함 데이터다. 20K train, 0.3K valid, 0.3K test로 이루어지며, 이름·회사·주소 등 11개 필드가 있다. 구조는 Ticket과 유사하게 비교적 단순하다.

**Receipt (In-Service Data).** 실제 서비스에서 사용되는 한국어 영수증 데이터다. 40K train, 1K valid, 1K test이고, 고유 필드가 81개로 많다. 상점 정보, 결제 정보, 가격 정보 등이 복합적으로 들어가며 구조도 복잡하다.

#### 3.1.3 Document Visual Question Answering

문서 VQA는 문서 이미지와 질문이 함께 주어졌을 때, 이미지 내부의 시각·텍스트 정보를 이용해 답변을 생성하는 문제다. Donut은 여기서도 별도 구조를 쓰지 않고, 질문을 시작 프롬프트로 주고 답변을 생성한다. 이렇게 하면 분류와 정보 추출과 마찬가지로 하나의 생성 인터페이스를 유지할 수 있다.

**DocVQA.** 이 데이터셋은 12K개 이상의 문서에 대해 50K개의 질문이 정의되어 있다. 40K train, 5K valid, 5K test로 구성되며, 평가는 ANLS(Average Normalized Levenshtein Similarity)로 한다. 이는 편집거리 기반 유사도 점수라서, 완전히 같은 문자열이 아니어도 부분적으로 유사하면 점수를 일부 받을 수 있다.

### 3.2 Setups

Donut의 시각 인코더는 Swin-B를 약간 수정한 버전이다. 레이어 수는 \(\{2, 2, 14, 2\}\), window size는 10으로 설정한다. 디코더는 BART 전체를 쓰지 않고 처음 네 레이어만 사용한다. 이는 속도와 정확도의 균형을 고려한 선택이다.

사전학습에는 2M 합성 문서와 11M IIT-CDIP 문서 이미지를 함께 쓴다. 학습은 64개의 A100 GPU, mini-batch 196으로 200K step 동안 수행한다. 최적화는 Adam을 사용하고, learning rate는 스케줄링하며 초기값은 \(1e{-5}\)에서 \(1e{-4}\) 사이에서 선택한다. 기본 입력 해상도는 \(2560 \times 1920\), 디코더 최대 길이는 1536이다. Ticket과 Business Card 파싱에는 \(960 \times 1280\)을 사용한다.

속도 측정은 P40 GPU에서 수행된다. 이는 A100보다 훨씬 느리므로, 실제 표의 absolute time은 보수적으로 봐야 한다. OCR 기반 기준선에는 MS OCR API와 CLOVA OCR API, 그리고 일부 실험에서 Paddle OCR 등이 사용된다. OCR 선택에 따라 전체 파이프라인 성능이 크게 달라질 수 있다는 점을 저자들은 후속 분석에서 별도로 다룬다.

Table 1은 RVL-CDIP 문서 분류 결과를 정리한다. Donut은 OCR 없이 동작하는데도 95.30% accuracy를 기록한다. 이는 LayoutLMv2의 95.25%를 약간 넘는 수치다. 더 중요한 점은 속도다. Donut은 752ms인데 LayoutLMv2는 1489ms다. 즉 정확도는 비슷하거나 약간 더 높고, 추론 시간은 거의 절반 수준이다. 또한 LayoutLM 계열은 표에 적힌 파라미터 외에 OCR 엔진 자체의 파라미터와 비용이 따로 필요하지만, Donut은 그렇지 않다.

【Table 1 삽입 위치 - OCR_free_Document_Understanding_Transformer_Table_01.png】

### 3.3 Experimental Results

#### 3.3.1 Document Classification

문서 분류 실험의 핵심 메시지는 단순하다. Donut은 일반 목적 문서 이해 백본들과 비교해도 정확도가 뒤지지 않으며, OCR에 대한 외부 의존성이 없다는 구조적 장점을 가진다. LayoutLMv2가 95.25%인 반면 Donut은 95.30%이고, 속도는 더 빠르다. 이 결과는 “OCR이 꼭 있어야 문서 분류가 잘 된다”는 관념을 약화시킨다.

저자들이 강조하는 또 하나의 지점은 전체 시스템 비용이다. OCR 기반 모델은 논문 표에서 언어모델·백본의 파라미터만 비교하면 비슷해 보일 수 있지만, 실제 서비스에서는 OCR 모델 또는 OCR API 비용이 추가된다. Donut은 이 숨은 비용을 줄여 주기 때문에 정확도 이상의 실제 장점이 있다.

Table 2는 네 가지 문서 IE 과제의 결과를 보여 준다. 논문은 각 데이터셋에 대해 field-level F1과 TED 기반 accuracy를 함께 제시한다. Donut은 네 도메인 모두에서 최고 성능을 기록한다. CORD에서는 F1 84.1, TED accuracy 90.9를 기록해 LayoutLMv2의 78.9 / 82.4를 앞선다. Ticket에서는 94.1 / 98.7로 매우 높고, Business Card에서도 57.8 / 84.4로 LayoutLMv2의 52.2 / 83.0보다 좋다. 가장 복잡한 Receipt 데이터셋에서도 Donut은 78.6 / 88.6으로 기준선들을 크게 앞선다.

【Table 2 삽입 위치 - OCR_free_Document_Understanding_Transformer_Table_02.png】

#### 3.3.2 Document Information Extraction

이 실험에서 비교군은 크게 세 부류다. 첫째는 OCR 결과를 직렬화한 뒤 BIO tagging을 수행하는 전통적 IE 접근이다. 여기에는 BERT, BROS, LayoutLM, LayoutLMv2가 포함된다. 둘째는 관계 예측 기반의 SPADE다. 셋째는 OCR 출력으로부터 구조를 직접 생성하는 encoder-decoder 계열인 WYVERN이다. Donut은 이들 모두와 달리 원시 이미지에서 바로 구조화 결과를 생성한다.

결과를 보면 Donut은 단순히 텍스트만 더 잘 읽는 것이 아니라, 구조까지 더 잘 맞힌다. CORD처럼 중첩 구조가 있는 데이터셋에서 TED accuracy 차이가 큰 것이 특히 중요하다. 이는 Donut이 각 메뉴 항목을 그룹으로 묶고, 그 안에서 이름·수량·가격을 연결하는 계층적 구조까지 더 잘 복원했다는 뜻이다.

저자들은 해상도와 데이터 규모의 관계도 언급한다. 예를 들어 CORD에서 \(1280 \times 960\) 해상도를 쓰면 0.7초/이미지, 91.1 accuracy를 얻는다. 즉 더 높은 해상도는 일반적으로 성능에 도움이 되지만, 추론 비용도 늘린다. 그럼에도 불구하고 Donut은 데이터셋 크기와 구조 복잡성이 달라져도 상대적으로 안정적인 성능을 보인다.

#### 3.3.3 Document Visual Question Answering

문서 VQA 결과는 조금 더 미묘하다. 전체 test set ANLS만 보면 Donut은 67.5로, LayoutLMv2의 78.1이나 LayoutLMv2-Large-QG의 86.7보다는 낮다. 따라서 전체 점수만 놓고 보면 OCR 기반의 강력한 대형 기준선이 여전히 우세하다. 하지만 논문은 여기서 끝나지 않고, 손글씨 문서에 대한 robustness를 따로 본다.

Table 3을 보면 Donut의 handwritten ANLS는 72.1이다. 이는 LayoutLMv2-Large-QG의 67.3보다 높다. 즉 손글씨처럼 OCR이 흔들리기 쉬운 영역에서는 OCR-free 방식이 오히려 강점을 보인다. 논문의 주장은 “Donut이 모든 DocVQA 설정에서 절대 최고다”가 아니라, “OCR에 묶이지 않는 구조 덕분에 특정 어려운 상황에서는 더 강하다”는 것이다.

【Table 3 삽입 위치 - OCR_free_Document_Understanding_Transformer_Table_03.png】

Figure 6은 DocVQA 예시를 보여 준다. 왼쪽과 가운데 예시는 OCR 오류 때문에 LayoutLMv2 계열이 상한선을 갖는 경우를 보여 준다. 반면 Donut은 이미지로부터 직접 답을 생성하므로 OCR 오타에 덜 묶인다. 오른쪽 예시는 반대로 Donut의 한계를 보여 준다. 입력 해상도 제약 때문에 큰 이미지 안의 매우 작은 글자를 놓칠 수 있다는 것이다. 즉 Donut의 약점은 “OCR이 없어서 못 읽는 것”이 아니라 “end-to-end 입력 해상도가 제한돼 있어 작은 글자를 놓칠 수 있는 것”이라고 해석할 수 있다.

【Figure 6 삽입 위치 - OCR_free_Document_Understanding_Transformer_Figure_06.png】

### 3.4 Further Studies

저자들은 Donut이 왜 잘 작동하는지 더 이해하기 위해 추가 분석을 수행한다. 여기서는 사전학습 전략, 이미지 백본, 입력 해상도, 텍스트 위치화, OCR 시스템의 영향, 저자원 학습 상황을 차례로 살핀다.

Figure 7은 세 가지 분석을 한 번에 보여 준다. (a)는 사전학습 전략 비교, (b)는 백본 비교, (c)는 해상도 비교다. 이 그림의 역할은 단순한 보조 자료가 아니라, Donut의 설계 선택이 임의적이지 않다는 점을 보여 주는 것이다.

【Figure 7 삽입 위치 - OCR_free_Document_Understanding_Transformer_Figure_07.png】

#### 3.4.1 On Pre-training Strategy

사전학습 전략 비교에서 저자들은 이미지 캡셔닝 같은 일반적인 비전-언어 사전학습보다, 문서 텍스트 읽기 자체를 목표로 하는 Donut식 사전학습이 더 효과적이라고 결론내린다. 특히 문서 IE 과제에서는 SynthDoG만으로도 꽤 강한 성능을 얻을 수 있었다. 이는 합성 문서가 실제 문서 구조를 배우는 데 충분한 신호를 준다는 뜻이다.

다만 DocVQA에서는 IIT-CDIP 같은 실제 문서 이미지가 중요했다. DocVQA의 이미지 분포가 실제 스캔 문서와 더 닮아 있기 때문에, 현실적인 시각 통계가 도움이 된 것으로 해석된다. 즉 합성 데이터만으로 충분한 영역과, 실제 이미지가 반드시 필요한 영역이 다르다는 점이 이 분석에서 드러난다.

#### 3.4.2 On Encoder Backbone

백본 비교에서는 ResNet-152, EfficientNetV2, ViT-B, Swin-B 등을 비교한다. 결과적으로 EfficientNetV2와 Swin Transformer가 가장 좋다. 저자들은 이를 높은 표현력 덕분으로 해석한다. 문서 이해는 작은 문자 패턴과 전역 레이아웃 정보를 동시에 다뤄야 하므로, 강한 시각 백본이 중요하다.

그중 Swin Transformer를 최종 선택한 이유는 Transformer 기반이라 확장성이 높고, 실험상 EfficientNetV2보다 조금 더 나은 성능을 보였기 때문이다. 즉 Donut의 OCR-free 성능은 디코더만의 공이 아니라, 문서 이미지 표현을 잘 만드는 시각 인코더 선택에도 크게 의존한다.

#### 3.4.3 On Input Resolution

해상도 분석에서는 입력 크기가 커질수록 성능이 빠르게 상승함을 보인다. 특히 작은 글자가 많은 DocVQA에서 이 효과가 두드러진다. 하지만 해상도를 키우면 계산량이 급격히 늘어난다. 저자들은 보다 효율적인 attention 메커니즘을 쓰면 이 문제를 완화할 수 있겠지만, 이 논문은 의도적으로 더 단순한 Transformer 구조를 유지했다고 설명한다.

즉 Donut의 약점 중 하나는 해상도에 따른 비용 증가다. 그러나 이것은 OCR-free라는 철학의 실패라기보다, 현재 사용한 단순한 end-to-end Transformer 구조의 계산 한계로 보는 것이 더 정확하다.

#### 3.4.4 On Text Localization

Donut은 텍스트 위치 supervision 없이 학습되지만, 디코더의 cross-attention을 시각화해 보면 실제로 의미 있는 위치를 바라본다. 저자들은 이를 통해 Donut이 단순한 문자열 생성기가 아니라, 문서 안에서 필요한 영역을 찾아가며 읽는 모델이라는 점을 보여 준다.

Figure 8은 특정 토큰을 생성할 때 디코더가 어떤 문자 영역에 주의를 두는지를 시각화한다. 주목할 점은 별도의 detector 없이도 올바른 텍스트 주변으로 attention이 모인다는 것이다. 이 결과는 Donut이 OCR 모듈을 제거했음에도, 내부적으로는 문서 내 텍스트 위치와 내용의 대응을 어느 정도 학습하고 있음을 시사한다.

【Figure 8 삽입 위치 - OCR_free_Document_Understanding_Transformer_Figure_08.png】

#### 3.4.5 On OCR System

OCR 기반 기준선은 어떤 OCR 엔진을 쓰느냐에 따라 성능과 속도가 크게 달라질 수 있다. 이는 레이아웃 이해 모델 자체의 품질뿐 아니라, 앞단 OCR 품질이 전체 시스템 결과를 결정한다는 뜻이다. 저자들은 이것이 OCR-dependent 접근의 구조적 약점이라고 본다.

Figure 9의 왼쪽과 가운데 그래프는 LayoutLMv2와 BERT 계열 성능이 OCR 엔진 선택에 따라 크게 출렁이는 것을 보여 준다. EasyOCR, PaddleOCR, MS OCR, CLOVA OCR이 각각 다른 정확도와 속도를 만든다. 반면 Donut은 OCR 엔진을 아예 쓰지 않기 때문에 이런 불확실성에서 자유롭다.

#### 3.4.6 On Low Resourced Situation

저자들은 CORD 학습 데이터를 줄여 가며 저자원 상황에서의 성능을 비교한다. 그 결과 Donut은 데이터가 아주 적어져도 성능이 비교적 안정적으로 유지된다. 특히 80개 샘플만 있는 극단적 저자원 상황에서도 강한 성능을 보인다. 더 큰 해상도 \(2560 \times 1920\)는 이런 저자원 상황에서 특히 도움이 된다.

Figure 9의 오른쪽 그래프는 샘플 수가 적을 때 Donut이 더 견고하다는 점을 보여 준다. 저자들은 Donut이 LayoutLMv2 정확도를 10% 데이터, 즉 약 80개 샘플만으로도 넘어섰다고 보고한다. 이는 OCR 없는 end-to-end 생성 방식이 적은 데이터에서도 비교적 강한 bias를 제공할 수 있음을 시사한다.

【Figure 9 삽입 위치 - OCR_free_Document_Understanding_Transformer_Figure_09.png】

## 4. Related Work

### 4.1 Optical Character Recognition

OCR은 일반적으로 두 단계로 이루어진다. 먼저 detector가 텍스트 영역을 찾고, 이후 recognizer가 잘라낸 텍스트 인스턴스에서 문자를 읽는다. 최근의 OCR 연구는 두 단계 모두에 딥러닝을 도입해 큰 발전을 이루었다. 합성 데이터와 실제 데이터 모두가 학습에 활용되며, detector와 recognizer 모두 높은 정확도를 보여 왔다.

텍스트 탐지 쪽에서는 초기 CNN 기반 지역 세그먼트 병합 방식에서 시작해, 이후 region proposal과 bounding box regression, 그리고 문자 성분 수준(component-level) 접근으로 발전했다. 텍스트 인식 쪽도 CNN 기반 특징 추출 뒤에 순차적 또는 attention 기반 디코더를 붙이는 구조가 주류가 되었다. 하지만 Donut의 핵심 주장은 이런 OCR 기술이 쓸모없다는 것이 아니라, 문서 이해 전체 시스템에서는 OCR을 별도 모듈로 떼어 두는 선택이 언제나 최선은 아니라는 것이다.

### 4.2 Visual Document Understanding

문서 분류는 오랫동안 일반 이미지 분류 문제처럼 다뤄졌다. 이후 BERT가 등장하면서 OCR로 추출한 텍스트를 시퀀스로 만들고, 여기에 시각 특징을 더한 뒤 언어모델에 넣는 방식이 급속히 확산됐다. LayoutLM, LayoutLMv2, StructuralLM, DocFormer 등이 이 흐름을 대표한다.

문서 IE도 실제 산업 응용이 매우 넓다. 영수증 디지털화, 송장 처리, 명함 인식 등은 모두 OCR 이후 구조 복원을 수행하는 전통적 파이프라인을 써 왔다. 최근 SPADE와 WYVERN 같은 모델이 parsing 단계를 단순화했지만, 여전히 OCR 출력에 의존한다. 문서 VQA 또한 대부분 OCR 후 BERT 계열 Transformer를 적용하는 파이프라인을 따른다. 다만 이런 추출형 접근은 답이 이미지 안에 문자 그대로 나타나지 않는 질문에서는 한계가 있다. 생성형 접근이 제안된 이유도 바로 여기에 있다.

## 5. Conclusions

이 논문은 문서 이미지를 원하는 구조화 출력으로 직접 매핑하는 OCR-free end-to-end 프레임워크 Donut을 제안한다. Donut은 OCR 없이도 문서를 읽고, 이해하고, 구조화 결과를 생성할 수 있음을 보여 준다. 핵심은 Transformer 기반 시각 인코더와 텍스트 디코더, 그리고 텍스트 읽기 중심의 사전학습 목표를 결합했다는 점이다.

또한 SynthDoG라는 합성 문서 생성기를 통해 대규모 실제 문서 코퍼스 의존성을 완화하고, 다국어 확장 가능성을 보여 준다. 공개 벤치마크와 실제 산업 데이터셋 모두에서 Donut은 높은 정확도와 비용 효율성을 보인다. 저자들은 향후 더 나은 사전학습 목표를 설계하는 것이 중요한 후속 과제라고 본다.

## References

아래 참고문헌 목록은 원문의 번호 체계를 유지한 정리본이다.

- **[1]** Afzal, M.Z., Capobianco, S., Malik, M.I., Marinai, S., Breuel, T.M., Dengel, A., Liwicki, M.: Deepdocclassifier: Document classification with deep convolutional neural network. In: 2015 13th International Conference on Document Analysis and Recognition (ICDAR). pp. 1111–1115 (2015). https://doi.org/10.1109/ICDAR.2015.7333933
- **[2]** Appalaraju, S., Jasani, B., Kota, B.U., Xie, Y., Manmatha, R.: Docformer: End-to-end transformer for document understanding. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 993–1003 (October 2021)
- **[3]** Baek, J., Kim, G., Lee, J., Park, S., Han, D., Yun, S., Oh, S.J., Lee, H.: What is wrong with scene text recognition model comparisons? dataset and model analysis. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) (October 2019)
- **[4]** Baek, Y., Lee, B., Han, D., Yun, S., Lee, H.: Character region awareness for text detection. In: 2019 IEEE/CVF Conference on Computer Vision and Pattern Recog-nition (CVPR). pp. 9357–9366 (2019). https://doi.org/10.1109/CVPR.2019.00959
- **[5]** Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.D., Dhariwal, P., Nee-lakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., Amodei, D.: Language models are few-shot learners. In: Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M.F., Lin, H. (eds.) Advances in Neural Information Processing Systems. vol. 33, pp. 1877– 1901. Curran Associates, Inc. (2020), https://proceedings.neurips.cc/paper/ 2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf
- **[6]** Davis, B., Morse, B., Cohen, S., Price, B., Tensmeyer, C.: Deep visual template-free form parsing. In: 2019 International Conference on Document Analysis and Recog-nition (ICDAR). pp. 134–141 (2019). https://doi.org/10.1109/ICDAR.2019.00030
- **[7]** Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: Imagenet: A large-scale hierarchical image database. In: 2009 IEEE conference on computer vision and pattern recognition. pp. 248–255. Ieee (2009)
- **[8]** Devlin, J., Chang, M.W., Lee, K., Toutanova, K.: BERT: Pre-training of deep bidirectional transformers for language understanding. In: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers). pp. 4171–4186. Association for Computational Linguistics, Minneapolis, Minnesota (Jun 2019). https://doi.org/10.18653/v1/N19-1423, https://aclanthology.org/ N19-1423
- **[9]** Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N.: An image is worth 16x16 words: Transformers for image recognition at scale. In: 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net (2021), https://openreview.net/ forum?id=YicbFdNTTy
- **[10]** Duong, Q., H¨am¨al¨ainen, M., Hengchen, S.: An unsupervised method for OCR post-correction and spelling normalisation for Finnish. In: Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa). pp. 240–248. Link¨oping University Electronic Press, Sweden, Reykjavik, Iceland (Online) (May 31–2 Jun 2021), https://aclanthology.org/2021.nodalida-main.24
- **[11]** Friedl, J.E.F.: Mastering Regular Expressions. O’Reilly, Beijing, edn. (2006), https://www.safaribooksonline.com/library/view/ mastering-regular-expressions/0596528124/
- **[12]** Guo, H., Qin, X., Liu, J., Han, J., Liu, J., Ding, E.: Eaten: Entity-aware at-tention for single shot visual text extraction. In: 2019 International Confer-ence on Document Analysis and Recognition (ICDAR). pp. 254–259 (2019). https://doi.org/10.1109/ICDAR.2019.00049
- **[13]** Gupta, A., Vedaldi, A., Zisserman, A.: Synthetic data for text localisation in nat-ural images. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (June 2016)
- **[14]** Hammami, M., H´eroux, P., Adam, S., d’Andecy, V.P.: One-shot field spotting on colored forms using subgraph isomorphism. In: 2015 13th International Con-ference on Document Analysis and Recognition (ICDAR). pp. 586–590 (2015). https://doi.org/10.1109/ICDAR.2015.7333829
- **[15]** Harley, A.W., Ufkes, A., Derpanis, K.G.: Evaluation of deep convolutional nets for document image classification and retrieval. In: 2015 13th International Con-ference on Document Analysis and Recognition (ICDAR). pp. 991–995 (2015). https://doi.org/10.1109/ICDAR.2015.7333910
- **[16]** Harley, A.W., Ufkes, A., Derpanis, K.G.: Evaluation of deep convolutional nets for document image classification and retrieval. In: 2015 13th International Con-ference on Document Analysis and Recognition (ICDAR). pp. 991–995 (2015). https://doi.org/10.1109/ICDAR.2015.7333910
- **[17]** He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 770–778 (2016). https://doi.org/10.1109/CVPR.2016.90
- **[18]** Hong, T., Kim, D., Ji, M., Hwang, W., Nam, D., Park, S.: Bros: A pre-trained language model focusing on text and layout for better key information extrac-tion from documents. Proceedings of the AAAI Conference on Artificial Intelli-gence 36(10), 10767–10775 (Jun 2022). https://doi.org/10.1609/aaai.v36i10.21322, https://ojs.aaai.org/index.php/AAAI/article/view/21322 2, 3, 4, 7, 9, 10, 22,
- **[19]** Huang, W., Qiao, Y., Tang, X.: Robust scene text detection with convolution neural network induced mser trees. In: Fleet, D., Pajdla, T., Schiele, B., Tuytelaars, T. (eds.) Computer Vision – ECCV 2014. pp. 497–511. Springer International Publishing, Cham (2014)
- **[20]** Huang, Z., Chen, K., He, J., Bai, X., Karatzas, D., Lu, S., Jawahar, C.V.: Ic-dar2019 competition on scanned receipt ocr and information extraction. In: 2019 International Conference on Document Analysis and Recognition (ICDAR). pp. 1516–1520 (2019). https://doi.org/10.1109/ICDAR.2019.00244
- **[21]** Hwang, A., Frey, W.R., McKeown, K.: Towards augmenting lexical resources for slang and African American English. In: Proceedings of the 7th Workshop on NLP for Similar Languages, Varieties and Dialects. pp. 160–172. International Committee on Computational Linguistics (ICCL), Barcelona, Spain (Online) (Dec 2020), https://aclanthology.org/2020.vardial-1.15
- **[22]** Hwang, W., Kim, S., Yim, J., Seo, M., Park, S., Park, S., Lee, J., Lee, B., Lee, H.: Post-ocr parsing: building simple and robust parser via bio tagging. In: Workshop on Document Intelligence at NeurIPS 2019 (2019)
- **[23]** Hwang, W., Lee, H., Yim, J., Kim, G., Seo, M.: Cost-effective end-to-end infor-mation extraction for semi-structured document images. In: Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. pp. 3375– 3383. Association for Computational Linguistics, Online and Punta Cana, Do-minican Republic (Nov 2021). https://doi.org/10.18653/v1/2021.emnlp-main.271, https://aclanthology.org/2021.emnlp-main.271
- **[24]** Hwang, W., Yim, J., Park, S., Yang, S., Seo, M.: Spatial depen-dency parsing for semi-structured document information extraction. In: Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021. pp. 330–343. Association for Computational Linguistics, Online (Aug 2021). https://doi.org/10.18653/v1/2021.findings-acl.28, https://aclanthology. org/2021.findings-acl.28
- **[25]** Hwang, W., Yim, J., Park, S., Yang, S., Seo, M.: Spatial depen-dency parsing for semi-structured document information extraction. In: Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021. pp. 330–343. Association for Computational Linguistics, Online (Aug 2021). https://doi.org/10.18653/v1/2021.findings-acl.28, https://aclanthology. org/2021.findings-acl.28
- **[26]** Jaderberg, M., Simonyan, K., Vedaldi, A., Zisserman, A.: Synthetic data and ar-tificial neural networks for natural scene text recognition. In: Workshop on Deep Learning, NIPS (2014)
- **[27]** Kang, L., Kumar, J., Ye, P., Li, Y., Doermann, D.S.: Convolutional neural networks for document image classification. 2014 22nd International Conference on Pattern Recognition pp. 3168–3172 (2014)
- **[28]** Karatzas, D., Gomez-Bigorda, L., Nicolaou, A., Ghosh, S., Bagdanov, A., Iwamura, M., Matas, J., Neumann, L., Chandrasekhar, V.R., Lu, S., Shafait, F., Uchida, S., Valveny, E.: Icdar 2015 competition on robust reading. In: 2015 13th Interna-tional Conference on Document Analysis and Recognition (ICDAR). pp. 1156–1160 (2015). https://doi.org/10.1109/ICDAR.2015.7333942
- **[29]** Kim, W., Son, B., Kim, I.: Vilt: Vision-and-language transformer without con-volution or region supervision. In: Meila, M., Zhang, T. (eds.) Proceedings of the 38th International Conference on Machine Learning. Proceedings of Ma-chine Learning Research, vol. 139, pp. 5583–5594. PMLR (18–24 Jul 2021), http://proceedings.mlr.press/v139/kim21k.html
- **[30]** Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. In: Bengio, Y., LeCun, Y. (eds.) 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings (2015), http://arxiv.org/abs/1412.6980
- **[31]** Klaiman, S., Lehne, M.: Docreader: Bounding-box free training of a document information extraction model. In: Document Analysis and Recognition – IC-DAR 2021: 16th International Conference, Lausanne, Switzerland, September 5–10, 2021, Proceedings, Part I. p. 451–465. Springer-Verlag, Berlin, Heidel-berg (2021). https://doi.org/10.1007/978-3-030-86549-8 29, https://doi.org/10. 1007/978-3-030-86549-8_29
- **[32]** Lewis, D., Agam, G., Argamon, S., Frieder, O., Grossman, D., Heard, J.: Building a test collection for complex document information processing. In: Proceedings of the 29th Annual International ACM SIGIR Conference on Research and Develop-ment in Information Retrieval. p. 665–666. SIGIR ’06, Association for Computing Machinery, New York, NY, USA (2006). https://doi.org/10.1145/1148170.1148307, https://doi.org/10.1145/1148170.1148307
- **[33]** Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., Stoy-anov, V., Zettlemoyer, L.: BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. In: Proceed-ings of the 58th Annual Meeting of the Association for Computational Lin-guistics. pp. 7871–7880. Association for Computational Linguistics, Online (Jul 2020). https://doi.org/10.18653/v1/2020.acl-main.703, https://aclanthology. org/2020.acl-main.703
- **[34]** Li, C., Bi, B., Yan, M., Wang, W., Huang, S., Huang, F., Si, L.: Struc-turalLM: Structural pre-training for form understanding. In: Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Vol-ume 1: Long Papers). pp. 6309–6318. Association for Computational Linguis-tics, Online (Aug 2021). https://doi.org/10.18653/v1/2021.acl-long.493, https: //aclanthology.org/2021.acl-long.493
- **[35]** Li, P., Gu, J., Kuen, J., Morariu, V.I., Zhao, H., Jain, R., Manjunatha, V., Liu, H.: Selfdoc: Self-supervised document representation learning. In: 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 5648–5656 (2021). https://doi.org/10.1109/CVPR46437.2021.00560
- **[36]** Liao, M., Shi, B., Bai, X., Wang, X., Liu, W.: Textboxes: A fast text detector with a single deep neural network. Proceedings of the AAAI Conference on Artificial Intelligence 31(1) (Feb 2017). https://doi.org/10.1609/aaai.v31i1.11196, https: //ojs.aaai.org/index.php/AAAI/article/view/11196
- **[37]** Liu, W., Chen, C., Wong, K.Y.K., Su, Z., Han, J.: Star-net: A spatial attention residue network for scene text recognition. In: Richard C. Wilson, E.R.H., Smith, W.A.P. (eds.) Proceedings of the British Machine Vision Conference (BMVC). pp. 43.1–43.13. BMVA Press (September 2016). https://doi.org/10.5244/C.30.43, https://dx.doi.org/10.5244/C.30.43
- **[38]** Liu, Y., Gu, J., Goyal, N., Li, X., Edunov, S., Ghazvininejad, M., Lewis, M., Zettlemoyer, L.: Multilingual denoising pre-training for neural machine translation. Transactions of the Association for Computational Linguistics 8, 726–742 (2020), https://aclanthology.org/2020.tacl-1.47
- **[39]** Liu, Y., Chen, H., Shen, C., He, T., Jin, L., Wang, L.: Abcnet: Real-time scene text spotting with adaptive bezier-curve network. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (June 2020)
- **[40]** Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin trans-former: Hierarchical vision transformer using shifted windows. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 10012– 10022 (October 2021)
- **[41]** Long, S., Yao, C.: Unrealtext: Synthesizing realistic scene text images from the unreal world. arXiv preprint arXiv:2003.10608 (2020)
- **[42]** Majumder, B.P., Potti, N., Tata, S., Wendt, J.B., Zhao, Q., Najork, M.: Rep-resentation learning for information extraction from form-like documents. In: Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. pp. 6495–6504. Association for Computational Linguistics, Online (Jul 2020). https://doi.org/10.18653/v1/2020.acl-main.580, https://www.aclweb. org/anthology/2020.acl-main.580
- **[43]** Majumder, B.P., Potti, N., Tata, S., Wendt, J.B., Zhao, Q., Najork, M.: Repre-sentation learning for information extraction from form-like documents. In: Pro-ceedings of the 58th Annual Meeting of the Association for Computational Lin-guistics. pp. 6495–6504. Association for Computational Linguistics, Online (Jul 2020). https://doi.org/10.18653/v1/2020.acl-main.580, https://aclanthology. org/2020.acl-main.580
- **[44]** Mathew, M., Karatzas, D., Jawahar, C.: Docvqa: A dataset for vqa on document images. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 2200–2209 (2021)
- **[45]** Park, S., Shin, S., Lee, B., Lee, J., Surh, J., Seo, M., Lee, H.: Cord: A consolidated receipt dataset for post-ocr parsing. In: Workshop on Document Intelligence at NeurIPS 2019 (2019)
- **[46]** Peng, D., Wang, X., Liu, Y., Zhang, J., Huang, M., Lai, S., Zhu, S., Li, J., Lin, D., Shen, C., Jin, L.: SPTS: Single-Point Text Spotting. CoRR abs/2112.07917 (2021), https://arxiv.org/abs/2112.07917
- **[47]** Phan, T.Q., Shivakumara, P., Tian, S., Tan, C.L.: Recognizing text with per-spective distortion in natural scenes. In: Proceedings of the IEEE International Conference on Computer Vision (ICCV) (December 2013)
- **[48]** Powalski, R., Borchmann, L., Jurkiewicz, D., Dwojak, T., Pietruszka, M., Pa lka, G.: Going full-tilt boogie on document understanding with text-image-layout trans-former. In: Llad´os, J., Lopresti, D., Uchida, S. (eds.) Document Analysis and Recognition – ICDAR 2021. pp. 732–747. Springer International Publishing, Cham (2021)
- **[49]** Riba, P., Dutta, A., Goldmann, L., Forn´es, A., Ramos, O., Llad´os, J.: Table de-tection in invoice documents by graph neural networks. In: 2019 International Conference on Document Analysis and Recognition (ICDAR). pp. 122–127 (2019). https://doi.org/10.1109/ICDAR.2019.00028
- **[50]** Rijhwani, S., Anastasopoulos, A., Neubig, G.: OCR Post Correction for Endangered Language Texts. In: Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP). pp. 5931–5942. Association for Computational Linguistics, Online (Nov 2020). https://doi.org/10.18653/v1/2020.emnlp-main.478, https://aclanthology.org/ 2020.emnlp-main.478
- **[51]** Schaefer, R., Neudecker, C.: A two-step approach for automatic OCR post-correction. In: Proceedings of the The 4th Joint SIGHUM Workshop on Com-putational Linguistics for Cultural Heritage, Social Sciences, Humanities and Lit-erature. pp. 52–57. International Committee on Computational Linguistics, Online (Dec 2020), https://aclanthology.org/2020.latechclfl-1.6
- **[52]** Shi, B., Bai, X., Yao, C.: An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition. IEEE Transac-tions on Pattern Analysis and Machine Intelligence 39, 2298–2304 (2017)
- **[53]** Shi, B., Wang, X., Lyu, P., Yao, C., Bai, X.: Robust scene text recog-nition with automatic rectification. In: IEEE Conference on Com-puter Vision and Pattern Recognition (CVPR). pp. 4168–4176 (2016). https://doi.org/10.1109/CVPR.2016.452
- **[54]** Taghva, K., Beckley, R., Coombs, J.: The effects of ocr error on the extraction of private information. In: Bunke, H., Spitz, A.L. (eds.) Document Analysis Systems VII. pp. 348–357. Springer Berlin Heidelberg, Berlin, Heidelberg (2006)
- **[55]** Tan, M., Le, Q.: Efficientnetv2: Smaller models and faster training. In: Meila, M., Zhang, T. (eds.) Proceedings of the 38th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 139, pp. 10096–10106. PMLR (18–24 Jul 2021), https://proceedings.mlr.press/v139/tan21a.html
- **[56]** Tian, Z., Huang, W., He, T., He, P., Qiao, Y.: Detecting text in natural image with connectionist text proposal network. In: Leibe, B., Matas, J., Sebe, N., Welling, M. (eds.) Computer Vision – ECCV 2016. pp. 56–72. Springer International Pub-lishing, Cham (2016)
- **[57]** Tito, R., Mathew, M., Jawahar, C.V., Valveny, E., Karatzas, D.: Icdar 2021 compe-tition on document visual question answering. In: Llad´os, J., Lopresti, D., Uchida, S. (eds.) Document Analysis and Recognition – ICDAR 2021. pp. 635–649. Springer International Publishing, Cham (2021)
- **[58]** Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L.u., Polosukhin, I.: Attention is all you need. In: Guyon, I., Luxburg, U.V., Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S., Garnett, R. (eds.) Advances in Neural Information Processing Systems. vol. 30. Curran Associates, Inc. (2017), https://proceedings.neurips.cc/paper/2017/file/ 3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
- **[59]** Wang, J., Hu, X.: Gated recurrent convolution neural network for ocr. In: Guyon, I., Luxburg, U.V., Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S., Gar-nett, R. (eds.) Advances in Neural Information Processing Systems. vol. 30. Curran Associates, Inc. (2017), https://proceedings.neurips.cc/paper/2017/ file/c24cd76e1ce41366a4bbe8a49b02a028-Paper.pdf
- **[60]** Wang, S., Li, B., Khabsa, M., Fang, H., Ma, H.: Linformer: Self-attention with linear complexity. arXiv preprint arXiv:2006.04768 (2020)
- **[61]** Wightman, R.: Pytorch image models. https://github.com/rwightman/ pytorch-image-models (2019). https://doi.org/10.5281/zenodo.4414861
- **[62]** Williams, R.J., Zipser, D.: A learning algorithm for continually running fully re-current neural networks. Neural computation 1(2), 270–280 (1989)
- **[63]** Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., Davison, J., Shleifer, S., von Platen, P., Ma, C., Jernite, Y., Plu, J., Xu, C., Le Scao, T., Gugger, S., Drame, M., Lhoest, Q., Rush, A.: Transformers: State-of-the-art natural language processing. In: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations. pp. 38–45. Association for Computational Linguistics, Online (Oct 2020). https://doi.org/10.18653/v1/2020.emnlp-demos.6, https://aclanthology.org/2020.emnlp-demos.6
- **[64]** Xu, Y., Xu, Y., Lv, T., Cui, L., Wei, F., Wang, G., Lu, Y., Florencio, D., Zhang, C., Che, W., Zhang, M., Zhou, L.: LayoutLMv2: Multi-modal pre-training for visually-rich document understanding. In: Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Vol-ume 1: Long Papers). pp. 2579–2591. Association for Computational Linguis-tics, Online (Aug 2021). https://doi.org/10.18653/v1/2021.acl-long.201, https: //aclanthology.org/2021.acl-long.201
- **[65]** Xu, Y., Li, M., Cui, L., Huang, S., Wei, F., Zhou, M.: Layoutlm: Pre-training of text and layout for document image understanding. In: Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Min-ing. p. 1192–1200. KDD ’20, Association for Computing Machinery, New York, NY, USA (2020). https://doi.org/10.1145/3394486.3403172, https://doi.org/ 10.1145/3394486.3403172
- **[66]** Xu, Y., Lv, T., Cui, L., Wang, G., Lu, Y., Florencio, D., Zhang, C., Wei, F.: Layoutxlm: Multimodal pre-training for multilingual visually-rich document un-derstanding. arXiv preprint arXiv:2104.08836 (2021)
- **[67]** Yim, M., Kim, Y., Cho, H.C., Park, S.: Synthtiger: Synthetic text image generator towards better text recognition models. In: Llad´os, J., Lopresti, D., Uchida, S. (eds.) Document Analysis and Recognition – ICDAR 2021. pp. 109–124. Springer International Publishing, Cham (2021)
- **[68]** Zhang, K., Shasha, D.: Simple fast algorithms for the editing distance be-tween trees and related problems. SIAM J. Comput. 18, 1245–1262 (12 1989). https://doi.org/10.1137/0218082
- **[69]** Zhang, Z., Zhang, C., Shen, W., Yao, C., Liu, W., Bai, X.: Multi-oriented text detection with fully convolutional networks. In: 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 4159–4167 (2016). https://doi.org/10.1109/CVPR.2016.451
- **[70]** Zhong, X., ShafieiBavani, E., Jimeno Yepes, A.: Image-based table recognition: Data, model, and evaluation. In: Vedaldi, A., Bischof, H., Brox, T., Frahm, J.M. (eds.) Computer Vision – ECCV 2020. pp. 564–580. Springer International Pub-lishing, Cham (2020)

## Appendix

### A.1 Details of OCR Engines (MS, CLOVA, Easy, Paddle)

OCR-dependent VDU 백본은 모두 OCR 결과를 입력 특징의 일부로 받는다. 따라서 어느 OCR 엔진을 쓰느냐가 전체 시스템 성능을 좌우한다. 논문은 네 가지 OCR 엔진을 사용했다. 두 개는 API형 제품(MS OCR, CLOVA OCR)이고, 두 개는 공개 오픈소스(EasyOCR, PaddleOCR)다.

MS OCR은 Microsoft의 최신 OCR API로, 여러 최신 VDU 연구에서 사용되는 강력한 기준선이다. 논문 시점 기준으로 printed text 164개 언어, handwritten text 9개 언어를 지원한다. CLOVA OCR은 NAVER CLOVA의 API 제품으로, 문서 IE에 특화되어 있으며 영어·일본어·한국어를 지원한다. 실제로 본문 Figure 9의 CORD 분석에서는 CLOVA OCR이 가장 높은 정확도를 냈다.

EasyOCR은 공개 다운로드 가능한 OCR 엔진이며 80개가 넘는 언어를 지원한다. 현대 딥러닝 OCR 모듈을 바탕으로 구현되었고, 전체 파라미터 수는 약 27M으로 비교적 작다. PaddleOCR은 CPU 환경에서도 가볍게 돌릴 수 있는 모바일 중심 경량 OCR을 사용했으며, 영어와 중국어에 맞춰 설계된 버전의 크기는 약 10M 수준이다. 논문은 중국어 Ticket 데이터에는 PaddleOCR을, 나머지 문서 IE에는 CLOVA OCR을 사용했다. 문서 분류와 VQA에서 LayoutLM 계열의 속도 측정은 기존 연구 관행을 따라 MS OCR로 수행했다.

### A.2 Details of Synthetic Document Generator (SynthDoG)

Appendix는 SynthDoG의 세부 구성을 더 자세히 설명한다. Figure A는 영어·중국어·일본어·한국어 합성 샘플을 더 많이 보여 준다. 본문 Figure 4가 대표 예시였다면, Figure A는 SynthDoG가 실제로 얼마나 다양한 외관을 만들어 내는지 보여 주는 확장판이라고 볼 수 있다.

【Figure A 삽입 위치 - OCR_free_Document_Understanding_Transformer_Figure_A.png】

SynthDoG의 첫 번째 요소는 **배경(background)**이다. 배경 이미지는 ImageNet에서 샘플링하고, out-of-focus 효과를 흉내 내기 위해 가우시안 블러를 무작위 적용한다. 두 번째 요소는 **문서 표면(document)**이다. 저자들이 수집한 종이 사진에서 질감을 뽑아 흰 바탕에 입히고, 여기에 elastic distortion과 Gaussian noise를 더해 실제 인쇄물 표면처럼 만든다. 또 서로 다른 촬영 각도를 흉내 내기 위해 perspective transformation도 적용한다.

세 번째 요소는 **텍스트 레이아웃과 패턴**이다. 규칙 기반 패턴 생성기가 문서 영역 안에 여러 개의 사각형 텍스트 블록을 놓고, 각 블록을 다시 여러 줄 텍스트 영역으로 해석한다. 텍스트 크기와 여백도 무작위다. 네 번째 요소는 **텍스트 내용과 스타일**이다. 다국어 Wikipedia로부터 문장을 뽑고, Noto 폰트를 사용해 여러 언어를 안정적으로 렌더링한다. 텍스트 색도 랜덤하게 바꾼다. 마지막으로 **후처리(post-processing)** 단계에서 색상, 밝기, 대비, 그림자, motion blur, Gaussian blur, JPEG compression을 적용한다. 이 모든 단계가 합쳐져 Donut 사전학습에 필요한 시각적 다양성을 만든다.

### A.3 Details of Document Information Extraction

문서 IE는 단순한 OCR보다 어렵다. 논문은 그 이유를 세 단계로 설명한다. 첫째, 텍스트를 읽어야 한다. 둘째, 텍스트 의미를 이해해야 한다. 셋째, 추출된 정보 사이의 관계와 구조를 예측해야 한다. 과거 연구 중 일부는 몇 개의 key field만 뽑는 문제에 초점을 맞췄지만, Donut은 그보다 넓은 구조 복원 문제를 겨냥한다.

Appendix는 이를 설명하기 위해 Donut의 출력 예시를 더 자세히 제시한다. Figure B는 Ticket 데이터 예시다. 중국어 기차표는 정보 깊이(depth)가 1이라 구조가 단순하고, 각 key 위치도 거의 고정돼 있다. 그래서 Donut이 잘 맞히는 경우에는 전체 TED accuracy가 100에 가깝게 나온다. 반면 일부 예시에서는 인명이나 역 이름처럼 OCR과 구조 판단이 동시에 필요한 부분에서 오차가 발생한다. 그림에서 빨간색으로 표시된 부분이 실패 예측이다.

【Figure B 삽입 위치 - OCR_free_Document_Understanding_Transformer_Figure_B.png】

Figure C는 CORD 예시다. 이 데이터는 깊이 2의 계층 구조를 가진다. 메뉴 항목 하나가 부모 그룹이고, 그 안에 수량·이름·가격이 자식 필드로 들어간다. 그림은 Donut이 단순히 텍스트 문자열만 생성하는 것이 아니라, 어떤 필드들이 같은 item에 속하는지도 함께 맞혀야 함을 보여 준다. 일부 실패 사례에서는 잔돈(changeprice)이나 메뉴 개수(menuqty_cnt) 같은 구조적 필드가 틀린다. 이 때문에 CORD는 Ticket보다 훨씬 어려운 파싱 문제로 해석된다.

【Figure C 삽입 위치 - OCR_free_Document_Understanding_Transformer_Figure_C.png】

Figure D는 비공개 산업 데이터셋인 Business Card와 Receipt의 고품질 유사 샘플을 보여 준다. 실제 서비스 데이터는 정책상 공개할 수 없기 때문에, 논문은 real-like 샘플을 대신 제시한다. 이 그림은 Donut이 단일한 벤치마크 세트에만 맞춰진 모델이 아니라, 일본어 명함과 한국어 영수증 같은 실제 서비스 문서까지 겨냥하고 있음을 시각적으로 보여 준다.

【Figure D 삽입 위치 - OCR_free_Document_Understanding_Transformer_Figure_D.png】

### A.4 Details of Model Training Scheme and Output Format

이 절은 Donut의 teacher forcing 학습 방식과 출력 형식을 조금 더 직관적으로 설명한다. Transformer의 기본 encoder-decoder 구조를 따르되, 학습 시에는 이전 시점의 정답 토큰을 입력으로 넣는다. 그 결과 각 시점에서의 다음 토큰 분류를 동시에 cross-entropy로 학습할 수 있다.

Figure E는 이 과정을 시각적으로 보여 준다. 왼쪽은 training이고 오른쪽은 inference다. training에서는 `<parsing> <item> <name> 3002 ...` 같은 정답 prefix를 계속 넣으며 각 위치의 다음 토큰 분포를 맞힌다. inference에서는 이전 시점의 **모델 예측 토큰**이 다음 입력으로 들어간다. 그래서 훈련과 추론 사이에는 exposure bias 차이가 있지만, teacher forcing 덕분에 학습은 안정적이다. 또한 예측된 토큰열은 최종적으로 JSON 형식으로 변환된다.

【Figure E 삽입 위치 - OCR_free_Document_Understanding_Transformer_Figure_E.png】

### A.5 Implementation and Training Hyperparameters

구현은 Hugging Face `transformers`와 TIMM(PyTorch image models)을 기반으로 한다. 모든 학습은 fp16 mixed precision으로 수행한다. 최적화는 Adam을 사용하며, 학습 진행에 따라 learning rate를 감소시킨다. pre-training 초기 learning rate는 \(1e{-4}\), fine-tuning에서는 \(1e{-5}\)에서 \(1e{-4}\) 사이를 사용한다.

사전학습은 64개의 NVIDIA A100 GPU, mini-batch 196, 200K steps로 수행되며 대략 2~3 GPU day가 소요된다. gradient clipping도 사용하며 최대 gradient norm은 0.05에서 1.0 사이에서 선택한다. pre-training 입력 해상도는 \(2560 \times 1920\)이다. 반면 CORD, Ticket, Business Card 같은 일부 다운스트림 문서 IE 태스크에서는 \(1280 \times 960\)처럼 더 작은 해상도도 시험한다. 이 경우 한 개의 A100 GPU로 CORD나 Ticket fine-tuning이 약 0.5시간이면 가능했다. 하지만 대규모 RVL-CDIP나 DocVQA에서 \(2560 \times 1920\)을 쓰면 비용이 급격히 증가하며, DocVQA는 약 1 GPU day, RVL-CDIP는 약 2 GPU day가 필요했다.

또한 OCR-dependent baseline은 `transformers`를 사용해 BERT, BROS, LayoutLMv2, WYVERN을 구현했고, SPADE는 공식 구현을 사용했다. 학습 장비는 P40, V100, A100 GPU가 혼합되어 쓰였으며, learning rate와 epoch 수는 validation score를 보며 조정했다.

### A.6 Preliminary Experiments in Smaller Resources

저자들은 본 실험에 앞서 더 작은 데이터와 더 적은 GPU로 예비 실험을 수행했다. 이를 DonutProto라고 부른다. 이 설정에서는 SynthDoG 1.2M만 사용하고, 8개의 V100 GPU로 5일 동안 사전학습했으며, 입력 크기는 \(2048 \times 1536\)이었다.

이렇게 자원을 크게 줄인 조건에서도 DonutProto는 상당히 괜찮은 성능을 냈다. RVL-CDIP accuracy는 94.5, CORD accuracy는 85.4였다. 이는 본 논문의 최종 대규모 설정보다는 낮지만, OCR-free 접근이 거대한 컴퓨팅 자원이 있어야만 성립하는 것은 아니라는 점을 보여 준다. 저자들은 이런 예비 실험을 바탕으로 이후 더 많은 데이터와 더 큰 자원으로 스케일업했다고 설명한다.

### A.7 Details of OCR-dependent Baseline Models

이 절은 LayoutLM, LayoutLMv2 같은 OCR-dependent VDU 백본이 각 태스크를 어떻게 푸는지 정리한다. 공통된 핵심은 OCR 엔진 출력을 입력 특징으로 삼는다는 것이다. OCR이 추출한 텍스트를 정렬해 토큰 시퀀스로 만들고, Transformer 인코더가 이를 contextualized vector로 바꾼다. 이후 과제별로 아주 조금씩 다른 출력층을 붙인다.

**Document Classification**에서는 입력 시퀀스 맨 앞에 `[CLS]` 토큰을 붙인다. 최종적으로 `[CLS]`의 출력 벡터에 linear + softmax를 적용해 class label을 예측한다.

**Document IE**에서는 출력 시퀀스 전체에 linear + softmax를 적용해 BIO tag sequence를 예측한다. 구조 깊이가 1인 문서에서는 tag set이 다음과 같이 정의된다.

$$
\{ B_k, I_k, O \mid k \in \text{pre-defined keys} \}
$$

여기서 \(B_k\)는 key \(k\)의 시작, \(I_k\)는 key \(k\) 내부, \(O\)는 어떤 key에도 속하지 않는 토큰을 뜻한다.

구조 깊이가 \(n\)인 문서, 특히 논문이 설명하는 \(n=2\) 경우에는 부모 그룹과 자식 필드를 동시에 표현하기 위해 더 복잡한 tag set이 필요하다.

$$
\{ B_g.B_k,\; B_g.I_k,\; I_g.B_k,\; I_g.I_k,\; O \mid g \in \text{parent keys},\; k \in \text{child keys} \}
$$

예를 들어 CORD의 `menu`가 부모 키이고 `cnt`, `nm`, `price`가 자식 키라면, 그룹의 시작과 지속 여부를 나타내는 \(B_g, I_g\)와 각 자식 필드의 시작·내부를 나타내는 \(B_k, I_k\)가 결합된다. 논문은 이것이 Group BIO-tagging으로 알려진 방식이라고 설명한다.

**Document VQA**에서는 출력 벡터 시퀀스를 span-tag sequence로 바꿔 답변의 시작점과 끝점을 찾는다. 즉 OCR-dependent 문서 VQA는 본질적으로 추출형(question answering over OCR tokens)에 가깝다. 반면 Donut은 답변 문자열을 생성하는 방식이므로, 인터페이스 철학 자체가 다르다.

## 추가 설명

### 1. Donut이 해결하려는 핵심 병목

이 논문의 가장 중요한 문제의식은 “문서 이해 시스템의 핵심 병목이 사실상 OCR”이라는 진단이다. 많은 기존 연구가 문서 이해 백본을 개선하는 데 집중했지만, 실제 서비스 관점에서는 OCR 품질과 OCR 비용이 전체 시스템의 상한선을 만든다. Donut은 이 병목을 아예 구조에서 제거하려고 한다. 그래서 Donut의 의미는 단순히 정확도가 약간 오른 새 모델이라기보다, 문서 이해 파이프라인의 중심을 OCR에서 end-to-end 생성 모델로 옮긴 시도라고 보는 편이 맞다.

### 2. 왜 “텍스트 읽기”를 먼저 배우게 하는가

사전학습 목표가 문서를 읽는 것이라는 점은 매우 중요하다. 문서 이해는 결국 읽기에서 출발한다. OCR-free라고 해서 텍스트를 건너뛰는 것이 아니다. 오히려 텍스트 읽기 자체를 모델 내부 능력으로 흡수한다. 이 전략 덕분에 fine-tuning 단계에서는 “이 문서에서 무엇이 중요한가”와 “그 정보를 어떤 구조로 내보내야 하는가”에 집중할 수 있다. 저자들이 캡셔닝보다 text reading pre-training이 더 낫다고 주장한 이유도 여기에 있다.

### 3. JSON 출력이 갖는 장점

Donut의 출력은 단순한 문자열이 아니라 JSON이다. 이것은 구현상 꽤 큰 장점이다. 문서 IE에서는 같은 정보를 여러 수준의 계층으로 묶어야 하는 경우가 많다. 예를 들어 영수증에서 메뉴 리스트는 여러 개의 item으로 이루어지고, 각 item 안에 이름·수량·가격이 들어간다. BIO tagging은 이런 구조를 복원하려면 추가 규칙이 많이 필요하다. 반면 JSON 생성은 구조를 그대로 출력할 수 있다. 그래서 Donut은 분류·IE·VQA를 모두 생성 문제로 통합하면서도, IE의 계층 구조를 자연스럽게 표현할 수 있다.

### 4. Table 2가 말해 주는 것

Table 2는 이 논문의 가장 중요한 실험 결과다. Donut은 CORD, Ticket, Business Card, Receipt 네 도메인 모두에서 최고 F1과 최고 TED accuracy를 기록한다. 특히 구조가 복잡한 CORD와 Receipt에서 차이가 크게 난다. 이것은 Donut이 “텍스트를 잘 읽는다”는 것만으로 설명되지 않는다. 더 정확히는 텍스트 읽기와 구조 복원을 하나의 생성 문제로 묶었기 때문에, 필드 사이 관계를 더 자연스럽게 학습한 결과라고 볼 수 있다.

### 5. DocVQA 결과를 어떻게 읽어야 하는가

DocVQA에서는 Donut이 전체 최고 점수는 아니다. 이 점을 숨기지 않는 것이 중요하다. 대신 손글씨 같은 OCR 취약 영역에서는 더 강하다. 따라서 이 결과는 Donut이 모든 문서 이해 문제를 완전히 지배했다는 의미가 아니라, OCR 기반 시스템과 다른 실패 양상을 가진 대안이라는 뜻이다. OCR 기반 방법은 OCR이 틀리면 그 위에 아무리 좋은 백본을 올려도 한계가 생긴다. Donut은 그런 상한선에서 더 자유롭지만, 대신 고해상도 큰 문서의 미세한 글자에서는 입력 해상도 제약을 받는다.

### 6. Figure 8이 암시하는 것

Figure 8의 cross-attention 시각화는 단순한 예쁜 그림이 아니다. Donut이 detector supervision 없이도 필요한 글자 위치를 찾고 있다는 간접 증거다. 이는 OCR-free라고 해서 공간 정보가 사라지는 것이 아니라, 오히려 위치 정보를 내부 attention 패턴으로 흡수하고 있음을 보여 준다. 다시 말해 Donut은 명시적인 bounding box를 출력하지 않아도, 내부적으로는 “어디를 보고 읽어야 하는지”를 학습한다.

### 7. SynthDoG의 역할

SynthDoG는 이 논문의 실용성을 뒷받침하는 핵심 장치다. 실제 문서 이미지와 정답 텍스트를 수천만 장 규모로 모으는 것은 매우 어렵다. 특히 영어 외 언어나 산업 문서에서는 더 그렇다. SynthDoG는 이 문제를 우회한다. 완벽하게 사실적인 문서를 만들지는 못하더라도, 문서를 읽는 사전학습에 필요한 시각적 다양성을 값싸게 공급한다. 본문 Figure 7(a)에서 synthetic data만으로도 IE에 충분히 강한 성능이 나온다는 분석은 이 아이디어의 타당성을 뒷받침한다.

### 8. OCR-free가 항상 공짜는 아니다

OCR를 제거하면 파이프라인은 단순해지지만, 그 대가가 완전히 없는 것은 아니다. Donut은 고해상도 입력에서 계산량이 크게 늘고, 큰 문서 안의 매우 작은 글자를 보려면 해상도를 올려야 한다. 따라서 OCR-free가 무조건 더 싸다고 단정할 수는 없다. 다만 저자들의 핵심 주장은 여전히 유효하다. OCR 엔진을 따로 운영하고, 그 오류를 보정하고, 언어와 도메인에 따라 유지하는 비용까지 합치면 end-to-end 모델이 전체 시스템 비용을 더 낮출 수 있다는 것이다.

### 9. 이 논문의 위치

Donut은 이후 등장한 여러 문서 이해 생성 모델의 출발점에 가깝다. “문서 이미지를 보고 구조화 결과를 직접 생성한다”는 발상, “OCR 없이도 문서 읽기와 이해를 통합할 수 있다”는 주장, “합성 문서 생성기를 활용해 다국어 사전학습을 확장한다”는 전략이 이후 연구의 중요한 기반이 되었다. 따라서 이 논문은 단순히 한 모델의 성능 보고서가 아니라, 문서 이해에서 OCR-free generation이라는 방향을 선명하게 제시한 작업으로 읽는 것이 적절하다.
