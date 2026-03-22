---
title: "저랭크 업데이트로 미세조정을 가볍게 — LoRA: Low-Rank Adaptation of Large Language Models"
source_paper: "Hu et al., 2021"
arxiv: "2106.09685"
---

# 저랭크 업데이트로 미세조정을 가볍게 — LoRA: Low-Rank Adaptation of Large Language Models

이 문서는 LoRA 논문을 “거대한 모델을 전부 다시 학습하지 않고도 어떻게 강하게 적응시킬 수 있는가”라는 질문에 답하는 해설본이다. 핵심은 사전학습 가중치 전체를 건드리는 대신, **저랭크 업데이트 \(\Delta W = BA\)** 만 학습해 미세조정 비용을 크게 낮추는 데 있다. 그래서 이 논문은 단순한 경량화 기법이 아니라, **저장·배포·학습 비용을 동시에 줄이는 파라미터 효율적 적응(PEFT)의 기준점**으로 읽는 편이 좋다. 처음 읽는다면 Section 4, Table 1, Table 2~4, Section 7 순서로 보면 전체 메시지가 가장 잘 잡힌다.

## 핵심 요약
- **푸는 문제:** 초거대 언어모델을 태스크별로 full fine-tuning 하면 저장·배포·학습 비용이 너무 커진다.
- **핵심 아이디어:** pretrained weight는 freeze하고, 선형층 업데이트를 저랭크 행렬 곱 \(BA\)로 제한해 작은 수의 파라미터만 학습한다.
- **주요 결과:** RoBERTa, DeBERTa, GPT-2, GPT-3 계열에서 full fine-tuning과 비슷하거나 더 나은 성능을 보이며, 학습 파라미터와 메모리 사용량을 크게 줄였다.
- **왜 중요한가:** 이후 LoRA는 LLM 미세조정의 사실상 표준 도구가 되었고, 다양한 adapter·routing 기법과 결합되는 기본 블록이 됐다.
- **한계 / 주의점:** 작은 rank가 항상 충분한 것은 아니며, 어떤 weight matrix에 LoRA를 넣을지와 rank를 어떻게 고를지가 성능에 직접적인 영향을 준다.

## 3분 요약: LoRA가 바꾸는 것
| 방식 | 무엇을 학습하는가 | 장점 | 주의할 점 |
| --- | --- | --- | --- |
| Full Fine-Tuning | 모델 전체 | 최대 표현력 | 저장/배포/옵티마이저 비용이 큼 |
| Adapter | 추가 모듈 | 파라미터 절감 | 추론 경로에 추가 연산이 남을 수 있음 |
| Prefix / Prompt | 입력 측면의 가상 토큰 | 구조 변경이 작음 | 길이 예산과 최적화 난도가 문제될 수 있음 |
| LoRA | 저랭크 업데이트 \(BA\) | 작은 파라미터, 병합 가능, 추가 추론 지연 거의 없음 | rank·적용 위치 선택이 중요 |

## 먼저 읽을 포인트
- Section 4는 LoRA의 핵심이다. \(\Delta W = BA\)라는 한 줄이 왜 효과적인지 먼저 이해해야 한다.
- Table 1은 adapter 대비 latency 차이를, Table 2~4는 실제 성능과 파라미터 효율을 보여 준다.
- Section 7은 LoRA가 단순히 “작은 모듈”이 아니라, 실제로 어떤 방향의 업데이트를 학습하는지 설명해 주는 분석 파트다.

## 문헌 정보
- 논문: *LoRA: Low-Rank Adaptation of Large Language Models*
- arXiv: 2106.09685v2
- 원문 링크: https://arxiv.org/abs/2106.09685
- PDF: https://arxiv.org/pdf/2106.09685.pdf

---

# Abstract

## 핵심 논점

- 이 섹션이 답하려는 질문  
  초거대 언어모델을 다운스트림 태스크에 적응시킬 때, full fine-tuning의 저장/배포/학습 비용이 급증한다. 이를 완화하면서도 성능을 유지하는 방법은 무엇인가?

- 선수 개념  
  - full fine-tuning과 파라미터 효율적 적응(PEFT)  
  - Transformer의 선형 변환 가중치 행렬  
  - low-rank(저랭크) 행렬 분해

- 읽으면서 확인할 포인트  
  1) LoRA가 **고정(freeze)** 하는 대상과 **학습** 하는 대상  
  2) 파라미터/메모리 절감이 “어떤 항목”에서 발생하는지(학습 파라미터, 옵티마이저 상태 등)  
  3) adapter 대비 추론 지연(latency) 특성의 차이  

## 본문 정리

- 자연어처리의 대표적 접근은 대규모 사전학습(pre-training) 이후 다운스트림 태스크에 대한 적응(adaptation)이다.   
- 그러나 모델 규모가 커질수록(예: GPT-3 175B) full fine-tuning은 태스크별로 거대한 모델 인스턴스를 별도 저장·배포해야 하므로 현실성이 떨어진다. 

- 논문은 **LoRA(Low-Rank Adaptation)** 를 제안한다. LoRA는  
  - 사전학습 가중치(pre-trained weights)를 **고정(freeze)** 하고,   
  - Transformer 각 레이어에 **학습 가능한 저랭크(rank decomposition) 행렬**을 주입(inject)하여,   
  - 다운스트림 태스크에서 학습해야 할 파라미터 수를 크게 줄인다. 

- GPT-3 175B를 예시로, LoRA는 Adam 기반 full fine-tuning 대비  
  - 학습 파라미터 수를 **10,000배**까지 줄일 수 있고,   
  - GPU 메모리 요구량을 **3배**까지 줄일 수 있다고 주장한다.   

- 또한 RoBERTa/DeBERTa/GPT-2/GPT-3에서 full fine-tuning과 유사하거나 더 나은 성능을 보인다고 보고한다.   
- 추가로 학습 파라미터가 적기 때문에 학습 처리량(training throughput)이 증가하며, adapter와 달리 **추론 지연(inference latency)** 이 추가되지 않도록 설계되었다고 설명한다. 

- 논문은 언어모델 적응에서 관찰되는 **rank-deficiency(랭크 결핍)** 를 실증적으로 조사하여 LoRA의 효율성에 대한 통찰을 제공한다고 밝힌다.   

- PyTorch 모델에 LoRA를 통합하기 위한 패키지와 RoBERTa/DeBERTa/GPT-2 구현 및 체크포인트를 제공한다고 명시한다(microsoft/LoRA). 

- (각주 0) v2에는 더 나은 베이스라인, GLUE 실험, adapter latency 분석이 추가되었다고 명시한다.   
- (각주 1) GPT-3는 few-shot으로도 성능이 나오지만, fine-tuning이 성능을 크게 향상시킨다고 Appendix A를 근거로 언급한다.   

## 해설

  
full fine-tuning은 태스크별로 “전체 모델 사본”을 갖게 되는 방식이므로, 모델이 커질수록 저장·배포·서빙 비용이 선형적으로 누적된다. LoRA는 “공유 가능한 베이스 모델”을 유지한 채, 태스크별로 작은 저랭크 업데이트만 별도 저장하는 구조를 통해 이러한 누적 비용을 줄이려는 접근으로 이해할 수 있다.

---

# 1 Introduction

## 핵심 논점

- 질문  
  기존의 PEFT(adapters, prefix/prompt tuning 등)가 있는데도 LoRA가 필요한 이유는 무엇인가? LoRA의 설계 선택(저랭크 업데이트, 병합 가능성)은 어떤 문제를 겨냥하는가?

- 선수 개념  
  - 태스크별 체크포인트 비용(저장/배포)  
  - 온라인 추론(배치가 작은 서빙)의 latency 특성  
  - prefix/prompt 방식의 시퀀스 길이 예산 제약

- 읽으면서 확인할 포인트  
  1) “intrinsic dimension/low-dimensional structure” 문헌에서 LoRA의 가설이 어떻게 도출되는지  
  2) LoRA가 제시하는 장점(불릿 4개)의 논리 구조  
  3) Transformer 표기: \(W_q, W_k, W_v, W_o\), \(d_{\text{model}}\), \(d_{\text{ffn}}\) 등

## 본문 정리

- NLP의 많은 응용은 하나의 대규모 사전학습 모델을 다양한 다운스트림 애플리케이션에 적응시키는 형태다.   
- 일반적인 적응은 full fine-tuning으로 이루어지며, 이 경우 태스크별 모델은 원 모델과 동일한 파라미터 수를 갖는다.   
- GPT-2나 RoBERTa-large 수준에서는 “불편” 수준이지만, GPT-3 175B에서는 태스크별로 독립 모델을 저장·배포하는 것이 “치명적 도전”이 된다고 서술한다.   

- 해결책으로 “일부 파라미터만 적응”하거나 “외부 모듈을 학습”하는 방식이 다수 제안되었다.   
  그러나 기존 기법은 종종  
  - adapter처럼 **추론 지연**을 증가시키거나,   
  - prefix/prompt 계열처럼 **사용 가능한 시퀀스 길이**를 줄이거나,   
  - full fine-tuning 성능을 안정적으로 따라가지 못하는 “효율–품질” 트레이드오프를 갖는다고 지적한다. 

- 저자들은 Li et al.(2018a), Aghajanyan et al.(2020) 등에서 관찰된 “과매개변수 모델의 낮은 intrinsic dimension”에 착안하여, **적응 과정의 가중치 변화(업데이트) 역시 낮은 intrinsic rank를 가질 것**이라는 가설을 제시한다. 

- LoRA는 원래 가중치 \(W\)를 직접 업데이트하는 대신, \(W\)의 변화분을 저랭크 행렬 분해 형태로 인코딩하여 학습한다는 아이디어로 소개된다(그림 1).   
  GPT-3 175B 예시에서 full rank가 매우 커도, 작은 rank로 충분할 수 있다는 직관을 제시한다. 

- LoRA의 장점 4가지를 원문은 다음과 같이 정리한다. 
  1) **공유 베이스 + 태스크별 소형 모듈**: 베이스 모델은 공유하고, 태스크별로 작은 LoRA 모듈(A,B)만 교체하면 된다.   
  2) **학습 효율/하드웨어 장벽 완화**: Adam 등 adaptive optimizer에서 optimizer state를 유지해야 하는 파라미터가 크게 줄어 효율이 개선된다(최대 3배).   
  3) **설계상 추가 추론 지연 없음**: 선형 구조로 인해 학습 후 가중치를 병합(merge)하여 배포 가능하므로, fully fine-tuned 모델 대비 inference latency가 구조적으로 추가되지 않는다.   
  4) **기존 방법과 직교(결합 가능)**: prefix-tuning 등 기존 기법과 결합 가능하며 Appendix E에 예시를 제시한다. 

- Terminologies and Conventions: Transformer 표기 및 관례를 정리한다.   
  - \(d_{\text{model}}\): Transformer layer의 입력/출력 차원  
  - self-attention의 projection matrices: \(W_q, W_k, W_v, W_o\)  
  - MLP의 feedforward 차원은 \(d_{\text{ffn}} = 4 \times d_{\text{model}}\)로 둔다고 언급한다.   
  - 최적화는 Adam을 관례로 언급한다. 

## 해설

  
LoRA는 “입력을 바꾸는 방식(prefix/prompt)”이 아니라 “가중치 업데이트를 저랭크로 제한하는 방식”이다. 이로 인해 (i) 시퀀스 길이 예산을 소모하지 않고, (ii) 학습 후에는 가중치 병합을 통해 추론 경로를 베이스 모델과 동일하게 유지할 수 있다는 점이 설계 상의 핵심 차별점으로 제시된다.

---

# 2 Problem Statement

## 핵심 논점

- 질문  
  다운스트림 적응을 수식으로 어떻게 정식화하며, “파라미터 효율적 적응”은 어떤 형태의 최적화 문제로 표현되는가?

- 선수 개념  
  - 오토리그레시브 언어모델 \(P_\Phi(y\mid x)\)  
  - 조건부 생성과 로그우도 최대화  
  - “베이스 파라미터 + 작은 업데이트” 관점

- 읽으면서 확인할 포인트  
  1) 데이터셋 \(Z=\{(x_i,y_i)\}\) 정의  
  2) full fine-tuning 목적식(식 1)  
  3) \(\Delta\Phi(\Theta)\)로의 재정의(식 2)와 의미  
  4) \(|\Theta|\ll|\Phi_0|\) 조건의 역할  

## 본문 정리

- 제안은 training objective에 대해 agnostic하나, 동기 예시로는 language modeling에 집중한다.   
- 사전학습된 오토리그레시브 LM을 \(P_\Phi(y\mid x)\)로 두고, 파라미터를 \(\Phi\)라 표기한다.   
- 다운스트림 태스크 데이터는 컨텍스트/타깃 쌍 데이터셋으로 주어진다:
  \[
  Z=\{(x_i,y_i)\}_{i=1,\ldots,N}
  \]
  여기서 \(x_i, y_i\)는 토큰 시퀀스다.   
  예: NL2SQL에서 \(x\)는 자연어 질의, \(y\)는 SQL; 요약에서 \(x\)는 기사, \(y\)는 요약문. 

- Full fine-tuning의 목적식(식 1)은 다음 로그우도 최대화로 제시된다:
  \[
  \max_{\Phi}\ \sum_{(x,y)\in Z}\ \sum_{t=1}^{|y|}\ \log\Big(P_{\Phi}(y_t \mid x, y_{<t})\Big) \tag{1}
  \]
  

- Full fine-tuning의 단점은 태스크별 업데이트 \(\Delta\Phi\)가 \(\Phi_0\)와 같은 규모를 갖게 되어, 태스크별로 전체 모델 크기만큼 저장·배포해야 한다는 점이다. 

- 파라미터 효율적 적응은 \(\Delta\Phi\)를 더 작은 파라미터 \(\Theta\)로 인코딩한다:
  \[
  \Delta\Phi=\Delta\Phi(\Theta),\quad |\Theta|\ll |\Phi_0|
  \]
  그리고 최적화는 다음 식(2)처럼 \(\Theta\)에 대해 이루어진다:
  \[
  \max_{\Theta}\ \sum_{(x,y)\in Z}\ \sum_{t=1}^{|y|}\ \log\Big(p_{\Phi_0 + \Delta\Phi(\Theta)}(y_t \mid x, y_{<t})\Big) \tag{2}
  \]
  

- 이후 섹션에서 \(\Delta\Phi\)를 저랭크 표현으로 인코딩하여 효율화하겠다고 예고하며, GPT-3 175B에서 \(|\Theta|\)가 \(|\Phi_0|\)의 0.01%까지 작아질 수 있다고 언급한다. 

## 해설

  
식 (1)은 “모델 전체 \(\Phi\)”를 직접 최적화하는 문제이고, 식 (2)는 “작은 모듈 \(\Theta\)”만 최적화하되 실제 모델 파라미터는 \(\Phi_0+\Delta\Phi(\Theta)\) 형태로 간접적으로 변한다. LoRA는 여기서 \(\Delta\Phi(\Theta)\)를 저랭크 행렬 곱으로 구체화한 사례로 이해할 수 있다.

---

# 3 Aren’t Existing Solutions Good Enough?

## 핵심 논점

- 질문  
  기존의 효율적 적응 방법(adapters, prefix/prompt 계열)이 존재하는데, 대규모·저지연 프로덕션 환경에서 어떤 한계가 있는가?

- 선수 개념  
  - adapter layer(추가 깊이)  
  - online inference(배치 크기 1 근접)에서의 latency  
  - prefix/prompt의 시퀀스 길이 점유

- 읽으면서 확인할 포인트  
  1) adapter가 “파라미터가 적어도” latency를 유의미하게 증가시킬 수 있는 조건  
  2) prefix/prompt 직접 최적화의 난점(최적화/스케일링/길이 예산)  
  3) 표/캡션에서 명시된 측정 조건(하드웨어, 배치/시퀀스 길이 등)

## 본문 정리

- 효율적 적응 연구는 오래되었고, 언어모델 맥락에서는 특히 다음 두 전략이 두드러진다고 서술한다.   
  1) **adapter layers 추가**  
  2) **입력 레이어 activation 최적화**(prefix/prompt 계열)  
  두 전략 모두 대규모·latency 민감한 프로덕션에서 한계를 가질 수 있다고 주장한다. 

---

## 3.2 Adapter Layers Introduce Inference Latency

- 다양한 adapter 변형이 있으나, 본문은  
  - Houlsby et al.(2019)의 설계(블록당 2개 adapter),  
  - Lin et al.(2020)의 설계(블록당 1개 + 추가 LayerNorm)  
  를 중심으로 논의한다고 밝힌다. 

- adapter는 bottleneck dimension을 작게 두어 파라미터와 FLOPs가 제한적이지만, 큰 모델의 latency는 하드웨어 병렬성에 좌우되며 adapter는 순차적으로 실행되어야 하므로 특히 online inference에서 차이가 발생할 수 있다고 설명한다. 

> **Table 1 삽입**

### Table 1 (GPT-2 medium, forward pass latency)
- 측정 조건(캡션): NVIDIA Quadro RTX8000, forward pass latency, 100 trials 평균.   
- 비교: Fine-Tune/LoRA vs AdapterL vs AdapterH.   
- Batch Size / Sequence Length / \(|\Theta|\) (adapter 학습 파라미터 수)가 함께 제시된다. 

**Table 1 수치(원문 그대로 재서술)**   
- Fine-Tune/LoRA:
  - 1449.4±0.8 (bs32, len512)  
  - 338.0±0.6 (bs16, len256)  
  - 19.8±2.7 (bs1, len128)  
- AdapterL:
  - 1482.0±1.0 (+2.2%)  
  - 354.8±0.5 (+5.0%)  
  - 23.9±2.1 (+20.7%)  
- AdapterH:
  - 1492.2±1.0 (+3.0%)  
  - 366.3±0.5 (+8.4%)  
  - 25.8±2.2 (+30.3%)  

- 또한 모델 병렬/샤딩 환경에서 추가 깊이는 AllReduce/Broadcast 같은 동기화 연산을 늘릴 수 있고, adapter 파라미터를 중복 저장하지 않는 한 문제가 커질 수 있다고 언급한다. 

---

## 3.3 Directly Optimizing the Prompt is Hard

- prefix tuning 계열은 다른 어려움이 있다고 서술한다.   
- 저자들은 prefix tuning이 최적화가 어렵고, 학습 파라미터 수를 늘려도 성능이 단조 증가하지 않는 현상을 관찰했다고 말하며, 이는 원 논문에서도 보고된 바 있다고 언급한다.   
- 또한 prefix/prompt에 시퀀스 길이를 할당하면 다운스트림 입력/출력에 사용할 유효 시퀀스 길이가 줄어들어 성능을 저하시킬 수 있다고 “의심한다”고 표현한다.   
- 정량 비교는 Section 5에서 수행한다고 예고한다. 

## 해설

  
Table 1은 “adapter의 오버헤드가 언제 두드러지는가”를 보여준다. 특히 배치가 작고 시퀀스가 짧은 환경에서(온라인 서빙에 가깝게) adapter의 추가 연산이 병목으로 작동할 가능성이 크다는 점이 수치로 제시된다. 반면 배치/시퀀스가 커지면 GPU 병렬성에 의해 상대적 오버헤드가 완화될 수 있다(논문은 Appendix B에서 더 विस्त게 분석).

---

# 4 Our Method

## 핵심 논점

- 질문  
  LoRA는 업데이트를 어떤 수학적 형태로 제한하며, 왜 추론 시 추가 latency 없이 배포 가능한가? Transformer에서는 어디에 적용하는가?

- 선수 개념  
  - 저랭크 분해: \(\Delta W = BA\)  
  - 가중치 동결(freeze)과 모듈 학습  
  - self-attention projection matrices

- 읽으면서 확인할 포인트  
  1) \(\Delta W\)의 저랭크 파라미터화와 학습 대상  
  2) 초기화 및 스케일링 \(\alpha/r\)의 목적  
  3) 추론 시 병합(merge) 가능성의 근거  
  4) Transformer에서의 적용 위치 및 실험의 기본 선택(MLP freeze 등)

## 본문 정리

- LoRA의 원리는 어떤 dense layer에도 적용 가능하지만, 실험은 Transformer LM의 특정 가중치에 집중한다고 밝힌다. 

---

## 4.1 Low-Rank-Parametrized Update Matrices

- pretrained weight \(W_0\in\mathbb{R}^{d\times k}\)에 대해 업데이트를 다음과 같이 제한한다:
  \[
  W_0 + \Delta W = W_0 + BA,\quad
  B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times k},\ r \ll \min(d,k)
  \]
  

- 학습 동안 \(W_0\)는 **freeze** 하고, \(A,B\)만 학습한다.   
- forward 계산은 다음과 같이 제시된다:
  \[
  h = W_0x + \Delta W x = W_0x + BAx \tag{3}
  \]
  

- 초기화와 스케일링:
  - \(A\): Gaussian 초기화  
  - \(B\): 0으로 초기화  
  → 초기 시점 \(\Delta W = BA = 0\)이 되어 모델은 원 모델과 동일하게 동작한다.   
  - \(\Delta W x\)에 \(\alpha/r\) 스케일링을 적용하며, \(\alpha\)는 \(r\)에 대한 상수로 둔다.   
  - Adam 최적화에서는 \(\alpha\) 튜닝이 학습률 튜닝과 유사한 효과를 낼 수 있어, 저자들은 별도 탐색 없이 처음 시도한 \(r\)로 \(\alpha\)를 설정한다고 말한다. 

- Full fine-tuning과의 관계:
  - 모든 weight matrix에 LoRA를 적용하고 bias도 학습하면(각주 2: bias는 weight 대비 매우 적음), rank \(r\)을 충분히 크게 둘 때 full fine-tuning의 표현력을 대략 회복할 수 있다고 설명한다.   
  - (각주 3) 어려운 태스크에서는 trainable parameter 수가 늘어나는 것이 불가피할 수 있다고 덧붙인다.   
  - \(r\) 증가 시 LoRA는 full fine-tuning에 수렴하는 반면, adapter는 MLP 구조로, prefix는 긴 입력을 받지 못하는 구조로 수렴한다는 대비를 제시한다. 

- No additional inference latency:
  - 배포 시 \(W=W_0+BA\)를 명시적으로 계산해 저장하고, 일반적인 inference를 수행할 수 있다고 한다.   
  - 태스크 전환 시에는 \(BA\)를 빼서 \(W_0\)를 복원하고 다른 태스크의 \(B'A'\)를 더하는 방식으로 교체 가능하며, 이 연산은 빠르고 메모리 오버헤드가 작다고 설명한다.   
  - 따라서 설계상 fully fine-tuned 모델 대비 추가 inference latency가 없다고 주장한다. 

---

## 4.2 Applying LoRA to Transformer

- 원칙적으로 어떤 weight matrix subset에도 LoRA를 적용할 수 있다고 말한다.   
- Transformer에서는 self-attention에 \(W_q, W_k, W_v, W_o\) 네 개, MLP에 두 개의 weight matrix가 존재한다고 정리한다.   
- \(W_q\) (또는 \(W_k, W_v\))는 출력이 head로 나뉘더라도 \(d_{\text{model}}\times d_{\text{model}}\) 하나의 행렬로 취급한다고 명시한다. 

- 실험 설정(동기 부여된 사용 사례): 단순성과 파라미터 효율을 위해  
  - downstream에서 attention weight만 adapt하고  
  - MLP 모듈은 freeze하는 구성을 택했다고 말한다.   
  어떤 attention weight를 adapt할지의 효과는 Section 7.1에서 추가로 연구한다고 예고한다.   
  MLP/LayerNorm/bias 적응에 대한 실증 연구는 future work로 남긴다. 

- Practical benefits and limitations:
  - Adam으로 큰 Transformer를 학습할 때 freeze된 파라미터의 optimizer state를 저장하지 않아도 되어 VRAM 사용량을 크게 줄일 수 있다고 한다. \(r\ll d_{\text{model}}\)이면 VRAM을 최대 2/3까지 줄일 수 있다고 서술하며, GPT-3 175B에서는 1.2TB→350GB로 줄였다고 보고한다.   
  - \(r=4\)이고 query/value projection만 adapt하면 체크포인트가 350GB→35MB로 약 10,000× 감소한다고 주장한다. (각주 4: 태스크 100개 저장 시 35TB vs 354GB 비교)   
  - GPT-3 175B에서 full fine-tuning 대비 LoRA가 25% speedup을 관찰했다고 말하며, 각주 5로 throughput(32.5→43.1 tokens/s per V100 GPU, 동일 shard 수)을 제시한다.   
  - 한계로는, 서로 다른 태스크의 서로 다른 LoRA 모듈을 가진 샘플을 하나의 배치에서 처리하는 것이(단일 forward) 병합 전략 하에서는 쉽지 않다는 점을 언급한다. latency가 중요하지 않으면 병합하지 않고 동적으로 선택하는 방법을 언급한다. 

## 해설

  
LoRA의 “추론 지연 없음” 주장은, 학습 시에는 \(W_0x\) 경로와 \(BAx\) 경로가 병렬로 더해지지만, 배포 시에는 \(W=W_0+BA\)로 단일 선형 변환으로 합칠 수 있다는 구조적 성질에 기반한다. 반면 adapter는 모델 계산 그래프에 추가 모듈이 삽입되므로, 배포 시에도 추가 연산이 남게 되는 차이가 있다.

---

# 5 Empirical Experiments

## 핵심 논점

- 질문  
  LoRA가 다양한 모델( RoBERTa/DeBERTa/GPT-2/GPT-3 )과 과제(NLU/NLG/NL→SQL/요약)에서 성능과 효율을 어떻게 보이는가? 기존 PEFT( BitFit, prefix, adapters )와 비교 시 어떤 양상이 나타나는가?

- 선수 개념  
  - GLUE 태스크와 지표(MNLI/CoLA/STS-B 등)  
  - NLG 자동평가 지표(BLEU/NIST/METEOR/ROUGE-L/CIDEr)  
  - 학습 파라미터 수 \(|\Theta|\)의 정의(“추가 학습 모듈” 기준)

- 읽으면서 확인할 포인트  
  1) 공정 비교를 위한 실험 세팅 정렬(특히 RoBERTa에서 † 표기)  
  2) 표에서 \(*\), \(†\) 표기의 의미  
  3) trainable parameter 규모 변화에 따른 성능 곡선(Figure 2)  
  4) GPT-3 175B에서의 대표 표준편차(typical fluctuation) 보고 방식  

## 본문 정리

- LoRA를 RoBERTa(base/large), DeBERTa XXL, GPT-2(medium/large), GPT-3 175B에 적용해 다운스트림 성능을 평가한다.   
- 과제는 RoBERTa/DeBERTa에서는 GLUE, GPT-2에서는 Li & Liang(2021)과의 직접 비교 세팅, GPT-3에서는 WikiSQL/SAMSum 등을 포함한다.   
- 모든 실험은 NVIDIA Tesla V100에서 수행되었다고 명시한다. 

---

## 5.1 Baselines

- 가능한 많은 베이스라인과 비교하기 위해, 일부는 이전 연구의 세팅을 재현하고, 일부는 이전 연구가 보고한 수치를 재사용한다고 말한다. 따라서 일부 베이스라인은 특정 실험에만 등장할 수 있다고 언급한다. 

### Fine-Tuning(FT) 및 부분 FT(FTTop2)
- FT: 사전학습 가중치/바이어스로 초기화한 뒤 모든 파라미터를 업데이트.   
- GPT-2에서는 마지막 두 레이어만 적응하는 FTTop2를 포함. 

### Bias-only / BitFit
- BitFit: 바이어스 벡터만 학습하고 나머지는 동결(Zaken et al., 2021 언급). 

### Prefix-embedding tuning (PreEmbed)
- 입력 토큰들 사이에 특수 토큰을 삽입하고, 그 토큰 임베딩만 학습한다. prefixing(앞)과 infixing(뒤) 모두 고려한다고 설명한다.   
- \(l_p\): prefix 토큰 수, \(l_i\): infix 토큰 수  
- 학습 파라미터 수:
  \[
  |\Theta| = d_{\text{model}} \times (l_p + l_i)
  \]
  

### Prefix-layer tuning (PreLayer)
- PreEmbed 확장으로, 매 레이어에서 특수 토큰 위치의 activation을 학습 가능한 값으로 둔다.   
- 학습 파라미터 수:
  \[
  |\Theta| = L \times d_{\text{model}} \times (l_p + l_i)
  \]
  여기서 \(L\)은 레이어 수. 

### Adapter tuning: AdapterH / AdapterL / AdapterP / AdapterD
- Houlsby et al.(2019) 원형 설계를 AdapterH로, Lin et al.(2020) 변형을 AdapterL로, Pfeiffer et al.(2021) 유사 설계를 AdapterP로, Rücklé et al.(2020) AdapterDrop을 AdapterD로 표기한다.   
- Adapter 학습 파라미터 수:
  \[
  |\Theta| = \hat L^{\text{Adpt}} \times (2\, d_{\text{model}}\, r + r + d_{\text{model}}) \;+\; 2 \times \hat L^{\text{LN}} \times d_{\text{model}}
  \]
  \(\hat L^{\text{Adpt}}\): adapter layer 개수, \(\hat L^{\text{LN}}\): 학습하는 LayerNorm 개수, \(r\): bottleneck 차원. 

### LoRA
- LoRA는 기존 가중치 행렬에 대해 저랭크 분해 행렬 쌍을 추가로 학습하며, 대부분 실험에서 \(W_q\), \(W_v\)에만 적용한다고 밝힌다(Section 4.2와 연결).   
- 학습 파라미터 수:
  \[
  |\Theta| = 2 \times \hat L^{\text{LoRA}} \times d_{\text{model}} \times r
  \]
  \(\hat L^{\text{LoRA}}\)는 LoRA를 적용한 가중치 행렬의 개수. 

- (각주 4) 배포 시에도 베이스 모델(예: 350GB)은 필요하지만, 태스크 100개를 저장할 때 LoRA 모듈만 추가하면 \(350\text{GB}+35\text{MB}\times 100\approx 354\text{GB}\)로 충분한 반면, full fine-tuning 모델 100개 저장은 \(\approx 35\text{TB}\)가 된다고 설명한다.   
- (각주 5) GPT-3 175B에서 full fine-tuning throughput은 32.5 tokens/s per V100 GPU, LoRA는 동일 shard 수에서 43.1 tokens/s per V100 GPU로 보고한다. 

---

## 5.2 RoBERTa base/large (GLUE)

- RoBERTa(base 125M, large 355M)를 HuggingFace Transformers에서 가져와 GLUE로 평가한다.   
- adapter와의 공정 비교를 위해 두 가지 변경을 적용한다:  
  1) 모든 태스크에서 동일 batch size, sequence length=128  
  2) MRPC/RTE/STS-B는 MNLI-adapted 모델이 아니라 pre-trained 모델에서 직접 초기화  
  이 세팅은 Houlsby et al.(2019)와 유사한 제한된 세팅이며, 해당 런에 \(†\)를 붙인다고 설명한다.   
- 하이퍼파라미터는 Appendix D.1을 참조하라고 안내한다. 

> **Table 2 삽입**

### Table 2 (GLUE 결과; RoBERTa 부분 재서술)
**지표 정의(캡션):** MNLI는 matched+mismatched 전체 정확도, CoLA는 Matthew’s correlation, STS-B는 Pearson correlation, 그 외는 accuracy. 모두 높을수록 좋음. \(*\)는 prior works 수치, \(†\)는 공정 비교 세팅. 

**RoBERTa base (trainable 125.0M: FT\(*\))**
- FT\(*\) 125.0M:  
  MNLI 87.6 / SST-2 94.8 / MRPC 90.2 / CoLA 63.6 / QNLI 92.8 / QQP 91.9 / RTE 78.7 / STS-B 91.2 / Avg 86.4  
- BitFit\(*\) 0.1M:  
  84.7 / 93.7 / 92.7 / 62.0 / 91.8 / 84.0 / 81.5 / 90.8 / Avg 85.2  
- AdapterD\(*\) 0.3M:  
  87.1±0.0 / 94.2±0.1 / 88.5±1.1 / 60.8±0.4 / 93.1±0.1 / 90.2±0.0 / 71.5±2.7 / 89.7±0.3 / Avg 84.4  
- AdapterD\(*\) 0.9M:  
  87.3±0.1 / 94.7±0.3 / 88.4±0.1 / 62.6±0.9 / 93.0±0.2 / 90.6±0.0 / 75.9±2.2 / 90.3±0.1 / Avg 85.4  
- LoRA 0.3M:  
  87.5±0.3 / 95.1±0.2 / 89.7±0.7 / 63.4±1.2 / 93.3±0.3 / 90.8±0.1 / 86.6±0.7 / 91.5±0.2 / Avg 87.2  

**RoBERTa large**
- FT\(*\) 355.0M:  
  90.2 / 96.4 / 90.9 / 68.0 / 94.7 / 92.2 / 86.6 / 92.4 / Avg 88.9  
- LoRA 0.8M:  
  90.6±0.2 / 96.2±0.5 / 90.9±1.2 / 68.2±1.9 / 94.9±0.3 / 91.6±0.1 / 87.4±2.5 / 92.6±0.2 / Avg 89.0  

**\(†\) 공정 비교 세팅(주요 행)**
- AdapterP\(†\) 3.0M:  
  90.2±0.3 / 96.1±0.3 / 90.2±0.7 / 68.3±1.0 / 94.8±0.2 / 91.9±0.1 / 83.8±2.9 / 92.1±0.7 / Avg 88.4  
- AdapterP\(†\) 0.8M:  
  90.5±0.3 / 96.6±0.2 / 89.7±1.2 / 67.8±2.5 / 94.8±0.3 / 91.7±0.2 / 80.1±2.9 / 91.9±0.4 / Avg 87.9  
- AdapterH\(†\) 6.0M:  
  89.9±0.5 / 96.2±0.3 / 88.7±2.9 / 66.5±4.4 / 94.7±0.2 / 92.1±0.1 / 83.4±1.1 / 91.0±1.7 / Avg 87.8  
- AdapterH\(†\) 0.8M:  
  90.3±0.3 / 96.3±0.5 / 87.7±1.7 / 66.3±2.0 / 94.7±0.2 / 91.5±0.1 / 72.9±2.9 / 91.5±0.5 / Avg 86.4  
- LoRA\(†\) 0.8M:  
  90.6±0.2 / 96.2±0.5 / 90.2±1.0 / 68.2±1.9 / 94.8±0.3 / 91.6±0.2 / 85.2±1.1 / 92.3±0.5 / Avg 88.6  

---

## 5.3 DeBERTa XXL (GLUE)

- DeBERTa XXL 1.5B에 대해 LoRA가 full fine-tuning 성능을 유지하는지 평가하며, 결과는 Table 2 하단에 제시한다. 하이퍼파라미터는 Appendix D.2를 참조하라고 한다. 

**DeBERTa XXL (Table 2 하단)**
- FT\(*\) 1500.0M:  
  MNLI 91.8 / SST-2 97.2 / MRPC 92.0 / CoLA 72.0 / QNLI 96.0 / QQP 92.7 / RTE 93.9 / STS-B 92.9 / Avg 91.1  
- LoRA 4.7M:  
  91.9±0.2 / 96.9±0.2 / 92.6±0.6 / 72.4±1.1 / 96.0±0.1 / 92.9±0.1 / 94.9±0.4 / 93.0±0.2 / Avg 91.3  

---

## 5.4 GPT-2 medium/large (E2E NLG Challenge)

- GPT-2 medium/large에 대해 NLG에서도 LoRA가 성립하는지 평가한다. Li & Liang(2021) 세팅을 따라 직접 비교하며, 본문에는 E2E NLG Challenge 결과(Table 3)만 제시하고 WebNLG/DART 결과는 Appendix F.1에 둔다. 하이퍼파라미터는 Appendix D.3을 참조하라고 한다. 

> **Table 3 삽입**

### Table 3 (E2E 결과; 원문 수치 재서술)
- 모든 지표는 높을수록 좋다. \(*\)는 prior works 수치. confidence interval을 표시했다고 캡션에 명시한다. 

**GPT-2 Medium (M)**
- FT\(*\) 354.92M: BLEU 68.2 / NIST 8.62 / MET 46.2 / ROUGE-L 71.0 / CIDEr 2.47  
- AdapterL\(*\) 0.37M: 66.3 / 8.41 / 45.0 / 69.8 / 2.40  
- AdapterL\(*\) 11.09M: 68.9 / 8.71 / 46.1 / 71.3 / 2.47  
- AdapterH 11.09M: 67.3±0.6 / 8.50±0.07 / 46.0±0.2 / 70.7±0.2 / 2.44±0.01  
- FTTop2\(*\) 25.19M: 68.1 / 8.59 / 46.0 / 70.8 / 2.41  
- PreLayer\(*\) 0.35M: 69.7 / 8.81 / 46.1 / 71.4 / 2.49  
- LoRA 0.35M: 70.4±0.1 / 8.85±0.02 / 46.8±0.2 / 71.8±0.1 / 2.53±0.02  

**GPT-2 Large (L)**
- FT\(*\) 774.03M: 68.5 / 8.78 / 46.0 / 69.9 / 2.45  
- AdapterL 0.88M: 69.1±0.1 / 8.68±0.03 / 46.3±0.0 / 71.4±0.2 / 2.49±0.0  
- AdapterL 23.00M: 68.9±0.3 / 8.70±0.04 / 46.1±0.1 / 71.3±0.2 / 2.45±0.02  
- PreLayer\(*\) 0.77M: 70.3 / 8.85 / 46.2 / 71.7 / 2.47  
- LoRA 0.77M: 70.4±0.1 / 8.89±0.02 / 46.8±0.2 / 72.0±0.2 / 2.47±0.02  

---

## 5.5 Scaling up to GPT-3 175B

- LoRA의 스케일 특성을 보기 위해 GPT-3 175B에 적용한다. 비용이 커서 각 엔트리의 표준편차 대신 데이터셋별 “전형적 변동폭(typical std)”만 보고한다고 밝힌다. 하이퍼파라미터는 Appendix D.4를 참조하라고 한다. 

> **Table 4 삽입**

### Table 4 (GPT-3 175B 결과; 원문 수치 재서술)
- FT 175,255.8M: WikiSQL 73.8 / MNLI-m 89.5 / SAMSum 52.0/28.0/44.5 (R1/R2/RL)  
- BitFit 14.2M: 71.3 / 91.0 / 51.3/27.4/43.5  
- PreEmbed 3.2M: 63.1 / 88.6 / 48.3/24.2/40.5  
- PreLayer 20.2M: 70.1 / 89.5 / 50.8/27.3/43.5  
- AdapterH 7.1M: 71.9 / 89.8 / 53.0/28.9/44.8  
- AdapterH 40.1M: 73.2 / 91.5 / 53.2/29.0/45.1  
- LoRA 4.7M: 73.4 / 91.7 / 53.8/29.8/45.9  
- LoRA 37.7M: 74.0 / 91.6 / 53.4/29.2/45.1  

**Table 4 변동폭(캡션):** WikiSQL ±0.5%, MNLI-m ±0.1%, SAMSum ±0.2/±0.2/±0.1 (R1/R2/RL). 

> **Figure 2 삽입**

### Figure 2 (성능 vs trainable params)
- x축: \(\log_{10}(\#\text{trainable params})\)  
- y축: validation accuracy  
- 두 패널: WikiSQL / MultiNLI-matched  
- 비교 방법: Fine-Tune, PrefixEmbed, PrefixLayer, Adapter(H), LoRA  
- 캡션은 LoRA가 더 나은 scalability와 task performance를 보인다고 요약하며, 상세 포인트는 Section F.2를 참조하라고 한다. 

- 저자들은 Prefix-embedding tuning에서 특수 토큰이 256개를 초과하면 성능이 크게 떨어지고, Prefix-layer tuning은 32개를 초과하면 성능이 크게 떨어지는 현상을 관찰했다고 보고하며, 이는 Li & Liang(2021) 관찰과 일치한다고 말한다.   
- 이 현상의 메커니즘은 본 논문 범위 밖이라면서도, 특수 토큰이 많아질수록 입력 분포가 pre-training 분포에서 멀어지는 shift가 생길 수 있다고 “추측(suspect)”한다. 저데이터 분석은 Section F.3에서 다룬다고 한다. 

## 해설

  
본 절의 비교는 “학습 가능한 파라미터 수”를 다양한 방식(바이어스/프롬프트 토큰/레이어별 프리픽스/어댑터/저랭크 업데이트)으로 배분할 때 성능이 어떻게 달라지는지를 다면적으로 보여준다. 특히 초대형 모델(GPT-3 175B)에서는 “파라미터 예산을 늘리면 성능이 단조 증가한다”는 단순 직관이 항상 성립하지 않을 수 있음을 시사한다(원문은 이를 특정 prefix 설정에서 관찰).

---

# 6 Related Works

## 핵심 논점

- 질문  
  Transformer LM의 발전, prompt engineering과 fine-tuning, parameter-efficient adaptation, 그리고 low-rank 구조 관련 문헌 속에서 LoRA의 위치는 무엇인가?

- 선수 개념  
  - GPT/BERT 계열의 사전학습–적응 패러다임  
  - adapters/prefix/prompt 계열  
  - low-rank 구조 및 관련 이론적 결과

- 읽으면서 확인할 포인트  
  1) adapter와의 구조적 유사점/차이점(특히 병합 가능성)  
  2) prefix/prompt 계열의 스케일링 제약(시퀀스 길이 점유)  
  3) 저랭크 문헌이 LoRA 가설을 어떻게 뒷받침하는지  

## 본문 정리

### (1) Transformer Language Models
- Transformer(Vaswani et al., 2017)는 self-attention 중심의 seq2seq 아키텍처이며, Radford et al.은 이를 autoregressive LM(GPT 계열)로 적용했다.   
- 이후 Transformer 기반 LM이 NLP에서 SOTA를 장악했으며, BERT/ GPT-2를 통해 “사전학습 후 태스크별 fine-tuning” 패러다임이 확립되었다.   
- GPT-3(Brown et al., 2020)는 175B 규모의 대형 단일 Transformer LM으로 언급된다. 

### (2) Prompt Engineering and Fine-Tuning
- GPT-3 175B는 few-shot로도 동작을 바꿀 수 있으나 프롬프트에 매우 의존하므로 prompt engineering이 필요하다고 서술한다.   
- fine-tuning은 사전학습 모델을 특정 태스크로 재학습하는 과정이며, 실무에서는 성능 극대화를 위해 전체 파라미터 업데이트가 자주 사용된다고 언급한다.   
- GPT-3 175B의 거대함 때문에 체크포인트 크기/하드웨어 장벽으로 일반적인 fine-tuning이 어렵다는 맥락을 다시 강조한다. 

### (3) Parameter-Efficient Adaptation
- adapter layer 삽입(Houlsby, Rebuffi, Lin 등) 계열과 비교해, LoRA도 병목 구조를 갖는 점에서 유사하다고 말한다.   
- 다만 LoRA의 학습된 가중치는 추론 시 메인 가중치에 **병합 가능**하므로, adapter 대비 추론 latency를 늘리지 않는 것이 핵심 차이라고 강조한다.   
- COMPACTER(Mahabadi et al., 2021)를 언급하며 Kronecker product 기반 파라미터화/공유 스킴을 설명한다. LoRA도 이러한 tensor product 기반 기법과 결합해 파라미터 효율을 더 높일 수 있으며 이는 future work로 남긴다고 한다.   
- 입력 임베딩(프롬프트)을 최적화하는 연구(Li & Liang, Lester et al., Hambardzumyan et al., Liu et al.)를 언급하고, 본 논문 실험에서 Li & Liang과 비교를 포함했다고 말한다.   
- 동시에 특수 토큰을 늘려 스케일 업할 때(특히 positional embedding이 학습되는 경우) 태스크 토큰에 사용할 시퀀스 길이를 잠식할 수 있다고 지적한다. 

### (4) Low-Rank Structures in Deep Learning
- low-rank 구조는 ML에서 널리 관찰되며, 과매개변수 딥넷이 학습 후 low-rank 특성을 보일 수 있다는 관찰(Oymak et al., 2019)을 인용한다.   
- 일부 연구는 원래 모델 학습 시 low-rank 제약을 명시적으로 부여하기도 했으나, 저자들이 아는 한 “동결된 모델에 대한 저랭크 업데이트로 다운스트림 적응”을 다룬 사례는 없다고 주장한다.   
- low-rank 구조가 있을 때 신경망이 NTK 등 고전 방법보다 유리할 수 있다는 이론적 결과, adversarial training에서 low-rank adaptation의 유용성(Allen-Zhu & Li, 2020b) 등을 언급하며 LoRA의 동기가 충분함을 강조한다. 

## 해설

  
Related Work는 LoRA가 PEFT 계열(특히 adapters)과 같은 “모듈 학습” 철학을 공유하면서도, 배포 관점에서 “병합을 통한 추론 경로 보존”을 차별점으로 내세운다는 점을 구조적으로 정리한다. 또한 “업데이트의 저랭크성”이 단순 휴리스틱이 아니라 기존 low-rank 문헌과 연결될 수 있음을 강조한다.

---

# 7 Understanding the Low-Rank Updates

## 핵심 논점

- 질문  
  (i) Transformer의 어떤 가중치에 LoRA를 적용할 때 효율이 좋은가?  
  (ii) LoRA rank \(r\)은 어떻게 선택하는 것이 적절한가?  
  (iii) 업데이트 \(\Delta W\)는 원래 가중치 \(W\)와 어떤 관계를 갖는가?

- 선수 개념  
  - SVD(특이값 분해), singular vectors  
  - 부분공간(subspace) 유사도  
  - Frobenius norm

- 읽으면서 확인할 포인트  
  1) 동일 파라미터 예산에서 적용 대상(\(W_q,W_k,W_v,W_o\)) 조합 비교(Table 5)  
  2) \(r\) 변화에 따른 성능 민감도(Table 6)와 부분공간 유사도 정의(식 4, Figure 3–4)  
  3) \(\Delta W\)와 \(W\)의 상관 및 증폭 양상(Table 7)  

## 본문 정리

- 저자들은 LoRA가 실험적으로 효과적임을 보인 뒤, downstream task에서 학습된 low-rank adaptation이 갖는 성질을 분석한다. 이 분석은 GPT-3 175B에서 수행되며, 해석 가능성과 효율성에 대한 추가 통찰을 제공하려는 목적을 밝힌다. 

---

## 7.1 Which weight matrices in Transformer should we apply LoRA to?

### 핵심 논점
- 질문  
  동일한 trainable parameter budget을 가정할 때, self-attention의 \(W_q, W_k, W_v, W_o\) 중 어디에 LoRA를 적용하는 것이 유리한가?

- 선수 개념  
  - self-attention projection matrices의 역할  
  - 예산 고정 하에서 rank 조정

- 읽으면서 확인할 포인트  
  1) 예산(18M) 고정 및 레이어 수(96) 고정 조건  
  2) 적용 대상 수에 따라 \(r\)을 조정하는 방식  
  3) WikiSQL/MultiNLI에서의 비교 결과(Table 5)

### 본문 정리

- 모델: GPT-3 175B  
- trainable parameter budget: 18M (FP16 저장 시 약 35MB)  
- 모든 96개 레이어에 동일 방식 적용  
- 예산을 맞추기 위해:  
  - 한 종류만 적응: \(r=8\)  
  - 두 종류 적응: \(r=4\)  
  - 네 종류 모두 적응: \(r=2\)  
- 결과는 Table 5에 제시한다. 

> **Table 5 삽입**

### Table 5 (budget=18M 고정, 원문 수치 재서술)
- WikiSQL (±0.5%)
  - \(W_q\)만: 70.4  
  - \(W_k\)만: 70.0  
  - \(W_v\)만: 73.0  
  - \(W_o\)만: 73.2  
  - \(W_q,W_k\): 71.4  
  - \(W_q,W_v\): 73.7  
  - \(W_q,W_k,W_v,W_o\): 73.7  
- MultiNLI (±0.1%)
  - \(W_q\)만: 91.0  
  - \(W_k\)만: 90.8  
  - \(W_v\)만: 91.0  
  - \(W_o\)만: 91.3  
  - \(W_q,W_k\): 91.3  
  - \(W_q,W_v\): 91.3  
  - \(W_q,W_k,W_v,W_o\): 91.7  

저자들은 \(W_q\) 또는 \(W_k\) 단독 적응 시 성능이 낮아지는 경향을 언급하며, 여러 행렬에 더 작은 \(r\)로 분산하는 구성이 유리할 수 있음을 시사한다.   
또한 ±는 랜덤 시드에 대한 표준편차로, 데이터셋별로 대략 일정하다고 설명한다. 

### 해설
  
동일 예산 하에서 “어느 변환을 조정할 것인가”는 모델이 downstream에서 재조정하는 자유도의 ‘배치 방식’을 바꾼다. 따라서 결과는 단순히 “파라미터 수”뿐 아니라 “파라미터가 들어가는 위치”가 성능에 실질적 영향을 준다는 점을 보여주는 사례로 해석할 수 있다.

---

## 7.2 What is the optimal rank \(r\) for LoRA?

### 핵심 논점
- 질문  
  LoRA rank \(r\)을 증가시킬 때 성능은 어떤 방식으로 변하는가? 적용 대상(weight set)에 따라 \(r\) 민감도는 달라지는가?

- 선수 개념  
  - \(r\)과 \(|\Theta|\)의 관계  
  - SVD 기반 부분공간 유사도(식 4)

- 읽으면서 확인할 포인트  
  1) \(r\) 스윕 실험(Table 6)  
  2) \(r\)이 다른 해의 부분공간 겹침(Figure 3)  
> **Figure 4 삽입**

  3) 동일 \(r\)에서 시드 변화의 영향(Figure 4)  
  4) 각주 6이 제시하는 일반화 한계

### 본문 정리

- 저자들은 \(r\) 영향 분석을 위해 적용 대상 weight를 세 가지로 두고 비교한다: \(W_q\)만, \(\{W_q,W_v\}\), \(\{W_q,W_k,W_v,W_o\}\). 

> **Table 6 삽입**

### Table 6 (rank 스윕; 원문 수치 재서술)
- WikiSQL (±0.5%)
  - \(W_q\): r=1 68.8 / r=2 69.6 / r=4 70.5 / r=8 70.4 / r=64 70.0  
  - \(W_q,W_v\): r=1 73.4 / r=2 73.3 / r=4 73.7 / r=8 73.8 / r=64 73.5  
  - \(W_q,W_k,W_v,W_o\): r=1 74.1 / r=2 73.7 / r=4 74.0 / r=8 74.0 / r=64 73.9  
- MultiNLI (±0.1%)
  - \(W_q\): r=1 90.7 / r=2 90.9 / r=4 91.1 / r=8 90.7 / r=64 90.7  
  - \(W_q,W_v\): r=1 91.3 / r=2 91.4 / r=4 91.3 / r=8 91.6 / r=64 91.4  
  - \(W_q,W_k,W_v,W_o\): r=1 91.2 / r=2 91.7 / r=4 91.7 / r=8 91.5 / r=64 91.4  

- (각주 6) 저자들은 작은 \(r\)이 모든 태스크/데이터셋에서 보장될 것으로 기대하지 않는다고 명시한다. 예로 downstream이 pretraining과 다른 언어라면, full retraining(LoRA에서 사실상 full rank)이 작은 \(r\)보다 유리할 수 있다는 “사고 실험”을 든다. 

> **Figure 3 삽입**

#### 부분공간 유사도 분석 (Figure 3, 4; 식 4)
- 공간 제약으로 96개 레이어 중 48번째 레이어만 본문에 싣고, 다른 레이어에서도 결론이 유사함을 Section H.1에서 보인다고 말한다. 

- normalized subspace similarity(식 4)를 다음과 같이 정의한다:
  \[
  \phi(A_{r=8}, A_{r=64}, i, j)
  = 
  \frac{\left\|\left(U^{i}_{A_{r=8}}\right)^{\top} U^{j}_{A_{r=64}}\right\|_{F}^{2}}{\min(i,j)}
  \in [0,1]
  \]
  여기서 \(U^{i}_{A_{r=8}}\)는 \(A_{r=8}\)의 singular vectors 중 top-i, \(U^{j}_{A_{r=64}}\)는 \(A_{r=64}\)의 singular vectors 중 top-j에 해당한다. \(\phi=1\)이면 부분공간이 동일하고, \(\phi=0\)이면 분리된 것으로 해석한다. 

- Figure 3에서 저자들은 top singular 방향의 겹침이 크고, 다른 방향의 겹침은 작을 수 있음을 관찰하며, 이러한 양상이 작은 \(r\)에서도 성능이 유지되는 현상을 설명하는 데 도움이 된다고 논의한다.   
- Figure 4에서는 동일 \(r=64\)에서 서로 다른 시드로 학습한 결과를 비교하고, 랜덤 Gaussian 행렬 대비 유의한 공통 방향이 관찰됨을 보여준다. 또한 \(\Delta W_q\)가 \(\Delta W_v\)보다 더 높은 intrinsic rank를 갖는 듯한 패턴을 논의하며, 이는 \(W_q\) 단독 적응에서 더 큰 \(r\)이 필요했던 경험적 관찰과 일치한다고 연결한다. 

### 해설
  
부분공간 유사도 분석은 “큰 \(r\)에서 추가로 얻는 자유도(추가 방향)가 실제로 일관된 신호인지”를 간접적으로 점검하는 도구로 읽을 수 있다. 논문은 특정 설정(GPT-3 175B, 특정 레이어 및 태스크)에서 top 방향의 공유가 상대적으로 크고, 나머지 방향은 덜 안정적일 수 있음을 관찰한다.

---

## 7.3 How does the adaptation matrix \(\Delta W\) compare to \(W\)?

### 핵심 논점
- 질문  
  \(\Delta W\)는 원래 가중치 \(W\)와 어느 정도 상관되어 있으며, \(W\)의 어떤 방향을 변화(증폭)시키는 경향이 있는가?

- 선수 개념  
  - projection과 Frobenius norm  
  - “top singular directions”와 “비강조 방향”의 구분

- 읽으면서 확인할 포인트  
  1) \(\Delta W\)에서 얻은 부분공간으로 \(W\)를 투영하는 실험 설계  
  2) 랜덤 대비 상관의 크기  
  3) 증폭 계수 예시(Table 7) 및 부록(H.4) 참조 안내

### 본문 정리

- 저자들은 \(\Delta W\)의 SVD에서 얻은 \(U,V\)를 사용해 \(W\)를 해당 \(r\)-차원 subspace로 projection한 \(U^\top W V^\top\)의 Frobenius norm을 측정하고, \(\|W\|_F\) 및 \(\|\Delta W\|_F\)와 비교한다.   
- 비교군으로는 (i) \(U,V\)를 \(\Delta W\)에서 가져오는 경우, (ii) \(U,V\)를 \(W\)의 top-r singular vectors로 쓰는 경우, (iii) 랜덤 행렬에서 가져오는 경우를 둔다.   
- (각주 7) 유사 분석을 \(B\)와 left singular unitary matrices로도 할 수 있으나, 본문 실험은 \(A\)에 대해 수행했다고 언급한다. 

> **Table 7 삽입**

### Table 7 (48th layer의 \(W_q\) 분석; 원문 수치 재서술)
- r=4:
  - \(\|U^\top W_q V^\top\|_F\)
    - \(U,V\) from \(\Delta W_q\): 0.32  
    - \(U,V\) from \(W_q\): 21.67  
    - random: 0.02  
  - \(\|W_q\|_F = 61.95\)  
  - \(\|\Delta W_q\|_F = 6.91\)  
- r=64:
  - \(\|U^\top W_q V^\top\|_F\)
    - \(\Delta W_q\): 1.90  
    - \(W_q\): 37.71  
    - random: 0.33  
  - \(\|\Delta W_q\|_F = 3.57\)  

- 저자들은 \(\Delta W\)가 랜덤보다 \(W\)와 더 강한 상관을 가지며, pretraining에서 학습되었으나 강하게 사용되지 않았던 방향을 downstream에 맞게 증폭할 수 있음을 시사한다고 논의한다.   
- r=4에서 증폭 계수 예로 \(21.5 \approx 6.91/0.32\)를 제시하며, r=64에서 증폭이 더 작아지는 이유는 Section H.4에서 설명한다고 덧붙인다. 

## 해설

  
본 분석은 \(\Delta W\)가 단순히 \(W\)의 “가장 큰 에너지 방향(top singular directions)”을 반복하는 것이 아니라, \(W\)에서 상대적으로 덜 강조된 방향을 과제에 맞게 재가중(증폭)할 수 있음을 정량적으로 뒷받침하려는 시도로 해석할 수 있다.

---

# 8 Conclusion and Future Work

## 핵심 논점

- 질문  
  LoRA가 제안하는 핵심 요지(효율적 적응과 배포 관점 장점)는 무엇이며, 저자들이 제시하는 후속 연구 과제는 무엇인가?

- 선수 개념  
  - PEFT의 배포/운영 상 제약  
  - “결합 가능성”과 “메커니즘 분석” 문제의식

- 읽으면서 확인할 포인트  
  1) 결론에서 재강조되는 LoRA의 장점  
  2) future work로 제시되는 4가지 방향과 그 의미

## 본문 정리

- 대규모 LM의 full fine-tuning은 하드웨어 비용뿐 아니라 태스크별 독립 모델 저장/스위칭에 따른 운영 비용이 크다고 다시 강조한다.   
- LoRA는 추가 inference latency를 유발하지 않고, 입력 시퀀스 길이를 줄이지 않으며, 모델 품질을 유지하는 효율적 적응 전략으로 요약된다.   
- 배포 관점에서 베이스 파라미터는 공유하고 태스크별로 작은 LoRA 모듈만 교체하는 방식으로 빠른 task-switching을 가능하게 한다고 강조한다.   
- 본 논문은 Transformer LM에 집중했으나, 원리는 일반적인 dense layer 기반 신경망에도 적용 가능하다고 언급한다. 

- Future work로 다음 네 가지를 제시한다. 
  1) LoRA는 다른 효율적 적응 방법들과 결합 가능(서로 직교적 개선 가능성).  
  2) fine-tuning/LoRA의 메커니즘은 여전히 불명확하며, LoRA는 full fine-tuning보다 분석을 용이하게 만들 수 있다.  
  3) LoRA 적용 weight 선택은 주로 휴리스틱에 의존했는데, 더 원칙적인 선택 방법의 가능성.  
  4) \(\Delta W\)의 rank-deficiency는 \(W\) 자체도 rank-deficient일 가능성을 시사하며, 향후 연구 영감이 될 수 있음.

## 해설

  
결론은 LoRA를 “단순한 경량화 기법”으로만 제시하기보다, (i) 배포·운영에서의 구조적 장점과 (ii) 적응 메커니즘 분석을 위한 관점(저랭크 업데이트의 해석 가능성)을 함께 강조하는 형태로 구성되어 있다.

---

# Appendix A. Large Language Models Still Need Parameter Updates

## 핵심 논점

- 질문  
  few-shot/prompt만으로 충분한가, 아니면 여전히 파라미터 업데이트(fine-tuning)가 필요한가?

- 선수 개념  
  - few-shot in-context learning  
  - fine-tuning과 성능 차이의 의미

- 읽으면서 확인할 포인트  
  1) few-shot 세팅(예: MNLI-m demo 개수)  
  2) Table 8 수치 비교

## 본문 정리

- 저자들은 few-shot learning이 샘플이 매우 적을 때 유리하다는 점을 인정하나, 실무에서는 성능 민감한 애플리케이션을 위해 수천 개 이상의 샘플을 준비할 수 있는 경우가 많다고 서술한다.   
- Table 8로 데이터 규모와 무관하게 fine-tuning이 few-shot 대비 큰 성능 향상을 보인다고 주장한다.   
- RTE의 GPT-3 few-shot 결과는 Brown et al.(2020)에서 가져왔다고 명시한다.   
- MNLI-matched few-shot은 클래스당 2개 demonstration, 총 6개 in-context examples를 사용했다고 설명한다. 

> **Table 8 삽입**

### Table 8 (원문 수치)
- GPT-3 Few-Shot: MNLI-m 40.6 / RTE 69.0  
- GPT-3 Fine-Tuned: MNLI-m 89.5 / RTE 85.4  

## 해설

  
Appendix A는 “prompt만으로 충분하므로 파라미터 업데이트가 불필요하다”는 관점을 반박하는 근거로 기능한다. LoRA와 같은 PEFT의 실용적 동기(업데이트는 필요하되 비용을 낮추자)를 강화한다.

---

# Appendix B. Inference Latency Introduced by Adapter Layers

## 핵심 논점

- 질문  
  adapter layer는 추론 latency를 어느 정도 증가시키는가? 이 증가가 어떤 조건에서 두드러지는가?

- 선수 개념  
  - sequential vs parallel 추가 모듈의 차이  
  - 배치/시퀀스 길이에 따른 GPU 병렬성

- 읽으면서 확인할 포인트  
  1) 측정 조건(하드웨어, trial 수, 측정 대상)  
> **Figure 5 삽입**

  2) batch size/sequence length/bottleneck \(r\) 변화에 따른 slowdown 패턴(Figure 5)

## 본문 정리

- adapter는 pre-trained 모델에 순차적으로 추가되는 외부 모듈이고, LoRA는 외부 모듈이지만 기존 계산과 병렬로 볼 수 있다고 설명한다. 따라서 adapter는 base 모델 계산에 추가 연산이 더해져 latency가 불가피하다고 주장한다.   
- Rücklé et al.(2020)의 관찰(큰 batch/sequence에서는 병렬성으로 latency가 완화될 수 있음)을 언급하며, GPT-2 medium에서 유사 실험을 수행한다고 말한다. 특히 온라인 추론(작은 배치)에서는 latency 증가가 유의미할 수 있음을 강조한다. 

- 실험 설정:
  - 하드웨어: NVIDIA Quadro RTX8000  
  - 측정: single forward pass latency, 100 trials 평균  
  - 변수: batch size, sequence length, adapter bottleneck dimension \(r\)  
  - 비교: AdapterH(Houlsby et al., 2019), AdapterL(Lin et al., 2020)  
  - 결과: no-adapter baseline(r=0) 대비 percentage slow-down을 Figure 5로 제시  
  

- Figure 5 캡션은 top row가 AdapterH, bottom row가 AdapterL임을 밝히며, 온라인/짧은 시퀀스 시나리오에서 slowdown이 30%를 넘을 수 있음을 언급한다. 

## 해설

  
Appendix B는 “adapter가 느리다”는 정성적 주장 대신, 병렬성이 충분히 활용되는 조건과 그렇지 않은 조건(특히 작은 배치/짧은 시퀀스)에서의 오버헤드가 어떻게 달라지는지 정량적으로 제시한다.

---

# Appendix C. Dataset Details

## 핵심 논점

- 목적  
  본문 실험에서 사용한 데이터셋의 정의/크기/입출력 인코딩/라이선스를 정리한다.

- 읽으면서 확인할 포인트  
  1) WikiSQL/SAMSum의 \(x/y\) 구성(컨텍스트/타깃 인코딩)  
  2) 데이터셋 크기(훈련/검증/테스트)  
  3) 라이선스(특히 비상업 조항 등)

## 본문 정리

### C.1 GLUE Benchmark
- GLUE는 MNLI, SST-2, MRPC, CoLA, QNLI, QQP, RTE, STS-B를 포함하는 NLU 태스크 모음으로 나열된다.   
- 폭넓은 커버리지로 NLU 모델 평가의 표준 지표로 사용된다고 설명한다.   
- 각 데이터셋은 서로 다른 permissive license로 배포된다고 덧붙인다.   
- 각주 8로 QQP(Quora Question Pairs) 관련 링크를 제공한다. 

### C.2 WikiSQL
- Zhong et al.(2017) 데이터셋.   
- 규모: 56,355 train / 8,421 validation   
- 태스크: 자연어 질문 + 테이블 스키마로부터 SQL 생성   
- 인코딩:
  - \(x=\{\text{table schema, query}\}\)
  - \(y=\{\text{SQL}\}\)
    
- 라이선스: BSD 3-Clause 

### C.3 SAMSum
- Gliwa et al.(2019) 데이터셋.   
- 규모: 14,732 train / 819 test   
- 내용: 2인 대화 + 언어학자 작성 abstractive summary   
- 인코딩:
  - 컨텍스트: 발화를 `"\n"`로 연결 후 `"\n\n"` 추가
  - \(y=\{\text{summary}\}\)
    
- 라이선스: Creative Commons BY-NC-ND 4.0 (비상업) 

### C.4 E2E NLG Challenge
- Novikova et al.(2017) data-to-text 데이터셋.   
- 규모(대략): 42,000 train / 4,600 validation / 4,600 test, 레스토랑 도메인   
- 입력: slot-value pair 시퀀스, 출력: 자연어 reference text   
- 각 입력은 multiple references를 가질 수 있다고 설명한다.   
- 라이선스: CC BY-NC-SA 4.0 

### C.5 DART
- Nan et al.(2020) 데이터셋.   
- 입력 구조: ENTITY—RELATION—ENTITY triples 시퀀스   
- 전체 82K examples, E2E보다 크고 복잡한 data-to-text 태스크   
- 라이선스: MIT 

### C.6 WebNLG
- Gardent et al.(2017) 데이터셋.   
- 전체 22K examples, 14개 카테고리   
- 14개 중 9개는 train에서 보이며 5개는 train에서 보지 못하지만 test에 존재하므로, seen(S)/unseen(U)/all(A)로 나누어 평가하는 것이 일반적이라고 설명한다.   
- 입력 구조: SUBJECT—PROPERTY—OBJECT triples 시퀀스   
- 라이선스: CC BY-NC-SA 4.0 

## 해설

  
Appendix C는 재현 관점에서 특히 중요하다. 동일한 “태스크 이름”이라도 컨텍스트/타깃 인코딩(특히 WikiSQL, SAMSum)은 모델 입력 형식과 직접 연결되며, 라이선스는 데이터 활용 가능 범위를 결정한다.

---
