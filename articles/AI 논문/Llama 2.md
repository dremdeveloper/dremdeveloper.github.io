---
title: "Llama 2: Open Foundation and Fine-Tuned Chat Models"
math: true
---

# Llama 2: Open Foundation and Fine-Tuned Chat Models

## Abstract

이 논문은 7B부터 70B까지의 규모를 갖는 사전학습 언어모델 계열과, 그 위에 대화형 정렬을 적용한 미세조정 모델 계열을 함께 제시한다. 저자들은 전자를 **Llama 2**, 후자를 **Llama 2-Chat**으로 구분하며, 두 계열을 하나의 연속된 시스템으로 설명한다. 핵심 주장은 세 가지다. 첫째, 공개형 foundation model로서도 경쟁력 있는 성능을 확보했다는 점이다. 둘째, 대화형 미세조정과 RLHF, 안전성 조정을 거친 Llama 2-Chat이 당시의 공개형 대화 모델들을 대부분의 평가에서 앞선다는 점이다. 셋째, 사람 평가 기준으로는 일부 비공개 모델을 일정 범위에서 대체할 가능성까지 보인다는 점이다.

초록은 모델 공개 자체보다 **어떻게 미세조정하고 어떻게 안전성을 개선했는가**를 비교적 자세히 공개하는 데 무게를 둔다. 저자들은 후속 연구자와 실무자가 이 레시피를 바탕으로 더 책임감 있게 모델을 개발할 수 있도록, fine-tuning 절차와 safety 개선 과정을 상세히 기술하겠다고 밝힌다.

<img width="879" height="511" alt="image" src="https://github.com/user-attachments/assets/0e5705ec-8e8a-4a1a-acd9-71b395fe21c1" />




## 1. Introduction

저자들은 대규모 언어모델이 프로그래밍, 창작, 지식 기반 질의응답, 보조적 추론 등 여러 작업에서 강한 잠재력을 보여 왔고, 채팅 인터페이스를 통해 대중적으로도 빠르게 확산되었다고 설명한다. 학습 방식 자체는 개념적으로 단순하다. 먼저 대규모 자기지도 말뭉치로 자기회귀 트랜스포머를 사전학습한 뒤, 그 모델을 사람 선호에 맞도록 미세조정하고 강화학습으로 정렬한다. 문제는 이 과정의 비용이 매우 크다는 데 있다. 거대한 연산 자원, 장기간의 데이터 큐레이션, 인력 기반 라벨링이 필요하기 때문에 실제 제품 수준의 대화형 LLM을 구축하는 주체는 소수 조직으로 제한되어 왔다.

이 배경에서 저자들은 공개형 사전학습 모델과 제품형 대화 모델 사이의 간극을 강조한다. BLOOM, LLaMA 1, Falcon 같은 공개형 모델은 foundation model 기준으로는 강했지만, ChatGPT, Bard, Claude 같은 제품형 대화 모델을 곧바로 대체할 수준의 사용성·안전성을 제공하지는 못했다는 것이다. 저자들이 보기에 그 이유는 단순히 파라미터 수의 차이가 아니라, **정렬 절차의 비공개성**과 **고비용의 미세조정 운영**에 있다. 실제 제품 모델은 대규모 인간 선호 데이터, 반복적 RLHF, 안전성 전용 튜닝, 레드팀 운영을 거치지만, 이러한 절차는 충분히 공개되지 않았고 재현 역시 쉽지 않았다.

이 논문은 그 간극을 줄이기 위해 Llama 2와 Llama 2-Chat을 함께 제시한다. 저자들은 7B, 13B, 34B, 70B 크기의 모델을 학습했고, 이 가운데 7B, 13B, 70B를 연구 및 상업적 사용을 위해 공개한다. 34B는 논문 안에서는 보고되지만, 공개 시점에는 충분한 레드팀 검증 시간이 확보되지 않아 배포 대상에서 제외되었다. 또한 평가와 안전성 검증의 중심은 영어이며, 모든 사용 시나리오를 포괄하지 못하므로 실제 배포 전에는 개발자가 자신의 환경에 맞는 별도 테스트와 추가 조정을 수행해야 한다고 못박는다.


<img width="875" height="521" alt="image" src="https://github.com/user-attachments/assets/49d4720f-d673-43e4-a8dd-26367e451b72" />



도입부의 그림들은 이후 본문 전체의 문제 구성을 압축해서 보여준다. Figure 1은 사람 평가 기준의 도움성을, Figure 2는 GPT-4를 판정자로 삼았을 때의 도움성·안전성 비교를, Figure 3은 적대적 프롬프트 기반 안전성 평가를 요약한다. 저자들은 사람 평가가 본질적으로 노이즈가 있고, 평가 프롬프트와 기준 자체도 특정 모델에 유리할 수 있다는 점을 직접 언급하면서 결과를 해석할 것을 요청한다.

<img width="883" height="527" alt="image" src="https://github.com/user-attachments/assets/d0a0c4f5-026a-49bf-ad4f-ce4c3d1fa6bb" />



Figure 4는 논문의 전체 학습 파이프라인을 요약한다. 먼저 공개 온라인 소스로 Llama 2를 사전학습하고, 그 위에 SFT를 적용해 초기 chat 모델을 만든 뒤, rejection sampling과 PPO를 포함한 RLHF를 반복 적용한다. 중요한 점은 정책 모델이 반복적으로 업데이트되는 동안, 보상모델을 위한 사람 선호 데이터도 병렬로 계속 수집되어야 한다는 것이다. 그렇지 않으면 보상모델이 더 이상 현재 정책이 생성하는 응답 분포를 제대로 판별하지 못하게 된다.

## 2. Pretraining

이 절은 Llama 2의 foundation model이 Llama 1 대비 무엇이 달라졌는지, 어떤 데이터와 하드웨어, 어떤 최적화 세팅을 사용했는지, 그리고 그 결과 어떤 기준 성능을 확보했는지를 설명한다. 저자들은 기본적으로 Llama 1의 학습 방식에서 출발하지만, 더 강건한 데이터 정제, 바뀐 데이터 혼합, 더 많은 사전학습 토큰, 더 긴 컨텍스트, 그리고 대형 모델에서의 GQA 도입을 통해 전반적인 성능을 끌어올렸다고 밝힌다.

### 2.1 Pretraining Data

사전학습 데이터는 공개적으로 이용 가능한 소스만으로 구성되며, Meta의 제품이나 서비스 데이터는 포함하지 않는다. 또한 저자들은 사적인 개인정보가 과도하게 담겼을 가능성이 높은 일부 웹 소스를 제외하려고 노력했다고 말한다. 단순히 데이터 양을 늘리는 데 그치지 않고, 모델이 보다 사실적인 지식을 학습하고 환각을 덜 일으키도록 상대적으로 factual한 소스를 업샘플링했다는 점도 강조한다.

총 사전학습 토큰 수는 2조 개다. 저자들은 이것을 성능과 학습 비용 사이의 합리적 절충으로 제시한다. 이 절 말미에서는 데이터 구성과 편향, 독성, 언어 분포 같은 보다 구체적인 조사 결과를 Section 4.1에서 다시 다룰 것이라고 예고한다.


### 2.2 Training Details

아키텍처는 전반적으로 Llama 1과 동일한 계열을 유지한다. 표준 Transformer에 RMSNorm 기반 pre-normalization, SwiGLU 활성화, RoPE를 사용한다. 구조적 차이로는 컨텍스트 길이를 2k에서 4k로 늘린 점, 그리고 34B와 70B 모델에 Grouped-Query Attention을 도입한 점이 핵심이다. 토크나이저는 Llama 1과 동일하게 SentencePiece BPE를 사용하고, 숫자는 자릿수 단위로 쪼개며, 알 수 없는 UTF-8 문자는 바이트 수준으로 분해한다. 어휘 크기는 32k다.

최적화는 AdamW로 수행된다. 저자들이 제시한 설정은 \(\beta_1 = 0.9\), \(\beta_2 = 0.95\), \(\epsilon = 10^{-5}\)이며, cosine learning rate schedule과 2,000 step warmup을 사용한다. 학습률은 마지막에 최대값의 10%까지 감쇠하고, weight decay는 0.1, gradient clipping은 1.0으로 둔다.


<img width="868" height="319" alt="image" src="https://github.com/user-attachments/assets/77d6395e-cc9c-4015-8674-1f80c472ff3c" />



<img width="868" height="461" alt="image" src="https://github.com/user-attachments/assets/4022d73f-f6c6-4713-bbf1-54dc0e2d9a7d" />


Table 1은 Llama 1과 Llama 2를 한눈에 비교하게 해 준다. Llama 2는 7B, 13B, 34B, 70B 크기로 학습되었고, 사전학습 토큰 수는 2조 개, 컨텍스트 길이는 4k다. 모든 모델은 전역 배치 크기 4M 토큰으로 학습되며, GQA는 34B와 70B에만 적용된다. Figure 5는 이러한 설정 아래에서의 학습 손실 곡선을 보여 주며, 2조 토큰까지도 손실이 뚜렷하게 포화되지 않았다는 점을 시사한다.


#### 2.2.1 Training Hardware and Carbon Footprint

저자들은 Meta의 Research Super Cluster와 내부 생산 클러스터를 활용해 학습을 수행했다고 설명한다. 하드웨어는 A100 계열 GPU 기반이며, 모델 크기에 따라 사용한 인터커넥트 구성이 다르다. 소프트웨어 측면에서는 이 환경에 맞춘 자체 학습 스택을 사용했다. 이 절은 단순히 속도만이 아니라, 모델 학습이 요구하는 인프라 규모와 탄소 배출의 문제를 함께 공개한다.


<img width="883" height="251" alt="image" src="https://github.com/user-attachments/assets/0ab2db94-0ff6-4124-8366-c6f182922d2a" />



Table 2는 각 모델 학습에 필요한 총 GPU 시간, 추정 전력 소비, 탄소 배출량을 정리한다. 저자들은 이 배출량을 100% 상쇄했다고 밝힌다.


### 2.3 Llama 2 Pretrained Model Evaluation

사전학습이 끝난 base model은 다양한 학술 벤치마크에서 평가된다. 저자들은 여러 개별 과제를 묶은 집계 지표와 개별 벤치마크 결과를 통해, Llama 2가 공개형 base model들 사이에서 강한 경쟁력을 확보했다고 보고한다. 특히 70B 모델은 같은 범주의 공개형 모델과 비교할 때 전반적으로 가장 높은 수준의 성능을 보이는 축에 속한다.

다만 저자들은 공개형 모델끼리의 비교와 비공개 모델과의 비교를 분리해서 제시한다. 일부 영역에서는 GPT-3.5 수준에 접근하거나 비슷한 결과를 보이기도 하지만, GPT-4나 PaLM-2-L 같은 대형 비공개 모델과는 여전히 격차가 남아 있다는 점을 숨기지 않는다.

Table 3은 이 비교를 코드, 상식·추론, 세계 지식, 읽기 이해, 수학, 집계형 학술 벤치마크로 묶어 보여 준다. 논문 본문이 강조하는 메시지는 단순하다. 공개형 base model들끼리 놓고 보면 Llama 2는 크기가 커질수록 거의 모든 묶음 과제에서 고르게 상승하고, 특히 70B가 가장 균형 잡힌 상단 성능을 낸다. 저자들은 Llama 2 70B가 Llama 1 65B 대비 MMLU에서 약 5점, BBH에서 약 8점 정도 올라간다고 요약한다. 이 변화는 단순히 파라미터가 커졌기 때문이라기보다, 2조 토큰 학습, 4k 문맥, GQA, 데이터 혼합 조정이 한꺼번에 작용한 결과로 읽는 편이 맞다.

Table 4는 그 결과를 GPT-3.5, GPT-4, PaLM, PaLM-2-L과 나란히 놓아 상대적 위치를 드러낸다. 여기서 논문은 과장하지 않는다. Llama 2 70B는 MMLU와 GSM8K에서 GPT-3.5에 가까운 수치를 보이고, PaLM 540B와는 상당수 항목에서 비슷하거나 더 나은 결과를 내지만, HumanEval 같은 코딩 과제와 GPT-4·PaLM-2-L과의 격차는 여전히 뚜렷하다. 즉, foundation model 자체는 공개형 생태계 최상위권에 올라왔지만, 폐쇄형 최상위 모델 전체를 대체했다고 말할 단계는 아니라는 점을 이 절에서 분명히 한다.


<img width="863" height="338" alt="image" src="https://github.com/user-attachments/assets/26fff03b-d8f6-4150-9bc2-7a18c25c2f40" />



<img width="866" height="240" alt="image" src="https://github.com/user-attachments/assets/f68b4285-17a2-4444-8bc9-f44db72626cf" />


## 3. Fine-tuning

이 절부터는 foundation model을 대화형 모델로 바꾸는 과정이 다뤄진다. 저자들은 Llama 2 base를 바로 제품형 대화 모델로 취급하지 않고, supervised fine-tuning과 RLHF, 그리고 시스템 메시지 일관성 보강을 통해 별도의 chat model 계열을 만든다.

### 3.1 Supervised Fine-Tuning (SFT)

SFT 단계의 목표는 모델을 기본적인 대화 형식과 지시 수행 스타일에 맞추는 것이다. 저자들은 공개 instruction-tuning 데이터에서 출발해 자체 데이터로 확장했지만, 이 절에서 반복해서 강조하는 점은 **양보다 질**이다. 수백만 개의 저품질 예시보다, 상대적으로 적더라도 잘 검수된 고품질 주석이 훨씬 유리하다고 보고한다. 실제로 저자들은 수만 건 수준의 고품질 내부 주석만으로도 충분히 강한 SFT 모델을 만들 수 있었다고 설명한다.

최종적으로는 27,540개의 SFT 주석에서 멈춘다. 이 데이터 역시 Meta 제품 사용자 데이터가 아니라, 별도로 설계된 주석 수집 절차에서 나온 것이다. 학습은 전체 시퀀스 길이 4,096, 배치 크기 64, learning rate \(2\times 10^{-5}\), weight decay 0.1로 2 epoch 수행한다. 프롬프트와 응답을 이어 붙여 학습하되, 손실은 응답 토큰에만 주고 프롬프트 토큰에는 주지 않는다.


<img width="865" height="453" alt="image" src="https://github.com/user-attachments/assets/c7c64931-3625-4179-ab02-6bdd38665629" />



Table 5는 도움성 중심 예시와 안전성 중심 예시가 SFT에서 어떤 형태로 작성되는지를 보여 준다. 저자들은 적절한 답을 직접 써 내려가는 이 단계가 모델의 기본 말투, 응답 구조, 거절 방식, 안전 응답의 틀을 형성한다고 본다.

여기서 중요한 것은 SFT가 단순한 instruction-following 예열 단계가 아니라는 점이다. 논문에 실린 도움성 예시는 시 쓰기처럼 형식과 어조를 동시에 만족해야 하는 작업이고, 안전성 예시는 모욕적이거나 공격적인 요청을 정중하지만 분명하게 거절하면서 대화의 방향을 바꾸는 작업이다. 따라서 SFT는 모델에게 “무엇을 답할 것인가”만이 아니라 “어떤 목소리로 답할 것인가”를 주입한다. 저자들이 공개 instruction data로 시작하되 최종적으로는 27,540개의 고품질 내부 주석으로 멈춘 이유도 바로 여기에 있다. 대화 모델이 처음 사용자에게 보여 주는 톤, 형식, 정렬된 기본 습관은 이 단계에서 거의 결정되기 때문이다.


### 3.2 Reinforcement Learning with Human Feedback (RLHF)

SFT 이후에는 사람 선호를 더 직접적으로 반영하는 RLHF 단계가 이어진다. 저자들은 이 과정을 단발성으로 보지 않고, 정책 모델과 보상모델을 함께 반복적으로 갱신하는 루프로 설계한다.

#### 3.2.1 Human Preference Data Collection

사람 선호 데이터 수집의 기본 형식은 이진 비교다. 같은 프롬프트에 대해 두 개의 후보 응답을 보여 주고, 어느 쪽이 더 나은지 고르게 한다. 저자들은 여기서 단순히 승패만 모으지 않고, **얼마나 더 나은지**까지 등급화한다. 예를 들어 현저히 더 낫다, 더 낫다, 약간 더 낫다, 거의 차이가 없거나 잘 모르겠다는 식의 세분화된 선호 강도를 함께 기록한다. 이 정보는 뒤의 보상모델 손실 함수에서 활용된다.

선호 데이터는 도움성과 안전성을 분리해 수집한다. 안전성 데이터에서는 단순 선호뿐 아니라 어느 응답이 안전하고 어느 응답이 위험한지, 혹은 둘 다 안전한지 같은 구조도 기록한다. 또 데이터는 한 번에 끝나는 것이 아니라 주차별 배치로 계속 누적된다. 모델이 좋아질수록 비교 대상 응답의 품질 차이가 줄어들고, 따라서 더 어려운 비교가 늘어나므로, 보상모델 역시 최신 응답 분포에 맞게 계속 갱신되어야 한다.

Table 6은 이 설계가 데이터 통계에 어떻게 반영되는지 보여 준다. 보상모델 학습에 쓰인 전체 비교쌍은 2,919,326개이고, 그중 Meta가 자체 수집한 Safety & Helpfulness 비교는 1,418,091개다. 특히 내부 데이터는 평균 3.9턴, 예제당 평균 798.5토큰으로 공개 선호 데이터보다 훨씬 대화형에 가깝다. 논문이 공개 데이터만으로 보상모델을 끝내지 않고 매주 새 비교를 계속 모은 이유가 바로 이 지점에 있다. 요약이나 포럼 답변에서 잘 작동하는 보상모델만으로는, 점점 더 좋아지는 최신 chat model의 분포를 따라잡기 어렵기 때문이다.


#### 3.2.2 Reward Modeling

보상모델은 하나가 아니라 둘이다. 저자들은 도움성과 안전성을 하나의 스칼라 보상으로 합치기보다, **helpfulness reward model**과 **safety reward model**을 분리한다. 이는 두 목표가 실제로 충돌할 수 있다고 보기 때문이다. 도움성 보상모델에는 선호 강도를 반영한 margin 항을 추가한 ranking loss를 쓰고, 안전성 보상모델에는 안전 관련 보조 손실을 추가로 고려한다. 학습에는 공개 선호 데이터와 내부 수집 데이터를 함께 사용하되, 과적합을 막기 위해 보상모델은 대체로 1 epoch만 학습한다.

저자들은 최종 보상모델이 내부 테스트셋에서 강한 성능을 내며, 선호 차이가 클수록 정확도도 높아진다고 보고한다. 또한 데이터 양과 모델 크기를 키울수록 정확도가 대체로 좋아지는 스케일링 경향이 나타난다고 설명한다.

보상모델의 핵심 수식도 이 절에서 분명히 제시된다. 프롬프트와 이전 문맥을 \(x\), 후보 응답을 \(y\), 보상모델 출력을 \(r_\theta(x,y)\) 로 두고, 사람이 선택한 응답을 \(y_c\), 탈락한 응답을 \(y_r\) 라고 두면 기본 pairwise ranking loss는 다음과 같다.

\[
L_{\mathrm{rank}}(\theta)
=
-\log \sigma \bigl(r_\theta(x,y_c)-r_\theta(x,y_r)\bigr).
\]

이 식은 선택된 응답의 점수가 탈락한 응답보다 커지도록 강제하는 가장 기본적인 보상모델 목적식이다. 점수 차이가 크게 벌어지면 손실이 작아지고, 둘의 순서가 뒤집히거나 차이가 작으면 손실이 커진다.

도움성 보상모델에서는 여기에 선호 강도 정보를 더 넣는다. 논문은 “현저히 더 낫다”, “더 낫다”, “약간 더 낫다”, “거의 비슷하거나 잘 모르겠다” 같은 4단계 등급을 margin으로 바꾸어 다음 식을 사용한다.

\[
L_{\mathrm{margin}}(\theta)
=
-\log \sigma \bigl(r_\theta(x,y_c)-r_\theta(x,y_r)-m(\rho)\bigr),
\]

여기서 \(m(\rho)\) 는 선호 강도 \(\rho\) 에 대응하는 이산 margin이다. 응답 차이가 분명한 쌍에는 큰 margin을, 거의 비슷한 쌍에는 작은 margin을 둠으로써 보상모델이 “얼마나 더 좋은가”까지 배운다. 부록의 Table 27과 Table 28은 이 margin이 특히 구분이 쉬운 쌍에서 정확도를 올린다는 점을 보여 준다.

Table 7의 수치도 이 설계를 뒷받침한다. Helpfulness RM은 Meta Helpfulness 테스트셋에서 63.2, Meta Safety에서도 62.8을 기록하고, Safety RM은 Meta Safety에서 64.5를 기록한다. Table 8을 보면 두 보상모델 모두 “significantly better”처럼 차이가 큰 비교쌍에서는 훨씬 높은 정확도를 보이고, “negligibly better / unsure” 구간으로 갈수록 성능이 내려간다. 즉 보상모델은 사람이 보아도 분명한 품질 차이를 가장 잘 학습한다. Figure 6은 이런 정확도가 아직 포화되지 않았음을 보여 준다. 모델이 커지고 데이터가 많아질수록 성능이 계속 오르기 때문이다.


<img width="867" height="439" alt="image" src="https://github.com/user-attachments/assets/30b1f1fd-7365-47e6-a5c0-76bec7c550db" />


<img width="895" height="536" alt="image" src="https://github.com/user-attachments/assets/9b9accfc-c9a5-4240-96af-f5784129d687" />




<img width="812" height="332" alt="image" src="https://github.com/user-attachments/assets/0398d611-32b8-4490-9a41-d1b4054b1d75" />



#### 3.2.3 Iterative Fine-Tuning

보상모델이 준비되면 정책 모델은 반복적인 RLHF 루프로 업데이트된다. 저자들은 rejection sampling과 PPO를 함께 사용한다. 먼저 여러 개의 후보 응답을 샘플링하고, 보상모델로 점수를 매겨 가장 좋은 응답을 골라 다시 미세조정한다. 이후에는 PPO를 통해 정책을 직접 최적화한다. 이때 보상은 단순히 도움이 되는 응답을 선호하는 것만이 아니라, 안전하지 않은 프롬프트에 대해서는 안전성을 우선하는 식으로 구성된다.

이 절에서 저자들이 중요한 실무 관찰로 보고하는 것 중 하나는 **망각 문제**다. 초기 반복에서는 직전 버전에서 뽑은 최상위 응답만 다음 단계 학습에 넣었더니, 특정 능력이 후퇴하는 현상이 나타났다. 예시로는 운율 있는 시 쓰기 능력 저하가 언급된다. 이를 완화하기 위해 후속 반복에서는 가장 최근 버전뿐 아니라 이전 반복들에서 얻은 상위 응답도 함께 유지하는 방식을 사용했다.

또 다른 관찰은 RLHF가 샘플링 온도 자체의 최적값을 바꾼다는 점이다. 즉, 같은 모델 계열이라도 SFT 단계와 RLHF 이후 단계에서 가장 잘 작동하는 탐색 온도가 다르다. 저자들은 이 때문에 rejection sampling을 운영할 때도 반복마다 온도를 다시 맞춰야 한다고 설명한다. PPO 단계의 구체 설정으로는 AdamW, learning rate \(1\times 10^{-6}\), batch size 512, PPO clip 0.2, minibatch 64 등이 제시되며, KL 페널티 계수는 모델 크기에 따라 다르게 둔다. 70B 모델의 경우 PPO 한 iteration이 평균 약 330초 걸렸고, 대규모 배치를 위해 FSDP를 활용했다.

논문은 PPO 목적식도 직접 적는다. 정책 \(\pi_\phi\) 가 데이터 분포 \(D\) 에서 프롬프트 \(x\) 를 받고 응답 \(y\sim\pi_\phi(\cdot\mid x)\) 를 생성한다고 할 때, 최적화의 기본 목표는 다음과 같이 쓸 수 있다.

\[
\max_{\phi}\;
\mathbb{E}_{x\sim D,\; y\sim \pi_\phi(\cdot\mid x)}[\,r(x,y)\,].
\]

여기서 최종 보상 \(r(x,y)\) 는 단순한 RM 점수 그 자체가 아니다. 논문은 원래 정책 \(\pi_0\) 와 너무 멀어지는 것을 막기 위해 KL 패널티를 더한 다음 식을 사용한다.

\[
r(x,y)
=
\tilde r(x,y)
-
\beta \log \frac{\pi_\phi(y\mid x)}{\pi_0(y\mid x)}.
\]

\(\tilde r(x,y)\) 는 다시 piecewise reward로 설계된다. 프롬프트가 잠재적으로 위험하다고 태깅되면 safety reward \(r_s(x,y)\) 를 우선 쓰고, 그렇지 않으면 helpfulness reward \(r_h(x,y)\) 를 사용한다. 즉 PPO는 언제나 “더 유용한 답”만을 쫓는 것이 아니라, 위험 프롬프트에 대해서는 애초에 다른 보상 축을 따르도록 설계되어 있다. 논문이 helpfulness RM과 safety RM을 분리한 이유가 여기서 최종 정책 수준까지 이어진다.

Figure 7은 rejection sampling의 직관을 가장 잘 보여 주는 그림이다. \(N\) 개 샘플 중 최대 보상과 중앙값 보상을 비교하면, 샘플 수가 커질수록 둘의 차이가 벌어진다. 중앙값은 거의 움직이지 않지만 최고점은 계속 올라가므로, 더 많이 탐색할수록 “고를 수 있는 좋은 답”의 상한이 오른다. Figure 8은 여기에 온도까지 얹어서, RLHF 후에는 최적 온도가 다시 달라진다는 점을 보여 준다. 저자들이 exploration을 단순한 하이퍼파라미터 탐색이 아니라 정렬 과정의 일부로 취급하는 이유가 바로 여기에 있다.


<img width="823" height="307" alt="image" src="https://github.com/user-attachments/assets/d741cefc-ccfb-46df-8ef6-6a5cce05bda2" />



<img width="816" height="289" alt="image" src="https://github.com/user-attachments/assets/42b25e11-8f52-4b2d-9aa1-9f224ad0bdc6" />



### 3.3 System Message for Multi-Turn Consistency

다중 턴 대화에서는 첫 턴에 준 지시가 이후 턴 전체에 걸쳐 유지되어야 한다. 예를 들어 짧게 답하라는 지시나 특정 인물처럼 말하라는 지시는 대화가 길어져도 사라지면 안 된다. 하지만 저자들은 초기 RLHF 모델이 몇 턴만 지나도 이런 시스템 수준 제약을 잊어버리는 경향을 확인했다.

이를 해결하기 위해 제안한 방법이 **Ghost Attention(GAtt)** 이다. 아이디어는 비교적 단순하다. 다중 턴 대화 데이터의 각 사용자 메시지에 동일한 지시를 합성적으로 덧붙여, 모델의 주의가 계속 그 지시를 보도록 유도한다. 다만 모든 턴에 같은 지시를 문자 그대로 넣으면 학습 시 문맥 불일치가 생길 수 있기 때문에, 중간 턴들에 대해서는 손실 계산을 조정해 그 부작용을 줄인다. 또한 학습 데이터에는 취미, 언어, 특정 공인 역할 같은 제약을 임의 조합으로 넣고, 표현도 일부는 더 짧고 덜 장황하게 바꿔 모델이 표면 문구가 아니라 제약 자체를 학습하게 한다.

논문식 표기로 쓰면 다중 턴 대화는 \((u_1,a_1,\dots,u_T,a_T)\) 형태의 사용자·어시스턴트 메시지 열로 볼 수 있고, 시스템 수준 제약은 별도의 지시 \(s\) 로 둘 수 있다. GAtt는 각 사용자 메시지를 \([s;u_t]\) 로 합성해 “이 제약은 모든 턴에 적용된다”는 사실을 학습 데이터 자체에 심는다. 다만 이렇게 만들면 첫 턴 이후의 중간 어시스턴트 메시지와 마지막 응답 사이에 훈련-추론 불일치가 생길 수 있으므로, 논문은 이전 턴 토큰들에 대해서는 loss를 0으로 두어 그 부작용을 줄인다. 결국 GAtt는 attention을 바꾸는 새로운 블록을 추가하는 방법이 아니라, 대화 데이터를 재구성해 시스템 메시지를 오랫동안 잊지 않게 만드는 정렬 기법이다.

저자들은 GAtt가 20턴 이상에서도 제약을 비교적 안정적으로 유지하며, 학습에 없던 제약 형식에도 어느 정도 zero-shot으로 일반화한다고 보고한다.

이 관찰은 부록의 Table 30과 Figure 28에서 더 선명해진다. Table 30에서 baseline은 2턴에서는 100%로 속성을 유지하지만 4턴에서 10%, 6턴과 20턴에서는 0%까지 무너진다. 반면 GAtt를 넣은 모델은 2, 4, 6, 20턴 모두 100%를 유지한다. Figure 28은 더 흥미롭다. 학습에 없던 제약, 예를 들어 한 문장으로만 답하라거나 특정 형식으로 계속 답하라는 제약도 여러 턴에 걸쳐 유지한다. 즉 GAtt의 효과는 단순 암기가 아니라 시스템 수준 제약 유지라는 행동 패턴 자체를 강화하는 데 있다.


<img width="718" height="392" alt="image" src="https://github.com/user-attachments/assets/5b0dbc0f-09c4-4cef-ab29-c926e224a0db" />


<img width="766" height="358" alt="image" src="https://github.com/user-attachments/assets/ba2d552c-3eb9-43c5-86ca-195b3e3483a5" />



### 3.4 RLHF Results

RLHF의 효과는 모델 기반 평가와 사람 평가 두 축에서 제시된다.

#### 3.4.1 Model-Based Evaluation

저자들은 보상모델을 단순한 학습 도구로만 쓰지 않고, 버전 간 비교를 위한 모델 기반 평가기에도 활용한다. 다만 Goodhart 현상, 즉 특정 보상에 과도하게 맞추는 문제가 생길 수 있으므로, 보상모델이 사람 판단과 얼마나 일치하는지 별도로 확인한다. 이 부분의 자세한 보정 결과는 부록의 Figure 29로 이어진다.

Figure 11은 여러 차례의 RLHF를 거치며 Llama 2-Chat이 어떻게 진화했는지를 보여 준다. 보상모델을 기준으로 보면 RLHF-V3 이후에는 ChatGPT와의 비교에서 도움성과 안전성 모두에서 우세한 구간이 나타난다. 다만 GPT-4를 판정자로 둔 비교에서는 최신 버전이 강해졌음에도 여전히 일정한 차이가 남아 있음을 보여 준다. 저자들은 이런 차이가 곧바로 절대적 우열을 뜻하는 것은 아니지만, 적어도 모델의 반복 개선이 한 방향으로 진행되었음을 시사한다고 해석한다.

논문은 여기서 Goodhart의 법칙을 직접 언급한다. 내부 보상모델이 최적화 대상이 되는 순간 그 점수만 높이고 사람에게는 별로인 응답을 만들 위험이 있기 때문이다. 그래서 Figure 11을 두 개의 판정자, 즉 내부 RM과 GPT-4로 병렬 제시한다. 내부 RM에서는 자사 모델이 더 빨리 올라가고, GPT-4 판정에서는 상승 폭이 더 보수적으로 보인다. 이 이중 비교 구조는 논문 전체의 태도와도 맞닿아 있다. 보상모델은 빠른 반복을 위한 중요한 프록시지만, 최종적인 모델 선택을 인간 평가와 더 일반적인 판정자 없이 맡길 수는 없다는 것이다.


<img width="717" height="329" alt="image" src="https://github.com/user-attachments/assets/da46ddf4-11c3-4a6b-b5b5-2941c9186ec6" />


#### 3.4.2 Human Evaluation

사람 평가는 약 4,000개의 도움성 프롬프트를 사용해 수행된다. 단일 턴과 다중 턴이 모두 포함되며, 공개형 모델뿐 아니라 일부 비공개 모델도 비교 대상에 들어간다. 저자들이 보고한 결과에 따르면 Llama 2-Chat은 전반적으로 공개형 모델들을 뚜렷하게 앞선다. 7B는 MPT-7B-Chat보다 많은 프롬프트에서 우세했고, 34B는 Vicuna-33B와 Falcon-40B를 크게 앞서는 구간을 보였으며, 70B는 ChatGPT와 비교해도 상당한 경쟁력을 보였다.

동시에 저자들은 이 사람 평가의 한계도 길게 적는다. 프롬프트셋이 실제 세계의 모든 사용을 대표하지 못하고, 코딩이나 수학 추론 같은 특정 고난도 영역이 충분히 포함되지 않았으며, 다중 턴 대화에서도 마지막 응답만 평가했다는 점이 대표적이다. 평가자 간 일치도 역시 완벽하지 않다. 따라서 저자들은 이 결과를 절대적 서열표가 아니라, 자신들이 설계한 평가 환경 안에서의 비교 결과로 받아들여야 한다고 말한다.

그럼에도 Figure 12가 주는 메시지는 분명하다. 7B는 MPT-7B-Chat을 약 60% 수준의 프롬프트에서 이기고, 34B는 Vicuna-33B와 Falcon-40B를 상대로 75%가 넘는 승률을 기록한다. 70B는 ChatGPT를 상대로 승률 36%, tie 31.5%를 보여 “완전히 앞선다”까지는 아니더라도, 공개형 모델이 닿기 어려웠던 구간에 근접했음을 시사한다. 논문이 굳이 tie 비율까지 밝히는 이유도 여기에 있다. 완전한 우세보다는, 사람이 봤을 때 비슷한 수준으로 느껴지는 구간이 이미 상당하다는 사실이 더 중요하기 때문이다.


<img width="716" height="345" alt="image" src="https://github.com/user-attachments/assets/90928271-a5a5-4ff6-a05e-7e80afdad53b" />


## 4. Safety

이 절은 안전성을 사전학습 단계와 미세조정 단계로 나누어 다룬다. 저자들은 foundation model 자체를 완전히 무해하게 만들었다고 주장하지 않는다. 대신 사전학습 데이터의 성격을 분석하고, 그 위에 별도의 safety fine-tuning, safety RLHF, context distillation, red teaming, 안전성 평가를 결합해 위험을 줄이는 전략을 설명한다.

### 4.1 Safety in Pretraining

먼저 저자들은 사전학습 코퍼스가 어떤 편향과 위험을 포함할 수 있는지 분석한다. 대명사 사용과 정체성 관련 표현을 조사한 결과, 성별·문화권·사회집단 표현이 균등하지 않으며 일부 서구권 인구집단과 남성형 대명사가 상대적으로 더 자주 나타난다. 언어 분포도 영어에 크게 치우쳐 있다. 이 때문에 Llama 2는 본질적으로 영어 중심 모델이며, 다른 언어에서는 성능과 안전성 모두가 더 불안정할 수 있다고 경고한다.

독성 데이터에 대해서는 강한 사전 삭제보다 **과도한 일반화 손상 없이 downstream 단계에서 제어하는 전략**을 택한다. 저자들은 독성 데이터를 지나치게 제거하면 모델이 위험하거나 공격적인 입력을 인식하고 안전하게 다루는 능력까지 잃을 수 있다고 본다. 그 결과, 영어 코퍼스 기준으로 일정 비율의 독성 문서는 그대로 남겨 두었다. 자동 안전성 벤치마크에서는 Llama 2 base가 Llama 1 대비 일부 지표에서 개선되지만, pretrained model만으로는 충분한 안전성을 기대하기 어렵고 추가적인 downstream 정렬이 필요하다는 결론을 명확히 한다.

Table 9는 이러한 편향을 추상적으로 말하지 않고 바로 데이터 표현의 비대칭으로 연결한다. 대명사와 정체성 기술어 분석에서는 서구권 인구집단과 남성형 표현의 비중이 더 높게 나타난다. Figure 13은 독성의 규모도 정량화한다. 10% 샘플에서 HateBERT 기반 분류기로 측정했을 때, 영어 문서 가운데 약 0.2%가 0.5 이상의 독성 가능성 점수를 받는다. 숫자만 보면 작아 보이지만, 코퍼스 전체 규모를 생각하면 무시할 수 없는 양이다.

Table 10은 언어 편향을 더 직접적으로 보여 준다. 전체 문서의 89.70%가 영어로 식별되고, 8.38%는 unknown, 그 외 언어는 모두 작은 꼬리 분포를 이룬다. 한국어는 0.06% 수준이다. 따라서 이 모델을 영어 중심이 아닌 다국어 assistant처럼 읽으면 곧바로 논문 해석을 잘못하게 된다.

Table 11은 이런 데이터 선택의 결과를 pretrained model 단계에서 확인시킨다. Llama 2 70B는 TruthfulQA의 truthful-and-informative 비율에서 50.18까지 올라가지만, ToxiGen 같은 독성 지표에서는 다른 모델을 압도하지 않는다. 저자들은 이를 실패라기보다 의도된 절충으로 해석한다. 사전학습 데이터에서 독성을 너무 강하게 긁어내지 않았기 때문에 독성 수치는 극단적으로 낮지 않을 수 있지만, 대신 downstream safety alignment에 필요한 일반화 능력은 더 잘 남겨 둔다는 논리다.

<img width="722" height="763" alt="image" src="https://github.com/user-attachments/assets/e7683ebf-9278-43ca-bdf7-76eb5484358a" />



<img width="721" height="332" alt="image" src="https://github.com/user-attachments/assets/433aa959-e362-4ea6-87f5-0d7d54b001dc" />



<img width="734" height="318" alt="image" src="https://github.com/user-attachments/assets/ba910076-5881-4054-a125-46a36a786b41" />



### 4.2 Safety Fine-tuning

안전성 미세조정은 세 갈래로 설명된다. 첫째는 안전 응답을 직접 쓰게 하는 supervised fine-tuning, 둘째는 사람 선호와 보상모델을 이용한 safety RLHF, 셋째는 안전한 preprompt를 활용해 출력을 증류하는 context distillation이다.

#### 4.2.1 Safety Categories and Annotation Guidelines

저자들은 안전성 주석 작업을 위해 위험 범주와 공격 벡터를 체계화한다. 위험 범주에는 범죄·불법 행위 지원, 혐오·유해 행위, 자격 없는 조언 제공 같은 항목이 포함된다. 공격 벡터에는 노골적 요청뿐 아니라 심리적 설득, 논리적 우회, 문법적 변형, 의미적 우회, 관점 전환, 비영어 프롬프트 등이 포함된다.

주석 지침의 핵심은 단순 거절만이 아니라 **안전하면서도 가능한 한 도움이 되는 응답**을 만드는 것이다. 즉, 즉각적인 안전 문제를 짚고, 왜 위험한지 설명하고, 가능하면 덜 위험한 방향의 추가 정보를 제공하는 응답이 가장 바람직한 형태로 간주된다. 이 지침은 실제 수집 과정에서 여러 차례 수정·정제되었다고 설명된다.


#### 4.2.2 Safety Supervised Fine-Tuning

안전성 SFT에서는 훈련된 주석자가 적대적 프롬프트에 대해 안전하면서도 유용한 답을 직접 작성한다. 데이터 수집과 학습 방식은 일반 SFT와 유사하지만, 프롬프트 설계와 응답 기준이 안전성에 초점을 맞춘다. 이 단계는 모델이 위험 요청에 대해 어떤 어조와 구조로 반응해야 하는지에 대한 기초 행동 규범을 심는 역할을 한다.


#### 4.2.3 Safety RLHF

저자들은 수천 건 규모의 안전 데모를 확보한 뒤에는, 더 미묘한 안전성 개선을 위해 RLHF로 중심을 이동한다. 이때도 도움성 보상모델과 안전성 보상모델을 분리해 운용한다. 특히 안전하지 않은 프롬프트에 대해서는 safety reward가 더 우선하도록 정책 업데이트를 설계한다.

이 절의 핵심 결과는, 충분한 safety RLHF가 **도움성을 크게 해치지 않으면서 긴 꼬리의 위험 응답을 줄일 수 있다**는 점이다. 저자들은 안전 데이터 비율을 0%, 1%, 10%, 25%, 50%, 100%로 달리한 실험을 제시하며, 안전 데이터가 늘수록 평균 안전성 점수는 올라가고 위험한 응답의 꼬리 분포는 거의 사라진다고 보고한다. 다만 부록에서는 이 과정이 과보수성이나 false refusal 증가로 이어질 수 있음도 별도로 다룬다.

Figure 14가 바로 이 주장을 시각적으로 뒷받침한다. 왼쪽 패널에서는 Meta Safety 테스트셋에 대한 safety RM 점수 분포가 위쪽으로 이동하고, 가장 위험한 응답들이 몰려 있던 낮은 점수의 긴 꼬리가 얇아진다. 오른쪽 패널에서는 Meta Helpfulness 테스트셋에 대한 helpfulness RM 점수 분포가 눈에 띄게 무너지지 않는다. 저자들이 “안전성 stage를 추가해도 충분한 helpfulness 데이터가 있으면 일반적인 도움성 저하는 거의 없다”고 말하는 근거가 이 그림이다. Table 12의 질적 비교는 이를 더 직관적으로 만든다. 초기 모델이 사기 메일 같은 요청에 노골적으로 협조하던 자리가, 최종 RLHF 모델에서는 즉각적 거절과 방향 전환으로 바뀐다.

Figure 15는 같은 이야기를 데이터 스케일 차원에서 반복한다. 안전 데이터 비율을 올릴수록 mean safety RM score는 빠르게 좋아지고, 분포의 왼쪽 꼬리, 즉 가장 위험한 응답군은 거의 사라진다. 반면 mean helpfulness score는 비교적 안정적으로 유지된다. 논문은 이 결과를 “helpfulness data가 이미 충분히 많기 때문”이라고 해석한다. 다만 이 절의 끝과 부록 A.4.5가 보여 주듯, 평균 도움성이 유지된다고 해서 과잉 거절 문제가 사라지는 것은 아니다.


<img width="564" height="708" alt="image" src="https://github.com/user-attachments/assets/ab2e6da2-9ea7-44a4-b4a1-2fa28d23068f" />



<img width="561" height="309" alt="image" src="https://github.com/user-attachments/assets/686ed857-4417-480a-b685-a0562205eb8c" />



#### 4.2.4 Context Distillation for Safety

context distillation은 안전한 preprompt를 먼저 붙여 더 바람직한 응답을 생성한 다음, 그 출력을 사용해 다시 모델을 미세조정하는 방식이다. 저자들은 범용적인 안전 assistant preprompt도 사용하지만, 위험 범주에 따라 더 구체적인 answer template을 주는 방식이 더 강하게 작동한다고 설명한다.

하지만 이 방법은 항상 이득만 주지 않는다. 이미 충분히 좋았던 응답을 오히려 뻣뻣하게 만들거나, 애초에 안전한 요청까지 거절하게 만드는 경우가 있다. 그래서 저자들은 context distillation을 모든 데이터에 일괄 적용하지 않고, 적대적 프롬프트에만 선택적으로 적용하며, 증류된 출력의 안전성 보상 점수가 실제로 개선된 경우에만 채택한다.

Table 13이 보여 주는 차이는 특히 중요하다. 범용 preprompt는 모델을 대체로 안전하게 만들지만, 위험 범주와 답변 템플릿까지 지정한 preprompt는 응답이 훨씬 구체적이고 해당 위험에 직접 맞닿는다. Figure 16의 왼쪽 패널에서는 generic preprompt보다 answer template이 붙은 targeted preprompt가 safety RM 점수를 더 크게 올린다. 오른쪽 패널에서는 이 효과가 원래 점수가 낮은 샘플에서 특히 크지만, 이미 점수가 높은 샘플에는 오히려 부정적으로 작용할 수 있음을 보여 준다. 그래서 논문은 context distillation을 전면적 해결책이 아니라, RM 점수가 실제로 개선되는 샘플에만 써야 하는 선택적 처방으로 다룬다.


<img width="568" height="465" alt="image" src="https://github.com/user-attachments/assets/e04a57d4-dd1c-4684-9faa-a093cbcc0d7f" />


<img width="559" height="357" alt="image" src="https://github.com/user-attachments/assets/4cb84a67-9961-4297-9931-8ca26bd13e8a" />



### 4.3 Red Teaming

모델 배포 전에는 대규모 레드팀이 운영된다. 저자들에 따르면 350명 이상의 내부 인력, 외부 계약자, 전문 벤더가 참여했고, 사이버 보안, 선거 사기, 허위정보, 법률, 정책, 시민권, 윤리, 머신러닝 등 여러 분야 전문가도 포함되었다. 점검 대상은 범죄, 무기, 사이버 악용, CBRN, 허위정보 같은 고위험 범주뿐 아니라, 다양한 우회 공격과 다중 턴 대화, 비영어 프롬프트까지 폭넓게 포함했다.

레드팀 과정에서 얻은 실무적 관찰도 제시된다. 예를 들어 초기 모델은 위험 요청을 겉으로는 거절하면서 실제 내용은 부분적으로 제공하는 경우가 있었고, 창의적 글쓰기나 역할극 형식이 안전장치를 우회하는 경향도 보였다고 설명한다. 또한 겉보기에는 긍정적 맥락이더라도 실제로는 위험한 요청을 숨길 수 있다는 점이 강조된다.

논문이 이 절을 짧게 쓰지 않는 이유는 레드팀이 단순한 점검 절차가 아니라, 다음 tuning iteration에 들어갈 데이터를 만드는 생산 공정이기 때문이다. 저자들은 350명 이상이 참여한 광범위한 레드팀을 운영했고, 각 exercise 뒤에 대화 길이, 위험 영역 분포, 허위정보 토픽 분포, 위험도 점수까지 분석해 다음 안전성 미세조정과 reward modeling에 다시 반영했다. 즉 레드팀은 모델을 “평가만” 하는 단계가 아니라, 안전성 데이터를 재공급하는 폐쇄 루프의 일부로 설계되어 있다.

### 4.4 Safety Evaluation of Llama 2-Chat

안전성 평가는 약 2,000개의 적대적 프롬프트로 수행된다. 이 중 1,351개는 단일 턴, 623개는 다중 턴이다. 응답은 5점 척도로 라벨링되고, 1점이나 2점이면 안전 위반으로 간주한다. 각 샘플은 세 명의 주석자가 검토하고 다수결로 최종 라벨을 정한다. 저자들은 평가자 간 일치도가 비교적 높았다고 보고한다.

결과적으로 Llama 2-Chat은 전체 위반 비율에서 경쟁 모델과 비슷하거나 더 낮은 수준을 보인다. 특히 다중 턴 상황에서 상대적으로 강한 결과를 보고하며, 위험 범주별로는 자격 없는 조언 제공 항목에서 더 많은 위반이 관찰된다. 자동 벤치마크 결과까지 합치면, fine-tuned Llama 2-Chat은 pretrained base 대비 진실성은 더 높아지고 독성 생성은 크게 줄어드는 방향으로 이동한다.

그럼에도 저자들은 안전성 평가의 프롬프트셋과 기준 역시 완전하지 않으며, 특정 문화권이나 정책 기준에 의존할 수 있다는 점을 다시 강조한다. 따라서 이 절의 결과 역시 절대적인 무해성 선언이 아니라, 자신들이 설계한 검증 체계 안에서 관찰된 상대적 개선으로 읽어야 한다.

Figure 17은 왜 논문이 violation percentage와 mean rating을 함께 보고하는지를 설명해 준다. 어떤 모델은 답을 매우 짧게만 해서 위반을 덜 일으킬 수 있지만, 그만큼 덜 유용할 수도 있다. 논문은 Falcon이 바로 그런 경우라고 지적한다. raw violation percentage만 보면 비슷해 보여도, 평균 평점에서는 Llama 2-Chat 34B보다 훨씬 낮다.

Figure 18은 모든 모델에서 multi-turn이 single-turn보다 더 위험하다는 점을 보여 준다. 사용자가 몇 턴에 걸쳐 안전장치를 우회할 기회가 많아지기 때문이다. 그럼에도 Llama 2-Chat은 특히 multi-turn에서 경쟁 모델 대비 강한 편이다. Figure 19는 위험 범주별로 나누었을 때 상대적으로 약한 지점도 숨기지 않는다. Llama 2-Chat은 다른 범주에서는 대체로 낮은 위반률을 보이지만, unqualified advice 범주에서는 비교적 더 많은 위반을 보인다. 다만 이 역시 절대량 자체는 낮고, 종종 전문직 disclaimer를 빠뜨리는 식의 사례가 포함된다고 설명된다.

Table 14는 사람 평가에서 본 개선이 자동 벤치마크와도 같은 방향인지 확인해 준다. TruthfulQA의 truthful-and-informative 비율은 Llama 2-Chat 7B, 13B, 34B, 70B에서 각각 57.04, 62.18, 67.20, 64.14이고, ToxiGen 독성 생성 비율은 0.00, 0.00, 0.02, 0.01로 매우 낮다. 따라서 이 절은 단지 “사람이 안전하다고 느꼈다”는 이야기만이 아니라, 사실성·독성 측면의 자동 지표도 함께 개선되었다는 점까지 보여 준다.


<img width="563" height="220" alt="image" src="https://github.com/user-attachments/assets/80765a41-ef6e-4198-a3b0-d69b1b143cf2" />


<img width="819" height="331" alt="image" src="https://github.com/user-attachments/assets/51a414c0-ffb0-4523-a2d8-90c9809ac62d" />


<img width="819" height="328" alt="image" src="https://github.com/user-attachments/assets/14dd147a-f418-492e-9412-902b7eb0335e" />



<img width="808" height="252" alt="image" src="https://github.com/user-attachments/assets/525e8661-2ae1-4a73-a0c3-3a22eee663bc" />



## 5. Discussion

### 5.1 Learnings and Observations

저자들은 가장 큰 관찰로 **RLHF의 실효성**을 꼽는다. 팀 내부에서도 사람이 직접 좋은 답을 쓰는 SFT가 더 중요할 것이라고 생각한 경우가 있었지만, 실제로는 비교형 피드백과 반복적 RLHF가 품질을 더 크게 끌어올렸다는 것이다. 사람은 항상 최적의 답을 새로 써 낼 수는 없지만, 둘 중 어느 쪽이 더 나은지는 비교적 안정적으로 판단할 수 있기 때문이다.

Figure 20은 이 관찰을 분포 수준에서 보여 준다. 논문이 강조하는 것은 평균 보상이 올랐다는 사실보다, SFT에서 RLHF로 갈수록 응답 분포의 왼쪽 꼬리, 즉 명백히 나쁜 답변들의 비중이 줄어든다는 점이다. RLHF는 최고의 문장을 하나 더 만드는 기술이기도 하지만, 실제로는 최악의 응답을 체계적으로 제거하는 기술에 더 가깝다는 해석이 여기서 나온다.

Figure 20은 SFT 단계에서 RLHF 단계로 갈수록 응답 분포가 어떻게 이동하는지 보여 준다. 저자들이 강조하는 것은 평균 점수 상승만이 아니라, 나쁜 꼬리 분포를 점차 줄여 간다는 점이다. 이는 RLHF가 단순히 더 화려한 답을 만드는 것이 아니라, 명백히 좋지 않은 응답을 제거하는 방향으로 작동했음을 시사한다.

또 다른 관찰은 RLHF가 프롬프트 유형에 따라 사실상 샘플링 온도를 다르게 학습하는 현상이다. Figure 21에서 factual 프롬프트는 다양성이 줄고, creative 프롬프트는 상대적으로 다양성을 유지한다. 이는 RLHF가 단순한 순위 최적화를 넘어, 어떤 종류의 질문에서 얼마나 확정적으로 답해야 하는지까지 내부적으로 조정하고 있음을 시사한다.

Figure 21의 측정 방식은 Self-BLEU다. 값이 낮을수록 여러 샘플 간 다양성이 크다. 논문은 창작 프롬프트 10개와 사실형 프롬프트 10개를 고르고, 각 온도에서 25개 응답을 샘플링해 Self-BLEU를 계산한다. 결과는 명확하다. 창작 프롬프트에서는 RLHF를 거친 뒤에도 온도를 올리면 다양한 응답이 계속 나오지만, factual 프롬프트에서는 온도를 높여도 결국 같은 답으로 수렴하는 방향이 강해진다. RLHF가 ‘정답이 있는 질문’과 ‘다양성이 허용되는 질문’을 내부적으로 구분하기 시작했다는 뜻이다.

저자들은 추가로 두 가지 흥미로운 현상을 보고한다. 첫째, 1,000개의 시간 관련 SFT 예시만으로도 모델이 시간 인식 개념을 일반화하는 경향이 나타난다. 둘째, 도구 사용을 명시적으로 학습시키지 않았는데도, 적절한 예시와 문맥만 주어지면 모델이 도구의 쓰임새와 인자 구조를 어느 정도 파악해 사용하는 모습이 나타난다.

Figure 22는 시간 인식 일반화의 사례를 모은다. 저자들은 날짜가 붙은 SFT 예시 1,000개만으로도 모델이 “지금 시점에서 이 사건은 얼마나 과거인가” 같은 질문을 비교적 안정적으로 처리하는 모습을 관찰한다. 이는 무작위 셔플된 next-token prediction만으로 학습한 모델 안에도 시간 구조를 재조직할 여지가 있다는 뜻이다.

Table 15와 Figure 23은 도구 사용 출현을 더 도발적으로 보여 준다. 계산기 접근이 가능할 때 Llama 2-Chat은 ASDiv 67.1, SVAMP 69.2, MAWPS 82.4를 기록해 Toolformer의 40.4, 29.4, 44.0을 크게 앞선다. 논문은 이를 “도구 사용을 위한 전용 주석 없이도 alignment 과정만으로 tool semantics와 API argument 구조를 이해하는 능력이 zero-shot으로 드러날 수 있다”는 신호로 읽는다. 물론 저자들은 바로 이어서, 이런 능력이 흥미로운 만큼 새로운 안전 문제도 가져온다고 경고한다.


<img width="818" height="227" alt="image" src="https://github.com/user-attachments/assets/bd1bfd48-7af4-4b1a-8893-27523900d499" />



<img width="815" height="828" alt="image" src="https://github.com/user-attachments/assets/da55e906-468b-42c4-ba74-84e8f48e4452" />



<img width="837" height="580" alt="image" src="https://github.com/user-attachments/assets/aba1a53f-dc5f-4d87-ae36-4970ed929a50" />



### 5.2 Limitations and Ethical Considerations

저자들은 Llama 2와 Llama 2-Chat이 여전히 일반적 LLM의 한계를 공유한다고 인정한다. 학습 이후 시점의 지식은 반영되지 않으며, 환각은 완전히 사라지지 않는다. 자격 없는 조언을 하거나, 문화적·언어적 맥락을 충분히 반영하지 못할 가능성도 남아 있다. 영어 중심 데이터 때문에 다른 언어에서는 품질과 안전성 모두가 더 약할 수 있다.

또한 안전성 조정을 거쳤더라도 모델이 해로운 내용, 편향된 내용, 공격적인 표현을 생성할 가능성은 여전히 남는다. 반대로 안전성 조정을 강하게 할수록 지나치게 보수적인 반응이나 잘못된 거절이 늘 수 있다. 저자들은 이러한 문제 때문에 base model 자체를 곧바로 고위험 환경에 투입해서는 안 되며, 실제 사용에서는 별도의 안전 장치와 도메인별 검증이 필요하다고 말한다.

### 5.3 Responsible Release Strategy

저자들은 Llama 2를 연구용뿐 아니라 상업적 사용도 가능한 형태로 공개하면서, 이를 책임 있는 공개 전략의 일부로 설명한다. 즉, 모델 가중치만 배포하는 것이 아니라, 라이선스, Acceptable Use Policy, 코드 예시, Responsible Use Guide 같은 문서를 함께 제공해 사용자가 위험과 한계를 인지한 상태에서 배포하도록 돕는다는 것이다.

이 절의 논지는 개방성 자체가 무조건 선하다는 주장이라기보다, 충분한 문서화와 사용 조건을 동반한 공개가 연구 재현성, 비용 절감, 책임 있는 혁신, 외부 감시에 도움이 될 수 있다는 주장에 가깝다. 동시에 저자들은 공개 모델도 악용될 수 있으며, 따라서 공개 이후에도 모니터링과 후속 완화가 필요하다는 점을 인정한다.

## 6. Related Work

관련 연구 절은 대규모 언어모델의 발전, 공개형과 비공개형 모델의 차이, instruction tuning과 RLHF의 계보, 자동 평가와 사람 평가의 한계, 그리고 안전성 연구와 레드팀 문헌을 폭넓게 묶는다. 저자들은 Llama 2를 완전히 새로운 패러다임으로 제시하기보다, 기존 연구 흐름 위에서 **공개 가능한 foundation model과 제품형 chat model 사이의 연결 고리**를 보다 체계적으로 문서화한 사례로 위치시킨다.

또한 이 절은 정렬과 안전성을 단순히 마지막 단계의 장식이 아니라, 모델 배포 가능성을 좌우하는 핵심 공학 문제로 다룬다는 점에서 앞선 instruction-tuning·RLHF 연구와 직접 연결된다. 공개형 생태계에서는 특히 fine-tuning 레시피와 평가 절차의 투명성이 중요하다는 점도 암묵적으로 강조된다.

## 7. Conclusion

결론에서 저자들은 7B부터 70B까지의 사전학습 및 미세조정 모델 계열을 공개했고, Llama 2-Chat이 당시의 공개형 대화 모델을 전반적으로 앞서는 결과를 보였다고 요약한다. 동시에 일부 비공개 모델에 접근하거나 대체 가능성을 보였지만, GPT-4 같은 최상위 모델과의 격차가 완전히 사라진 것은 아니라는 점도 인정한다.

이 논문의 더 큰 기여는 결과표 자체보다, foundation model에서 chat model로 넘어가는 실제 절차를 비교적 투명하게 공개했다는 데 있다. 저자들은 이 공개가 커뮤니티의 후속 개선, 더 책임 있는 배포, 더 나은 안전성 연구로 이어지기를 기대한다.

## References

> [References 삽입 위치]

## Appendix A

### A.1 Contributions

부록은 먼저 기여자와 각 팀의 역할을 정리한다. 여기에는 사전학습 인프라, 데이터 수집, SFT, RLHF, 안전성 튜닝, 평가, 정책 검토, 릴리스 준비 등 모델 제작과 배포에 필요한 여러 기능이 어떻게 분담되었는지가 포함된다.

#### A.1.1 Acknowledgments

감사의 말에서는 주석자, 레드팀 참여자, 외부 도메인 전문가, 인프라 엔지니어, 법무·정책·커뮤니케이션 팀 등 광범위한 협업 주체를 언급한다. 저자들은 특히 사람 평가와 안전성 검토가 모델 성능 개선만큼이나 중요한 작업이었다는 점을 드러낸다.

### A.2 Additional Details for Pretraining

#### A.2.1 Architecture Changes Compared to Llama 1

이 절은 컨텍스트 길이 확장과 attention 구조 변경의 효과를 보다 세부적으로 분석한다. 컨텍스트를 2k에서 4k로 늘리면 긴 문맥 과제에서 성능이 개선되며, 일반 과제에서는 성능 저하 없이 유지되는 경향이 보고된다. attention 구조 측면에서는 MHA, MQA, GQA를 비교한 뒤, 품질과 효율의 절충으로 GQA를 선택한 이유를 설명한다. 저자들은 특히 대형 모델에서 GQA가 MHA에 가까운 품질을 유지하면서도 더 나은 처리량을 제공한다고 본다.

Table 16은 4k 문맥 확장이 장식이 아니라는 점을 명확하게 보여 준다. NarrativeQA F1은 0.21에서 17.26으로, Qasper는 0.71에서 18.52로, QuALITY 정확도는 26.1에서 29.6으로, ContractNLI EM은 11.76에서 16.33으로 오른다. QMSum과 SQuAD도 모두 개선된다. 반면 Table 17의 일반 과제에서는 변화가 작다. HellaSwag은 거의 유지되고, NQ도 비슷하며, GSM8K는 4.9에서 6.5로 오르고, HumanEval은 7.9에서 7.3으로 약간 내려간다. 즉 긴 문맥 능력의 이득이 일반 능력의 광범위한 붕괴를 대가로 하지는 않는다는 뜻이다.

Table 18과 Figure 24는 왜 34B와 70B에서 GQA를 택했는지 설명한다. 품질 면에서는 GQA가 MHA와 비슷하고 MQA보다 평균적으로 낫다. 효율 면에서는 MHA가 KV cache 때문에 긴 문맥과 큰 배치에서 빠르게 메모리 한계에 걸리는데, GQA와 MQA는 훨씬 더 높은 배치까지 버틴다. Figure 24가 보여 주는 throughput 차이는, GQA가 단순한 미세 최적화가 아니라 큰 모델을 실제 서비스 가능한 형태로 가져가기 위한 구조적 선택이었다는 점을 드러낸다.


<img width="624" height="290" alt="image" src="https://github.com/user-attachments/assets/856151be-4795-46c3-aa79-056c2c702a08" />



<img width="816" height="583" alt="image" src="https://github.com/user-attachments/assets/2c490b6a-7d53-4859-9e85-b6c8c769553c" />


#### A.2.2 Additional Details for Pretrained Models Evaluation

이 절은 본문에서 요약만 제시했던 base model 평가 결과를 세부 표로 확장한다. MMLU, 일반 상식·추론 벤치마크, 코드 생성, 세계 지식, 읽기 이해, AGI Eval, 수학 추론 등 다양한 과제가 따로 정리되며, Llama 2가 어느 과제에서 강하고 어디에서 상대적으로 약한지가 보다 촘촘하게 드러난다.

가장 대표적인 것은 Table 19의 MMLU 세부 결과다. Llama 2 70B는 Humanities 65.0, STEM 58.0, Social Sciences 80.3, Other 74.6, Average 68.9로 공개형 base model 가운데 가장 높은 평균을 보인다. Social Sciences와 Other 축에서 특히 강한 수치를 보이는 반면, STEM은 상대적으로 낮다. 이어지는 Table 20~25는 이 패턴이 단일 평균에 가려지지 않도록 과제별 성격을 분해한다. 코드 생성에서는 여전히 상위 폐쇄형 모델과 차이가 남고, 반대로 세계 지식·읽기 이해·수학 추론에서는 크기에 따라 비교적 안정적인 이득이 누적된다. 부록을 읽으면 본문이 말한 “전반적 향상”이 실제로는 어떤 항목에서 강하고 어떤 항목에서 약한지 더 구체적으로 보이게 된다.


<img width="809" height="322" alt="image" src="https://github.com/user-attachments/assets/bb5bd784-0f0d-4676-bcaa-14c5e419c7f2" />



<img width="815" height="710" alt="image" src="https://github.com/user-attachments/assets/0463a912-584f-4652-b6c5-28150bf49f03" />


<img width="796" height="758" alt="image" src="https://github.com/user-attachments/assets/c28b3f76-de65-4769-a636-823b74903ed1" />



<img width="768" height="291" alt="image" src="https://github.com/user-attachments/assets/f502dfec-e932-4ca9-8658-4099c306ea84" />


<img width="826" height="344" alt="image" src="https://github.com/user-attachments/assets/e9803e18-9fdc-4f3d-b98e-b5c2a8970d36" />



### A.3 Additional Details for Fine-tuning

#### A.3.1 Detailed Statistics of Meta Human Preference Data

이 절은 내부에서 수집한 선호 데이터의 주차별 통계를 자세히 보여 준다. 배치가 진행될수록 비교 수가 커지고, 대화 길이도 늘어나며, 다중 턴 샘플 비중이 커진다. 동시에 모델이 좋아질수록 두 응답의 차이가 미세해져, 평가자가 '거의 비슷하다'거나 '판단하기 어렵다'고 답하는 비율이 증가한다.

Table 26은 보상모델 데이터가 시간이 지나면서 단순히 양만 늘어난 것이 아니라, 길이와 대화성까지 함께 증가했다는 점을 보여 준다. 논문이 RLHF를 정적인 파이프라인이 아니라 주차별 annotation curriculum으로 보는 이유가 여기에 있다. Figure 25에서 'negligibly better / unsure' 쪽 분포가 후기로 갈수록 커지는 것도 같은 맥락이다. 모델이 개선될수록 annotator는 더 어려운 비교를 하게 되고, 바로 그 어려운 비교가 다음 reward model의 품질을 밀어 올린다.


<img width="822" height="470" alt="image" src="https://github.com/user-attachments/assets/e420e52e-d13b-4492-8ca6-65140c72ac11" />



<img width="818" height="421" alt="image" src="https://github.com/user-attachments/assets/6dc0b196-e494-472f-a94a-5b4bef1ebe80" />



#### A.3.2 Curriculum Strategy for Meta Human Preference Data

저자들은 선호 데이터 수집이 사실상 **annotation curriculum**을 형성한다고 본다. 초기에는 쉬운 비교가 많고, 후기로 갈수록 모델들이 비슷해져 더 미묘한 비교가 늘어난다. Figure 26은 시간이 지날수록 배치별 프롬프트의 난도가 올라가는 경향을 보여 준다.

이런 curriculum 관점은 RLHF를 해석할 때 중요하다. 같은 양의 선호 데이터라 해도, 쉬운 비교만 많이 모으는 것과 점점 더 미세한 비교를 모으는 것은 전혀 다르다. 논문은 후반 배치의 낮은 agreement를 데이터 품질 저하로 보지 않고, 오히려 모델들이 이미 일정 수준 이상으로 올라왔기 때문에 생기는 자연스러운 어려움으로 본다.


<img width="812" height="366" alt="image" src="https://github.com/user-attachments/assets/d62f4398-491b-4e5e-b7fb-4c4dbde3a975" />



#### A.3.3 Ablation on Ranking Loss with Preference Rating-based Margin for Reward Modeling

이 절은 도움성 보상모델의 ranking loss에 선호 강도 기반 margin을 넣는 것이 왜 유용한지 분석한다. 저자들은 사람이 '현저히 더 낫다'고 표시한 비교와 '약간 더 낫다'고 표시한 비교를 같은 세기로 다루지 않는 편이 더 낫다고 본다. 다만 margin을 너무 크게 두면, 원래도 차이가 미세한 샘플까지 지나치게 이분화할 위험이 있다.

Table 27은 이 아이디어를 small margin과 large margin이라는 두 변형으로 구체화한다. small margin은 1, 2/3, 1/3, 0, large margin은 3, 2, 1, 0으로 각 선호 강도를 수치화한다. Table 28을 보면 margin이 없는 기본식보다 margin을 넣었을 때 특히 'significantly better'와 'better' 구간의 정확도가 더 오른다. Figure 27은 이를 점수 분포 수준에서 보여 준다. chosen과 rejected의 보상 점수 차이가 더 분명해지고, reward model이 쉬운 비교에서 더 자신 있게 순위를 매긴다.


<img width="817" height="342" alt="image" src="https://github.com/user-attachments/assets/bfbfef6f-1194-4411-bd39-bc162872d7da" />


<img width="820" height="339" alt="image" src="https://github.com/user-attachments/assets/24e00322-3d9d-46aa-97c9-00b72f773f2b" />



#### A.3.4 Ablation on Ranking Loss with Safety Auxiliary Loss for Reward Modeling

안전성 보상모델에는 ranking loss 외에 안전 관련 보조 손실을 더했을 때의 효과가 따로 분석된다. 저자들은 이 보조 손실이 특히 위험 응답을 더 잘 구분하게 해 주며, 범주별 정확도 개선에도 도움이 된다고 보고한다.

Table 29의 핵심은 평균 정확도뿐 아니라 unsafe response recall이다. 안전성 보상모델의 목적은 두 응답 중 더 나은 쪽을 고르는 것만이 아니라, 실제로 위험한 응답을 확실히 낮은 점수로 밀어내는 것이다. 논문은 safety auxiliary loss를 넣었을 때 세 범주 모두에서 정확도가 올라가고, 0.5 임계치 기준으로 unsafe response를 잡아내는 비율도 좋아진다고 보고한다.


<img width="822" height="168" alt="image" src="https://github.com/user-attachments/assets/c64c64b5-b0f7-47a7-b01a-6a189528a39e" />



#### A.3.5 Additional Results for GAtt

GAtt의 추가 결과에서는 최대 20턴까지 제약을 유지하는 정량 결과와, 학습에 없던 제약에도 zero-shot으로 반응하는 예시가 제시된다. 저자들은 GAtt가 단순한 표면 문구 암기가 아니라, 시스템 수준 제약을 지속적으로 참조하게 만드는 장치라고 해석한다.


<img width="814" height="153" alt="image" src="https://github.com/user-attachments/assets/c7a4567c-0c9d-4171-a505-cacfbaa47966" />


<img width="823" height="527" alt="image" src="https://github.com/user-attachments/assets/c5f339bf-9c82-4de8-9ad5-85578a98931f" />


#### A.3.6 How Far Can Model-Based Evaluation Go?

이 절은 보상모델 점수가 사람의 세밀한 품질 평가와 어느 정도 맞아떨어지는지 살펴본다. 세 명의 사람 평가자가 7점 척도로 응답을 다시 평가한 결과와 보상모델 점수를 비교했을 때, 보상모델이 완벽하지는 않지만 실질적으로 유용한 point-wise 품질 신호를 제공한다고 결론짓는다.


<img width="824" height="409" alt="image" src="https://github.com/user-attachments/assets/1b4c76ec-34ff-486f-b39e-064e278bdd42" />



#### A.3.7 Human Evaluation

사람 평가 부록은 본문보다 훨씬 자세한 실험 설정을 제공한다. 어떤 시스템 프롬프트를 각 모델에 사용했는지, 단일 턴과 다중 턴 프롬프트 수를 어떻게 배분했는지, 어떤 예시 프롬프트와 실제 응답 비교가 들어갔는지까지 표로 정리한다. 이는 사람 평가 결과가 단지 숫자 하나가 아니라, 프롬프트 설계와 시스템 메시지 선택에 크게 영향을 받는다는 점을 보여 준다.

Table 31은 각 비교 모델에 실제로 어떤 system prompt를 붙였는지를 보여 주고, Table 32는 사람이 본 프롬프트 수를 구체적으로 적는다. 예를 들어 ChatGPT 비교에는 1,917개의 single-turn과 2,256개의 multi-turn prompt가, PaLM-chat에는 1,869개와 2,143개가 사용된다. Falcon, MPT, Vicuna는 multi-turn 수가 더 적다. 즉 본문 Figure 12의 승률은 “같은 prompt 수, 같은 system prompt 조건”의 순수 추상 비교가 아니라, 실제 서비스형 인터페이스를 최대한 공정하게 재구성한 조건부 결과다. Figure 30과 Figure 31이 system prompt, turn 수, 전체 word count에 따른 민감도를 따로 보여 주는 것도 그 때문이다.

또한 ChatGPT와의 비교에서 시스템 프롬프트의 차이가 승률에 영향을 줄 수 있다는 점, 턴 수와 응답 길이에 따라 승률이 크게 흔들리지 않는다는 점도 별도 분석으로 제시된다. 이 절은 사람 평가를 해석할 때, 모델 자체의 능력뿐 아니라 **평가 인터페이스 설계**가 결과에 미치는 영향을 반드시 봐야 한다는 메시지를 준다.


<img width="821" height="536" alt="image" src="https://github.com/user-attachments/assets/fb2cf184-6e44-4705-b3e8-ad6b5efef221" />


<img width="822" height="820" alt="image" src="https://github.com/user-attachments/assets/8ad64d49-18d3-4eff-b90e-3ee565008e35" />


<img width="815" height="331" alt="image" src="https://github.com/user-attachments/assets/12fecafb-538f-4407-9d75-44f1d9f47b8b" />


<!-- Table 34 -->
<img width="830" height="885" alt="image" src="https://github.com/user-attachments/assets/e8cb1f37-4a80-407c-a274-55767347f0a7" />


### A.4 Additional Details for Safety

#### A.4.1 Tension between Safety and Helpfulness in Reward Modeling

이 절은 도움성과 안전성이 실제로 충돌할 수 있음을 예시로 보여 준다. 어떤 응답은 매우 자세하고 유용하지만 안전성 기준에서는 낮은 점수를 받을 수 있고, 반대로 지나치게 짧고 조심스러운 응답은 안전하지만 도움성은 떨어질 수 있다. 저자들이 두 개의 보상모델을 분리한 이유가 여기서 다시 확인된다.

Figure 32의 산점도는 이 긴장을 매우 직관적으로 그린다. safe response 집합에서는 안전 점수는 높지만 도움성 점수가 낮은 샘플이 나타나고, unsafe response 집합에서는 도움성은 높아 보이지만 안전 점수가 낮은 샘플이 나타난다. Table 35는 이런 불일치의 구체 예시를 제시한다. 즉 어떤 응답은 정보를 많이 주기 때문에 helpfulness RM에는 높게 보이지만, 바로 그 정보가 위험해서 safety RM에는 낮게 보일 수 있다. 이 부록은 본문 3.2.2의 두-RM 설계가 임의 선택이 아니라 데이터 분포의 실제 충돌 구조에 대응한 것임을 보여 준다.


<!-- Figure 32 -->
> [Figure 32 삽입 위치]


<!-- Table 35 -->
> [Table 35 삽입 위치]


#### A.4.2 Qualitative Results on Safety Data Scaling

안전 데이터 비율을 높였을 때의 질적 변화도 구체 예시로 제시된다. 저자들은 더 많은 안전 데이터가 위험하거나 모욕적인 출력을 줄이는 데 분명히 도움이 되지만, 동시에 점점 더 보수적인 응답과 맥락 오해를 유발할 수 있음을 예시로 보여 준다.


<img width="602" height="847" alt="image" src="https://github.com/user-attachments/assets/ebb6b76f-114b-4b00-be0a-c851f707f79d" />


<img width="614" height="868" alt="image" src="https://github.com/user-attachments/assets/4c204ba9-6868-43ca-af0c-9c8b9290bc25" />



<img width="634" height="814" alt="image" src="https://github.com/user-attachments/assets/a44392a7-b045-4f5d-914a-e09b77485ff8" />



#### A.4.3 English Pronouns

이 절은 영어 대명사 분석을 보충 설명한다. 사전학습 데이터에서 특정 성별 대명사가 더 많이 등장한다는 사실이 왜 모델 표현 편향과 연결될 수 있는지, 그리고 단순 빈도 분석이 한계를 가지는 이유를 간단히 적는다.

#### A.4.4 Context Distillation Preprompts

여기서는 context distillation에 사용한 안전 preprompt들을 모아 보여 준다. 일부는 범용 안전 assistant 역할을 부여하고, 일부는 특정 위험 범주에 더 직접적인 answer template을 주는 방식이다.


<!-- Table 39 -->
> [Table 39 삽입 위치]


#### A.4.5 Safety Errors: False Refusals and Vague Responses

저자들은 안전성 개선이 부작용을 낳을 수 있다는 점을 정면으로 다룬다. context distillation이나 과도한 safety data는 응답을 지나치게 모호하게 만들거나, 실제로는 무해한 요청까지 거절하게 만들 수 있다. 이 절의 예시들은 안전성 튜닝이 언제나 일방향 개선이 아니라는 점을 보여 준다.

Figure 33은 안전 데이터 비율이 늘수록 false refusal도 증가하는 경향을 시각화한다. 즉, 안전성을 높이는 것과 과잉 거절을 줄이는 것은 서로 독립적이지 않으며, 실제 배포에서는 이 균형을 별도로 조정해야 한다는 메시지가 나온다.

<img width="629" height="580" alt="image" src="https://github.com/user-attachments/assets/b2e79e32-d136-41c7-888a-605318f79954" />


<img width="611" height="514" alt="image" src="https://github.com/user-attachments/assets/41e24c2b-657a-4823-aeb3-81cd1acb7510" />



<img width="630" height="295" alt="image" src="https://github.com/user-attachments/assets/37ca76c0-4df6-4174-bb9b-924f7d1d3992" />


#### A.4.6 Examples of Safety Evaluation

이 절은 안전성 평가에 실제로 사용된 프롬프트와, 동일 프롬프트에 대해 여러 모델이 어떻게 응답했는지 보여 준다. 본문 수치가 어떤 종류의 적대적 요청에서 나왔는지 감을 잡게 해 주는 부분이다.


<img width="556" height="435" alt="image" src="https://github.com/user-attachments/assets/e4ffa74b-671e-4ed2-8c3c-58f2fed74cfc" />



<!-- Table 43 -->
> [Table 43 삽입 위치]


#### A.4.7 Description of Automatic Safety Benchmarks

자동 안전성 벤치마크 설명 절에서는 TruthfulQA, ToxiGen, BOLD가 각각 무엇을 측정하는지 정리한다. TruthfulQA는 거짓되거나 오해를 부르는 답을 얼마나 피하는지, ToxiGen은 인구집단 관련 프롬프트에 대해 독성 출력을 얼마나 생성하는지, BOLD는 다양한 사회집단을 다룰 때의 감성 분포를 살핀다.

저자들은 이러한 벤치마크가 유용하지만 완전하지 않다고 말한다. 벤치마크는 특정 형식의 위험만 측정하며, 실제 배포 환경의 복합적 안전성 문제를 전부 대표하지 못한다.

#### A.4.8 Automatic Safety Benchmark Evaluation Results

이 절은 자동 안전성 벤치마크의 세부 결과를 모두 제시한다. TruthfulQA에서는 truthfulness와 informativeness를 함께 보고하고, ToxiGen에서는 인구집단별 독성 생성 비율을 나누어 본다. BOLD에서는 인종, 성별, 종교 이념, 정치 이념, 직업 도메인별 감성 분포를 따로 정리해, 특정 집단에서 감성 편향이 과도하게 나타나는지 확인한다.

Table 44를 보면 fine-tuned Llama 2-Chat의 TruthfulQA 수치는 7B 57.04, 13B 62.18, 34B 67.20, 70B 64.14로 34B까지는 뚜렷하게 상승하고 70B에서 약간 흔들린다. Table 45는 평균 독성 비율 하나만으로는 보이지 않는 subgroup 차이를 드러내기 위해 ToxiGen을 인구집단별로 쪼갠다. Tables 46~50도 같은 철학을 따른다. BOLD의 인종, 성별, 종교 이념, 정치 이념, 직업 도메인 각각에 대해 감성 분포를 분리해 보여 줌으로써, 안전성·편향 평가는 평균 한 줄로 끝낼 수 없다는 점을 강조한다. Table 50이 profession domain을 별도로 두는 것도 같은 이유다. 직업 범주에서는 사회적 고정관념과 감성 편향이 다른 축으로 나타날 수 있기 때문이다.

<img width="480" height="421" alt="image" src="https://github.com/user-attachments/assets/1ce23742-d6fb-44c6-9072-feda2dcf1b92" />


<img width="613" height="793" alt="image" src="https://github.com/user-attachments/assets/e80e5ddb-b1f1-4b53-a32d-449f98c0d194" />

<img width="597" height="403" alt="image" src="https://github.com/user-attachments/assets/7efbdd3a-720a-4be2-bf88-280e8321e6ec" />


<img width="639" height="786" alt="image" src="https://github.com/user-attachments/assets/994834b1-ee69-4b21-babf-b36b596a71a7" />


<img width="777" height="305" alt="image" src="https://github.com/user-attachments/assets/87320de6-8328-4a79-908e-c22472433bde" />


### A.5 Data Annotation

이 절은 데이터 주석의 운영 절차를 모아 설명한다. SFT, 선호 데이터, 안전성 데이터가 어떤 기준으로 수집되고 검수되었는지를 다룬다.

#### A.5.1 SFT Annotation Instructions

SFT 주석자는 정보성, 진실성, 관련성, 명확성, 무해성을 함께 만족하는 응답을 작성하도록 지시받는다. 만약 이 기준이 충돌하면, 저자들은 무해성을 우선하되 가능한 한 도움이 되도록 답을 구성하도록 요구한다.

#### A.5.2 Negative User Experience Categories

이 절은 피해야 할 부정적 사용자 경험 범주를 정리한다. 범죄 조장, 위험 행동 유도, 혐오·학대성 표현, 노골적 성적 내용, 명백히 오해를 부르는 정보 등이 여기에 포함된다. 이러한 범주는 주석자에게 무엇을 생성하지 말아야 하는지, 혹은 어떤 경우에 안전하게 방향을 바꿔야 하는지를 알려 주는 기준으로 쓰인다.

#### A.5.3 Quality Assurance Process

주석 품질 보증은 숙련된 콘텐츠 매니저의 검토를 통해 이뤄진다. 지침 준수 여부, 대화 문맥의 일관성, instruction following, 명백한 오류나 정책 위반 여부 등을 점검한 뒤 통과된 데이터만 학습에 사용한다.

#### A.5.4 Annotator Selection

주석자 선정에는 문법·독해·쓰기 능력 평가, 기준 부합성 테스트, 민감한 주제 대응 능력 확인 등 여러 단계가 포함된다. 저자들은 사람 데이터의 품질이 SFT와 RLHF 전체 성능을 크게 좌우한다고 보기 때문에, 주석자 선발과 교육을 중요한 공정으로 취급한다.

### A.6 Dataset Contamination

이 절은 평가 데이터 오염 가능성을 분석한다. 저자들은 단순 n-gram 중복 검출보다 더 세밀한 방식으로, 토큰화된 입력을 전개해 부분적 중복 비율을 계산하는 절차를 사용했다고 설명한다. 그 결과 충분한 근거가 확인된 일부 데이터셋에 대해서만 오염 결과를 보고한다.

저자들의 메시지는 두 가지다. 첫째, 오염 분석은 반드시 해야 하지만 과도하게 단순한 기준으로는 실제 오염을 정확히 측정하기 어렵다. 둘째, 일부 영향이 있더라도 전체 결론을 단순히 무효화하기보다는, 어느 벤치마크가 얼마나 영향을 받았는지를 구체적으로 보는 편이 낫다는 것이다.

Table 51은 그래서 의도적으로 보수적이다. 논문은 충분한 증거가 있는 데이터셋만 affected로 표기하고, 나머지는 단순 중복 가능성만으로 오염 판정을 내리지 않는다. 논쟁적으로 보일 수 있는 주제를 과장 없이 다루기 위해, ‘무엇이 오염되었는가’보다 ‘무엇을 오염되었다고 볼 만한 충분한 근거가 있는가’를 먼저 묻는 셈이다.


<img width="772" height="541" alt="image" src="https://github.com/user-attachments/assets/81ee2ae1-aef5-4527-bdd9-45c6392b6f5c" />



### A.7 Model Card

마지막 부록은 Llama 2 모델 카드다. 여기에는 모델 개요, 의도된 사용, 권장되지 않는 사용, 성능과 안전성 관련 요인, 알려진 한계, 책임 있는 사용 지침이 정리된다. 본문 전체의 기술적 설명이 실제 배포 문서로 어떻게 연결되는지를 보여 주는 마무리다.


<img width="571" height="838" alt="image" src="https://github.com/user-attachments/assets/8497ff17-b3cd-4fa0-8ea3-e21cf2e9efc7" />


## 추가 설명

이 논문의 핵심 기여는 모델 하나를 발표했다는 데만 있지 않다. 더 중요한 것은 공개형 foundation model을 실제 대화형 assistant로 바꾸는 공정을, 사전학습·SFT·보상모델·rejection sampling·PPO·safety tuning·red teaming·human evaluation이라는 단계로 나누어 비교적 투명하게 문서화했다는 점이다. 그래서 이 문서는 단순한 성능 보고서라기보다, 공개 생태계가 닫힌 제품형 모델에 접근하기 위해 어떤 공학적 절차를 밟았는지 보여 주는 제작 기록에 가깝다.

두 번째로 중요한 점은 도움성과 안전성을 하나의 점수에 억지로 합치지 않았다는 것이다. 논문은 helpfulness reward model과 safety reward model을 분리하고, 위험 프롬프트에는 safety reward를 우선 적용한다. 이는 “안전해질수록 덜 유용해질 수밖에 없다”는 단순 도식을 피하고, 실제 충돌이 생기는 위치를 별도 축으로 관리하려는 설계다. 부록의 Figure 32와 Table 35는 이 긴장이 이론적 가정이 아니라 실제 데이터 분포에서 관찰되는 현상임을 보여 준다.

세 번째는 RLHF를 단순한 마지막 미세조정 단계가 아니라, 분포 자체를 재형성하는 과정으로 본다는 점이다. Figure 20은 RLHF가 응답 분포의 왼쪽 꼬리, 즉 명백히 좋지 않은 답변들을 점차 제거한다는 사실을 보여 준다. Figure 21은 RLHF가 창작형 프롬프트와 사실형 프롬프트를 다르게 다루면서, 사실상 프롬프트 종류에 따라 다른 온도를 내부적으로 학습한다는 점을 시사한다. 이 논문이 흥미로운 이유는 바로 이런 현상을 단순 결과가 아니라 관찰 가능한 메커니즘으로 제시한다는 데 있다.

네 번째는 안전성 서술의 톤이다. 논문은 안전성을 달성했다고 선언하지 않는다. 어떤 데이터가 남아 있었고, 어떤 독성이 사전학습 코퍼스에 포함되었으며, 어떤 자동 벤치마크와 사람 평가로 그것을 측정했고, 어떤 false refusal이 생겼는지까지 함께 적는다. 그래서 이 문서는 “우리는 안전하다”는 홍보 문서보다, “어떤 완화가 얼마나 효과를 냈고 어떤 부작용이 남았는가”를 기록한 실험 문서에 더 가깝다.

마지막으로 이 논문은 공개 전략 자체를 기술적 설계의 일부로 다룬다. 가중치 공개, 상업적 사용 허용, 라이선스, Acceptable Use Policy, Responsible Use Guide, 모델 카드가 한 묶음으로 제시된다. 따라서 Llama 2 논문은 2023년 공개형 LLM 경쟁에서 성능 보고서이면서 동시에 배포 설계 문서이기도 하다. 성능, 정렬, 안전성, 릴리스 정책이 한 문서 안에서 함께 다뤄진다는 점이 이 논문의 가장 큰 역사적 의미다.
