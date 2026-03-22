# Mixture of Experts in Large Language Models

## Abstract

이 논문은 대규모 언어모델에서 Mixture-of-Experts(MoE) 아키텍처를 폭넓게 검토한다. 핵심 주장은, 전체 파라미터를 항상 모두 활성화하지 않고도 높은 모델 용량과 성능을 확보할 수 있다는 점이다. 저자들은 이론적 배경, 핵심 구조, 게이팅과 라우팅, 계층형·희소형 MoE, 메타러닝, 멀티모달·멀티태스크 학습, 실제 배포 사례, 그리고 최근의 한계와 도전 과제를 하나의 흐름으로 정리한다. 또한 MoE가 동등한 베이지안식 접근보다 더 큰 표현 용량을 가질 수 있고, 태스크 특화 성능과 확장 효율을 높일 수 있다고 본다. 다만 expert diversity, calibration, 신뢰할 수 있는 inference aggregation이 확보되지 않으면 MoE의 장점이 충분히 살아나지 않는다는 점도 함께 강조한다. 마지막으로 현재의 연구 공백, 남아 있는 난제, 그리고 앞으로의 유망한 연구 방향을 정리한다.

## Index Terms

Large language models, mixture of experts, expert routing, meta learning, knowledge transfer, sparse activation, large language models architecture, natural language processing

## I. Introduction and Fundamentals

지난 10여 년 동안 딥러닝 모델의 용량은 폭발적으로 증가했다. 특히 대형 Transformer의 등장은 성능을 크게 끌어올렸지만, 동시에 FLOPs와 에너지 사용량도 빠르게 키웠다. 파라미터 수를 선형으로 늘릴수록 실제 계산 비용은 더 가파르게 증가하고, 이 추세는 실서비스 환경에서 지속 가능하지 않다.

이런 배경에서 연구자들은 dense하고 단일한 모델 대신 sparse하고 modular한 계산 구조를 대안으로 보기 시작했다. 그중 Mixture-of-Experts(MoE)는 전체 파라미터를 항상 모두 쓰지 않고, 입력마다 일부 expert만 조건부로 활성화한다는 점에서 가장 유망한 구조로 제시된다. 이 방식은 전체 모델 크기와 추론 비용을 분리해, 총 용량은 크게 늘리되 실제 활성 계산은 제한할 수 있게 만든다.

MoE의 개념적 기원은 adaptive learning system의 초기 연구들로 거슬러 올라간다. 당시의 모델들은 입력 공간의 서로 다른 부분을 담당하는 여러 expert의 앙상블로 구성되었고, gating function이 각 입력을 가장 적절한 expert로 보내는 역할을 맡았다. 즉, MoE의 출발점은 단순한 파라미터 절약이 아니라, 입력 공간을 분할하고 전문가를 경쟁적으로 선택한다는 학습 원리였다.

다만 초창기에는 계산 자원과 학습 인프라의 한계 때문에 이런 구조가 대형 고성능 모델로 이어지지 못했다. MoE가 실제로 큰 의미를 가지기 시작한 것은 sparse routing이 현대 deep network와 결합되고, 분산 학습과 대규모 병렬화가 가능해진 뒤였다. 특히 sparsely gated network의 등장은 전체 성능을 유지하면서도 학습과 추론에서 일부 파라미터만 활성화할 수 있음을 보여주었다.

MoE는 단순히 더 적게 계산하는 방법이 아니라, 딥러닝 설계 자체를 바꾸는 구조적 전환으로 제시된다. 기존 dense 네트워크가 모든 입력과 태스크에 대해 모델의 모든 부분을 동일하게 중요하다고 가정한다면, MoE는 조건부 계산을 통해 각 expert가 특정 도메인, 언어 패턴, 혹은 모달리티에 특화되도록 만든다. 훈련 과정에서 경량의 gating network가 어떤 expert가 입력을 처리해야 하는지 결정하며, 이 과정은 expert 간 기능적 다양성을 유도하고 일반화와 견고성을 높이는 방향으로 작동한다.

이런 재조명은 모델이 커질수록 무조건 더 좋다는 단순한 스케일링 관점에 대한 반성 위에서 나온다. MoE는 구조와 계산 모두에서 이질성과 선택성을 받아들이며, 같은 규모의 dense 모델보다 더 유연한 대안을 제공한다. 언어모델, 기계번역, 비전-언어 추론 등 다양한 영역에서 MoE가 성공적으로 적용된 것은 이 접근이 특정 분야에 한정되지 않는다는 점을 보여준다.

[Figure 1 삽입]

*Figure 1. Mixture of Experts(MoE) 모델 발전 타임라인. 초기 개념 정립부터 현대의 대규모 구현까지 이어지는 핵심 이정표를 시간축 위에 정리한다.*

**From sparse gating to billion-scale deployment.** 2020년을 전후로 MoE는 개념적 구조에서 대규모 실전 구조로 넘어간다. 이 전환점의 대표가 GShard이다. GShard는 600B 규모의 다국어 모델로, auto-sharding과 token-level expert routing을 통해 초대형 sparse MoE가 실제로 가능하다는 것을 보여주었다. 이어 Switch Transformer와 GLaM은 이 패러다임을 언어모델링으로 확장했다. 이들은 입력당 1–2개의 expert만 활성화하는 token-choice gating을 사용하면서도, dense Transformer의 대안이 될 수 있는 확장성을 입증했다.

2021–2022년에는 MoE가 proof-of-concept 단계를 지나 널리 채택되는 계산 프레임워크로 성숙한다. Meta-MoE, CMP-MoE, V-MoE 같은 공개 연구들이 modular computation에 대한 관심을 넓혔고, 상업 및 도메인 특화 모델들도 빠르게 늘어났다. NLLB-MoE는 다국어 번역에, Swin-MoE는 vision task에, LIMoE는 multimodal learning에 적용되며, MoE가 NLP 바깥으로 확장되기 시작했다.

2023년 이후의 흐름은 산업 규모의 확장과 구조적 다변화로 요약된다. DeepSeekV3, Skywork 3.0, Arctic 같은 모델은 MoE가 현대 foundation model의 핵심 부품이 되었음을 보여준다. 여기에 MoE-LLaVA, MM1, Omni-SMoLA 같은 멀티모달 계열이 결합되면서, expert routing은 retrieval, instruction tuning, grounding, agentic control과 함께 더 복합적인 시스템 설계로 묶인다. 또한 Jamba, Qwen1.5-MoE, Mistral-8x22B 같은 오픈소스 모델이 등장하면서, 최근의 혁신은 단순 규모 경쟁보다 효율, 제어 가능성, 멀티태스크 일반화 쪽으로 이동한다.

논문은 2025년 이후의 관심사가 더 이상 파라미터 수 자체를 키우는 데 있지 않다고 본다. 이제 핵심은 장시간 학습과 실제 배포 조건 아래에서 routing을 얼마나 안정적으로 유지할 수 있는가이다. 비슷한 입력이 비슷한 expert를 고르게 만드는 similarity-preserving load balancing, expert-capacity 제약으로 발생하는 token dropping과 padding 낭비를 줄이는 MAXSCORE류 접근은, MoE가 연구용 장난감이 아니라 배포 가능한 계산 정책으로 다듬어지고 있음을 보여준다.

[Table I 삽입]

*Table I. 여섯 개 응용 도메인에 걸친 대표적 MoE 아키텍처 분류. expert 수, routing 전략, 등장 시기(2017–2024), 핵심 혁신 및 활용 사례를 함께 정리한다.*

| Category | Model (#Experts) | Routing | Year | Key Innovation / Use Case |
| --- | --- | --- | --- | --- |
| Language LLM | Switch Transformer (64), GLaM (64) | Token-choice | 2021–22 | 토큰당 1/64의 파라미터만 활성화하는 초거대 LLM |
| Translation | GShard MoE (128), DeepSpeed-MoE (256) | Token-choice | 2020–21 | Auto sharding과 pipeline parallelism으로 대규모 다국어 번역 처리 |
| Multimodal | Omni-SMoLA (16), T-REX2 (32) | Cross-modal gate | 2023–24 | 저랭크 expert와 dual vision-text prompt를 결합해 open-set detection 수행 |
| Computer Vision | MoCaE-DET (8), Deep-MoE (32) | Attention gate | 2017–23 | Calibration-aware fusion으로 COCO AP를 단일 detector 대비 2.5 향상 |
| Param-Efficient | LoRA-MoE (4), Nexus (8) | Frozen router | 2023–24 | 1% 미만 파라미터 업데이트로 PEFT를 수행하고 dense checkpoint를 adaptive MoE로 전환 |
| Hierarchical | H-MoE (32), MixER (10) | 2-level / Top-1 | 2024 | Coarse-to-fine quadratic gating과 K-means routing으로 계층적 expert 선택 구현 |

**Diversity of designs across modalities and tasks.** Table I가 보여주듯이 MoE는 이제 하나의 단일 구조가 아니라, 규모·라우팅 방식·도메인 특화가 서로 다른 넓은 설계 공간이 되었다. Switch Transformer와 GLaM은 language modeling에서 token-level sparse activation으로 수백억~수천억 규모 확장을 이끌었고, GShard와 DeepSpeed-MoE는 pipeline parallelism과 expert sharding을 결합해 번역과 시스템 처리량을 강조했다. Omni-SMoLA와 T-REX2 같은 multimodal 계열은 텍스트와 이미지 단서를 함께 사용해 open-set recognition과 grounded reasoning을 수행한다. 또 H-MoE와 MixER는 multi-stage routing을 통해 해석성과 모듈성을 높이려 하고, LoRA-MoE와 Nexus는 fine-tuning 비용을 최소화하는 방향으로 설계를 밀어간다.

이 표를 자세히 읽으면 논문의 시선이 분명해진다. MoE를 설명할 때 중요한 것은 단순히 expert 개수가 많은가가 아니라, 어떤 입력 단위를 기준으로 expert를 고르는가, 그리고 그 선택이 어느 응용 도메인에서 실질적인 장점을 주는가이다. 즉, 동일한 MoE라는 이름 아래에서도 목적 함수, 시스템 제약, 입력 구조가 달라지면 최선의 설계는 전혀 달라진다.

**Routing complexity and the stability efficiency tradeoff.** 모든 MoE의 중심에는 routing algorithm이 있다. 초기 설계는 softmax 기반 top-\(k\) selection을 사용했지만, 이후 entropy 기반, auxiliary load balancing 기반, differentiable attention 기반 routing이 계속 제안되었다. 목표는 expert diversity를 확보하면서도 중복 계산을 줄이는 것이다. 그러나 게이트가 지나치게 자신감을 가지면 일부 expert로 붕괴하고, 반대로 너무 균등하게 보내면 specialization이 약해진다. 따라서 routing의 표현력, 안정성, 학습 복잡도 사이의 긴장은 MoE 전체를 관통하는 핵심 문제로 남는다.

**Deployment constraints.** MoE는 계산량 절감이라는 장점이 있지만, 실제 배포에서는 irregular memory access, cross-device communication, unstable batching 같은 문제를 만든다. sparse activation은 하드웨어 친화적이지 않은 경우가 많고, routing의 stochasticity는 재현성과 latency를 해친다. 그래서 최근 구조들은 scale만이 아니라 serving compatibility를 함께 고려한다. LoRA-MoE와 Nexus처럼 router를 고정하거나 low-rank adapter를 쓰는 설계는 inference variance를 줄이고 caching과 fine-tuning을 단순화한다.

**Scope of this work.** 이 논문은 먼저 expert module, gating, load balancing 같은 핵심 구성요소를 정리하고, 이어서 NLP, computer vision, multimodal learning 등 분야별 적용을 다룬다. 그리고 routing instability, expert underutilization, scalability limitation 같은 기술적 난제를 검토한 뒤, sparse mixture fusion, expert replay, lifelong modularity 같은 미래 방향까지 연결한다. 즉, 이 논문 전체는 MoE가 어떻게 발전했고, 지금 어디에 쓰이며, 앞으로 무엇이 막히는가를 구조적으로 정리하려는 서베이다.

## II. Core Architectures and Routing Mechanisms

이 절은 MoE의 핵심 아키텍처와 라우팅 메커니즘을 다룬다. 저자들의 관심은 expert module 자체보다, expert를 어떻게 고르고, 얼마나 sparsely 활성화하며, 그 과정에서 specialization과 efficiency를 어떻게 동시에 확보할 것인가에 있다. Figure 2는 이를 일곱 개 축으로 정리한 큰 지도 역할을 한다.

[Figure 2 삽입]

*Figure 2. Mixture of Experts(MoE) 모델의 종합 taxonomy. 언어 모델, 멀티모달 모델, 아키텍처 혁신, 학습 전략, 라우팅 메커니즘, 응용 시나리오, 도전과제의 일곱 범주로 방법론과 대표 모델을 조직한다.*

이 분류도는 먼저 Language Models 축에서 일반 LLM과 특화 LLM을 나눈다. 일반 LLM 범주에는 Mixtral, Qwen2-MoE, DeepSeek-MoE, Switch Transformer 같은 오픈소스 계열과 GPT-4(논문에서는 루머 수준으로 언급), PaLM-2, GLaM, DBRX 같은 상업·대표 계열, 그리고 GShard, ST-MoE, Expert Router 같은 연구 계열이 함께 놓인다. 특화 LLM 축에는 NLLB-200, NLLB, Multilingual-MoE 같은 번역 모델, DeepSeek-Coder-MoE와 CodeT5-MoE 같은 코드 생성 모델, MoGU와 BioBERT-MoE 같은 도메인 특화 모델이 배치된다.

Multimodal Models 축에서는 LIMoE, MoE-LLaVA, LLaVA-MoLE 같은 vision-language understanding 계열과 Omni-SMoLA, DALLE-MoE 같은 generation 계열이 분리되고, Computer Vision 하위에는 V-MoE, Swin-MoE, pMoE 같은 image classification 계열과 ADVMOE 같은 adversarial robustness 계열이 묶인다. Multi-task Unified 범주에는 Uni-MoE, MM1, MoCLE처럼 하나의 구조 안에서 여러 태스크를 통합하려는 시도가 놓인다.

Architectural Innovations 축은 expert selection, structural variants, optimization techniques의 세 갈래로 세분된다. 여기에는 Standard Top-K와 Adaptive Top-K, Weighted Routing과 Probabilistic Selection 같은 선택 방식, HMoE·Nested MoE·OMoE·MoCaE 같은 구조 변형, 그리고 DeepSpeed-MoE·FasterMoE·HyperMoE 같은 효율화 기법이 함께 들어간다. Training Strategies 축은 expert initialization, end-to-end 혹은 stage-wise training, expert dropout·noise injection·sparsity regularization 같은 학습 방법을 포괄한다.

Routing Mechanisms 축은 learned gating, attention-based gating, hash-based routing, load balancing, adaptive expert selection, context-aware routing을 묶는다. Application Scenarios 축은 NLP, computer vision, multimodal task, scientific computing으로 이어지고, Challenges & Outlook 축은 scalability issue, training challenge, deployment challenge, future direction을 정리한다. 즉, Figure 2는 MoE를 단일 아키텍처가 아니라 연구 축이 여러 갈래로 분기되는 방법론 생태계로 파악하게 만든다.

**Core concepts and mathematical principles.** 하나의 MoE layer는 입력 \(x\)를 전체 \(N\)개 expert 중 일부 \(k\)개로만 보낸다. 따라서 출력은 모든 expert의 dense 평균이 아니라, 선택된 소수 expert의 가중합으로 계산된다.

$$
y=\sum_{i=1}^{N} g_i(x)E_i(x) \tag{1}
$$

여기서 \(E_i(x)\)는 \(i\)번째 expert의 출력이고, \(g_i(x)\)는 최대 \(k\)개 expert에만 비영 값을 주는 gating function이다. 다시 말해 MoE의 핵심은 expert를 많이 두는 데 있지 않고, 그중 아주 일부만 활성화하는 sparse dispatch 규칙에 있다. 식 (1)은 그 규칙이 출력 수준에서 어떻게 구현되는지를 보여준다.

게이팅은 보통 Noisy Top-\(k\) routing으로 구현된다.

$$
H(x)_i=(x\cdot W_g)_i+\mathcal{N}(0,\sigma^2) \tag{2}
$$

여기서 \(W_g\)는 입력을 expert score로 사상하는 gating weight이고, \(\mathcal{N}(0,\sigma^2)\)는 score에 더해지는 잡음이다. 이 노이즈는 초기 학습에서 특정 expert가 지나치게 우세해지는 현상을 막고, 덜 선택되던 expert도 한동안 탐색되게 만들어 준다. 동시에 top-\(k\) sparsity는 전체 expert 수가 아니라 실제 선택된 expert 수에 비례하는 계산 비용을 가능하게 한다. 그래서 MoE의 희소 활성화는 device 간 병렬화와 함께 대규모 학습에 잘 맞는다.

### II-A. Foundational MoE Architectures

**Sparse activation via gating networks.** 각 MoE layer는 여러 expert와, 입력마다 소수 expert를 고르는 gating network로 구성된다. expert는 보통 서로 독립적인 feedforward module이며, gate는 각 expert에 대한 relevance score를 계산한다. 이 점수는 Gaussian noise와 soft activation term을 포함한다.

$$
H(x)_i=(x\cdot W_g)_i+\mathcal{N}(0,1)\cdot \operatorname{Softplus}((x\cdot W_n)_i) \tag{3}
$$

$$
G(x)=\operatorname{Softmax}(\operatorname{TopK}(H(x),k)) \tag{4}
$$

여기서 \(\operatorname{TopK}\)는 가장 큰 \(k\)개를 제외한 score를 \(-\infty\)로 마스킹한 뒤 softmax를 적용하는 구조다. 식 (3)은 라우팅 score를 만들고, 식 (4)는 그 score를 실제 dispatch probability로 바꾼다. 즉, score 생성 단계와 sparse selection 단계가 분리되어 있다는 점이 중요하다. 이 분리 덕분에 모델은 탐색과 선택을 동시에 조절할 수 있다.

[Figure 3 삽입]

*Figure 3. decoder-only Transformer에서 sparsely gated MoE가 어떻게 들어가는지 보여주는 예시. 이 설정에서는 \(k=2\)인 top-\(k\) routing을 사용하며, router softmax 확률이 가장 높은 두 FFN expert를 각 token마다 선택한다. 선택된 expert는 병렬로 계산되고, 출력은 가중합으로 결합된다.*

Figure 3가 시각화하는 핵심은 self-attention 뒤의 FFN 자리가 더 이상 하나의 동일한 feedforward block이 아니라는 점이다. 같은 token이라도 어떤 token은 FFN1과 FFN4를, 다른 token은 FFN2와 FFN3를 사용할 수 있다. 따라서 MoE는 Transformer의 블록 구조를 깨뜨리지 않으면서도, FFN 단계에서 token별 계산 경로를 달리하는 방식으로 거대한 용량을 확보한다.

By evaluating only a small subset of experts per token, MoE models achieve substantial computational savings and parallel scalability. Large-scale implementations such as Switch Transformer, GShard, and DeepSpeed-MoE demonstrate that this design enables efficient training of models with hundreds of billions of parameters.

**Load balancing objectives.** MoE 학습의 대표적 문제는 expert collapse다. 일부 expert만 대부분의 token을 받고 나머지는 거의 쓰이지 않는 현상이다. 이를 막기 위해 auxiliary load balancing loss를 추가한다.

$$
L_{\text{balance}}=\alpha\sum_{i=1}^{N} f_iP_i \tag{5}
$$

여기서 \(f_i\)는 expert \(i\)에 실제로 할당된 token 비율이고, \(P_i\)는 평균 gate probability다. 이 손실은 기대 사용량과 실제 사용량이 크게 어긋나는 경우를 벌점으로 준다. 결과적으로 expert starvation을 줄이고 자원을 더 고르게 쓰게 만든다. 다만 balance를 너무 강하게 강제하면 routing accuracy나 specialization이 약해질 수 있으므로, \(\alpha\)의 선택은 중요한 trade-off로 남는다.

이 식의 의미를 더 풀어 쓰면, MoE는 원래 특정 expert가 특정 입력 패턴에 강하게 특화되길 바라지만, 동시에 너무 특화된 나머지 나머지 expert가 놀게 되면 전체 시스템의 용량이 낭비된다. 식 (5)는 바로 그 모순을 조절하는 장치다. 잘 설계된 MoE는 완전한 균등 분배도 아니고, 완전한 편중도 아닌 상태에서 가장 높은 효율을 낸다.

**From prototypes to production.** 개념적 sparse 구조를 실제 서비스 가능한 아키텍처로 옮기기 위해서는 communication overhead, memory fragmentation, inference latency 같은 시스템 문제가 해결되어야 했다. GShard는 auto-sharded tensor computation으로 expert parallelism을 보여주었지만, 큰 인프라 지원이 필요했다. 이후 Mixtral은 static top-2 routing과 fused attention layer로 통신 비용을 줄였고, DBRX는 fused MoE kernel과 memory prefetching으로 추론 효율을 높였다. Qwen2와 DeepSeek-v3는 quantized MoE layer와 expert dropout을 결합해 추론 비용을 낮춘다. 이 흐름은 MoE가 연구용 prototype에서 production-ready backbone으로 이동하고 있음을 보여준다.

**Theoretical capacity and scaling behavior.** MoE의 표현력은 여러 subnetworks의 조합으로 형성되는 composite hypothesis space로 설명된다. 논문은 전체 MoE 가설 공간이 expert hypothesis class와 gating space의 결합으로 확장된다고 본다.

$$
\mathcal{H}_{\text{MoE}}
=
\bigcup_{g\in\mathcal{G}}
\left\{
\sum_{i=1}^{N} g_i(x)E_i(x)\;\middle|\;E_i\in\mathcal{H}_i
\right\} \tag{6}
$$

핵심은, MoE가 modular specialization을 통해 높은 용량을 가지지만, routing이 불안정하면 공유된 inductive bias 부족 때문에 generalization이 약해질 수도 있다는 점이다. 그럼에도 대규모 실험에서는 MoE가 dense 모델과 비슷한 성능을 내면서도 token당 활성 파라미터 수를 크게 줄이는 경우가 많아, 계산 효율과 표현 전문화 사이의 실질적 균형점을 제공한다.

식 (6)은 MoE를 단순히 여러 expert의 평균이 아니라, gating function이 고르는 하위 가설들의 합집합으로 보게 만든다. 그래서 논문은 MoE의 장점을 더 많은 파라미터 그 자체가 아니라, 더 넓은 함수 공간을 조건부로 탐색할 수 있는 능력으로 해석한다.
### II-B. Advanced Architectural Variants

**Orthogonal training.** expert 간 중복을 줄이고 specialization을 강하게 유도하기 위해 OMoE는 expert weight 사이의 정렬을 벌점으로 주는 orthogonal regularization을 도입한다.

$$
L_{\text{orth}}=\sum_{i\neq j}\langle W_i,W_j\rangle^2 \tag{7}
$$

이 손실은 서로 다른 expert가 비슷한 방향의 표현을 학습하는 것을 막아, 기능적 중복을 줄이는 데 목적이 있다. 즉, 서로 다른 expert는 실제로 다른 일을 하도록 만들자는 발상이다. 이 식은 diversity를 우연한 부산물로 남겨 두지 않고, 아예 목적함수 안에 넣어 직접 최적화하려는 시도로 이해할 수 있다.

**Mutual distillation and expert interaction.** 반대로 expert들 사이의 지식 공유를 강화하려는 방향도 있다. MoDE와 관련 연구들은 expert output 사이의 KL divergence를 활용한 mutual distillation을 도입한다.

$$
L_{\text{distill}}
=
\sum_{i=1}^{N}\sum_{j\neq i}
\operatorname{KL}\big(E_i(x)\,\|\,E_j(x)\big) \tag{8}
$$

이 접근은 각 expert가 다른 expert의 지식을 일부 받아들이게 만들어 robustness를 높인다. 그러나 distillation이 너무 강하면 expert들이 서로 닮아가며 homogenization이 발생할 수 있으므로, diversity와 consistency 사이의 긴장을 조절해야 한다. 다시 말해 식 (7)이 expert를 서로 밀어내는 힘이라면, 식 (8)은 expert를 부분적으로 서로 끌어당기는 힘이다. 논문은 이 두 방향이 경쟁 관계가 아니라, 어떤 수준에서 균형을 이룰 때 가장 좋은 MoE가 만들어진다고 본다.

**Parameter efficient tuning.** 초기 MoE는 fine-tuning 시 모든 expert를 저장하고 갱신해야 해서 parameter-efficient tuning과 잘 맞지 않았다. 이를 완화하기 위해 일부 expert head나 low-rank adapter만 업데이트하는 방식이 제안되었다. 이때 실제 업데이트되는 파라미터 비율은 다음처럼 표현된다.

$$
\rho=\frac{\|\Delta \theta_{\text{train}}\|_0}{\|\theta_{\text{full}}\|_0}\ll 1 \tag{9}
$$

여기서 \(\Delta \theta_{\text{train}}\)은 실제로 학습되는 파라미터, \(\theta_{\text{full}}\)은 전체 파라미터다. MoCE-IR, Adamix류 접근은 shared layer를 고정하고 expert별 경량 파트만 업데이트함으로써 sparse computation의 장점을 유지하면서도 tuning 비용을 크게 줄인다. 논문이 이 식을 강조하는 이유는 분명하다. LLM 시대의 MoE는 잘 훈련하는 것만큼, 얼마나 싸게 적응시키는가도 중요한 문제이기 때문이다.

**Hierarchical and multi-head extensions.** 더 큰 expert pool과 더 유연한 specialization을 위해 HMoE는 2단계 gating을 사용한다. 먼저 coarse gate가 super-expert group을 고르고, 그 안에서 finer gate가 최종 expert를 선택한다.

$$
y=\sum_{i\in G^{(1)}(x)} G_i^{(2)}(x)E_i(x) \tag{10}
$$

이 구조는 coarse-to-fine selection을 통해 routing overhead가 폭발하지 않도록 하면서, 서로 다른 추상화 수준에서 specialization을 가능하게 만든다. 논문은 multi-head MoE도 함께 언급한다. 이 구조는 서로 다른 입력 차원이나 태스크별로 다른 expert subset을 병렬로 할당하며, vision이나 speech처럼 공간·시간 분해가 자연스러운 영역에서 유리하다. 결국 계층형과 multi-head 확장은 더 많은 expert를 두는 것보다, 어떤 수준에서 expert를 조직할 것인가에 초점을 맞춘다.

**Heterogeneous and adaptive experts.** 최근 MoE는 모든 expert가 같은 구조와 비용을 가진다는 가정을 넘어서기 시작했다. 각 expert가 depth, width, modality 같은 서로 다른 computational profile \(\phi_i\)를 가지게 하고, router가 입력 난이도에 따라 더 비싼 expert 혹은 더 가벼운 expert를 선택하게 하는 것이다.

$$
g(x)=\arg\max_i\Big[S_i(x)-\lambda\cdot \operatorname{Cost}(\phi_i)\Big] \tag{11}
$$

이 cost-aware routing은 성능 점수와 계산 비용을 함께 고려하므로, specialization과 hardware efficiency를 동시에 노린다. 복잡한 입력은 더 큰 expert로, 단순한 입력은 더 저렴한 expert로 보내는 식의 serving-aware design이 가능해진다. 이 식은 MoE가 추상적 모델 이론에서 실제 시스템 설계로 옮겨 가는 지점을 잘 보여준다.

**Knowledge integration from unselected experts.** HyperMoE는 top-\(k\)로 뽑히지 않은 expert가 완전히 놀게 되는 문제를 완화하려고 한다. 선택되지 않은 expert의 intermediate signal을 요약해 활성 expert의 출력에 추가로 반영하는 식이다.

$$
y=
\sum_{i\in A(x)} g_i(x)E_i(x)
+
\gamma\sum_{j\notin A(x)} h_j(x) \tag{12}
$$

여기서 \(A(x)\)는 active expert set이고, \(h_j(x)\)는 inactive expert의 side information을 인코딩한 신호다. 이 방식은 full expert evaluation 없이도 더 넓은 지식을 간접적으로 끌어와 multitask나 low-resource setting에서 generalization을 개선하려는 시도로 이해할 수 있다. 즉, 선택되지 않은 expert를 완전히 버리는 대신, 비용이 낮은 보조 신호로 재활용하는 것이다.

### II-C. Routing Strategies and Specialization Patterns

**Token choice vs. expert choice.** Sparse routing은 크게 Token Choice와 Expert Choice 두 패러다임으로 나뉜다. Token Choice에서는 각 token이 자신이 갈 top-\(k\) expert를 선택한다.

$$
y_t=\sum_{i\in A_t} g_i(x_t)E_i(x_t),
\qquad
A_t=\operatorname{TopK}_{i=1}^{N}[g_i(x_t)] \tag{13}
$$

예를 들어 어떤 token은 Expert 1과 3을 고르고, 다른 token은 Expert 2와 4를 고를 수 있다. 즉, routing의 제어권이 token에 있다. 이 방식은 입력마다 가장 적절한 계산 경로를 세밀하게 고를 수 있다는 장점이 있지만, 전체 expert 부하가 어떻게 분포될지는 token 쪽 결정이 누적된 결과로 나타난다.

Expert Choice에서는 반대로 각 expert가 고정된 budget \(B\) 아래 자신이 처리할 token 집합을 선택한다.

$$
y_t=\sum_{i:x_t\in T_i}\tilde g_i(x_t)E_i(x_t),
\qquad
T_i=\operatorname{TopB}_{t=1}^{T}[s_i(x_t)] \tag{14}
$$

이 구조는 expert가 자신의 workload를 직접 조절하므로, load balancing과 token grouping에서 장점을 가진다. 논문은 이 차이를 단순 구현 차이로 보지 않고, control flow 자체가 달라지는 구조적 차이로 본다. Token Choice가 입력 단위 최적화에 가깝다면, Expert Choice는 자원 단위 최적화에 가깝다.

[Figure 4 삽입]

*Figure 4. Token Choice와 Expert Choice MoE 라우팅 전략 비교. (A) Token Choice에서는 각 token이 affinity score를 기준으로 자신에게 가장 적합한 expert를 고른다. 예시에서는 “We”가 Expert 1과 Expert 3으로, “Like”가 Expert 3과 Expert 4로 확률 가중치와 함께 라우팅된다. (B) Expert Choice에서는 각 expert가 고정된 예산 안에서 선호하는 token을 선택한다. 예시에서는 Expert 1이 ["We", "Love", "To", "Study"]를, Expert 2가 ["We", "Love", "Quite", "Library"]를 처리해 workload를 균형 있게 분배하면서도 token이 필요하면 여러 expert에 의해 처리될 수 있음을 보여준다.*

Figure 4는 라우팅의 제어권이 어디에 놓이는지가 시스템 성질 전체를 바꾼다는 점을 시각적으로 설명한다. Token Choice는 입력에 맞는 expert를 더 직접적으로 찾지만, Expert Choice는 expert 예산과 병렬 처리 효율을 더 안정적으로 관리한다. 그래서 논문은 이 비교를 단순 알고리즘 비교가 아니라, MoE를 어떤 계산 체계로 볼 것인가의 차이로 해석한다.

**Learned routers vs. Fixed routers.** 일반적으로는 router를 학습시키는 것이 당연해 보이지만, 최근 연구는 무작위 초기화된 fixed router가 learned router와 비슷하거나 더 나은 성능을 보일 수 있다고 보고한다. 이 경우 gate는 학습 가능한 함수라기보다 초기화 시점에 정해진 sparse mask가 된다.

$$
g_i(x)=
\begin{cases}
1, & i\in A_{\text{fixed}}(x) \\
0, & \text{otherwise}
\end{cases} \tag{15}
$$

이 접근은 특히 초기 학습에서 routing instability와 gradient-induced variance를 줄인다는 장점이 있다. 또한 learned routing이 항상 specialization을 더 잘 만든다는 가정 자체를 흔든다. 논문이 이 결과를 중요하게 다루는 이유는 명확하다. MoE의 성능 향상이 정말로 “똑똑한 router”에서 오는지, 아니면 “희소한 모듈 분해” 그 자체에서 오는지를 다시 묻게 만들기 때문이다.

routing granularity 역시 중요하다. sequence-level gating은 입력 전체에 대해 한 번 routing을 결정하므로, expert가 topic이나 discourse structure 같은 더 거친 수준에 특화되기 쉽다. 반면 token-level gating은 명사·동사 같은 syntactic category와 더 잘 정렬되는 미세 specialization을 만든다. 즉, router를 학습하느냐의 문제와 별개로, 어떤 단위에서 routing하느냐가 expert가 학습하는 기능의 종류를 결정한다.

**Emergent linguistic structure in expert assignments.** probing 연구는 MoE layer가 명시적 supervision 없이도 POS나 형태소 역할에 따라 token을 암묵적으로 클러스터링한다는 점을 보여준다. 논문은 expert \(i\)에 배정된 token 집합을 \(S_i\)로 두고, POS와 expert assignment의 mutual information을 다음처럼 적는다.

$$
S_i=\{x_t\mid g_i(x_t)>0\},
\qquad
I(\mathrm{POS};\mathrm{Expert}_i)=H(\mathrm{POS})-H(\mathrm{POS}\mid S_i) \tag{16}
$$

즉, 특정 expert에 들어간 token만 보아도 품사 정보를 상당히 알 수 있다면, routing은 단순한 임의 분기가 아니라 언어학적 구조를 반영하는 specialization을 형성한 것이다. 저자들은 이런 해석 가능성이 modular debugging, domain adaptation, controllable generation에도 도움이 된다고 본다. 이는 MoE가 단지 효율적인 모델이 아니라, 내부 구조를 더 잘 들여다볼 수 있는 모델일 수도 있음을 시사한다.

**Adaptive expert selection.** 일부 adaptive MoE는 입력 복잡도에 따라 활성 expert 수 자체를 바꾸는 입력-의존적 capacity control을 도입한다.

$$
k(x)=\min\big(K_{\max},\lfloor \tau\cdot \|x\|\rfloor\big) \tag{17}
$$

여기서 \(\tau\)는 scaling coefficient이고, \(\|x\|\)는 norm, entropy, difficulty proxy 같은 입력 복잡도 지표다. 복잡하거나 애매한 입력에는 더 많은 expert를 쓰고, 단순한 입력에는 더 가벼운 경로를 택하게 만드는 것이다. 이 방식은 multimodal setting이나 domain shift 상황에서 sample efficiency와 robustness를 높일 수 있으며, continual learning에서도 오래된 지식을 보존하면서 새로운 입력 분포에 적응하는 데 도움이 된다고 정리된다.

식 (17)은 MoE가 더 이상 expert 수를 고정한 정적인 희소 구조가 아니라, 필요에 따라 계산량 자체를 조절하는 동적 시스템으로 가고 있음을 보여준다. 이는 추론 시간, 하드웨어 예산, 입력 난이도가 모두 달라지는 실제 배포 환경에서 특히 중요하다.

## III. Meta-Learning and Knowledge Transfer in MoE

앞 절이 MoE의 구조적 토대를 설명했다면, 여기서는 한 단계 더 나아가 MoE가 어떻게 더 빨리 적응하고, 어떻게 지식을 옮길 수 있는가를 다룬다. 질문은 단순하다. expert를 잘 나누고 고르는 것만으로 충분한가, 아니면 태스크 분포 전체를 보며 라우팅 정책 자체를 meta-level에서 학습해야 하는가?

### III-A. Meta-Learning Framework Design

meta-learning은 MoE가 새로운 태스크를 만날 때 처음부터 다시 훈련하지 않고도 빠르게 일반화하도록 돕는다. 논문은 task-specific routing을 각 태스크마다 따로 학습하는 대신, 여러 태스크 분포 \(\mathcal{T}\) 전반에 걸친 routing policy를 최적화해 새로운 태스크에서도 빠르게 적응하는 구조를 강조한다.

$$
\theta_{T_{\text{new}}}=\theta-\eta\nabla_{\theta}L_{\text{support}}(\theta) \tag{18}
$$

여기서 \(\theta\)는 meta-learned parameter 혹은 routing policy이고, \(L_{\text{support}}(\theta)\)는 sparse expert output으로부터 유도된 support loss다. 새 태스크가 들어오면 support set에서 한두 번의 gradient step만으로 빠른 적응을 유도하는 방식이다. 논문은 이 식을 통해 MoE가 단지 입력마다 expert를 고르는 모델이 아니라, 태스크 분포를 기준으로 자신이 어떤 방식으로 고를지도 학습할 수 있는 모델임을 보여준다.

**Hierarchical meta-learning with MixER.** MixER(Mixture of Expert Reconstructors)는 nested dynamical system을 다루기 위해 고안된 구조로, 기존 MoE layer의 router에 입력 \(x\)뿐 아니라 context vector \(\xi\)도 함께 넣는다. 그리고 softmax gating 대신, K-means에 가까운 objective로 discrete expert selection을 수행한다.

$$
z(x,\xi)=\arg\min_j \left\|f_{\theta}(x,\xi)-\mu_j\right\|^2 \tag{19}
$$

여기서 \(f_{\theta}(x,\xi)\)는 latent space로의 매핑이고, \(\mu_j\)는 expert \(j\)의 prototype이다. 이 구조는 top-1 routing을 가능하게 하고, 왜 이 expert가 선택되었는가를 latent space에서의 cluster assignment로 해석하게 만든다. 동시에 미분가능한 soft selection의 오버헤드를 피한다. 다만 contextual hierarchy가 약하면 token-to-expert assignment가 겹치며 specialization이 무너질 수 있다는 한계도 논문은 바로 언급한다.

[Figure 5 삽입]

*Figure 5. standard MoE와 MixER layer의 아키텍처 비교. (A) Standard MoE에서는 입력 \(x\)가 gating network를 거쳐 routing 결정을 만들고, 선택된 expert의 출력이 전통적인 softmax-weighted 합으로 결합된다. (B) MixER에서는 입력 \(x\)와 추가 context vector \(\xi\)가 함께 router에 들어가며, 이 구조는 기존 softmax-weighted output combination을 제거한다.*

Figure 5는 MixER가 왜 Section III에 놓이는지를 보여 준다. 이 구조의 차별점은 단순히 router 입력이 하나 더 늘었다는 데 있지 않다. context를 함께 받아 discrete cluster 선택에 가깝게 동작함으로써, MoE가 입력 단위의 희소 선택을 넘어 태스크 구조나 시스템 상태를 함께 반영하는 적응형 라우팅으로 이동한다.

**Meta-distillation for domain adaptation.** Meta-DMoE는 domain shift를 test-time adaptation 문제이자 meta-distillation 문제로 본다. 서로 다른 source domain에서 사전학습된 domain-specific expert들의 출력을 transformer 기반 aggregator가 묶고, 그 결과로 가벼운 student model을 지도한다.

$$
L_{\text{meta}}
=
\operatorname{KL}\Big(
S(x)\ \|\ A(E_1(x),\ldots,E_N(x))
\Big) \tag{20}
$$

aggregator는 expert output 사이의 inter-domain dependency를 학습하고, meta-optimization은 이 지식이 unseen target domain으로 옮겨질 수 있게 만든다. 따라서 domain label이 없거나, heterogeneous domain을 하나의 통합 dense 모델로 학습하는 것이 비효율적인 상황에서 더 나은 일반화를 기대할 수 있다. 이 식은 MoE가 expert를 직접 쓰는 구조일 뿐 아니라, 여러 expert의 지식을 압축해 다른 모델로 옮기는 교사 집합으로도 기능할 수 있음을 보여 준다.

### III-B. Knowledge Transfer Mechanisms

**Sparse-to-Dense knowledge integration.** sparse MoE의 지식을 dense model로 이전하는 일은 쉽지 않다. 논문은 이를 해결하는 방향으로 multi-teacher distillation을 소개한다. 여러 expert teacher가 하나의 dense student를 함께 가르치며, student는 분산된 전문 지식을 통합해 일반화 능력을 키운다. 이 프레임워크는 knowledge gathering 단계와 knowledge distillation 단계로 나뉘고, knowledge gathering에는 summation, averaging, Top-KG, SVD-KG의 네 가지 방식이 제시된다.

결과적으로 dense student OneS는 ImageNet에서 MoE 이점의 61.7%를 유지하면서도 15M 파라미터로 78.4% top-1 accuracy를 달성한다. NLP 데이터셋에서는 MoE 이점의 88.2%를 유지하고 baseline 대비 51.7% 더 좋은 성능을 보였으며, 추론 속도는 MoE 대비 3.7배 빨라진다. 논문의 요지는, sparse 구조가 하드웨어와 잘 맞지 않는다는 문제를 multi-teacher → single-student 구조로 우회할 수 있다는 것이다.

**Mutual distillation among experts.** MoE expert는 각자 제한된 샘플만 보기 때문에 학습 범위가 좁아질 수 있다. 이 문제를 다루기 위해 MoDE는 moderate mutual distillation을 expert 간에 도입한다. gate가 expert를 좁은 하위분포로 제한하더라도, 서로의 출력을 약하게 증류하면서 다른 expert가 배운 feature를 받아들이게 만드는 것이다. 논문은 이를 통해 tabular, NLP, computer vision 전반에서 성능과 robustness가 좋아졌다고 정리하며, expert probing 연구를 통해 적절한 정도의 distillation이 개별 expert와 전체 MoE 성능을 함께 높인다고 본다.

### III-C. System Platform Support

meta-learning 기반 MoE는 모델 아이디어만으로는 충분하지 않고, 이를 재사용 가능한 시스템 구성요소로 바꾸는 플랫폼이 필요하다. AwesomeMeta+는 이를 위해 core meta-learning component를 reusable, configurable module로 묶는 prototyping platform을 제안한다. 이 플랫폼은 task conditioning, gradient aggregation, adaptation loop 같은 반복적 구조를 composable unit로 추상화하여, 실험을 더 확장 가능하고 재현 가능하게 만든다.

논문은 AwesomeMeta+의 핵심 기여를 세 가지로 정리한다. 첫째, task descriptor를 expert selector에 연결하는 declarative model interface. 둘째, 자원 제약 아래 expert instantiation을 최적화하는 scheduler. 셋째, few-shot accuracy나 expert stability 같은 transferability metric을 추적하는 evaluation monitor다. 50명 이상의 연구자 피드백과 benchmark 결과를 통해, 이 플랫폼은 meta-adaptation logic 이해를 돕고 시스템 조립 시간을 줄이며, 추가 오버헤드는 거의 없이 일관된 배포를 가능하게 한다고 정리한다.

이 절의 전체 메시지는 분명하다. MoE의 다음 단계는 더 복잡한 라우팅 규칙을 하나 더 붙이는 것이 아니라, 적응과 전이를 시스템 차원에서 반복 가능하게 만드는 데 있다. 즉, meta-learning과 knowledge transfer는 MoE를 “큰 모델”에서 “재사용 가능한 모듈 시스템”으로 바꾸는 축이다.
## IV. MIXTURE OF EXPERTS APPLICATIONS AND DOMAIN-SPECIFIC MODELS

이제 논문은 구조와 학습 메커니즘을 넘어, MoE가 실제 영역에서 어떻게 쓰이는지로 이동한다. 추천 시스템, 검색, 컴퓨터 비전, NLP, 헬스케어 등 서로 다른 문제 영역이 등장하지만, 공통적인 메시지는 같다. 현실의 데이터와 목표가 이질적일수록, 모두를 하나의 dense 표현으로 억지로 통합하는 것보다 적절한 expert decomposition과 routing을 설계하는 편이 유리하다는 것이다.

### IV-A. Recommendation Systems and Search

추천과 검색에서는 multi-domain, multi-task personalization이 핵심 문제다. 사용자, 아이템, 시나리오가 달라질수록 domain-specific signal과 shared knowledge를 동시에 잘 다루어야 하지만, 전통적 dense model은 이 균형을 맞추기 어렵다.

이를 해결하는 대표 예로 논문은 M3oE를 든다. M3oE는 shared preference, domain-specific behavior, task-specific pattern을 담당하는 세 개의 평행 expert 모듈을 두고, 이를 hierarchical gating으로 결합한다. 더 나아가 AutoML 기반 structure search를 써서 expert composition 자체를 동적으로 조정하며, 개인화 목표가 바뀌어도 구조가 함께 적응하게 만든다. 핵심은 사용자·도메인·태스크 이질성을 하나의 tower에 밀어 넣지 않고, 그 차이를 expert 분해의 출발점으로 삼는 것이다.

AESM2는 shared embedding 위에 multi-scenario layer를 쌓고, 그 위에 multi-task tower를 올리는 구조다. feature, scenario indicator, task indicator를 함께 encoding한 뒤, scenario와 task 수준에서 expert를 자동 선택한다. 이렇게 하면 어떤 expert는 여러 시나리오에 공유되고, 어떤 expert는 특정 scenario에 특화되도록 계산이 할당된다. 논문은 이 구조가 dynamic traffic distribution에서도 retrieval quality와 training stability를 개선한다고 설명한다.

[Figure 6 삽입]

*Figure 6. multi-task learning을 위한 AESM2 아키텍처. raw categorical/numerical feature를 연속 임베딩으로 바꾸는 Shared Embedding Layer, expert selection을 수행하는 Multi-Scenario Layer, 그리고 multi-task prediction을 담당하는 Multi-Task Layer로 구성된다.*

Figure 6가 강조하는 것은 단순히 층이 많다는 사실이 아니다. 추천 시스템에서 실제로 중요한 것은 모든 트래픽 구간을 동일한 방식으로 처리하지 않고, scenario-aware routing을 통해 표현 공유와 특화를 동시에 조절하는 것이다. AESM2는 바로 그 지점을 구조 수준에서 구현한다.

### IV-B. Multimodal and Multitask Learning

멀티모달·멀티태스크 학습에서는 서로 다른 입력 모달리티와 목적 함수를 동시에 다뤄야 하므로, shared representation만으로는 간섭이 쉽게 생긴다. 논문은 MoVA, DeepSeek-VL2, Omni-SMoLA, T-REX2, MoME, MoTE를 최근 흐름으로 묶으며, 단순 fine-tuning만으로는 heterogeneous task 전반의 강한 일반화를 얻기 어렵다고 본다.

**Omni-SMoLA.** Omni-SMoLA는 large multimodal model에서 shared representation이 만드는 task interference를 줄이려 한다. modality와 task별로 특화된 low-rank expert module을 추가하고 sparse routing으로 이들을 조합함으로써, generalist capability를 유지하면서도 모듈화된 specialization을 형성한다. 논문은 표준 LMM fine-tuning보다 더 안정적인 수렴과 더 좋은 task 전반 성능을 보고한다. 이 지점에서 MoE는 멀티모달 모델의 부가 기능이 아니라, 서로 다른 모달리티가 서로를 방해하지 않게 만드는 핵심 구조가 된다.

**Multimodal specialization via hybrid prompts.** T-REX2는 open-set object detection에서 textual prompt와 visual prompt의 상보성을 이용한다. 하나의 encoder는 “dog”, “bird” 같은 추상 텍스트 범주를 처리하고, 다른 encoder는 exemplar image에서 instance-level feature를 뽑는다. 두 stream은 deformable cross-attention을 통해 결합되고, contrastive alignment loss가 aggregated visual embedding과 text encoder의 [CLS] representation을 의미적으로 정렬한다. 이 구조는 text가 개념적 grounding을, visual prompt가 세밀한 instance cue를 제공해 unseen class에 대한 zero-shot detection을 강화한다.

[Figure 7 삽입]

*Figure 7. T-Rex2 프레임워크의 구조 개요. DETR 기반 end-to-end object detection 설계를 따르며, visual/textual prompt는 deformable cross-attention과 CLIP text encoding을 거쳐 contrastive learning으로 정렬된다.*

Figure 7는 멀티모달 MoE가 왜 단순한 late fusion이 아닌지를 잘 보여 준다. 텍스트 프롬프트는 범주 수준의 의미를, 비주얼 프롬프트는 instance 수준의 구체성을 제공한다. T-REX2는 이 둘을 동일한 detection pipeline 안에 넣어, 모달리티별 전문성이 실제 예측 단계까지 이어지도록 만든다.

**Expert reuse via hypernet-based modulation.** HyperMoE는 sparse expert model의 핵심 한계인 underutilization을 줄이려 한다. 표준 MoE에서는 top-\(k\) expert만 활성화되어 나머지 expert는 forward 단계에서 완전히 빠지는데, HyperMoE는 inactive expert의 hidden state를 활용하는 hypernetwork를 도입해 lightweight modulation signal을 생성하고 이를 active expert의 출력 경로에 주입한다. 덕분에 full expert evaluation 없이도 broader model knowledge를 active path에 끌어올 수 있으며, low-resource나 multitask 환경에서 downstream 성능과 expert diversity를 높일 수 있다.

논문은 이어 I2MoE, Uni3D-MoE, SMAR, MoDES처럼 interaction-aware routing, 3D multimodal scene understanding, soft modality-aware routing, training-free dynamic expert skipping을 다루는 후속 방향도 함께 언급한다. 공통된 방향은 표현력을 늘리되, 모달리티 간 상호작용과 추론 비용을 함께 통제하려는 것이다. 즉, multimodal MoE 연구의 핵심은 더 많은 expert를 쓰는 것이 아니라, 어느 expert가 어떤 상호작용을 담당해야 하는지를 더 정교하게 배우는 쪽으로 이동하고 있다.

### IV-C. Healthcare and Life Sciences

헬스케어에서 MoE는 환자 진료, 임상 의사결정, 시스템 효율이라는 세 가지 문제를 동시에 다루기 위한 구조로 제시된다. 논문은 Med-MoE, BiMediX, LoRA 기반 medical MoE를 대표 예로 들며, 이 영역에서는 정확도뿐 아니라 modularity와 interpretability가 특히 중요하다고 본다. 의료 시스템은 안전 제약이 강하기 때문에, 단순히 큰 모델보다 제어 가능한 전문가 구조가 더 적합할 수 있다.

**Embodied intelligence in healthcare.** Figure 8이 보여주듯이, embodied AI는 bedside assistance부터 rehabilitation, surgical support까지 다양한 의료 현장에 스며들 수 있다. 다만 clinical workflow와의 통합 부족, simulation-to-reality gap, 표준화된 평가 benchmark 부재가 광범위한 배포를 막는 장애로 남아 있다.

[Figure 8 삽입]

*Figure 8. healthcare에서 embodied AI가 적용되는 예시. pre-intervention(virtual triage nurse, medical consultant, remote ultrasound, endoscopic navigator), in-intervention(patient digital twin, mental healer, surgical operator, surgical planner), post-intervention(recovery coach, intelligent exoskeleton, medication controller, health wearable)으로 정리된다.*

Figure 8의 의미는 의료용 MoE가 반드시 텍스트와 영상 모델에만 머무르지 않는다는 데 있다. perception, planning, memory, action이 한 시스템 안에서 결합될수록, 하나의 monolithic policy보다 역할이 분리된 expert 집합이 더 자연스럽다. 논문은 이 점을 embodied healthcare라는 맥락에서 보여 준다.

**Medical data scarcity and specialized routing.** Syn-Mediverse는 48,000개 이상의 hyper-realistic image와 150만 개 annotation을 포함한 synthetic dataset으로 의료 시각 과제의 데이터 부족을 보완한다. AT-MoE는 LoRA-tuned expert layer와 grouped adaptive routing을 이용해 interpretability와 controllable decision-making을 강화한다. 최근에는 MoE-Health가 EHR, clinical note, medical imaging처럼 불완전하고 이질적인 입력 모달리티를 동적으로 처리하면서 in-hospital mortality, length-of-stay, readmission prediction을 개선한다.

MedMoE는 vision-language MoE 안에서 modality-specialized expert branch를 두어 의료 영상과 텍스트 소견의 정렬을 강화하고, REN은 anatomical prior를 이용해 region-specific expert를 학습해 interstitial lung disease 진단의 성능과 해석 가능성을 높인다. 또 adaptive expert grouping류 접근은 expert 사이의 협업 성향을 활용해 routing overhead를 낮추면서도 의료 맥락에서의 generalization을 유지하려 한다. 이들 흐름은 결국 의료용 MoE의 핵심이 더 많은 용량이 아니라, 도메인 제약을 반영한 전문화임을 보여준다.

이 절 전체를 관통하는 포인트는 분명하다. 헬스케어에서는 누가 더 큰 모델을 가졌는가보다, 어느 expert가 어떤 의료 단서를 맡고 그 결정 경로를 얼마나 설명할 수 있는가가 더 중요하다. 논문은 MoE가 바로 այդ 요구를 구조적으로 수용할 수 있다고 본다.
### IV-D. Computer Vision and Image Processing

현대 computer vision은 CNN에서 Transformer와 diffusion 계열로 이동하고 있으며, MoE는 object detection, image classification, scene understanding의 복잡성을 다루기 위한 희소 모듈 구조로 편입되고 있다. 논문은 AdaMV-MoE, GNT-MOVE, expert-based image decomposition 연구들을 넓은 배경으로 제시한다.

**Mixture of Calibrated Experts (MoCaE).** object detection에서는 여러 detector의 prediction을 합칠 때 confidence miscalibration이 큰 문제다. MoCaE는 각 expert output을 경험적 신뢰도에 맞게 calibration한 뒤 fusion함으로써, 과도하게 자신감 있는 expert가 전체 합의를 덮어쓰는 문제를 줄인다. COCO benchmark에서 최대 +2.5 AP 향상을 보여 주며, calibrated aggregation이 visual MoE의 성능을 실질적으로 바꿀 수 있음을 보여준다.

**Specialization and hierarchical visual experts.** refined gating architecture와 entropy-minimizing regularizer는 서로 겹침이 적고 의미적으로 정렬된 expert selection을 유도해 homogeneous expert behavior를 줄인다. MNIST, CIFAR, FashionMNIST 실험에서는 정확도와 해석 가능성이 함께 좋아진다. 더 깊은 계열인 Deep Mixture of Experts는 stacked routing layer를 사용해 앞단에서는 where expert가 공간적 위치를, 뒷단에서는 what expert가 의미적 정체성을 다루도록 분리한다. 이런 흐름은 visual MoE가 멀티스케일 feature hierarchy와 adaptive capacity allocation을 어떻게 결합하는지를 보여준다.

논문이 computer vision 예시를 넣는 이유는 단순히 응용 폭을 넓히기 위해서가 아니다. 비전에서의 expert specialization은 실제로 어떤 feature가 어디서 분기되고 다시 합쳐지는지를 더 선명하게 보여 준다. 즉, 언어모델에서 추상적으로 보이던 routing 문제가 비전에서는 공간적·의미적 분해 문제로 더 직접적으로 드러난다.

### IV-E. Natural Language Processing and Large Language Models

모든 응용 영역 가운데 NLP와 LLM은 MoE 채택이 가장 넓고 강하게 진행된 분야다. 기본 동기는 분명하다. 추론과 학습 비용을 비례적으로 늘리지 않으면서 capacity를 확장하는 것이다. 이 목표는 오픈소스 커뮤니티와 산업계의 효율 패러다임을 모두 바꾸어 놓았다.

**Parameter-efficient MoE for NLP.** 초기 MoE는 full set of experts를 저장하고 업데이트해야 하므로 parameter-efficient fine-tuning과 상충했다. 이를 줄이기 위해 논문은 11B급 모델에서 1% 미만 파라미터만 업데이트하면서도 full fine-tuning과 맞먹는 성능을 내는 extremely parameter-efficient MoE framework를 언급한다. 이는 constrained environment에서도 task-specific adaptability를 유지하게 만든다. LLM 시대의 MoE가 진짜로 널리 쓰이기 위해서는 pretraining 효율뿐 아니라 adaptation 효율까지 확보해야 한다는 점이 여기서 드러난다.

**Flexible composition for domain adaptation.** MoDE toolkit은 학습된 adapter나 full model을 domain-specific expert pool로 조합할 수 있게 하며, scratch에서 다시 학습하지 않고도 modular domain composition을 가능하게 한다. 계산 제약이 있는 multi-domain 시나리오에서 특히 실용적이다. 논문은 이 방향을, 거대한 범용 모델을 매번 다시 훈련하는 대신 이미 확보한 domain skill을 expert 단위로 재조합하는 흐름으로 읽는다.

**Hypothesis construction and representational properties.** 최근 MoE hypothesis construction 연구는 MoE가 uncertainty distribution 전반을 평균하는 Bayesian ensemble과 달리, discrete routing으로 hypothesis를 선택한다는 점을 이론적으로 밀고 나간다. 논문은 이를 abductive reasoning over hypothesis space로 설명하며, 약한 inductive prior 하에서도 MoE가 더 높은 functional capacity를 가질 수 있다고 요약한다. 즉, MoE는 불확실성을 평균내는 구조라기보다, 상황에 맞는 가설을 골라 쓰는 구조라는 해석이 가능해진다.

**Recent NLP-specific extensions.** MoxE는 entropy-aware routing을 쓰는 extended LSTM 기반 MoE로, rare token handling과 scalability를 개선한다. L-MoE는 MoE와 LoRA를 end-to-end trainable framework로 묶어, task-specialized low-rank expert adapter를 differentiable routing으로 조합한다. 이 흐름은 NLP용 MoE가 단순히 거대한 sparse FFN을 넘어서, 적응성과 skill composition을 직접 설계하기 시작했음을 보여준다.

이 절의 핵심은 언어모델에서 MoE가 더 이상 선택적 실험 구조가 아니라는 데 있다. 이제 MoE는 LLM의 비용-성능 곡선을 다시 그리는 핵심 도구가 되었고, fine-tuning·domain adaptation·가설 선택까지 언어모델 설계의 여러 층위에 직접 개입한다.

### IV-F. Methodological Innovation and Theoretical Foundations

MoE의 성공은 단지 큰 실험 결과만으로 설명되지 않는다. 이를 떠받치는 이론과 방법론의 축도 함께 커지고 있다. 논문은 scalability, convergence, expert diversity, model integration 문제를 이 절에 묶어 다룬다.

**Scaling behavior under memory constraints.** unified scaling law 연구는 active parameter count, dataset size, expert configuration을 함께 고려했을 때, MoE가 dense model보다 memory efficiency에서 더 나을 수 있음을 보인다. 최대 5B parameter regime까지의 실험은, 이전의 통념과 달리 sparse 구조가 메모리 제약 환경에서도 경쟁력이 있다는 점을 정리한다. 즉, MoE의 강점은 계산량만이 아니라 메모리-성능 trade-off에서도 나타날 수 있다.

**Convergence in Gaussian-gated models.** Gaussian-gated MoE의 maximum likelihood estimation 분석은 gating covariate 아래의 비균일 수렴을 Voronoi 기반 loss로 기술한다. 위치 파라미터 설정에 따라 solution space가 달라지고, polynomial system 형태의 구조가 나타난다는 점은 sparse optimization의 동역학을 이해하는 실마리로 제시된다. 이는 MoE가 경험적으로 잘 돌아간다는 사실을 넘어, 왜 어떤 설정에서는 잘 되고 어떤 설정에서는 불안정한지를 더 정밀하게 설명하려는 움직임이다.

**Structured optimization for routing.** Omi 등의 similarity-preserving load-balancing objective는 비슷한 입력이 일관된 expert assignment를 받도록 유도하여 expert collapse와 variance를 줄이려 한다. MaxScore routing은 expert selection을 minimum-cost maximum-flow 문제로 바꾸어, expert capacity·token assignment·communication cost를 하나의 명시적 최적화 프레임으로 다룬다. 이는 heuristic regularization에서 principled optimization으로 이동하는 흐름이다.

**Expert diversity, reuse, and model merging.** OMoE는 Gram-Schmidt orthogonalization으로 expert redundancy를 줄이고, MoDE 계열의 mutual distillation은 feature coverage를 넓힌다. Nexus는 dense model을 retraining 없이 sparse expert system으로 바꾸는 adaptive routing과 parameter reuse를 강조한다. HMoE는 hypernetwork로 expert parameter를 동적으로 생성해 domain alignment를 돕고, parameter merging framework는 heterogeneous expert integration 시의 충돌을 alignment와 reparameterization으로 줄인다. 즉, 이 절의 메시지는 MoE가 더 이상 단일 아키텍처가 아니라, expert 생성·선택·정렬·병합을 포괄하는 방법론 집합이라는 데 있다.

## V. Evaluations, Challenges and Future Directions

MoE가 실제 배포 영역으로 들어갈수록, 어떻게 더 크게 만들까 못지않게 무엇을 제대로 평가해야 하는가가 중요해진다. 논문은 기존 LLM benchmark만으로는 MoE의 본질을 포착하기 어렵다고 본다. MoE는 conditional computation 구조이기 때문에, 최종 정확도만 보는 평가는 중간의 expert assignment, load balancing, specialization quality를 놓치기 쉽다.

### V-A. Evaluation Framework and Methodology

**Theoretical foundation and evaluation principles.** MoE 평가는 divide-and-conquer 원리의 성공 여부를 봐야 한다. 문제 공간 분할이 잘 되었는지, expert assignment가 유효한지, load balancing과 knowledge distribution이 end-to-end로 작동하는지를 함께 측정해야 한다. 즉, evaluation은 output-level metric만으로 끝나지 않고 intermediate process까지 포함해야 한다.

**Standardized evaluation framework.** 논문은 Mixtral 8x7B benchmark와 LibMoE를 중요한 플랫폼으로 거론한다. 특히 LibMoE는 연구·학습·평가 전 과정을 모듈화한 full-lifecycle framework로 소개된다. 5개의 state-of-the-art MoE 알고리즘을 3개의 LLM과 11개 dataset에 zero-shot으로 체계 평가한 결과, 서로 다른 목적에서 개발된 알고리즘들이 surprisingly similar average performance를 보였다는 점이 강조된다. 이는 특정 MoE algorithm choice의 중요성이 생각보다 덜할 수 있음을 시사한다.

**System-level multi-dimensional evaluation.** MoE-CAP은 model accuracy, application performance, deployment cost의 세 축을 함께 보는 CAP triangle을 제안한다. 소프트웨어 스택(attention, router, expert network)과 하드웨어 스택(CPU, GPU, DRAM, HBM)을 함께 고려하고, routing sparsity와 hardware utilization의 상호작용까지 반영해 CAP radar plot을 만든다. 이 프레임은 정확도는 좋은데 배포가 안 되는 MoE를 걸러내기 위한 실전형 평가 틀이다.

[Figure 9 삽입]

*Figure 9. MoE-CAP 방법론 프레임워크. 왼쪽은 deployment cost, model accuracy, application performance 사이의 삼각 관계를 보여주고, 오른쪽은 sparsity-aware metric과 CAP radar visualization을 이용해 MoE 아키텍처를 종합 평가하는 MoE-CAP 프레임워크를 나타낸다.*

Figure 9가 말하는 바는 명확하다. MoE의 좋은 설계는 정확도 하나로 정의되지 않는다. 어떤 MoE는 정확도는 높아도 배포 비용이 과도하고, 또 어떤 MoE는 비용은 낮지만 실제 응용 성능이 떨어질 수 있다. CAP triangle은 이런 세 축의 균형을 정면으로 평가 대상으로 삼는다.

**Specific evaluation method examples.** MoCaE는 prediction miscalibration 문제를 직접 겨냥한 좋은 예다. raw prediction을 평균내기 전에 각 expert output을 empirical reliability에 맞게 calibration하여 fusion하면, overconfident expert가 전체 결정을 왜곡하는 현상을 줄일 수 있다. COCO에서 최대 +2.5 AP를 얻는다는 점은 calibration-aware evaluation이 단순 분석 도구를 넘어 실제 성능 향상으로 이어질 수 있음을 보여준다.

**Expert diversity and representation learning challenges.** MoE의 근본적 한계 중 하나는 expert가 실제로는 매우 비슷한 representation으로 수렴한다는 점이다. 논문은 diverse input에서 expert similarity가 99%를 넘는 사례를 언급하며, 이는 underperforming model에서만 나타나는 특수 현상이 아니라 좋은 모델에서도 나타나는 구조적 문제라고 본다. 이런 representational homogeneity는 divide-and-conquer 원리를 직접 훼손한다. expert가 많다는 사실만으로 MoE가 자동으로 강해지지 않는 이유가 여기에 있다.

**Architectural design and integration challenges.** shared layer를 MoE 안에 섣불리 통합하면, 동일한 shared feature에 노출된 expert가 서로 redundant하거나 conflict하는 representation을 배우면서 specialization이 약해질 수 있다. 또 incremental learning에서 post hoc으로 expert를 추가하는 경우, parallel expert 간 inconsistent output이 생겨 training instability나 suboptimal prediction으로 이어질 수 있다. 따라서 conflict-aware routing이나 mediation strategy가 필요하다.

**Routing mechanisms and specialization challenges.** learned routing의 필요성과 효용 자체도 여전히 논쟁적이다. frozen random router가 여러 benchmark에서 learned routing과 비슷한 성능을 보인다는 결과는, adaptive routing이 언제나 핵심이라는 통념에 의문을 던진다. 결국 routing expressiveness와 architectural simplicity 사이의 trade-off를 더 정밀하게 이해해야 한다.

**Challenges in Theoretical Grounding of MoE Architectures.** MoE는 경험적으로는 강력하지만, expert diversity·specialization dynamics·system-level generalization을 하나의 이론으로 묶어 설명하는 틀이 아직 약하다. 논문은 expert count, sparsity, gating choice를 task와 data에 맞게 원리적으로 정해 줄 정량 프레임워크가 필요하다고 본다. 정보이론이나 학습이론에 기반한 routing·selection 분석이 이루어져야, 고비용 empirical tuning을 줄일 수 있다.

**Technical method innovation.** DeepSeekMoE, TableMoE, Pre-gated MoE는 input-aware expert allocation과 gating preconditioning으로 specialization을 높이려는 방향을 보여준다. 여기에 OMoE류의 orthogonality regularization과 RLHF 같은 feedback-based optimization을 결합하면, 구조적 규제와 적응적 학습을 함께 쓰는 hybrid strategy가 가능해진다. 논문은 이런 혼합 전략이 더 robust하고 generalizable한 MoE로 가는 유망한 길이라고 본다.

## VI. Conclusion

이 서베이는 Mixture of Experts 아키텍처의 최근 발전을 이론적 기원에서 대규모 구현까지 추적하고, 핵심 구조와 설계 원리를 분석한 뒤, 고급 변형, 메타러닝, 지식 전이, 도메인별 응용까지 차례대로 검토한다. 이어 평가 방법론, routing stability와 expert specialization 같은 핵심 난제, 그리고 미래 연구 방향까지 하나의 흐름으로 연결한다.

결론적으로 MoE는 단순한 parameter-saving trick이 아니라, scalable하고 efficient한 foundation model을 설계하는 핵심 패러다임으로 자리 잡고 있다. 하지만 expert diversity, routing stability, deployability, principled evaluation, theoretical grounding이 함께 해결되지 않으면 그 잠재력을 완전히 실현하기 어렵다. 이 논문은 바로 그 지점까지 포함해 현재의 지형을 정리하는 작업으로 읽힌다.

## 추가 설명

### 1. 이 논문이 MoE를 구조적 전환으로 보는 이유

이 논문은 MoE를 단순한 비용 절감 기법으로 소개하지 않는다. 오히려 모든 입력에 동일한 계산을 수행하는 dense 모델의 가정을 깨고, 입력마다 다른 계산 경로를 활성화하는 조건부 계산 체계로 본다. 그래서 서론에서 바로 sparse activation의 효율보다 먼저 modularity와 specialization을 이야기한다. MoE의 핵심은 expert가 많다는 사실이 아니라, 어떤 입력이 어떤 expert를 거치는지가 모델의 표현 자체를 바꾼다는 데 있다.

이 관점으로 보면 MoE의 장점은 두 층위로 나뉜다. 첫째는 계산 효율이다. 전체 파라미터 수를 늘려도 실제 활성 파라미터 수는 제한할 수 있다. 둘째는 표현 구조다. 서로 다른 expert가 입력 공간의 다른 부분을 맡으면서, 하나의 dense 네트워크보다 더 다양한 기능 분해가 가능해진다. 논문이 MoE를 foundation model 시대의 핵심 아키텍처로 보는 이유도 여기에 있다.

### 2. 식 (1)부터 식 (5)까지는 하나의 파이프라인으로 읽어야 한다

식 (1)은 MoE의 가장 기본적인 출력 정의다. 전체 expert의 출력을 합치는 것처럼 보이지만, 실제로는 gate가 거의 모든 expert의 값을 0으로 만들어 소수 expert만 활성화한다. 식 (2)는 이 선택이 어떻게 결정되는지를 보여 주는 score 식이고, 식 (3)과 식 (4)는 그 score를 noisy top-\(k\) routing으로 구현하는 구체적 방식이다. 즉, 식 (1)은 결과이고, 식 (2)–(4)는 그 결과를 만드는 선택 규칙이다.

여기에 식 (5)가 더해지면 왜 MoE 학습이 어려운지가 드러난다. 모델은 한편으로는 특정 expert가 특정 입력 패턴에 강하게 특화되길 바라지만, 다른 한편으로는 일부 expert만 과도하게 바빠지는 것도 원하지 않는다. 그래서 라우팅은 정확할수록 좋지만, 너무 정확하면 collapse가 일어나고, 너무 균등하면 specialization이 사라진다. 식 (5)는 바로 이 모순을 드러내는 식이다.

### 3. 식 (7)부터 식 (17)까지는 모두 specialization의 질을 다르게 조절하는 방법이다

식 (7)은 expert를 서로 멀어지게 만들어 중복을 줄이고, 식 (8)은 expert 사이에 약한 지식 공유를 만들어 지나친 단절을 막는다. 식 (9)는 적응 비용을 낮추고, 식 (10)은 coarse-to-fine 조직 구조를 도입하며, 식 (11)은 계산 비용까지 고려한 expert 선택을 가능하게 만든다. 식 (12)는 선택되지 않은 expert도 간접적으로 활용하는 방식이고, 식 (13)과 식 (14)는 라우팅의 제어권이 token에 있는지 expert에 있는지를 구분한다. 식 (16)은 그 결과가 실제 언어 구조와 얼마나 연결되는지를 측정하고, 식 (17)은 입력 난이도에 따라 expert 수 자체를 바꾼다.

즉, 논문 속 수식들은 제각기 다른 아이디어처럼 보이지만 사실은 하나의 질문을 서로 다른 각도에서 다루고 있다. 어떻게 하면 expert가 서로 다른 일을 하도록 만들면서도, 전체 시스템은 효율적이고 안정적으로 유지할 수 있는가. 이 질문이 논문의 중심축이다.

### 4. 응용 사례가 다양한 이유는 입력 이질성이 큰 곳일수록 MoE가 잘 맞기 때문이다

추천 시스템에서는 사용자와 시나리오가 다르고, 멀티모달 학습에서는 텍스트와 이미지가 다르며, 의료에서는 데이터 모달리티와 안전 제약이 다르다. dense 모델은 이런 차이를 하나의 표현 공간에 전부 우겨 넣으려 하지만, MoE는 애초에 차이가 존재한다는 사실을 계산 구조에 반영한다. 그래서 추천에서는 scenario-aware routing이, 멀티모달에서는 modality-aware routing이, 의료에서는 domain-constrained routing이 중요하게 등장한다.

논문이 LLM만 길게 다루지 않고 recommendation, healthcare, vision을 모두 포함하는 이유도 여기에 있다. MoE의 본질은 특정 도메인에 한정된 최적화 기술이 아니라, 이질적인 신호를 분해하고 다시 조합하는 범용적 계산 원리라는 점을 보여 주기 위해서다.

### 5. 이 논문이 마지막에 평가를 길게 다루는 이유

MoE는 정확도만으로는 설명되지 않는 모델이다. 같은 정확도를 내더라도 어떤 모델은 특정 expert만 과도하게 쓰고, 어떤 모델은 고르게 분산하며, 어떤 모델은 하드웨어 비효율 때문에 실제 배포가 어렵다. 그래서 논문은 마지막 절에서 evaluation을 단순 부록이 아니라 핵심 주제로 끌어올린다. CAP triangle이 중요한 이유도, 정확도·응용 성능·배포 비용이 서로 다른 축이라는 사실을 전제로 하기 때문이다.

이 부분은 논문의 결론과도 연결된다. MoE가 진짜로 dense 모델의 대안이 되려면 더 큰 모델을 만드는 것만으로는 부족하다. expert diversity, routing stability, calibration, hardware utilization까지 함께 좋아져야 한다. 논문이 제시하는 future direction은 결국 더 강한 MoE보다 더 믿을 수 있고 더 설명 가능하며 더 배포 가능한 MoE를 향해 있다.
