# HybridFlow: A Flexible and Efficient RLHF Framework

## Abstract

이 논문은 RLHF(Reinforcement Learning from Human Feedback)를 하나의 분산 데이터플로우 문제로 다시 바라본다. 기존 RL에서는 각 노드가 신경망 연산을, 각 엣지가 데이터 의존성을 뜻하는 DAG로 작업을 설명할 수 있지만, RLHF에서는 상황이 훨씬 복잡해진다. 각 노드는 단순한 신경망이 아니라 대규모 언어 모델의 분산 학습·추론·생성 프로그램이 되고, 노드 사이의 엣지는 단순한 텐서 전달이 아니라 서로 다른 병렬화 전략을 쓰는 다수의 GPU 집합 사이에서 발생하는 many-to-many 멀티캐스트가 된다. 따라서 RLHF를 효율적으로 돌리려면, 모델 내부의 분산 계산과 모델 사이의 데이터 재분할을 동시에 잘 다뤄야 한다.

문제는 기존 접근이 양 극단에 치우쳐 있다는 점이다. 전통적인 RL 프레임워크의 단일 컨트롤러(single-controller) 방식은 전체 데이터플로우를 유연하게 제어하기에는 좋지만, 거대한 LLM 내부 연산까지 중앙에서 지시하려 하면 dispatch overhead가 너무 커진다. 반대로 기존 RLHF 시스템의 다중 컨트롤러(multi-controller) 방식은 각 디바이스가 자기 계산을 독립적으로 진행하므로 모델 내부 계산은 빠르지만, 모델 간 통신과 실행 순서가 코드 안에 깊게 뒤엉켜 있어 새로운 알고리즘이나 배치를 적용하기가 어렵다.

저자들은 이 딜레마를 해결하기 위해 **HybridFlow**를 제안한다. 핵심 아이디어는 **모델 내부(intra-node) 계산에는 multi-controller를 쓰고, 모델 사이(inter-node) 데이터 흐름 제어에는 single-controller를 쓰는 하이브리드 구조**다. 이를 위해 계층형 API를 설계해 각 모델의 학습·추론·생성 연산을 primitive로 캡슐화하고, 모델 간 데이터 재분할은 transfer protocol로 숨긴다. 그 위에 actor 모델의 학습-생성 전환을 최적화하는 **3D-HybridEngine**과, 데이터플로우에 맞는 GPU 배치와 병렬화 전략을 자동으로 찾는 **Auto Device Mapping**을 더한다.

실험에서는 PPO, ReMax, Safe-RLHF 같은 서로 다른 RLHF 알고리즘과 7B-70B 규모의 모델, 8-128개 GPU 환경에서 HybridFlow를 평가한다. 결과는 단순한 구조 개선 수준을 넘는다. 저자들은 기존 최강 베이스라인 대비 **1.53배에서 20.57배까지의 처리량 향상**을 보고하며, 이 성능 향상이 단순한 구현 차이가 아니라 데이터플로우 표현, actor 전환 비용, 병렬화 전략, 모델 배치 최적화가 함께 맞물려 나온 결과라고 주장한다.

## 1. Introduction

대규모 언어 모델은 사전학습과 SFT만으로도 강력하지만, 실제 서비스 수준의 정렬에는 여전히 RLHF가 핵심 기법으로 쓰인다. RLHF에서는 보통 actor, critic, reference policy, reward model 같은 여러 모델이 함께 등장하고, 한 iteration 안에서도 생성, 점수 계산, 가치 평가, 정책 갱신이 서로 다른 방식으로 얽힌다. 이 논문은 여기서 출발해, RLHF를 “여러 개의 분산 LLM 프로그램이 서로 데이터를 주고받는 복합 데이터플로우”로 본다.

PPO 기반 RLHF를 예로 들면, actor는 프롬프트에 대한 응답을 생성하고, critic은 value를 계산하며, reference policy는 KL 기준이 되는 log probability를 제공하고, reward model은 선호 점수를 계산한다. 이어서 actor와 critic이 다시 갱신된다. Safe-RLHF는 여기에 cost model과 보조 pretrain loss를 더하고, ReMax는 critic을 제거하는 대신 추가 생성 패스를 넣는다. 즉 RLHF 알고리즘이 달라질수록 노드 수, 노드 간 의존성, 데이터 전달 방식이 달라진다. 따라서 어떤 RLHF 프레임워크가 진짜 실용적이려면, 특정 알고리즘 하나만 빠르게 돌리는 것이 아니라 **다양한 데이터플로우를 유연하게 표현하고 실행할 수 있어야 한다.**

![Figure 1](/assets/images/A%20Flexible%20and%20Efficient%20RLHF%20Framework/hybridflow_flexible_efficient_rlhf_framework_figure_01.png)

기존 RL 프레임워크는 전체 그래프를 중앙 컨트롤러가 관리하는 single-controller 방식을 사용한다. 이런 방식은 “어떤 노드를 어디에 배치할지, 어떤 순서로 실행할지”를 전역 시야로 조정할 수 있다는 장점이 있다. 하지만 RLHF에서는 각 노드 자체가 이미 거대한 분산 프로그램이다. 예컨대 actor 하나만 해도 tensor parallel, pipeline parallel, data parallel이 결합된 LLM 학습·생성 엔진으로 동작한다. 이 모든 내부 연산을 중앙 컨트롤러가 일일이 dispatch하면 제어 오버헤드가 커질 수밖에 없다.

그래서 기존 RLHF 시스템들은 보통 multi-controller 구조를 선택한다. 각 GPU 프로세스가 로컬 컨트롤러를 가지고 계산을 진행하면 LLM 내부 연산은 효율적으로 수행된다. 문제는 그 다음이다. actor가 만든 결과를 critic과 reward model로 보내고, 각 모델이 서로 다른 병렬화 전략을 쓰는 상황에서, send/recv와 collective communication, 모델 연산 코드가 한 프로그램 안에 뒤섞이기 쉽다. 이런 구조에서는 데이터 의존성을 조금만 바꾸어도 여러 모델의 내부 구현을 동시에 수정해야 하고, 다른 학습·추론 엔진을 끼워 넣기도 어렵다.

논문은 이 문제를 해결하기 위해 RLHF 프레임워크를 두 층으로 나눈다. **모델 내부 계산은 각 모델 클래스가 캡슐화하고, 모델 간 데이터 흐름은 단일 컨트롤러가 제어한다.** 여기서 actor 학습과 생성처럼 RLHF에서 가장 비용이 큰 부분은 3D-HybridEngine으로 최적화하고, 전체 데이터플로우 차원의 자원 배치는 Auto Device Mapping으로 자동화한다. 저자들이 소개하는 핵심 기여는 네 가지로 정리된다. 첫째, RLHF 데이터플로우를 표현하고 실행하기 위한 계층형 하이브리드 프로그래밍 모델이다. 둘째, actor 학습-생성 전환 비용을 줄이는 3D-HybridEngine이다. 셋째, 모델 배치와 병렬화를 자동으로 정하는 매핑 알고리즘이다. 넷째, 다양한 알고리즘·모델 규모·클러스터 조건에서의 대규모 실험 검증이다.

## 2. Background and Motivation

### 2.1 Reinforcement Learning from Human Feedback

RLHF는 인간 선호를 반영해 LLM의 출력 분포를 바꾸는 절차다. 보통 actor는 응답을 생성하고, critic은 가치 추정을 제공하며, reference policy는 현재 정책이 원래 정책에서 얼마나 벗어났는지 측정하는 기준을 제공하고, reward model은 인간 선호를 근사한 점수를 준다. 이 여러 모델은 하나의 monolithic 시스템이 아니라, 서로 다른 계산 양상과 서로 다른 병렬화 요구를 가진 개별 프로그램들에 가깝다.

PPO를 기준으로 보면 RLHF는 세 단계로 분해된다. 첫째는 **Generation** 단계로, actor가 프롬프트에 대한 응답을 auto-regressive하게 생성한다. 둘째는 **Preparation** 단계로, critic이 value를 계산하고, reference policy가 reference log probability를 계산하고, reward model이 reward를 계산한다. 셋째는 **Learning/Training** 단계로, 이전 단계에서 모은 정보를 바탕으로 actor와 critic을 업데이트한다. 이 구조는 Safe-RLHF와 ReMax에서도 유지되지만, 세부 노드 구성과 계산 순서는 달라진다. Safe-RLHF는 cost model과 auxiliary pretrain loss를 포함하고, ReMax는 critic 없이 actor 중심 데이터플로우를 사용한다.

중요한 점은 RLHF 알고리즘이 활발히 변형되고 있다는 사실이다. 정렬 목적이 안전성인지, 도움 됨인지, 샘플 효율인지에 따라 필요한 모델과 데이터 의존성이 달라진다. 따라서 RLHF 시스템이 진화하는 알고리즘을 따라가려면, 데이터플로우를 딱 하나의 고정 실행 패턴으로 구현하는 방식에서 벗어나야 한다.

### 2.2 Programming Model for Distributed ML

분산 ML 시스템의 프로그래밍 모델은 크게 single-controller와 multi-controller로 나눌 수 있다. single-controller는 중앙 컨트롤러가 전체 실행 순서를 관리한다. 사용자는 하나의 전역 프로그램 관점에서 데이터플로우를 기술할 수 있고, 컨트롤러는 이를 각 워커에게 분배한다. 이 방식은 노드 배치, 실행 순서, 자원 가상화처럼 **그래프 수준의 조정**에는 유리하다. 하지만 대규모 LLM의 연산 그래프를 매번 중앙에서 지시하면 dispatch overhead가 커진다.

반대로 multi-controller에서는 각 디바이스가 자체 컨트롤러를 가진다. Megatron-LM, DeepSpeed, vLLM 같은 현대 LLM 학습·서빙 시스템이 이 방식을 선호하는 이유는 분명하다. 각 워커가 자기 로컬 상태를 바탕으로 빠르게 collective를 호출하고 연산을 수행할 수 있기 때문이다. 하지만 이 구조는 시스템 전체를 보는 중앙 시야가 없으므로, **모델 간 데이터 전달과 실행 순서를 표현하는 코드가 사용자 프로그램 안으로 스며든다.** RLHF에서는 특히 이 문제가 심하다. actor가 send를 하고 critic과 reward model이 정확한 시점에 receive를 해야 하며, 그 사이에 all_gather와 모델 연산이 섞인다.

![Figure 2](/assets/images/A%20Flexible%20and%20Efficient%20RLHF%20Framework/hybridflow_flexible_efficient_rlhf_framework_figure_02.png)

논문이 문제 삼는 것은 바로 이 결합도다. 기존 RLHF 프레임워크에서는 “어떤 모델이 어떤 데이터를 받아 무엇을 계산하는가”라는 논리와, “어떤 collective와 point-to-point를 언제 어떻게 호출하는가”라는 실행 세부가 한 덩어리로 묶여 있다. 그 결과 특정 알고리즘에 맞춘 코드는 빠르게 만들 수 있어도, 다른 RLHF 데이터플로우나 다른 병렬화 엔진으로 일반화하기 어렵다.

### 2.3 RLHF Characteristics

RLHF 데이터플로우가 어려운 이유는 모델 수가 많기 때문만이 아니다. **각 모델의 workload가 서로 이질적**이라는 점이 더 중요하다. reference policy와 reward model은 주로 forward pass만 수행하므로 파라미터만 GPU 메모리에 있으면 된다. 반면 actor와 critic은 학습 대상이므로 파라미터, gradient, optimizer state까지 유지해야 한다. 게다가 actor는 생성과 학습을 모두 수행하고, critic은 추론과 학습을 모두 수행한다. RLHF 시스템은 처음부터 이 이질성을 전제로 설계되어야 한다.

또 다른 특징은 **actor 학습과 actor 생성의 계산 성격이 다르다**는 점이다. actor 학습은 일반적으로 compute-bound라서 더 큰 model parallel 크기가 유리하다. 반면 actor 생성은 memory-bound에 가까워 작은 model parallel과 큰 data parallel의 조합이 더 나은 경우가 많다. 예를 들어 8개 GPU에서 7B 모델을 학습할 때는 8-way model parallel이 효율적일 수 있지만, 생성에서는 2-way model parallel에 4-way 복제를 두는 편이 더 높은 throughput을 낼 수 있다. 문제는 이 두 단계를 오갈 때 가중치 재분할 비용이 커진다는 점이다. 논문은 70B actor 기준으로 iteration마다 140GB의 모델 가중치를 넘기는 사례를 제시하며, 이 전환이 iteration 시간의 36.4%까지 차지할 수 있다고 말한다.

마지막으로 **모델 배치 전략도 상황에 따라 달라져야 한다.** 서로 의존성이 없는 모델은 다른 GPU 집합에 두어 병렬 실행할 수 있지만, 이렇게 하면 어떤 단계에서는 일부 GPU가 놀게 된다. 반대로 여러 모델을 같은 GPU 집합에 colocate하면 메모리를 공유하면서 순차 실행해야 하므로 충돌과 OOM을 피할 수 있지만, 병렬성이 줄어든다. 따라서 좋은 RLHF 프레임워크는 단순히 “빠른 엔진”이 아니라, 모델 간 배치와 실행 패턴을 유연하게 바꿀 수 있어야 한다.

![Figure 3](/assets/images/A%20Flexible%20and%20Efficient%20RLHF%20Framework/hybridflow_flexible_efficient_rlhf_framework_figure_03.png)

### 2.4 Limitations of existing RLHF systems

저자들은 기존 RLHF 시스템의 한계를 두 갈래로 정리한다. 첫째는 **유연성 부족**이다. 기존 시스템은 대부분 multi-controller 방식으로 구현되어 있어 collective communication, 모델 연산, point-to-point transfer 코드가 복잡하게 얽혀 있다. 그래서 PPO 같은 특정 알고리즘은 빠르게 지원할 수 있어도, 다른 RLHF 데이터플로우를 만들려면 사실상 시스템 전체를 뜯어고쳐야 한다. 논문은 DeepSpeed-Chat에 3D parallelism을 본격 적용하려면 전체 시스템을 거의 새로 구현해야 할 수 있다고 지적한다.

둘째는 **실행 효율의 제한**이다. DeepSpeed-Chat과 OpenRLHF는 actor 학습에 ZeRO-3를 쓰고, 생성에는 TP를 쓴다. 이때 DeepSpeed-Chat은 학습과 생성 사이에서 actor 가중치를 다시 재분할해야 하고, OpenRLHF는 아예 두 단계용 actor 복사본을 따로 유지한다. NeMo-Aligner는 학습과 생성에 같은 3D parallel 설정을 사용해 전환 비용은 적지만, 생성 throughput이 낮아진다. 또한 각 시스템은 사실상 하나의 고정된 모델 placement와 하나의 실행 패턴만 자연스럽게 지원한다.

![Table 1](/assets/images/A%20Flexible%20and%20Efficient%20RLHF%20Framework/hybridflow_flexible_efficient_rlhf_framework_table_01.png)

Table 1은 이 차이를 구조적으로 보여 준다. DeepSpeed-Chat은 모든 모델을 한 GPU 집합에 올려 순차 실행하는 경향이 강하고, OpenRLHF는 모델을 분리 배치하지만 두 actor 복사본을 유지한다. NeMo-Aligner는 actor와 reference, critic과 reward를 각각 묶는 방식이지만 학습·생성의 병렬화 구성을 동일하게 가져간다. HybridFlow는 여기서 두 가지를 동시에 노린다. **학습/생성 병렬화를 다르게 설정하면서도 actor 복사본 중복을 없애고, placement 자체도 workload와 자원 규모에 맞게 바꿀 수 있는 프레임워크**를 만드는 것이다.

### 2.5 Design Considerations

이 절의 결론은 명확하다. RLHF 데이터플로우에서 노드 수는 아주 많지 않다. 따라서 모델 간 데이터 전달과 실행 순서는 single-controller로 조정해도 dispatch overhead가 지배적이지 않다. 반면 각 모델 내부의 연산은 거대한 분산 LLM 프로그램이므로 multi-controller가 필수다. 이 두 요구를 결합하면, 가장 자연스러운 해법은 **inter-node에는 single-controller, intra-node에는 multi-controller를 쓰는 계층형 하이브리드 모델**이다.

즉 HybridFlow는 “전체 흐름은 하나의 중앙 프로그램으로 작성하되, 각 모델의 실제 분산 계산은 각자 최적의 엔진이 알아서 수행하게 하는” 설계를 선택한다. 이 설계 덕분에 시스템은 유연성과 효율을 동시에 확보할 수 있다.

## 3. HybridFlow Overview

3절은 HybridFlow의 전체 구조를 한 번에 조망하는 절이다. 논문은 HybridFlow를 세 개의 핵심 구성요소로 설명한다. 첫째는 **Hybrid Programming Model**로, RLHF 데이터플로우를 primitive API와 transfer protocol의 조합으로 표현한다. 둘째는 **3D-HybridEngine**으로, actor의 학습-생성 전환을 효율화한다. 셋째는 **Auto-Mapping Algorithm**으로, 주어진 GPU 수와 모델 조합에서 최적의 배치와 병렬화를 찾는다.

![Figure 4](/assets/images/A%20Flexible%20and%20Efficient%20RLHF%20Framework/hybridflow_flexible_efficient_rlhf_framework_figure_04.png)

사용자는 RLHF를 시작할 때 actor·critic·reference·reward 모델의 구조와 크기, 데이터플로우 안에서의 배치 계획, 그리고 각 단계에서 쓸 병렬화 전략을 입력한다. single controller는 이 정보를 바탕으로 모델을 초기화하고, ResourcePool 위에 모델을 매핑하고, 각 모델에 primitive operation을 호출한다. 한편 각 모델 내부의 distributed worker는 자신에게 할당된 병렬화 구성에 따라 parallel group을 만들고, 학습·추론·생성을 해당 엔진 위에서 수행한다.

핵심은 시스템이 사용자에게 “분산 collective를 언제 호출할지”까지 드러내지 않는다는 점이다. 사용자는 actor가 응답을 생성하고, critic이 value를 계산하고, reward model이 점수를 계산한다는 **논리적 흐름**만 작성하면 된다. 실제 데이터 재분할과 장치 간 통신은 transfer protocol과 3D-HybridEngine이 처리한다.

## 4. Hybrid Programming Model

### 4.1 Hierarchical APIs

HybridFlow의 프로그래밍 모델은 계층형이다. 가장 아래에는 물리 디바이스가 있고, 그 위에 ResourcePool이 있다. ResourcePool은 GPU 집합을 가상화한 단위다. 같은 ResourcePool을 공유하는 모델은 같은 GPU 집합에 colocate되고, 서로 다른 ResourcePool을 쓰는 모델은 서로 다른 장치 집합에 배치된다. 이 추상화 덕분에 placement를 바꾸더라도 모델 연산 코드 자체는 건드리지 않아도 된다.

그 위에는 worker class가 있다. 논문은 기본 클래스로 **3DParallelWorker**를 두고, 여기에 FSDPWorker와 ZeROWorker 같은 파생 클래스를 더한다. actor, critic, reference, reward 모델은 이런 worker를 기반으로 구현되며, 각 모델은 `generate_sequence`, `compute_values`, `compute_reward`, `update_actor` 같은 primitive API를 외부에 노출한다. 중요한 점은 이 API가 “분산 학습 프로그램 전체”를 하나의 연산처럼 캡슐화한다는 것이다. 즉 외부에서는 actor.generate_sequence를 호출했을 때 내부에서 Megatron-LM이든 vLLM이든 알아서 돌아가고, 호출자는 결과만 받는다.

![Figure 5](/assets/images/A%20Flexible%20and%20Efficient%20RLHF%20Framework/hybridflow_flexible_efficient_rlhf_framework_figure_05.png)

모델 간 데이터 전달은 **transfer protocol**이 담당한다. 각 model operation은 `@register`를 통해 어떤 protocol을 사용할지 연결된다. 하나의 protocol은 `collect`와 `distribute` 두 함수로 이뤄진다. 예를 들어 actor 학습이 3D parallelism을 쓴다면, actor의 출력은 각 DP group에서 collect되어 single controller 쪽으로 올라오고, 다시 critic 입력 형식에 맞게 distribute된다. 이 구조는 모델 연산과 데이터 재분할을 분리한다. 즉 actor 내부 병렬화를 바꿔도 critic 코드까지 수정할 필요가 없다.

또 하나 중요한 요소가 **asynchronous execution**이다. 모델이 서로 다른 GPU 집합에 배치되어 있으면, 어떤 모델의 출력 future가 준비되는 즉시 다음 모델을 실행할 수 있다. 반대로 같은 GPU 집합에 있는 colocated model은 호출 순서대로 순차 실행된다. 이 덕분에 HybridFlow는 동일한 RLHF 알고리즘 코드로도 placement에 따라 다양한 실행 패턴을 실현할 수 있다.

### 4.2 Implementation of different RLHF algorithms

논문은 이 API 설계의 장점을 다양한 RLHF 알고리즘 구현 예시로 보여 준다. PPO는 actor 생성, critic value 계산, reward 계산, advantage 계산, actor/critic 업데이트를 순서대로 호출하는 짧은 single-process 프로그램으로 작성할 수 있다. Safe-RLHF는 여기에 cost model 초기화와 cost 계산, auxiliary pretrain loss만 더하면 된다. ReMax는 critic 관련 코드를 빼고 actor 생성 호출을 하나 더 넣으면 된다.

![Figure 6](/assets/images/A%20Flexible%20and%20Efficient%20RLHF%20Framework/hybridflow_flexible_efficient_rlhf_framework_figure_06.png)

저자들이 강조하는 포인트는 “다양한 RLHF 알고리즘을 몇 줄의 코드 차이로 표현할 수 있다”는 것이다. 이는 단순한 코드 길이 자랑이 아니라, **분산 계산 코드와 알고리즘 로직이 분리되었기 때문**에 가능한 일이다. 연구자는 손실식과 데이터플로우만 바꾸면 되고, 각 모델의 분산 학습·생성 엔진은 재사용할 수 있다. 결국 HybridFlow는 RLHF 시스템을 특정 알고리즘에 종속된 구현물에서, 다양한 데이터플로우를 조합 가능한 실행 플랫폼으로 바꿔 놓는다.

## 5. 3D-HybridEngine

### 5.1 Parallel Groups

3D-HybridEngine은 actor의 학습과 생성을 같은 디바이스 집합 위에서 모두 효율적으로 수행하기 위해 설계되었다. 논문은 actor를 위해 학습 단계와 생성 단계에 서로 다른 3D parallel group을 구성한다. 학습은 일반적인 \(p\)-\(t\)-\(d\) 구조를 쓰고, 생성은 \(p_g\)-\(t_g\)-\(d_g\)-\(d\) 구조를 사용한다. 여기서 \(p\)와 \(t\)는 pipeline/tensor parallel 크기, \(d\)는 학습 단계의 data parallel 복제 수를 뜻한다. 생성 단계에서는 \(d_g\)개의 micro DP group을 도입해 더 큰 data parallel 효과를 낸다.

저자들은 동일한 총 GPU 수 \(N_a\)에 대해
\[
N_a = p \times t \times d = p_g \times t_g \times d_g \times d
\]
와 같은 대응을 사용한다. 핵심은 학습과 생성이 같은 GPU 수를 쓰더라도, **어떻게 그룹을 나누느냐에 따라 actor의 처리량과 전환 비용이 크게 달라진다**는 점이다. 생성은 auto-regressive 특성상 작은 model parallel과 큰 data parallel의 이득을 보는 경우가 많고, 학습은 반대로 큰 model parallel이 유리하다.

### 5.2 3D-HybridEngine Workflow

3D-HybridEngine의 workflow는 RLHF iteration 사이에서 actor 가중치를 어떻게 재배치하는가를 설명한다. iteration \(i\)의 actor 학습이 끝나면, 업데이트된 actor 가중치를 생성용 parallel group 안에서 사용할 수 있도록 먼저 all-gather한다. 그다음 프롬프트를 각 생성 replica에 배포하고 응답을 생성한다. 생성 결과는 micro DP group 안에서 다시 all-gather한 뒤, 학습용 parallel group에 맞게 재분할한다. 이후 actor loss를 계산하고, actor 가중치를 다시 업데이트한다.

![Figure 7](/assets/images/A%20Flexible%20and%20Efficient%20RLHF%20Framework/hybridflow_flexible_efficient_rlhf_framework_figure_07.png)

이 절차의 포인트는 “학습용 actor”와 “생성용 actor”를 아예 따로 두지 않고, **같은 장치 위에서 같은 actor를 서로 다른 grouping으로 재해석한다**는 점이다. 그러면 actor 복사본 중복을 피할 수 있고, 전환 자체를 엔진 수준에서 최적화할 수 있다.

### 5.3 Zero redundancy model resharding

기존 방식에서 actor 재분할 비용이 큰 이유는, 학습과 생성에 서로 다른 parallel group을 쓰면 어떤 GPU는 학습 시 들고 있던 가중치 조각과 생성 시 필요한 가중치 조각이 겹치지 않기 때문이다. 이 경우 새로운 생성용 가중치를 따로 모으고, 동시에 이후 학습을 위해 기존 학습용 가중치도 유지해야 해서 메모리 중복이 생긴다. 논문은 이런 vanilla한 grouping 방식을 **HybridFlow-V**라고 부른다.

![Figure 8](/assets/images/A%20Flexible%20and%20Efficient%20RLHF%20Framework/hybridflow_flexible_efficient_rlhf_framework_figure_08.png)

HybridFlow는 생성 단계의 parallel group을 다시 짜서, 같은 GPU 위에서 학습용 가중치와 생성용 가중치가 최대한 겹치도록 만든다. 예컨대 생성용 TP/PP group을 연속 rank 기준으로만 묶지 않고, micro DP group 단위로 재배열한다. 그 결과 많은 GPU에서 학습 단계에 이미 들고 있던 가중치 조각을 그대로 생성에도 재사용할 수 있다. 이것이 논문이 말하는 **zero redundancy model resharding**이다.

이 설계의 이점은 두 가지다. 첫째, 디바이스 메모리에서 추가 가중치 복제본이 필요 없어 peak memory가 줄어든다. 둘째, all-gather도 전체 GPU가 아니라 각 micro DP group 안에서 동시에 수행되므로 통신량이 줄어든다. 즉 3D-HybridEngine은 “전환을 없애는” 것이 아니라, **전환을 가중치 재사용이 많은 형태로 재구성해 비용을 줄인다.**

### 5.4 Transition overhead

이 절에서는 DeepSpeed-Chat, HybridFlow-V, HybridFlow를 actor 전환 비용 관점에서 비교한다. DeepSpeed-Chat은 전환 시 전체 GPU에 걸쳐 all-gather를 수행하므로 통신량과 메모리 부담이 가장 크다. HybridFlow-V는 training TP/PP group 안으로 all-gather 범위를 줄이지만, 여전히 생성용 가중치와 학습용 가중치가 겹치지 않는 GPU에서 메모리 중복이 생긴다. HybridFlow는 all-gather를 micro DP group 안으로 더 제한하고, 학습용 가중치를 생성에 재사용해 중복을 없앤다.

![Table 2](/assets/images/A%20Flexible%20and%20Efficient%20RLHF%20Framework/hybridflow_flexible_efficient_rlhf_framework_table_02.png)

Table 2의 메시지는 명확하다. HybridFlow는 통신량을 더 줄이고, peak memory를 generation partition 크기 수준으로 맞추며, **redundancy를 0으로 만든다.** 따라서 3D-HybridEngine의 이점은 단순히 생성 throughput을 높이는 데만 있지 않고, RLHF iteration마다 반복되는 학습-생성 전환을 구조적으로 가볍게 만드는 데 있다.

## 6. Auto Device Mapping

HybridFlow는 placement를 고정하지 않는다. 대신 주어진 RLHF 데이터플로우와 GPU 수에 대해, **어떤 모델을 같은 GPU 집합에 colocate할지**, **각 집합에 GPU를 몇 개 줄지**, **그 위에서 각 모델이 어떤 parallelism을 쓸지**를 자동으로 찾는다. 논문은 이것을 mapping 문제로 정의한다.

알고리즘 1의 흐름은 다음과 같다. 먼저 데이터플로우의 모델들을 여러 colocated set으로 나누는 모든 placement를 열거한다. PPO처럼 모델이 4개인 경우 가능한 placement가 15개가 된다. 그다음 각 colocated set에 대해 OOM이 나지 않는 최소 device allocation \(A_{min}\)을 찾는다. 이후 그 최소값에서 시작해 가능한 GPU allocation을 열거하고, 각 모델에 대해 `auto_parallel` 모듈로 최적 병렬화를 찾는다. 마지막으로 `d_cost` 모듈이 각 stage의 지연을 합산해 end-to-end RLHF iteration latency를 계산하고, 가장 짧은 mapping을 고른다.

논문의 pseudocode를 글로 풀면 다음과 같다.

1. 주어진 placement \(plm\)마다 평가를 시작한다.  
2. 각 colocated set에 OOM을 피하는 최소 GPU 수를 할당한다.  
3. 그보다 큰 feasible allocation을 모두 검사한다.  
4. allocation이 정해지면, 각 모델마다 `auto_parallel`로 \((p,t,d)\)를 고른다.  
5. 그런 뒤 stage별 실행 시간을 추정해 전체 RLHF iteration 비용을 계산한다.  
6. 가장 낮은 비용을 내는 placement와 allocation, parallelism 조합을 최종 mapping으로 채택한다.

`d_cost`가 중요한 이유는 RLHF가 단일 모델 학습이 아니기 때문이다. 같은 stage에 있는 모델이라도 서로 다른 GPU 집합에 있으면 병렬로 실행될 수 있고, 같은 colocated set 안에 있으면 순차 실행해야 한다. 그래서 stage latency는 단순 합이 아니라, **각 stage에서 병렬 집합 사이의 최대 지연과 순차 지연을 함께 고려한 값**이 된다.

논문은 또한 이 탐색의 worst-case 복잡도가 커질 수 있음을 인정한다. 배치 경우의 수는 결국 \(N\)개의 장치를 \(k\)개 모델에 나누는 조합 문제와 닮아 있으며, placement가 많아질수록 탐색 공간도 커진다. 대신 HybridFlow는 각 모델이 특정 GPU 수를 받을 때 최적 parallelism을 캐시해, 서로 다른 placement 간에도 같은 per-model 전략 탐색 결과를 재사용한다.

Appendix C의 Algorithm 2는 `auto_parallel`의 내부를 설명한다. 이 알고리즘은 입력으로 device allocation \(A\), 최소 allocation \(A_{min}\), workload \(W\), machine당 GPU 수 \(U\)를 받고, 가능한 \(p\)와 \(t\) 조합을 순회하면서 `simu` 모듈로 latency를 계산해 최소 비용 전략을 선택한다. 훈련, 추론, 생성 workload는 서로 다른 시뮬레이터로 다뤄지며, 생성 단계에서는 KVCache까지 고려한다.

## 7. Implementation

HybridFlow는 전체적으로 약 12k lines of code 규모로 구현되었다. 이 중 **Hybrid programming model**은 1.8k LoC 정도이며, 중앙 single controller는 Ray 위에서 RPC로 동작한다. 중간 데이터는 TensorDict에 저장되고, 각 모델의 함수 호출은 디바이스별 프로세스로 분산된다. 이 계층은 Megatron-LM, PyTorch FSDP, DeepSpeed, vLLM 같은 기존 학습·추론 엔진을 감싸는 역할을 한다.

**3D-HybridEngine**은 약 2.4k LoC로 구현되었고, Megatron-LM과 vLLM 위에 올라간다. 구현의 요점은 actor 학습용 가중치와 생성용 가중치를 메모리/CPU buffer와 NCCL collective로 관리하는 방식이다. 논문은 학습 중 generation weight를 CPU memory로 offload하고, 전환 시 다시 GPU로 가져오며, 생성에 필요한 KVCache 역시 생성 후 CPU로 내렸다가 다음 iteration에서 다시 불러오는 구조를 설명한다. 또한 micro DP group 단위 collect/concat을 통해 전환 통신을 국소화한다.

**Auto-Mapping Algorithm**은 약 1.9k LoC로 구현되었고, 학습·추론·생성 세 workload를 위한 simulator와 함께 CPU에서 사전 실행된다. 즉 RLHF 본 실행 전에 미리 최적 mapping을 찾은 뒤, 그 결과로 데이터플로우를 초기화한다. 이 구현 방식은 온라인 적응보다는 초기 배치 최적화에 가깝지만, RLHF처럼 iteration이 길고 반복적인 워크로드에는 충분히 실용적이다.

## 8. Evaluation

### 8.1 Experimental Setup

평가는 16대 머신, 총 128개의 NVIDIA A100 80GB GPU 클러스터에서 수행된다. 머신당 GPU는 8개이며, intra-machine 연결은 600Gb/s NVLink, inter-machine 대역폭은 200Gbps다. 소프트웨어 스택은 CUDA 12.1, PyTorch 2.1.2, Megatron-core 0.6.0, NCCL 2.18.1, vLLM 0.3.1이다.

실험 대상 RLHF 알고리즘은 PPO, ReMax, Safe-RLHF다. 모델군은 Llama 7B, 13B, 34B, 70B 규모를 사용한다. 기본 설정에서는 actor, critic, reference, reward 모델이 같은 크기를 가지며, Safe-RLHF는 여기에 cost model을 하나 더 둔다. ReMax는 critic 없이 actor 중심으로 동작한다. 혼합정밀도는 actor/critic 학습에 BF16 parameter와 FP32 gradient·optimizer state를 사용하고, 추론과 생성도 BF16로 수행한다.

비교 대상은 DeepSpeed-Chat v0.14.0, OpenRLHF v0.2.5, NeMo-Aligner v0.2.0이다. NeMo-Aligner는 ReMax를 지원하지 않으므로 해당 비교에서는 빠진다. 데이터셋은 HuggingFace의 `Dahoas/full-hh-rlhf`를 사용하며, 각 실험에서 입력 프롬프트 길이와 출력 응답 길이는 각각 1024, actor 입력 프롬프트의 global batch size는 1024로 맞춘다. PPO epoch는 1, update iteration은 8이다. 성능 지표는 RLHF throughput(tokens/sec)이며, 10회 warm-up 이후 5회 training iteration 평균을 사용한다.

### 8.2 End-to-End performance

![Figure 9](/assets/images/A%20Flexible%20and%20Efficient%20RLHF%20Framework/hybridflow_flexible_efficient_rlhf_framework_figure_09.png)

![Figure 10](/assets/images/A%20Flexible%20and%20Efficient%20RLHF%20Framework/hybridflow_flexible_efficient_rlhf_framework_figure_10.png)

![Figure 11](/assets/images/A%20Flexible%20and%20Efficient%20RLHF%20Framework/hybridflow_flexible_efficient_rlhf_framework_figure_11.png)

8.2의 핵심 메시지는 간단하다. **HybridFlow는 PPO, ReMax, Safe-RLHF 모두에서, 그리고 7B-70B 전 모델 규모에서 기존 시스템보다 일관되게 빠르다.** 특히 PPO 결과를 보면 DeepSpeed-Chat 대비 평균 3.67배, OpenRLHF 대비 평균 3.25배, NeMo-Aligner 대비 평균 12.52배 향상을 보이며, 최대 20.57배까지 올라간다. 절대 수치만 보면 PPO 34B 조건에서 가장 큰 격차가 나타나는데, 이는 HybridFlow가 생성 단계와 전환 단계를 동시에 줄여 주기 때문이다.

Figures 9-11을 보면 속도 향상은 특정 모델 하나에만 나타나는 것이 아니다. 7B처럼 작은 모델에서도 HybridFlow가 우세하고, 34B와 70B처럼 큰 모델에서는 격차가 더 커진다. 저자들은 그 이유를 세 가지로 설명한다. 첫째, 각 stage의 workload에 맞춰 병렬화 전략을 다르게 잡을 수 있다. 둘째, actor 학습-생성 전환이 훨씬 싸다. 셋째, placement를 workload에 맞게 조정해 GPU idle time과 KVCache 병목을 줄일 수 있다.

논문은 scalability도 별도로 강조한다. HybridFlow는 8 GPU 같은 비교적 작은 클러스터에서도 최소 2.09배 이상의 speedup을 보이고, GPU 수가 늘어나도 강한 scaling을 유지한다. OpenRLHF는 큰 클러스터에서 더 잘 버티는 편이지만 작은 클러스터에서는 비효율이 커지고, NeMo-Aligner는 생성 단계의 병렬화 제약이 병목이 된다. 결국 HybridFlow의 장점은 특정 baseline 하나를 이기는 것이 아니라, **작은 클러스터와 큰 클러스터 양쪽에서 모두 좋은 placement를 찾는다는 점**에 있다.

### 8.3 Model Placement

![Figure 12](/assets/images/A%20Flexible%20and%20Efficient%20RLHF%20Framework/hybridflow_flexible_efficient_rlhf_framework_figure_12.png)

![Figure 13](/assets/images/A%20Flexible%20and%20Efficient%20RLHF%20Framework/hybridflow_flexible_efficient_rlhf_framework_figure_13.png)

8.3에서는 placement 자체가 throughput에 얼마나 큰 영향을 주는지 따로 분석한다. 비교 대상은 네 가지다. `colocate`는 DeepSpeed-Chat처럼 모든 모델을 같은 GPU 집합에 두는 방식이고, `standalone`은 OpenRLHF처럼 모델을 분리하는 방식이다. `split`은 NeMo-Aligner처럼 actor/ref와 critic/rm을 나누는 방식이며, `hybridflow`는 알고리즘 1이 찾아낸 최적 placement다.

Figure 12에서 보이듯, **GPU 수에 따라 최적 placement가 달라진다.** 13B 모델에서는 16-64 GPU 구간에서 colocate가 가장 강하고, 128 GPU에서는 standalone이 더 좋다. 34B 모델에서는 96-128 GPU에서 split 쪽이 유리해진다. 즉 작은 클러스터에서는 모델을 같은 장치에 몰아 순차 실행하는 편이 stage 전환과 통신을 줄여 유리하고, 큰 클러스터에서는 actor와 critic 같은 큰 workload를 서로 다른 GPU에 나눠 병렬 실행하는 편이 더 좋다.

Figure 13은 actor와 reference는 13B지만 critic과 reward는 70B인 비대칭 설정을 본다. 여기서는 64 GPU까지는 colocate가 평균 44.8% 정도 더 빠르고, 96 GPU에서는 split이 유리해진다. 128 GPU에서는 알고리즘 1이 actor, reference, reward를 64 GPU에 묶고 critic을 나머지 64 GPU에 배치하는 구조를 선택한다. 이 결과는 placement가 단순한 구현 선택이 아니라 **모델 상대 크기와 stage별 compute/communication 균형에 따라 달라지는 최적화 문제**임을 잘 보여 준다.

### 8.4 3D-HybridEngine

![Figure 14](/assets/images/A%20Flexible%20and%20Efficient%20RLHF%20Framework/hybridflow_flexible_efficient_rlhf_framework_figure_14.png)

![Figure 15](/assets/images/A%20Flexible%20and%20Efficient%20RLHF%20Framework/hybridflow_flexible_efficient_rlhf_framework_figure_15.png)

8.4는 HybridFlow의 actor 엔진이 실제로 얼마나 큰 차이를 만드는지 보여 준다. Figure 14는 actor 학습과 생성 사이의 transition time을 비교한 결과다. HybridFlow는 OpenRLHF 대비 평균 55.2%(11.7초), 최대 89.1%(78.2초)까지 전환 시간을 줄인다. 특히 70B에서 격차가 극단적으로 커지는데, 이는 기존 방식이 큰 actor 모델을 두 단계 사이에서 동기화하거나 재분할하는 데 큰 비용을 쓰기 때문이다.

Figure 15는 16 GPU 환경에서 generation parallel size를 바꾸었을 때, transition time과 generation time의 trade-off를 보여 준다. 7B 모델에서는 \(t_g=2\), 13B에서는 \(t_g=4\) 정도가 가장 좋은 균형을 이룬다. \(t_g=8\), 즉 학습과 같은 TP 크기를 유지하면 generation latency가 가장 커지는데, 이는 GPU가 충분히 활용되지 못하기 때문이다. 반대로 \(t_g\)를 너무 더 줄이면 각 GPU가 더 큰 KVCache를 떠안아야 해서 오히려 생성이 느려진다. 이 실험은 HybridFlow가 단순히 전환 비용만 낮추는 것이 아니라, **학습과 생성의 병렬화 목표를 분리해 둘 다 최적화한다**는 점을 확인해 준다.

### 8.5 Algorithm Runtime

![Figure 16](/assets/images/A%20Flexible%20and%20Efficient%20RLHF%20Framework/hybridflow_flexible_efficient_rlhf_framework_figure_16.png)

Figure 16은 Auto Device Mapping 자체의 실행 시간을 보여 준다. 모델 크기와 GPU 수가 함께 커질수록 탐색 시간도 증가하지만, 증가 패턴은 비교적 완만하다. 중요한 것은 이 시간이 실제 RLHF 학습 시간보다 훨씬 짧다는 점이다. 논문은 큰 설정에서도 최적 placement 탐색 시간이 반나절이나 며칠이 아니라, 캐시까지 포함하면 **최대 반 시간 정도** 수준이라고 본다. 따라서 Auto Mapping은 “지나치게 무거운 사전 탐색”이 아니라, RLHF를 본격 실행하기 전에 충분히 지불할 만한 초기화 비용으로 해석된다.

## 9. Discussions

9절은 시스템 자체의 확장성과 적용 범위를 논의한다. 먼저 **Fault Tolerance** 측면에서 HybridFlow는 기존 체크포인팅 및 장애 복구 기법과 직교적이라고 설명한다. NCCL 오류와 silent corruption은 checksum으로 감지할 수 있고, single controller가 RPC를 통해 checkpoint를 조정해 actor/critic의 파라미터, dataloader ID, RNG state까지 저장할 수 있다. 또한 여유 replica가 있다면 redundancy 기반 fault tolerance나 CPU checkpoint도 적용 가능하다.

다음으로 저자들은 **placement insight**를 세 가지로 정리한다. 첫째, actor 생성은 병렬화가 제한적이므로 actor에 더 많은 GPU를 주는 것이 overall throughput에 직접적이다. 둘째, 각 모델 계산이 GPU를 충분히 활용할 수 있는 작은 클러스터에서는 colocate가 유리하다. 셋째, 큰 클러스터에서는 actor와 critic을 다른 디바이스에 배치해 학습·준비 단계를 병렬 실행하는 편이 좋다. 즉 “언제나 colocate”나 “언제나 split” 같은 고정 규칙은 없고, 자원 규모에 따라 답이 달라진다.

**Resource multiplexing**에 대한 논의도 흥미롭다. ResourcePool은 같은 디바이스에 여러 모델을 colocate하게 해 주지만, 현재 HybridFlow는 기본적으로 sequential execution 쪽에 더 가깝다. 이는 RLHF에서 모델이 크고 메모리 압박이 심해, fine-grained GPU sharing이 자칫 contention과 OOM을 부를 수 있기 때문이다. 저자들은 앞으로는 finer-grained auto-mapping, model offload, heterogeneous device 통합까지 고려한 방향이 유망하다고 본다.

마지막으로 **From alignment to reasoning**에서는 RLHF의 보상 신호를 꼭 neural reward model로 한정할 필요가 없다고 말한다. 코드 생성이라면 sandbox 실행 결과가 reward가 될 수 있고, 수학 추론이라면 정답 검산 함수가 reward 역할을 할 수 있다. HybridFlow는 이런 비신경 reward module도 remote function으로 감싸 single controller 아래에서 orchestration할 수 있으므로, RLHF를 넘어 보다 넓은 강화학습 응용에도 사용할 수 있다는 것이 저자들의 주장이다.

## 10. Related Work

관련 연구는 크게 세 묶음이다. 첫째는 **RL frameworks**다. 일반적인 RL 프레임워크들은 소형 DNN 기준으로 설계되었고, 최근 RLHF 시스템들 역시 대부분 multi-controller와 하드코딩된 동기화 로직 위에 구축되어 있다. 경험 재생이나 actor-learner 분리 같은 전통 RL 패턴은 발전했지만, RLHF처럼 여러 대형 LLM이 상호작용하는 데이터플로우 전체를 다루는 데는 한계가 있다.

둘째는 **LLM training and serving systems**다. TorchDDP, Horovod, DeepSpeed, Megatron-LM, vLLM 같은 시스템은 data/model parallelism, pipeline parallelism, continuous batching, chunked prefill 등으로 단일 모델 학습·생성을 최적화한다. 하지만 이들은 본질적으로 **하나의 모델을 빠르게 돌리는 엔진**이지, actor·critic·reward·reference가 함께 움직이는 RLHF 다중 모델 데이터플로우를 직접 표현하는 프레임워크는 아니다.

셋째는 **dataflow systems**다. MapReduce, Spark, Dryad, Naiad, Ray, Pathways 같은 시스템은 분산 작업 그래프를 실행하는 관점에서 영감을 준다. 특히 Ray는 전역 제어와 동적 task graph를 제공하고, Pathways는 단일 모델 내부에서 복잡한 병렬성을 표현한다. HybridFlow는 이 계열과 친연성이 있지만, 초점은 어디까지나 **RLHF라는 다중 LLM 데이터플로우를 표현하고, 모델 간 재분할을 포함한 end-to-end 실행을 최적화하는 것**에 있다.

## 11. Conclusion

HybridFlow의 결론은 분명하다. RLHF는 더 이상 “몇 개 모델을 적당히 붙여 놓고 PPO를 돌리는 시스템”으로 보기 어렵고, **복합적인 분산 데이터플로우**로 이해해야 한다. 이 논문은 그 관점에 맞춰, inter-node에는 single-controller를, intra-node에는 multi-controller를 쓰는 하이브리드 프로그래밍 모델을 제시한다. 여기에 actor 전환 비용을 줄이는 3D-HybridEngine과 placement를 자동으로 정하는 mapping 알고리즘을 결합해, RLHF 시스템을 더 유연하고 더 빠르게 만든다.

실험 결과도 그 메시지를 뒷받침한다. HybridFlow는 PPO, ReMax, Safe-RLHF 전반에서 7B-70B 모델과 8-128 GPU 환경에 걸쳐 기존 시스템 대비 1.53배에서 20.57배까지 throughput 향상을 달성한다. 따라서 이 논문의 핵심 기여는 특정 baseline을 이긴 것이 아니라, **RLHF 프레임워크를 하나의 일반적인 분산 시스템 문제로 재정의하고, 그에 맞는 계층형 실행 구조를 설계했다**는 데 있다.

## Acknowledgments

저자들은 shepherd인 Y. Charlie Hu와 anonymous reviewer들에게 감사를 표한다. 또한 Xin Liu, Yangrui Chen, Ningxin Zheng의 프로젝트 관련 피드백에 감사를 전한다. 연구비는 ByteDance Research Collaboration Project와 홍콩 RGC의 HKU 17204423, C7004-22G(CRF) 지원에서 일부 제공되었다.

## References

이 논문의 참고문헌은 크게 RLHF 알고리즘(PPO, Safe-RLHF, ReMax, GRPO), 일반 RL 프레임워크, 분산 LLM 학습·서빙 시스템(Megatron-LM, DeepSpeed, vLLM, FSDP, ZeRO), 데이터플로우 시스템(MapReduce, Spark, Dryad, Naiad, Ray, Pathways), 그리고 LLM 정렬 및 안전성 연구를 포괄한다. 본문 설명은 원문 참고문헌의 흐름을 따라 정리했으며, 상세한 서지 정보는 원문 논문을 직접 참고하면 된다.

## Appendix A. Primitive APIs in HybridFlow

HybridFlow의 primitive API는 “모델 내부 분산 계산을 재사용 가능한 함수 단위로 추상화한다”는 점에서 핵심적이다. actor, critic, reference policy, reward 모델은 각각 자신에게 필요한 연산을 primitive로 노출하고, RLHF 알고리즘 구현자는 이 함수를 조합해 데이터플로우를 만든다. 중요한 것은 이 primitive가 단순한 Python 함수가 아니라, 내부적으로는 Megatron-LM, FSDP, ZeRO, vLLM 같은 분산 엔진을 감싼 고수준 연산이라는 점이다.

Actor는 `generate_sequence`, `compute_log_prob`, `compute_loss`, `update_actor`를 제공한다. `generate_sequence`는 prompt batch를 받아 응답 batch와 응답 토큰별 log probability를 반환한다. `compute_log_prob`는 prompts와 responses에 대한 token-level log probability를 계산하며, PPO에서는 generation 시점의 log prob를 재현하는 데 쓰일 수 있다. `compute_loss`는 pretrain loss를 계산하고, `update_actor`는 advantage와 return, pretrain loss 등을 받아 실제 actor 업데이트를 수행한다.

Critic은 `compute_values`와 `update_critic`를 제공한다. 전자는 prompt-response 쌍에 대한 value를 계산하고, 후자는 value와 return을 받아 제곱오차 형태의 critic loss로 가중치를 갱신한다. Reference policy는 `compute_ref_log_prob`를 통해 reference log probability를 계산하고, reward model은 `compute_reward`로 scalar reward를 계산한다. 마지막으로 `compute_advantage`는 value와 reward를 이용해 advantage를 추정하는 순수 수치 계산 함수로, 별도 모델 forward를 필요로 하지 않는다.

![Table 4](/assets/images/A%20Flexible%20and%20Efficient%20RLHF%20Framework/hybridflow_flexible_efficient_rlhf_framework_table_04.png)

Table 4는 이 primitive API의 의미를 모델별로 정리한다. 결국 Appendix A의 핵심은, HybridFlow가 RLHF를 “분산 시스템 수준의 복잡한 프로그램”이 아니라, **primitive API를 조합한 상위 데이터플로우**로 바라보게 만든다는 점이다.

## Appendix B. Transfer Protocols

Transfer protocol은 HybridFlow에서 모델 간 데이터 재분할을 담당하는 핵심 추상화다. 각 protocol은 `collect function`과 `distribute function`의 쌍으로 구성된다. collect는 어떤 모델 연산의 출력을 single controller 또는 다른 형식으로 모으고, distribute는 다음 모델의 입력 병렬화 구조에 맞게 데이터를 나눈다. 이 추상화 덕분에 모델 계산 코드는 오직 자기 local computation만 신경 쓰고, 모델 사이의 데이터 reshaping과 multicast는 protocol 레벨에서 처리할 수 있다.

![Table 3](/assets/images/A%20Flexible%20and%20Efficient%20RLHF%20Framework/hybridflow_flexible_efficient_rlhf_framework_table_03.png)

Table 3에는 대표 protocol이 정리되어 있다. `ONE_TO_ALL`은 모든 rank에 broadcast하는 가장 단순한 경우다. `3D_PROTO`는 3D parallel training에서 흔한 경우로, DP group 내에서 데이터를 gather/concatenate한 뒤 다시 broadcast한다. `3D_ALL_MICRO_DP`는 HybridEngine과 함께 쓰이며, training과 inference 사이를 오갈 때 policy model의 3D parallel 스킴을 처리한다. `3D_PP_ONLY`는 TP와 DP 그룹이 동일할 때 weight name 등을 점검하는 데 유용하다. `DP_PROTO`는 data-parallel training 모델용이고, `ALL_TO_ALL`은 주로 debugging처럼 각 워커 입력을 수동으로 지정하고 출력을 직접 확인하고 싶을 때 쓰인다.

Appendix B가 보여 주는 것은, HybridFlow의 유연성이 단순한 API 분리만에서 오지 않는다는 점이다. 진짜 핵심은 **모델 간 데이터 재분할 자체를 1급 추상화로 끌어올렸다는 것**이다.

## Appendix C. Auto-Parallelism Algorithm

Appendix C의 Algorithm 2는 `auto_parallel`이 한 모델의 병렬화 전략을 어떻게 고르는지 설명한다. 입력은 device allocation \(A\), minimal allocation \(A_{min}\), workload \(W\), machine당 GPU 수 \(U\)다. 알고리즘은 먼저 현재 모델 \(l\)에 할당된 GPU 수 \(N_l\)과 최소 병렬화 크기 \(p_{min}, t_{min}\)을 가져온다. 그다음 가능한 \(t\)와 \(p\) 범위를 순회하고, 각 조합에 대해
\[
d = \frac{N_l}{p \times t}
\]
를 계산해 \((p,t,d)\) 형태의 후보 병렬화를 만든 뒤, `simu(para_plan, l, W[l])`로 latency를 추정한다. 가장 작은 cost를 주는 계획이 최종 parallelism이 된다.

논문은 이 시뮬레이터가 training, inference, generation 각각에 대해 서로 다른 analytical model을 가진다고 설명한다. 특히 actor 모델은 먼저 training 단계에서 필요한 메모리를 계산하고, generation 단계에서는 batch size와 max sequence length로부터 KVCache 요구량을 계산한다. 만약 현재 model-parallel 크기로 generation의 weights와 KVCache를 함께 담을 수 없으면, model-parallel 크기를 늘린 뒤 다시 latency를 비교한다. 결국 `auto_parallel`은 단순히 계산량만 보는 것이 아니라, **메모리 적합성까지 포함한 latency minimization**을 수행한다.

저자들은 동시에 한계도 인정한다. variable KVCache 크기를 더 정확히 반영하는 autoregressive generation simulator가 있다면, auto-mapping 품질은 더 좋아질 수 있다. 즉 Appendix C는 현재 알고리즘이 충분히 실용적이지만, 장기적으로는 더 정교한 workload model로 확장될 수 있음을 보여 준다.
