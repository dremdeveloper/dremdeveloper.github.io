# Efficient Memory Management for Large Language Model Serving with PagedAttention

## Abstract

대규모 언어 모델 서비스의 처리량을 높이려면 한 번에 충분히 많은 요청을 배치로 묶어 처리해야 한다. 하지만 기존 시스템은 요청마다 필요한 KV cache의 크기가 매우 크고, 생성이 진행되면서 이 메모리가 계속 늘어나고 줄어드는 특성 때문에 높은 배치 효율을 내기 어렵다. 메모리 관리가 비효율적이면 단편화와 중복 저장 때문에 실제로 사용할 수 있는 KV cache 공간이 크게 줄어들고, 결국 동시에 처리할 수 있는 요청 수가 제한된다.

이 논문은 이 문제를 해결하기 위해 운영체제의 가상 메모리와 페이징 기법에서 영감을 얻은 **PagedAttention**을 제안한다. PagedAttention은 attention이 참조하는 key/value를 반드시 연속된 메모리에 둘 필요가 없도록 만들고, KV cache를 고정 크기 블록 단위로 관리한다. 이 위에 구축된 시스템이 **vLLM**이다. vLLM은 KV cache 메모리 낭비를 거의 0에 가깝게 줄이고, 하나의 요청 내부 또는 서로 다른 요청 사이에서도 KV cache를 유연하게 공유할 수 있도록 설계된다.

논문의 실험 결과는 vLLM이 FasterTransformer, Orca 같은 기존 최첨단 시스템과 비교해 비슷한 지연시간 수준을 유지하면서도 처리량을 **2배에서 4배 정도** 높일 수 있음을 보여준다. 특히 시퀀스가 길수록, 모델이 클수록, 디코딩 방식이 복잡할수록 개선폭이 더 커진다.

## 1. Introduction

최근 GPT, PaLM 같은 대규모 언어 모델은 코드 보조, 범용 챗봇, 번역, 질의응답 같은 다양한 서비스를 가능하게 만들었다. 문제는 이러한 서비스를 운영하는 비용이 매우 크다는 점이다. LLM 요청 하나를 처리하는 비용은 전통적인 검색 질의보다 훨씬 비싸고, 따라서 서비스 관점에서는 **요청당 비용을 줄이기 위해 처리량을 극대화하는 것**이 중요해진다.

LLM의 핵심은 autoregressive Transformer이다. 모델은 입력 프롬프트와 지금까지 생성한 토큰들을 바탕으로 다음 토큰을 하나씩 생성한다. 이때 토큰 생성은 순차적이므로, 특히 생성 단계에서는 GPU의 연산 성능을 충분히 활용하지 못하고 메모리 접근이 병목이 되기 쉽다. 논문은 이 지점을 LLM serving의 본질적인 한계로 본다.

13B 파라미터 모델을 NVIDIA A100 40GB에서 서빙하는 예를 보면, GPU 메모리의 상당 부분은 모델 파라미터가 고정적으로 차지하고 있고, 그 다음 큰 비중을 차지하는 것이 요청마다 동적으로 증가하는 KV cache이다. activation은 상대적으로 작은 비중만 차지한다. 결국 한 번에 얼마나 많은 요청을 처리할 수 있는지는 KV cache를 얼마나 효율적으로 저장하느냐에 크게 좌우된다.

![Figure 1. Memory layout and throughput curve.](../../assets/images/Efficient%20Memory%20Management%20for%20Large%20Language%20Model%20Serving%20with%20PagedAttention/vllm_figure_01_memory_layout_and_throughput_curve.png)

기존 LLM serving 시스템은 보통 하나의 요청에 해당하는 KV cache를 **연속된 메모리 공간**에 저장한다. 이는 기존 딥러닝 프레임워크의 텐서 연산이 연속 메모리를 전제로 하는 경우가 많기 때문이다. 그러나 KV cache는 일반적인 딥러닝 텐서와 다르다. 첫째, 생성이 진행될수록 길이가 계속 늘어난다. 둘째, 요청이 언제 끝날지, 최종 길이가 얼마나 될지 미리 알 수 없다. 이 특성 때문에 연속 메모리 기반 방식은 두 가지 문제를 낳는다.

첫 번째는 **메모리 단편화**다. 최대 시퀀스 길이를 기준으로 공간을 미리 크게 예약해 두면 실제로 사용되지 않는 내부 단편화가 커지고, 요청별 할당 크기가 들쭉날쭉하면 외부 단편화도 발생한다. 두 번째는 **메모리 공유 불가**다. 병렬 샘플링이나 beam search처럼 여러 시퀀스가 프롬프트의 일부를 공유하는 상황에서도, 기존 시스템은 각 시퀀스의 KV cache를 별도 연속 공간에 두기 때문에 같은 prefix를 중복 저장하게 된다.

논문은 실제 프로파일링 결과를 통해 기존 시스템에서 실제 토큰 상태를 담는 데 사용되는 KV cache 비율이 매우 낮음을 보인다. 즉, GPU 메모리는 이미 충분히 큰데도 제대로 활용되지 못하고 있으며, 이것이 곧 배치 크기와 처리량의 상한으로 이어진다.

![Figure 2. KV cache memory waste breakdown.](../../assets/images/Efficient%20Memory%20Management%20for%20Large%20Language%20Model%20Serving%20with%20PagedAttention/vllm_figure_02_kv_cache_memory_waste_breakdown.png)

이 문제를 해결하기 위해 논문은 KV cache를 고정 크기 블록으로 쪼개고, 논리적 순서와 물리적 위치를 분리하는 PagedAttention을 제안한다. 운영체제의 가상 메모리처럼, 논리적으로 이어진 KV block이 물리적으로는 떨어진 위치에 있어도 attention 계산이 가능하도록 만든 것이다. 여기에 block-level 메모리 관리와 선점형 스케줄링을 결합한 시스템이 vLLM이다.

논문이 정리하는 기여는 네 가지로 볼 수 있다. 첫째, LLM serving에서 KV cache 메모리 관리가 처리량을 결정하는 핵심 병목임을 구체적으로 드러낸다. 둘째, 비연속 메모리에 저장된 KV cache 위에서도 정확한 attention을 수행하는 PagedAttention을 제안한다. 셋째, 이를 실제 분산 serving 엔진인 vLLM으로 구현한다. 넷째, 다양한 모델과 워크로드, 다양한 디코딩 방식에서 기존 시스템보다 크게 높은 처리량을 실험으로 입증한다.

## 2. Background

### 2.1 Transformer-Based Large Language Models

언어 모델링의 목표는 토큰 시퀀스 $x=(x_1,\dots,x_n)$의 확률을 모델링하는 것이다. 언어는 순차 구조를 가지므로, 전체 시퀀스의 결합 확률은 다음과 같이 조건부 확률의 곱으로 분해할 수 있다.

$$
P(x)=P(x_1)\cdot P(x_2\mid x_1)\cdots P(x_n\mid x_1,\dots,x_{n-1})
\tag{1}
$$

Transformer는 이 autoregressive 분해를 대규모로 구현하는 사실상의 표준 아키텍처다. 그 핵심 연산이 self-attention이다. 입력 은닉 상태 시퀀스 $(x_1,\dots,x_n)\in\mathbb{R}^{n\times d}$가 주어지면, 각 위치 $i$의 hidden state로부터 query, key, value를 다음처럼 만든다.

$$
q_i=W_qx_i,\qquad k_i=W_kx_i,\qquad v_i=W_vx_i
\tag{2}
$$

그리고 위치 $i$에서의 attention은 앞선 모든 위치의 key와 value를 이용해 계산된다.

$$
a_{ij}=
\frac{\exp\left(q_i^\top k_j/\sqrt{d}\right)}
{\sum_{t=1}^{i}\exp\left(q_i^\top k_t/\sqrt{d}\right)},
\qquad
o_i=\sum_{j=1}^{i}a_{ij}v_j
\tag{3}
$$

여기서 $a_{ij}$는 위치 $i$의 query가 위치 $j$의 key를 얼마나 참조하는지를 나타내는 attention score이고, $o_i$는 value의 가중합으로 만들어진 출력이다. 결국 Transformer는 “현재 위치가 앞선 모든 토큰의 정보를 얼마나 읽어올 것인가”를 이 연산으로 결정한다.

논문은 attention 외의 나머지 구성 요소들, 예를 들어 embedding, feed-forward, layer normalization, residual connection, output logit 계산, 그리고 query/key/value를 만드는 선형 변환은 대부분 위치별로 독립 적용된다고 설명한다. 따라서 **이전 모든 토큰의 상태를 참조해야 하는 부분은 attention이며, 이 때문에 key/value를 캐시하는 KV cache가 중요해진다.**

### 2.2 LLM Service & Autoregressive Generation

실제 LLM 서비스에서는 사용자 프롬프트 $(x_1,\dots,x_n)$가 들어오고, 모델은 출력 토큰 $(x_{n+1},\dots,x_{n+T})$를 생성한다. 논문은 프롬프트와 출력의 연결된 전체를 **sequence**라고 부른다.

autoregressive 구조 때문에 모델은 한 번에 전체 출력을 만들 수 없고, 다음 토큰을 하나씩 샘플링해야 한다. 이때 매번 이전 모든 토큰의 key/value가 필요하므로, 이미 계산한 key/value를 다시 쓰기 위해 저장한 것이 **KV cache**다. 중요한 점은 “같은 토큰 문자열이라도 위치가 다르면 KV cache가 다르다”는 점이다. 즉, KV cache는 단순 토큰 ID의 캐시가 아니라 **문맥과 위치를 반영한 상태**다.

논문은 LLM 생성 과정을 두 단계로 나눈다.

첫 번째는 **prompt phase**다. 이 단계에서는 전체 프롬프트가 한 번에 주어지므로, 프롬프트의 모든 토큰에 대한 key/value를 한꺼번에 계산할 수 있다. 행렬-행렬 곱 형태로 병렬화가 잘 되기 때문에 GPU 활용률도 높다.

두 번째는 **autoregressive generation phase**다. 이 단계에서는 매 반복마다 새 토큰 하나를 입력으로 받아 다음 토큰의 확률을 계산한다. 이전 위치들의 key/value는 KV cache에서 가져오고, 현재 위치의 key/value만 새로 만든다. 이 과정은 반복 간 의존성이 있어 병렬화가 어렵고, 행렬-벡터 곱 중심이라 GPU를 덜 효율적으로 쓴다. 그래서 생성 단계는 계산보다 메모리 접근이 병목이 되는 **memory-bound workload**가 된다.

### 2.3 Batching Techniques for LLMs

LLM serving의 처리량을 높이려면 여러 요청을 한 번에 배치로 처리해야 한다. 동일한 모델 가중치를 여러 요청이 공유하므로, 배치가 커질수록 가중치 이동 비용이 여러 요청에 amortize되고 GPU 계산 효율도 좋아진다.

하지만 단순 배치는 두 가지 이유로 어렵다. 첫째, 요청은 서로 다른 시점에 도착한다. 너무 단순한 방식은 먼저 온 요청을 뒤에 온 요청 때문에 기다리게 하거나, 반대로 새 요청을 오래 미루게 만든다. 둘째, 요청마다 입력 길이와 출력 길이가 크게 다르다. 길이를 맞추려고 padding을 넣으면 메모리와 연산이 낭비된다.

이를 해결하기 위해 기존 연구들은 **cellular batching**이나 **iteration-level scheduling** 같은 세밀한 배칭 기법을 제안했다. 이 방식들은 “요청 전체”가 아니라 “생성 반복 단위”로 스케줄링한다. 한 iteration이 끝날 때마다 완료된 요청은 배치에서 빼고 새 요청을 넣는다. 그러면 새 요청은 전체 배치가 끝날 때까지 기다리지 않고 한 iteration만 기다린 뒤 실행될 수 있다. 또한 특수한 GPU kernel을 이용하면 padding도 줄일 수 있다.

그러나 이런 정교한 배칭이 있어도, 실제로 동시에 담아둘 수 있는 요청 수는 결국 **GPU 메모리에 KV cache를 얼마나 많이 올릴 수 있는가**에 의해 제한된다. 이 지점에서 논문은 계산보다 메모리 관리가 더 중요하다고 본다.

## 3. Memory Challenges in LLM Serving

논문은 LLM serving의 처리량이 메모리 병목을 받는 이유를 세 가지로 정리한다.

첫째는 **KV cache 자체가 너무 크다**는 점이다. 예를 들어 13B OPT 모델에서 토큰 하나의 KV cache는 약 800KB다. 계산식은 key와 value 두 종류, hidden size 5120, layer 수 40, FP16의 2바이트를 곱해 얻는다. OPT의 최대 길이가 2048 토큰이므로 요청 하나의 KV cache는 최대 1.6GB까지 커질 수 있다. GPU 메모리가 수십 GB 수준이라는 점을 생각하면, 아무리 잘 써도 동시에 올릴 수 있는 요청 수는 많지 않다.

둘째는 **디코딩 방식이 복잡해질수록 메모리 관리도 복잡해진다**는 점이다. 병렬 샘플링에서는 같은 프롬프트를 공유하는 여러 출력이 있으므로 프롬프트 부분의 KV cache를 공유할 수 있다. 논문은 실험에서 프롬프트 부분이 전체 KV cache의 12%를 차지하는 경우를 예로 든다. beam search에서는 공유 비율이 더 커져 최대 55% 수준의 메모리 절감이 가능하지만, 어떤 부분을 공유할 수 있는지는 디코딩 진행에 따라 계속 바뀐다.

셋째는 **입력 길이와 출력 길이를 미리 모른다는 점**이다. 프롬프트 길이는 요청마다 크게 다르고, 출력 길이는 입력과 모델의 샘플링 결과에 따라 달라진다. 따라서 어떤 시점에 어떤 요청의 KV cache가 얼마나 커질지 정확히 예측하기 어렵다. 메모리가 부족해지면 일부 요청의 KV cache를 내리거나, 나중에 다시 올리는 스케줄링이 필요해진다.

### 3.1 Memory Management in Existing Systems

기존 시스템은 대체로 하나의 요청에 대한 KV cache를 **하나의 연속 텐서**로 잡아 둔다. 이때 최종 출력 길이를 미리 모르기 때문에, 요청의 최대 가능 길이를 기준으로 큰 메모리 chunk를 예약해 둔다.

![Figure 3. Existing systems KV cache fragmentation.](../../assets/images/Efficient%20Memory%20Management%20for%20Large%20Language%20Model%20Serving%20with%20PagedAttention/vllm_figure_03_existing_systems_kv_cache_fragmentation.png)

Figure 3은 이 방식의 문제를 직관적으로 보여준다. 논문은 두 요청을 예로 든다. 요청 A는 최대 길이 2048, 요청 B는 최대 길이 512다. 이 구조에서는 세 가지 낭비가 발생한다.

첫 번째는 **reserved slots**다. 아직 생성되지 않았지만 미래 토큰을 위해 미리 잡아 둔 공간이다. 언젠가는 쓰일 수 있지만, 요청이 살아 있는 동안 다른 요청이 이 공간을 쓸 수 없으므로 현재 시점에서는 사실상 배치를 제한하는 낭비다.

두 번째는 **internal fragmentation**이다. 실제 출력이 최대 길이보다 훨씬 짧게 끝나면, 미리 크게 잡아 둔 나머지 공간은 끝까지 쓰이지 않는다. 이 공간은 요청이 끝나야 비로소 낭비였다는 사실이 확정된다.

세 번째는 **external fragmentation**이다. 서로 다른 크기의 chunk를 계속 할당하고 해제하다 보면, 총량은 충분해도 연속된 큰 공간을 만들 수 없게 된다. 이 경우 남은 메모리가 있어도 새 요청을 넣지 못한다.

논문은 실제 실험에서 기존 시스템의 실효 메모리 비율이 **최저 20.4% 수준**까지 떨어질 수 있다고 보고한다. 즉, GPU 메모리 대부분이 실제 토큰 상태가 아니라 예약 공간, 단편화, 과도한 여유분에 묶여 있다는 뜻이다.

단편화를 줄이기 위한 compaction 같은 방법을 생각할 수도 있다. 하지만 LLM serving에서 KV cache는 매우 크고, 그것을 계속 재배치하는 것은 성능에 치명적이다. 더구나 compaction만으로는 병렬 샘플링, beam search, shared prefix에서 필요한 **메모리 공유 문제**를 해결하지 못한다.

## 4. Method

논문은 이러한 문제를 해결하기 위해 **PagedAttention**이라는 attention 알고리즘과, 이를 실제 시스템으로 구현한 **vLLM**을 제안한다. 전체 구조에서 vLLM은 중앙의 scheduler가 여러 GPU worker를 조율하고, KV cache manager가 물리 메모리의 KV block을 관리하는 구조를 가진다.

![Figure 4. vLLM system overview.](../../assets/images/Efficient%20Memory%20Management%20for%20Large%20Language%20Model%20Serving%20with%20PagedAttention/vllm_figure_04_vllm_system_overview.png)

핵심 아이디어는 단순하다. 요청의 KV cache를 “길게 이어진 하나의 연속 버퍼”로 보지 않고, **고정 크기 block들의 논리적 열**로 본다. 그 block들이 물리적으로 어디에 놓이는지는 별도의 block table이 관리한다. 이렇게 하면 최대 길이를 미리 다 예약하지 않아도 되고, 같은 prefix를 여러 시퀀스가 공유하는 것도 자연스럽게 지원된다.

### 4.1 PagedAttention

PagedAttention의 출발점은 운영체제의 가상 메모리다. 사용자 프로그램은 마치 연속 주소 공간을 가진 것처럼 보이지만, 실제 물리 메모리는 page 단위로 흩어져 있어도 된다. 논문은 이를 KV cache에 그대로 옮긴다. 즉, 논리적으로 연속된 토큰들의 KV cache를 **KV block**이라는 고정 크기 단위로 나누고, 물리적으로는 비연속 공간에 둘 수 있게 만든다.

논문은 Transformer에서 한 토큰이 여러 layer와 여러 attention head에 걸쳐 key/value를 가진다는 점도 짚는다. 구현 관점에서는 한 토큰의 모든 key/value를 한 block으로 묶어도 되고, layer나 head마다 따로 관리해도 된다. 논문은 성능 차이는 없고 구현 편의상 후자를 택했다고 설명한다.

![Figure 5. PagedAttention with non-contiguous KV blocks.](../../assets/images/Efficient%20Memory%20Management%20for%20Large%20Language%20Model%20Serving%20with%20PagedAttention/vllm_figure_05_pagedattention_non_contiguous_kv_blocks.png)

block size를 $B$라고 두면, $j$번째 key block과 value block은 다음처럼 쓸 수 있다.

$$
K_j=(k_{(j-1)B+1},\dots,k_{jB}),\qquad
V_j=(v_{(j-1)B+1},\dots,v_{jB})
$$

논문은 기존 attention 식을 block 단위로 다시 표현한다. 핵심은 query $q_i$가 각 block의 key들과 곱해져 block별 attention score를 만들고, 정규화는 지금까지의 모든 block을 대상으로 수행되며, 최종 출력은 각 block의 value와 그 block의 attention score를 곱해 더하는 형태라는 점이다. 논문이 쓰는 blockwise 표현을 그대로 쓰면 다음과 같다.

$$
A_{ij}=(a_{i,(j-1)B+1},\dots,a_{i,jB}),\qquad
o_i=\sum_{j=1}^{\lceil i/B\rceil}V_jA_{ij}^{\top}
\tag{4}
$$

여기서 중요한 것은 수식의 겉모양보다도 계산 방식의 변화다. 기존 attention은 과거 위치들의 key/value가 연속 텐서라고 가정하기 쉽지만, PagedAttention은 **각 block을 독립적으로 찾아가 읽고 계산한 뒤 결과를 합치는 방식**으로 attention을 수행한다. 그래서 물리 메모리에서 block이 떨어져 있어도 논리적 순서만 보존되면 정확한 attention이 가능하다.

논문은 Figure 5에서 query 토큰 “forth”가 세 개의 흩어진 block에 저장된 key/value를 읽는 예를 보여준다. block 0에는 “Four score and seven”, block 1에는 그 다음 토큰들, block 2에는 더 뒤의 토큰들이 저장되어 있을 수 있다. attention kernel은 이들을 물리 위치와 무관하게 block 단위로 찾아가 점곱과 가중합을 수행한다.

요컨대 PagedAttention의 의미는 “attention 계산을 block table을 통한 간접 참조 위에서 수행할 수 있게 만든 것”이다. 이 덕분에 그 위의 메모리 관리자도 가상 메모리처럼 훨씬 유연해진다.

### 4.2 KV Cache Manager

vLLM의 KV cache manager는 운영체제의 virtual memory manager와 거의 같은 역할을 한다. 논리적으로는 요청마다 KV block들이 순서대로 이어져 있지만, 물리적으로는 GPU DRAM 안의 어느 block과도 연결될 수 있다.

하나의 요청은 **logical KV blocks**의 연속으로 표현된다. 새 토큰이 생성될수록 왼쪽에서 오른쪽으로 block이 차오르고, 마지막 block의 빈 칸은 미래 생성용으로 남는다. 반면 GPU 쪽에는 block engine이 큰 연속 메모리 chunk를 확보해 놓고, 그것을 다시 **physical KV blocks**로 나눈다. 그리고 요청별로 block table을 유지해 “logical block $j$가 physical block 몇 번에 대응하는가”, “그 block 안에 몇 칸이 실제로 채워졌는가”를 기록한다.

이 구조의 장점은 명확하다. 더 이상 최대 길이 전체를 미리 예약할 필요가 없다. 실제로 토큰이 늘어날 때에만 새로운 physical block을 할당하면 된다. 논리 block과 물리 block을 분리했기 때문에 외부 단편화가 사실상 사라지고, 내부 낭비도 마지막 block의 일부 빈 칸 정도로만 제한된다.

### 4.3 Decoding with PagedAttention and vLLM

논문은 Figure 6을 통해 vLLM이 하나의 입력 시퀀스를 어떻게 디코딩하는지 단계별로 설명한다.

![Figure 6. Block table translation in vLLM.](../../assets/images/Efficient%20Memory%20Management%20for%20Large%20Language%20Model%20Serving%20with%20PagedAttention/vllm_figure_06_block_table_translation_in_vllm.png)

예시 프롬프트는 7개의 토큰으로 이루어져 있다. block size가 4라고 하면, 프롬프트의 KV cache를 담기 위해 처음 두 개의 logical block이면 충분하다. vLLM은 logical block 0과 1을 각각 physical block 7과 1에 매핑한다. prefill 단계에서는 프롬프트 전체와 첫 출력 토큰에 대한 KV cache를 일반 self-attention으로 계산하고, 앞 4개 토큰은 logical block 0에, 나머지 3개 토큰은 logical block 1에 저장한다. 이때 logical block 1의 마지막 칸 하나는 아직 비어 있고, 이후 생성 단계에서 쓰이게 된다.

첫 번째 autoregressive decoding step에서는 새 토큰 하나를 생성하고, 기존 KV cache는 physical block 7과 1에서 읽는다. 마지막 logical block에 빈 칸이 있으므로 새 토큰의 KV cache는 같은 physical block 안에 그냥 이어서 저장된다. 이 경우 바뀌는 것은 block table의 “filled count”뿐이다.

두 번째 step에 가면 마지막 logical block이 가득 찼으므로, 이제는 새 logical block이 필요하다. vLLM은 그 시점에 가서 새 physical block 하나를 할당하고, 이를 block table에 추가한다. 즉, **길이가 늘어날 때에만 block이 새로 생긴다.**

이 구조는 한 요청만이 아니라 전체 배치 수준에서도 작동한다. 매 iteration마다 vLLM은 현재 배치할 candidate sequence들을 고르고, 새로 필요한 logical block에 대응하는 physical block을 먼저 할당한다. 그 다음 현재 iteration에서 들어갈 입력 토큰들을 하나로 묶어 모델에 넣는다. prompt phase에 있는 요청은 프롬프트 토큰 전체가 들어가고, generation phase에 있는 요청은 가장 최근 토큰 하나가 들어간다. attention layer에서는 PagedAttention kernel이 block table을 보고 과거 KV cache를 읽고, 이번 step의 새 KV cache를 해당 physical block에 쓴다.

논문은 block size의 trade-off도 강조한다. block이 너무 작으면 마지막 block 낭비는 줄지만, 너무 잘게 쪼개져 병렬성이 떨어지고 kernel 호출 및 간접 참조 부담이 커질 수 있다. 반대로 block이 너무 크면 한 block 안의 빈 칸이 많아져 내부 단편화가 커진다. 이 trade-off는 뒤의 ablation에서 다시 분석된다.

가장 중요한 결론은, vLLM에서는 한 요청 때문에 낭비되는 메모리가 **최대 한 block 분량**으로 억제된다는 점이다. block은 왼쪽에서 오른쪽으로 차고, 앞 block들이 다 찬 뒤에만 새 block을 할당하므로, 낭비는 마지막 partially-filled block에만 머문다. 이것이 곧 더 많은 요청을 동시에 메모리에 올릴 수 있게 하고 처리량을 높인다.

![Figure 7. Two requests sharing physical KV space.](../../assets/images/Efficient%20Memory%20Management%20for%20Large%20Language%20Model%20Serving%20with%20PagedAttention/vllm_figure_07_two_requests_shared_physical_kv_space.png)

Figure 7은 서로 다른 두 요청의 logical block들이 GPU DRAM의 physical block 공간 안에서 섞여 저장되는 모습을 보여준다. 각각의 요청 입장에서는 logical block들이 순서대로 이어져 있지만, 실제 물리 메모리에서는 이웃한 logical block이 떨어져 있을 수 있다. 중요한 것은 연속된 물리 공간이 아니라 **block table에 의해 논리 순서가 보장된다**는 점이다.

### 4.4 Application to Other Decoding Scenarios

지금까지의 설명은 하나의 프롬프트에서 하나의 출력 시퀀스를 생성하는 기본적인 greedy decoding이나 sampling에 해당한다. 하지만 실제 서비스는 병렬 샘플링, beam search, shared prefix처럼 훨씬 복잡한 시나리오를 다룬다. 논문은 vLLM이 이러한 경우에도 자연스럽게 적용된다고 보여준다.

#### Parallel sampling

코드 보조 같은 응용에서는 하나의 입력에 대해 여러 개의 샘플 출력을 동시에 생성하는 경우가 많다. 이때 여러 출력은 같은 프롬프트를 공유하므로, 프롬프트 부분의 KV cache도 공유할 수 있다.

![Figure 8. Parallel sampling example.](../../assets/images/Efficient%20Memory%20Management%20for%20Large%20Language%20Model%20Serving%20with%20PagedAttention/vllm_figure_08_parallel_sampling_example.png)

Figure 8의 예에서 sample A1과 A2는 같은 프롬프트를 가진다. vLLM은 프롬프트를 저장한 logical block 0, 1이 두 시퀀스 모두에서 동일한 physical block 7, 1을 가리키게 만든다. 하나의 physical block이 여러 logical block에 매핑될 수 있으므로, vLLM은 각 physical block에 **reference count**를 둔다.

문제는 두 샘플이 이후 서로 다른 토큰을 생성하기 시작할 때다. 이때 shared block을 그대로 같이 쓰면 안 되므로, vLLM은 운영체제의 fork와 유사한 **copy-on-write**를 block 단위로 적용한다. 예를 들어 A1이 아직 여러 시퀀스가 공유 중인 physical block 1에 새 KV cache를 쓰려고 하면, vLLM은 refcount가 1보다 큰 것을 확인하고 새로운 physical block을 하나 할당한 뒤 기존 내용을 복사한다. 그 다음 A1은 새 block에 쓰고, A2는 기존 block에 쓴다. 이런 식으로 대부분의 프롬프트 KV cache는 공유한 채, 실제로 달라지는 시점의 block만 분기한다.

#### Beam search

beam search는 parallel sampling보다 더 강한 공유를 만든다. beam width를 $k$라고 하면, 각 step에서 후보 시퀀스를 확장한 뒤 확률이 높은 상위 $k$개만 남긴다. 이 과정에서 후보들은 긴 prefix를 공유하다가 어느 시점에 갈라지고, 다시 다른 후보의 계통이 살아남기도 한다.

![Figure 9. Beam search example.](../../assets/images/Efficient%20Memory%20Management%20for%20Large%20Language%20Model%20Serving%20with%20PagedAttention/vllm_figure_09_beam_search_example.png)

Figure 9에서 후보 0, 1, 2는 세 block까지 공유하고, 후보 3은 더 일찍 갈라진다. 이후 iteration에서 살아남는 top-4 후보가 후보 1과 2의 후손들로 바뀌면, 더 이상 쓰이지 않는 candidate 0과 3의 block은 refcount를 줄이고 필요하면 해제된다. 그리고 새 후보를 위한 block을 다시 할당한다. 이렇게 하면 서로 다른 beam candidate들이 공통 prefix를 가능한 길게 공유할 수 있고, 예전 시스템처럼 큰 KV cache 조각을 자주 복사할 필요가 없다. copy-on-write가 실제로 일어나는 경우는 “오래된 shared block 내부에 새 토큰을 써야 하는 순간”으로 제한되며, 그때도 한 block만 복사하면 된다.

#### Shared prefix

실제 LLM 서비스에서는 공통 instruction이나 few-shot example을 포함한 긴 system prompt를 여러 요청이 공유하는 경우가 많다. 번역, 포맷 변환, 문체 변환 같은 작업이 대표적이다.

![Figure 10. Shared prompt example for machine translation.](../../assets/images/Efficient%20Memory%20Management%20for%20Large%20Language%20Model%20Serving%20with%20PagedAttention/vllm_figure_10_shared_prompt_example_for_machine_translation.png)

Figure 10은 영어-프랑스어 번역 예시에서 여러 요청이 같은 설명과 예시 문장을 prefix로 공유하는 상황을 보여준다. vLLM에서는 서비스 제공자가 자주 쓰는 shared prefix의 KV cache를 미리 physical block에 올려 둘 수 있다. 이후 실제 사용자 입력이 들어오면, 그 요청의 logical block이 이 prefix용 physical block을 그대로 가리키게 만들면 된다. 마지막 block만 copy-on-write로 표시해 두면, 이후 입력이나 생성이 prefix를 손상시키지 않고 이어질 수 있다. 덕분에 prefix 계산을 매번 다시 하지 않아도 되고, 중복 저장도 피할 수 있다.

#### Mixed decoding methods

논문은 서로 다른 디코딩 방식을 동시에 처리하는 상황도 강조한다. 어떤 요청은 단일 샘플링, 어떤 요청은 병렬 샘플링, 어떤 요청은 beam search를 요구할 수 있다. 기존 시스템은 이런 복합 상황에서 메모리 공유 패턴이 복잡해지면 처리하기 어렵다. 반면 vLLM은 logical-to-physical block mapping이라는 공통 추상화를 두기 때문에, kernel 입장에서는 각 시퀀스별 physical block ID 목록만 보면 된다. 시퀀스들 사이에서 누가 무엇을 공유하는지는 block table 계층이 감춘다. 이것이 서로 다른 sampling preference를 가진 요청도 더 넓게 같이 배치할 수 있게 해 준다.

### 4.5 Scheduling and Preemption

요청률이 시스템 용량을 넘으면, 어떤 요청을 먼저 처리하고 어떤 요청을 잠시 밀어둘지 결정해야 한다. vLLM은 기본 정책으로 **FCFS(first-come-first-serve)**를 택한다. 이는 공정성을 보장하고 starvation을 막기 위한 선택이다. 선점이 필요할 때도 가장 늦게 도착한 요청부터 밀어낸다.

여기서 논문이 강조하는 점은 LLM의 선점 단위가 일반 메모리 시스템과 다르다는 것이다. 하나의 시퀀스를 계속 생성하려면 그 시퀀스의 모든 이전 토큰 상태가 한꺼번에 필요하다. 그래서 vLLM은 블록 단위로 조금씩 evict하는 대신, **한 시퀀스는 전부 내리거나 전부 남겨 두는 all-or-nothing eviction**을 쓴다. 또 beam search처럼 여러 시퀀스가 하나의 요청 그룹을 이루는 경우에는 서로 KV cache를 공유할 수 있으므로, 그 sequence group 전체를 함께 preempt하거나 함께 복구한다.

블록을 내린 뒤 다시 복구하는 방법으로 논문은 두 가지를 검토한다.

첫째는 **swapping**이다. 이는 운영체제의 고전적인 방식과 유사하다. GPU에서 밀려난 block을 CPU RAM으로 복사해 두고, 다시 실행할 때 GPU로 가져온다. 그래서 vLLM에는 GPU block allocator뿐 아니라 CPU block allocator도 있다. 논문은 설계를 통해 CPU 쪽 swap 공간의 필요량이 GPU의 KV cache 메모리 총량을 넘지 않도록 만든다.

둘째는 **recomputation**이다. 이 경우 블록을 저장해 두지 않고, 나중에 다시 필요해졌을 때 KV cache를 새로 계산한다. 흥미로운 점은 이 recomputation이 생각보다 비싸지 않다는 것이다. 이미 생성된 토큰들을 원래 프롬프트 뒤에 이어 붙여 새로운 긴 프롬프트처럼 보고 한 번의 prompt phase로 재계산할 수 있기 때문이다. 즉, 원래는 여러 autoregressive step에 걸쳐 만들었던 KV cache를 다시 만들 때는 더 큰 병렬성을 활용할 수 있다.

어느 쪽이 더 좋은지는 CPU-GPU 대역폭과 GPU 연산 성능에 따라 달라지므로, 논문은 뒤의 ablation에서 둘을 비교한다.

### 4.6 Distributed Execution

대형 LLM은 하나의 GPU에 다 올릴 수 없는 경우가 많다. 그래서 모델 병렬화가 필요하다. 논문은 vLLM이 Megatron-LM 스타일의 **tensor model parallelism**을 지원한다고 설명한다. 이 방식에서는 Transformer의 선형 계층과 attention head가 여러 GPU에 나뉘어 배치되고, 중간 결과는 all-reduce로 동기화된다.

중요한 관찰은 모델이 여러 GPU에 쪼개져 있어도 **모든 shard가 같은 입력 토큰 위치들을 처리한다**는 점이다. 즉, 각 GPU worker는 서로 다른 파라미터 조각을 가지지만, 필요한 KV cache 위치 인덱스는 동일하다. 그래서 vLLM은 중앙 scheduler 안에 **하나의 KV cache manager**를 두고, 모든 worker가 동일한 logical-to-physical block mapping을 공유하게 만든다.

실행 시에는 scheduler가 각 요청의 입력 토큰 ID와 block table을 포함한 control message를 만들어 GPU worker들에게 broadcast한다. worker들은 그 정보를 바탕으로 attention layer에서 필요한 physical block을 읽고, 계산 중에는 scheduler의 개입 없이 all-reduce로 중간 결과를 동기화한다. 마지막에는 이번 iteration의 sampled token을 scheduler에 돌려준다. 이 구조의 장점은 GPU worker가 매 순간 메모리 관리자와 복잡하게 상호작용할 필요 없이, iteration 시작 시점에 필요한 block mapping만 받아 계산하면 된다는 것이다.

## 5. Implementation

vLLM은 FastAPI 기반 프론트엔드와 GPU 기반 inference engine으로 구성된 end-to-end serving 시스템이다. 프론트엔드는 OpenAI API와 유사한 인터페이스를 확장해, 요청마다 최대 시퀀스 길이, beam width 같은 샘플링 파라미터를 설정할 수 있게 한다. 논문에 따르면 vLLM engine은 약 **8.5K lines의 Python 코드**와 **2K lines의 C++/CUDA 코드**로 구현되었다.

모델 실행부는 PyTorch와 Transformers 위에서 GPT, OPT, LLaMA 같은 주요 LLM을 구현하며, 분산 환경의 텐서 통신에는 NCCL을 사용한다. scheduler, block manager 같은 제어 계층은 Python으로 구현하고, PagedAttention처럼 성능이 민감한 부분은 전용 CUDA kernel로 구현한 것이 특징이다.

![Table 1. Model sizes and server configurations.](../../assets/images/Efficient%20Memory%20Management%20for%20Large%20Language%20Model%20Serving%20with%20PagedAttention/vllm_table_01_model_sizes_and_server_configurations.png)

Table 1은 논문이 평가에 사용한 모델 크기와 서버 구성을 정리한다. 13B 모델은 A100 한 장, 66B는 A100 네 장, 175B는 A100-80GB 여덟 장을 사용한다. 파라미터만으로도 각각 26GB, 132GB, 346GB를 차지하며, 남는 GPU 메모리 중 상당량이 KV cache용으로 쓰인다. 13B에서는 KV cache에 12GB를 쓸 수 있어 최대 약 15.7K slot, 66B에서는 21GB로 9.7K slot, 175B에서는 264GB로 60.1K slot을 담을 수 있다. 이 수치는 왜 메모리 관리가 serving 성능의 핵심인지 다시 보여준다.

### 5.1 Kernel-level Optimization

PagedAttention은 block table을 통해 흩어진 block을 읽는 구조이므로, 기존 연속 메모리 기반 attention kernel을 그대로 쓰면 비효율적일 수 있다. 그래서 논문은 세 가지 GPU kernel 최적화를 제시한다.

첫째는 **fused reshape and block write**다. Transformer layer마다 새로 생성된 KV cache를 block 형태로 쪼개고, block read에 적합한 레이아웃으로 reshape한 다음, block table이 가리키는 위치에 써야 한다. 이 과정을 여러 개의 작은 kernel로 나누면 launch overhead가 크므로 하나의 kernel로 합친다.

둘째는 **block read와 attention의 결합**이다. FasterTransformer의 attention kernel을 바탕으로 하되, 연속 메모리 대신 block table을 보고 KV cache를 읽어오도록 수정한다. 논문은 memory access를 coalesced하게 만들기 위해 각 block을 읽을 때 GPU warp를 대응시키고, 요청 배치 안에서 시퀀스 길이가 달라도 처리할 수 있게 지원한다.

셋째는 **fused block copy**다. copy-on-write가 일어나면 여러 discontinuous block을 복사해야 할 수 있다. 이것을 매번 `cudaMemcpyAsync`로 처리하면 아주 작은 복사가 많이 발생해 오버헤드가 커진다. 논문은 여러 block의 복사를 한 번의 kernel launch로 묶어 수행하는 kernel을 구현해 이 문제를 줄인다.

### 5.2 Supporting Various Decoding Algorithms

vLLM은 다양한 디코딩 알고리즘을 세 가지 기본 연산으로 추상화한다. 그것이 **fork**, **append**, **free**다.

- `fork`는 기존 시퀀스에서 새 시퀀스를 만든다.
- `append`는 시퀀스 뒤에 새 토큰을 붙인다.
- `free`는 시퀀스를 삭제한다.

예를 들어 병렬 샘플링에서는 하나의 입력 시퀀스에서 여러 출력 시퀀스를 `fork`로 만들고, 각 iteration마다 `append`를 수행하며, 종료 조건을 만족한 시퀀스는 `free`로 제거한다. beam search나 prefix sharing도 같은 틀 안에서 구현된다. 논문은 이 조합만으로도 미래의 새로운 디코딩 알고리즘 상당수를 지원할 수 있다고 본다.

## 6. Evaluation

논문은 다양한 모델, 데이터셋, 디코딩 시나리오에서 vLLM의 성능을 측정한다. 핵심 평가지표는 **normalized latency**, 즉 각 요청의 end-to-end latency를 출력 토큰 수로 나눈 값이다. request rate를 높여 가면서 latency가 언제 급격히 폭증하는지를 보면, 시스템이 감당할 수 있는 처리량 한계를 비교할 수 있다.

### 6.1 Experimental Setup

평가 모델로는 OPT 13B, 66B, 175B와 LLaMA 13B를 사용한다. 13B와 66B는 당시 LLM leaderboard에서 많이 쓰이던 크기이며, 175B는 GPT-3급 대형 모델을 대표한다. 실험 환경은 Google Cloud Platform의 A2 인스턴스와 NVIDIA A100 GPU다. 구체적인 메모리 구성은 앞의 Table 1에 정리되어 있다.

워크로드는 **ShareGPT**와 **Alpaca** 데이터셋을 바탕으로 합성한다. ShareGPT는 실제 ChatGPT 대화 로그를 기반으로 하며, Alpaca는 self-instruct를 이용해 생성된 instruction-following 데이터셋이다. 논문은 두 데이터셋의 실제 텍스트를 토크나이즈한 뒤, 입력 길이와 출력 길이 분포를 이용해 요청을 합성한다. 또한 timestamp가 없기 때문에 요청 도착 시간은 서로 다른 request rate를 갖는 Poisson process로 생성한다.

![Figure 11. ShareGPT and Alpaca length distributions.](../../assets/images/Efficient%20Memory%20Management%20for%20Large%20Language%20Model%20Serving%20with%20PagedAttention/vllm_figure_11_sharegpt_and_alpaca_length_distributions.png)

Figure 11을 보면 두 데이터셋의 성격 차이가 뚜렷하다. ShareGPT는 입력 평균이 약 **161.31 토큰**, 출력 평균이 약 **337.99 토큰**으로 길고 분산도 크다. 반면 Alpaca는 입력 평균 **19.31 토큰**, 출력 평균 **58.45 토큰**으로 훨씬 짧다. 이 차이는 뒤의 실험 결과에서 “긴 시퀀스일수록 vLLM의 이점이 더 커지는 이유”를 이해하는 데 중요하다.

비교 대상은 두 종류다. 첫 번째는 **FasterTransformer**다. FasterTransformer 자체는 scheduler가 없으므로, 논문은 Triton 같은 기존 시스템을 닮은 동적 배칭 scheduler를 직접 얹어 비교한다. 각 실험에서는 GPU 메모리 용량이 허용하는 최대 batch size $B$를 정하고, 가장 먼저 도착한 요청부터 최대 $B$개까지 묶어 처리한다.

두 번째는 **Orca**다. Orca는 처리량 중심으로 최적화된 당시의 대표적인 LLM serving 시스템이지만 공개 구현이 없어서 논문 저자들이 직접 재구현했다. 메모리 할당은 buddy allocator를 쓴다고 가정하고, 출력 길이를 얼마나 과대예약하는지에 따라 세 버전을 둔다. `Orca (Oracle)`은 실제 출력 길이를 미리 안다고 가정한 비현실적 상한선, `Orca (Pow2)`는 실제 길이의 최대 2배까지만 예약하는 버전, `Orca (Max)`는 항상 모델 최대 길이까지 예약하는 버전이다.

평가는 대부분 1시간 분량의 trace로 수행하되, 비용 때문에 OPT-175B만 15분 trace를 쓴다.

### 6.2 Basic Sampling

기본 실험은 요청당 하나의 출력만 생성하는 **basic sampling**이다. 논문은 세 가지 OPT 모델과 두 데이터셋에 대해 request rate를 올려 가며 latency 변화를 본다.

![Figure 12. Single-sequence generation on ShareGPT and Alpaca.](../../assets/images/Efficient%20Memory%20Management%20for%20Large%20Language%20Model%20Serving%20with%20PagedAttention/vllm_figure_12_single_sequence_generation_sharegpt_alpaca.png)

Figure 12의 곡선은 공통적으로, request rate가 낮을 때는 latency가 완만히 증가하다가 어느 임계점을 넘으면 갑자기 폭증하는 모습을 보인다. 이는 시스템이 처리 가능한 용량을 넘는 순간 queue가 계속 쌓이기 때문이다. 따라서 각 시스템이 어디까지 낮은 normalized latency를 유지하는지가 곧 처리량 한계다.

![Figure 13. Average batched requests for OPT-13B.](../../assets/images/Efficient%20Memory%20Management%20for%20Large%20Language%20Model%20Serving%20with%20PagedAttention/vllm_figure_13_average_batched_requests_opt13b.png)

ShareGPT에서 vLLM은 Orca (Oracle)보다 **1.7배에서 2.7배**, Orca (Max)보다 **2.7배에서 8배** 높은 request rate까지 비슷한 latency를 유지한다. 논문은 그 이유를 PagedAttention이 메모리 낭비를 줄여 더 많은 요청을 동시에 배치할 수 있기 때문이라고 설명한다. Figure 13(a)의 OPT-13B 예시에서는 vLLM이 같은 시점에 평균 **30.42개 요청**을 함께 처리하는 반면, Orca (Oracle)은 **13.62개**, Orca (Max)는 **7.00개** 수준이다. 즉, vLLM은 OPT-13B ShareGPT 조건에서 Orca (Oracle)보다 약 **2.2배**, Orca (Max)보다 약 **4.3배** 많은 요청을 한 번에 담는다.

FasterTransformer와 비교하면 차이는 더 크다. FasterTransformer는 fine-grained scheduling이 없고 메모리 관리 역시 Orca (Max)에 가까운 비효율을 보이므로, vLLM은 최대 **22배** 높은 request rate를 버틴다.

Alpaca에서도 전반적 경향은 비슷하다. 다만 OPT-175B 환경에서는 vLLM의 이점이 Orca (Oracle), Orca (Pow2) 대비 덜 두드러진다. 이유는 두 가지다. 첫째, 175B를 위한 서버 구성은 KV cache에 쓸 수 있는 GPU 메모리가 매우 크다. 둘째, Alpaca는 시퀀스가 짧다. 그래서 이 경우에는 기존 시스템도 상당수 요청을 이미 batch에 담을 수 있고, 병목이 메모리가 아니라 계산으로 이동한다. 논문은 이것을 **compute-bound regime**라고 설명한다.

### 6.3 Parallel Sampling and Beam Search

논문은 이어서 PagedAttention의 block sharing이 병렬 샘플링과 beam search에서 얼마나 효과적인지 측정한다.

![Figure 14. Parallel generation and beam search with OPT-13B on Alpaca.](../../assets/images/Efficient%20Memory%20Management%20for%20Large%20Language%20Model%20Serving%20with%20PagedAttention/vllm_figure_14_parallel_generation_and_beam_search_opt13b_alpaca.png)

Figure 14의 첫 번째 행은 병렬 샘플링 결과다. 하나의 프롬프트에서 출력 시퀀스를 더 많이 샘플링할수록, 공유 가능한 프롬프트 KV cache의 비중이 커지므로 vLLM의 상대적 이점도 커진다. 두 번째 행은 beam search 결과다. beam search는 프롬프트뿐 아니라 생성 중간의 긴 prefix까지 후보들 사이에서 공유할 수 있으므로, 병렬 샘플링보다도 효과가 더 크게 나타난다.

![Figure 15. Memory saving from shared KV blocks.](../../assets/images/Efficient%20Memory%20Management%20for%20Large%20Language%20Model%20Serving%20with%20PagedAttention/vllm_figure_15_memory_saving_from_shared_kv_blocks.png)

논문은 OPT-13B Alpaca 조건에서, Orca (Oracle) 대비 vLLM의 개선폭이 basic sampling에서는 약 **1.3배**였지만 beam width가 6인 beam search에서는 약 **2.3배**까지 커진다고 보고한다. 즉, PagedAttention의 장점은 단순히 fragmentation을 줄이는 데만 있지 않고, **공유 가능한 KV cache를 block 단위로 실제 공유할 수 있게 만든다**는 데 있다.

Figure 15는 이 메모리 절약량을 수치로 보여준다. Alpaca에서 병렬 샘플링은 **6.1%에서 9.8%**, beam search는 **37.6%에서 55.2%**의 메모리 절약을 보인다. ShareGPT에서는 시퀀스가 더 길기 때문에 효과가 더 커져, 병렬 샘플링에서 **16.2%에서 30.5%**, beam search에서 **44.3%에서 66.3%**의 절감이 나타난다. 결국 prefix가 길고 공유 구간이 길수록 vLLM의 강점이 커진다.

### 6.4 Shared Prefix

공통 instruction과 few-shot example을 많은 요청이 같이 쓰는 상황을 따로 측정한 것이 shared prefix 실험이다.

![Figure 16. Shared-prefix translation workload.](../../assets/images/Efficient%20Memory%20Management%20for%20Large%20Language%20Model%20Serving%20with%20PagedAttention/vllm_figure_16_shared_prefix_translation_workload.png)

여기서는 LLaMA-13B를 사용하고, 영어-독일어 번역 작업에 대해 WMT16 데이터셋을 이용한다. 프롬프트는 두 가지로 만든다. 하나는 번역 예시 1개가 포함된 one-shot prefix이고, 다른 하나는 예시 5개가 포함된 few-shot prefix다. prefix 길이는 각각 약 80토큰, 341토큰이다.

결과적으로 one-shot prefix가 공유되는 경우 vLLM은 Orca (Oracle)보다 **1.67배** 높은 처리량을 보이고, five-shot prefix가 공유되는 경우에는 **3.58배**까지 올라간다. 공유되는 prefix가 길수록 중복 계산과 중복 저장을 더 많이 없앨 수 있기 때문이다.

### 6.5 Chatbot

챗봇은 LLM의 대표적인 응용이다. 논문은 ShareGPT를 바탕으로 대화 이력과 마지막 사용자 질의를 합쳐 프롬프트를 만든다. OPT-13B의 context length 제한 때문에 프롬프트는 마지막 **1024토큰**만 남기고 자르며, 출력도 최대 **1024토큰**까지만 생성하게 한다.

![Figure 17. Chatbot workload performance.](../../assets/images/Efficient%20Memory%20Management%20for%20Large%20Language%20Model%20Serving%20with%20PagedAttention/vllm_figure_17_chatbot_workload_performance.png)

이 설정에서 논문은 서로 다른 대화 라운드 사이의 KV cache를 유지하지 않는다. 그 이유는 대화가 끊겨 있는 동안 그 KV cache가 다른 요청이 쓸 수 있는 공간을 계속 점유하게 되기 때문이다.

Figure 17의 결과에서 vLLM은 세 Orca baseline보다 약 **2배** 높은 request rate까지 버틴다. ShareGPT 대화는 긴 경우가 많기 때문에, 많은 요청의 입력 프롬프트가 이미 1024토큰에 가깝다. buddy allocator 기반 Orca 계열은 이런 경우 출력 공간도 크게 예약해 버려 세 버전이 비슷하게 나빠진다. 반면 vLLM은 긴 프롬프트 자체로 인한 fragmentation과 reservation 문제를 해결하기 때문에 훨씬 안정적으로 긴 대화 요청을 처리할 수 있다.

## 7. Ablation Studies

논문은 vLLM의 설계 선택들이 실제로 어떤 trade-off를 가지는지 ablation으로 분석한다. 주요 질문은 세 가지다. 첫째, block table을 통한 간접 참조가 attention kernel 자체를 얼마나 느리게 만드는가. 둘째, block size는 어느 정도가 좋은가. 셋째, 메모리가 부족할 때 swapping과 recomputation 중 어느 쪽이 더 유리한가.

### 7.1 Kernel Microbenchmark

![Figure 18. Ablation experiments.](../../assets/images/Efficient%20Memory%20Management%20for%20Large%20Language%20Model%20Serving%20with%20PagedAttention/vllm_figure_18_ablation_experiments.png)

PagedAttention은 block table 접근, 분기, 가변 길이 처리 때문에 기존의 고도로 최적화된 연속 메모리 attention kernel보다 손해를 볼 수 있다. 논문은 Figure 18(a)를 통해 실제 attention kernel latency를 비교하고, vLLM의 PagedAttention kernel이 FasterTransformer보다 **20%에서 26%** 정도 느리다고 보고한다.

하지만 논문은 이 차이를 “attention operator 자체에만 생기는 국소적 오버헤드”로 해석한다. Linear 같은 다른 연산들은 영향을 거의 받지 않고, 무엇보다 시스템 전체 성능에서는 메모리 활용 개선으로 얻는 이득이 훨씬 더 크기 때문에 end-to-end 결과에서는 vLLM이 FasterTransformer를 크게 앞선다. 즉, kernel 하나만 보면 조금 느릴 수 있지만, **더 많은 요청을 동시에 담을 수 있게 된 효과가 전체 처리량에서 압도적**이라는 뜻이다.

### 7.2 Impact of Block Size

Figure 18(b)는 block size가 end-to-end latency에 어떤 영향을 주는지 보여준다. 논문이 지적하는 핵심 trade-off는 분명하다.

- block이 너무 작으면 KV cache를 읽고 처리할 때 GPU 병렬성을 충분히 활용하기 어렵다.
- block이 너무 크면 마지막 partially-filled block이 커져 내부 단편화가 늘고, 공유 세분성도 떨어진다.

ShareGPT trace에서는 block size **16에서 128** 사이가 가장 좋은 성능을 보인다. 시퀀스가 길기 때문에 block이 다소 커도 크게 손해 보지 않는 것이다. 반면 Alpaca trace는 시퀀스가 짧아서 block size가 커질수록 금방 비효율적이 된다. 이 경우에는 **16이나 32**가 좋고, 더 큰 block은 성능을 크게 악화시킨다.

논문은 실무적으로 **block size 16**이 가장 무난하다고 결론 내린다. GPU 병렬성을 충분히 활용할 만큼은 크고, 대부분의 워크로드에서 심각한 내부 단편화를 만들 만큼 크지는 않다는 판단이다. 그래서 vLLM의 기본값도 16으로 설정된다.

### 7.3 Comparing Recomputation and Swapping

메모리 압박이 있을 때 preempted sequence를 되살리는 두 방법, recomputation과 swapping을 비교한 결과가 Figure 19다.

![Figure 19. Recomputation versus swapping.](../../assets/images/Efficient%20Memory%20Management%20for%20Large%20Language%20Model%20Serving%20with%20PagedAttention/vllm_figure_19_recomputation_vs_swapping.png)

Figure 19(a)의 microbenchmark를 보면, **작은 block size에서는 swapping의 비용이 매우 커진다.** CPU와 GPU 사이를 오가는 데이터가 작은 block으로 잘게 나뉘면, PCIe 대역폭을 충분히 활용하지 못하고 많은 작은 전송이 발생하기 때문이다. 반대로 recomputation은 KV block 자체를 이동시키지 않으므로 block size가 작아져도 오버헤드가 거의 일정하다.

큰 block size로 가면 swapping의 상대적 비효율이 줄어들 수 있다. 하지만 논문은 recomputation의 지연이 swapping보다 커지는 경우가 없고, 최악이어도 **swapping latency의 20% 이내**라고 보고한다. 또한 end-to-end 성능 기준으로 보면 block size **16에서 64** 정도에서는 recomputation과 swapping이 비슷한 성능을 낸다. 따라서 어떤 방식을 쓸지는 하드웨어 환경과 block size에 따라 달라질 수 있지만, 작은 block 중심의 설계에서는 recomputation이 특히 매력적이다.

## 8. Discussion

논문은 vLLM의 아이디어가 모든 GPU 워크로드에 그대로 적용되는 것은 아니라고 분명히 말한다. 가상 메모리와 페이징이 잘 맞는 경우는 크게 두 조건이 있을 때다. 하나는 **동적 메모리 할당이 필수적인 경우**, 다른 하나는 **성능이 계산보다 메모리 수용량에 의해 제한되는 경우**다.

예를 들어 일반적인 DNN 학습은 텐서 shape가 대부분 정적이어서, 메모리 배치를 사전에 최적화하기 쉽다. 또 LLM이 아닌 일반 DNN 추론에서는 메모리 효율이 조금 좋아져도 전체 성능이 계산 병목이라 체감 이득이 없을 수 있다. 그런 환경에 vLLM식 간접 참조와 비연속 block 메모리를 억지로 도입하면, 오히려 indirection 오버헤드 때문에 느려질 수도 있다.

반대로 LLM serving은 동적 길이, 큰 token state, online scheduling이라는 조건이 동시에 있기 때문에 페이징이 특히 잘 맞는다. 논문은 이런 점에서 vLLM이 운영체제 아이디어를 “그대로 가져온 것”이 아니라, LLM serving이라는 응용의 의미론에 맞춰 재해석했다고 본다. all-or-nothing swap-out 정책, recomputation을 통한 복구, memory access와 attention의 kernel fusion이 그런 예다.

## 9. Related Work

논문은 관련 연구를 세 갈래로 정리한다.

첫째는 **일반적인 모델 서빙 시스템**이다. Clipper, TensorFlow Serving, Nexus, InferLine, Clockwork 같은 시스템들은 batching, caching, placement, scheduling을 연구했다. 하지만 이들은 autoregressive LLM 추론의 핵심인 “토큰 상태가 시간이 지나며 계속 늘어나는 구조”를 중심 문제로 다루지 않았기 때문에, KV cache 최적화 관점에서는 한계가 있다.

둘째는 **Transformer 전용 serving 시스템**이다. 이 계열은 GPU kernel 최적화, 고급 batching, model parallelism, parameter sharing 같은 기법을 사용한다. 논문은 특히 Orca를 가장 직접적인 비교 대상으로 본다.

논문이 보는 vLLM과 Orca의 관계는 경쟁이라기보다 **상보적**이다. Orca는 request interleaving과 fine-grained scheduling을 통해 GPU 활용률을 높인다. vLLM은 메모리 낭비를 줄이고 공유를 가능하게 해 더 많은 요청의 working set을 메모리에 담게 만든다. 즉, Orca가 “언제 무엇을 함께 계산할 것인가”의 문제를 잘 푼다면, vLLM은 “그 계산에 필요한 상태를 메모리에 얼마나 많이, 얼마나 효율적으로 올려 둘 것인가”의 문제를 푼다. 논문은 미세한 scheduling이 있을수록 오히려 메모리 관리의 중요성이 더 커진다고 본다.

셋째는 **메모리 최적화 연구**다. 학습 과정에서는 swapping, recomputation, 혹은 둘의 결합이 peak memory를 줄이는 데 오래 전부터 쓰여 왔다. FlexGen은 제한된 GPU 메모리에서 LLM 추론을 수행하기 위해 가중치와 token state를 swap하는 방법을 다루지만, online serving 상황을 겨냥하지는 않는다. OLLA는 텐서의 위치와 생애를 조정해 fragmentation을 줄이지만, block-level online serving이나 prefix sharing까지 다루지는 않는다. FlashAttention은 attention 계산의 메모리 사용량과 I/O를 줄이지만, 이 논문처럼 KV cache 자체를 block 단위로 관리하며 온라인 서빙을 최적화하지는 않는다.

## 10. Conclusion

이 논문은 key/value를 비연속 paged memory에 저장할 수 있게 하는 **PagedAttention**을 제안하고, 그 위에 고성능 LLM serving 시스템 **vLLM**을 구현했다. 핵심은 운영체제의 가상 메모리, 페이징, copy-on-write 같은 오래된 아이디어를 LLM serving의 KV cache 문제에 맞게 다시 적용했다는 점이다.

논문의 논지는 분명하다. LLM serving의 병목은 단순히 attention 연산이 느리기 때문이 아니라, **동적으로 커지고 줄어드는 거대한 KV cache를 얼마나 잘 관리하느냐**에 달려 있다. PagedAttention은 attention 계산이 block table을 따라 비연속 메모리를 정확히 읽도록 만들고, vLLM은 이를 기반으로 near-zero waste, prefix sharing, copy-on-write, preemption, distributed execution을 결합한다. 그 결과 기존 최첨단 시스템 대비 2배에서 4배 수준의 처리량 향상을 실험으로 보인다.

## Acknowledgement

논문은 Xiaoxuan Liu, Zhifeng Chen, Yanping Huang, SOSP 리뷰어들, 그리고 shepherd인 Lidong Zhou에게 감사의 뜻을 전한다. 또한 Andreessen Horowitz, Anyscale, Astronomer, Google, IBM, Intel, Lacework, Microsoft, Mohamed Bin Zayed University of Artificial Intelligence, Samsung SDS, Uber, VMware 등의 지원을 받았다고 밝힌다.

## 추가 설명

이 논문을 이해할 때 가장 중요한 포인트는 **vLLM이 attention 수학 자체를 바꾼 논문이 아니라, attention이 필요한 KV cache를 저장하고 읽는 방식을 바꾼 논문**이라는 점이다. 모델이 생성하는 확률분포나 attention의 의미는 그대로 유지된다. 달라지는 것은 key/value가 GPU 메모리에 놓이는 방식, 그리고 그 메모리를 얼마나 유연하게 공유할 수 있느냐이다.

기존 시스템의 문제를 더 직관적으로 말하면, “미래에 얼마나 길어질지 모르는 출력 때문에 너무 큰 창고를 먼저 통째로 빌려 놓는 방식”에 가깝다. 요청 하나가 짧게 끝나면 빈 공간이 많이 남고, 긴 요청과 짧은 요청이 섞이면 중간중간 쓸모없는 틈이 생긴다. vLLM은 이것을 “작은 상자(block)를 필요할 때마다 붙여 나가는 방식”으로 바꾼다. 그러면 마지막 상자의 일부만 비는 정도만 감수하면 되고, 전체 공간 활용률이 크게 올라간다.

또 하나 중요한 포인트는 **KV cache sharing의 실질적 가치**다. 병렬 샘플링, beam search, few-shot prefix 같은 상황에서는 여러 시퀀스가 긴 prefix를 공유한다. 기존 방식에서는 이 prefix를 시퀀스 수만큼 중복 저장해야 했다. 반면 vLLM은 같은 physical block을 여러 logical block이 가리키게 만들고, 진짜로 달라지는 순간에만 copy-on-write를 적용한다. 그래서 prefix가 길수록, 동시에 여러 후보를 유지할수록 이득이 커진다. 논문에서 beam search나 shared prefix 실험의 개선폭이 basic sampling보다 더 큰 이유가 바로 여기에 있다.

PagedAttention이 운영체제와 닮았다는 비유도 그냥 비유 차원에서 끝나지 않는다. 논리 주소와 물리 주소를 분리하고, 공통 페이지를 여러 프로세스가 공유하며, 수정 시에만 복사하는 copy-on-write를 쓴다는 점이 실제로 거의 동일하다. 다만 vLLM은 여기에 LLM serving 특유의 의미론을 더한다. 예를 들어 sequence 전체 상태가 있어야 생성이 가능하므로 all-or-nothing preemption을 쓰고, CPU 메모리에서 복사해 오는 대신 프롬프트 형태로 다시 계산하는 recomputation도 활용한다. 즉, 운영체제의 추상화를 가져오되 LLM에 맞게 다시 설계한 것이다.

마지막으로, 논문 결과를 해석할 때는 “항상 2배에서 4배 빨라진다”라고 단순화하면 안 된다. 개선폭은 **시퀀스가 길수록**, **모델이 클수록**, **메모리 병목이 강할수록**, **공유 가능한 prefix가 많을수록** 커진다. 반대로 시퀀스가 짧고 GPU 메모리가 매우 넉넉해서 이미 compute-bound라면, vLLM의 장점은 상대적으로 작아진다. 논문에서 OPT-175B + Alpaca 조합이 그 예다. 따라서 vLLM의 핵심 가치는 “메모리가 병목인 실제 온라인 LLM serving 조건”에서 가장 분명하게 드러난다.
