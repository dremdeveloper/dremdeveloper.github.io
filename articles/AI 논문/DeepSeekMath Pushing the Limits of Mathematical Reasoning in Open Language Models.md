# DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y.K. Li, Y. Wu, Daya Guo  
DeepSeek-AI, Tsinghua University, Peking University  
GitHub: https://github.com/deepseek-ai/DeepSeek-Math

## Abstract

수학적 추론은 언어 모델에게 특히 어려운 과제다. 문제를 올바르게 풀기 위해서는 단순한 언어적 패턴 완성만으로는 부족하고, 중간 추론 단계의 일관성, 계산의 정확성, 기호 조작 능력, 그리고 정답 검증 가능성이 동시에 요구되기 때문이다. 이 논문은 이러한 문제의식 위에서 DeepSeekMath 7B를 제안한다. 모델은 DeepSeek-Coder-Base-v1.5 7B를 초기점으로 삼아, Common Crawl에서 수집한 120B 규모의 수학 관련 토큰과 자연어 및 코드 데이터를 함께 사용해 추가 사전학습을 수행한다. 그 결과 DeepSeekMath 7B는 외부 도구나 voting 기법 없이 경쟁 수준의 MATH 벤치마크에서 51.7%를 달성하며, Gemini Ultra와 GPT-4에 근접하는 성능을 보인다. 또한 64개 샘플에 대한 self-consistency를 적용하면 MATH에서 60.9%까지 올라간다.

논문은 이러한 성능 향상의 핵심 원인을 두 가지로 정리한다. 첫째는 공개 웹 데이터에서 수학적으로 유의미한 문서를 정밀하게 추출하는 데이터 선택 파이프라인이다. 단순히 수학 데이터셋을 모으는 수준이 아니라, Common Crawl 전역에서 수학 관련 웹페이지를 반복적으로 발굴하고, benchmark contamination을 제거하며, 다국어 수학 데이터까지 포괄하는 대규모 수학 코퍼스를 구축한다. 둘째는 PPO를 대체하는 강화학습 알고리즘인 GRPO(Group Relative Policy Optimization)이다. GRPO는 PPO처럼 별도의 value model을 요구하지 않으면서도 수학적 추론 성능을 효과적으로 개선하며, 특히 메모리 사용량 측면에서 더 효율적이다.

Figure 1은 외부 툴과 voting 없이 경쟁 수준 MATH 벤치마크에서 오픈소스 모델들이 어느 수준까지 도달했는지를 비교한다. DeepSeekMath 계열이 기존 오픈소스 모델을 뚜렷하게 앞서며, 논문의 전체 목표가 “오픈소스 7B 모델로 수학 추론의 상한을 어디까지 끌어올릴 수 있는가”에 놓여 있음을 보여준다.

![Figure 1](/assets/images/DeepSeekMath%20Pushing%20the%20Limits%20of%20Mathematical%20Reasoning%20in%20Open%20Language%20Models/DeepSeekMath_Pushing_the_Limits_of_Mathematical_Reasoning_in_Open_Language_Models_01_Figure_1.png)

## 1. Introduction

대규모 언어 모델은 자연어 생성과 일반적 추론에서는 눈에 띄는 진전을 이루었지만, 수학적 추론은 여전히 가장 까다로운 영역으로 남아 있다. 수학 문제는 문제 해석, 조건 정리, 중간 식 변형, 경우 분기, 계산 정확성, 최종 검증이라는 여러 단계를 요구한다. 따라서 언어적 유창함만으로는 높은 성능을 얻기 어렵고, 학습 데이터의 질과 구조, 그리고 정답 신호를 어떻게 학습에 반영하느냐가 특히 중요해진다.

이 논문은 DeepSeekMath 프로젝트를 통해 그 문제를 정면으로 다룬다. 저자들은 우선 수학 분야에서 의미 있는 추가 사전학습이 가능하려면 웹 전체에서 고품질 수학 데이터를 대규모로 추출해야 한다고 본다. 기존 수학 코퍼스는 규모가 작거나, 영어 중심이거나, arXiv 위주로 편중되어 있었고, 실제 수학 벤치마크 성능 향상과 얼마나 연결되는지도 충분히 검증되지 않았다. 이에 따라 논문은 OpenWebMath를 시드로 삼되, Common Crawl 전역에서 fastText 기반 분류기와 반복적 확장 절차를 이용해 수학 웹페이지를 재귀적으로 발굴하는 방식을 제안한다. 그 결과 35.5M개의 수학 웹페이지, 120B 토큰 규모의 DeepSeekMath Corpus를 구축한다.

다음으로 저자들은 베이스 모델의 출발점을 일반 언어 모델이 아니라 코드 특화 모델인 DeepSeek-Coder-Base-v1.5 7B로 선택한다. 논문 전반의 실험은 이 선택이 우연이 아님을 보여준다. 코드 학습은 프로그램 기반 도구 사용을 동반한 수학 추론에 직접적인 도움을 줄 뿐 아니라, 도구를 쓰지 않는 순수한 chain-of-thought 수학 추론에서도 이점을 가져온다. 실제로 DeepSeekMath-Base 7B는 GSM8K에서 64.2%, MATH에서 36.2%를 달성하며, 이는 Minerva 540B와 같은 훨씬 큰 폐쇄형 수학 베이스 모델도 넘어서는 결과다.

여기서 끝나지 않고, 저자들은 수학 instruction tuning과 강화학습까지 연결한다. 수학 문제를 CoT(chain-of-thought), PoT(program-of-thought), tool-integrated reasoning 형식으로 정리한 776K instruction 데이터로 DeepSeekMath-Instruct 7B를 만든 뒤, 다시 제한된 범위의 수학 질문만 사용해 RL을 적용한다. 그럼에도 RL 이후의 DeepSeekMath-RL 7B는 GSM8K 88.2%, MATH 51.7%, CMATH 88.8%로 올라가며, 대다수 오픈소스 모델과 많은 폐쇄형 모델을 뛰어넘는다.

논문이 서론에서 강조하는 핵심 메시지는 분명하다. 고품질 수학 데이터의 확보, 코드 기반 초기화, 수학 특화 instruction tuning, 그리고 효율적인 RL 알고리즘이 결합되면, 상대적으로 작은 7B 규모 오픈소스 모델도 수학적 추론에서 매우 높은 성능에 도달할 수 있다는 것이다.

### 1.1. Contributions

논문의 기여는 다음과 같은 흐름으로 정리된다.

첫째, Common Crawl에는 수학적 학습에 실제로 유용한 대규모 웹 데이터가 존재하며, 이를 잘 선별하면 기존 공개 수학 코퍼스보다 더 강력한 성능 향상을 얻을 수 있음을 실험적으로 보였다. DeepSeekMath Corpus는 120B 토큰 규모로, Minerva나 OpenWebMath에서 활용한 수학 웹 데이터보다 훨씬 크고, 실제 벤치마크 성능에서도 가장 강한 효과를 나타낸다.

둘째, DeepSeekMath-Base 7B는 수학 텍스트에 대한 지속학습을 통해 GSM8K, MATH, CMATH, Gaokao 계열 문제, formal mathematics까지 폭넓게 향상된다. 특히 MATH에서 36.2%, GSM8K에서 64.2%라는 수치는 7B 모델로서는 매우 강력하며, 더 큰 규모의 선행 모델과도 경쟁한다.

셋째, 코드 학습이 수학 추론에 도움을 준다는 가설을 단순 인상비평이 아니라 체계적 ablation으로 검증했다. 코드 학습은 툴 사용 기반의 수학 추론을 강화하며, 상황에 따라 도구 없는 추론과 일반 reasoning에도 긍정적인 영향을 준다.

넷째, 반대로 arXiv 논문만으로 구성된 수학 데이터는 기대만큼 효과적이지 않다는 점을 보였다. 저자들은 수학 사전학습에서 arXiv가 거의 관습처럼 포함되어 왔음에도, 실제 수학 벤치마크에서는 별다른 이득이 없거나 오히려 성능 저하가 나타날 수 있음을 제시한다.

다섯째, PPO를 대체하는 GRPO를 제안했다. 이 방법은 PPO의 클리핑 기반 정책 최적화 구조를 유지하면서도 value function을 제거하고, 동일 질문에 대해 생성한 여러 출력의 상대적 보상으로 advantage를 계산한다. 그 결과 메모리와 계산량을 절약하면서도 수학 RL에서 높은 효과를 얻는다.

여섯째, RL이 단순히 학습한 문제 분포 안에서만 이득을 주는 것이 아니라, 제한된 RL 데이터만 사용했는데도 out-of-domain 평가까지 광범위하게 개선된다는 점을 보였다. 이는 RL이 특정 문제의 암기가 아니라 출력 분포 전체를 더 좋은 방향으로 재조정했음을 시사한다.

### 1.2. Summary of Evaluations and Metrics

논문은 평가를 크게 세 부류로 나눈다.

첫 번째는 자연어 기반 수학 문제 풀이 능력이다. 영어권에서는 GSM8K, MATH, SAT, OCW Courses, MMLU-STEM을 사용하고, 중국어권에서는 MGSM-zh, CMATH, Gaokao-MathCloze, Gaokao-MathQA를 사용한다. 이들 벤치마크는 초등 수준 산술부터 올림피아드 수준 수학 문제까지 난이도와 형식을 다양하게 포괄한다.

두 번째는 툴 사용 기반 수학 추론과 formal mathematics이다. 툴 사용 평가에서는 few-shot program-of-thought prompting을 적용해 모델이 Python 프로그램을 생성하고, math나 sympy 같은 라이브러리를 활용하게 한다. formal mathematics 쪽에서는 miniF2F에서 informal-to-formal proving을 Isabelle 환경으로 평가한다.

세 번째는 일반적인 언어 이해, 추론, 코드 생성 능력이다. MMLU, BBH, HumanEval, MBPP를 통해 수학 특화 학습이 일반 역량을 손상시키는지, 혹은 오히려 일부 일반 추론에도 이득을 주는지를 확인한다.

이 구성은 논문의 의도를 잘 드러낸다. DeepSeekMath는 단순히 수학 벤치마크 숫자 하나를 올리는 프로젝트가 아니라, 수학 데이터와 수학 RL이 언어 모델 전체 역량에 어떤 구조적 영향을 주는지를 함께 분석하는 프로젝트다.

## 2. Math Pre-Training

### 2.1. Data Collection and Decontamination

DeepSeekMath Corpus 구축의 출발점은 OpenWebMath다. OpenWebMath는 Common Crawl에서 수학 관련 페이지를 추출한 공개 코퍼스인데, 저자들은 이것을 곧바로 최종 코퍼스로 쓰지 않는다. 대신 OpenWebMath를 “양성 예시가 충분히 있는 시드 집합”으로 보고, 이를 기반으로 더 넓고 더 다양한 수학 웹페이지를 반복적으로 찾는 파이프라인을 설계한다.

핵심 아이디어는 fastText 분류기를 이용해 수학 관련 여부를 빠르게 점수화한 뒤, 그 결과를 다시 새로운 양성 예시 수집에 활용하는 것이다. 첫 번째 단계에서 저자들은 OpenWebMath에서 500K개의 positive examples를 뽑고, Common Crawl에서 무작위로 500K개의 negative examples를 뽑아 fastText 분류기를 학습한다. 이때 사용한 설정은 vector dimension 256, learning rate 0.1, max word n-gram 3, minimum word occurrences 3, epochs 3이다. Common Crawl 전체는 먼저 중복 제거를 거쳐 40B HTML 페이지 수준으로 줄인다. 그 다음 fastText 점수로 페이지를 정렬하고 상위권 페이지를 수집한다.

하지만 저자들은 첫 번째 fastText만으로는 수학 웹의 다양성을 충분히 포착하기 어렵다고 본다. 특히 OpenWebMath에 없는 유형의 사이트나 수학 하위 분야는 첫 번째 분류기에서 놓칠 수 있다. 그래서 각 수집 라운드 이후, Common Crawl을 base URL 기준의 domain 단위로 묶은 다음, 첫 번째 수집에서 10% 이상 페이지가 선택된 domain을 수학 관련 domain으로 간주한다. 예를 들어 mathoverflow.net 같은 사이트가 여기에 해당한다. 그 후 이러한 도메인 안에서 실제로 수학 콘텐츠가 있는 URL path를 사람이 수동으로 표시하고, 그 경로에 연결된 아직 미수집 페이지를 새로운 시드 집합에 넣는다. 이렇게 확장된 시드로 다시 fastText를 학습하고, 다시 Common Crawl 전체에 적용한다.

이 과정을 4번 반복한 결과 총 35.5M개의 수학 웹페이지, 120B 토큰을 확보한다. 특히 4번째 반복에서 새로 얻은 데이터의 거의 98%가 이미 3번째 반복에서 포착된 내용과 겹쳤기 때문에, 저자들은 그 시점에서 수집을 중단한다. 즉 단순히 “많이 모았다”가 아니라, 추가 수집의 한계수익이 거의 사라질 때까지 반복을 진행한 것이다.

또 하나의 핵심은 contamination 제거다. 수학 벤치마크는 웹에서 그대로 노출돼 있을 가능성이 높기 때문에, 사전학습 코퍼스 안에 benchmark 질문이나 정답이 섞이면 성능 해석이 왜곡된다. 저자들은 DeepSeek-Coder 논문의 방식에 따라 benchmark 텍스트와 정확히 일치하는 n-gram을 제거한다. 구체적으로, 영어 수학 벤치마크(GSM8K, MATH 등)와 중국어 벤치마크(CMATH, AGIEval 등)의 질문이나 답과 10-gram exact match가 되는 텍스트 구간을 모두 제거한다. benchmark 텍스트가 10-gram보다 짧지만 최소 3-gram 이상이면, 그 짧은 문자열에 대해서도 exact match를 수행해 오염 가능성을 제거한다.

Figure 2는 이 전체 절차를 반복 파이프라인으로 도식화한다. OpenWebMath 같은 초기 수학 웹 코퍼스를 출발점으로 삼고, Common Crawl에서 positive/negative를 구성해 fastText 분류기를 학습한 뒤, Common Crawl 전체에 적용하고, 선택된 페이지와 도메인 정보를 다시 이용해 다음 라운드의 시드를 확장하는 흐름이다. 핵심은 “분류기 → 수집 → 도메인 분석 → 시드 확장 → 재학습”의 순환 구조이며, 이것이 단회성 필터링보다 훨씬 넓은 수학 웹 공간을 포착하게 만든다.

![Figure 2](/assets/images/DeepSeekMath%20Pushing%20the%20Limits%20of%20Mathematical%20Reasoning%20in%20Open%20Language%20Models/DeepSeekMath_Pushing_the_Limits_of_Mathematical_Reasoning_in_Open_Language_Models_02_Figure_2.png)

### 2.2. Validating the Quality of the DeepSeekMath Corpus

DeepSeekMath Corpus가 실제로 더 좋은 수학 코퍼스인지 검증하기 위해, 논문은 이를 최근 공개된 수학 코퍼스들과 직접 비교한다. 비교 대상은 세 가지다. 첫째, MathPile은 8.9B 토큰 규모로 교과서, Wikipedia, ProofWiki, CommonCrawl, StackExchange, arXiv 등을 섞은 코퍼스이며, 85% 이상이 arXiv에서 온다. 둘째, OpenWebMath는 13.6B 토큰 규모의 수학 웹 코퍼스다. 셋째, Proof-Pile-2는 OpenWebMath, AlgebraicStack의 수학 코드, arXiv를 섞은 51.9B 토큰 규모 코퍼스이며, 실험에서는 arXiv:Web:Code 비율을 2:4:1로 사용한다.

#### 2.2.1. Training Setting

비교 실험에서는 DeepSeek-LLM 계열 구조를 따르는 1.3B 파라미터 일반 사전학습 모델을 사용한다. 각 수학 코퍼스에 대해 별도로 150B 토큰씩 학습하며, HAI-LLM 프레임워크 위에서 AdamW를 사용한다. 최적화 설정은 $\beta_1=0.9$, $\beta_2=0.95$, weight decay 0.1이다. 학습률은 2,000 warmup step 후 최고치 5.3e-4에 도달하고, 전체 학습의 80% 시점에서 최고치의 31.6%, 90% 시점에서 10.0%로 추가 감소하는 multi-step schedule을 사용한다. batch size는 4M tokens, context length는 4K다.

Table 1은 이 동일한 학습 예산 아래에서 각 코퍼스가 실제 수학 벤치마크 성능에 어떤 차이를 만드는지 보여준다. 여기서 가장 중요한 비교는 DeepSeekMath Corpus 120.2B가 단순히 크기만 큰 것이 아니라, 거의 모든 영어·중국어 수학 벤치마크에서 명확한 우위를 보인다는 점이다. 예를 들어 DeepSeekMath Corpus로 학습한 1.3B 모델은 GSM8K 23.8%, MATH 13.6%, SAT 56.3%, MMLU-STEM 33.1%, CMATH 41.5%, Gaokao-MathQA 23.6%를 기록한다. 반면 Proof-Pile-2는 GSM8K 14.3%, MATH 11.2%, SAT 43.8%, CMATH 19.9%, Gaokao-MathQA 11.7%에 머문다. OpenWebMath는 영어 수학에서는 일정 개선을 보이지만 중국어 수학에서는 매우 제한적이다. MathPile은 오히려 여러 벤치마크에서 기본 모델보다 못한 결과를 보인다.

이 표가 중요한 이유는 “수학 데이터”라는 이름이 붙은 코퍼스라고 해서 다 같은 효과를 내지 않는다는 점을 보여주기 때문이다. 어떤 코퍼스는 양이 적고, 어떤 코퍼스는 영어 편향이 강하고, 어떤 코퍼스는 실제 문제 해결과 연결되는 텍스트 밀도가 낮다. DeepSeekMath Corpus는 크기, 다국어성, 실제 문제 해결과 연결된 웹페이지 비중 측면에서 모두 더 유리하다고 해석할 수 있다.

![Table 1](/assets/images/DeepSeekMath%20Pushing%20the%20Limits%20of%20Mathematical%20Reasoning%20in%20Open%20Language%20Models/DeepSeekMath_Pushing_the_Limits_of_Mathematical_Reasoning_in_Open_Language_Models_03_Table_1.png)

#### 2.2.2. Evaluation Results

저자들은 DeepSeekMath Corpus의 장점을 세 가지로 정리한다. 첫째는 high-quality다. Table 1에서 보듯 DeepSeekMath Corpus로 학습한 모델은 여덟 개 수학 벤치마크 전반에서 가장 높은 성능을 낸다. 단순히 한두 개 벤치마크만 잘하는 것이 아니라, 초등 수준 산술, 고난도 수학 문제, 객관식 STEM 문제, 중국어 수학 문제까지 넓게 이득을 준다.

둘째는 multilingual 특성이다. DeepSeekMath Corpus는 영어와 중국어가 특히 많이 포함된 다국어 수학 데이터다. 이 때문에 영어 중심 코퍼스들이 중국어 수학 평가에서 거의 개선을 보이지 못하거나 오히려 악화되는 것과 달리, DeepSeekMath Corpus는 영어와 중국어 모두에서 이득을 준다. 특히 CMATH 41.5%라는 결과는 같은 1.3B 규모에서 다른 코퍼스와 비교해 압도적으로 높다.

셋째는 large-scale이다. Figure 3은 각 코퍼스에 대해 학습 토큰 수를 늘려가며 benchmark curve를 그린 결과다. 이 그림은 DeepSeekMath Corpus가 더 큰 규모일 뿐 아니라, 학습이 진행될수록 성능이 더 오래 상승하며 plateau에 늦게 도달함을 보여준다. 반면 다른 코퍼스는 크기가 작아 150B 토큰 학습 동안 같은 데이터를 여러 번 반복하게 되고, 성능이 빠르게 포화된다. 즉 DeepSeekMath Corpus의 이점은 단순한 정적 품질이 아니라, 더 오래 유효한 학습 신호를 제공한다는 데 있다.

![Figure 3](/assets/images/DeepSeekMath%20Pushing%20the%20Limits%20of%20Mathematical%20Reasoning%20in%20Open%20Language%20Models/DeepSeekMath_Pushing_the_Limits_of_Mathematical_Reasoning_in_Open_Language_Models_04_Figure_3.png)

### 2.3. Training and Evaluating DeepSeekMath-Base 7B

이제 저자들은 1.3B 비교 실험을 넘어, 실제 주력 모델인 DeepSeekMath-Base 7B를 학습한다. 초기 모델은 DeepSeek-Coder-Base-v1.5 7B이며, 총 500B 토큰을 지속학습한다. 데이터 비중은 DeepSeekMath Corpus 56%, AlgebraicStack 4%, arXiv 10%, GitHub code 20%, Common Crawl 자연어 10%(영어와 중국어 포함)다. 기본 학습 설정은 2.2.1과 유사하되, maximum learning rate는 4.2e-4, batch size는 10M tokens로 조정한다.

저자들은 이 모델을 세 가지 관점에서 평가한다. 첫째는 도구 없이 자기완결적으로 수학 문제를 푸는 능력, 둘째는 도구를 사용한 수학 문제 해결과 formal theorem proving, 셋째는 일반 언어 이해와 코드 능력이다.

#### Mathematical Problem Solving with Step-by-Step Reasoning

Table 2는 few-shot chain-of-thought prompting으로 수학 문제를 푸는 능력을 비교한다. 비교 대상에는 폐쇄형 Minerva 7B/62B/540B와, 오픈소스 Mistral 7B, Llemma 7B/34B가 포함된다. DeepSeekMath-Base 7B는 GSM8K 64.2%, MATH 36.2%, OCW 15.4%, SAT 84.4%, MMLU-STEM 56.5%, CMATH 71.7%, Gaokao-MathCloze 20.3%, Gaokao-MathQA 35.3%를 기록한다.

이 결과는 몇 가지 점에서 매우 인상적이다. 우선 MATH에서 36.2%는 Llemma 34B의 25.3%보다 10%p 이상 높다. 즉 수학 특화 공개 베이스 모델 가운데도 뚜렷한 격차가 난다. 또한 Minerva 540B가 MATH에서 33.6%였다는 점을 감안하면, DeepSeekMath-Base 7B는 훨씬 작은 규모로 더 높은 점수를 낸다. GSM8K에서도 64.2%로 Minerva 540B의 58.8%를 앞선다. 저자들은 이를 통해 웹 기반 수학 코퍼스와 코드 기반 초기화가 단순한 보조 요인이 아니라, 베이스 모델 단계에서 이미 강력한 수학 능력을 형성한다고 해석한다.

![Table 2](/assets/images/DeepSeekMath%20Pushing%20the%20Limits%20of%20Mathematical%20Reasoning%20in%20Open%20Language%20Models/DeepSeekMath_Pushing_the_Limits_of_Mathematical_Reasoning_in_Open_Language_Models_05_Table_2.png)

#### Mathematical Problem Solving with Tool Use

도구 사용 평가에서는 few-shot program-of-thought prompting을 사용해 모델이 Python 프로그램을 생성하고 실행 결과로 정답을 내도록 한다. Table 3에서 DeepSeekMath-Base 7B는 GSM8K+Python 66.9%, MATH+Python 31.4%를 기록한다. 이는 이전 강력한 공개 모델인 Llemma 34B의 64.6%, 26.3%를 모두 넘어선다. CodeLlama 34B도 각각 52.7%, 23.5%로 차이가 꽤 크다.

이 결과는 베이스 모델이 이미 자연어 추론과 프로그램 생성 사이의 연결을 상당히 잘 학습했음을 의미한다. 단지 “코드를 잘 쓴다”가 아니라, 수학적 의미를 프로그램 형태로 외재화하는 능력이 좋아졌다는 뜻이다. 이는 후반부 ablation에서 코드 학습이 수학 추론에 유익하다는 주장과도 연결된다.

#### Formal Mathematics

같은 Table 3에는 informal-to-formal theorem proving 결과도 포함된다. 과제는 informal statement, formal counterpart, informal proof를 바탕으로 Isabelle 형식의 formal proof를 생성하는 것이다. 저자들은 proof sketch를 생성하게 한 다음, Sledgehammer로 누락된 세부 단계를 채운다. DeepSeekMath-Base 7B는 miniF2F-valid 25.8%, miniF2F-test 24.6%를 기록하며, 역시 Llemma 34B(21.0%, 21.3%)를 앞선다. 이는 DeepSeekMath가 단지 산술 word problem에 강한 모델이 아니라, 형식 증명으로 연결되는 보다 구조적 수학에도 의미 있는 강점을 갖는다는 점을 보여준다.

![Table 3](/assets/images/DeepSeekMath%20Pushing%20the%20Limits%20of%20Mathematical%20Reasoning%20in%20Open%20Language%20Models/DeepSeekMath_Pushing_the_Limits_of_Mathematical_Reasoning_in_Open_Language_Models_06_Table_3.png)

#### Natural Language Understanding, Reasoning, and Code

수학 특화 학습이 일반 능력을 얼마나 유지하는지 보기 위해, 저자들은 MMLU, BBH, HumanEval, MBPP를 평가한다. Table 4에서 DeepSeekMath-Base 7B는 MMLU 54.9%, BBH 59.5%, HumanEval 40.9%, MBPP 52.6%를 기록한다. DeepSeek-Coder-Base-v1.5의 특정 checkpoint와 비교하면 MMLU와 BBH는 상당히 향상되었고, 코딩 성능은 대체로 유지된다. 특히 자연어 reasoning 쪽 지표가 개선된 것은 수학 데이터가 일반 reasoning에도 일부 긍정적 전이를 줄 수 있음을 시사한다.

다만 코딩 성능은 DeepSeek-Coder-Base-v1.5 최종 체크포인트의 43.2/60.4보다는 낮다. 저자들은 이를 상쇄하기 위해 수학 지속학습 중에도 20%의 GitHub code를 섞었다고 설명한다. 그 결과 수학 능력 향상과 코딩 능력 보존 사이에서 일정한 균형을 맞춘 셈이다.

![Table 4](/assets/images/DeepSeekMath%20Pushing%20the%20Limits%20of%20Mathematical%20Reasoning%20in%20Open%20Language%20Models/DeepSeekMath_Pushing_the_Limits_of_Mathematical_Reasoning_in_Open_Language_Models_07_Table_4.png)

## 3. Supervised Fine-Tuning

### 3.1. SFT Data Curation

사전학습만으로는 충분하지 않기 때문에, 저자들은 별도의 수학 instruction-tuning 데이터셋을 구축한다. 이 데이터는 영어와 중국어를 모두 포함하며, 난이도도 초등 수학부터 고급 수학까지 다양하다. 답안 형식 역시 단순한 final answer가 아니라 CoT, PoT, tool-integrated reasoning 세 가지를 폭넓게 포함한다. 전체 학습 예제 수는 776K다.

영어 데이터는 GSM8K와 MATH에 대해 tool-integrated solution을 직접 주석하고, 여기에 MathInstruct의 일부와 Lila-OOD 학습 세트를 더해 구성한다. 이 안에는 algebra, probability, number theory, calculus, geometry 등 여러 하위 분야가 포함된다. 중국어 데이터는 K-12 수학 문제를 76개 세부 주제로 수집하고, 각 문제에 CoT와 tool-integrated reasoning 형식의 해설을 부여한다.

이 설계는 매우 중요하다. 사전학습은 모델 내부에 수학적 지식과 패턴을 심어주는 역할을 하지만, 실제 추론을 어떤 형식으로 외부화할지, 도구를 언제 쓰고 어떻게 쓰는지를 정교하게 가르치는 데는 instruction tuning이 더 직접적이기 때문이다. 즉 776K 데이터는 단순히 정답을 알려주는 데이터셋이 아니라, “수학 문제를 어떻게 풀어야 하는가”를 형식 수준에서 정렬하는 데이터셋이다.

### 3.2. Training and Evaluating DeepSeekMath-Instruct 7B

DeepSeekMath-Instruct 7B는 DeepSeekMath-Base를 출발점으로 수학 instruction tuning을 수행한 모델이다. 학습 시에는 여러 예제를 무작위로 이어 붙여 최대 4K context length가 되도록 구성하고, batch size 256, constant learning rate 5e-5로 500 step 학습한다.

평가는 도구 없는 step-by-step reasoning과 tool-integrated reasoning 두 설정에서 수행된다. 비교 대상은 GPT-4, GPT-4 Code Interpreter, Gemini Ultra/Pro, Inflection-2, Grok-1 같은 폐쇄형 모델들과, DeepSeek-LLM-Chat 67B, Qwen 72B, InternLM2-Math 20B, Math-Shepherd-Mistral 7B, WizardMath 계열, MetaMath 70B, ToRA 34B, MAmmoTH 70B 같은 오픈소스 모델들이다.

도구 사용이 금지된 CoT 평가에서 DeepSeekMath-Instruct 7B는 GSM8K 82.9%, MATH 46.8%, MGSM-zh 73.2%, CMATH 84.6%를 기록한다. 이는 대다수 오픈소스 모델을 앞서는 성능이며, 특히 MATH 46.8%는 WizardMath-v1.1 7B의 33.0%, Math-Shepherd-Mistral 7B의 33.0%, Qwen 72B의 35.2%보다 크게 높다. 규모가 훨씬 큰 70B급 오픈소스 모델들과 비교해도 우위가 뚜렷하다. GPT-4와 Gemini Ultra가 여전히 더 높지만, 오픈소스 7B 모델로서는 상당히 높은 수준이다.

도구 사용이 허용된 설정에서는 DeepSeekMath-Instruct 7B가 GSM8K 83.7%, MATH 57.4%, MGSM-zh 72.0%, CMATH 84.3%를 기록한다. MATH 57.4%는 GPT-4 Code Interpreter의 69.7%보다는 낮지만, 기존 오픈소스 중에서는 매우 강하며 DeepSeek-LLM-Chat 67B의 51.1%, ToRA 34B의 50.8%보다 높다. 특히 7B 모델이라는 점을 고려하면 도구 사용과 자연어 추론을 통합하는 instruction tuning의 효과가 상당히 컸음을 알 수 있다.

Table 5는 여기까지의 DeepSeekMath-Instruct 7B 결과와, 뒤에서 설명할 DeepSeekMath-RL 7B 결과를 한 표 안에 함께 배치해 전체 비교를 보여준다. CoT와 tool-integrated reasoning 두 설정 모두에서 RL 버전이 instruct 버전을 전반적으로 상회하며, 특히 MATH와 중국어 수학 벤치마크에서 추가 향상이 확인된다.

![Table 5](/assets/images/DeepSeekMath%20Pushing%20the%20Limits%20of%20Mathematical%20Reasoning%20in%20Open%20Language%20Models/DeepSeekMath_Pushing_the_Limits_of_Mathematical_Reasoning_in_Open_Language_Models_08_Table_5.png)

## 4. Reinforcement Learning

### 4.1. Group Relative Policy Optimization

수학 instruction tuning 이후에도 성능을 더 끌어올리기 위해, 논문은 강화학습 단계를 도입한다. 기존 LLM alignment 문맥에서 PPO는 널리 쓰여 왔지만, value model을 별도로 학습해야 하고 메모리 사용량이 크며, 수학처럼 보상이 마지막에만 주어지는 문제에서는 token별 value 추정이 쉽지 않다. 이를 해결하기 위해 저자들은 GRPO를 제안한다.

#### 4.1.1. From PPO to GRPO

먼저 PPO의 surrogate objective는 다음과 같이 주어진다.

$$
J_{PPO}(\theta)
=
\mathbb{E}_{q \sim P(Q),\; o \sim \pi_{\theta_{old}}(O|q)}
\left[
\frac{1}{|o|}
\sum_{t=1}^{|o|}
\min\left(
\frac{\pi_\theta(o_t|q,o_{<t})}{\pi_{\theta_{old}}(o_t|q,o_{<t})} A_t,\;
\operatorname{clip}\left(
\frac{\pi_\theta(o_t|q,o_{<t})}{\pi_{\theta_{old}}(o_t|q,o_{<t})},
1-\varepsilon, 1+\varepsilon
\right) A_t
\right)
\right].
$$

여기서 $\pi_\theta$는 현재 정책, $\pi_{\theta_{old}}$는 데이터를 생성한 이전 정책, $q$는 질문, $o$는 출력, $\varepsilon$은 PPO의 clipping 하이퍼파라미터다. 핵심은 advantage $A_t$를 이용해 좋은 token은 강화하고 나쁜 token은 약화하되, 정책이 한 번에 너무 크게 이동하지 않도록 비율을 자른다는 점이다.

LLM RL에서는 reward model 과적합을 막기 위해 토큰별 KL penalty를 보상에 추가하는 경우가 많다. 논문은 이를 다음과 같이 쓴다.

$$
r_t
=
r_\phi(q, o_{\le t})
-
\beta \log
\frac{\pi_\theta(o_t|q,o_{<t})}{\pi_{ref}(o_t|q,o_{<t})}.
$$

여기서 $r_\phi$는 reward model, $\pi_{ref}$는 보통 초기 SFT 모델인 reference model, $\beta$는 KL penalty 계수다. PPO에서는 이런 보상과 별도로 value function $V_\psi$를 학습해 GAE로 $A_t$를 계산한다. 문제는 이 value model이 정책 모델과 비슷한 규모가 되기 쉽고, 그 자체로 메모리와 계산 비용이 크다는 점이다. 더구나 수학 RL에서는 보상이 보통 출력의 마지막이나 reasoning step 말단에만 주어지므로, 모든 token에 대해 정확한 value를 예측하는 것은 특히 어렵다.

바로 이 지점에서 GRPO가 등장한다. Figure 4는 PPO와 GRPO의 구조 차이를 보여준다. PPO는 policy model, reward model, value model, reference model이 모두 관여하지만, GRPO는 value model을 제거하고 동일한 질문에 대해 여러 개의 출력을 뽑은 뒤, 그 그룹 내부의 상대적 보상만으로 baseline을 구성한다. 따라서 추가 value network 없이도 advantage를 계산할 수 있다.

![Figure 4](/assets/images/DeepSeekMath%20Pushing%20the%20Limits%20of%20Mathematical%20Reasoning%20in%20Open%20Language%20Models/DeepSeekMath_Pushing_the_Limits_of_Mathematical_Reasoning_in_Open_Language_Models_09_Figure_4.png)

GRPO의 목표 함수는 다음과 같다.

$$
J_{GRPO}(\theta)
=
\mathbb{E}_{q \sim P(Q),\; \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)}
\left[
\frac{1}{G}
\sum_{i=1}^G
\frac{1}{|o_i|}
\sum_{t=1}^{|o_i|}
\left(
\min\left(
\frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q,o_{i,<t})} \hat{A}_{i,t},
\operatorname{clip}\left(
\frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q,o_{i,<t})},
1-\varepsilon, 1+\varepsilon
\right) \hat{A}_{i,t}
\right)
-
\beta D_{KL}[\pi_\theta \Vert \pi_{ref}]
\right)
\right].
$$

여기서 가장 중요한 차이는 $\hat{A}_{i,t}$다. PPO의 $A_t$는 value function 기반의 절대적 advantage인 반면, GRPO의 $\hat{A}_{i,t}$는 “같은 질문에 대해 생성한 여러 답변들 사이에서 상대적으로 얼마나 더 좋은가”를 나타내는 값이다. 수학 문제에서는 같은 질문에 대해 여러 풀이를 샘플링했을 때, 어느 풀이가 더 낫고 어느 풀이가 더 나쁜지 비교하는 것이 절대적인 token value를 추정하는 것보다 더 자연스럽고 안정적일 수 있다.

또한 GRPO는 KL penalty를 보상에 억지로 섞지 않고, loss 항에 KL divergence를 직접 추가한다. 논문은 Schulman(2020)의 unbiased estimator를 사용해 이를 다음과 같이 계산한다.

$$
D_{KL}[\pi_\theta \Vert \pi_{ref}]
=
\frac{\pi_{ref}(o_{i,t}|q,o_{i,<t})}{\pi_\theta(o_{i,t}|q,o_{i,<t})}
-
\log
\frac{\pi_{ref}(o_{i,t}|q,o_{i,<t})}{\pi_\theta(o_{i,t}|q,o_{i,<t})}
-
1.
$$

이 추정량은 항상 양수이며, reference policy와 지나치게 멀어지는 것을 막는 정규화 역할을 한다.

#### 4.1.2. Outcome Supervision RL with GRPO

가장 단순한 형태의 GRPO는 output-level reward만 사용하는 outcome supervision이다. 각 질문 $q$에 대해 기존 정책 $\pi_{\theta_{old}}$로부터 $G$개의 출력 $\{o_1, o_2, \dots, o_G\}$를 샘플링하고, reward model로 각 출력에 대한 스칼라 보상 $r = \{r_1, r_2, \dots, r_G\}$를 얻는다. 그런 다음 같은 그룹 안에서 평균을 빼고 표준편차로 나눈 normalized reward를 만든다.

$$
\tilde{r}_i = \frac{r_i - \operatorname{mean}(r)}{\operatorname{std}(r)}.
$$

Outcome supervision에서는 출력 $o_i$의 모든 token에 대해 동일한 advantage를 부여한다.

$$
\hat{A}_{i,t} = \tilde{r}_i.
$$

즉 어떤 답변이 그룹 평균보다 좋으면 그 답변의 모든 token을 전반적으로 강화하고, 평균보다 나쁘면 전반적으로 약화하는 방식이다. 구조는 단순하지만, 같은 질문 안에서 상대 비교를 하기 때문에 보상의 스케일 편차를 줄이고 학습을 안정화하는 장점이 있다.

#### 4.1.3. Process Supervision RL with GRPO

수학 문제는 결과만 맞는지 아닌지보다 중간 추론 단계가 중요하다. 그래서 저자들은 process supervision도 실험한다. 이 경우 reward model은 각 출력의 각 reasoning step 끝에서 보상을 준다. 질문 $q$와 $G$개의 출력이 주어졌을 때, process reward model은 각 출력의 step 말단마다 보상들을 생성하고, 그 집합 전체를 평균과 표준편차로 정규화한다.

$$
\tilde{r}_i^{index(j)} = \frac{r_i^{index(j)} - \operatorname{mean}(R)}{\operatorname{std}(R)}.
$$

그 다음 token $t$의 advantage는 그 token 이후에 끝나는 모든 step reward의 합으로 정의된다.

$$
\hat{A}_{i,t}
=
\sum_{index(j) \ge t}
\tilde{r}_i^{index(j)}.
$$

이렇게 하면 어떤 reasoning step이 이후 결과에 긍정적이면 그 앞쪽 token들도 함께 보상을 받고, 잘못된 step은 그 구간 이후 token들을 약화시킨다. 논문은 이런 step-aware advantage가 outcome supervision보다 더 세밀한 학습 신호를 제공한다고 본다.

#### 4.1.4. Iterative RL with GRPO

강화학습이 진행될수록 policy model은 계속 바뀌는데, 초기에 만든 reward model이 그 변화에 끝까지 잘 맞는다고 볼 수는 없다. 그래서 저자들은 iterative RL도 도입한다. Algorithm 1의 절차는 다음과 같다.

1. 초기 policy model $\pi_\theta \leftarrow \pi_{\theta_{init}}$로 시작한다.  
2. 각 iteration마다 현재 policy를 reference model $\pi_{ref}$로 고정한다.  
3. task prompt 집합 $D$에서 batch $D_b$를 뽑는다.  
4. old policy를 현재 policy로 갱신한 뒤, 각 질문에 대해 $G$개의 출력을 샘플링한다.  
5. reward model $r_\phi$로 각 출력의 보상을 계산한다.  
6. group-relative advantage $\hat{A}_{i,t}$를 계산한다.  
7. 내부 GRPO iteration $\mu$회 동안 policy를 GRPO objective로 갱신한다.  
8. sampling 결과로 만든 새 데이터를 이용하되, 과거 데이터 10%를 replay로 섞어 reward model을 계속 업데이트한다.

이 구조의 핵심은 reward model과 policy model을 함께 시대별로 갱신한다는 점이다. policy가 좋아질수록 더 어려운 출력을 생성하게 되고, 그에 맞춰 reward model도 더 좋은 감독 신호를 주도록 재훈련된다. 수학 추론처럼 출력 분포가 빠르게 변하는 영역에서는 이런 iterative refinement가 특히 중요하다.

### 4.2. Training and Evaluating DeepSeekMath-RL

실제 RL 실험은 DeepSeekMath-Instruct 7B를 시작점으로 한다. RL 학습 데이터는 SFT 데이터 중에서도 GSM8K와 MATH에 해당하는 CoT 형식 질문 약 144K개만 사용한다. 다른 질문은 의도적으로 제외해, RL이 학습 중 직접 보지 못한 out-of-domain 벤치마크에도 일반화되는지를 확인한다.

초기 reward model은 DeepSeekMath-Base 7B 위에서 learning rate 2e-5로 학습한다. 정책 모델의 learning rate는 1e-6, KL coefficient는 0.04다. 각 질문마다 64개의 출력을 샘플링하고, max length는 1024, batch size는 1024로 둔다. 그리고 각 exploration stage 뒤에는 정책을 한 번만 업데이트한다.

Table 5에서 RL 결과를 보면, DeepSeekMath-RL 7B는 CoT 기준으로 GSM8K 88.2%, MATH 51.7%, MGSM-zh 79.6%, CMATH 88.8%를 기록한다. tool-integrated reasoning에서는 GSM8K 86.7%, MATH 58.8%, MGSM-zh 78.4%, CMATH 87.6%다. 중요한 점은 RL 학습 데이터가 GSM8K와 MATH의 CoT 형식에 제한되어 있었음에도, 중국어 수학 벤치마크나 tool-integrated reasoning 같은 다른 평가 설정까지 전반적으로 개선되었다는 것이다. 이는 RL이 단순히 특정 benchmark 형식에 overfit한 것이 아니라, 정책 분포 자체를 더 좋은 방향으로 정렬했음을 시사한다.

## 5. Discussion

### 5.1. Lessons Learnt in Pre-Training

이 절에서는 사전학습에 관한 두 가지 중요한 교훈을 정리한다. 하나는 코드 학습이 수학 추론에 실제로 도움이 된다는 점이고, 다른 하나는 arXiv 논문이 기대만큼 효과적이지 않다는 점이다. 별도 언급이 없는 한, 여기서의 DeepSeekMath Corpus는 데이터 수집 두 번째 iteration에서 확보한 89B-token 버전을 의미한다.

#### 5.1.1. Code Training Benefits Mathematical Reasoning

코드 학습이 reasoning을 돕는다는 가설은 오래 있었지만, 구체적으로 어떤 방식으로 도움이 되는지는 명확하지 않았다. 저자들은 DeepSeek-LLM 1.3B를 사용해 두 가지 two-stage setting과 두 가지 one-stage setting을 비교한다.

Two-stage 설정에서는  
- 일반 토큰 400B 학습 후 수학 토큰 150B 학습,  
- 코드 토큰 400B 학습 후 수학 토큰 150B 학습  
을 비교한다.

One-stage 설정에서는  
- 수학 토큰 150B만 학습,  
- 코드 400B와 수학 150B를 섞어 한 번에 학습  
을 비교한다.

Table 6은 수학 추론 자체를, 그것도 도구 없이 푸는 경우와 Python을 활용하는 경우로 나누어 보여준다. 결과를 보면 코드 학습은 특히 tool use 기반 수학 추론에서 매우 강한 효과를 낸다. 예를 들어 two-stage에서 코드만 400B 학습한 단계에서도 GSM8K+Python 12.4%, MATH+Python 10.0%로, 일반 사전학습만 한 경우의 3.3%, 2.3%보다 훨씬 높다. 이후 수학 학습까지 더하면 17.4%, 9.4%로 더 올라간다. one-stage mixed training도 GSM8K+Python 19.7%, MATH+Python 13.5%로 강력하다.

도구 없는 순수 CoT 수학 추론에서도 코드 학습은 도움이 된다. two-stage code→math는 최종적으로 GSM8K 21.9%, MATH 15.3%, CMATH 39.7%로, general→math의 19.1%, 14.4%, 37.2%보다 높다. 즉 코드는 단순히 Python 사용법을 가르치는 데이터가 아니라, 알고리즘적 분해와 정밀한 단계 서술 능력을 강화해 수학 추론 자체에도 긍정적 영향을 줄 수 있다.

![Table 6](/assets/images/DeepSeekMath%20Pushing%20the%20Limits%20of%20Mathematical%20Reasoning%20in%20Open%20Language%20Models/DeepSeekMath_Pushing_the_Limits_of_Mathematical_Reasoning_in_Open_Language_Models_10_Table_6.png)

Table 7은 같은 학습 설정이 일반 이해·추론·코딩에 미치는 영향을 보여준다. 코드 400B만 학습하면 HumanEval 25.0%, MBPP 40.0%로 크게 상승하고, 이후 수학 학습을 거치면 MMLU 36.2%, BBH 35.3%로 일반 reasoning도 좋아진다. one-stage code+math mixed training은 MMLU 33.5%, BBH 35.6%, HumanEval 29.3%, MBPP 39.4%로, 수학과 코딩을 동시에 비교적 잘 유지한다. 저자들은 이를 catastrophic forgetting 완화 효과로 해석한다.

다만 one-stage mixed training은 도구 없는 순수 수학 추론에서는 two-stage code→math보다 약하다. 논문은 그 이유를 1.3B라는 작은 모델 규모 때문이라고 추정한다. 즉 작은 모델은 코드와 수학을 동시에 충분히 흡수하기에 용량이 부족할 수 있고, 그래서 어떤 설정에서는 섞어 학습하는 것보다 순차 학습이 더 유리하다.

![Table 7](/assets/images/DeepSeekMath%20Pushing%20the%20Limits%20of%20Mathematical%20Reasoning%20in%20Open%20Language%20Models/DeepSeekMath_Pushing_the_Limits_of_Mathematical_Reasoning_in_Open_Language_Models_11_Table_7.png)

#### 5.1.2. ArXiv Papers Seem Ineffective in Improving Mathematical Reasoning

수학 사전학습에서 arXiv는 거의 관습처럼 포함되어 왔다. 그러나 저자들은 “정말 arXiv가 수학 문제 해결 능력을 높이는가?”를 별도로 점검한다. 비교 대상은 수학 데이터로 널리 쓰이는 MathPile과, arXiv LaTeX 원문에서 preamble, comments, macros, bibliographies를 제거한 ArXiv-RedPajama다.

DeepSeek-LLM 1.3B에는 각 코퍼스로 150B 토큰씩 학습하고, DeepSeek-Coder-Base-v1.5 7B에는 40B 토큰씩 학습한다. Table 8을 보면 결과는 다소 반직관적이다. 1.3B 모델에서 MathPile은 GSM8K 2.7%, MATH 3.3%, CMATH 1.2%로 거의 개선이 없고, ArXiv-RedPajama는 GSM8K 3.3%, MATH 3.4%, MMLU-STEM 9.0%처럼 오히려 좋지 않은 경우가 많다. 7B 모델에서도 ArXiv-RedPajama는 SAT나 일부 객관식에서는 소폭 이득이 있지만, GSM8K, MATH, MMLU-STEM 등 핵심 지표에서는 명확한 개선이 없거나 저하된다.

![Table 8](/assets/images/DeepSeekMath%20Pushing%20the%20Limits%20of%20Mathematical%20Reasoning%20in%20Open%20Language%20Models/DeepSeekMath_Pushing_the_Limits_of_Mathematical_Reasoning_in_Open_Language_Models_12_Table_8.png)

Table 9는 formal mathematics에서의 영향도 보여준다. DeepSeek-Coder-Base-v1.5 7B 기준으로 miniF2F-valid/test는 no math training이 20.1/21.7인데, MathPile은 16.8/16.4, ArXiv-RedPajama는 14.8/11.9로 더 낮다. 즉 적어도 이 논문이 사용한 설정에서는 arXiv-only 수학 학습이 theorem proving에도 유리하지 않다.

![Table 9](/assets/images/DeepSeekMath%20Pushing%20the%20Limits%20of%20Mathematical%20Reasoning%20in%20Open%20Language%20Models/DeepSeekMath_Pushing_the_Limits_of_Mathematical_Reasoning_in_Open_Language_Models_13_Table_9.png)

저자들은 이 결론을 지나치게 일반화하지는 않는다. 특정 과제, 다른 데이터와의 혼합, 더 큰 모델 규모에서는 arXiv의 가치가 달라질 수 있다고 인정한다. 그럼에도 논문이 던지는 메시지는 분명하다. “수학과 관련된 텍스트”라는 이유만으로 arXiv를 자동으로 고품질 수학 reasoning 데이터라고 가정해서는 안 되며, 실제 benchmark 성능으로 검증해야 한다는 것이다.

### 5.2. Insights of Reinforcement Learning

#### 5.2.1. Towards to a Unified Paradigm

저자들은 SFT, RFT, DPO, PPO, GRPO 같은 여러 학습 방법을 하나의 통일된 관점에서 해석하려 한다. 이때 각 방법의 gradient는 다음과 같은 일반 형태로 쓸 수 있다.

$$
\nabla_\theta J_A(\theta)
=
\mathbb{E}_{(q,o)\sim D}
\left[
\frac{1}{|o|}
\sum_{t=1}^{|o|}
GC_A(q,o,t,\pi_{rf})
\nabla_\theta \log \pi_\theta(o_t|q,o_{<t})
\right].
$$

여기에는 세 가지 요소가 있다. 첫째, 데이터가 어디서 왔는가를 정하는 Data Source $D$. 둘째, 어떤 보상 신호를 쓰는가를 정하는 Reward Function $\pi_{rf}$. 셋째, 그 보상 신호를 실제 gradient coefficient로 어떻게 바꾸는가를 정하는 Algorithm $A$다. 즉 학습 방법들의 차이는 결국 “어떤 데이터를 보고, 어떤 신호를 보상으로 삼고, 그것을 어떻게 gradient 크기로 바꾸는가”의 차이로 환원된다.

Table 10은 이 관점을 요약한다.  
- SFT는 사람 손으로 선택된 $(q,o)$ 데이터에서 gradient coefficient가 항상 1이다.  
- RFT는 SFT 질문에 대해 SFT 모델이 생성한 출력 중 정답인 것만 받아들여 학습한다.  
- DPO는 선호/비선호 출력 쌍을 이용한다.  
- Online RFT는 샘플링 출력을 현재 policy에서 얻는다.  
- PPO와 GRPO는 현재 policy 출력에 reward model을 적용해 gradient coefficient를 계산한다.

![Table 10](/assets/images/DeepSeekMath%20Pushing%20the%20Limits%20of%20Mathematical%20Reasoning%20in%20Open%20Language%20Models/DeepSeekMath_Pushing_the_Limits_of_Mathematical_Reasoning_in_Open_Language_Models_14_Table_10.png)

Figure 5는 DeepSeekMath-Instruct 1.3B를 바탕으로 RFT, Online RFT, GRPO+OS, GRPO+PS를 비교한 결과다. 여기서 첫 번째 관찰은 online sampling의 중요성이다. RFT와 DPO처럼 초기 SFT 모델이 생성한 고정 데이터로 학습하는 것보다, 현재 policy가 실제로 생성하는 데이터를 다시 학습에 쓰는 Online RFT와 GRPO가 후반부에 훨씬 더 강해진다. 초반에는 actor와 초기 SFT 모델이 비슷해서 차이가 크지 않지만, 학습이 진행될수록 actor가 생성하는 출력 분포가 달라지기 때문이다.

두 번째 관찰은 gradient coefficient의 표현력이 중요하다는 점이다. Online RFT는 정답이면 모두 같은 강도로 강화하고, 오답은 강화하지 않는다. 반면 GRPO는 reward model이 부여한 상대적 점수에 따라 정답 안에서도 더 좋은 풀이를 더 강하게 밀어주고, 오답은 더 적극적으로 억제할 수 있다. 이 차이가 결국 더 강한 성능 향상으로 이어진다.

세 번째 관찰은 process supervision의 가치다. Figure 5에서 GRPO+PS는 GRPO+OS보다 더 높은 성능을 보인다. 이는 중간 reasoning step을 세밀하게 감독하는 것이 단순히 최종 정답 여부만 보는 것보다 수학 문제에 더 적합하다는 사실을 보여준다.

![Figure 5](/assets/images/DeepSeekMath%20Pushing%20the%20Limits%20of%20Mathematical%20Reasoning%20in%20Open%20Language%20Models/DeepSeekMath_Pushing_the_Limits_of_Mathematical_Reasoning_in_Open_Language_Models_15_Figure_5.png)

마지막으로 iterative RL도 중요한 차이를 만든다. Figure 6은 DeepSeekMath-Instruct 7B에 대해 iteration 0, 1, 2를 비교한 결과인데, 첫 번째 iteration에서 특히 큰 향상이 나타나고, 그 뒤에도 추가 향상이 이어진다. reward model을 고정하는 대신 policy 변화에 맞추어 계속 업데이트하는 것이 실제로 효과가 있다는 뜻이다.

![Figure 6](/assets/images/DeepSeekMath%20Pushing%20the%20Limits%20of%20Mathematical%20Reasoning%20in%20Open%20Language%20Models/DeepSeekMath_Pushing_the_Limits_of_Mathematical_Reasoning_in_Open_Language_Models_16_Figure_6.png)

#### 5.2.2. Why RL Works?

저자들은 RL이 왜 효과적인지 더 깊이 이해하기 위해, Instruct 모델과 RL 모델의 Pass@K와 Maj@K를 비교한다. Figure 7은 GSM8K와 MATH에서 temperature 0.7로 여러 개 후보를 뽑았을 때의 결과를 보여준다. 여기서 중요한 패턴은 RL이 Pass@K는 거의 늘리지 않지만, Maj@K는 뚜렷하게 끌어올린다는 점이다.

Pass@K는 $K$개 후보 중 하나라도 정답이 있으면 성공으로 보는 지표이므로, 모델이 잠재적으로 정답 풀이를 생성할 수 있는지에 더 가깝다. 반면 Maj@K는 여러 후보 중 다수결 또는 대표 응답이 얼마나 안정적으로 정답 쪽으로 모이는지를 본다. RL 이후 Maj@K만 주로 향상된다는 것은, 모델의 근본적인 문제 해결 능력이 갑자기 생겼다기보다는, 이미 존재하던 올바른 풀이들을 출력 분포 상 더 앞쪽, 더 안정적인 위치로 끌어올렸다는 뜻이다.

저자들은 이것을 reasoning에서의 alignment 문제로 해석한다. 즉 SFT 모델도 정답 풀이를 만들 잠재력은 어느 정도 가지고 있지만, 그 풀이가 top of distribution에 충분히 집중되어 있지 않다. RL은 이 분포를 재정렬해 올바른 reasoning path가 더 자주, 더 일관되게 선택되도록 만든다.

![Figure 7](/assets/images/DeepSeekMath%20Pushing%20the%20Limits%20of%20Mathematical%20Reasoning%20in%20Open%20Language%20Models/DeepSeekMath_Pushing_the_Limits_of_Mathematical_Reasoning_in_Open_Language_Models_17_Figure_7.png)

#### 5.2.3. How to Achieve More Effective RL?

논문은 더 효과적인 RL을 위해 세 가지 축을 제안한다.

첫째, Data Source다. 현재 논문은 instruction tuning 단계의 질문과 비교적 단순한 nucleus sampling만 사용했다. 저자들은 이것이 RL이 주로 Maj@K를 개선하고 Pass@K는 덜 개선한 한 이유라고 본다. 앞으로는 out-of-distribution 질문, tree-search 계열의 고급 decoding, 더 다양한 탐색 전략을 도입해 exploration 자체를 강화할 필요가 있다. 또한 speculative decoding이나 더 효율적인 serving 기법도 RL의 탐색 효율을 높이는 데 중요하다.

둘째, Algorithms다. 현재 대부분의 방법은 reward signal을 거의 신뢰하고, 그에 따라 조건부 확률을 올리거나 내린다. 그러나 실제 reward model은 noisy할 수 있다. 특히 복잡한 수학 문제에서는 잘못된 보상이나 불확실한 보상이 자주 발생한다. 따라서 앞으로는 noisy reward에 강인한 RL, 약한 보상 모델로부터 강한 정책을 끌어내는 weak-to-strong alignment 방식이 중요해질 수 있다.

셋째, Reward Function이다. reward model은 out-of-distribution 질문과 고급 decoding 출력에 잘 일반화되어야 하고, 자신의 uncertainty를 표현할 수 있어야 하며, reasoning step 단위로 정교한 보상을 주는 process reward model도 더 효율적으로 구축되어야 한다. 결국 수학 RL의 병목은 policy만이 아니라, 좋은 보상 신호를 얼마나 신뢰성 있게 만들 수 있는가에도 달려 있다.

## 6. Conclusion, Limitation, and Future Work

논문은 DeepSeekMath를 통해 오픈소스 수학 언어 모델의 성능 상한을 크게 끌어올렸다고 결론짓는다. DeepSeekMath는 DeepSeek-Coder-v1.5 7B에서 출발해 500B 토큰을 지속학습했고, 그중 핵심은 Common Crawl에서 추출한 120B 수학 토큰이다. 실험은 웹페이지 기반 수학 데이터가 매우 큰 잠재력을 가지며, 반대로 arXiv는 기대만큼 유익하지 않을 수 있음을 보여준다. 또한 GRPO는 PPO보다 적은 메모리로 수학 추론을 유의미하게 향상시키는 RL 방법으로 제시된다. 여기에 더해 논문은 SFT, RFT, DPO, PPO, GRPO를 하나의 통일된 프레임으로 정리하고, 더 나은 RL을 위한 방향까지 제안한다.

한편 한계도 분명히 언급한다. DeepSeekMath는 양적 추론 벤치마크에서는 매우 강하지만, geometry와 theorem proving 쪽은 폐쇄형 모델보다 여전히 약하다. 저자들은 삼각형이나 타원 관련 문제를 모델이 제대로 처리하지 못한 예를 언급하며, 이것이 사전학습·미세조정 데이터 선택의 편향을 반영할 수 있다고 본다. 또한 모델 규모의 제약 때문에 few-shot 활용 능력은 GPT-4보다 약하다. GPT-4는 few-shot 예시가 주어질 때 성능이 더 오르지만, DeepSeekMath는 zero-shot과 few-shot의 차이가 상대적으로 작다.

따라서 미래 과제로는 더 정교한 데이터 선택 파이프라인 구축, geometry나 형식 수학 같은 취약 영역 보강, 그리고 Section 5.2.3에서 제시한 더 강력한 RL 방향 탐색이 제시된다.

## References

아래 참고문헌은 원문 순서를 유지하여 정리한다.

R. Anil, S. Borgeaud, Y. Wu, J. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth,
K. Millican, D. Silver, S. Petrov, M. Johnson, I. Antonoglou, J. Schrittwieser, A. Glaese, J. Chen,
E. Pitler, T. P. Lillicrap, A. Lazaridou, O. Firat, J. Molloy, M. Isard, P. R. Barham, T. Hennigan,
B. Lee, F. Viola, M. Reynolds, Y. Xu, R. Doherty, E. Collins, C. Meyer, E. Rutherford, E. Moreira,
K. Ayoub, M. Goel, G. Tucker, E. Piqueras, M. Krikun, I. Barr, N. Savinov, I. Danihelka,
B. Roelofs, A. White, A. Andreassen, T. von Glehn, L. Yagati, M. Kazemi, L. Gonzalez,
M. Khalman, J. Sygnowski, and et al. Gemini: A family of highly capable multimodal
models. CoRR, abs/2312.11805, 2023. doi: 10.48550/ARXIV.2312.11805. URL https:
//doi.org/10.48550/arXiv.2312.11805.
J. Austin, A. Odena, M. Nye, M. Bosma, H. Michalewski, D. Dohan, E. Jiang, C. Cai, M. Terry,
Q. Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732,
2021.
Z. Azerbayev, H. Schoelkopf, K. Paster, M. D. Santos, S. McAleer, A. Q. Jiang, J. Deng, S. Bider-
man, and S. Welleck. Llemma: An open language model for mathematics. arXiv preprint
arXiv:2310.10631, 2023.
J. Bai, S. Bai, Y. Chu, Z. Cui, K. Dang, X. Deng, Y. Fan, W. Ge, Y. Han, F. Huang, et al. Qwen
technical report. arXiv preprint arXiv:2309.16609, 2023.
C. Burns, P. Izmailov, J. H. Kirchner, B. Baker, L. Gao, L. Aschenbrenner, Y. Chen, A. Ecoffet,
M. Joglekar, J. Leike, et al. Weak-to-strong generalization: Eliciting strong capabilities with
weak supervision. arXiv preprint arXiv:2312.09390, 2023.
ChatGLM3 Team. Chatglm3 series: Open bilingual chat llms, 2023. URL https://github.c
om/THUDM/ChatGLM3.
M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. de Oliveira Pinto, J. Kaplan, H. Edwards, Y. Burda,
N. Joseph, G. Brockman, A. Ray, R. Puri, G. Krueger, M. Petrov, H. Khlaaf, G. Sastry, P. Mishkin,
B. Chan, S. Gray, N. Ryder, M. Pavlov, A. Power, L. Kaiser, M. Bavarian, C. Winter, P. Tillet,
F. P. Such, D. Cummings, M. Plappert, F. Chantzis, E. Barnes, A. Herbert-Voss, W. H. Guss,
A. Nichol, A. Paino, N. Tezak, J. Tang, I. Babuschkin, S. Balaji, S. Jain, W. Saunders, C. Hesse,
A. N. Carr, J. Leike, J. Achiam, V. Misra, E. Morikawa, A. Radford, M. Knight, M. Brundage,
M. Murati, K. Mayer, P. Welinder, B. McGrew, D. Amodei, S. McCandlish, I. Sutskever, and
W. Zaremba. Evaluating large language models trained on code. CoRR, abs/2107.03374, 2021.
URL https://arxiv.org/abs/2107.03374.
W. Chen, X. Ma, X. Wang, and W. W. Cohen. Program of thoughts prompting: Disentangling
computation from reasoning for numerical reasoning tasks. CoRR, abs/2211.12588, 2022. doi:
10.48550/ARXIV.2211.12588. URL https://doi.org/10.48550/arXiv.2211.12588.
K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek,
J. Hilton, R. Nakano, et al. Training verifiers to solve math word problems. arXiv preprint
arXiv:2110.14168, 2021.
T. Computer. Redpajama: an open dataset for training large language models, Oct. 2023. URL
https://github.com/togethercomputer/RedPajama-Data.
DeepSeek-AI. Deepseek LLM: scaling open-source language models with longtermism. CoRR,
abs/2401.02954, 2024. doi: 10.48550/ARXIV.2401.02954. URL https://doi.org/10.485
50/arXiv.2401.02954.

Z. Du, Y. Qian, X. Liu, M. Ding, J. Qiu, Z. Yang, and J. Tang. Glm: General language model
pretraining with autoregressive blank infilling. In Proceedings of the 60th Annual Meeting
of the Association for Computational Linguistics (Volume 1: Long Papers), pages 320–335,
2022.
L. Gao, A. Madaan, S. Zhou, U. Alon, P. Liu, Y. Yang, J. Callan, and G. Neubig. PAL: program-
aided language models. In A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and
J. Scarlett, editors, International Conference on Machine Learning, ICML 2023, 23-29 July
2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research,
pages 10764–10799. PMLR, 2023. URL https://proceedings.mlr.press/v202/gao23f.
html.
Z. Gou, Z. Shao, Y. Gong, Y. Shen, Y. Yang, M. Huang, N. Duan, and W. Chen. Tora: A tool-
integrated reasoning agent for mathematical problem solving. CoRR, abs/2309.17452, 2023.
doi: 10.48550/ARXIV.2309.17452. URL https://doi.org/10.48550/arXiv.2309.1745
2.
D. Guo, Q. Zhu, D. Yang, Z. Xie, K. Dong, W. Zhang, G. Chen, X. Bi, Y. Wu, Y. K. Li, F. Luo,
Y. Xiong, and W. Liang. Deepseek-coder: When the large language model meets programming
– the rise of code intelligence, 2024.
D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt. Measuring
massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.
D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt. Mea-
suring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874,
2021.
High-flyer. Hai-llm: 高效且轻量的大模型训练工具, 2023. URL https://www.high-flyer.c
n/en/blog/hai-llm.
Inflection AI. Inflection-2, 2023. URL https://inflection.ai/inflection-2.
A. Q. Jiang, S. Welleck, J. P. Zhou, W. Li, J. Liu, M. Jamnik, T. Lacroix, Y. Wu, and G. Lample. Draft,
sketch, and prove: Guiding formal theorem provers with informal proofs. arXiv preprint
arXiv:2210.12283, 2022.
A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. d. l. Casas, F. Bressand,
G. Lengyel, G. Lample, L. Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.
A. Joulin, E. Grave, P. Bojanowski, M. Douze, H. Jégou, and T. Mikolov. Fasttext. zip: Compress-
ing text classification models. arXiv preprint arXiv:1612.03651, 2016.
W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica.
Efficient memory management for large language model serving with pagedattention. In
Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.
Y. Leviathan, M. Kalman, and Y. Matias. Fast inference from transformers via speculative
decoding. In International Conference on Machine Learning, pages 19274–19286. PMLR,
2023.
A. Lewkowycz, A. Andreassen, D. Dohan, E. Dyer, H. Michalewski, V. Ramasesh, A. Slone,
C. Anil, I. Schlag, T. Gutman-Solo, et al. Solving quantitative reasoning problems with
language models. Advances in Neural Information Processing Systems, 35:3843–3857, 2022a.

A. Lewkowycz, A. Andreassen, D. Dohan, E. Dyer, H. Michalewski, V. V. Ramasesh, A. Slone,
C. Anil, I. Schlag, T. Gutman-Solo, Y. Wu, B. Neyshabur, G. Gur-Ari, and V. Misra. Solving
quantitative reasoning problems with language models. In S. Koyejo, S. Mohamed, A. Agarwal,
D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems
35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New
Orleans, LA, USA, November 28 - December 9, 2022, 2022b. URL http://papers.nips.
cc/paper_files/paper/2022/hash/18abbeef8cfe9203fdf9053c9c4fe191-Abstr
act-Conference.html.
H. Lightman, V. Kosaraju, Y. Burda, H. Edwards, B. Baker, T. Lee, J. Leike, J. Schulman,
I. Sutskever, and K. Cobbe. Let’s verify step by step. arXiv preprint arXiv:2305.20050, 2023.
I. Loshchilov and F. Hutter.
Decoupled weight decay regularization.
arXiv preprint
arXiv:1711.05101, 2017.
H. Luo, Q. Sun, C. Xu, P. Zhao, J. Lou, C. Tao, X. Geng, Q. Lin, S. Chen, and D. Zhang.
Wizardmath: Empowering mathematical reasoning for large language models via reinforced
evol-instruct. arXiv preprint arXiv:2308.09583, 2023.
S. Mishra, M. Finlayson, P. Lu, L. Tang, S. Welleck, C. Baral, T. Rajpurohit, O. Tafjord, A. Sab-
harwal, P. Clark, and A. Kalyan. LILA: A unified benchmark for mathematical reasoning.
In Y. Goldberg, Z. Kozareva, and Y. Zhang, editors, Proceedings of the 2022 Conference on
Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab
Emirates, December 7-11, 2022, pages 5807–5832. Association for Computational Linguistics,
2022. doi: 10.18653/V1/2022.EMNLP-MAIN.392. URL https://doi.org/10.18653/v1/
2022.emnlp-main.392.
X. Nguyen, W. Zhang, X. Li, M. M. Aljunied, Q. Tan, L. Cheng, G. Chen, Y. Deng, S. Yang,
C. Liu, H. Zhang, and L. Bing. Seallms - large language models for southeast asia. CoRR,
abs/2312.00738, 2023. doi: 10.48550/ARXIV.2312.00738. URL https://doi.org/10.485
50/arXiv.2312.00738.
OpenAI. GPT4 technical report. arXiv preprint arXiv:2303.08774, 2023.
L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal,
K. Slama, A. Ray, et al. Training language models to follow instructions with human feedback.
Advances in Neural Information Processing Systems, 35:27730–27744, 2022.
K. Paster, M. D. Santos, Z. Azerbayev, and J. Ba. Openwebmath: An open dataset of high-quality
mathematical web text. CoRR, abs/2310.06786, 2023. doi: 10.48550/ARXIV.2310.06786. URL
https://doi.org/10.48550/arXiv.2310.06786.
L. C. Paulson. Three years of experience with sledgehammer, a practical link between auto-
matic and interactive theorem provers. In R. A. Schmidt, S. Schulz, and B. Konev, editors,
Proceedings of the 2nd Workshop on Practical Aspects of Automated Reasoning, PAAR-2010,
Edinburgh, Scotland, UK, July 14, 2010, volume 9 of EPiC Series in Computing, pages 1–10.
EasyChair, 2010. doi: 10.29007/TNFD. URL https://doi.org/10.29007/tnfd.
S. Polu and I. Sutskever. Generative language modeling for automated theorem proving. CoRR,
abs/2009.03393, 2020. URL https://arxiv.org/abs/2009.03393.
R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning, and C. Finn. Direct preference
optimization: Your language model is secretly a reward model. 2023.

J. Schulman. Approximating kl divergence, 2020. URL http://joschu.net/blog/kl-app
rox.html.
J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel. High-dimensional continuous
control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015.
J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization
algorithms. arXiv preprint arXiv:1707.06347, 2017.
F. Shi, M. Suzgun, M. Freitag, X. Wang, S. Srivats, S. Vosoughi, H. W. Chung, Y. Tay, S. Ruder,
D. Zhou, D. Das, and J. Wei. Language models are multilingual chain-of-thought reasoners.
In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali,
Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview.net/pdf?id=
fR3wGCk-IXp.
F. Song, B. Yu, M. Li, H. Yu, F. Huang, Y. Li, and H. Wang. Preference ranking optimization for
human alignment. arXiv preprint arXiv:2306.17492, 2023.
M. Suzgun, N. Scales, N. Schärli, S. Gehrmann, Y. Tay, H. W. Chung, A. Chowdhery, Q. V. Le,
E. H. Chi, D. Zhou, et al. Challenging big-bench tasks and whether chain-of-thought can solve
them. arXiv preprint arXiv:2210.09261, 2022.
T. Tao. Embracing change and resetting expectations, 2023. URL https://unlocked.micro
soft.com/ai-anthology/terence-tao/.
H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra,
P. Bhargava, S. Bhosale, D. Bikel, L. Blecher, C. Canton-Ferrer, M. Chen, G. Cucurull, D. Esiobu,
J. Fernandes, J. Fu, W. Fu, B. Fuller, C. Gao, V. Goswami, N. Goyal, A. Hartshorn, S. Hosseini,
R. Hou, H. Inan, M. Kardas, V. Kerkez, M. Khabsa, I. Kloumann, A. Korenev, P. S. Koura,
M. Lachaux, T. Lavril, J. Lee, D. Liskovich, Y. Lu, Y. Mao, X. Martinet, T. Mihaylov, P. Mishra,
I. Molybog, Y. Nie, A. Poulton, J. Reizenstein, R. Rungta, K. Saladi, A. Schelten, R. Silva, E. M.
Smith, R. Subramanian, X. E. Tan, B. Tang, R. Taylor, A. Williams, J. X. Kuan, P. Xu, Z. Yan,
I. Zarov, Y. Zhang, A. Fan, M. Kambadur, S. Narang, A. Rodriguez, R. Stojnic, S. Edunov, and
T. Scialom. Llama 2: Open foundation and fine-tuned chat models. CoRR, abs/2307.09288,
2023. doi: 10.48550/arXiv.2307.09288. URL https://doi.org/10.48550/arXiv.2307.
09288.
T. H. Trinh, Y. Wu, Q. V. Le, H. He, and T. Luong. Solving olympiad geometry without human
demonstrations. Nature, 625(7995):476–482, 2024.
P. Wang, L. Li, L. Chen, F. Song, B. Lin, Y. Cao, T. Liu, and Z. Sui. Making large language models
better reasoners with alignment. arXiv preprint arXiv:2309.02144, 2023a.
P. Wang, L. Li, Z. Shao, R. Xu, D. Dai, Y. Li, D. Chen, Y. Wu, and Z. Sui. Math-shepherd: Verify
and reinforce llms step-by-step without human annotations. CoRR, abs/2312.08935, 2023b.
Z. Wang, R. Xia, and P. Liu. Generative AI for math: Part I - mathpile: A billion-token-scale
pretraining corpus for math. CoRR, abs/2312.17120, 2023c. doi: 10.48550/ARXIV.2312.17120.
URL https://doi.org/10.48550/arXiv.2312.17120.
J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. H. Chi, Q. V. Le, and D. Zhou.
Chain-of-thought prompting elicits reasoning in large language models. In NeurIPS, 2022.
URL http://papers.nips.cc/paper_files/paper/2022/hash/9d5609613524ecf
4f15af0f7b31abca4-Abstract-Conference.html.

T. Wei, J. Luan, W. Liu, S. Dong, and B. Wang. Cmath: Can your language model pass chinese
elementary school math test?, 2023.
M. Wenzel, L. C. Paulson, and T. Nipkow. The isabelle framework. In O. A. Mohamed, C. A.
Muñoz, and S. Tahar, editors, Theorem Proving in Higher Order Logics, 21st International
Conference, TPHOLs 2008, Montreal, Canada, August 18-21, 2008. Proceedings, volume 5170
of Lecture Notes in Computer Science, pages 33–38. Springer, 2008. doi: 10.1007/978-3-540-7
1067-7\_7. URL https://doi.org/10.1007/978-3-540-71067-7_7.
H. Xia, T. Ge, P. Wang, S.-Q. Chen, F. Wei, and Z. Sui. Speculative decoding: Exploiting
speculative execution for accelerating seq2seq generation. In H. Bouamor, J. Pino, and K. Bali,
editors, Findings of the Association for Computational Linguistics: EMNLP 2023, pages 3909–
3925, Singapore, Dec. 2023. Association for Computational Linguistics. doi: 10.18653/v1/20
23.findings-emnlp.257. URL https://aclanthology.org/2023.findings-emnlp.257.
H. Xia, Z. Yang, Q. Dong, P. Wang, Y. Li, T. Ge, T. Liu, W. Li, and Z. Sui. Unlocking efficiency
in large language model inference: A comprehensive survey of speculative decoding. arXiv
preprint arXiv:2401.07851, 2024.
S. Yao, D. Yu, J. Zhao, I. Shafran, T. L. Griffiths, Y. Cao, and K. Narasimhan. Tree of thoughts:
Deliberate problem solving with large language models. arXiv preprint arXiv:2305.10601,
2023.
L. Yu, W. Jiang, H. Shi, J. Yu, Z. Liu, Y. Zhang, J. T. Kwok, Z. Li, A. Weller, and W. Liu.
Metamath: Bootstrap your own mathematical questions for large language models. CoRR,
abs/2309.12284, 2023. doi: 10.48550/ARXIV.2309.12284. URL https://doi.org/10.485
50/arXiv.2309.12284.
Z. Yuan, H. Yuan, C. Li, G. Dong, C. Tan, and C. Zhou. Scaling relationship on learning
mathematical reasoning with large language models. arXiv preprint arXiv:2308.01825, 2023a.
Z. Yuan, H. Yuan, C. Tan, W. Wang, S. Huang, and F. Huang. Rrhf: Rank responses to align
language models with human feedback without tears. arXiv preprint arXiv:2304.05302, 2023b.
X. Yue, X. Qu, G. Zhang, Y. Fu, W. Huang, H. Sun, Y. Su, and W. Chen. Mammoth: Building
math generalist models through hybrid instruction tuning. CoRR, abs/2309.05653, 2023. doi:
10.48550/ARXIV.2309.05653. URL https://doi.org/10.48550/arXiv.2309.05653.
K. Zheng, J. M. Han, and S. Polu. Minif2f: a cross-system benchmark for formal olympiad-level
mathematics. arXiv preprint arXiv:2109.00110, 2021.
W. Zhong, R. Cui, Y. Guo, Y. Liang, S. Lu, Y. Wang, A. Saied, W. Chen, and N. Duan. AGIEval: A
human-centric benchmark for evaluating foundation models. CoRR, abs/2304.06364, 2023.
doi: 10.48550/arXiv.2304.06364. URL https://doi.org/10.48550/arXiv.2304.06364.

## Appendix A. Analysis of Reinforcement Learning

이 부록은 본문 5.2.1의 통일된 관점을 수식 수준에서 더 자세히 전개한다. 핵심은 SFT, RFT, Online RFT, DPO, PPO, GRPO가 모두 데이터 원천, 보상 신호, 그리고 gradient coefficient의 차이로 이해될 수 있다는 점이다.

### A.1. Analysis of Reinforcement Learning

#### A.1.1. Supervised Fine-tuning

Supervised Fine-tuning의 목적함수는 다음과 같다.

$$
J_{SFT}(\theta)
=
\mathbb{E}_{q,o \sim P_{sft}(Q,O)}
\left[
\frac{1}{|o|}
\sum_{t=1}^{|o|}
\log \pi_\theta(o_t|q,o_{<t})
\right].
$$

이에 대한 gradient는 다음과 같다.

$$
\nabla_\theta J_{SFT}(\theta)
=
\mathbb{E}_{q,o \sim P_{sft}(Q,O)}
\left[
\frac{1}{|o|}
\sum_{t=1}^{|o|}
\nabla_\theta \log \pi_\theta(o_t|q,o_{<t})
\right].
$$

SFT에서 데이터 원천은 사람이 정제한 instruction dataset 자체다. 보상 함수는 명시적 reward model이 아니라 사람의 선택으로 간주할 수 있으며, gradient coefficient는 항상 1이다. 즉 SFT는 “좋은 답변을 그대로 모사하는 최대우도 학습”이다.

#### A.1.2. Rejection Sampling Fine-tuning

RFT는 SFT 모델이 생성한 여러 출력 중 정답인 출력만 골라 다시 학습하는 방식이다. 목적함수는 다음과 같다.

$$
J_{RFT}(\theta)
=
\mathbb{E}_{q \sim P_{sft}(Q),\; o \sim \pi_{sft}(O|q)}
\left[
\frac{1}{|o|}
\sum_{t=1}^{|o|}
I(o) \log \pi_\theta(o_t|q,o_{<t})
\right].
$$

gradient는 다음과 같이 쓸 수 있다.

$$
\nabla_\theta J_{RFT}(\theta)
=
\mathbb{E}_{q \sim P_{sft}(Q),\; o \sim \pi_{sft}(O|q)}
\left[
\frac{1}{|o|}
\sum_{t=1}^{|o|}
I(o) \nabla_\theta \log \pi_\theta(o_t|q,o_{<t})
\right].
$$

여기서 gradient coefficient는 다음과 같다.

$$
GC_{RFT}(q,o,t) = I(o) =
\begin{cases}
1 & \text{the answer of } o \text{ is correct} \\
0 & \text{the answer of } o \text{ is incorrect}
\end{cases}
$$

따라서 RFT는 정답 출력 전체를 강화하고 오답 출력은 버린다. 중간 추론의 질 차이나 정답 내부의 상대적 우열은 반영하지 못한다.

#### A.1.3. Online Rejection Sampling Fine-tuning

Online RFT는 RFT와 거의 같지만, 출력 $o$를 더 이상 고정된 SFT 모델에서 뽑지 않고 현재 policy $\pi_\theta$에서 샘플링한다. 따라서 gradient는 다음과 같다.

$$
\nabla_\theta J_{OnRFT}(\theta)
=
\mathbb{E}_{q \sim P_{sft}(Q),\; o \sim \pi_\theta(O|q)}
\left[
\frac{1}{|o|}
\sum_{t=1}^{|o|}
I(o) \nabla_\theta \log \pi_\theta(o_t|q,o_{<t})
\right].
$$

데이터 원천이 실시간 policy exploration으로 바뀌었다는 점이 핵심이다. 논문 본문의 Figure 5에서 Online RFT가 RFT보다 후반부에 강해지는 이유가 여기에 있다.

#### A.1.4. Direct Preference Optimization (DPO)

DPO는 선호 출력 $o^+$와 비선호 출력 $o^-$의 쌍을 이용한다. 목적함수는 본질적으로 “선호 출력의 상대 로그확률은 올리고, 비선호 출력의 상대 로그확률은 내리는” sigmoid 분류 문제다.

$$
J_{DPO}(\theta)
=
\mathbb{E}_{q \sim P_{sft}(Q),\; o^+,o^- \sim \pi_{sft}(O|q)}
\left[
\log \sigma \left(
\beta \frac{1}{|o^+|} \sum_t \log \frac{\pi_\theta(o_t^+|q,o_{<t}^+)}{\pi_{ref}(o_t^+|q,o_{<t}^+)}
-
\beta \frac{1}{|o^-|} \sum_t \log \frac{\pi_\theta(o_t^-|q,o_{<t}^-)}{\pi_{ref}(o_t^-|q,o_{<t}^-)}
\right)
\right].
$$

논문은 gradient coefficient를 다음과 같이 적는다.

$$
GC_{DPO}(q,o,t)
=
\sigma \left(
\beta \log \frac{\pi_\theta(o_t^-|q,o_{<t}^-)}{\pi_{ref}(o_t^-|q,o_{<t}^-)}
-
\beta \log \frac{\pi_\theta(o_t^+|q,o_{<t}^+)}{\pi_{ref}(o_t^+|q,o_{<t}^+)}
\right).
$$

DPO의 중요한 특징은 pairwise preference를 사용하지만, 여전히 데이터는 SFT 모델이 오프라인으로 생성한 출력 쌍에 의존한다는 점이다.

#### A.1.5. Proximal Policy Optimization (PPO)

부록은 PPO의 목적함수를 다시 적고, 단순화를 위해 매 exploration stage마다 정책을 한 번만 업데이트한다고 가정한다. 그러면 $\pi_{\theta_{old}} = \pi_\theta$로 보고 clipping 및 min 연산을 제거한 분석이 가능해진다.

$$
J_{PPO}(\theta)
=
\mathbb{E}_{q \sim P_{sft}(Q),\; o \sim \pi_{\theta_{old}}(O|q)}
\left[
\frac{1}{|o|}
\sum_{t=1}^{|o|}
\frac{\pi_\theta(o_t|q,o_{<t})}{\pi_{\theta_{old}}(o_t|q,o_{<t})}
A_t
\right].
$$

따라서 gradient는

$$
\nabla_\theta J_{PPO}(\theta)
=
\mathbb{E}_{q \sim P_{sft}(Q),\; o \sim \pi_{\theta_{old}}(O|q)}
\left[
\frac{1}{|o|}
\sum_{t=1}^{|o|}
A_t \nabla_\theta \log \pi_\theta(o_t|q,o_{<t})
\right]
$$

가 되며, gradient coefficient는

$$
GC_{PPO}(q,o,t,\pi_\theta^{rm}) = A_t
$$

이다. 여기서 $A_t$는 reward들과 value function $V_\psi$로부터 GAE를 사용해 계산된다. 즉 PPO의 핵심 비용은 좋은 reward model뿐 아니라, token별 baseline 역할을 하는 value model까지 함께 가져가야 한다는 데 있다.

#### A.1.6. Group Relative Policy Optimization (GRPO)

GRPO 역시 단순화를 위해 $\pi_{\theta_{old}} = \pi_\theta$라고 놓고 분석한다. 이때 목적함수는 다음과 같이 쓸 수 있다.

$$
J_{GRPO}(\theta)
=
\mathbb{E}_{q \sim P_{sft}(Q),\; \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)}
\left[
\frac{1}{G}
\sum_{i=1}^G
\frac{1}{|o_i|}
\sum_{t=1}^{|o_i|}
\left(
\frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q,o_{i,<t})}
\hat{A}_{i,t}
-
\beta
\left(
\frac{\pi_{ref}(o_{i,t}|q,o_{i,<t})}{\pi_\theta(o_{i,t}|q,o_{i,<t})}
-
\log
\frac{\pi_{ref}(o_{i,t}|q,o_{i,<t})}{\pi_\theta(o_{i,t}|q,o_{i,<t})}
-1
\right)
\right)
\right].
$$

gradient는

$$
\nabla_\theta J_{GRPO}(\theta)
=
\mathbb{E}_{q \sim P_{sft}(Q),\; \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)}
\left[
\frac{1}{G}
\sum_{i=1}^G
\frac{1}{|o_i|}
\sum_{t=1}^{|o_i|}
\left(
\hat{A}_{i,t}
+
\beta \left(
\frac{\pi_{ref}(o_{i,t}|o_{i,<t})}{\pi_\theta(o_{i,t}|o_{i,<t})} - 1
\right)
\right)
\nabla_\theta \log \pi_\theta(o_{i,t}|q,o_{i,<t})
\right]
$$

가 된다. 따라서 gradient coefficient는

$$
GC_{GRPO}(q,o,t,\pi_\theta^{rm})
=
\hat{A}_{i,t}
+
\beta \left(
\frac{\pi_{ref}(o_{i,t}|o_{i,<t})}{\pi_\theta(o_{i,t}|o_{i,<t})} - 1
\right).
$$

이 수식은 본문의 핵심 아이디어를 압축적으로 보여준다. PPO가 value function으로 advantage를 구성한다면, GRPO는 같은 질문에 대한 여러 출력의 상대적 점수로 $\hat{A}_{i,t}$를 구성한다. 그래서 additional critic 없이도 정책을 효과적으로 업데이트할 수 있다.

## 추가 설명

이 논문의 가장 큰 의미는 “수학 성능을 올리기 위해 어떤 데이터와 어떤 학습 절차가 실제로 중요한가”를 매우 구체적으로 분해했다는 데 있다. 우선 데이터 측면에서 보면, 수학 reasoning에 직접 도움이 되는 텍스트는 단순히 수학 기호가 많이 들어간 문서가 아니다. DeepSeekMath Corpus가 강했던 이유는 웹페이지라는 형식 자체 때문이라기보다, 문제 풀이 문맥, 직관적 설명, 단계적 전개, 다국어 질의응답 흔적, 실제 교육용 콘텐츠 등이 대량으로 포함되어 있었기 때문으로 해석할 수 있다. 반대로 arXiv는 엄밀한 수학·과학 텍스트이지만, 언어 모델이 benchmark 스타일의 문제 해결로 연결하기에는 신호가 너무 간접적일 수 있다. 이 논문은 그 차이를 수치로 보여준다.

또한 코드 학습의 역할도 중요하다. 수학 추론은 결국 상태를 구조화하고, 중간 결과를 보존하고, 절차적으로 문제를 분해하는 과정인데, 코드는 그 구조를 매우 명시적으로 표현하는 데이터다. 그래서 코드 학습은 Python tool use뿐 아니라, 자연어 기반 CoT reasoning에도 간접적으로 도움을 줄 수 있다. 다만 작은 모델에서는 코드와 수학을 동시에 대량 혼합하는 것이 항상 최선은 아니며, 학습 순서와 혼합 비율이 중요하다는 점도 함께 드러난다.

GRPO의 의의는 RLHF 문맥의 일반적 논리를 수학 추론에 더 잘 맞는 방식으로 재설계했다는 데 있다. 수학 문제에서는 하나의 질문에 대해 여러 풀이를 생성해 보면, 어느 풀이가 더 논리적이고 어느 풀이가 더 허술한지 상대 비교가 가능하다. GRPO는 바로 그 구조를 이용해, 절대적 value를 token마다 예측하는 대신 그룹 내부 상대 순위를 baseline으로 삼는다. 이것은 메모리를 줄이는 공학적 이점뿐 아니라, 수학이라는 과제 구조에 더 자연스럽게 맞는 학습 신호를 제공한다는 점에서도 중요하다.

마지막으로 RL이 Pass@K보다 Maj@K를 더 많이 올렸다는 결과는 이 논문의 RL 해석에서 핵심적이다. 이는 RL이 모델에 완전히 새로운 능력을 갑자기 주입했다기보다, 이미 존재하던 올바른 reasoning path를 더 자주 선택하게 만들었다는 뜻이다. 다시 말해 DeepSeekMath 프로젝트의 성과는 “사전학습으로 잠재적 능력을 충분히 만들고, instruction tuning으로 형식을 정렬한 뒤, RL로 출력 분포를 안정화한다”는 3단계 전략이 효과적으로 맞물렸기 때문에 가능했다. 이 관점은 이후 수학 reasoning LLM 설계에서도 매우 중요한 기준점이 된다.

