# DeepMath-103K

Zhiwei He, Tian Liang, Jiahao Xu, Qiuzhi Liu, Xingyu Chen, Yue Wang, Linfeng Song, Dian Yu, Zhenwen Liang, Wenxuan Wang, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, Dong Yu

Tencent, Shanghai Jiao Tong University

GitHub: https://github.com/zwhe99/DeepMath  
Dataset: https://hf.co/datasets/zwhe99/DeepMath-103K

## Abstract

이 논문은 수학 추론용 강화학습 데이터의 가장 핵심적인 세 조건, 즉 **충분히 높은 난도**, **평가 벤치마크와의 오염 제거**, **규칙 기반 보상에 바로 사용할 수 있는 검증 가능한 정답**을 동시에 만족하는 공개 자원이 부족하다는 문제에서 출발한다. 저자들은 이 공백을 메우기 위해 DeepMath-103K를 제안한다. 이 데이터셋은 약 103K개의 수학 문제로 구성되며, 핵심 95K는 난도 5 이상 문제만 남긴 고난도 코어 세트이고, 추가로 8K개의 난도 3-5 문제를 더해 난도 범위를 보완한다.

각 샘플에는 문제 본문, 검증 가능한 최종 정답, 난도 점수, 계층적 주제 라벨, 그리고 DeepSeek-R1이 생성한 세 개의 서로 다른 풀이가 포함된다. 이 설계는 RLVR에서의 정답 보상뿐 아니라 SFT, distillation, solution imitation 같은 여러 후속 학습 방식까지 동시에 지원하려는 목적을 갖는다. 즉 이 데이터셋은 단순한 문제-정답 모음이 아니라, “최종 정답 중심 RL”과 “풀이 중심 SFT”를 모두 염두에 둔 구조로 설계된 것이다.

DeepMath-103K의 또 다른 특징은 원천 데이터의 성격이다. 기존 공개 수학 데이터셋 상당수는 이미 많이 재활용된 대회 문제나 잘 정제된 기존 리소스를 다시 모으는 방식이었는데, 이 논문은 Math StackExchange 같은 덜 정형화된 수학 포럼 데이터까지 적극적으로 끌어들인다. 그 대신 강한 정제 파이프라인을 적용하여 비정형 원문을 표준화된 질의응답 형식으로 변환한다. 그 결과 기존 데이터셋과 겹치지 않는 새로운 문제가 많이 유입되고, 임베딩 분포 자체도 기존 공개 데이터셋과 다른 패턴을 보인다.

논문은 이렇게 구축한 DeepMath-103K로 학습한 DeepMath 계열 모델이 수학 벤치마크에서 강한 성능 향상을 보일 뿐 아니라 GPQA-Diamond처럼 수학 바깥의 과학 추론에서도 더 나은 결과를 내는 점을 보여 준다. 저자들의 핵심 주장은 단순하다. 강한 수학 추론 모델을 학습시키려면 단순히 큰 데이터가 아니라, **어렵고, 깨끗하고, 검증 가능하며, 다양한** 데이터가 필요하고, DeepMath-103K는 그 네 요소를 동시에 만족하는 공개 자원이라는 것이다.

## 1. Introduction

최근 대규모 언어모델의 수학 추론 능력을 끌어올리는 핵심 방법 중 하나는 RL, 그중에서도 정답의 맞고 틀림을 자동으로 판정할 수 있는 RLVR이다. 이 방식은 사람이 사고 과정 전체를 평가하지 않아도 최종 답만으로 보상을 줄 수 있으므로 확장성이 좋고, 실제로 강한 reasoning model들을 만드는 데 큰 역할을 해 왔다. 그러나 RL이 강하다고 해서 아무 데이터에나 붙일 수 있는 것은 아니다. 정답이 검증 가능해야 하고, 모델이 이미 다 맞히는 쉬운 문제만으로는 충분한 학습 신호가 생기지 않으며, 평가 데이터와 훈련 데이터가 섞여 있어서는 안 된다.

저자들은 기존 공개 수학 데이터셋이 이 조건을 동시에 만족하지 못한다고 본다. 첫째, 다수의 공개 RL/SFT용 수학 데이터는 현재의 강한 모델을 더 밀어붙이기에는 너무 쉽다. 둘째, 공개 train split만 사용하더라도 유명 벤치마크와의 중복이 상당하여, 성능 향상이 진짜 일반화인지 데이터 누출인지 분간하기 어렵다. 셋째, 정답이 서술형이거나 너무 길고 복잡해 규칙 기반 추출과 검증이 어려운 샘플이 많다. 넷째, 여러 데이터셋이 결국 비슷한 원천 문제를 재조합하거나 재필터링한 결과물이라 서로 간 중복과 분포 유사성이 크다.

DeepMath-103K는 이 문제를 네 갈래로 해결하려 한다. 첫 번째는 **고난도 문제 비중을 높이는 것**이다. 두 번째는 **광범위한 평가 벤치마크에 대한 오염 제거**다. 세 번째는 **각 문제가 반드시 검증 가능한 최종 정답을 갖도록 보장하는 것**이다. 네 번째는 **각 문제마다 세 개의 R1 풀이를 추가하여 RL뿐 아니라 SFT에도 바로 쓰이도록 만드는 것**이다. 이 네 가지는 논문의 도입부 전반을 관통하는 설계 원리다.

Figure 1의 왼쪽 패널은 DeepMath-103K와 기존 공개 수학 데이터셋들의 난도 분포를 비교한다. Open-RS, DAPO-17K, DSR-Preview, STILL-3-RL, Open-R1, ORZ-129K와 나란히 놓고 보면 DeepMath-103K는 난도 5 이상 영역이 훨씬 두껍고, 특히 난도 6-9의 문제 비중이 뚜렷하게 높다. 즉 이 데이터셋은 “문제 수가 많은 데이터”가 아니라, **고난도 구간에 집중된 데이터**다.

오른쪽 패널은 DeepMath-103K를 이용해 학습한 DeepMath 계열 모델들의 AIME25 성능을 요약한다. Zero RL 설정에서는 DeepMath-Zero-7B가 17.5, DeepMath-Zero-Math-7B가 23.5를 기록하고, RL 설정에서는 DeepMath-1.5B가 30.8, DeepMath-Omn-1.5B가 57.3을 기록한다. 논문은 이를 통해 데이터셋 설계가 모델 성능 향상으로 직접 연결된다고 주장한다.

![Figure 1](/assets/images/DeepMath-103K/DeepMath_103K_A_Large_Scale_Challenging_Decontaminated_and_Verifiable_Mathematical_Dataset_for_Advancing_Reasoning_01_Figure_1.png)

저자들은 도입부에서 세 가지 기여를 분명히 한다. 첫째, DeepMath-103K 데이터셋 자체를 공개한다. 둘째, 이 데이터셋을 만들기 위해 어떤 수집·정제·난도 추정·정답 검증 파이프라인을 사용했는지 상세히 제시한다. 셋째, 그 데이터셋으로 학습한 DeepMath 시리즈 모델이 수학과 비수학 추론 모두에서 강력한 결과를 보인다는 실험을 제시한다.

## 2. Overview of DeepMath-103K

DeepMath-103K의 각 샘플은 단순한 문제-정답 쌍이 아니다. Figure 2에 나온 예시처럼 한 샘플 안에는 문제 본문, 최종 정답, 주제 계층, 난도, 그리고 세 개의 R1 풀이가 함께 들어 있다. 논문은 이 구조를 통해 하나의 샘플이 여러 연구 목적을 동시에 지원하도록 설계되었다고 설명한다.

![Figure 2](/assets/images/DeepMath-103K/DeepMath_103K_A_Large_Scale_Challenging_Decontaminated_and_Verifiable_Mathematical_Dataset_for_Advancing_Reasoning_02_Figure_2.png)

이 구조에서 **Question**은 원래의 수학 문제 진술이다. **Final Answer**는 규칙으로 추출하고 비교할 수 있는 최종 정답이다. **Topic**은 계층적 분류 체계를 따라 상위 분야와 하위 분야를 함께 제공한다. **Difficulty**는 수치형 난도 라벨이다. **R1 Solutions**는 DeepSeek-R1이 생성한 서로 다른 세 개의 풀이 경로다. 이 세 개의 풀이가 함께 있다는 점은 중요하다. 정답만 있는 RL 데이터와 달리, DeepMath-103K는 “같은 문제를 풀어 가는 다양한 reasoning trajectory”까지 제공하므로 SFT나 distillation으로 연결하기 쉽다.

### 2.1 Higher Difficulty

DeepMath-103K는 난도 3부터 9까지를 포함하지만, 중심은 난도 5-9 구간이다. 코어 데이터 95K가 이 구간에서 선별되었고, 난도 범위를 넓히기 위해 SimpleRL에서 난도 3-5의 문제 8K를 추가했다. 논문의 입장은 명확하다. 강한 reasoning model을 더 끌어올리려면 모델이 이미 쉽게 맞히는 문제보다 아직 충분히 어렵고 실수할 가능성이 있는 문제를 더 많이 보여 주는 편이 낫다.

도입부 Figure 1a가 이미 보여 주듯이 DeepMath-103K의 분포는 기존 공개 RLVR 학습 데이터보다 눈에 띄게 더 어렵다. 저자들은 이것을 단순한 통계 차이가 아니라 설계 철학의 반영으로 해석한다. 즉 처음부터 “강한 모델을 더 강하게 만들기 위한 데이터”를 목표로 했기 때문에 난도 상위 구간 중심으로 데이터셋을 만들었다는 뜻이다.

### 2.2 Rigorous Data Decontamination

DeepMath-103K는 기존 공개 자원의 훈련 분할만 사용했지만, 저자들이 원천 데이터 풀을 조사해 보니 평가 벤치마크와의 오염률이 매우 높았다. Figure 3은 이 사실을 정면으로 보여 준다. Omni-MATH 83-24는 92.6%, AIME 83-24는 91.1%, AIME24와 AMC23은 각각 90.0%, Math Odyssey는 88.4%, GaoKao(MC)는 78.0%, MATH500은 76.6%, GaoKao(MQA)는 70.1%, MATH는 67.7%의 오염률을 보인다. JEEBench 49.3%, Minerva Math 35.7%, MMLU-STEM 35.1%, CMATH 33.9%, OlympiadBench 33.6%, Olympiad Arena 32.3%도 결코 낮지 않다. GSM8K 9.8%, GPQA 5.0%도 완전히 무시할 수 있는 수준은 아니다.

이 결과는 “train split만 썼다”는 사실만으로는 평가의 청결성을 보장할 수 없음을 의미한다. 그래서 DeepMath-103K는 표준 벤치마크들과의 의미적 중복을 별도로 탐지하여 제거하는 강한 decontamination 절차를 거친다.

![Figure 3](/assets/images/DeepMath-103K/DeepMath_103K_A_Large_Scale_Challenging_Decontaminated_and_Verifiable_Mathematical_Dataset_for_Advancing_Reasoning_03_Figure_3.png)

### 2.3 Broad Topical Diversity

DeepMath-103K는 고난도라는 특성만 갖는 것이 아니라, 주제 분포도 넓다. Figure 4는 데이터셋의 계층적 주제 분해를 시각화한 것이다. 상위 수준에서 Calculus, Algebra, Precalculus, Applied Mathematics, Geometry, Discrete Mathematics, Number Theory, Differential Equations 등이 나타나고, 그 아래에는 Single-variable, Applications of Integrals, Multi-variable, Group Theory, Field Theory, Polynomial Operations, Probability, Combinatorics, Graph Theory, Congruences 같은 더 세부적인 주제가 이어진다.

논문이 강조하는 포인트는 “단지 다양한 문제를 모았다”가 아니다. 깊이 있는 수학 추론을 학습시키려면 모델이 특정 대회 스타일이나 특정 주제군에 과도하게 편중되지 않아야 하고, 다양한 개념 체계를 오가며 문제를 풀어야 한다. DeepMath-103K는 이러한 요구를 만족하도록 고전적 기초 주제와 고급 주제를 함께 포함한다.

![Figure 4](/assets/images/DeepMath-103K/DeepMath_103K_A_Large_Scale_Challenging_Decontaminated_and_Verifiable_Mathematical_Dataset_for_Advancing_Reasoning_04_Figure_4.png)

### 2.4 Data Novelty and Uniqueness

논문은 DeepMath-103K가 기존 공개 데이터셋과 얼마나 겹치는지를 정량적으로 측정한다. 방법은 비교적 단순하지만 의미가 크다. 먼저 모든 샘플을 `paraphrase-multilingual-MiniLM-L12-v2`로 임베딩한 뒤, 임베딩 유사도가 0.98을 넘는 경우 같은 문제로 본다. 이렇게 각 데이터셋을 임베딩 집합으로 보고, 다른 데이터셋에 없는 고유 샘플 수를 계산한다.

Figure 5에 따르면 DeepMath-103K는 82.81K개의 고유 문제를 갖는다. 이에 비해 Open-R1은 43.66K, STILL-3-RL은 28.3K, DSR-Preview는 11.07K, DAPO-17K는 9.37K, ORZ-129K는 4.28K의 고유 문제만을 가진다. 반대로 비고유 문제 수는 Open-R1 50.32K, STILL-3-RL 59.76K, DSR-Preview 29.25K, DAPO-17K 8.03K, ORZ-129K 125.04K로 나타난다. 이 비교는 DeepMath-103K가 기존 공개 데이터 재조합물과는 꽤 다른 성격을 가진다는 점을 시사한다.

![Figure 5](/assets/images/DeepMath-103K/DeepMath_103K_A_Large_Scale_Challenging_Decontaminated_and_Verifiable_Mathematical_Dataset_for_Advancing_Reasoning_05_Figure_5.png)

### 2.5 Embedding-space Comparison

고유 문제 수 비교에 더해, 논문은 t-SNE로 임베딩 분포를 시각화한다. Figure 6에서 ORZ-129K, Open-R1, STILL-3-RL, DSR-Preview, DAPO-17K는 서로 매우 비슷한 분포 모양을 보인다. 이는 이들 데이터셋이 독립적으로 구축되었다고 해도 결국 비슷한 원천 분포에서 나왔을 가능성을 강하게 암시한다. 반면 DeepMath-103K는 다른 군과 구분되는 분포를 보인다.

이 시각화는 DeepMath-103K의 차별점이 단순히 샘플 수나 난도에서만 나오지 않음을 보여 준다. 이 데이터셋은 **주제 구성, 표현 양식, 문제 출처의 성격 자체가 다르기 때문에** 임베딩 공간에서 별도 영역을 형성한다는 것이 논문의 해석이다.

![Figure 6](/assets/images/DeepMath-103K/DeepMath_103K_A_Large_Scale_Challenging_Decontaminated_and_Verifiable_Mathematical_Dataset_for_Advancing_Reasoning_06_Figure_6.png)

## 3. Construction of DeepMath-103K

Figure 7은 DeepMath-103K 구축 파이프라인 전체를 한 장으로 정리한다. MMIQC 1,202K, WebInstSub 1,612K, NuminaMath-CoT 55K를 합쳐 2,869K의 raw pool을 만들고, 여기서 decontamination을 거쳐 2,670K의 decontaminated pool을 얻는다. 이 과정에서 199K가 제거된다. 이후 difficulty filtering을 거쳐 1,090K의 high-difficulty pool을 만들고, 1,580K를 제거한다. 마지막으로 answer verifiability filtering을 적용해 95K를 남기고 995K를 제거한다. 그 뒤 SimpleRL의 8K를 추가하여 최종 DeepMath-103K가 된다.

![Figure 7](/assets/images/DeepMath-103K/DeepMath_103K_A_Large_Scale_Challenging_Decontaminated_and_Verifiable_Mathematical_Dataset_for_Advancing_Reasoning_07_Figure_7.png)

논문은 이 전체 파이프라인이 상당한 계산 비용을 요구했다고 밝힌다. 총 GPT-4o API 비용은 약 138,000달러였고, H20 GPU 사용량은 총 127,000시간 규모였다. 즉 DeepMath-103K는 단순 크롤링 결과가 아니라, 상당한 비용을 들여 정제된 고가치 자원이다.

### 3.1 Stage 1: Source Analysis and Collection

저자들은 먼저 공개 수학 데이터 원천들의 난도 분포를 조사한다. MetaMathQA, dart-math-hard, OpenMathInstruct-2처럼 GSM8K/MATH에서 파생된 계열 데이터와 NuminaMath-CoT, MMIQC, WebInstructSub처럼 웹 전반에서 문제를 모은 계열 데이터를 비교한다. Figure 8이 보여 주듯 GSM8K/MATH 계열 파생 데이터는 낮은 난도(1-5)에 더 많이 몰려 있고, MMIQC와 WebInstructSub는 중간 이상 난도(5-9)의 비중이 훨씬 크다.

이 결과를 바탕으로 저자들은 MMIQC와 WebInstructSub의 Math StackExchange 관련 서브셋을 주 원천으로 선택하고, 초기 수집본의 주제 다양성을 보완하기 위해 NuminaMath-CoT도 추가한다. 이렇게 얻은 초기 raw pool은 2,869K 문제다.

![Figure 8](/assets/images/DeepMath-103K/DeepMath_103K_A_Large_Scale_Challenging_Decontaminated_and_Verifiable_Mathematical_Dataset_for_Advancing_Reasoning_08_Figure_8.png)

### 3.2 Stage 2: Data Decontamination

이 단계의 목표는 원천 데이터와 표준 평가 벤치마크가 겹치지 않도록 하는 것이다. 대상 벤치마크는 MATH, AIME, AMC, Minerva Math, OlympiadBench, Omni-MATH, MathOdyssey, GAOKAO, JEEBench, MMLU-STEM, CMATH, OlympiadArena, GSM8K, GPQA까지 매우 넓다.

방법은 두 단계다. 먼저 각 후보 문제에 대해 `paraphrase-multilingual-MiniLM-L12-v2` 임베딩을 사용해 각 벤치마크 test set에서 top-k=5 유사 예제를 검색한다. 이어서 Llama-3.3-70B-Instruct를 LLM judge로 써서 각 후보와 검색된 예제들이 동일 문제인지, 혹은 의미적으로 사실상 같은 paraphrase인지 판정한다. 어떤 비교에서든 중복 가능성이 포착되면 그 후보는 버린다.

Table 1은 이 semantic decontamination이 단순한 문자열 비교보다 훨씬 강력하다는 것을 보여 주는 예시다. AIME24 예제에서는 20x20 격자에서 좌상단에서 우하단으로 가는 경로 수를 묻는 raw question과, 8x8 격자에서 우하단에서 좌상단으로 가되 방향을 정확히 네 번 바꾸는 경로 수를 묻는 benchmark question 사이의 의미적 유사성이 강조된다. AMC23 예제에서는 3,5,9 paise 조합으로 정확히 지불할 수 없는 최대 금액을 묻는 문제와, 6,10,15 cent 동전으로 exact change가 불가능한 가장 비싼 물건 값을 묻는 문제 사이의 개념적 대응이 드러난다. 표면 문자열이 완전히 같지 않아도, 실제 문제 구조가 같다면 제거 대상이라는 것이 이 표의 메시지다.

![Table 1](/assets/images/DeepMath-103K/DeepMath_103K_A_Large_Scale_Challenging_Decontaminated_and_Verifiable_Mathematical_Dataset_for_Advancing_Reasoning_09_Table_1.png)

### 3.3 Stage 3: Difficulty Filtering

DeepMath-103K는 단순히 깨끗한 데이터셋이 아니라, 어려운 문제를 남기는 데이터셋이어야 한다. 이를 위해 저자들은 Omni-MATH의 난도 라벨링 방식을 따라 GPT-4o에게 각 문제의 난도를 평가하게 한다. 난도 기준은 AoPS의 competition rating 지침에 기반하며, 단일 판정의 불안정을 줄이기 위해 각 문제를 여섯 번 평가한 뒤 평균을 사용한다. 그리고 난도 5 이상만 남긴다.

Table 2는 difficulty filtering을 통과한 geometry 문제 예시다. 난도 5에서는 무작위 점들의 convex polygon 부분집합 기대 크기를 묻는 문제가 등장하고, 난도 6에서는 직선과 포물선 조건을 만족하는 정사각형의 최소 넓이를 묻는다. 난도 7은 회전하는 두 정다각형의 교집합 변 개수 수열을 요구하고, 난도 8은 단위원에 내접한 n각형의 변과 대각선 제곱합 최대값을 다룬다. 난도 9는 projective plane의 동형성 제약 아래 가능한 최대 family 크기를 묻는다. 표를 따라갈수록 계산량만 늘어나는 것이 아니라, 개념 깊이와 문제 구조가 함께 어려워진다는 것이 논문의 포인트다.

![Table 2](/assets/images/DeepMath-103K/DeepMath_103K_A_Large_Scale_Challenging_Decontaminated_and_Verifiable_Mathematical_Dataset_for_Advancing_Reasoning_10_Table_2.png)

### 3.4 Stage 4: Answer Verification

RLVR에서 가장 중요한 것은 정답이 규칙 기반으로 안정적으로 검증 가능해야 한다는 점이다. 그러나 실제 수학 문제에는 두 가지 장애가 있다. 하나는 애초에 명확한 단일 최종 답이 존재하지 않는 개방형 문제다. 다른 하나는 답은 있지만 너무 길거나 기호 구조가 복잡해서 자동 추출과 비교가 어렵다는 점이다.

이를 해결하기 위해 논문은 두 단계 검증 절차를 쓴다. 첫 번째는 **question filtering and formatting**이다. GPT-4o로 raw question을 처리하여 애초에 검증 불가능한 유형은 버리고, 대화체나 비정형 진술은 하나의 명확한 수치/기호 답을 요구하는 형식으로 다시 쓴다. 두 번째는 **consistency-based answer verification**이다. 이 단계를 통과한 문제에 대해 DeepSeek-R1으로 세 개의 풀이를 생성하고, 규칙 기반 verifier로 세 풀이와 원본 해설에서 각각 최종 답을 추출한다. 그리고 네 곳에서 나온 최종 답이 모두 일치하는 경우에만 최종 데이터셋에 남긴다.

이 단계는 단지 “답을 하나 붙이는 것”이 아니라, **풀이 경로가 달라도 동일한 최종 정답으로 수렴하는 문제만 남기는 것**이다. 따라서 최종 DeepMath-103K는 RL reward로 쓰기에 훨씬 안정적인 데이터가 된다.

## 4. DeepMath Series Models

이 장은 DeepMath-103K로 학습한 모델들이 실제로 어떤 성능을 내는지 다룬다. 논문은 단순히 데이터셋만 공개하는 데 그치지 않고, 그 데이터셋으로 학습한 DeepMath 시리즈 모델을 통해 데이터의 실질적 가치를 검증한다.

### 4.1 Setup

저자들은 두 가지 RL 학습 패러다임을 사용한다. 첫 번째는 **Zero RL**이다. 이는 비instruction-tuned base model에서 시작하여 GRPO로 바로 학습하는 방식이다. 여기서는 Qwen-2.5-7B와 Qwen-2.5-Math-7B를 사용하고, 정답이 맞으면 +1, 틀리면 -1 보상을 준다. 두 번째는 **RL from instruct models**이다. 이미 어느 정도 수학 추론 능력을 가진 instruction-tuned 모델에서 시작해 RL을 추가한다. 여기서는 R1-Distill-Qwen-1.5B와 OpenMath-Nemotron-1.5B를 기반 모델로 사용한다.

평가는 MATH500, AMC23, OlympiadBench, Minerva Math, AIME24, AIME25, English PolyMath, 그리고 비수학 일반화 평가를 위한 GPQA-Diamond로 구성된다. 모든 실험에서 지표는 16회 샘플 평균 pass@1이고, decoding 설정은 temperature=0.6, top-p=0.95, max tokens=32K로 통일한다. 논문은 스크립트 차이로 인한 편차를 줄이기 위해 비교 대상 baseline도 같은 설정으로 다시 평가했다고 명시한다.

### 4.2 Mathematical Reasoning Results

Table 3은 수학 벤치마크 전반의 핵심 결과다. 표는 proprietary models, zero RL from base model, RL from instruct models 세 블록으로 구성된다.

![Table 3](/assets/images/DeepMath-103K/DeepMath_103K_A_Large_Scale_Challenging_Decontaminated_and_Verifiable_Mathematical_Dataset_for_Advancing_Reasoning_11_Table_3.png)

먼저 proprietary block에서는 o1-mini가 AIME24에서 63.6, o3-mini(low effort)가 AIME24에서 60.0을 기록한다. 공개 비교 기준으로 쓰인 수치다.

**Zero RL from base model** 블록을 보면, Qwen-2.5-7B는 MATH500 54.8, AMC23 35.3, OlympiadBench 27.8, Minerva Math 16.2, AIME24 7.7, AIME25 5.4, PolyMath 28.1이다. 여기에 DeepMath-103K로 zero RL을 적용한 DeepMath-Zero-7B는 각각 85.5, 64.7, 51.0, 45.3, 20.4, 17.5, 42.7로 크게 오른다. Open-Reasoner-Zero-7B(81.8, 58.9, 47.9, 38.4, 15.6, 14.4, 40.7)나 Qwen-2.5-7B-SRL-Zoo(77.0, 55.8, 41.0, 41.2, 15.6, 8.7, 33.1)보다도 전반적으로 높다.

Qwen-2.5-Math-7B 계열에서도 같은 경향이 나타난다. 원본 모델은 46.9, 31.9, 15.8, 15.5, 11.2, 4.4, 22.7이지만, DeepMath-Zero-Math-7B는 86.9, 74.7, 52.3, 49.5, 34.2, 23.5, 46.6까지 올라간다. Qwen-2.5-Math-7B-SRL-Zoo, Qat-Zero-7B, Eurus-2-7B-PRIME보다도 높은 수치다. 즉 DeepMath-103K는 base model에서 reasoning ability를 “처음부터 끌어올리는” zero RL 시나리오에서 특히 강한 효용을 보인다.

**RL from instruct models** 블록에서도 이점은 유지된다. R1-Distill-Qwen-1.5B는 84.7, 72.0, 53.1, 36.6, 29.4, 24.8, 39.9인데, DeepMath-1.5B는 89.9, 82.3, 61.8, 42.5, 37.3, 30.8, 46.6으로 오른다. OpenMath-Nemotron-1.5B는 91.8, 90.5, 70.3, 26.3, 61.3, 50.6, 56.8인데, DeepMath-Omn-1.5B는 93.2, 94.2, 73.4, 28.3, 64.0, 57.3, 58.7로 더 높다. 특히 DeepMath-Omn-1.5B는 AIME24 64.0, AIME25 57.3으로 1.5B급 공개 모델 중 매우 강력한 결과를 보이며, AIME24 기준 o1-mini 63.6과 low-effort o3-mini 60.0도 넘어선다.

이 장의 핵심 해석은 분명하다. DeepMath-103K는 단순한 데이터 보강이 아니라, zero RL과 instruct RL 모두에서 일관된 성능 상승을 만들 수 있는 고품질 reasoning dataset이다.

### 4.3 Generalizable Reasoning beyond Mathematics

논문은 수학 성능만 보는 데 그치지 않고, GPQA-Diamond로 비수학 일반화를 측정한다. Table 4가 그 결과를 요약한다.

![Table 4](/assets/images/DeepMath-103K/DeepMath_103K_A_Large_Scale_Challenging_Decontaminated_and_Verifiable_Mathematical_Dataset_for_Advancing_Reasoning_12_Table_4.png)

Qwen-2.5-7B는 Biology 33.6, Physics 27.8, Chemistry 21.4, Overall 25.3인데, DeepMath-Zero-7B는 57.2, 53.0, 28.2, 41.7로 크게 오른다. Qwen-2.5-Math-7B는 32.2, 26.0, 21.1, 24.3이고, DeepMath-Zero-Math-7B는 47.4, 56.3, 26.0, 41.2다. Instruct RL에서도 R1-Distill-Qwen-1.5B 13.5, 36.2, 4.4, 19.1이 DeepMath-1.5B에서 18.1, 47.6, 12.2, 28.2로 오르고, OpenMath-Nemotron-1.5B 12.8, 23.5, 18.9, 20.3은 DeepMath-Omn-1.5B에서 17.1, 28.4, 21.5, 24.1로 오른다.

저자들은 이 일반화 이득을 데이터 다양성과 정제 품질에서 찾는다. Math StackExchange 같은 덜 정형화된 소스에서 가져온 문제는 표면적으로 수학 문제지만, 실제로는 복잡한 정보 해석, 조건 관리, 추론 경로 점검 같은 더 일반적인 reasoning skill을 요구한다. 따라서 그 위에서 학습한 모델이 생물, 물리, 화학 질문에서도 더 나은 추론을 보인다는 해석이다.

### 4.4 Analysis of Zero RL Using DeepMath-103K

Figure 9는 DeepMath-Zero-7B의 zero RL 훈련 중 어떤 현상이 나타나는지 분석한다. 패널 (a)는 학습이 진행될수록 rollout response length가 꾸준히 증가하는 모습을 보여 준다. 이는 모델이 더 길고 복잡한 추론 과정을 전개하게 되었음을 의미한다.

패널 (b)는 subgoal, verification, backtracking, enumeration 네 가지 cognitive behavior의 증가를 추적한다. 논문은 Gandhi et al.과 Zeng et al.의 방법을 따라 이 행동들을 계수한다. 학습이 진행될수록 특히 subgoal 설정과 verification, backtracking이 증가한다는 점은 모델이 단순히 장황해지는 것이 아니라, 목표를 쪼개고 중간 결과를 검산하며 틀린 경로를 되돌아가는 식의 reasoning pattern을 더 많이 드러낸다는 뜻으로 해석된다.

패널 (c)는 평가 벤치마크에서의 평균 응답 길이를 비교한다. Qwen-2.5-7B-SRL-Zoo는 약 1.1K, Open-Reasoner-Zero는 2.7K, Qwen-2.5-7B는 3.1K, DeepMath-Zero-7B는 5.9K 수준으로 나타난다. DeepMath-103K로 훈련한 모델이 훨씬 긴 추론을 수행한다는 점은, 이 데이터셋이 long reasoning model 연구에 적합한 자원이라는 논문의 해석으로 이어진다.

Figure 9의 캡션 문구는 도식 내용과 완전히 일치하지 않는 복사 흔적이 있지만, 본문 설명 기준으로 이 그림이 다루는 내용은 **응답 길이 증가**, **인지적 행동의 출현**, **테스트 시 긴 reasoning 경향**이다.

![Figure 9](/assets/images/DeepMath-103K/DeepMath_103K_A_Large_Scale_Challenging_Decontaminated_and_Verifiable_Mathematical_Dataset_for_Advancing_Reasoning_13_Figure_9.png)

## 5. Related Work

논문은 수학 추론 데이터 연구를 LLM post-training의 세 단계에 대응하는 세 줄기로 정리한다. 첫째는 continue pre-training(CPT)이며, OpenWebMath, MathPile, InfiMM-Web-Math, FineMath, MegaMath 같은 대규모 수학 텍스트 코퍼스가 여기에 속한다. 둘째는 SFT이며, MATH와 GSM8K 같은 고전적 데이터셋에서 시작해 MetaMathQA, OpenMathInstruct, NuminaMath-CoT, MMIQC, dart-math-hard, OpenMathReasoning처럼 더 크고 더 어려운 SFT 데이터셋들이 이어졌다. 셋째는 RLVR이며, Open-R1, ORZ-129K, DSR-Preview, Open-RS, DAPO-17K, BigMath 같은 검증 가능한 답 중심 데이터셋들이 최근 빠르게 등장했다.

이 맥락에서 DeepMath-103K는 “수학 RL 데이터셋”이라는 범주 안에서도 고난도, 강한 decontamination, verifiable answers, 그리고 다중 풀이까지 결합했다는 점에서 차별화된다고 정리된다.

## 6. Conclusion

저자들은 결론에서 DeepMath-103K를 RLVR을 통해 LLM의 reasoning capability를 끌어올리기 위해 특별히 설계된 대규모 수학 데이터셋으로 규정한다. 이 데이터셋의 강점은 어렵고, 깨끗하고, 검증 가능하고, 풀이 다양성까지 갖췄다는 데 있다. 또한 덜 정형화된 수학 포럼 데이터를 적극적으로 정제하여, 기존 공개 자원보다 더 큰 새로움과 다양성을 확보했다고 주장한다.

실험 결과는 이 주장을 뒷받침한다. DeepMath 시리즈 모델은 다수의 수학 벤치마크에서 새로운 SOTA 또는 그에 준하는 강력한 결과를 보이며, GPQA-Diamond에서도 일반화 이득을 보인다. 논문은 데이터셋, 코드, 모델 가중치를 공개함으로써 후속 reasoning 연구의 기반을 마련하려 한다.

## Appendix A. Contamination Analysis of Existing Datasets

부록 A는 기존 공개 RL 수학 데이터셋들의 오염 분석을 별도로 제시한다. 이 분석은 MATH500과의 중복을 보는 방식이며, 본문에서 사용한 의미적 decontamination보다 느슨한 **문자열 기반** 비교를 사용한다. 구체적으로 normalized indel similarity가 90%를 넘으면 오염 샘플로 본다. 느슨한 기준이기 때문에 절대적 의미의 오염 정의라기보다 “표면적으로 상당히 비슷한 문제”를 잡아내는 보조 분석에 가깝다.

Figure 10에 따르면 ORZ-129K는 223개, STILL-3-RL은 160개, DSR-Preview는 120개, DAPO-17K는 24개, Open-RS는 22개, Open-R1은 3개의 오염 샘플을 보인다. DeepMath-103K는 0개다. 이 결과는 본문 3장의 의미적 decontamination이 실제로 강력하게 작동했음을 뒷받침하는 추가 증거로 제시된다.

![Figure 10](/assets/images/DeepMath-103K/DeepMath_103K_A_Large_Scale_Challenging_Decontaminated_and_Verifiable_Mathematical_Dataset_for_Advancing_Reasoning_14_Figure_10.png)

## Appendix B. SFT Results

부록 B는 DeepMath-103K가 RL뿐 아니라 SFT에도 유용하다는 점을 보여 준다. 저자들은 Qwen-2.5-7B를 대상으로 각 문제마다 첫 번째 R1 풀이 하나만 쓰는 경우와, 세 개의 R1 풀이를 모두 쓰는 경우를 비교한다.

Table 5를 보면, 원본 Qwen-2.5-7B는 MATH500 54.8, AMC23 35.3, OlympiadBench 27.8, Minerva Math 16.2, AIME24 7.7, AIME25 5.4다. 여기에 **SFT with 1 R1 Solution**을 적용하면 69.2, 47.3, 35.9, 29.8, 12.3, 8.7로 오른다. **SFT with 3 R1 Solutions**은 74.1, 50.0, 40.2, 34.1, 13.8, 14.0으로 더 좋다. 즉 하나의 풀이만 써도 이득이 있지만, 세 개의 서로 다른 reasoning path를 모두 활용하면 추가 이득이 있다. 다만 RL counterpart인 DeepMath-Zero-7B(85.5, 64.7, 51.0, 45.3, 20.4, 17.5)가 여전히 가장 높기 때문에, 논문은 DeepMath-103K가 SFT에도 유용하지만 RL과 결합될 때 가장 큰 효과를 낸다고 정리한다.

![Table 5](/assets/images/DeepMath-103K/DeepMath_103K_A_Large_Scale_Challenging_Decontaminated_and_Verifiable_Mathematical_Dataset_for_Advancing_Reasoning_15_Table_5.png)

## Appendix C. Training Details

부록 C는 DeepMath 시리즈 모델들의 학습 구성을 정리한다. 프레임워크는 `verl`을 사용한다.

Table 6에 따르면 DeepMath-Zero-7B와 DeepMath-Zero-Math-7B는 둘 다 learning rate 1e-6, KL coefficient 0.0, train batch size 512, PPO mini-batch size 32, clip ratio low 0.20, clip ratio high 0.28, temperature 1.0, rollout.n 16, total steps 500을 사용한다. 차이는 max prompt length와 max response length, overlong buffer 길이다. Zero-7B는 prompt 2K, response 10K, overlong buffer 2K이고, Zero-Math-7B는 prompt 1K, response 3K, overlong buffer 512다.

반면 instruct RL 계열인 DeepMath-1.5B와 DeepMath-Omn-1.5B는 둘 다 learning rate 1e-6이지만 KL coefficient 1e-3을 사용하고, response length를 24K까지 늘린다. batch size는 128, PPO mini-batch는 64, temperature는 0.6이다. DeepMath-1.5B는 rollout.n 16, total steps 1800이고, DeepMath-Omn-1.5B는 rollout.n 18, total steps 700이다. 이 설정 차이는 base model zero RL과 instruction-tuned RL의 안정성 요구와 길어진 reasoning budget을 반영한다.

![Table 6](/assets/images/DeepMath-103K/DeepMath_103K_A_Large_Scale_Challenging_Decontaminated_and_Verifiable_Mathematical_Dataset_for_Advancing_Reasoning_16_Table_6.png)

## Appendix D. Licenses for Existing Assets

부록 D는 사용한 외부 자산들의 라이선스를 정리한다. 데이터 자산인 MMIQC, WebInstSub, NuminaMath-CoT는 모두 Apache 2.0이다. 코드 자산인 `verl`과 `NeMo-Skills`도 Apache 2.0이다. 모델 자산은 Qwen-2.5-7B가 Apache 2.0, R1-Distill-Qwen-1.5B가 MIT, OpenMath-Nemotron-1.5B가 CC BY 4.0이다.

이 정보는 단순 부록이 아니라 실용적 의미가 있다. DeepMath-103K를 실제로 재현하거나 파생 연구에 활용할 때 어떤 자산을 어떤 조건에서 쓸 수 있는지를 명확히 해 주기 때문이다.

![Table 7](/assets/images/DeepMath-103K/DeepMath_103K_A_Large_Scale_Challenging_Decontaminated_and_Verifiable_Mathematical_Dataset_for_Advancing_Reasoning_17_Table_7.png)

## Appendix E. Limitations and Broader Impacts

논문은 제한점도 분명히 언급한다. 첫째, 난도 평가는 GPT-4o 기반이므로 라벨 자체에 LLM bias가 들어갈 수 있다. 둘째, 주제 다양성이 넓다고 해도 완벽하게 균형 잡혀 있다고 장담할 수는 없다. 셋째, 데이터 구축 자체가 매우 큰 계산 자원을 요구했으므로 누구나 동일 파이프라인을 그대로 반복하기는 어렵다. 넷째, 저자들의 수작업 점검에 따르면 일부 judgment 또는 multiple-choice 유형 문제는 극단적으로 보면 운 좋게 답을 맞히는 경우가 있을 수 있다.

그럼에도 논문은 긍정적 파급효과가 더 크다고 본다. DeepMath-103K는 RL reasoning 연구의 진입 장벽을 낮추고, 더 어려운 문제를 대상으로 한 학습을 촉진하며, 오염이 적은 평가를 가능하게 하고, 결과적으로 더 일반화 가능한 AI를 만드는 데 기여할 수 있다.

## References

- **[1]** Alon Albalak, Duy Phung, Nathan Lile, Rafael Rafailov, Kanishk Gandhi, Louis Castricato, Anikait Singh, Chase Blagden, Violet Xiang, Dakota Mahan, and Nick Haber. Big-math: A large-scale, high-quality math dataset for reinforcement learning in language models, 2025. URL https://arxiv.org/abs/2502.17387.
- **[2]** Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Martı́n Blázquez, Guilherme Penedo, Lewis Tunstall, Andrés Marafioti, Hynek Kydlı́ček, Agustı́n Piqueres Lajarı́n, Vaibhav Srivastav, Joshua Lochner, Caleb Fahlgren, Xuan-Son Nguyen, Clémentine Fourrier, Ben Burtenshaw, Hugo Larcher, Haojun Zhao, Cyril Zakka, Mathieu Morlon, Colin Raffel, Leandro von Werra, and Thomas Wolf. Smollm2: When smol goes big – data-centric training of a small language model, 2025. URL https://arxiv.org/abs/2502.02737.
- **[3]** Daman Arora, Himanshu Singh, and Mausam. Have LLMs advanced enough? a challenging problem solving benchmark for large language models. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 7527–7543, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.468. URL https://aclanthology.org/2023.emnlp-main.468.
- **[4]** Hritik Bansal, Arian Hosseini, Rishabh Agarwal, Vinh Q. Tran, and Mehran Kazemi. Smaller, weaker, yet better: Training LLM reasoners via compute-optimal sampling. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=3OyaXFQuDl.
- **[5]** Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu, Mengfei Zhou, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, and Dong Yu. Do not think that much for 2+3=? on the overthinking of o1-like llms, 2024. URL https://arxiv.org/abs/2412.21187.
- **[6]** Zhipeng Chen, Yingqian Min, Beichen Zhang, Jie Chen, Jinhao Jiang, Daixuan Cheng, Wayne Xin Zhao, Zheng Liu, Xu Miao, Yang Lu, Lei Fang, Zhongyuan Wang, and Ji-Rong Wen. An empirical study on eliciting and improving r1-like reasoning models. arXiv preprint arXiv:2503.04548, 2025.
- **[7]** Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.
- **[8]** Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, et al. Process reinforcement through implicit rewards. arXiv preprint arXiv:2502.01456, 2025. Quy-Anh Dang and Chris Ngo. Reinforcement learning for reasoning in small llms: What works and what doesn’t, 2025. URL https://arxiv.org/abs/2503.16219.
- **[9]** Hugging Face. Open r1: A fully open reproduction of deepseek-r1, January 2025. URL https://github.com/huggingface/open-r1.
- **[10]** Meng Fang, Xiangpeng Wan, Fei Lu, Fei Xing, and Kai Zou. Mathodyssey: Benchmarking mathematical problem-solving skills in large language models using odyssey math data. arXiv preprint arXiv:2406.18321, 2024.
- **[11]** Kanishk Gandhi, Ayush Chakravarthy, Anikait Singh, Nathan Lile, and Noah D Goodman. Cognitive behaviors that enable self-improving reasoners, or, four habits of highly effective stars. arXiv preprint arXiv:2503.01307, 2025.
- **[12]** Bofei Gao, Feifan Song, Zhe Yang, Zefan Cai, Yibo Miao, Qingxiu Dong, Lei Li, Chenghao Ma, Liang Chen, Runxin Xu, Zhengyang Tang, Benyou Wang, Daoguang Zan, Shanghaoran Quan, Ge Zhang, Lei Sha, Yichang Zhang, Xuancheng Ren, Tianyu Liu, and Baobao Chang. Omnimath: A universal olympiad level mathematic benchmark for large language models, 2024. URL https://arxiv.org/abs/2410.07985.
- **[13]** Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.
- **[14]** Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- **[15]** Xiaotian Han, Yiren Jian, Xuefeng Hu, Haogeng Liu, Yiqi Wang, Qihang Fan, Yuang Ai, Huaibo Huang, Ran He, Zhenheng Yang, and Quanzeng You. Infimm-webmath-40b: Advancing multimodal pre-training for enhanced mathematical reasoning, 2024. URL https://arxiv.org/abs/2409.12568.
- **[16]** Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems, 2024.
- **[17]** Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. Proceedings of the International Conference on Learning Representations (ICLR), 2021a.
- **[18]** Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. NeurIPS, 2021b.
- **[19]** Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, and Heung-Yeung Shum Xiangyu Zhang. Openreasoner-zero: An open source approach to scaling reinforcement learning on the base model. https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero,2025.
- **[20]** Zhen Huang, Zengzhi Wang, Shijie Xia, Xuefeng Li, Haoyang Zou, Ruijie Xu, Run-Ze Fan, Lyumanshan Ye, Ethan Chern, Yixin Ye, Yikai Zhang, Yuqing Yang, Ting Wu, Binjie Wang, Shichao Sun, Yang Xiao, Yiyuan Li, Fan Zhou, Steffi Chern, Yiwei Qin, Yan Ma, Jiadi Su, Yixiu Liu, Yuxiang Zheng, Shaoting Zhang, Dahua Lin, Yu Qiao, and Pengfei Liu. Olympicarena: Benchmarking multi-discipline cognitive reasoning for superintelligent ai. arXiv preprint arXiv:2406.12753, 2024. URL https://arxiv.org/abs/2406.12753.
- **[21]** Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. Solving quantitative reasoning problems with language models. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems, volume 35, pp. 3843–3857. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper_files/paper/2022/file/18abbeef8cfe9203fdf9053c9c4fe191-Paper-Conference.pdf.
- **[22]** Jia LI, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Costa Huang, Kashif Rasul, Longhui Yu, Albert Jiang, Ziju Shen, Zihan Qin, Bin Dong, Li Zhou, Yann Fleureau, Guillaume Lample, and Stanislas Polu. Numinamath. [https://huggingface.co/AI-MO/NuminaMath-CoT](https://github.com/project-numina/aimo-progress-prize/blob/main/report/numina_dataset.pdf),2024.
- **[23]** Haoxiong Liu, Yifan Zhang, Yifan Luo, and Andrew Chi-Chih Yao. Augmenting math word problems via iterative question composing, 2024. URL https://arxiv.org/abs/2401.09003.
- **[24]** Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.
- **[25]** Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y. Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Tianjun Zhang, Li Erran Li, Raluca Ada Popa, and Ion Stoica. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl, 2025. Notion Blog.
- **[26]** MAA. American invitational mathematics examination (AIME). Mathematics Competition Series, n.d.a. URL https://maa.org/math-competitions/aime.
- **[27]** MAA. American mathematics competitions (AMC 10/12). Mathematics Competition Series, n.d.b. URL https://maa.org/math-competitions/amc.
- **[28]** Ivan Moshkov, Darragh Hanley, Ivan Sorokin, Shubham Toshniwal, Christof Henkel, Benedikt Schifferer, Wei Du, and Igor Gitman. Aimo-2 winning solution: Building state-of-the-art mathematical reasoning models with openmathreasoning dataset. arXiv preprint arXiv:2504.16891, 2025.
- **[29]** Keiran Paster, Marco Dos Santos, Zhangir Azerbayev, and Jimmy Ba. Openwebmath: An open dataset of high-quality mathematical web text, 2023. Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 11 2019. URL http://arxiv.org/abs/1908.10084.
- **[30]** David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. GPQA: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024. URL https://openreview.net/forum?id=Ti67584b98.
- **[31]** Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/2402.03300.
- **[32]** Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.
- **[33]** Qwen Team. Qwen2.5: A party of foundation models, September 2024. URL https://qwenlm.github.io/blog/qwen2.5/.
- **[34]** Yuxuan Tong, Xiwen Zhang, Rui Wang, Ruidong Wu, and Junxian He. DART-math: Difficulty-aware rejection tuning for mathematical problem-solving. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id=zLU21oQjD5.
- **[35]** Shubham Toshniwal, Wei Du, Ivan Moshkov, Branislav Kisacanin, Alexan Ayrapetyan, and Igor Gitman. Openmathinstruct-2: Accelerating ai for math with massive open-source instruction data. arXiv preprint arXiv:2410.01560, 2024a.
- **[36]** Shubham Toshniwal, Ivan Moshkov, Sean Narenthiran, Daria Gitman, Fei Jia, and Igor Gitman. Openmathinstruct-1: A 1.8 million math instruction tuning dataset. arXiv preprint arXiv: Arxiv2402.10176, 2024b.
- **[37]** Yiming Wang, Pei Zhang, Jialong Tang, Haoran Wei, Baosong Yang, Rui Wang, Chenshu Sun, Feitong Sun, Jiran Zhang, Junxuan Wu, Qiqian Cang, Yichang Zhang, Fei Huang, Junyang Lin, Fei Huang, and Jingren Zhou. Polymath: Evaluating mathematical reasoning in multilingual contexts. arXiv preprint arXiv:2504.18428, 2025a. URL https://arxiv.org/abs/2504.18428.
- **[38]** Yue Wang, Qiuzhi Liu, Jiahao Xu, Tian Liang, Xingyu Chen, Zhiwei He, Linfeng Song, Dian Yu, Juntao Li, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, and Dong Yu. Thoughts are all over the place: On the underthinking of o1-like llms, 2025b. URL https://arxiv.org/abs/2501.18585.
- **[39]** Zengzhi Wang, Xuefeng Li, Rui Xia, and Pengfei Liu. Mathpile: A billion-token-scale pretraining corpus for math. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024. URL https://openreview.net/forum?id=RSvhU69sbG.
- **[40]** Tianwen Wei, Jian Luan, Wei Liu, Shuang Dong, and Bin Wang. Cmath: Can your language model pass chinese elementary school math test?, 2023.
- **[41]** Longhui Yu, Weisen Jiang, Han Shi, Jincheng YU, Zhengying Liu, Yu Zhang, James Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=N8N0hgNDRt.
- **[42]** Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Weinan Dai, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. Dapo: An open-source llm reinforcement learning system at scale, 2025. URL https://arxiv.org/abs/2503.14476.
- **[43]** Xiang Yue, Tianyu Zheng, Ge Zhang, and Wenhu Chen. Mammoth2: Scaling instructions from the web. Advances in Neural Information Processing Systems, 37:90629–90660, 2024.
- **[44]** Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. Simplerlzoo: Investigating and taming zero reinforcement learning for open base models in the wild, 2025a. URL https://arxiv.org/abs/2503.18892.
- **[45]** Weihao Zeng, Yuzhen Huang, Wei Liu, Keqing He, Qian Liu, Zejun Ma, and Junxian He. 7b model and 8k examples: Emerging reasoning with reinforcement learning is both effective and efficient. https://hkust-nlp.notion.site/simplerl-reason, 2025b. Notion Blog.
- **[46]** Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. Agieval: A human-centric benchmark for evaluating foundation models, 2023.
- **[47]** Fan Zhou, Zengzhi Wang, Nikhil Ranjan, Zhoujun Cheng, Liping Tang, Guowei He, Zhengzhong Liu, and Eric P. Xing. Megamath: Pushing the limits of open math corpora. arXiv preprint arXiv:2504.02807, 2025. Preprint.

## Additional Explanation

이 논문이 가진 가장 큰 의미는 “좋은 RL 수학 데이터셋의 조건”을 구체적으로 분해해서 보여 준 데 있다. 많은 공개 데이터는 크기만 크거나, 정답 검증 가능성만 있거나, 혹은 대회 문제를 많이 담고 있다는 장점 하나에 기대는 경우가 많았다. DeepMath-103K는 그런 단일 장점 데이터와 다르게 **난도**, **청결성**, **정답 검증 가능성**, **주제 다양성**, **풀이 다양성**을 동시에 묶으려 한 시도다. 논문 전체를 따라가 보면 이 다섯 요소가 서로 보완 관계에 있다는 점이 분명해진다. 난도가 높아야 강한 학습 신호가 생기고, decontamination이 되어야 결과를 믿을 수 있으며, 정답이 검증 가능해야 RLVR이 성립하고, 주제와 풀이가 다양해야 특정 스타일에 과적합되지 않는다.

Figure 5와 Figure 6의 조합도 중요하다. 고유 문제 수가 많다는 것만으로는 데이터셋의 본질적 차이를 다 말할 수 없고, 반대로 임베딩 분포가 다르다는 것만으로는 실제 중복 감소 효과를 정량화하기 어렵다. 이 논문은 두 지표를 함께 제시함으로써 DeepMath-103K가 기존 데이터 재조합물과는 다른 분포적 성격을 갖는다는 점을 입체적으로 보여 준다. 다시 말해 “새 문제를 많이 담고 있다”와 “전체 분포도 다르다”를 동시에 입증하려고 한 셈이다.

Table 3과 Table 5를 함께 보면, 이 데이터셋의 가치가 단순히 RL reward에만 있는 것이 아니라는 점도 보인다. 세 개의 R1 풀이를 가진 구조 덕분에 SFT만으로도 baseline 대비 큰 향상이 나온다. 하지만 RL이 여전히 더 강하다는 결과는, DeepMath-103K가 “SFT에도 쓸 수 있는 RL 데이터셋”이라기보다 “RL을 가장 잘 지원하지만 SFT로도 상당히 유용한 데이터셋”이라고 이해하는 편이 정확함을 보여 준다.

Figure 9는 또 다른 측면에서 흥미롭다. DeepMath-103K가 가져오는 이득은 단순히 정답률 상승만이 아니라, 모델이 더 긴 추론을 수행하고, subgoal 설정과 검산, backtracking을 더 자주 드러내게 만든다는 것이다. 이는 이 데이터셋이 모델에게 단순 pattern matching이 아니라 실제 multi-step reasoning을 강요하고 있음을 시사한다. 동시에 지나치게 긴 추론이 항상 좋은 것은 아니므로, DeepMath-103K는 long reasoning 연구의 재료이기도 하고, overthinking/underthinking을 함께 연구할 실험장이라는 해석도 가능하다.

마지막으로 이 논문은 “좋은 reasoning model은 좋은 reward 설계만으로 나오지 않는다”는 점을 다시 확인해 준다. reward가 맞더라도 데이터가 너무 쉽거나, 평가와 오염되어 있거나, 정답 표현이 불안정하면 RL이 주는 이점을 충분히 끌어내기 어렵다. DeepMath-103K는 그 반대 사례를 제시한다. 데이터를 제대로 만들면 모델은 더 잘 풀고, 더 길게 생각하고, 더 넓은 영역으로 일반화할 수 있다는 것이 이 논문의 최종 메시지다.
