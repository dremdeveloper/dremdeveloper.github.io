# LLM 학습 레시피와 실전 인사이트

하이브리드 구조, 프리트레이닝 데이터 구성, 롱컨텍스트 확장, 포스트트레이닝 순서, DPO와 RL 데이터 설계, 소형 모델 안정화라는 여섯 축으로 다시 정리한 것이다.  
---

## 1. Nemotron-H
**논문**: *Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models*  
**링크**: https://arxiv.org/abs/2504.03624

![Nemotron-H 하이브리드 아키텍처](/assets/images/LLM%20학습%20레시피와%20실전%20인사이트/01_nemotron_h_architecture.png)

### 키포인트
이 논문은 하이브리드 Mamba/Transformer 구조를 왜 쓰는지, 프리트레이닝에서 SFT 성격의 데이터를 실제로 얼마나 적극적으로 섞는지, 그리고 큰 모델에서 작은 모델을 만드는 과정이 왜 별도 문제인지까지 한 번에 보여준다.  
특히 긴 reasoning trace가 늘어날수록 attention 중심 구조는 KV-cache와 지연시간 부담이 빠르게 커지는데, Nemotron-H는 attention 비중을 과감하게 줄이고 Mamba-2와 FFN 중심으로 재구성해 이 부담을 낮추려 한다. 그래서 이 논문은 단순히 “새 구조를 썼다”는 사례가 아니라, 긴 추론 구간에서 계산비용을 줄이면서 성능을 유지하려는 설계 의도를 읽기 좋은 자료다.  
또 하나 중요한 점은 프리트레이닝 데이터에 synthetic SFT-style token을 대규모로 포함했다는 점이다. SFT에 가까운 데이터를 프리트레이닝 블렌드에 넣는 전략이 실제 레시피로 이미 쓰이고 있다는 직접 근거가 된다. 작은 모델 생성도 단순 축소가 아니라 pruning, NAS, distillation을 묶은 별도 압축 파이프라인으로 다룬다.

### 확인된 내용
- 전체 층의 약 **8%만 self-attention**이고, 나머지는 **Mamba-2와 FFN**이 교대로 배치된다.
- 목표는 긴 reasoning workload에서 **추론 효율**을 높이는 것이다.
- 프리트레이닝 데이터에 **synthetic SFT-style tokens 230B**를 추가했다고 명시한다.
- **MiniPuzzle = pruning + NAS + distillation** 조합으로 56B를 47B로 압축한다.
- 포스트트레이닝에서는 reasoning trace가 있는 샘플과 reasoning을 벗긴 샘플을 함께 써서 **reasoning on/off 제어**를 학습한다.

### 읽을 때 붙잡을 포인트
이 논문에서 가장 실무적인 부분은 두 가지다.  
첫째, 프리트레이닝 단계에서 이미 instruction-like 신호를 섞고 있다는 점이다. 따라서 “이런 데이터를 PT에 넣어도 되나”라는 질문보다 “어느 비율에서 도움이 되고 어느 시점부터 과적합이나 스타일 편향이 생기나”가 더 현실적인 질문이 된다.  
둘째, 소형 모델화는 단순한 너비·깊이 축소가 아니라 압축 레시피 전체를 다시 설계해야 하는 문제로 다뤄진다. 작은 모델 성능이 아쉬울 때 단순 스케일 다운만 반복하는 방식이 왜 자주 막히는지 설명해 주는 사례다.

### 실험으로 옮기면
- PT blend에서 **synthetic SFT-style 데이터 비율**을 바꿔가며 ablation
- reasoning on/off **paired-example** 유무 비교
- 작은 모델 생성 시 **pruning only**와 **pruning + distillation** 비교

### 아직 확인이 더 필요한 부분
- reasoning trace를 **1K에서 hard truncate**한다는 수치는 현재 확인한 원문에서 직접 찾지 못했다. reasoning / non-reasoning 제어 자체는 확인되지만, 고정 컷오프 길이는 추가 확인이 필요하다.

---

## 2. NVIDIA Nemotron 3
**논문**: *NVIDIA Nemotron 3: Efficient and Open Intelligence*  
**링크**: https://arxiv.org/abs/2512.20856

![NVIDIA Nemotron 3 하이브리드 MoE 구조](/assets/images/LLM%20학습%20레시피와%20실전%20인사이트/02_nemotron3_hybrid_moe.png)

![NVIDIA Nemotron 3 롱컨텍스트 확장 결과](/assets/images/LLM%20학습%20레시피와%20실전%20인사이트/03_nemotron3_long_context.png)

### 키포인트
Nemotron 3는 하이브리드 MoE 구조, multi-token prediction, 초장문 컨텍스트 확장, 다중 환경 RL을 한 번에 묶어서 보여주는 자료다. 그래서 구조와 학습 스케줄을 따로 보지 않고, “긴 컨텍스트와 긴 생성에서 시스템 전체를 어떻게 설계하는가”라는 관점으로 읽는 편이 더 좋다.  
특히 이 논문이 흥미로운 지점은 롱컨텍스트 확장에서 흔히 떠올리는 staged increase를 반드시 전제하지 않는다는 점이다. Nano에서는 512K 길이의 CPT를 직접 수행했고, 이런 설정에서 굳이 8K, 16K, 32K 식으로 천천히 늘리는 절차가 필수적이지 않았다고 적는다.  
포스트트레이닝 쪽에서도 태스크를 잘게 나눠 순차적으로 붙이기보다 여러 RL 환경을 동시에 학습하는 접근을 택한다. 그래서 이 논문은 “joint RL이 언제 먹히는가”를 볼 때 좋은 대조군이 된다.

### 확인된 내용
- **Hybrid Mamba-Transformer MoE** 아키텍처를 사용한다.
- 최대 **1M tokens** 컨텍스트를 지원하도록 설계한다.
- Super/Ultra 모델에는 **multi-token prediction**을 넣는다.
- Nano는 CPT를 **512K sequence length**에서 수행했고, SFT는 **256K**에서 수행한다.
- CPT에서 **8K→16K→…→512K 같은 staged increase가 필요하지 않았다**고 직접 적는다.
- 포스트트레이닝은 태스크별 분리 stage보다 **여러 RL 환경을 동시에 학습**하는 방식을 채택한다.
- 이 방식이 더 안정적이고 reward hacking에 덜 취약하다고 주장한다.

### 읽을 때 붙잡을 포인트
이 논문은 롱컨텍스트를 늘릴 때 항상 단계적 확장이 정답이라는 가정을 약하게 만든다. 다만 여기서 바로 “staged training은 불필요하다”로 일반화하면 과하다. 더 정확한 해석은, 적어도 이 계열과 이 인프라에서는 direct high-length CPT가 가능했고, staged increase가 절대 조건은 아니었다는 정도다.  
또 하나는 joint RL 해석이다. 이 논문은 simultaneous RL이 더 안정적이라고 보지만, 이 결론은 reward 설계와 task mix가 잘 맞을 때 특히 강해진다. 뒤에서 볼 Cascade 2와 나란히 보면, 결국 핵심은 joint냐 cascade냐의 이념 대결이 아니라 능력 간섭을 어떤 방식으로 관리하느냐에 가깝다.

### 실험으로 옮기면
- **direct 256K/512K CPT**와 **staged CPT** 비교
- MTP head 유무에 따른 rollout 효율 측정
- **all-task simultaneous RL**과 **task-wise cascade RL** 비교

---

## 3. Nemotron-Cascade 2
**논문**: *Nemotron-Cascade 2: Post-Training LLMs with Cascade RL and Multi-Domain On-Policy Distillation*  
**링크**: https://arxiv.org/abs/2603.19220

![Nemotron-Cascade 2 포스트트레이닝 파이프라인](/assets/images/LLM%20학습%20레시피와%20실전%20인사이트/04_cascade2_pipeline.png)

![Nemotron-Cascade 2 MOPD 개요](/assets/images/LLM%20학습%20레시피와%20실전%20인사이트/05_cascade2_mopd.png)

### 키포인트
이 논문은 포스트트레이닝에서 순서가 왜 중요한지 가장 또렷하게 보여준다. instruction following, human alignment, long-context reasoning, code, SWE 같은 능력은 한 번에 다 좋아지지 않고, 어떤 능력을 먼저 밀어 올리느냐에 따라 다른 능력이 흔들릴 수 있다는 전제를 분명하게 둔다.  
특히 IF-RL을 앞에 두고, 그 뒤에 multi-domain RL과 RLHF를 배치하며, 중간에 teacher distillation으로 흔들린 성능을 복구하는 설계는 “한 번 좋아진 능력이 다음 단계에서 무너지면 어떻게 회복할 것인가”까지 포함하는 구조다. 단순히 순차 학습을 선호한다기보다, 능력 간 충돌을 줄이는 순서를 세밀하게 설계하고 있다는 점이 핵심이다.  
또한 RL이 reasoning trace를 짧게 만들고 math 성능을 해칠 수 있다고 직접 언급한다. 그래서 이 논문은 RL score만 보고 지나가기보다, trace length와 entropy를 같이 봐야 한다는 점을 분명하게 남긴다.

### 확인된 내용
- SFT 뒤에 **sequential, domain-wise Cascade RL**을 적용한다.
- 대표 ordering은 **Instruction-Following RL → Multi-domain RL → MOPD → RLHF → Long-context RL → Code RL → SWE RL**이다.
- **IF-RL을 앞에 둔 이유**를 명시한다.
  - IF-RL은 strict instruction adherence를 올리지만 human alignment를 해칠 수 있다.
  - 뒤의 RLHF는 instruction following을 크게 해치지 않는다.
  - 먼저 좋은 IF teacher를 만들면 이후 **MOPD teacher**로 쓰기 좋다.
- **MOPD**는 strongest intermediate teacher들을 모아 **reverse-KL 기반 token-level distillation**으로 capability drift를 복원한다.
- 일부 RLVR 학습은 **entropy를 줄이고 reasoning trace를 짧게 만들어 math 성능을 해칠 수 있다**고 적는다.
- IF-RL과 multi-domain RL에서는 **KL loss coefficient를 0**으로 둔다.

### 읽을 때 붙잡을 포인트
이 논문이 주는 메시지는 “순차 학습이 무조건 좋다”가 아니다. 더 중요한 메시지는 순서가 틀리면 잘하던 능력도 쉽게 무너질 수 있다는 점이다.  
또한 성능 복구를 GRPO나 RL을 더 돌려서 해결하기보다, 특정 시점의 teacher를 distillation에 다시 써서 capability drift를 복원하는 방식이 꽤 실용적으로 보인다. 이 부분은 regression이 잦은 파이프라인에서 특히 참고할 만하다.

### 실험으로 옮기면
- **IF → Math**와 **Math → IF** 비교
- **cascade only**와 **cascade + teacher distillation** 비교
- RL 중 **entropy / 평균 trace length / math score** 동시 로깅
- regression 발생 시 **teacher checkpoint distillation**으로 복구 가능한지 확인

---

## 4. OLMo 3
**논문**: *OLMo 3*  
**링크**: https://arxiv.org/abs/2512.13961

![OLMo 3 모델 플로우](/assets/images/LLM%20학습%20레시피와%20실전%20인사이트/06_olmo3_model_flow.png)

![OLMo 3 Delta Learning 테이블](/assets/images/LLM%20학습%20레시피와%20실전%20인사이트/07_olmo3_delta_learning_table.png)

![OLMo 3 길이 제어 분석](/assets/images/LLM%20학습%20레시피와%20실전%20인사이트/08_olmo3_length_control.png)

### 키포인트
OLMo 3는 DPO와 RL을 실제 운영 단계에서 어떻게 다루는지 가장 자세하게 보여주는 자료다. 모델 흐름을 base → think/instruct → DPO → RL까지 공개하고, DPO pair 품질을 볼 때 chosen의 절대 점수보다 chosen/rejected 사이의 contrast를 더 중시한다.  
이 점이 중요한 이유는, DPO 이후 RL이 기대만큼 오르지 않을 때 원인을 알고리즘에서만 찾기 쉽기 때문이다. 그런데 이 논문은 pair contrast가 약하거나, 길이 차이가 너무 커서 길이 편향이 생기거나, RL 데이터에 너무 쉬운 문제가 섞여 있으면 뒤 단계가 힘을 제대로 못 받는다고 읽히는 근거를 준다.  
또한 최종 checkpoint를 고를 때 평균 점수만 보지 않고 length analysis와 vibe test를 함께 사용한다. 수치상으로는 좋아 보여도 실제 응답 품질이 미묘하게 흐트러지는 경우를 걸러내려는 장치다.

### 확인된 내용
- base → think/instruct → DPO → RL까지 전체 model flow를 공개한다.
- **Delta Learning** 관점에서 DPO pair의 절대 품질보다 **chosen/rejected contrast**를 더 중요하게 본다.
- **chosen-only SFT는 오히려 해롭고**, DPO가 큰 개선을 만든다고 보고한다.
- **DPO 모델이 SFT 모델보다 RLVR의 더 좋은 시작점**이라고 정리한다.
- 7B Think에서는 RL 전에 **pass rate > 62.5%인 쉬운 문제를 제거**한다.
- chosen/rejected 길이 차이가 과도해지지 않도록 **100 tokens 제한**을 둔다.
- final RL checkpoint 선택에서 **평균 점수 + length analysis + vibe test**를 함께 본다.

### 읽을 때 붙잡을 포인트
이 논문은 DPO를 “좋은 답변을 더 많이 보여주는 단계”로 보는 관점을 교정해 준다. 핵심은 좋은 답변 자체보다 두 응답의 차이를 얼마나 선명하게 주느냐다.  
또 하나 눈여겨볼 부분은 길이 편향 관리다. chosen이 항상 훨씬 길고 rejected가 항상 짧다면, 모델은 내용보다 길이를 학습해 버릴 수 있다. 그러면 RL에 들어가서도 길게 말하는 방향으로만 보상이 쏠릴 수 있다. 그래서 DPO 뒤 RL 상승폭이 작을 때는 보상함수만 보기보다 pair 구성과 길이 분포를 먼저 확인하는 편이 자연스럽다.

### 실험으로 옮기면
- DPO pair에서 **contrast maximization** 적용
- **chosen/rejected 길이 차이 상한** 도입
- RL 전 **easy prompt filtering** 도입
- 정량 점수 외에 **length / vibe test / undesirable behavior** 체크리스트 추가

---

## 5. Ministral 3
**논문**: *Ministral 3*  
**링크**: https://arxiv.org/abs/2601.08584

![Ministral 3 학습 레시피](/assets/images/LLM%20학습%20레시피와%20실전%20인사이트/09_ministral3_training_recipe.png)

![Ministral 3 Long CoT 동작 특성](/assets/images/LLM%20학습%20레시피와%20실전%20인사이트/10_ministral3_long_cot_behavior.png)

![Ministral 3 ODPO 결과](/assets/images/LLM%20학습%20레시피와%20실전%20인사이트/11_ministral3_odpo.png)

### 키포인트
Ministral 3는 소형 모델, 특히 3B급 reasoning 모델이 왜 쉽게 흔들리는지 가장 선명하게 보여준다. 큰 모델에서는 어느 정도 버티는 설정도 작은 모델에서는 verbosity가 과해지거나 RL이 불안정해지면서 곧바로 깨질 수 있는데, 이 논문은 그런 구간을 구체적으로 드러낸다.  
특히 3B reasoning에서는 vanilla SFT만으로는 brittle했고, stronger teacher의 logit distillation이 안정화 수단으로 쓰였다. 이 점은 distillation을 단순 성능 향상 기법으로 보기보다, 작은 모델에서 학습을 성립시키기 위한 안전장치로 봐야 한다는 뜻에 가깝다.  
또한 instruct branch와 reasoning branch를 같은 레시피로 밀어붙이지 않고 분리해서 다루며, Long CoT 비율을 올릴수록 STEM 성능은 좋아지지만 일반 대화에서는 과도한 reflection과 backtracking이 늘어날 수 있다고 적는다. reasoning 성능과 대화 자연스러움을 동시에 챙기기 어렵다는 점을 잘 보여준다.

### 확인된 내용
- Mistral Small 3.1에서 시작해 **Cascade Distillation**으로 **14B → 8B → 3B**를 만든다.
- instruct branch는 **SFT + ODPO**를 사용한다.
- reasoning branch는 **SFT(with CoT) + GRPO + ODPO**를 사용한다.
- 3B reasoning에서는 vanilla SFT가 brittle했고, **Magistral Small 1.2 teacher logit distillation**으로 verbosity를 줄이고 RL을 안정화했다.
- reasoning RL 순서는 **STEM RL → General RL**이다.
- Long CoT 비율을 늘리면 STEM 성능은 좋아지지만, 일반 채팅 관점에서는 **과도한 reflection / internal monologue / backtracking**이 생긴다.
- ODPO를 reasoning 모델 위에 올리면 **14B/8B는 alignment benchmark가 크게 좋아진다**. 3B는 public benchmark 개선이 약하지만 internal human eval은 좋아진다.

### 읽을 때 붙잡을 포인트
3B 구간에서는 같은 기법이라도 훨씬 예민하게 반응한다. reasoning trace 길이, teacher 품질, ODPO 강도 같은 요소가 조금만 흔들려도 전체 출력 스타일이 바로 무너질 수 있다.  
그래서 이 논문은 reasoning과 instruct를 너무 빨리 합치기보다, 각각 안정화한 뒤 합류시키는 전략이 왜 필요한지 잘 보여준다. 또 Long CoT를 무작정 늘리는 것이 늘 좋은 방향이 아니라는 점도 중요하다. STEM 점수는 오르더라도 사용자 체감 대화 품질은 쉽게 나빠질 수 있다.

### 실험으로 옮기면
- 3B에서 **teacher logit distillation 유무** 비교
- **STEM → General**과 **General → STEM** 순서 비교
- Long CoT ratio sweep 후 **STEM 점수 / chat 자연스러움** 동시 평가
- reasoning checkpoint에 **ODPO 추가 적용** 후 alignment recovery 측정

---

## 6. Magistral
**논문**: *Magistral*  
**링크**: https://arxiv.org/abs/2506.10910

![Magistral RL 파이프라인](/assets/images/LLM%20학습%20레시피와%20실전%20인사이트/12_magistral_pipeline.png)

### 키포인트
Magistral은 RL 레시피를 운영 관점에서 읽기 좋은 논문이다. 여기서는 모델을 어떻게 크게 바꿨는가보다, 학습을 실제로 굴릴 때 어떤 제어변수가 자주 문제를 일으키는가가 더 잘 보인다.  
특히 difficulty filtering, KL 제거, 최대 생성 길이의 점진적 증가, 그리고 너무 쉽거나 너무 어려운 문제를 동시에 피하는 goldilocks difficulty 개념이 핵심이다. 즉, RL을 한 덩어리 알고리즘으로 보지 않고 데이터 난이도, 응답 길이, 시스템 안정성을 함께 조절해야 하는 운영 문제로 본다.  
그래서 이 논문은 “점수가 안 오른다”는 현상을 볼 때 보상식만 바꾸기보다, 현재 데이터 난이도가 정책 수준에 맞는지, 길이 스케줄이 과도하지 않은지, KL이 실제로 필요한지부터 다시 보게 만든다.

### 확인된 내용
- **Magistral Medium**은 reasoning model distillation 없이 **pure RL**로 훈련한다.
- **Magistral Small**은 Medium이 만든 traces로 cold-start SFT 뒤 RL을 진행한다.
- GRPO 수정점으로 아래를 둔다.
  - **KL divergence 제거**
  - loss normalization
  - minibatch advantage normalization
  - Clip-Higher 스타일 trust region 상한 완화
  - zero-advantage group 제거
- difficulty filtering은 2단계다.
  - 약한 초기 모델로 1차 filtering
  - 더 강해진 RL 모델로 2차 re-grading
- 목표는 **goldilocks difficulty**, 즉 너무 쉽지도 너무 어렵지도 않은 문제대다.
- training curriculum에서 최대 completion length를 **16K → 24K → 32K**로 늘린다.
- **length penalty**도 둔다.

### 읽을 때 붙잡을 포인트
이 논문은 curriculum을 단순한 easy-to-hard 순서로 설명하지 않는다. 난이도, 생성 길이, 계산 자원, policy 안정성이 서로 얽혀 있다는 쪽에 가깝다.  
또한 KL off는 특이한 실험 아이디어가 아니라 실제 선택지로 제시된다. Cascade 2와 같이 읽으면, KL을 강하게 주는 것이 반드시 안전한 기본값은 아니라는 점도 자연스럽게 보인다.

### 실험으로 옮기면
- RL 세팅에서 **KL on/off ablation**
- **difficulty curriculum** 도입
- **max completion length schedule** 도입
- reward 외에 **group entropy / length plateau**를 control signal로 같이 사용

---

## 주제별로 다시 묶어 보면

### 하이브리드 Mamba/Transformer 구조는 언제 설득력이 생기나
Nemotron-H와 Nemotron 3가 공통으로 보여주는 방향은 분명하다. 긴 컨텍스트와 긴 reasoning trace가 자주 등장하는 환경에서는 attention을 전부 유지하는 방식이 메모리와 지연시간 측면에서 부담이 크다. 하이브리드 구조는 이 문제를 줄이면서도 필요한 위치에는 attention을 남기는 절충안으로 읽힌다.  
즉, 이 구조를 쓰는 이유는 새로운 블록을 도입하는 실험적 재미보다, 긴 추론에서 토큰당 비용과 KV-cache 부담을 줄여 실제 처리량을 확보하려는 데 가깝다. 구조적 이점이 필요한지 보려면 같은 크기의 순수 Transformer와 throughput, memory footprint, 장문 응답 품질을 함께 비교해야 한다.

### 프리트레이닝에서 SFT 성격의 데이터를 섞는 문제
Nemotron-H는 synthetic SFT-style token을 프리트레이닝 데이터에 대규모로 포함했다고 직접 밝힌다. 이 한 줄만으로도, instruction-like 데이터를 PT 블렌드에 넣는 방식은 더 이상 주변적인 아이디어가 아니라 실제로 사용된 레시피라고 볼 수 있다.  
중요한 것은 넣느냐 마느냐보다, 어느 비율에서 도움이 되는지와 어떤 부작용이 생기는지다. 비율이 높아질수록 초기 helpfulness나 format adherence는 좋아질 수 있지만, 반대로 스타일이 과도하게 고정되거나 데이터 다양성이 줄어드는 문제가 생길 수도 있다. 따라서 이 쟁점은 찬반보다 blend 설계 문제로 다루는 편이 맞다.

### 롱컨텍스트는 단계적으로만 늘려야 하나
Nemotron 3는 direct high-length CPT가 가능하다는 사례를 준다. 최소한 이 논문에서는 512K 컨텍스트 학습을 위해 반드시 8K, 16K, 32K 식의 단계별 길이 확장이 필요하지 않았다고 적는다.  
다만 이 결론을 모든 모델에 그대로 옮기기는 어렵다. optimizer 안정성, 배치 구성, 데이터 품질, 하드웨어 조건이 다르면 staged schedule이 다시 유리해질 수 있다. 그래서 가장 적절한 해석은 “단계적 확장이 유일한 정답은 아니다” 정도다. 실제로는 direct setup과 staged setup을 같은 자원 조건에서 직접 비교하는 것이 가장 깔끔하다.

### Joint RL과 Cascade RL은 어느 쪽이 더 맞나
Nemotron 3는 여러 RL 환경을 동시에 학습하는 쪽을 선호하고, Cascade 2는 domain-wise 순차 학습과 teacher distillation을 조합한다. 표면적으로는 두 논문이 상반돼 보이지만, 실제로는 둘 다 능력 간섭을 어떻게 관리할지에 대한 답을 내고 있다고 보는 편이 정확하다.  
reward 설계가 안정적이고 task mix가 자연스럽게 섞이면 joint RL이 더 단순하고 강할 수 있다. 반대로 특정 능력을 먼저 올리면 다른 능력이 쉽게 무너지거나, 다음 단계에서 regression이 반복된다면 cascade order와 중간 teacher 복구가 훨씬 유용할 수 있다. 결국 중요한 것은 철학이 아니라 간섭의 형태다.

### DPO 뒤 RL 개선폭이 작을 때 먼저 볼 것
OLMo 3가 가장 실무적인 기준을 준다. 이 경우에는 우선 알고리즘을 의심하기보다 데이터와 선택 로직을 점검하는 편이 맞다.  
먼저 DPO pair의 contrast가 충분히 큰지 봐야 한다. chosen과 rejected의 차이가 약하면 모델은 무엇을 선호해야 하는지 선명하게 배우지 못한다. 그다음에는 길이 편향을 확인해야 한다. 좋은 답이 늘 더 길고 나쁜 답이 늘 더 짧다면, 모델이 내용 대신 길이를 학습할 수 있다. RL 데이터의 난이도도 중요하다. 너무 쉬운 문제는 학습 신호를 약하게 만들고, 너무 어려운 문제는 실패만 반복하게 만든다. 마지막으로 checkpoint selection 단계에서 평균 점수만 보는 습관도 위험하다. OLMo 3가 vibe test와 length analysis를 같이 둔 이유가 바로 여기에 있다.

### 3B급 소형 모델에서는 무엇이 특히 예민한가
Ministral 3를 보면 3B reasoning 모델은 큰 모델과 전혀 다른 난이도로 다뤄야 한다. teacher 품질, trace 길이, ODPO 강도, branch 병합 시점이 조금만 흔들려도 출력 스타일과 안정성이 금방 무너질 수 있다.  
그래서 distillation은 선택적인 보정이 아니라 학습을 성립시키는 장치로 읽히기도 한다. 또한 reasoning branch와 instruct branch를 한꺼번에 묶기보다 따로 안정화한 뒤 합치는 편이 더 안전해 보인다. Long CoT 역시 작은 모델에서는 특히 조심스럽다. STEM 점수는 좋아질 수 있지만, 사용자 관점의 응답 품질은 쉽게 거칠어질 수 있다.

### KL, 난이도, 길이 제어가 계속 등장하는 이유
Cascade 2와 Magistral을 같이 보면, RL 품질은 보상함수만으로 결정되지 않는다. KL 계수, 데이터 난이도 분포, 최대 생성 길이, entropy 변화가 모두 결과를 크게 바꾼다.  
특히 Magistral의 goldilocks difficulty는 좋은 RL 데이터가 단순히 어려운 데이터와 같지 않다는 점을 잘 보여준다. 너무 쉬우면 배울 것이 적고, 너무 어려우면 실패만 누적된다. 또 길이를 한 번에 크게 열기보다 16K에서 24K, 다시 32K로 올리는 방식은 모델 품질뿐 아니라 시스템 부담 관리와도 연결된다. 이 영역은 알고리즘보다 운영 레시피의 색채가 강하다.

### 아직 단정하기 어려운 부분
- Nemotron-H에서 reasoning trace를 **정확히 어느 길이에서 hard truncate하는지**는 현재 확인한 본문으로는 분명하지 않다.
- concat 기반 long-context 학습에서 **샘플 간 masking 처리**가 어떻게 명시되는지는 논문마다 공개 수준 차이가 있다.
- Nemotron 3의 **multi-environment RL 데이터 비율**은 큰 방향은 보이지만 세부 blend는 제한적으로만 공개된다.
- Ministral 3에서 **ODPO가 3B public benchmark에는 약하게 드러나는 이유**는 시사점은 있지만 원인 분해까지 완전히 제공되지는 않는다.

---

## 함께 보면 좋은 자료

### Generalizing Verifiable Instruction Following
**링크**: https://arxiv.org/abs/2507.02833  
Instruction following을 검증 가능한 형태로 다루며, IFBench와 RLVR 기반 개선 해석에 도움이 된다. Cascade 2의 IF-RL 관련 결과를 읽을 때 같이 보면 맥락이 또렷해진다.

### The Delta Learning Hypothesis: Preference Tuning on Weak Data can Yield Strong Gains
**링크**: https://arxiv.org/abs/2507.06187  
OLMo 3에서 강조하는 delta learning 관점을 더 직접적으로 설명한다. chosen의 절대 품질보다 chosen/rejected 차이가 중요하다는 해석을 보강하는 데 유용하다.

### Better & Faster Large Language Models via Multi-Token Prediction
**링크**: https://arxiv.org/abs/2404.19737  
Nemotron 3의 MTP 해석을 조금 더 일반적인 관점에서 이해하는 데 도움이 된다. 품질 개선과 생성 효율을 동시에 보려는 시도를 비교해 보기 좋다.

---
