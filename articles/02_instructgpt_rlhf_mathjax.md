    ---
    title: "Ouyang et al. (2022) 확장 해설본"
    math: true
    ---

    <script>
window.MathJax = {
  tex: {
    inlineMath: [['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    processEscapes: true
  },
  svg: { fontCache: 'global' }
};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>

> 편집 원칙  
> - 수식은 MathJax 기준으로 다시 정리했습니다.  
> - Figure/Table은 실제 이미지를 직접 삽입하지 않고, **어떤 원문 도판을 넣어야 하는지**를 안내하는 placeholder로 통일했습니다.  
> - 이해 점검 질문, 스터디 체크리스트, 로드맵 등 부속 학습 패키지는 제거해 **논문 해설 본문 중심**으로 재구성했습니다.  


## 전문가 관점 요약

InstructGPT 논문은 오늘날 RLHF 계열 정렬 파이프라인의 기준 문헌으로 읽힌다. 그러나 이 논문의 핵심은 “사람이 선호하는 답을 만들었다”는 선언보다도, **사전학습 언어모델의 목표와 실제 사용자 목표 사이의 간극을 운영 가능한 학습 파이프라인으로 번역했다**는 데 있다. 저자들은 이를 SFT → Reward Model → PPO라는 3단계로 구현했으며, 각 단계는 서로 다른 실패 모드를 줄이기 위한 역할 분담을 가진다.

전문가 관점에서 중요한 독해 포인트는 세 가지다. 첫째, 이 논문은 단순히 성능 향상을 보고하는 것이 아니라, **alignment objective를 어떻게 데이터화할 것인가**라는 문제를 다룬다. 즉, “좋은 답변”이란 무엇인지에 대한 라벨러 지침, 프롬프트 분포, 비교 기준 자체가 모델 품질의 일부가 된다. 둘째, PPO-ptx가 보여 주듯 RLHF는 보상을 올리는 방향으로만 가면 충분하지 않다. 사전학습 분포를 완전히 버리면 일반 언어능력이 훼손될 수 있어 **alignment tax**가 발생한다. 셋째, 평가 역시 절대적 진실이 아니라 **누가, 어떤 지침으로, 어떤 분포에서 평가하는가**의 함수다.

이 논문을 과장 없이 읽으려면 저자들이 **무엇을 주장하지 않는지**도 봐야 한다. 저자들은 보편적 인간 가치 정렬을 달성했다고 주장하지 않으며, 라벨러 집단의 선호를 특정한 프롬프트 분포 위에서 더 잘 따르도록 만들었다고 본다. 따라서 이 논문은 철학적 최종해라기보다, **선호 기반 정렬을 위한 현실적 엔지니어링 프레임**으로 이해하는 것이 정확하다.

이후의 DPO, RLAIF, constitutional AI, direct alignment 방법론을 이해하려면 이 논문을 기준점으로 삼아야 한다. 즉, InstructGPT는 “왜 reward model이 필요한가?”, “왜 KL 제약이 필요한가?”, “왜 선호 데이터가 capability와 safety를 동시에 흔드는가?”를 설명하는 출발점이다. 이 문서를 읽을 때는 각 설계 요소가 어떤 실패를 막기 위한 것인지 계속 추적해 보라.


# Training Language Models to Follow Instructions with Human Feedback
## 확장 해설본 (InstructGPT / RLHF)
- 논문: Long Ouyang et al., 2022  
- arXiv: 2203.02155 (v1: 2022-03-04)  
- 링크: https://arxiv.org/abs/2203.02155 (PDF: https://arxiv.org/pdf/2203.02155)

본 문서는 위 논문 **원문을 기반으로** 작성한 “확장 해설(annotated expansion)”이다.  
논문에 직접 근거가 있는 서술은 **【원문 근거】**, 원문 밖 배경 설명·직관·예시는 **【추가 설명(일반 지식)】**으로 구분하여 표기한다.  
(근거 표시는 원문 섹션/그림/표/부록 번호를 가능한 한 함께 적는다.)

---

## 논문 전체 구조 지도

이 논문이 일관되게 답하려는 질문은 다음 네 가지로 정리된다.

1. **왜** 사전학습된 대규모 언어모델은 사용자 의도(user intent)를 안정적으로 따르지 못하는가?  
2. **어떻게** 사람 피드백을 통해 instruction-following 행동을 학습시키는가?  
3. 그 결과가 **실제 사용자 프롬프트 분포**에서 유의미한 품질 개선으로 나타나는가?  
4. 이 과정에서 “정렬(alignment)”은 **누구의 선호에** 맞춰지며, 어떤 비용/리스크가 남는가?  

(근거: Abstract, Section 1, Section 5)

---

# 1. Abstract 해설

## A. 섹션 개요
- **핵심 문제:** 사전학습 LM이 유용한 능력을 보이더라도, 실제 사용 맥락에서는 “도움됨/진실성/무해성”이 보장되지 않는다.  
- **핵심 접근:** 사람 시연(demonstration)과 선호 비교(preference comparison)를 이용한 후단 학습.  
- **핵심 결과:** 인간 평가에서 선호도가 개선되며, 진실성·독성 관련 지표에서 개선이 관찰된다. 동시에 단순 실수와 안전성 한계를 명시한다.  
(근거: Abstract)

## B. 원문 전개(재구성)
**【원문 근거】**  
Abstract는 다음 논리로 구성된다.

- 사전학습 언어모델은 다양한 작업을 수행할 수 있지만, 실제 사용에서는 **허위 생성(untruthful), 독성(toxic), 비도움(not helpful)** 같은 의도치 않은 행동이 나타난다.  
- 저자들은 사전학습 목표(웹 텍스트 다음 토큰 예측)와 실제 목표(사용자 지시를 도움이 되고 안전하게 따르기) 사이의 불일치를 핵심 원인으로 본다.  
- 이를 완화하기 위해, (i) 라벨러가 작성한 프롬프트 및 API 프롬프트를 수집하고, (ii) 라벨러가 “바람직한 응답”을 작성한 데이터로 **지도 미세조정(SFT)** 을 수행한 뒤, (iii) 모델 출력에 대한 **선호 순위 데이터**로 **보상 모델(RM)** 을 학습하고, (iv) RM을 보상으로 **강화학습(RLHF; PPO)** 을 수행한다.  
- 이로써 얻은 모델을 InstructGPT라 부르며, 인간 평가에서 선호 개선을 보이고, truthfulness 향상 및 toxic 출력 감소를 보고한다. 다만 단순 실수와 안전성 한계는 남아 있다고 명시한다.  
(근거: Abstract)

## C. 해설 및 직관
**【추가 설명(일반 지식)】**  
Abstract가 제시하는 연구 프레임은 “능력(capability)”과 “행동 정렬(behavioral alignment)”을 분리하여 다룬다는 점에 있다. 사전학습은 언어적·추론적 능력을 부여하지만, 사용자가 원하는 응답 양식/우선순위(예: 도움이 되되 무해하게)를 자동으로 내재화하지 않는다. 본 논문은 후단 학습을 통해 그 행동 규범을 명시적으로 학습시키는 설계를 제안한다.

---

# 2. Introduction 해설

## A. 섹션 개요
- **문제정의:** 사전학습 LM의 목표 함수와 배포 환경의 목표 함수가 다르며, 이로 인해 의도치 않은 행동이 발생한다.  
- **정렬 개념화:** 사용자 의도를 “명시적 지시 준수 + 암묵적 기대(진실성/무해성 등)”로 확장해 정의한다.  
- **평가 철학:** 공개 NLP 벤치마크만으로는 실제 사용 분포의 품질을 대표하기 어렵다는 전제를 둔다.  
(근거: Section 1)

## B. 원문 전개(재구성)
**【원문 근거】**  
서론은 다음을 강조한다.

1. 대규모 언어모델은 프롬프팅으로 다양한 작업을 수행하지만, **사실을 만들어내거나**, **편향/독성 텍스트를 생성하거나**, **사용자 지시를 따르지 않는** 문제가 발생한다.  
2. 이는 사전학습 목표(웹 문서 다음 토큰 예측)가 “사용자 지시를 도움이 되고 안전하게 따르기” 목표와 다르기 때문이며, 저자들은 이를 **misaligned** 라고 표현한다.  
3. 정렬(alignment)은 “사용자 의도(user intentions)에 부합하는 행동”으로 정의되며, Askell et al.의 표현을 빌려 **helpful / honest / harmless** 프레임을 사용한다. 단, honest는 직접 측정이 어려워 본 논문에서는 truthfulness와 hallucination 감소로 일부만 포착한다고 한다.  
4. 본 논문은 RLHF를 사용하여 GPT-3를 광범위한 written instruction에 맞춰 fine-tune하는 접근을 제시한다(도식: Figure 2). 또한 이 과정은 특정 집단(라벨러/연구자)의 선호에 맞춘 정렬이며 “보편적 인간 가치”로 일반화되지 않는다는 점을 서론에서부터 분명히 한다.  
(근거: Section 1, Figure 2)

## C. 해설 및 직관
**【추가 설명(일반 지식)】**  
서론의 실질적 기여는 “LM 성능”을 단일 축으로 보지 않고, **배포 환경의 사용자 경험/위험**을 중심으로 목표를 재정의했다는 점이다. 즉, 모델이 어떤 벤치마크를 얼마나 맞히는지보다, 실제 사용에서 **의도 준수(Instruction following), 통제 용이성(Constraint following), 사실성(Hallucination 감소), 유해성(Toxicity/Bias 완화)** 같은 행동 특성이 어떻게 변화하는지를 분해해 측정한다.

---

# 3. Related Work 해설

## A. 섹션 개요
- RLHF 계열(특히 텍스트 요약에서의 사람 피드백 학습)과의 연속성  
- 공개 NLP 데이터 기반 instruction tuning(FLAN/T0 계열)과의 대비  
- harm(독성/편향/프라이버시/오정보) 평가 벤치마크의 맥락  
(근거: Section 2)

## B. 분류 체계로 재구성
**【원문 근거】**  
논문이 인접 연구를 묶는 방식은 다음과 같이 재정리할 수 있다.

| 범주 | 원문이 언급하는 선행 흐름 | 본 논문과의 관계 |
|---|---|---|
| RLHF / human feedback | 로봇·게임·요약 등에서 선호 학습 및 RL 적용 | RLHF를 **광범위한 자연어 태스크 분포**에 적용 |
| Instruction tuning / cross-task generalization | 다수 공개 NLP 태스크에 instruction prefix를 부착해 미세조정(FLAN, T0 등) | 공개 데이터 기반 접근과 달리 **실제 API 프롬프트 분포**에 초점을 둠 |
| Harms 평가 | toxicity, stereotype, bias, privacy leakage, misinformation 등 | harms는 문맥 의존적이므로 proxy metric과 사람 평가를 병행 |
| 행동 제어 기법 | 데이터 필터링, control token, 블로킹, 별도 모델로 steering 등 | RLHF를 중심 축으로 두되 상보적 관계로 위치시킴 |

(근거: Section 2)

## C. 원문 전개(재구성)
**【원문 근거】**  
저자들은 RLHF를 텍스트 요약에 적용한 선행 연구(Stiennon et al.)를 중요한 전 단계로 두고, 이를 더 넓은 태스크 분포로 확장한다. 또한 공개 NLP 데이터로 instruction tuning을 수행한 연구 계열과 비교하며, 본 논문의 핵심은 **“어떤 분포에 대해” 정렬하는가**(public NLP tasks vs 실제 사용자 프롬프트 분포)의 차이에 있음을 강조한다. harms 평가에 관해서는 문헌을 인용하며, 특정 개입이 부작용을 가져올 수 있음을 언급한다.  
(근거: Section 2)

## D. 해설 및 직관
**【추가 설명(일반 지식)】**  
본 논문은 “instruction following”이라는 표면적 목표가 유사하더라도, (i) 학습 신호(정답/시연 vs 선호 비교), (ii) 목표 분포(public task mixture vs product prompts), (iii) 평가 방식(자동평가 중심 vs 인간 선호 중심)에 따라 결과가 실질적으로 달라질 수 있음을 명시적으로 실험한다는 점에서 구분된다.

---

# 4. Methods and Experimental Details 해설

---

## 4.1 Section 3.1 — High-level methodology

### A. 섹션 개요
- 파이프라인은 **(1) SFT, (2) RM 학습, (3) PPO 기반 RLHF**의 3단계로 제시된다.  
- Step 2–3은 반복 가능한 루프(정책 갱신 → 비교 데이터 추가 수집 → RM 갱신)로 기술된다.  
(근거: Section 3.1, Figure 2)

### B. 원문 전개(재구성)
**【원문 근거】**  
Figure 2에 따라 방법은 다음과 같다.

1. **Demonstration 수집 → SFT**  
   라벨러가 프롬프트에 대한 바람직한 응답을 작성한다. 이 데이터를 사용해 pretrained GPT-3를 supervised fine-tuning 한다.
2. **Comparison 수집 → Reward Model 학습**  
   동일 프롬프트에 대한 여러 모델 출력을 라벨러에게 제시하고, 더 나은 출력을 선택하게 하여 선호 비교 데이터를 수집한다. 이를 사용해 RM을 학습한다.
3. **PPO 기반 강화학습(RLHF)**  
   RM을 보상 함수로 두고, SFT 정책을 초기 정책으로 하여 PPO로 추가 미세조정한다.

또한 비교 데이터는 supervised policy에서 주로 생성되지만, 일부는 PPO 정책에서도 생성될 수 있으며, 이로써 반복적으로 개선 가능한 루프를 구성한다.  
(근거: Section 3.1, Figure 2)

### C. 해설 및 직관
**【추가 설명(일반 지식)】**  
3단계 구성은 기능적으로 분리된다. SFT는 “기본적으로 적절한 응답”의 초기화를 제공하고, RM은 “여러 응답 중 무엇이 더 낫나”라는 선호 함수를 근사하며, PPO는 그 선호 함수를 정책 업데이트로 연결한다. 이 분해가 없으면, 선호 학습을 안정적으로 수행하기 어렵거나, 반대로 정책이 보상 모델을 과도하게 exploit할 가능성이 커진다.

### D. 의사코드(해설용)
**【추가 설명(일반 지식)】**

```text
Initialize pretrained model π0

# Step 1: supervised fine-tuning (SFT)
Collect (x, y*) demonstrations
Train π_SFT to maximize log π(y*|x)

# Step 2: reward model (RM)
For each x, sample K candidate outputs {y1..yK}
Collect human ranking / pairwise preferences
Train rθ(x, y) so that rθ(x, yw) > rθ(x, yl)

# Step 3: RLHF (PPO)
Initialize π_RL ← π_SFT
Repeat:
  Sample x ~ prompt distribution
  Sample y ~ π_RL(.|x)
  Reward ← rθ(x, y) - β KL(π_RL || π_SFT)   (plus pretraining mix for PPO-ptx)
  Update π_RL with PPO to maximize expected reward
```

---

## 4.2 Section 3.2 — Dataset

### A. 섹션 개요
- 데이터 출처: OpenAI API(특히 Playground)에서 수집된 프롬프트 + 라벨러가 작성한 bootstrap 프롬프트  
- 분할: 사용자 ID 단위로 train/valid/test 분리  
- 분포: 공개 NLP 벤치마크와 달리 **open-ended generation/brainstorming** 비중이 큼  
(근거: Section 3.2, Table 1, Table 6, Appendix A)

### B. 원문 전개(재구성)
**【원문 근거】**  
프롬프트는 주로 OpenAI API Playground에서 초기 InstructGPT 모델을 사용하던 고객이 입력한 텍스트 프롬프트로부터 수집된다. 저자들은 production API 데이터는 사용하지 않았다고 명시한다. 프롬프트는 긴 공통 접두(prefix)를 공유하는 경우를 휴리스틱하게 deduplicate하며, 한 사용자 ID당 최대 200개 프롬프트만 허용한다. 또한 train/valid/test는 사용자 ID 기준으로 분할하여 사용자 단위 누수를 줄인다. 훈련 split의 프롬프트에서 PII를 필터링한다.  
(근거: Section 3.2)

**【원문 근거】**  
초기에는 사용자 프롬프트가 충분하지 않아 라벨러가 직접 프롬프트를 작성해 bootstrap했다. bootstrap 프롬프트는 plain, few-shot, user-based(대기명단 신청서의 use-case 기반) 세 범주로 구성된다. 이 프롬프트에서 SFT용 demonstration 데이터, RM용 ranking 데이터, PPO 입력용 unlabeled prompt 풀을 구성한다.  
(근거: Section 3.2, Appendix A)

**【원문 근거】**  
Table 6은 split별 데이터 규모를 제시한다(예: SFT의 labeler/customer prompt 수, RM의 train/valid prompt 수, PPO의 prompt 수).  
(근거: Table 6)

**【원문 근거】**  
Table 1은 API 프롬프트의 use-case 분포를 제시하며, Generation(45.6%), Open QA(12.4%), Brainstorming(11.2%), Chat(8.4%), Rewrite(6.6%), Summarization(4.2%), Classification(3.5%), Other(3.5%), Closed QA(2.6%), Extract(1.9%) 등으로 구성된다. 즉 정형 QA/분류 중심의 벤치마크 분포와 상이하다.  
(근거: Table 1)

### C. Appendix A의 데이터 통계(핵심 항목)
**【원문 근거】**  
Appendix A는 프롬프트 다양성 및 길이 통계를 추가로 제시한다.

- RM train/valid에서 ambiguous prompt, sensitive content, closed-domain, continuation style, explicit constraints 등의 비율이 보고된다(Table 7).  
- prompt/customer당 프롬프트 수 분포(Table 8), 프롬프트 길이 분포(Table 9–10), 라벨러/고객 프롬프트 및 demonstration 길이 비교(Table 11) 등을 제공한다.  
- 언어 분포는 langid 기반으로 대체로 영어가 대부분이며(약 96%로 보고), 다국어가 소량 포함된다고 기술한다.  
(근거: Appendix A, Table 7–11)

### D. 해설 및 직관
**【추가 설명(일반 지식)】**  
데이터 설계의 핵심은 “instruction-following”의 학습 목표가 곧 **프롬프트 분포 선택 문제**라는 점이다. 공개 태스크의 instruction tuning과 제품/서비스 프롬프트 기반 정렬은 목표 분포가 다르므로, 일반화 성질과 실패 모드도 달라질 수 있다.

---

## 4.3 Section 3.3 — Tasks

### A. 원문 전개(재구성)
**【원문 근거】**  
저자들은 프롬프트가 단순한 명령문뿐 아니라, few-shot 패턴, 혹은 continuation 스타일(서두를 주고 이어 쓰게 하는 형태)로도 태스크를 규정할 수 있음을 명시한다. 라벨러는 이런 입력에서 사용자 의도를 추론하여 응답을 작성/평가하며, 지나치게 모호한 입력은 건너뛰도록 지시받는다.  
(근거: Section 3.3)

### B. 해설 및 직관
**【추가 설명(일반 지식)】**  
실제 사용에서는 “Please do X” 형태의 명령문보다, 예시 제시, 템플릿 기반 입력, 글쓰기 연속 등 간접적 형태의 과제가 빈번하다. 따라서 instruction following을 “명령문 파싱” 문제로 축소하면 실제 분포의 상당 부분을 놓치게 된다.

---

## 4.4 Section 3.4 — Human data collection

### A. 섹션 개요
- 약 40명 contractor를 선발해 demonstration/comparison/evaluation을 수행  
- screening 기준과 instruction 문서, 평가 rubrics의 변화(훈련 vs 최종 평가)가 부록에 상세히 서술  
(근거: Section 3.4, Appendix B)

### B. 원문 전개(재구성)
**【원문 근거】**  
저자들은 Upwork/ScaleAI를 통해 약 40명의 contractor를 고용해 라벨링을 수행한다. 특히 민감한 콘텐츠를 다루므로 screening을 수행했다고 설명한다.  
(근거: Section 3.4)

**【원문 근거】**  
Appendix B.1에 따르면 screening은 (i) 민감 발화 플래깅 일치도, (ii) 출력 순위 판단에서 연구자와의 일치도, (iii) 민감 프롬프트에서 demonstration writing 품질(Likert), (iv) 자신감 등으로 구성된다. soft cutoff(예: 75% 일치도, 6/7 demo score)가 언급된다.  
(근거: Appendix B.1)

**【원문 근거】**  
라벨러와 연구팀은 지속적으로 협업하며, edge case 처리를 위한 커뮤니케이션 채널이 있었음을 적는다. 또한 일반화 점검을 위해 학습 데이터 생성에는 참여하지 않는 held-out labeler도 고용했으나, 이 held-out pool은 screening을 받지 않았다고 한다. inter-annotator agreement 수치도 함께 보고된다.  
(근거: Section 3.4, Appendix B)

### C. Appendix B 세부(지침 변화·인구통계·만족도·UI)
**【원문 근거】**  
Appendix B.2는 라벨링 지침이 프로젝트 진행 중 변화했음을 명시한다. 특히 training 데이터 생성 시에는 helpfulness를 상대적으로 우선했고, 최종 평가에서는 truthful/harmless를 더 중시하도록 지침이 조정되었다.  
(근거: Appendix B.2, Figure 10)

**【원문 근거】**  
Appendix B.3–B.4는 일부 라벨러의 익명 자발 설문을 통한 인구통계 및 만족도 응답 분포를 제시하고, Figure 12는 평가 UI의 구성(선호 선택 및 메타데이터 수집)을 보여준다.  
(근거: Appendix B.3–B.4, Figure 12)

### D. 해설 및 직관
**【추가 설명(일반 지식)】**  
RLHF는 “모델 학습”일 뿐 아니라 “사람 평가 프로토콜 설계” 문제이기도 하다. 누가, 어떤 지침을 받고, 어떤 인터페이스에서 판단했는지가 곧 reward 함수의 정의를 구성한다. 본 논문이 Section 5.2에서 “누구에게 정렬되는가”를 길게 논의하는 배경이 여기에 있다.

---

## 4.5 Section 3.5 — Models

### A. 표기법 정리(핵심)
**【원문 근거】**

| 기호 | 의미 |
|---|---|
| \(x\) | prompt |
| \(y\) | response/completion |
| \(y_w\), \(y_l\) | 선호된 응답(winner), 비선호 응답(loser) |
| \(r_\theta(x,y)\) | reward model의 scalar 점수 |
| \(K\) | 한 prompt에 대해 함께 rank된 응답 수 |
| \(\pi^{SFT}\) | supervised fine-tuned 정책 |
| \(\pi^{RL}_\phi\) | RLHF(PPO)로 최적화된 정책 |
| \(\beta\) | KL penalty 계수 |
| \(\gamma\) | pretraining loss 계수(PPO-ptx) |

(근거: Section 3.5)

### B. 공통 아키텍처/학습 설정
**【원문 근거】**  
모든 모델은 GPT-3 아키텍처를 사용한다. RM과 value function은 출력 헤드를 scalar로 바꿔 사용한다. fp16 가중치/활성, fp32 master weights를 사용하며, GPT-3와 동일한 BPE를 사용한다. context length는 2k이며, prompt와 response 길이에 상한을 둔다. optimizer는 Adam을 사용한다.  
(근거: Section 3.5, Appendix C)

### C. SFT
**【원문 근거】**  
SFT는 라벨러 demonstration에 대해 supervised fine-tuning한 모델이다. 논문은 학습 epoch 수, dropout, cosine LR schedule 등 세부를 Appendix C에 명시한다. 또한 validation loss 최적과 인간 선호 최적이 일치하지 않을 수 있음을 언급하며, checkpoint 선택을 validation RM score 기준으로 수행했다고 기술한다.  
(근거: Section 3.5, Appendix C.1)

### D. Reward Model (RM)
**【원문 근거】**  
RM은 prompt-response pair를 입력으로 받아 scalar reward를 출력한다. 본 논문은 PPO 단계 전반에서 **6B RM 하나**를 사용했다고 밝히며, 그 이유(안정성/비용/초기화 적합성 등)를 제시한다. 학습은 선호 비교 데이터에 대한 pairwise ranking objective로 수행된다.  
(근거: Section 3.5, Appendix C.2)

RM loss는 다음과 같이 제시된다.

\[
\mathcal{L}(\theta)=
-\frac{1}{\binom{K}{2}}
\mathbb{E}_{(x,y_w,y_l)\sim D}
\left[
\log \sigma\!\left(r_\theta(x,y_w)-r_\theta(x,y_l)\right)
\right]
\]

(근거: Section 3.5, Equation (1))

### E. PPO / PPO-ptx
**【원문 근거】**  
RL 단계는 bandit 설정으로 기술되며, prompt가 주어지고 모델이 응답을 생성하면 RM이 보상을 부여하고 종료된다. 이때 SFT 정책과의 KL penalty를 두어 과도한 정책 이동을 억제한다.  
(근거: Section 3.5)

**【원문 근거】**  
PPO-ptx는 여기에 pretraining 데이터에 대한 log-likelihood 항을 섞어 capability regression을 줄이는 변형으로 제시된다. 목적함수는 다음과 같다.

\[
\text{objective}(\phi)=
\mathbb{E}_{(x,y)\sim D_{\pi^{RL}_\phi}}
\left[
r_\theta(x,y)
-
\beta \log \frac{\pi^{RL}_\phi(y\mid x)}{\pi^{SFT}(y\mid x)}
\right]
+
\gamma
\mathbb{E}_{x\sim D_{\text{pretrain}}}
\left[
\log \pi^{RL}_\phi(x)
\right]
\]

(근거: Section 3.5, Equation (2))

### F. Baselines(요지)
**【원문 근거】**  
비교 대상으로 GPT-3, prompted GPT-3, SFT, PPO, PPO-ptx를 둔다. 또한 FLAN 및 T0(T0++) 데이터셋으로 fine-tuning한 175B GPT-3 baseline을 구성하여 비교한다.  
(근거: Section 3.5, Appendix C.5)

### G. 해설 및 직관
**【추가 설명(일반 지식)】**  
RM은 “정답 판별기”라기보다 “선호 판정기”로 설계된다. 따라서 RM이 학습한 것은 사실성 그 자체가 아니라 라벨러가 선호하는 응답의 패턴(간결성, 정중함, 제약 준수, 위험 회피 등)을 포함한다. PPO-ptx는 선호 최적화 과정이 기본 언어모델 능력을 훼손하는 현상을 완화하기 위해, 사전학습 분포에 대한 학습 신호를 함께 유지하는 장치로 이해할 수 있다.

---

## 4.6 Section 3.6 — Evaluation

### A. 섹션 개요
- alignment를 helpful/honest/harmless로 개념화하되, honest는 truthfulness로 부분 대체  
- 실제 API 프롬프트 분포에서의 **인간 선호 평가**와, 공개 NLP 데이터에서의 **자동 평가**를 병행  
- 선호 외에도 실패 모드별 메타데이터를 수집(Table 3)  
(근거: Section 3.6, Table 3, Appendix D)

### B. 원문 전개(재구성)
**【원문 근거】**  
저자들은 alignment를 사용자 의도에 맞게 행동하는 것으로 정의하고, 이를 helpful/honest/harmless 프레임으로 다룬다. 단 honest는 직접 측정이 어려우므로 TruthfulQA 및 closed-domain hallucination 지표로 일부만 측정한다고 밝힌다. harmlessness는 문맥 의존적이어서 직접 측정이 어려우며, 대신 customer assistant 맥락의 부적절성, protected class 폄하, 성적/폭력적 내용 포함 여부 등 proxy criteria를 사용한다.  
(근거: Section 3.6)

**【원문 근거】**  
정량 평가는 (i) API distribution에서 held-out customer prompts에 대한 인간 선호 평가, (ii) GPT-3 distribution prompts에서의 별도 평가, (iii) 공개 NLP 데이터셋에서의 자동 평가로 구성된다. Table 3의 메타데이터 필드는 응답 실패 유형(예: instruction 실패, hallucination, 유해 조언, protected class denigration 등)을 분해해 기록한다.  
(근거: Section 3.6, Table 3)

### C. Appendix D: 자동평가 프로토콜(요지)
**【원문 근거】**  
Appendix D는 각 벤치마크에 대한 prompt 템플릿, decoding 설정, 채점 규칙을 상세히 기술한다. 예를 들어 sampling 기반 태스크는 \(T=0\)에서 생성하고 첫 줄바꿈에서 truncate하며, multiple-choice는 후보 completion의 평균 per-token log probability로 선택한다. toxicity/bias 평가를 위해 basic/respectful/biased prompt 조건을 둔다.  
(근거: Appendix D)

### D. 해설 및 직관
**【추가 설명(일반 지식)】**  
이 평가는 “선호(총괄) + 실패 모드(원인 분해) + 공개 벤치마크(능력 보존)”의 삼각 구조를 가진다. 특히 Table 3의 실패 모드 기록은 “선호 개선이 무엇 때문인지”를 해석 가능하게 만든다는 점에서 중요하다.

---

# 5. Results 해설

---

## 5.1 Section 4.1 — Results on the API distribution

### 5.1.1 선호도 결과(그림 기반 해설)
**【원문 근거】**  
Figure 1(및 본문 서술)은 GPT-3 → prompted GPT-3 → SFT → PPO/PPO-ptx로 진행할수록 API 분포에서의 인간 선호가 개선되는 패턴을 제시한다. error bar는 95% 신뢰구간이다.  
(근거: Figure 1, Section 4.1)

**【원문 근거】**  
또한 본문은 175B InstructGPT(PPO-ptx)가 175B GPT-3 및 few-shot prompted GPT-3와의 비교에서 더 높은 선호를 얻는다고 보고한다(구체 수치는 본문/그림 캡션에서 제시).  
(근거: Section 4.1, Figure 1/3)

### 5.1.2 실패 모드 분해(메타데이터)
**【원문 근거】**  
Figure 4는 선호도 개선이 단일 요인 때문이 아니라, instruction 수행 실패 감소, customer assistant 맥락 적절성 개선, explicit constraint 준수 개선, closed-domain hallucination 감소 등으로 분해될 수 있음을 보여준다.  
(근거: Figure 4, Section 4.1)

### 5.1.3 FLAN / T0 비교
**【원문 근거】**  
Figure 5는 공개 NLP 데이터 기반 instruction tuning(FLAN, T0)으로 fine-tuning한 175B GPT-3와 InstructGPT의 비교를 제시하며, 본 논문 데이터 분포에서 InstructGPT가 더 선호된다는 결과를 보고한다. 저자들은 이 차이를 (i) 공개 데이터의 태스크 분포 편향, (ii) 실제 사용자 입력의 다양성 부족 가능성과 연결해 해석한다.  
(근거: Figure 5, Section 4.1)

### 5.1.4 held-out labeler 일반화
**【원문 근거】**  
Figure 3은 training labeler뿐 아니라 held-out labeler에서도 유사한 선호 패턴이 관찰됨을 보여준다. Appendix E.2는 RM의 cross-validation 기반 예측 성능도 함께 보고한다.  
(근거: Figure 3, Appendix E.2)

### 5.1.5 해설 및 직관
**【추가 설명(일반 지식)】**  
여기서 핵심은 “선호가 올랐다”는 결론 자체보다, Figure 4처럼 **어떤 실패 모드가 줄었는지**를 함께 제시하여 행동 변화의 방향을 구체화했다는 점이다. 또한 공개 데이터 기반 instruction tuning과 실제 서비스 프롬프트 기반 정렬이 목표 분포 차이로 인해 성능/선호가 달라질 수 있음을 실험적으로 드러낸다.

---

## 5.2 Section 4.2 + Appendix E — Public NLP, truthfulness, toxicity, bias

### 5.2.1 Truthfulness / Hallucination
**【원문 근거】**  
저자들은 TruthfulQA 및 closed-domain hallucination 평가에서 개선 경향을 보고한다. 다만 TruthfulQA 자동 평가 지표의 한계와, 자동 메트릭이 개선 폭을 과대평가할 수 있었음을 acknowledge에서 언급한다.  
(근거: Section 4.2, Appendix E, Acknowledgements)

### 5.2.2 Toxicity
**【원문 근거】**  
RealToxicityPrompts에서 respectful prompt 조건에서 독성 감소가 관찰되며, biased prompt 조건에서는 instruction-following 모델이 유해한 지시를 더 잘 따를 위험을 시사하는 패턴이 부록에서 제시된다.  
(근거: Section 4.2, Appendix E)

### 5.2.3 Bias
**【원문 근거】**  
Winogender, CrowS-Pairs 등 편향 벤치마크에서 일관된 유의미 개선은 뚜렷하지 않다고 보고된다. 일부 조건에서는 지표 해석이 단순하지 않음을 저자들이 언급한다.  
(근거: Appendix E.5, Table 14)

### 5.2.4 Alignment tax 및 PPO-ptx의 효과
**【원문 근거】**  
PPO가 일부 공개 NLP 태스크에서 성능 회귀를 보이며, PPO-ptx가 이를 상당 부분 완화하는 결과가 제시된다. Appendix E.6은 \(\gamma\) 증가가 회귀를 완화하는 반면, \(\beta\)만 증가시키는 것은 충분치 않음을 보여주는 ablation을 포함한다.  
(근거: Appendix E.6, Figure 33–34, Table 14)

### 5.2.5 해설 및 직관
**【추가 설명(일반 지식)】**  
이 결과군은 RLHF가 단일 방향의 “개선”이라기보다, 목표 분포의 선호를 올리는 과정에서 다른 능력 지표가 손상될 수 있는 **trade-off**를 명시적으로 다룬다는 점에서 중요하다. PPO-ptx는 그 trade-off를 완화하기 위한 정규화(혹은 다중 목적 최적화)로 이해할 수 있다.

---

## 5.3 Section 4.3 — Qualitative results

### 5.3.1 비영어 및 코드 예시
**【원문 근거】**  
저자들은 비영어 instruction 및 코드 관련 프롬프트에 대한 질적 예시를 제공하며, fine-tuning 데이터에서 희소한 영역에도 instruction-following 행동이 일부 전이될 수 있음을 보인다. 동시에 일부 예시에서는 언어 혼용 등 불완전성이 나타난다.  
(근거: Section 4.3, Figure 8, Figure 42–43)

### 5.3.2 단순 실수(simple mistakes)
**【원문 근거】**  
Figure 9는 false premise를 그대로 수용하거나, 과도한 hedging, 복수 제약 준수 실패 등 단순하지만 중요한 오류를 제시한다. 저자들은 이런 문제를 adversarial data collection 등으로 완화할 가능성을 논의한다.  
(근거: Figure 9, Section 4.3)

### 5.3.3 Appendix F 샘플
**【원문 근거】**  
Appendix F는 추가 샘플을 제공한다. 저자들은 일부 프롬프트가 behavior를 보여주기 위해 선택되었음을 밝히되, 출력은 cherry-pick하지 않았다고 기술한다. harmful prompt 사례는 “instruction following 강화가 유해 지시 수행 강화로도 이어질 수 있음”을 시사한다.  
(근거: Appendix F)

### 5.3.4 해설 및 직관
**【추가 설명(일반 지식)】**  
질적 예시는 모델이 “새로운 능력”을 획득했다기보다, instruction-following이라는 행동 양식이 희귀 도메인(비영어/코드)에도 부분적으로 전이된다는 점을 보여준다. 동시에 harmful prompt 사례는 “도움됨”을 중심으로 최적화한 정렬이 **오용 리스크**를 자동으로 해결하지 않는다는 사실을 부각한다.

---

# 6. Discussion 해설

---

## 6.1 Section 5.1 — Implications for alignment research

### A. 원문 전개(재구성)
**【원문 근거】**  
저자들은 본 연구의 함의를 다음과 같이 정리한다.

- 정렬 비용(compute)은 사전학습에 비해 상대적으로 작다고 보고하며, SFT/PPO-ptx 학습 비용과 GPT-3 사전학습 비용을 비교한다.  
- 특정 태스크 분포에서는 정렬이 단순 스케일 증가보다 사용자 선호 개선에 효과적일 수 있음을 시사한다.  
- instruction-following 행동이 일부 도메인에 전이되는 가능성을 질적 예시로 보인다.  
- PPO-ptx를 통해 alignment tax를 상당 부분 완화할 수 있음을 보이며, 이는 실제 채택 가능성 측면에서 중요하다고 주장한다.  
- production-adjacent 환경에서의 empirical feedback loop의 중요성을 강조한다.  
(근거: Section 5.1)

### B. 해설 및 직관
**【추가 설명(일반 지식)】**  
Section 5.1의 메시지는 “RLHF가 정답”이라기보다, 실제 배포 환경을 기준으로 목표 함수를 정하고 그에 맞춘 측정과 반복 개선을 수행해야 한다는 실증적 방법론에 가깝다.

---

## 6.2 Section 5.2 — Who are we aligning to?

### A. 원문 전개(재구성)
**【원문 근거】**  
저자들은 InstructGPT가 보편적 인간 가치에 정렬된 것이 아니라, **특정 라벨러 집단과 연구자, 그리고 API 고객 프롬프트 분포**의 함수로서 형성된 행동을 학습한 것임을 명시한다. 라벨러의 인구통계적 대표성, 불일치(disagreement), 그리고 “평균 라벨러 선호” 자체가 바람직한 목표인지에 대한 문제가 남는다고 논의한다.  
(근거: Section 5.2, Appendix B)

### B. 해설 및 직관
**【추가 설명(일반 지식)】**  
RLHF의 “정렬 대상”은 기술적 파이프라인 밖에서 결정된다(라벨러 구성, 지침, 데이터 출처, 평가 기준). 따라서 Section 5.2는 단순 윤리 코멘트가 아니라, reward 함수 정의의 범위와 한계를 명확히 규정하는 방법론적 논의로 읽는 것이 타당하다.

---

## 6.3 Section 5.3 — Limitations

### A. 원문 전개(재구성)
**【원문 근거】**  
한계로는 (i) 제한된 라벨러 집단과 단일 라벨링 중심의 데이터 수집, (ii) disagreement를 충분히 반영하기 어려운 점, (iii) 모델이 여전히 사실 오류/독성/편향/성적·폭력적 내용 등에서 완전하지 않다는 점, (iv) instruction following 강화가 유해 instruction 수행 강화로 이어질 수 있다는 점 등이 제시된다.  
(근거: Section 5.3)

### B. 해설 및 직관
**【추가 설명(일반 지식)】**  
이 한계들은 RLHF가 “안전성 완성”이 아니라, 목표·데이터·평가·거버넌스를 포함한 시스템 설계의 일부임을 강조한다.

---

## 6.4 Section 5.4 — Open questions

### A. 원문 전개(재구성)
**【원문 근거】**  
저자들은 adversarial data collection, 유해 지시 거부(refusal) 정책의 설계, control code/steering과의 결합, PPO 이외의 학습 알고리즘, 비교 피드백 외에 critique/edit 같은 더 정보량이 큰 피드백 포맷, pretraining mix의 부작용과 데이터 필터링/보강 문제, “instruction vs intention vs value”의 구분 등 다양한 열린 질문을 제시한다.  
(근거: Section 5.4)

### B. 해설 및 직관
**【추가 설명(일반 지식)】**  
Section 5.4는 본 논문의 파이프라인을 “종결된 해법”이 아니라, 후속 연구가 풀어야 할 구체적 병목과 설계 선택지의 목록으로 정리한다는 점에서 가치가 크다.

---

## 6.5 Section 5.5 — Broader impacts

### A. 원문 전개(재구성)
**【원문 근거】**  
저자들은 본 접근이 언어모델의 긍정적 영향(도움됨/진실성/무해성 개선 가능성)을 확대할 수 있는 동시에, 오용(misuse) 리스크를 낮추지 못하거나 오히려 높일 가능성도 있음을 논의한다. 고위험 도메인(의료/법률/신용/고용/정치 등)에서의 주의, 배포 형태(API vs 오픈 배포)의 trade-off도 함께 언급한다.  
(근거: Section 5.5)

---

# 7. Appendix / Supplementary 핵심 정리

## 7.1 Appendix A — 데이터/프롬프트
**【원문 근거】**  
bootstrap prompt 작성 방식(plain/few-shot/user-based), prompt 예시(카테고리별), 데이터 통계(Table 6–11) 등이 제공된다.  
(근거: Appendix A)

## 7.2 Appendix B — 사람 데이터 수집
**【원문 근거】**  
라벨러 선발, 지침 변화, 인구통계 설문, 만족도 설문, UI 스크린샷 등이 포함된다.  
(근거: Appendix B)

## 7.3 Appendix C — 학습 세부
**【원문 근거】**  
SFT/RM/RLHF 하이퍼파라미터와 초기화, PPO-ptx 구성, FLAN/T0 baseline 튜닝 등이 기술된다.  
(근거: Appendix C)

## 7.4 Appendix D — 자동평가 템플릿/규칙
**【원문 근거】**  
데이터셋별 템플릿과 평가 규칙(샘플링/MC scoring/독성·편향 프롬프트 조건)이 상세히 제시된다.  
(근거: Appendix D)

## 7.5 Appendix E — 추가 결과/ablation
**【원문 근거】**  
Table 14 및 Figures 28–41을 통해 alignment tax, \(\beta/\gamma\) 민감도, 학습 길이/비율/학습률 등의 영향을 분석한다.  
(근거: Appendix E)

## 7.6 Appendix F — 샘플
**【원문 근거】**  
정렬 모델과 GPT-3 출력 비교 샘플을 제공하며, 일부 프롬프트 선택과 출력 선택의 원칙을 설명한다.  
(근거: Appendix F)

---

# (추가) 심화 해설: 수식·도표 중심 재독

이 장은 본문을 한 번 읽은 뒤, 핵심 수식과 도표를 중심으로 다시 읽을 때 유용하도록 구성한다.

## 1. 왜 3단계(SFT → RM → PPO)인가?

### A. 섹션 개요
- SFT: 바람직한 응답의 초기화  
- RM: 선호 비교로부터 reward 함수를 근사  
- PPO: 근사된 reward 함수를 실제 정책 업데이트로 연결  
(근거: Section 3.1, Figure 2)

### B. 원문 전개(재구성)
**【원문 근거】**  
Figure 2에 따라, 시연 기반 지도학습(SFT)과 선호 비교 기반 RM 학습을 선행한 후, RM을 보상으로 PPO를 수행한다. Step 2–3은 반복 가능한 개선 루프다.  
(근거: Section 3.1, Figure 2)

### C. 해설 및 직관
**【추가 설명(일반 지식)】**  
시연은 “좋은 답 하나”를 주지만, 실제 품질 판단은 “여러 답의 상대적 우열”로 더 안정적으로 드러날 수 있다. RM은 이 상대적 우열을 근사하고, PPO는 그 근사를 정책 업데이트에 사용한다.

---

## 2. Reward Model: Equation (1) 정밀 해설

### A. 원문 수식
\[
\mathcal{L}(\theta)=
-\frac{1}{\binom{K}{2}}
\mathbb{E}_{(x,y_w,y_l)\sim D}
\left[
\log \sigma\!\left(r_\theta(x,y_w)-r_\theta(x,y_l)\right)
\right]
\]
(근거: Section 3.5, Eq (1))

### B. 기호 해설
- \(x\): prompt  
- \(y_w\): 더 선호된 응답(winner)  
- \(y_l\): 덜 선호된 응답(loser)  
- \(r_\theta(x,y)\): RM의 scalar score  
- \(K\): 한 prompt에서 함께 순위화된 후보 수  
(근거: Section 3.5)

### C. 의미(직관)
**【추가 설명(일반 지식)】**  
이 loss는 “선호된 응답이 비선호 응답보다 높은 reward를 받도록” 하는 pairwise ranking objective다. 절대 점수의 기준점보다 **차이**가 중요하며, \(\sigma(r_w-r_l)\)가 1에 가까울수록(차이가 클수록) loss가 감소한다.

---

## 3. PPO-ptx: Equation (2) 정밀 해설

### A. 원문 수식
\[
\text{objective}(\phi)=
\mathbb{E}_{(x,y)\sim D_{\pi^{RL}_\phi}}
\left[
r_\theta(x,y)
-
\beta \log \frac{\pi^{RL}_\phi(y\mid x)}{\pi^{SFT}(y\mid x)}
\right]
+
\gamma
\mathbb{E}_{x\sim D_{\text{pretrain}}}
\left[
\log \pi^{RL}_\phi(x)
\right]
\]
(근거: Section 3.5, Eq (2))

### B. 항별 해석
- \(r_\theta(x,y)\): 선호 보상(RM) 최대화  
- KL penalty 항: SFT 정책으로부터의 과도한 이탈 억제  
- pretraining 항(\(\gamma\)): 언어모델링 능력 보존(정렬 과정의 회귀 완화)  
(근거: Section 3.5, Appendix E.6)

### C. 부록 결과와의 연결
**【원문 근거】**  
Appendix E.6은 \(\gamma\) 증가가 공개 NLP 회귀를 완화하는 한편, \(\beta\)만으로는 충분하지 않을 수 있음을 보여준다.  
(근거: Appendix E.6, Figure 33–34)

---

## 4. 주요 도표의 독해 포인트(선별)

### 4.1 Figure 1

> [Figure 삽입 안내] 이 위치에 **논문 원문의 Figure 1** crop을 넣으십시오.  
> - 권장 캡션: `4.1 Figure 1`  
> - 편집 원칙: 도판 자체는 원문 이미지를 사용하고, 본문에서는 **이 도판이 무엇을 보여 주는지**만 해설합니다.

**【원문 근거】**  
API 분포에서의 인간 선호(175B SFT 대비 win-rate)로 여러 모델을 비교한다. GPT-3 계열 대비 PPO/PPO-ptx 계열의 우위를 보여주며, error bar는 95% CI다.  
(근거: Figure 1, 캡션)

### 4.2 Table 1

> [Table 삽입 안내] 이 위치에 **논문 원문의 Table 1** crop을 넣으십시오.  
> - 권장 캡션: `4.2 Table 1`  
> - 편집 원칙: 도판 자체는 원문 이미지를 사용하고, 본문에서는 **이 도판이 무엇을 보여 주는지**만 해설합니다.

**【원문 근거】**  
API 사용 분포가 정형 QA/분류 중심이 아니라 generation/brainstorming 비중이 크다는 점을 정량적으로 제시한다.  
(근거: Table 1)

### 4.3 Figure 4 / Table 3

> [Figure 삽입 안내] 이 위치에 **논문 원문의 Figure 4** crop을 넣으십시오.  
> - 권장 캡션: `4.3 Figure 4 / Table 3`  
> - 편집 원칙: 도판 자체는 원문 이미지를 사용하고, 본문에서는 **이 도판이 무엇을 보여 주는지**만 해설합니다.

**【원문 근거】**  
선호의 개선을 실패 모드(지시 실패, 제약 준수 실패, hallucination 등)로 분해해 해석 가능성을 제공한다.  
(근거: Figure 4, Table 3)

### 4.4 Figure 5

> [Figure 삽입 안내] 이 위치에 **논문 원문의 Figure 5** crop을 넣으십시오.  
> - 권장 캡션: `4.4 Figure 5`  
> - 편집 원칙: 도판 자체는 원문 이미지를 사용하고, 본문에서는 **이 도판이 무엇을 보여 주는지**만 해설합니다.

**【원문 근거】**  
공개 NLP 데이터 기반 instruction tuning(FLAN/T0)과의 비교를 통해, 목표 분포 차이가 품질 판단에 결정적일 수 있음을 논의한다.  
(근거: Figure 5)

## 5. 빈번한 오해(정리)

1. **“InstructGPT는 보편적 인간 가치에 정렬되었다.”**  
   **【원문 근거】** Section 5.2는 정렬 결과가 특정 라벨러/연구자/데이터 분포의 함수임을 명시한다. (근거: Section 5.2)

2. **“RLHF는 자동으로 안전해진다.”**  
   **【원문 근거】** 독성/편향/유해 instruction 수행 등에서 한계와 trade-off를 논의하며, biased prompt에서 위험 패턴이 관찰됨을 보고한다. (근거: Appendix E, Appendix F)

3. **“KL penalty가 있으면 capability 회귀는 없다.”**  
   **【원문 근거】** Appendix E.6은 \(\beta\)만으로 회귀가 충분히 해결되지 않을 수 있음을 보여주고, \(\gamma\)의 역할을 제시한다. (근거: Appendix E.6)

4. **“작은 aligned 모델이 큰 모델을 이겼으니 스케일은 중요하지 않다.”**  
   **【원문 근거】** 본 논문의 주장은 특정 분포에서 정렬이 강력한 레버라는 것이며, 스케일 자체의 중요성을 부정하지 않는다. (근거: Section 4–5 전반)

---
