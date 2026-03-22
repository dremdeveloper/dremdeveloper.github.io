---
title: "RLHF의 표준 파이프라인 — Training Language Models to Follow Instructions with Human Feedback"
source_paper: "Long Ouyang et al., 2022"
arxiv: "2203.02155"
---

# RLHF의 표준 파이프라인 — Training Language Models to Follow Instructions with Human Feedback

이 문서는 InstructGPT 논문을 “사람 피드백으로 언어모델 행동을 재정렬하는 운영 가능한 파이프라인”이라는 관점에서 다시 정리한 해설본이다. 이 논문의 핵심은 새로운 언어 능력을 만드는 것이 아니라, 이미 큰 모델이 가진 능력을 **사용자 지시를 더 잘 따르는 방향**으로 옮기는 데 있다. 그래서 가장 핵심 논점은 세부 하이퍼파라미터가 아니라 **SFT → Reward Model → PPO / PPO-ptx**라는 세 단계의 연결 구조다. 처음 읽는다면 Figure 2, Figure 1, Figure 4, Appendix E.6 순서로 훑으면 전체 흐름이 빠르게 잡힌다.

## 핵심 요약
- **푸는 문제:** 다음 토큰 예측을 잘하는 대형 언어모델이 실제 사용자 지시를 더 잘 따르도록 만들 수 있는가.
- **핵심 아이디어:** 시연 기반 SFT로 초기 정책을 만든 뒤, 선호 비교 데이터로 보상모델을 학습하고, PPO로 정책을 사람 선호 쪽으로 미세조정한다.
- **주요 결과:** 사람 평가에서 InstructGPT는 더 큰 GPT-3 base model보다도 더 선호되는 경우가 많았고, 일부 truthfulness/독성 지표도 개선됐다.
- **왜 중요한가:** 이후 RLHF 계열 연구의 표준 파이프라인이 이 논문에서 거의 완성된 형태로 제시된다.
- **한계 / 주의점:** 이 정렬은 특정 라벨러 집단과 특정 프롬프트 분포에 맞춰진 결과이며, capability regression과 유해 지시 수행 같은 위험은 여전히 남는다.

## 3분 요약: RLHF 파이프라인
| 단계 | 입력 | 학습 목표 | 결과물 |
| --- | --- | --- | --- |
| SFT | 프롬프트 + 라벨러 시연 응답 | 바람직한 답변 형태를 먼저 학습 | instruction-following 초기 정책 |
| Reward Model | 같은 프롬프트에 대한 여러 응답 + 선호 비교 | 어떤 응답이 더 나은지 점수화 | scalar reward를 주는 보상모델 |
| PPO / PPO-ptx | 현재 정책이 생성한 응답 + RM 점수 | 선호되는 응답 확률을 높이되, 기본 LM 성질은 너무 잃지 않기 | 더 잘 정렬된 정책 |

## 먼저 읽을 포인트
- Figure 2는 이 논문의 뼈대다. 나머지 섹션은 거의 전부 이 그림을 풀어 쓰는 방식으로 읽어도 된다.
- Figure 1은 “정말 더 좋아졌는가”를, Figure 4와 Table 3은 “무엇이 좋아졌는가”를 보여 준다.
- Appendix E.6은 PPO-ptx가 왜 필요한지를 가장 선명하게 보여 준다. 정렬이 capability regression을 부를 수 있다는 점을 여기서 확인할 수 있다.

## 문헌 정보
- 논문: Long Ouyang et al., 2022
- arXiv: 2203.02155
- 원문 링크: https://arxiv.org/abs/2203.02155
- PDF: https://arxiv.org/pdf/2203.02155

---

# 1. Abstract 해설

## 핵심 논점
Abstract는 (i) 사전학습 언어모델과 사용자 의도 간 불일치 문제, (ii) 시연 기반 지도 미세조정과 선호 기반 강화학습(RLHF)으로 구성된 학습 절차, (iii) 인간 평가 및 자동 평가에서의 관찰, (iv) 잔존 한계에 대한 언급으로 구성된다.
(Abstract)

## 본문 정리

Abstract는 다음 논리로 구성된다.

- 사전학습 언어모델은 다양한 작업을 수행할 수 있지만, 실제 사용에서는 **허위 생성(untruthful), 독성(toxic), 비도움(not helpful)** 같은 의도치 않은 행동이 나타난다.
- 저자들은 사전학습 목표(웹 텍스트 다음 토큰 예측)와 실제 목표(사용자 지시를 도움이 되고 안전하게 따르기) 사이의 불일치를 핵심 원인으로 본다.
- 이를 완화하기 위해, (i) 라벨러가 작성한 프롬프트 및 API 프롬프트를 수집하고, (ii) 라벨러가 “바람직한 응답”을 작성한 데이터로 **지도 미세조정(SFT)** 을 수행한 뒤, (iii) 모델 출력에 대한 **선호 순위 데이터**로 **보상 모델(RM)** 을 학습하고, (iv) RM을 보상으로 **강화학습(RLHF; PPO)** 을 수행한다.
- 이로써 얻은 모델을 InstructGPT라 부르며, 인간 평가에서 선호 개선을 보이고, truthfulness 향상 및 toxic 출력 감소를 보고한다. 다만 단순 실수와 안전성 한계는 남아 있다고 명시한다.
(Abstract)

## 해설

Abstract가 제시하는 연구 프레임은 “능력(capability)”과 “행동 정렬(behavioral alignment)”을 분리하여 다룬다는 점에 있다. 사전학습은 언어적·추론적 능력을 부여하지만, 사용자가 원하는 응답 양식/우선순위(예: 도움이 되되 무해하게)를 자동으로 내재화하지 않는다. 본 논문은 후단 학습을 통해 그 행동 규범을 명시적으로 학습시키는 설계를 제안한다.

---

# 2. Introduction 해설

## 핵심 논점
- **문제정의:** 사전학습 LM의 목표 함수와 배포 환경의 목표 함수가 다르며, 이로 인해 의도치 않은 행동이 발생한다.
- **정렬 개념화:** 사용자 의도를 “명시적 지시 준수 + 암묵적 기대(진실성/무해성 등)”로 확장해 정의한다.
- **평가 철학:** 공개 NLP 벤치마크만으로는 실제 사용 분포의 품질을 대표하기 어렵다는 전제를 둔다.
(Section 1)

## 본문 정리

서론은 다음을 강조한다.

1. 대규모 언어모델은 프롬프팅으로 다양한 작업을 수행하지만, **사실을 만들어내거나**, **편향/독성 텍스트를 생성하거나**, **사용자 지시를 따르지 않는** 문제가 발생한다.
2. 이는 사전학습 목표(웹 문서 다음 토큰 예측)가 “사용자 지시를 도움이 되고 안전하게 따르기” 목표와 다르기 때문이며, 저자들은 이를 **misaligned** 라고 표현한다.
3. 정렬(alignment)은 “사용자 의도(user intentions)에 부합하는 행동”으로 정의되며, Askell et al.의 표현을 빌려 **helpful / honest / harmless** 프레임을 사용한다. 단, honest는 직접 측정이 어려워 본 논문에서는 truthfulness와 hallucination 감소로 일부만 포착한다고 한다.
4. 본 논문은 RLHF를 사용하여 GPT-3를 광범위한 written instruction에 맞춰 fine-tune하는 접근을 제시한다(도식: Figure 2). 또한 이 과정은 특정 집단(라벨러/연구자)의 선호에 맞춘 정렬이며 “보편적 인간 가치”로 일반화되지 않는다는 점을 서론에서부터 분명히 한다.
(Section 1, Figure 2)

## 해설

서론의 실질적 기여는 “LM 성능”을 단일 축으로 보지 않고, **배포 환경의 사용자 경험/위험**을 중심으로 목표를 재정의했다는 점이다. 즉, 모델이 어떤 벤치마크를 얼마나 맞히는지보다, 실제 사용에서 **의도 준수(Instruction following), 통제 용이성(Constraint following), 사실성(Hallucination 감소), 유해성(Toxicity/Bias 완화)** 같은 행동 특성이 어떻게 변화하는지를 분해해 측정한다.

---

# 3. Related Work 해설

## 핵심 논점
- RLHF 계열(특히 텍스트 요약에서의 사람 피드백 학습)과의 연속성
- 공개 NLP 데이터 기반 instruction tuning(FLAN/T0 계열)과의 대비
- harm(독성/편향/프라이버시/오정보) 평가 벤치마크의 맥락
(Section 2)

## 분류와 비교

논문이 인접 연구를 묶는 방식은 다음과 같이 재정리할 수 있다.

| 범주 | 원문이 언급하는 선행 흐름 | 본 논문과의 관계 |
|---|---|---|
| RLHF / human feedback | 로봇·게임·요약 등에서 선호 학습 및 RL 적용 | RLHF를 **광범위한 자연어 태스크 분포**에 적용 |
| Instruction tuning / cross-task generalization | 다수 공개 NLP 태스크에 instruction prefix를 부착해 미세조정(FLAN, T0 등) | 공개 데이터 기반 접근과 달리 **실제 API 프롬프트 분포**에 초점을 둠 |
| Harms 평가 | toxicity, stereotype, bias, privacy leakage, misinformation 등 | harms는 문맥 의존적이므로 proxy metric과 사람 평가를 병행 |
| 행동 제어 기법 | 데이터 필터링, control token, 블로킹, 별도 모델로 steering 등 | RLHF를 중심 축으로 두되 상보적 관계로 위치시킴 |

(Section 2)

## 본문 정리

저자들은 RLHF를 텍스트 요약에 적용한 선행 연구(Stiennon et al.)를 중요한 전 단계로 두고, 이를 더 넓은 태스크 분포로 확장한다. 또한 공개 NLP 데이터로 instruction tuning을 수행한 연구 계열과 비교하며, 본 논문의 핵심은 **“어떤 분포에 대해” 정렬하는가**(public NLP tasks vs 실제 사용자 프롬프트 분포)의 차이에 있음을 강조한다. harms 평가에 관해서는 문헌을 인용하며, 특정 개입이 부작용을 가져올 수 있음을 언급한다.
(Section 2)

## 해설

본 논문은 “instruction following”이라는 표면적 목표가 유사하더라도, (i) 학습 신호(정답/시연 vs 선호 비교), (ii) 목표 분포(public task mixture vs product prompts), (iii) 평가 방식(자동평가 중심 vs 인간 선호 중심)에 따라 결과가 실질적으로 달라질 수 있음을 명시적으로 실험한다는 점에서 구분된다.

---

# 4. Methods and Experimental Details 해설

---

## 4.1 Section 3.1 — High-level methodology

### 핵심 논점
- 파이프라인은 **(1) SFT, (2) RM 학습, (3) PPO 기반 RLHF**의 3단계로 제시된다.
- Step 2–3은 반복 가능한 루프(정책 갱신 → 비교 데이터 추가 수집 → RM 갱신)로 기술된다.
(Section 3.1, Figure 2)

### 본문 정리

> **Figure 2 삽입**

Figure 2에 따라 방법은 다음과 같다.

1. **Demonstration 수집 → SFT**
   라벨러가 프롬프트에 대한 바람직한 응답을 작성한다. 이 데이터를 사용해 pretrained GPT-3를 supervised fine-tuning 한다.
2. **Comparison 수집 → Reward Model 학습**
   동일 프롬프트에 대한 여러 모델 출력을 라벨러에게 제시하고, 더 나은 출력을 선택하게 하여 선호 비교 데이터를 수집한다. 이를 사용해 RM을 학습한다.
3. **PPO 기반 강화학습(RLHF)**
   RM을 보상 함수로 두고, SFT 정책을 초기 정책으로 하여 PPO로 추가 미세조정한다.

또한 비교 데이터는 supervised policy에서 주로 생성되지만, 일부는 PPO 정책에서도 생성될 수 있으며, 이로써 반복적으로 개선 가능한 루프를 구성한다.
(Section 3.1, Figure 2)

### 해설

3단계 구성은 기능적으로 분리된다. SFT는 “기본적으로 적절한 응답”의 초기화를 제공하고, RM은 “여러 응답 중 무엇이 더 낫나”라는 선호 함수를 근사하며, PPO는 그 선호 함수를 정책 업데이트로 연결한다. 이 분해가 없으면, 선호 학습을 안정적으로 수행하기 어렵거나, 반대로 정책이 보상 모델을 과도하게 exploit할 가능성이 커진다.

즉 RLHF를 완전히 새로운 학습 패러다임처럼 볼 필요는 없다. 실제 운영 구조는 **좋은 답변 형식을 먼저 SFT로 만들고, 사람 선호를 점수 함수로 바꾼 뒤, PPO로 정책을 미세하게 이동시키는 3단계 파이프라인**으로 이해하는 것이 가장 정확하다. 이 단순한 분해가 InstructGPT를 이후 RLHF 계열 연구의 기준점으로 만들었다.

### 의사코드

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

### 핵심 논점
- 데이터 출처: OpenAI API(특히 Playground)에서 수집된 프롬프트 + 라벨러가 작성한 bootstrap 프롬프트
- 분할: 사용자 ID 단위로 train/valid/test 분리
- 분포: 공개 NLP 벤치마크와 달리 **open-ended generation/brainstorming** 비중이 큼
(Section 3.2, Table 1, Table 6, Appendix A)

### 본문 정리

프롬프트는 주로 OpenAI API Playground에서 초기 InstructGPT 모델을 사용하던 고객이 입력한 텍스트 프롬프트로부터 수집된다. 저자들은 production API 데이터는 사용하지 않았다고 명시한다. 프롬프트는 긴 공통 접두(prefix)를 공유하는 경우를 휴리스틱하게 deduplicate하며, 한 사용자 ID당 최대 200개 프롬프트만 허용한다. 또한 train/valid/test는 사용자 ID 기준으로 분할하여 사용자 단위 누수를 줄인다. 훈련 split의 프롬프트에서 PII를 필터링한다.
(Section 3.2)

초기에는 사용자 프롬프트가 충분하지 않아 라벨러가 직접 프롬프트를 작성해 bootstrap했다. bootstrap 프롬프트는 plain, few-shot, user-based(대기명단 신청서의 use-case 기반) 세 범주로 구성된다. 이 프롬프트에서 SFT용 demonstration 데이터, RM용 ranking 데이터, PPO 입력용 unlabeled prompt 풀을 구성한다.
(Section 3.2, Appendix A)

> **Table 6 삽입**

Table 6은 split별 데이터 규모를 제시한다(예: SFT의 labeler/customer prompt 수, RM의 train/valid prompt 수, PPO의 prompt 수).
(Table 6)

> **Table 1 삽입**

Table 1은 API 프롬프트의 use-case 분포를 제시하며, Generation(45.6%), Open QA(12.4%), Brainstorming(11.2%), Chat(8.4%), Rewrite(6.6%), Summarization(4.2%), Classification(3.5%), Other(3.5%), Closed QA(2.6%), Extract(1.9%) 등으로 구성된다. 즉 정형 QA/분류 중심의 벤치마크 분포와 상이하다.
(Table 1)

### Appendix A의 데이터 통계(핵심 항목)

Appendix A는 프롬프트 다양성 및 길이 통계를 추가로 제시한다.

- RM train/valid에서 ambiguous prompt, sensitive content, closed-domain, continuation style, explicit constraints 등의 비율이 보고된다(Table 7).
> **Table 11 삽입**

> **Table 9 삽입**

> **Table 8 삽입**

- prompt/customer당 프롬프트 수 분포(Table 8), 프롬프트 길이 분포(Table 9–10), 라벨러/고객 프롬프트 및 demonstration 길이 비교(Table 11) 등을 제공한다.
- 언어 분포는 langid 기반으로 대체로 영어가 대부분이며(약 96%로 보고), 다국어가 소량 포함된다고 기술한다.
> **Table 7 삽입**

(Appendix A, Table 7–11)

### 해설

데이터 설계의 핵심은 “instruction-following”의 학습 목표가 곧 **프롬프트 분포 선택 문제**라는 점이다. 공개 태스크의 instruction tuning과 제품/서비스 프롬프트 기반 정렬은 목표 분포가 다르므로, 일반화 성질과 실패 모드도 달라질 수 있다.

---

## 4.3 Section 3.3 — Tasks

### 본문 정리

저자들은 프롬프트가 단순한 명령문뿐 아니라, few-shot 패턴, 혹은 continuation 스타일(서두를 주고 이어 쓰게 하는 형태)로도 태스크를 규정할 수 있음을 명시한다. 라벨러는 이런 입력에서 사용자 의도를 추론하여 응답을 작성/평가하며, 지나치게 모호한 입력은 건너뛰도록 지시받는다.
(Section 3.3)

### 해설

실제 사용에서는 “Please do X” 형태의 명령문보다, 예시 제시, 템플릿 기반 입력, 글쓰기 연속 등 간접적 형태의 과제가 빈번하다. 따라서 instruction following을 “명령문 파싱” 문제로 축소하면 실제 분포의 상당 부분을 놓치게 된다.

---

## 4.4 Section 3.4 — Human data collection

### 핵심 논점
- 약 40명 contractor를 선발해 demonstration/comparison/evaluation을 수행
- screening 기준과 instruction 문서, 평가 rubrics의 변화(훈련 vs 최종 평가)가 부록에 상세히 서술
(Section 3.4, Appendix B)

### 본문 정리

저자들은 Upwork/ScaleAI를 통해 약 40명의 contractor를 고용해 라벨링을 수행한다. 특히 민감한 콘텐츠를 다루므로 screening을 수행했다고 설명한다.
(Section 3.4)

Appendix B.1에 따르면 screening은 (i) 민감 발화 플래깅 일치도, (ii) 출력 순위 판단에서 연구자와의 일치도, (iii) 민감 프롬프트에서 demonstration writing 품질(Likert), (iv) 자신감 등으로 구성된다. soft cutoff(예: 75% 일치도, 6/7 demo score)가 언급된다.
(Appendix B.1)

라벨러와 연구팀은 지속적으로 협업하며, edge case 처리를 위한 커뮤니케이션 채널이 있었음을 적는다. 또한 일반화 점검을 위해 학습 데이터 생성에는 참여하지 않는 held-out labeler도 고용했으나, 이 held-out pool은 screening을 받지 않았다고 한다. inter-annotator agreement 수치도 함께 보고된다.
(Section 3.4, Appendix B)

### Appendix B 세부(지침 변화·인구통계·만족도·UI)

Appendix B.2는 라벨링 지침이 프로젝트 진행 중 변화했음을 명시한다. 특히 training 데이터 생성 시에는 helpfulness를 상대적으로 우선했고, 최종 평가에서는 truthful/harmless를 더 중시하도록 지침이 조정되었다.
> **Figure 10 삽입**

(Appendix B.2, Figure 10)

> **Figure 12 삽입**

Appendix B.3–B.4는 일부 라벨러의 익명 자발 설문을 통한 인구통계 및 만족도 응답 분포를 제시하고, Figure 12는 평가 UI의 구성(선호 선택 및 메타데이터 수집)을 보여준다.
(Appendix B.3–B.4, Figure 12)

### 해설

RLHF는 “모델 학습”일 뿐 아니라 “사람 평가 프로토콜 설계” 문제이기도 하다. 누가, 어떤 지침을 받고, 어떤 인터페이스에서 판단했는지가 곧 reward 함수의 정의를 구성한다. 본 논문이 Section 5.2에서 “누구에게 정렬되는가”를 길게 논의하는 배경이 여기에 있다.

---

## 4.5 Section 3.5 — Models

### 표기법 정리(핵심)

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

(Section 3.5)

### 공통 아키텍처/학습 설정

모든 모델은 GPT-3 아키텍처를 사용한다. RM과 value function은 출력 헤드를 scalar로 바꿔 사용한다. fp16 가중치/활성, fp32 master weights를 사용하며, GPT-3와 동일한 BPE를 사용한다. context length는 2k이며, prompt와 response 길이에 상한을 둔다. optimizer는 Adam을 사용한다.
(Section 3.5, Appendix C)

### SFT

SFT는 라벨러 demonstration에 대해 supervised fine-tuning한 모델이다. 논문은 학습 epoch 수, dropout, cosine LR schedule 등 세부를 Appendix C에 명시한다. 또한 validation loss 최적과 인간 선호 최적이 일치하지 않을 수 있음을 언급하며, checkpoint 선택을 validation RM score 기준으로 수행했다고 기술한다.
(Section 3.5, Appendix C.1)

### Reward Model (RM)

RM은 prompt-response pair를 입력으로 받아 scalar reward를 출력한다. 본 논문은 PPO 단계 전반에서 **6B RM 하나**를 사용했다고 밝히며, 그 이유(안정성/비용/초기화 적합성 등)를 제시한다. 학습은 선호 비교 데이터에 대한 pairwise ranking objective로 수행된다.
(Section 3.5, Appendix C.2)

RM loss는 다음과 같이 제시된다.

\[
\mathcal{L}(\theta)=
-\frac{1}{\binom{K}{2}}
\mathbb{E}_{(x,y_w,y_l)\sim D}
\left[
\log \sigma\!\left(r_\theta(x,y_w)-r_\theta(x,y_l)\right)
\right]
\]

(Section 3.5, Equation (1))

### E. PPO / PPO-ptx

RL 단계는 bandit 설정으로 기술되며, prompt가 주어지고 모델이 응답을 생성하면 RM이 보상을 부여하고 종료된다. 이때 SFT 정책과의 KL penalty를 두어 과도한 정책 이동을 억제한다.
(Section 3.5)

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

(Section 3.5, Equation (2))

### F. Baselines(요지)

비교 대상으로 GPT-3, prompted GPT-3, SFT, PPO, PPO-ptx를 둔다. 또한 FLAN 및 T0(T0++) 데이터셋으로 fine-tuning한 175B GPT-3 baseline을 구성하여 비교한다.
(Section 3.5, Appendix C.5)

### G. 해설

RM은 “정답 판별기”라기보다 “선호 판정기”로 설계된다. 따라서 RM이 학습한 것은 사실성 그 자체가 아니라 라벨러가 선호하는 응답의 패턴(간결성, 정중함, 제약 준수, 위험 회피 등)을 포함한다. PPO-ptx는 선호 최적화 과정이 기본 언어모델 능력을 훼손하는 현상을 완화하기 위해, 사전학습 분포에 대한 학습 신호를 함께 유지하는 장치로 이해할 수 있다.

### H. Appendix C 기반 학습 세부사항(원문 수치 정리)

원문은 Appendix C에서 학습 설정을 비교적 상세히 기술한다. 아래는 해당 수치를 문맥에 맞게 정리한 것이다.
(Appendix C, 특히 C.1–C.5)

#### H.1 공통 모델/토크나이징 설정
- 모든 모델은 GPT-3 아키텍처를 사용한다. RM과 value function은 원래 모델의 unembedding 층을 **스칼라 출력 projection 층**으로 치환한다.
- fp16 가중치/활성값을 사용하되, fp32 master copy를 유지한다.
- Brown et al. (2020)과 동일한 BPE 인코딩을 사용한다.
- 컨텍스트 길이는 2k 토큰이며, **프롬프트가 1k 토큰을 초과하면 필터링**하고 **최대 응답 길이를 1k 토큰으로 제한**한다.
- 최적화는 Adam을 사용하며,
  \(\beta_1=0.9\), \(\beta_2=0.95\)로 명시한다.
(Appendix C 서두)

#### H.2 SFT 학습(C.1)
- 16 epochs 학습, residual dropout 0.2.
- cosine learning-rate schedule을 사용하며, 학습 종료 시점에 초기 학습률의 10%까지 감소시킨다. warmup은 사용하지 않는다.
- 1.3B/6B: learning rate \(9.65\times 10^{-6}\), batch size 32.
- 175B: learning rate \(5.03\times 10^{-6}\), batch size 8.
- learning rate는 (1.3B/6B에 대해 7개, 175B에 대해 5개) 후보를 두고 geometric search로 선택했으며, epoch 수도 geometric search로 조정했다고 보고한다.
(Appendix C.1)

#### H.3 RM 학습(C.2)
- 모든 PPO 정책(1.3B/6B/175B)에 대해 **단일 6B reward model**을 학습하여 공통으로 사용한다.
- 175B RM은 더 낮은 validation loss를 달성할 수 있었으나, (i) 학습 불안정성, (ii) PPO에서 RM/value가 175B가 되면서 계산량이 크게 증가하는 문제로 인해 최종 설정에서 배제된다.
- RM 초기화: 다양한 공개 NLP 데이터셋(ARC, BoolQ, CoQA, DROP, MultiNLI, OpenBookQA, QuAC, RACE, Winogrande)에 fine-tuning된 6B GPT-3에서 초기화했다고 기술한다(역사적 이유라고 설명).
- 학습: 전체 RM 학습셋에 대해 **1 epoch**, learning rate \(9\times 10^{-6}\), cosine schedule(종료 시 10%까지 decay), batch size 64(여기서 batch는 ‘프롬프트 개수’ 기준).
- 비교 수집: 프롬프트마다 \(K\in\{4,\dots,9\}\)개의 응답을 순위화하며, tie는 제거한다. 한 배치가 최대 \(64\times \binom{K}{2}\le 2304\) 비교쌍을 포함할 수 있음을 명시한다.
- 일반화 점검: Appendix E.2에서 held-out labeler에 대해 human-preferred output을 예측하는 inter-/intra-group validation accuracy를 각각 72.4±0.4%, 69.6±0.9%로 보고한다.
(Appendix C.2–C.3, Appendix E.2)

#### H.4 RLHF 초기화 모델(C.3)
- RL 정책은 pretrained GPT-3에서 시작하여, demonstration 데이터에 대해 2 epochs의 supervised fine-tuning을 수행한다.
- 이때 10% pretraining data mix를 포함하며(이 선택이 PPO 단계에서 유리하다고 Appendix E.11에서 언급), cosine schedule로 최종 10%까지 decay.
- batch size: 1.3B/6B는 32, 175B는 8.
- peak learning rate는 sweep으로 탐색 후, 1.3B/6B/175B에 대해 각각 \(5\times 10^{-6}\), \(1.04\times 10^{-5}\), \(2.45\times 10^{-6}\)을 사용한다.
(Appendix C.3)

#### H.5 PPO 및 PPO-ptx 학습(C.4)
- KL penalty 계수: \(\beta=0.02\).
- 학습 에피소드 수: 256k episodes. 약 31k unique prompts를 포함하며, PII 필터링 및 common-prefix 기반 dedup 이후의 수치라고 설명한다.
- batch size 512, minibatch size 64. 각 batch는 8개 minibatch로 나뉘며, **single inner epoch**만 수행한다.
- learning rate: 상수 learning rate를 사용하되 첫 10 iterations 동안 warmup(peak의 1/10에서 시작).
- exponential moving average(EMA) decay 0.992.
- GAE 추정에서 discount를 적용하지 않는다고 명시한다.
- PPO clip ratio 0.2, rollout sampling temperature 1.
- 모든 PPO 정책에서 RM과 value function은 6B를 사용하며, value function은 RM에서 초기화한다. value function learning rate는 1.3B/6B 정책에서는 \(9\times 10^{-6}\), 175B 정책에서는 \(5\times 10^{-6}\).
- PPO-ptx의 pretraining mix: RL 에피소드 수 대비 **8배의 pretraining example**을 사용하며, PPO gradient와 pretraining gradient를 연속 단계로 계산해 함께 누적한다. pretraining gradient에는 계수 \(\gamma=27.8\)을 곱해 상대적 강도를 조절한다.
(Appendix C.4)

#### H.6 FLAN/T0 비교 베이스라인(C.5)
- 175B GPT-3를 FLAN 및 T0++ 데이터셋으로 fine-tuning한 베이스라인을 구성한다.
- 데이터 규모 정렬: T0(96M datapoints)이 FLAN(1.2M datapoints)보다 훨씬 크므로, T0를 1M datapoints로 subsample하여 학습량을 맞춘다.
- 학습: cosine schedule(종료 시 10%까지 decay), batch size 64. 초기 learning rate로 4e-6 및 6e-6을 비교한다.
- 체크포인트 선택: 6B RM으로 validation completions를 scoring하여 선택한다. FLAN은 learning rate 4e-6, 896k examples 학습 체크포인트를 선택했다고 보고한다. T0는 두 설정(batch 128, lr 4e-6, 1.28M examples / batch 64, lr 6e-6, 1M examples) 중 전자에서 896k examples 지점 체크포인트를 선택한다.
> **Figure 13 삽입**

(Appendix C.5, Figure 13)

---

## 4.6 Section 3.6 — Evaluation

### 핵심 논점
- alignment를 helpful/honest/harmless로 개념화하되, honest는 truthfulness로 부분 대체
- 실제 API 프롬프트 분포에서의 **인간 선호 평가**와, 공개 NLP 데이터에서의 **자동 평가**를 병행
- 선호 외에도 실패 모드별 메타데이터를 수집(Table 3)
> **Table 3 삽입**

(Section 3.6, Table 3, Appendix D)

### 본문 정리

저자들은 alignment를 사용자 의도에 맞게 행동하는 것으로 정의하고, 이를 helpful/honest/harmless 프레임으로 다룬다. 단 honest는 직접 측정이 어려우므로 TruthfulQA 및 closed-domain hallucination 지표로 일부만 측정한다고 밝힌다. harmlessness는 문맥 의존적이어서 직접 측정이 어려우며, 대신 customer assistant 맥락의 부적절성, protected class 폄하, 성적/폭력적 내용 포함 여부 등 proxy criteria를 사용한다.
(Section 3.6)

정량 평가는 (i) API distribution에서 held-out customer prompts에 대한 인간 선호 평가, (ii) GPT-3 distribution prompts에서의 별도 평가, (iii) 공개 NLP 데이터셋에서의 자동 평가로 구성된다. Table 3의 메타데이터 필드는 응답 실패 유형(예: instruction 실패, hallucination, 유해 조언, protected class denigration 등)을 분해해 기록한다.
(Section 3.6, Table 3)

### Appendix D: 자동평가 프로토콜(요지)

Appendix D는 각 벤치마크에 대한 prompt 템플릿, decoding 설정, 채점 규칙을 상세히 기술한다. 예를 들어 sampling 기반 태스크는 \(T=0\)에서 생성하고 첫 줄바꿈에서 truncate하며, multiple-choice는 후보 completion의 평균 per-token log probability로 선택한다. toxicity/bias 평가를 위해 basic/respectful/biased prompt 조건을 둔다.
(Appendix D)

### 해설

이 평가는 “선호(총괄) + 실패 모드(원인 분해) + 공개 벤치마크(능력 보존)”의 삼각 구조를 가진다. 특히 Table 3의 실패 모드 기록은 “선호 개선이 무엇 때문인지”를 해석 가능하게 만든다는 점에서 중요하다.

---

# 5. Results 해설

---

## 5.1 Section 4.1 — Results on the API distribution

### 5.1.1 선호도 결과(그림 기반 해설)

> **Figure 1 삽입**

Figure 1(및 본문 서술)은 GPT-3 → prompted GPT-3 → SFT → PPO/PPO-ptx로 진행할수록 API 분포에서의 인간 선호가 개선되는 패턴을 제시한다. error bar는 95% 신뢰구간이다.
(Figure 1, Section 4.1)

또한 본문은 175B InstructGPT(PPO-ptx)가 175B GPT-3 및 few-shot prompted GPT-3와의 비교에서 더 높은 선호를 얻는다고 보고한다(구체 수치는 본문/그림 캡션에서 제시).
(Section 4.1, Figure 1/3)

### 5.1.2 실패 모드 분해(메타데이터)

> **Figure 4 삽입**

Figure 4는 선호도 개선이 단일 요인 때문이 아니라, instruction 수행 실패 감소, customer assistant 맥락 적절성 개선, explicit constraint 준수 개선, closed-domain hallucination 감소 등으로 분해될 수 있음을 보여준다.
(Figure 4, Section 4.1)

### 5.1.3 FLAN / T0 비교

> **Figure 5 삽입**

Figure 5는 공개 NLP 데이터 기반 instruction tuning(FLAN, T0)으로 fine-tuning한 175B GPT-3와 InstructGPT의 비교를 제시하며, 본 논문 데이터 분포에서 InstructGPT가 더 선호된다는 결과를 보고한다. 저자들은 이 차이를 (i) 공개 데이터의 태스크 분포 편향, (ii) 실제 사용자 입력의 다양성 부족 가능성과 연결해 해석한다.
(Figure 5, Section 4.1)

### 5.1.4 held-out labeler 일반화

> **Figure 3 삽입**

Figure 3은 training labeler뿐 아니라 held-out labeler에서도 유사한 선호 패턴이 관찰됨을 보여준다. Appendix E.2는 RM의 cross-validation 기반 예측 성능도 함께 보고한다.
(Figure 3, Appendix E.2)

### 5.1.5 해설

여기서 주목할 점은 “선호가 상승했다”는 관찰을 단순히 보고하는 데 그치지 않고, Figure 4처럼 **어떤 실패 모드가 줄었는지**를 함께 제시하여 행동 변화의 방향을 구체화했다는 점이다. 또한 공개 데이터 기반 instruction tuning과 실제 서비스 프롬프트 기반 정렬이 목표 분포 차이로 인해 성능/선호가 달라질 수 있음을 실험적으로 드러낸다.

---

## 5.2 Section 4.2 + Appendix E — Public NLP, truthfulness, toxicity, bias

### 5.2.1 Truthfulness / Hallucination

저자들은 TruthfulQA 및 closed-domain hallucination 평가에서 개선 경향을 보고한다. 다만 TruthfulQA 자동 평가 지표의 한계와, 자동 메트릭이 개선 폭을 과대평가할 수 있었음을 acknowledge에서 언급한다.
(Section 4.2, Appendix E, Acknowledgements)

### 5.2.2 Toxicity

RealToxicityPrompts에서 respectful prompt 조건에서 독성 감소가 관찰되며, biased prompt 조건에서는 instruction-following 모델이 유해한 지시를 더 잘 따를 위험을 시사하는 패턴이 부록에서 제시된다.
(Section 4.2, Appendix E)

### 5.2.3 Bias

Winogender, CrowS-Pairs 등 편향 벤치마크에서 일관된 유의미 개선은 뚜렷하지 않다고 보고된다. 일부 조건에서는 지표 해석이 단순하지 않음을 저자들이 언급한다.
(Appendix E.5, Table 14)

### 5.2.4 Alignment tax 및 PPO-ptx의 효과

PPO가 일부 공개 NLP 태스크에서 성능 회귀를 보이며, PPO-ptx가 이를 상당 부분 완화하는 결과가 제시된다. Appendix E.6은 \(\gamma\) 증가가 회귀를 완화하는 반면, \(\beta\)만 증가시키는 것은 충분치 않음을 보여주는 ablation을 포함한다.
> **Figure 33 삽입**

(Appendix E.6, Figure 33–34, Table 14)

### 5.2.5 해설

이 결과군은 RLHF가 단일 방향의 “개선”이라기보다, 목표 분포의 선호를 올리는 과정에서 다른 능력 지표가 손상될 수 있는 **trade-off**를 명시적으로 다룬다는 점에서 중요하다. PPO-ptx는 그 trade-off를 완화하기 위한 정규화(혹은 다중 목적 최적화)로 이해할 수 있다.

---

## 5.3 Section 4.3 — Qualitative results

### 5.3.1 비영어 및 코드 예시

저자들은 비영어 instruction 및 코드 관련 프롬프트에 대한 질적 예시를 제공하며, fine-tuning 데이터에서 희소한 영역에도 instruction-following 행동이 일부 전이될 수 있음을 보인다. 동시에 일부 예시에서는 언어 혼용 등 불완전성이 나타난다.
> **Figure 42 삽입**

> **Figure 8 삽입**

(Section 4.3, Figure 8, Figure 42–43)

### 5.3.2 단순 실수(simple mistakes)

> **Figure 9 삽입**

Figure 9는 false premise를 그대로 수용하거나, 과도한 hedging, 복수 제약 준수 실패 등 단순하지만 중요한 오류를 제시한다. 저자들은 이런 문제를 adversarial data collection 등으로 완화할 가능성을 논의한다.
(Figure 9, Section 4.3)

### 5.3.3 Appendix F 샘플

Appendix F는 추가 샘플을 제공한다. 저자들은 일부 프롬프트가 behavior를 보여주기 위해 선택되었음을 밝히되, 출력은 cherry-pick하지 않았다고 기술한다. harmful prompt 사례는 “instruction following 강화가 유해 지시 수행 강화로도 이어질 수 있음”을 시사한다.
(Appendix F)

### 5.3.4 해설

질적 예시는 모델이 “새로운 능력”을 획득했다기보다, instruction-following이라는 행동 양식이 희귀 도메인(비영어/코드)에도 부분적으로 전이된다는 점을 보여준다. 동시에 harmful prompt 사례는 “도움됨”을 중심으로 최적화한 정렬이 **오용 리스크**를 자동으로 해결하지 않는다는 사실을 부각한다.

---

# 6. Discussion 해설

---

## 6.1 Section 5.1 — Implications for alignment research

### 본문 정리

저자들은 본 연구의 함의를 다음과 같이 정리한다.

- 정렬 비용(compute)은 사전학습에 비해 상대적으로 작다고 보고하며, SFT/PPO-ptx 학습 비용과 GPT-3 사전학습 비용을 비교한다.
- 특정 태스크 분포에서는 정렬이 단순 스케일 증가보다 사용자 선호 개선에 효과적일 수 있음을 시사한다.
- instruction-following 행동이 일부 도메인에 전이되는 가능성을 질적 예시로 보인다.
- PPO-ptx를 통해 alignment tax를 상당 부분 완화할 수 있음을 보이며, 이는 실제 채택 가능성 측면에서 중요하다고 주장한다.
- production-adjacent 환경에서의 empirical feedback loop의 중요성을 강조한다.
(Section 5.1)

### 해설

Section 5.1의 메시지는 “RLHF가 정답”이라기보다, 실제 배포 환경을 기준으로 목표 함수를 정하고 그에 맞춘 측정과 반복 개선을 수행해야 한다는 실증적 방법론에 가깝다.

---

## 6.2 Section 5.2 — Who are we aligning to?

### 본문 정리

저자들은 InstructGPT가 보편적 인간 가치에 정렬된 것이 아니라, **특정 라벨러 집단과 연구자, 그리고 API 고객 프롬프트 분포**의 함수로서 형성된 행동을 학습한 것임을 명시한다. 라벨러의 인구통계적 대표성, 불일치(disagreement), 그리고 “평균 라벨러 선호” 자체가 바람직한 목표인지에 대한 문제가 남는다고 논의한다.
(Section 5.2, Appendix B)

### 해설

RLHF의 “정렬 대상”은 기술적 파이프라인 밖에서 결정된다(라벨러 구성, 지침, 데이터 출처, 평가 기준). 따라서 Section 5.2는 단순 윤리 코멘트가 아니라, reward 함수 정의의 범위와 한계를 명확히 규정하는 방법론적 논의로 읽는 것이 타당하다.

---

## 6.3 Section 5.3 — Limitations

### 본문 정리

한계로는 (i) 제한된 라벨러 집단과 단일 라벨링 중심의 데이터 수집, (ii) disagreement를 충분히 반영하기 어려운 점, (iii) 모델이 여전히 사실 오류/독성/편향/성적·폭력적 내용 등에서 완전하지 않다는 점, (iv) instruction following 강화가 유해 instruction 수행 강화로 이어질 수 있다는 점 등이 제시된다.
(Section 5.3)

### 해설

이 한계들은 RLHF가 “안전성 완성”이 아니라, 목표·데이터·평가·거버넌스를 포함한 시스템 설계의 일부임을 강조한다.

---

## 6.4 Section 5.4 — Open questions

### 본문 정리

저자들은 adversarial data collection, 유해 지시 거부(refusal) 정책의 설계, control code/steering과의 결합, PPO 이외의 학습 알고리즘, 비교 피드백 외에 critique/edit 같은 더 정보량이 큰 피드백 포맷, pretraining mix의 부작용과 데이터 필터링/보강 문제, “instruction vs intention vs value”의 구분 등 다양한 열린 질문을 제시한다.
(Section 5.4)

### 해설

Section 5.4는 본 논문의 파이프라인을 “종결된 해법”이 아니라, 후속 연구가 풀어야 할 구체적 병목과 설계 선택지의 목록으로 정리한다는 점에서 가치가 크다.

---

## 6.5 Section 5.5 — Broader impacts

### 본문 정리

저자들은 본 접근이 언어모델의 긍정적 영향(도움됨/진실성/무해성 개선 가능성)을 확대할 수 있는 동시에, 오용(misuse) 리스크를 낮추지 못하거나 오히려 높일 가능성도 있음을 논의한다. 고위험 도메인(의료/법률/신용/고용/정치 등)에서의 주의, 배포 형태(API vs 오픈 배포)의 trade-off도 함께 언급한다.
(Section 5.5)

---

# 7. Appendix / Supplementary 핵심 정리

## 7.1 Appendix A — 데이터/프롬프트

bootstrap prompt 작성 방식(plain/few-shot/user-based), prompt 예시(카테고리별), 데이터 통계(Table 6–11) 등이 제공된다.
(Appendix A)

## 7.2 Appendix B — 사람 데이터 수집

라벨러 선발, 지침 변화, 인구통계 설문, 만족도 설문, UI 스크린샷 등이 포함된다.
(Appendix B)

## 7.3 Appendix C — 학습 세부

SFT/RM/RLHF 하이퍼파라미터와 초기화, PPO-ptx 구성, FLAN/T0 baseline 튜닝 등이 기술된다.
(Appendix C)

## 7.4 Appendix D — 자동평가 템플릿/규칙

데이터셋별 템플릿과 평가 규칙(샘플링/MC scoring/독성·편향 프롬프트 조건)이 상세히 제시된다.
(Appendix D)

## 7.5 Appendix E — 추가 결과/ablation

> **Table 14 삽입**

Table 14 및 Figures 28–41을 통해 alignment tax, \(\beta/\gamma\) 민감도, 학습 길이/비율/학습률 등의 영향을 분석한다.
(Appendix E)

## 7.6 Appendix F — 샘플

정렬 모델과 GPT-3 출력 비교 샘플을 제공하며, 일부 프롬프트 선택과 출력 선택의 원칙을 설명한다.
(Appendix F)

---
