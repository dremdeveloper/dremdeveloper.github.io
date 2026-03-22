---
title: "도메인마다 정답 형식이 달라진다 — Instruction Tuning for Large Language Models: A Survey — Section 6"
math: true
---

<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    processEscapes: true,
    tags: 'ams'
  },
  options: {
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
  },
  svg: { fontCache: 'global' }
};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>

# 도메인마다 정답 형식이 달라진다 — Instruction Tuning for Large Language Models: A Survey — Section 6

이 글은 *Instruction Tuning for Large Language Models: A Survey*의 **Section 6**을 중심으로, instruction tuning이 도메인마다 무엇이 달라지는지 정리하는 해설이다. 의료, 정보추출, 글쓰기, 산술, 코드처럼 과제가 바뀌면 **데이터 구성**, **정답 형식**, **평가 기준**, **실패 위험**도 함께 바뀐다. 그래서 도메인 특화 instruction tuning은 단순히 데이터를 더 넣는 문제가 아니라, **무엇을 맞는 답으로 볼지 다시 정의하는 작업**에 가깝다. 처음 읽는다면 Table 6, Figure 13, Figure 14를 먼저 보라.

## 핵심 요약
- **푸는 문제:** instruction tuning이 여러 도메인으로 확장될 때, 데이터와 정답 형식과 평가 기준은 어떻게 달라지는가.
- **핵심 아이디어:** 도메인별 대표 사례를 통해, 같은 “instruction tuning”이라도 입력/출력 구조와 학습 목표가 크게 달라진다는 점을 비교한다.
- **주요 결과:** 정보추출은 구조화 출력, 편집/글쓰기는 제약 기반 생성, 의료는 안전성과 신뢰도, 산술은 정답 정확도, 코드는 실행 가능성과 테스트 통과가 핵심이 된다.
- **왜 중요한가:** 도메인 적응은 단순한 데이터 추가가 아니라 **평가 문제를 다시 정의하는 일**이라는 점을 보여 준다.
- **한계 / 주의점:** 이 절은 개별 모델의 절대 우열보다, **도메인별 설계 차이**를 이해하는 데 초점을 맞춰 읽어야 한다.

## 먼저 읽을 포인트
| 도메인 | 주로 달라지는 것 | 읽을 때 볼 포인트 |
| --- | --- | --- |
| Information Extraction | 구조화 출력 형식 | instruction + options + output 포맷 |
| Editing / Writing | 입력 보존 + 목표 속성 변경 | 제약형 generation |
| Medical | 안전성, 사실성, 책임성 | 데이터 출처와 평가 기준 |
| Arithmetic | 정답 정확도 | 템플릿 다양화와 계산 정확도 |
| Code | 실행 가능성, 테스트 통과 | complex instruction과 expected output |

## Section 6. Domain-specific Instruction Tuning
- 원문: *Instruction Tuning for Large Language Models: A Survey* (arXiv:2308.10792v10)
- https://arxiv.org/abs/2308.10792

## 6 Domain-specific Instruction Tuning

### 6절 개요

- 본 절은 다양한 도메인 및 응용에서의 인스트럭션 튜닝 사례를 정리한다. (Section 6)
- 도메인 유형은 대화, 의도 분류·슬롯 태깅, 정보추출, 감성(ABSA), 글쓰기, 의료, 산술, 코드 등으로 제시된다. (Section 6, Table 6)

---

> **Table 6 삽입**

### Table 6. Domain-specific Instruction Fine-tuned LLMs

| 도메인 | 대표 모델 | 기반 모델 | 입력-출력 형식 | 평가 관점 |
| --- | --- | --- | --- | --- |
| Dialogue | InstructDial | T0-3B / BART0 | text-to-text 대화 응답 | unseen dialogue task, few-shot generalization |
| Intent / Slot | LINGUIST | AlexaTM 5B | intent + slot 구조화 출력 | novel intent, cross-lingual generalization |
| Information Extraction | InstructUIE | FlanT5 11B | instruction + options + text → 구조화 텍스트 | supervised / zero-shot IE |
| ABSA | IT-MTL | T5 220M | QA 템플릿 기반 factorized output | F1 중심 비교 |
| Writing / Editing | Writing-Alpaca-7B, CoEdIT | LLaMA-7B / FlanT5 | 제약 기반 편집·재작성 | edit quality, simplification, style transfer |
| Medical | Radiology-GPT, ChatDoctor, Med-ChatGLM | Alpaca-7B 등 | 임상 대화 / 의료 QA | 안전성, 사실성, 신뢰도 |
| Arithmetic | Goat | LLaMA-7B | 자연어 산술 질의 → 정확한 수치 응답 | 정답 정확도 |
| Code | WizardCoder | StarCoder 15B | complex instruction → expected output 포함 코드 | 실행 가능성, task completion |

- Table 6은 도메인별 대표 “domain-specific instruction fine-tuned LLMs”를 **모델명 / 기반 모델(Base Model) / 파라미터 수(# Params) / 학습 데이터 크기(Trainset Size)**로 요약한다. (Table 6)

> 표에서 `-` 또는 공란은 해당 항목이 Table 6에 수치로 기재되어 있지 않음을 의미한다. (Table 6)

### Table 6 각주(프로젝트/리소스 링크)

- Table 6에는 관련 코드/리소스 링크가 각주로 제시된다. (Table 6 각주)

```text
1  https://github.com/prakharguptaz/Instructdial
2  https://github.com/BeyonderXX/InstructUIE
3  https://github.com/amazon-science/instruction-tuning-for-absa
4  https://github.com/facebookresearch/EditEval
5  https://github.com/vipulraheja/coedit
6  https://github.com/vishakhpk/creative-instructions
7  https://huggingface.co/spaces/allen-eric/radiology-gpt
8  https://github.com/Kent0n-Li/ChatDoctor
9  https://github.com/SCIR-HI/Med-ChatGLM
10 https://github.com/liutiedong/goat
11 https://github.com/nlpxucan/WizardLM
```

## 6.1 Dialogue

### 개요
- 대화 도메인에서의 인스트럭션 튜닝 사례로 InstructDial을 소개한다. (Section 6.1)

### InstructDial (Gupta et al., 2022)

#### 본문 정리
- InstructDial은 대화용 인스트럭션 튜닝 프레임워크로, **59개 대화 데이터셋**에서 구성한 **48개 대화 태스크**를 **일관된 text-to-text 포맷**으로 통합한다. (Section 6.1)
- 각 태스크 인스턴스는 다음 요소를 포함한다:
  **task description, instance inputs, constraints, instructions, output**. (Section 6.1)
- 인스트럭션 준수 능력을 강화하기 위해 **2개의 메타 태스크(meta-tasks)**를 도입한다. (Section 6.1)
  1) **Instruction selection task**: 주어진 (input-output pair)에 대해 해당하는 instruction을 모델이 선택
  2) **Instruction binary task**: 특정 instruction이 주어진 input으로부터 특정 output을 유도하는지 여부를 `"yes"/"no"`로 판정
- InstructDial 태스크들로 두 기반 모델을 파인튜닝한다. (Section 6.1)
  - **T0-3B**: T0(Sanh et al., 2021)의 3B 버전(서베이에서는 T5의 3B 파라미터 버전으로 연결해 설명)
  - **BART0**: Lin et al.(2022), 406M 파라미터, Bart-large(Lewis et al., 2019) 기반
- 보지 못한(unseen) 대화 데이터셋/태스크(예: dialogue evaluation, intent detection 포함)에서 인상적인 성능을 보였으며, few-shot setting에서도 더 나은 결과를 보인다고 서술한다. (Section 6.1)

#### 해설
- 대화는 동일한 입력이라도 “요구되는 출력의 규범(톤/제약/목표)”이 instruction에 의해 달라지는 경우가 많다. InstructDial의 메타 태스크는 이러한 “instruction–출력 정합성”을 판별·선택 문제로 변환함으로써 지시 준수 학습을 강화하는 구성으로 해석할 수 있다.

---

## 6.2 Intent Classification and Slot Tagging

### 개요
- 의도 분류(intent classification) 및 슬롯 태깅(slot tagging)을 위한 LINGUIST를 소개한다. (Section 6.2)

### LINGUIST (Rosenbaum et al., 2022)

#### 본문 정리
- LINGUIST는 의도 분류 및 슬롯 태깅을 위한 instruction 데이터로 **AlexaTM 5B**(Soltan et al., 2022)를 파인튜닝한다. AlexaTM은 **다국어(multilingual) 모델**로 서술된다. (Section 6.2, Table 6)
- 각 instruction은 **5개 블록**으로 구성된다. (Section 6.2)
  1) 생성 출력의 **언어(language)**
  2) **의도(intention)**
  3) 출력에 포함할 **slot type 및 slot value**
     - 예시로 `[3, snow]`에서 `3`은 slot type, `snow`는 slot value로 설명
  4) slot type 라벨과 숫자 간 **매핑(mapping)**
  5) 출력 포맷 학습을 위한 **최대 10개 예시(examples)**
- 성능 서술은 다음과 같이 요약된다. (Section 6.2)
  - **SNIPS**(Coucke et al., 2018)에서 **10-shot novel intent** 세팅에서 SOTA 대비 큰 향상을 보임
  - **mATIS++**(Xu et al., 2020)에서 **zero-shot cross-lingual**(6개 언어) 세팅에서 강한 베이스라인인 *Machine Translation with Slot Alignment*를 능가하면서 intent classification 성능을 유지
- LINGUIST의 Trainset Size는 Table 6에서 **13K**로 제시된다.

#### 해설
- 슬롯 태깅/의도 분류는 출력 제약이 강한 구조화 과제이며, instruction 내에서 스키마(슬롯 타입, 라벨 매핑, 예시)를 명시하는 방식은 출력 포맷 안정화에 유리하게 작동할 수 있다.

---

## 6.3 Information Extraction

### 개요
- 통합 정보추출(unified information extraction)을 위한 InstructUIE를 소개한다. (Section 6.3)

### InstructUIE (Wang et al., 2023d)

#### 본문 정리
- InstructUIE는 instruction tuning 기반의 **통합(unified) IE 프레임워크**로, 다양한 IE 태스크를 **seq2seq 포맷**으로 변환해 **FlanT5 11B**를 파인튜닝한다. (Section 6.3, Table 6)
- InstructUIE는 **IE INSTRUCTIONS** 벤치마크를 도입하며, 이는 **32개** IE 데이터셋을 **통합된 text-to-text 포맷**과 **전문가 작성(expert-written) instruction**으로 구성한 것이라고 서술한다. (Section 6.3)
- 각 태스크 인스턴스는 **4가지 속성(property)**으로 구성된다. (Section 6.3)
  1) **task instruction**: 추출 대상 타입, 출력 구조(format), 추가 제약/규칙 등
  2) **options**: 출력 라벨 제약(output label constraints)
  3) **text**: 입력 문장
  4) **output**: 원래 샘플의 태그를 문장 형태로 변환한 출력
     - 예: NER의 경우 `"entity tag: entity span"` 형태로 변환
- 성능 서술은 다음과 같다. (Section 6.3)
  - **supervised setting**에서는 BERT와 유사한 성능
  - **zero-shot setting**에서는 SOTA 및 GPT-3.5를 능가

> **Figure 13 삽입**

#### Figure 13. InstructUIE 개요 프레임워크(텍스트 재서술)

- Figure 13은 InstructUIE의 overview framework를 제시하며, Wang et al.(2023d)에서 복사된 그림임이 캡션에 명시된다. (Figure 13)
- 도식은 다음의 흐름을 나타낸다. (Figure 13)
  - 좌측에 여러 IE 하위 태스크 예시가 배치됨:
    - **NER**(예: CoNLL 2003, ACE 2005)
    - **RE**(예: CoNLL 2004, SciERC, NYT11)
    - **EE**(예: CASIE, GENIA)
  - 각 태스크는 “instruction + options + text + output” 템플릿 형태로 표현됨
  - 중앙의 LLM에 대해 **Multi-Task Instruction Tuning**이 표시되며, 다태스크 인스트럭션 튜닝을 통해 단일 모델을 학습함을 나타냄
  - 하단에는 **Zero-Shot Evaluation** 및 **Unseen Datasets**가 표시되어, 보지 못한 데이터셋(예: MIT-Movie 등 표기)에서의 평가 흐름을 제시함

#### 해설
- IE 태스크(개체/관계/이벤트)는 출력 구조가 상이하지만, 이를 일관된 텍스트 생성 문제로 규격화하면 단일 seq2seq 모델로의 통합 학습이 가능해진다. 본 서베이는 InstructUIE의 핵심 설계로 instruction과 options(라벨 제약)의 분리를 강조한다.

---

## 6.4 Aspect-based Sentiment Analysis

### 개요
- ABSA(Aspect-based Sentiment Analysis)에 대한 통합 instruction tuning 사례(IT-MTL)를 소개한다. (Section 6.4)

### IT-MTL (Varia et al., 2022)

#### 본문 정리
- Varia et al.(2022)은 ABSA를 위한 통합 instruction tuning 프레임워크를 제안하며, **T5(220M)**를 기반으로 한다. (Section 6.4, Table 6)
- 해당 프레임워크는 ABSA의 다음 **4개 요소**를 포함하는 factorized sub-tasks를 다룬다. (Section 6.4)
  - Aspect Term
  - Aspect Category
  - Opinion Term
  - Sentiment
- 이를 **5개의 QA 태스크 조합**으로 취급하고, 각 문장을 태스크별 instruction 템플릿으로 변환한다. (Section 6.4)
  - 템플릿 예: `"What are the aspect terms in the text: $TEXT?"`
  - 서베이 본문은 “5개 QA 태스크”의 구체 목록을 본 절에서 열거하지는 않는다.
- 성능 서술: few-shot 학습에서 SOTA 대비 평균 **8.29 F1** 향상, full fine-tuning에서는 comparable(유사)하다고 서술한다. (Section 6.4)

#### 해설
- ABSA는 하나의 문장에서 여러 요소(대상·의견·극성)를 분리 추출해야 하므로, 요소별 질문(템플릿)으로 분해하면 단일 모델이 다양한 서브태스크를 instruction만 바꿔 수행하도록 만들 수 있다.

---

## 6.5 Writing

### 개요
- 글쓰기(writing) 및 편집(editing), 창작(creative writing)을 위한 인스트럭션 튜닝 사례로 Writing-Alpaca-7B, CoEdIT, CoPoet을 소개한다. (Section 6.5)

### 6.5.1 Writing-Alpaca-7B (Zhang et al., 2023d)

#### 본문 정리
- Zhang et al.(2023d)은 글쓰기 보조를 위해 **LLaMA-7B**를 글쓰기 instruction 데이터로 파인튜닝한 **Writing-Alpaca-7B**를 제안한다. (Section 6.5, Table 6)
- 제안 데이터셋은 **EDITEVAL(Dwivedi-Yu et al., 2022)** 벤치마크의 확장으로,
  - **Updating task를 제거**하고
  - **문법성(grammaticality) 태스크를 추가**했다고 서술한다. (Section 6.5)
- instruction 스킴은 Stanford Alpaca의 것을 **엄격히(strictly)** 따른다고 하며, 구성 요소로
  - universal preface
  - instruction field(태스크 안내)
  - input field(편집할 텍스트 제공)
  - response field(모델 출력)
  를 명시한다. (Section 6.5)
- Writing-Alpaca-7B가 LLaMA의 writing 태스크 성능을 개선하고, 더 큰 off-the-shelf LLM들보다도 낫다고 서술한다(본 절에는 정량 수치 미제시). (Section 6.5)

> 서베이 본문에서는 “LLaMa-7B (Peng et al., 2023)”와 같은 표기(인용)도 등장하며, Table 6에서는 base model을 LLaMA로 제시한다. 서베이 내에서 이 표기 차이에 대한 추가 설명은 본 절에 포함되지 않는다. (Section 6.5, Table 6)

---

### 6.5.2 CoEdIT (Raheja et al., 2023)

#### 본문 정리
- CoEdIT는 글쓰기 보조를 위해 **FlanT5(770M/3B/11B)**를 text editing instruction 데이터로 파인튜닝한다. (Section 6.5)
- 데이터셋은 약 **82K**의 ``<instruction: source, target>`` 페어로 구성된다. (Section 6.5)
- 사용자는 예를 들어 `"Make the sentence simpler"`와 같은 instruction으로 편집 목표를 지정할 수 있다. (Section 6.5, Figure 14 설명)
- CoEdIT는 다음 태스크에서 SOTA를 달성했다고 서술한다. (Section 6.5)
  - grammatical error correction
  - text simplification
  - iterative text editing
  - stylistic editing 3종: formality style transfer, neutralization, paraphrasing
- 또한 fine-tuning에 포함되지 않은 “인접한(adjacent)” 새 태스크에도 일반화 가능하다고 서술한다. (Section 6.5)

> **Figure 14 삽입**

#### Figure 14. COEDIT 개요 프레임워크(텍스트 재서술)

- Figure 14는 COEDIT의 overview framework를 제시하며, Raheja et al.(2023)에서 복사된 그림임이 캡션에 명시된다. (Figure 14)
- 도식은 다음과 같은 구조를 나타낸다. (Figure 14)
  - 좌측에 편집 태스크 예시들이 나열됨: **GEC**, **Simplification**, **Coherence**, **Formality** 등(중간 “…” 표기 포함)
  - 각 태스크는 “instruction + 원문 문장”을 입력으로 하여
  - 중앙의 “LLM (Pre-trained Instruction-tuned)”로 처리되고
  - 우측에 “편집된 문장”을 출력하는 흐름을 보임

#### 해설
- 편집 과제는 “입력 보존 + 목표 속성 변경”이라는 구조적 제약이 강하므로, instruction을 통해 목표(단순화/격식/교정 등)를 명확히 지정하는 포맷이 유효하게 작동할 수 있다.

---

### 6.5.3 CoPoet (Chakrabarty et al., 2022)

#### 본문 정리
- CoPoet은 협업 시(poetry) 글쓰기 도구로, **T5-3B, T5-11B, T0-3B** 등의 LLM을 시 instruction 데이터로 학습한다. (Section 6.5)
- 데이터는 ``<instruction, poem_line>`` 페어이며, instruction 유형은 3가지로 서술된다. (Section 6.5)
  1) **Continuation**
  2) **Lexical Constraints**
  3) **Rhetorical Techniques**
- 사용자는 “사랑에 대한 문장 쓰기”처럼 주제 지시를 주거나, “fly로 끝내라”와 같은 제약을 줄 수 있다. 또한 InstructGPT 같은 instruction 학습 LLM들과 경쟁력 있으며, 보지 못한 조합형(compositional) instruction도 만족시킬 수 있다고 서술한다. (Section 6.5)

#### 해설
- 창작 과제에서 제약(어휘/수사기법)을 명시적으로 학습하면, 복수 제약을 결합한 형태의 지시(조합 일반화)에 대한 대응력이 향상될 여지가 있다.

---

## 6.6 Medical

### 개요
- 의료 도메인에서 Radiology-GPT, ChatDoctor, ChatGLM-Med를 소개한다. (Section 6.6)

### 6.6.1 Radiology-GPT (Liu et al., 2023c)

#### 본문 정리
- Radiology-GPT는 방사선학(radiology)용으로 **Alpaca-7B** 기반 모델을 파인튜닝한 것으로 소개된다. (Section 6.6, Table 6)
- 방사선 리포트는 보통 두 섹션으로 구성된다고 서술한다. (Section 6.6)
  - **Findings**: 영상에서 관찰한 상세 소견
  - **Impression**: 소견을 바탕으로 한 해석/요약
- Findings 텍스트에 대해 `"Derive the impression from findings in the radiology report"`와 같은 instruction을 주고, 동일 리포트의 Impression을 target output으로 사용한다. (Section 6.6)
- StableLM, Dolly, LLaMA 등 일반 언어 모델과 비교했을 때 Radiology-GPT가 방사선 진단/연구/커뮤니케이션에서 유의미한 적응력을 보인다고 서술한다(정량 수치 미제시). (Section 6.6)
- Trainset Size는 Table 6에서 **122K**로 제시된다.

---

### 6.6.2 ChatDoctor (Li et al., 2023j)

#### 본문 정리
- ChatDoctor는 **LLaMA-7B** 기반이며,
  - Alpaca instruction 데이터와
  - **HealthCareMagic100k** 환자–의사 대화 데이터셋을 사용한다고 서술한다. (Section 6.6)
- 또한 프롬프트 템플릿(prompt templates)을 설계하여 대화 중 **Disease Database, Wikipedia retrieval** 등 외부 지식 DB를 검색해 더 정확한 출력을 유도한다고 서술한다. (Section 6.6)
- 결과적으로 환자 요구 이해 및 조언 제공 능력이 개선되며, 신뢰할 수 있는 온라인/오프라인 출처에서의 self-directed information retrieval을 통해 정확도가 크게 향상된다고 서술한다(본 절에서는 정량·프로토콜 미제시). (Section 6.6)
- Trainset Size는 Table 6에서 **100K**로 제시된다.

---

### 6.6.3 ChatGLM-Med (Wang et al., 2023a)

#### 본문 정리
- ChatGLM-Med는 **ChatGLM-6B** 기반이며, 중국어 의료 instruction 데이터셋으로 파인튜닝한 모델로 소개된다. (Section 6.6, Table 6)
- instruction 데이터셋은 의료 관련 QA 페어로 구성되며, **GPT-3.5 API** 및 **Medical Knowledge Graph**를 사용해 만들었다고 서술한다. (Section 6.6)
- ChatGLM의 의료 분야 QA 성능을 향상시킨다고 서술한다. (Section 6.6)
- Table 6에서 Trainset Size는 `-`로 표기되어, 본 표에는 크기가 제시되지 않는다. (Table 6)

---

## 6.7 Arithmetic

### 개요
- 산술(arithmetic) 도메인에서 Goat를 소개한다. (Section 6.7)

### Goat (Liu and Low, 2023)

#### 본문 정리
- Goat는 산술 문제 해결을 목표로 **LLaMA-7B**를 instruction 기반으로 파인튜닝한 모델로 소개된다. (Section 6.7, Table 6)
- 산술 문제를 자연어 QA 형태로 표현하며, 예로 `"What is 8914/64?"`를 제시한다. (Section 6.7)
- ChatGPT(OpenAI, 2022)를 사용해 **수백 개의 instruction 템플릿**을 생성한다고 서술한다. (Section 6.7)
- 다양한 질문 형식에 대한 적응을 위해 입력을 변형한다. (Section 6.7)
  - 숫자와 기호 사이 공백을 랜덤하게 제거
  - `"*"`를 `"x"` 또는 `"times"`로 치환
- BIG-bench의 arithmetic subtask에서 SOTA를 달성하며, 특히 **zero-shot Goat-7B**가 **few-shot PaLM-540B** 수준의 정확도에 필적하거나 초과한다고 서술한다. (Section 6.7)
- Trainset Size는 Table 6에서 **1.0M**로 제시된다.

---

## 6.8 Code

### 개요
- 코드 도메인에서 WizardCoder를 소개한다. (Section 6.8)

### WizardCoder (Luo et al., 2023)

#### 본문 정리
- WizardCoder는 **StarCoder 15B** 기반으로 “복잡한 instruction tuning(complex instruction tuning)”을 수행한 모델로 소개된다. (Section 6.8, Table 6)
- 이를 위해 **Evol-Instruct 방법(Xu et al., 2023a)**을 코드 도메인에 맞게 적응(adapt)했다고 서술한다. (Section 6.8)
- 학습 데이터는 **Code Alpaca 데이터셋**에 Evol-Instruct를 반복 적용(iterative application)하여 생성되며, 각 샘플은 **instruction, input, expected output** 속성을 갖는다. (Section 6.8)
- 예시로 SQL 쿼리를 수정해 `distinct`를 선택하도록 하는 instruction, 입력 SQL, expected output을 제시한다. (Section 6.8)
- WizardCoder가 다른 오픈소스 코드 LLM들을 능가하고, HumanEval/HumanEval+에서 Anthropic Claude, Google Bard 등 closed LLM보다도 성능이 좋다고 서술한다. (Section 6.8)
- Trainset Size는 Table 6에서 **78K**로 제시된다.

#### 해설
- 코드 과제는 출력의 실행 가능성/정확성이 중요하며, “instruction–input–expected output”의 삼분 구조는 과제 정의와 평가 가능성(테스트/정적 분석 등)을 명확히 하는 데 유리한 표준 포맷으로 활용될 수 있다.

---
