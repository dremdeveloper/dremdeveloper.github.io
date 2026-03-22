# Instruction Tuning for Large Language Models: A Survey

원문 링크: https://arxiv.org/abs/2308.10792  
기준 버전: arXiv v10

## Abstract

이 논문은 instruction tuning(IT) 연구를 체계적으로 정리한 survey다. 논문은 IT를 대규모 언어 모델이 사용자 지시를 더 잘 따르도록 만들기 위한 핵심 기법으로 보고, `(instruction, output)` 쌍으로 구성된 데이터셋에 대해 추가 학습하는 과정을 instruction tuning으로 정의한다. 이는 다음 토큰 예측 중심의 사전학습 목표와, 사용자의 지시를 유용하고 안전하게 따르는 실제 사용 목표 사이의 간극을 줄이기 위한 방법이다.

논문은 IT의 전반적인 방법론, 데이터셋 구축 방식, instruction-tuned 모델들의 학습 및 활용, 멀티모달·도메인 특화 응용, 그리고 instruction 생성 방식이나 데이터 규모 같은 성능 영향 요인을 함께 정리한다. 또한 IT의 한계, 비판점, 현재 전략의 부족한 부분, 앞으로의 연구 방향도 함께 검토한다.

## 1 Introduction

최근 몇 년 사이 대규모 언어 모델은 매우 빠르게 발전했다. GPT-3, PaLM, LLaMA 같은 모델들은 다양한 자연어 처리 과제에서 강한 성능을 보였고, 그 결과 LLM은 범용 텍스트 처리의 핵심 도구가 되었다.

하지만 논문은 LLM의 기본 학습 목표와 사용자가 원하는 목표 사이에 뚜렷한 불일치가 있다고 본다. LLM은 대규모 말뭉치에서 다음 단어를 예측하도록 학습되지만, 실제 사용자는 모델이 자신의 지시를 정확하고 유용하며 안전하게 따르기를 기대한다. Instruction tuning은 바로 이 간극을 줄이기 위해 제안된 방식이다. 사람이 작성한 지시문과 그에 대응하는 바람직한 출력으로 모델을 추가 학습시켜, 지시 이행 능력과 제어 가능성을 높인다.

논문은 IT의 장점을 세 가지로 정리한다. 첫째, instruction 데이터셋으로 미세조정하면 사전학습 목표와 사용자 목표의 차이를 줄일 수 있다. 둘째, 지시문을 통해 출력의 형식과 내용, 반응 방식에 제약을 줄 수 있어 일반 LLM보다 더 예측 가능하고 통제 가능한 모델 행동을 얻을 수 있다. 셋째, 전체 재학습이나 구조 변경 없이도 특정 도메인에 빠르게 적응할 수 있어 계산 비용 측면에서 효율적이다.

동시에 논문은 IT의 난점도 분명히 지적한다. 고품질 instruction을 충분히 다양하게 만드는 일 자체가 쉽지 않고, 기존 instruction 데이터셋은 양·다양성·창의성 모두 제한적일 수 있다. 또 instruction tuning이 실제로는 학습 데이터에 많이 포함된 과제에만 강해지는 것이 아니냐는 우려가 있으며, 모델이 과제 자체를 이해하기보다 출력 형식이나 표면 패턴만 복사하는 것 아니냐는 비판도 존재한다. 논문은 이러한 문제 때문에 instruction adherence를 더 잘 높이고, 예상 밖의 응답을 더 안정적으로 다루는 연구가 계속 필요하다고 본다.

이 survey는 바로 그 공백을 메우기 위해 작성되었다. 논문은 이후에 방법론, 대표 데이터셋, instruction-tuned LLM, 멀티모달 확장, 도메인별 적용, 효율적 튜닝 기법, 평가·분석·비판의 순서로 논의를 전개한다.

**[Figure 1 삽입 위치]**
- 넣을 것: instruction tuning의 전체 파이프라인 도식
- 원문 캡션: Figure 1: General pipeline of instruction tuning.

## 2 Methodology

논문은 이 절에서 instruction tuning에 사용되는 일반적 파이프라인을 설명한다.

### 2.1 Instruction Dataset Construction

instruction 데이터셋의 각 인스턴스는 기본적으로 세 요소로 구성된다. 첫째는 과업을 자연어로 지정하는 instruction이고, 둘째는 필요한 경우 제공되는 추가 문맥인 input이며, 셋째는 instruction과 input을 올바르게 수행한 결과인 output이다.

논문은 instruction 데이터셋을 만드는 대표적 방법을 두 가지로 나눈다. 첫 번째는 기존의 주석된 자연어 처리 데이터셋을 instruction 형식으로 변환하는 방식이다. 이 방법에서는 기존의 텍스트-라벨 쌍이나 과제용 입력-출력 쌍을 템플릿으로 감싸 `(instruction, output)` 구조로 바꾼다. Flan과 P3가 이 전략의 대표 사례다.

두 번째는 LLM을 이용해 output을 생성하는 방식이다. 이 경우 instruction은 사람이 직접 모으거나, 소수의 seed instruction에서 LLM으로 확장할 수 있다. 이후 수집된 instruction을 GPT-3.5-Turbo나 GPT-4 같은 모델에 넣어 응답을 생성하고, 이를 학습 데이터로 사용한다. InstructWild와 Self-Instruct가 이러한 접근에 속한다. 멀티턴 대화 데이터셋의 경우에는 사용자와 AI assistant 역할을 LLM이 번갈아 수행하면서 self-play 방식으로 대화 메시지를 생성할 수 있다.

### 2.2 Instruction Tuning

instruction 데이터셋이 준비되면, 사전학습된 모델을 완전지도학습 방식으로 직접 미세조정할 수 있다. 주어진 instruction과 input을 바탕으로, 모델이 output의 각 토큰을 순차적으로 예측하도록 학습하는 것이 기본 형태다.

## 3 Datasets

이 절에서는 커뮤니티에서 널리 사용되는 instruction tuning 데이터셋을 정리한다. 논문은 전체 개요를 Table 1에서 제시한 뒤, 대표 데이터셋을 순서대로 설명한다.

### 3.1 Natural Instructions

**[Figure 2 삽입 위치]**
- (a) 넣을 것: Natural Instruction dataset의 instructions 예시 이미지
- (b) 넣을 것: Natural Instruction dataset의 instances 예시 이미지
- 원문 캡션: Figure 2: The figure is adapted from Mishra et al. (2021).

Natural Instructions(Mishra et al., 2021)는 사람이 직접 작성한 영어 instruction 데이터셋으로, 61개의 서로 다른 NLP 과제에서 나온 193K 인스턴스로 구성된다. 데이터는 크게 `instructions`와 `instances`로 나뉜다.

`instructions`는 각 과제를 설명하는 메타 정보이며, 제목, 정의, 주의할 점, 프롬프트, 긍정 예시, 부정 예시 등으로 구성된다. Figure 2(a)는 이 `instructions`의 형태를 보여준다. `instances`는 실제 입력과 정답 출력으로 이루어진 `(input, output)` 쌍이며, Figure 2(b)는 그 예시다.

원 데이터는 61개 과제에 대응하는 기존 NLP 데이터셋에서 가져왔다. 저자들은 데이터셋의 주석 지침 파일을 참고해 instruction을 모았고, 각 데이터 인스턴스를 공통된 `(input, output)` 형식으로 통일해 instances를 만들었다.

### 3.2 P3

P3(Public Pool of Prompts)(Sanh et al., 2021)는 170개의 영어 NLP 데이터셋과 2,052개의 영어 프롬프트를 통합해 만든 instruction tuning 데이터셋이다. 여기서 프롬프트는 전통적인 NLP 과제의 데이터를 자연어 입력-출력 쌍으로 바꾸는 함수 역할을 한다.

P3의 각 인스턴스는 `inputs`, `answer_choices`, `targets`의 세 요소를 가진다. `inputs`는 자연어로 기술된 과제 설명이고, `answer_choices`는 가능한 응답 후보들의 목록이며, `targets`는 정답 응답이다. 저자들은 PromptSource라는 협업형 프롬프트 제작 도구를 만들고, 그 안의 여러 프롬프트 중 하나를 무작위로 골라 각 인스턴스를 통일된 구조로 변환했다.

### 3.3 xP3

xP3(Crosslingual Public Pool of Prompts)(Muennighoff et al., 2022)는 46개 언어에 걸친 16가지 자연어 과제로 구성된 다국어 instruction 데이터셋이다. 각 인스턴스는 `inputs`와 `targets` 두 요소로 구성되며, `inputs`는 자연어 과제 설명, `targets`는 그 지시에 맞는 텍스트 결과다.

xP3의 원천은 세 가지다. 영어 instruction 데이터셋 P3, P3에 없던 네 가지 영어 과제, 그리고 30개의 다국어 NLP 데이터셋이다. 저자들은 PromptSource의 사람이 작성한 템플릿을 샘플링한 뒤 이를 각 과제 데이터에 채워 넣어 다양한 NLP 과제를 통일된 형식으로 변환했다. 예를 들어 자연어 추론 과제는 전제와 가설 관계를 묻는 영어 템플릿으로 바뀌고, 원래 라벨은 yes/maybe/no 같은 자연어 응답으로 치환된다.

### 3.4 Flan 2021

Flan 2021(Longpre et al., 2023)은 SST-2, SNLI, AG News, MultiRC 같은 62개 대표 NLP 벤치마크를 자연어 입력-출력 쌍으로 변환해 만든 영어 instruction 데이터셋이다. 각 인스턴스는 `input`과 `target`으로 구성된다.

`input`은 과제를 자연어 instruction 형태로 설명한 텍스트이고, `target`은 그 지시에 맞는 텍스트 결과다. 저자들은 먼저 instruction 템플릿과 target 템플릿을 수작업으로 만들고, 그다음 데이터셋 인스턴스를 템플릿에 채워 넣어 데이터를 생성했다.

### 3.5 Unnatural Instructions

Unnatural Instructions(Honovich et al., 2022)는 InstructGPT(text-davinci-002)를 사용해 만든 약 240,000개 규모의 instruction 데이터셋이다. 각 인스턴스는 `instruction`, `input`, `constraints`, `output` 네 요소를 가진다.

`instruction`은 과제를 자연어로 설명하고, `input`은 그 과제를 구체화하는 인자이며, `constraints`는 출력 공간에 대한 제한이다. `output`은 instruction·input·constraints를 만족하는 정답 텍스트다. 저자들은 먼저 사람이 만든 Super-Natural Instructions에서 seed instruction을 샘플링한 뒤, 세 개의 seed를 예시로 넣어 InstructGPT가 새로운 `(instruction, input, constraints)` 조합을 생성하도록 했다. 이후 instruction 또는 input을 다시 바꿔 다양성을 높였고, 그 결합을 다시 InstructGPT에 넣어 output을 얻었다.

**[Table 1 삽입 위치]**
- 넣을 것: instruction tuning datasets 전체 개요 표
- 표 핵심 열: Type / Dataset Name / # of Instances / # of Tasks / # of Lang / Construction / Open-source
- 원문 표 하단 링크/각주:
  1. https://github.com/allenai/unifiedqa
  2. https://github.com/LAION-AI/Open-Instruction-Generalist
  3. https://github.com/hkunlp/unifiedskg
  4. https://github.com/allenai/natural-instructions-v1
  5. https://github.com/allenai/natural-instructions
  6. https://huggingface.co/datasets/bigscience/P3
  7. https://github.com/bigscience-workshop/xmtf
  8. https://github.com/google-research/FLAN
  9. https://github.com/BAAI-Zlab/COIG
  10. https://github.com/orhonovich/unnatural-instructions
  11. https://github.com/yizhongw/self-instruct
  12. https://github.com/XueFuzhao/InstructionWild
  13. https://github.com/nlpxucan/evol-instruct
  14. https://github.com/tatsu-lab/stanford_alpaca
  15. https://github.com/csitfun/LogiCoT
  16. https://huggingface.co/datasets/databricks/databricks-dolly-15k
  17. https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM
  18. https://huggingface.co/datasets/GAIR/lima
  19. https://huggingface.co/datasets/JosephusCheung/GuanacoDataset
  20. https://github.com/LAION-AI/Open-Assistant
  21. https://github.com/project-baize/baize-chatbot
  22. https://github.com/thunlp/UltraChat#data
- 원문 캡션: Table 1: An overview of instruction tuning datasets.

### 3.6 Self-Instruct

Self-Instruct(Wang et al., 2022c)는 InstructGPT를 사용해 만든 영어 instruction 데이터셋으로, 52K개의 학습 instruction과 252개의 평가 instruction으로 구성된다. 각 데이터는 `instruction`, `input`, `output`으로 이뤄진다. `instruction`은 과제 정의이고, `input`은 선택적으로 제공되는 보충 정보이며, `output`은 올바른 응답이다.

논문은 Self-Instruct의 생성 과정을 단계적으로 설명한다. 먼저 175개의 seed task에서 8개 instruction을 무작위로 골라 예시로 넣고, InstructGPT가 새로운 task instruction을 만들게 한다. 이후 그것이 분류 과제인지 여부를 판단해, 분류라면 가능한 출력 옵션을 생성한 뒤 해당 카테고리에 맞는 input을 만들고, 분류가 아니라면 input-first 전략 또는 output-first 전략을 사용해 input과 output을 생성한다. 마지막으로 유사 instruction 제거, 중복 제거 같은 후처리를 거쳐 최종 52K instruction을 얻는다.

### 3.7 Evol-Instruct

Evol-Instruct(Xu et al., 2023a)는 영어 instruction 학습 세트 52K와 평가 세트 218개로 이루어진 데이터셋이다. 저자들은 ChatGPT를 사용해 기존 instruction을 더 어렵고 더 다양하게 바꾸는 방식으로 데이터를 확장한다.

이때 두 가지 전략이 사용된다. 하나는 in-depth evolving으로, 제약 추가, 추론 단계 증가, 입력 복잡화 같은 방식으로 같은 instruction을 더 깊게 만든다. 다른 하나는 in-breadth evolving으로, 단순한 instruction을 더 복잡한 instruction으로 바꾸거나 아예 새로운 instruction을 생성해 다양성을 높인다. 저자들은 52K개의 초기 `(instruction, response)` 쌍에서 시작해 진화 전략을 반복 적용했고, 규칙과 ChatGPT를 이용해 품질이 낮은 결과를 걸러낸 뒤 총 250K instruction pair를 수집했다. 별도로 실제 시나리오에서 사람이 작성한 218개 instruction으로 평가 세트도 만들었다.

### 3.8 LIMA

LIMA(Zhou et al., 2023)는 1K 학습 인스턴스와 300개 테스트 인스턴스로 구성된 영어 instruction 데이터셋이다. 학습 세트는 1K개의 `(instruction, response)` 쌍으로 이뤄진다.

학습 데이터의 75%는 Stack Exchange, wikiHow, Pushshift Reddit 같은 커뮤니티 질의응답 사이트에서 뽑았고, 20%는 저자 그룹이 직접 작성했으며, 5%는 Super-Natural Instructions에서 샘플링했다. 검증 세트는 저자 작성 데이터 중 50개를 골라 만들었고, 테스트 세트 300개는 다른 저자 그룹이 작성한 예시와 Pushshift Reddit 데이터에서 구성했다. 논문은 LIMA를 통해 소량의 고품질 instruction만으로도 강한 결과를 낼 수 있다는 가설을 탐색한다.

### 3.9 Super-Natural Instructions

**[Figure 3 삽입 위치]**
- (a) 넣을 것: Super-Natural Instruction dataset의 instructions 예시 이미지
- (b) 넣을 것: Super-Natural Instruction dataset의 instances 예시 이미지
- 원문 캡션: Figure 3: The figure is adapted from Wang et al. (2022d).

Super-Natural Instructions(Wang et al., 2022d)는 1,616개의 NLP 과제와 5M개의 task instance를 포함하는 다국어 instruction 모음이다. 76개의 과제 유형과 55개 언어를 포괄한다. 각 과제는 `instruction`과 `task instances`로 구성된다.

`instruction`은 자연어 과제 정의, 긍정 예시, 부정 예시로 구성되며, Figure 3(a)가 그 형태를 보여준다. `task instances`는 텍스트 입력과 허용 가능한 출력 목록으로 구성되며, Figure 3(b)가 예시다. 원 데이터는 공개 NLP 데이터셋, 크라우드소싱 과정에서 나온 중간 주석, 그리고 수치 비교 같은 기호적 과제를 자연어로 바꾼 합성 과제에서 온다.

### 3.10 Dolly

**[Table 2 삽입 위치]**
- 넣을 것: Dolly V1의 instruction 예시 표
- 표 핵심 열: Instruction Type / Example
- 원문 캡션: Table 2: Examples of instructions in Dolly V1 (Conover et al., 2023).

Dolly(Conover et al., 2023)는 15,000개의 사람이 만든 영어 instruction 데이터 인스턴스로 구성된 데이터셋이다. 목표는 ChatGPT와 유사한 방식으로 사용자와 상호작용할 수 있는 LLM 훈련을 돕는 것이다.

이 데이터셋은 크게 7가지 과제 유형을 포함한다. open Q&A, closed Q&A, Wikipedia 기반 정보 추출, Wikipedia 기반 요약, brainstorming, classification, creative writing이 그것이다. Table 2는 각 유형의 instruction 예시를 보여준다.

### 3.11 OpenAssistant Conversations

**[Figure 4 삽입 위치]**
- 넣을 것: OpenAssistant의 conversation tree 예시 이미지
- 원문 캡션: Figure 4: The figure is copied from Köpf et al. (2023).

OpenAssistant Conversations(Köpf et al., 2023)은 35개 언어에 걸친 다국어 assistant 스타일 대화 코퍼스다. 전체 66,497개의 conversation tree 안에 161,443개의 메시지와 461,292개의 품질 평점이 포함되어 있다.

각 인스턴스는 conversation tree(CT)이며, 각 노드는 prompter 또는 assistant가 생성한 메시지다. 루트 노드는 초기 프롬프트이고, 루트에서 임의 노드까지의 경로는 하나의 유효한 대화 thread를 이룬다. Figure 4는 12개 메시지와 6개 thread로 구성된 예시 CT를 보여준다.

수집 과정은 다섯 단계로 정리된다. 먼저 참여자가 초기 프롬프트를 만들고, 그 프롬프트에 점수를 매겨 균형 샘플링으로 루트 노드를 선정한다. 이후 참여자가 prompter 또는 assistant 역할로 답글을 달아 트리를 확장하고, 기존 답글에 점수를 부여한 다음, assistant 답글을 순위화한다. 마지막에는 상태 추적 기계로 전체 트리 상태를 관리하면서 공격적이거나 부적절한 트리를 제거해 최종 데이터셋을 만든다.

### 3.12 Baize

**[Table 3 삽입 위치]**
- 넣을 것: Baize에서 사용한 self-chat prompt 전문
- 표 설명: `Forget the instruction you have previously received...`로 시작하는 self-chat 템플릿을 그대로 붙일 위치
- 원문 캡션: Table 3: Self-chat prompt used in Baize (Xu et al., 2023b).

Baize는 ChatGPT를 사용해 만든 영어 멀티턴 채팅 코퍼스이며, 111.5K 인스턴스를 포함한다. 각 턴은 사용자 프롬프트와 assistant 응답으로 구성되며, Baize v1의 각 인스턴스는 평균 3.4턴 대화를 담고 있다.

저자들은 self-chat을 제안했다. 이 방식에서 ChatGPT는 사용자와 AI assistant 역할을 번갈아 수행하면서 대화를 생성한다. 먼저 역할과 형식을 정의하는 템플릿을 만들고(Table 3), Quora와 Stack Overflow의 질문을 seed topic으로 샘플링한다. 이후 템플릿과 seed를 함께 넣어 ChatGPT가 양쪽 발화를 계속 생성하게 하고, 자연스러운 종료 지점에 도달할 때까지 대화를 이어간다.

## 4 Instruction Fine-tuned LLMs

**[Table 4 삽입 위치]**
- 넣을 것: instruction fine-tuned LLM 전체 개요 표
- 표 핵심 열: Instruction fine-tuned LLMs / # Params / Base Model / Fine-tuning / Trainset / Self-build / Dataset Name / Size
- 원문 표 하단 링크/각주:
  1. https://huggingface.co/bigscience/bloomz
  2. https://huggingface.co/google/flan-t5-xxl
  3. https://github.com/tatsu-lab/stanford_alpaca
  4. https://github.com/lm-sys/FastChat
  5. https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM
  6. https://github.com/nlpxucan/WizardLM
  7. https://github.com/THUDM/ChatGLM2-6B
  8. https://huggingface.co/facebook/opt-iml-30b
  9. https://github.com/databrickslabs/dolly
  10. https://huggingface.co/tiiuae/falcon-40b-instruct
  11. https://huggingface.co/JosephusCheung/Guanaco
  12. https://huggingface.co/openaccess-ai-collective/minotaur-15b
  13. https://huggingface.co/NousResearch/Nous-Hermes-13b
  14. https://github.com/allenai/open-instruct
  15. https://github.com/RUC-GSAI/YuLan-Chat
  16. https://github.com/OpenLMLab/MOSS
  17. https://github.com/jondurbin/airoboros
  18. https://github.com/thunlp/UltraChat
- 원문 캡션: Table 4: An overview of LLMs tuned on IT datasets.

이 절에서는 instruction fine-tuning을 거친 대표적인 LLM을 정리한다.

### 4.1 InstructGPT

InstructGPT(Ouyang et al., 2022)는 GPT-3(176B)를 기반으로 사람의 instruction에 맞추어 미세조정한 모델이다. 논문은 이 과정을 세 단계로 설명한다. 첫째, Playground API 기록에서 얻은 instruction 데이터를 사람이 걸러낸 뒤 supervised fine-tuning을 수행한다. 둘째, 하나의 instruction에 대해 여러 응답을 수집하고 사람이 순위를 매긴 데이터를 이용해 reward model을 학습한다. 셋째, 첫 단계의 모델을 reward model과 새로운 instruction으로 다시 최적화한다.

이 마지막 최적화에는 PPO가 사용된다. 두 번째와 세 번째 단계는 성능 향상이 멈출 때까지 번갈아 수행된다. 논문은 InstructGPT가 GPT-3보다 TruthfulQA의 truthfulness와 RealToxicityPrompts의 toxicity 측면에서 더 나은 결과를 보였고, WSC 같은 NLP 데이터셋에서도 비슷하거나 더 좋은 성능을 낸다고 정리한다. 사람 평가에서도 instruction 준수, 명시적 제약 준수, hallucination 감소, 적절한 응답 생성 측면에서 개선이 보고된다.

### 4.2 BLOOMZ

BLOOMZ(Muennighoff et al., 2022)는 BLOOM(176B)을 xP3 데이터셋으로 미세조정한 모델이다. xP3는 영어 instruction-영어 응답 데이터인 P3와, 영어 instruction-다국어 응답 데이터셋을 함께 포함한다.

논문에 따르면 BLOOMZ는 zero-shot 설정에서 BLOOM보다 coreference resolution, sentence completion, natural language inference 과제에서 더 좋은 성능을 보였고, HumanEval의 Pass@100과 생성 과제 BLEU 점수에서도 개선을 보였다. 즉, 다국어 instruction 데이터로 fine-tuning했을 때 zero-shot 일반화가 확실히 향상된다는 사례로 소개된다.

### 4.3 Flan-T5

Flan-T5는 T5(11B)를 기반으로 FLAN 데이터셋에 instruction fine-tuning한 모델이다. FLAN 데이터는 62개 데이터셋, 12개 NLP 과제를 통합한 `(instruction, pair)` 형식의 대규모 집합으로, 다양한 instruction 템플릿 아래서 통일된 과제 형식으로 재구성된다.

논문은 Flan-T5가 JAX 기반 T5X 프레임워크로 fine-tuning되었고, held-out task 성능을 보면서 최적 체크포인트를 고른다고 설명한다. 계산량은 T5 사전학습의 약 0.2% 수준으로 상대적으로 작다. 평가에서는 T5보다 확실히 강하고, 더 큰 모델인 PaLM과도 few-shot 설정에서 비슷하거나 일부 과제에서는 더 나은 결과를 보인다.

### 4.4 Alpaca

Alpaca(Taori et al., 2023)는 LLaMA(7B)를 InstructGPT가 생성한 instruction 데이터셋으로 미세조정한 모델이다. 논문은 8개의 80GB A100에서 약 3시간이면 학습이 가능하다고 설명한다.

사람 평가 기준으로 Alpaca는 InstructGPT(text-davinci-003)와 비슷한 수준의 결과를 보였다고 소개된다. 즉, 더 작은 오픈 모델도 적절한 instruction 데이터만 있으면 훨씬 큰 상용 모델과 경쟁 가능한 instruction-following 능력을 가질 수 있다는 사례로 제시된다.

### 4.5 Vicuna

Vicuna(Chiang et al., 2023)는 LLaMA(13B)를 ShareGPT의 사용자 공유 ChatGPT 대화 데이터로 미세조정한 모델이다. 저자들은 ShareGPT.com에서 수집한 대화 기록 중 저품질 샘플을 걸러 70K 대화 데이터를 확보했다.

멀티턴 대화를 더 잘 다루기 위해 손실 함수를 수정했고, 최대 컨텍스트 길이도 512에서 2048로 늘렸다. 학습 시에는 gradient checkpointing과 flash attention을 사용해 메모리 비용을 줄였다. 또 별도의 테스트 세트를 구성해 GPT-4에게 응답을 평가하게 했는데, 논문은 Vicuna가 Alpaca와 LLaMA를 대부분의 질문에서 앞서고, 일부 비율에서는 ChatGPT와 동등하거나 더 좋은 평가를 받았다고 보고한다.

### 4.6 GPT-4-LLM

GPT-4-LLM(Peng et al., 2023)은 LLaMA(7B)를 GPT-4가 생성한 instruction 데이터셋으로 학습한 모델이다. 논문은 이를 두 단계로 정리한다. 첫 단계에서는 Alpaca의 instruction을 사용하되 응답은 GPT-4로 수집하여 supervised fine-tuning을 수행한다. 두 번째 단계에서는 GPT-4, InstructGPT, OPT-IML의 응답을 비교해 GPT-4가 1~10점으로 채점한 데이터로 reward model을 만들고, PPO로 추가 최적화한다.

평가에서 GPT-4-LLM은 Alpaca(7B)뿐 아니라 일부 더 큰 모델보다도 나은 자동 평가 결과를 보였고, 사람 평가에서도 helpfulness, honesty, harmlessness 측면에서 개선을 보였다. 논문은 강한 teacher model의 출력이 고품질 instruction tuning에 매우 유효함을 보여주는 예시로 이 모델을 다룬다.

### 4.7 Claude

논문은 Claude를, 사전학습 언어 모델을 instruction 데이터셋으로 미세조정해 helpful하고 harmless한 응답을 생성하도록 만든 모델로 소개한다. 설명 방식은 InstructGPT와 유사한 두 단계 구조다. 먼저 instruction 데이터에 대해 supervised fine-tuning을 수행하고, 이후 여러 LLM의 응답을 평가한 비교 데이터를 이용해 reward model을 학습한 뒤 PPO 기반 최적화를 진행한다.

이 survey의 정리에 따르면 Claude는 backbone 모델보다 더 helpful하고 harmless한 응답을 생성하며, 독성 감소와 사람 평가 기준에서 개선을 보였다. 이 절은 instruction tuning과 preference optimization이 함께 결합되는 흐름을 보여주는 사례로 읽을 수 있다.

### 4.8 WizardLM

WizardLM(Xu et al., 2023a)은 Evol-Instruct 데이터셋으로 LLaMA(7B)를 fine-tuning한 모델이다. Vicuna와 공정 비교를 위해 Evol-Instruct의 70K 서브셋을 사용했고, 8개의 V100에서 3 epoch 학습에 약 70시간이 든다고 설명한다. 추론 시 최대 생성 길이는 2048이다.

복잡한 instruction 성능을 평가하기 위해 저자들은 실제 시나리오에서 수집한 218개의 인간 작성 instruction으로 Evol-Instruct test set을 만들었다. 사람 평가와 GPT-4 기반 자동 평가 모두에서 WizardLM은 Alpaca와 Vicuna를 앞섰고, 일부 비율에서는 ChatGPT에 필적하는 응답을 낸다고 보고된다.

### 4.9 ChatGLM2

ChatGLM2(Du et al., 2022)는 GLM(6B)을 영어·중국어 bilingual instruction 데이터셋으로 fine-tuning한 모델이다. 논문은 1.4T 토큰 규모의 bilingual instruction 데이터가 사용되며, 중국어와 영어 비율은 1:1이라고 설명한다. 데이터는 질의응답 및 대화 완성 과제에서 샘플링된다.

학습 전략은 InstructGPT와 유사한 3단계 구조를 따르며, 멀티턴 문맥 처리를 위해 최대 컨텍스트 길이를 1024에서 32K로 늘렸다. 또한 multi-query attention과 causal mask 전략을 도입해 메모리 비용을 줄였다. INT4 양자화 시에는 더 작은 GPU 메모리로도 긴 대화를 지원한다. 평가에서 ChatGLM2는 GLM과 기존 ChatGLM을 MMLU, C-Eval, GSM8K, BBH에서 모두 앞선다.

### 4.10 LIMA

LIMA(65B)는 LLaMA(65B)를 아주 적은 수의 고품질 instruction 데이터로 fine-tuning한 모델이다. 핵심은 superficial alignment hypothesis인데, 모델의 지식과 능력 대부분은 이미 사전학습 단계에서 형성되어 있고, alignment 학습은 그 능력을 사용자가 선호하는 형식으로 드러내는 역할을 한다는 가설이다.

이 가설을 검증하기 위해 저자들은 instruction train/valid/test 세트를 직접 구성했다. 논문은 사람 평가에서 LIMA가 InstructGPT와 Alpaca를 앞서고, BARD·Claude·GPT-4와도 비슷한 수준에 근접한다고 설명한다. GPT-4 기반 자동 평가에서도 Alpaca와 InstructGPT보다 높으며, 적은 수의 정제된 demonstration만으로도 강한 alignment가 가능하다는 주장을 뒷받침한다.

### 4.11 Others

#### OPT-IML (175B)

OPT-IML(Iyer et al., 2022)은 OPT(175B)를 IML 데이터셋으로 미세조정한 모델이다. IML 데이터셋은 PromptSource, FLAN, Super-Natural Instructions 등 공개 벤치마크에서 온 1500개 이상의 NLP 과제를 포함한다. fine-tuning 이후 OPT-IML은 여러 벤치마크 전반에서 मूल OPT를 능가한다.

#### Dolly 2.0 (12B)

Dolly 2.0은 Pythia(12B)를 databricks-dolly-15k instruction 데이터셋으로 fine-tuning한 모델이다. 이 데이터셋은 분류, 정보 추출 등 7개 범주의 NLP 과제를 포함한다. 논문은 Dolly 2.0이 Pythia보다 큰 폭으로 좋아졌고, 더 파라미터가 큰 GPT-NeoX와도 비슷한 성능을 낸다고 설명한다.

#### Falcon-Instruct (40B)

Falcon-Instruct는 Falcon(40B)을 영어 대화 데이터셋으로 미세조정한 모델이다. 학습 데이터는 Baize의 150M 토큰과 RefinedWeb의 일부 데이터로 구성된다. flash attention과 multi-query 기법으로 메모리 사용량을 줄였고, Open LLM Leaderboard에서 baseline Falcon보다 더 높은 성능을 보이며 더 큰 Guanaco보다도 좋은 결과를 냈다고 소개된다.

#### Guanaco (7B)

Guanaco는 LLaMA(7B)를 영어와 다국어 멀티턴 대화 데이터로 fine-tuning한 모델이다. Alpaca의 52K 영어 instruction pair와 534K+ 규모의 다국어 대화 데이터를 함께 사용한다. 목적은 역할에 맞는 응답과 주제 지속형 멀티턴 응답 생성을 강화하는 것이다.

#### Minotaur (15B)

Minotaur는 Starcoder Plus(15B)를 WizardLM, GPTeacher-General-Instruct 등 오픈 instruction 데이터셋으로 fine-tuning한 모델이다. 추론 시 최대 18K 토큰의 긴 컨텍스트를 지원한다. 논문은 이를 여러 오픈 instruction 소스를 혼합한 파생 모델의 한 예로 배치한다.

#### Nous-Hermes (13B)

Nous-Hermes는 LLaMA(13B)를 300K+ 규모의 instruction 데이터셋으로 fine-tuning한 모델이다. 데이터는 GPTeacher, CodeAlpaca, GPT-4-LLM, Unnatural Instructions, Camel-AI의 특정 과학 하위셋 등에서 샘플링되며, 응답은 GPT-4가 생성한다. 논문은 ARC challenge와 BoolQ 같은 여러 과제에서 GPT-3.5-turbo에 근접한 성능을 보인다고 정리한다.

#### TÜLU (6.7B)

TÜLU(Wang et al., 2023c)는 OPT(6.7B)를 FLAN V2, CoT, Dolly, Open Assistant-1, GPT4-Alpaca, Code-Alpaca, ShareGPT 등을 섞은 mixed instruction dataset으로 fine-tuning한 모델이다. 논문은 fine-tuning 이후 TÜLU가 평균적으로 ChatGPT 성능의 83%, GPT-4 성능의 68% 수준에 도달했다고 보고한다.

#### YuLan-Chat (13B)

YuLan-Chat은 LLaMA(13B)를 250,000개의 중영 bilingual instruction pair로 fine-tuning한 모델이다. 논문은 이 모델이 ChatGLM 수준에 근접하고, BBH3K 같은 영어 벤치마크에서는 Vicuna보다도 강한 결과를 보인다고 설명한다.

#### MOSS (16B)

MOSS는 멀티턴 대화와 다양한 플러그인 활용을 목표로 하는 bilingual dialogue 모델이다. 대화 instruction에 대해 fine-tuning되며, fine-tuning 이후 backbone보다 사람 선호에 더 잘 맞는 응답을 생성한다고 논문은 정리한다.

#### Airoboros (13B)

Airoboros는 LLaMA(13B)를 Self-Instruct 데이터셋으로 fine-tuning한 모델이다. 논문은 이 모델이 여러 벤치마크에서 기본 LLaMA보다 크게 향상되며, 일부 특정 벤치마크 전용 모델들과도 비교 가능한 결과를 낸다고 소개한다.

#### UltraLM (13B)

UltraLM은 LLaMA(13B)를 fine-tuning한 instruction-following 모델이다. 논문은 Dolly보다 훨씬 높은 승률을 보였고, Vicuna와 WizardLM보다도 더 높은 승률을 기록했다고 정리한다. 이 절은 오픈 instruction-tuned 모델 생태계가 매우 빠르게 파생·확장되고 있음을 보여주는 마무리 역할을 한다.

## 5 Multi-modality Instruction Fine-tuning

### 5.1 Multi-modality Datasets

**[Table 5 삽입 위치]**
- 넣을 것: 멀티모달 instruction fine-tuning 데이터셋 개요 표
- 표 핵심 열: Multi-modality Instruction Fine-tuning Dataset / Modalities / Modality Pair / # Instance / # Tasks
- 원문 표 하단 링크/각주:
  1. https://github.com/VT-NLP/MultiInstruct
  2. https://github.com/xiaoman-zhang/PMC-VQA
  3. https://github.com/OpenLAMM/LAMM
- 원문 캡션: Table 5: An overview of multi-modality instruction fine-tuning datasets.

#### MUL-TIINSTRUCT

MUL-TIINSTRUCT(Xu et al., 2022)는 62개의 다양한 멀티모달 과제를 통일된 seq-to-seq 형식으로 정리한 데이터셋이다. 10개의 상위 카테고리를 포괄하며, 21개의 공개 데이터셋에서 과제를 도출한다. 각 과제에는 전문가가 작성한 5개의 instruction이 붙는다.

기존 과제는 공개 데이터의 input/output pair를 이용해 인스턴스를 만들고, 새로운 과제는 기존 과제의 인스턴스를 재구성하거나 필요한 정보를 추출해 5K에서 5M 수준까지 확장한다. 논문은 OFA를 MUL-TIINSTRUCT로 fine-tuning하면 unseen task에 대한 zero-shot 성능이 전반적으로 향상된다고 설명한다.

#### PMC-VQA

PMC-VQA(Zhang et al., 2023c)는 149K 이미지에 대응하는 227K image-question pair를 포함하는 대규모 의료 시각 질의응답 데이터셋이다. 개방형 질의와 선택형 질의를 모두 지원하며, 다양한 질환과 시각 모달리티를 다룬다.

생성 파이프라인은 PMC-OA의 이미지-캡션 쌍을 수집하고, ChatGPT로 question-answer pair를 만든 뒤 일부를 수작업으로 검증하는 방식이다. 논문은 이 데이터셋 위에서 사전학습된 MedVInT가 VQA-RAD와 SLAKE에서 기존 모델을 능가하는 성능을 보였다고 정리한다.

#### LAMM

LAMM(Yin et al., 2023)은 2D 이미지 이해와 3D point cloud 이해를 모두 아우르는 멀티모달 instruction tuning 데이터셋이다. 186K개의 language-image instruction-response pair와 10K개의 language-point cloud instruction-response pair를 포함한다.

저자들은 공개 데이터셋의 원래 라벨을 기반으로 GPT API와 self-instruction 방법을 사용해 instruction과 response를 생성했다. LAMM은 데이터셋뿐 아니라 9개의 공통 이미지 과제와 3개의 point cloud 과제를 평가하는 LAMM-Benchmark, 그리고 서로 다른 모달리티 간 충돌을 줄이기 위해 encoder·projector·LLM fine-tuning 블록을 분리한 LAMM-Framework도 함께 제안한다.

### 5.2 Multi-modality Instruction Fine-tuning Models

**[Table 6 삽입 위치]**
- 넣을 것: 멀티모달 instruction fine-tuned LLM 개요 표
- 표 핵심 열: Multi-modality Instruction Fine-tuned LLMs / Model Name / # Params / Modality / Base Model / # Params / Fine-tuning / Self-build / Trainset Size
- 원문 표 하단 링크/각주:
  1. https://github.com/timothybrooks/instruct-pix2pix
  2. https://github.com/haotian-liu/LLaVA
  3. https://github.com/DAMO-NLP-SG/Video-LLaMA
  4. https://github.com/salesforce/LAVIS/tree/main/projects/instructblip
  5. https://github.com/Luodian/Otter
  6. https://github.com/open-mmlab/Multimodal-GPT
- 원문 캡션: Table 6: An overview of multi-modality instruction fine-tuned LLMs. I/T/V/A stand for Image/Text/Video/Audio

#### InstructPix2Pix (983M)

InstructPix2Pix(Brooks et al., 2022)는 Stable Diffusion(983M)을 멀티모달 편집 데이터셋으로 fine-tuning한 조건부 diffusion 모델이다. 학습 데이터는 450K개가 넘는 텍스트 편집 instruction과, 편집 전후 이미지 쌍으로 구성된다.

데이터 생성에는 GPT-3와 Stable Diffusion이 함께 사용된다. GPT-3는 이미지 프롬프트에 기반해 편집 instruction을 생성하고, Stable Diffusion은 그 instruction을 실제 이미지 편집 결과로 변환한다. 이후 생성된 데이터셋으로 latent diffusion objective를 사용해 모델을 학습한다. 논문은 이 모델이 SDEdit, Text2Live 같은 기존 방법과 비교했을 때 이미지 설명이 아니라 편집 instruction 자체를 더 잘 따르는 편집 결과를 낸다고 설명한다.

**[Figure 5 삽입 위치]**
- 넣을 것: image editing dataset 생성 과정과 diffusion model 학습 과정 도식
- 원문 캡션: Figure 5: Image editing dataset generation and diffusion model training. The figure is copied from Brooks et al. (2022).

#### LLaVA (13B)

LLaVA(Liu et al., 2023b)는 CLIP의 시각 인코더와 LLaMA의 언어 디코더를 연결해 만든 대규모 멀티모달 모델이다. 158K개의 vision-language instruction-following 샘플로 fine-tuning되며, 이 데이터는 conversation, detailed description, complex reasoning 프롬프트를 포함한다.

논문은 GPT-4를 이용해 image-text pair를 instruction-following 형식으로 바꾸고, caption과 bounding box 같은 시각 특징을 인코딩에 활용한다고 설명한다. 평가에서는 synthetic multimodal instruction-following 데이터에서 GPT-4 대비 높은 상대 점수를 기록했고, ScienceQA fine-tuning에서는 LLaVA와 GPT-4의 결합이 새로운 최고 정확도를 달성했다고 보고한다.

#### Video-LLaMA

Video-LLaMA(Zhang et al., 2023b)는 비디오의 시각·청각 내용을 모두 이해할 수 있도록 설계된 멀티모달 프레임워크다. 구조는 Vision-Language Branch와 Audio-Language Branch, 그리고 language decoder로 구성된다. VL branch에는 BLIP-2 기반 이미지 인코더, position embedding, video Q-former, linear layer가 들어가고, AL branch에는 ImageBind 기반 오디오 인코더와 Audio Q-former가 들어간다.

VL branch는 Webvid-2M 비디오 캡션 데이터셋으로 먼저 학습되고, 이후 MiniGPT-4, LLaVA, VideoChat 계열 instruction 데이터로 fine-tuning된다. AL branch는 video/image instru-caption 데이터로 학습된다. 논문은 이 구조를 통해 Video-LLaMA가 이미지 이해, 비디오 이해, 상식 개념 인식, 시간적 변화 파악까지 수행할 수 있다고 설명한다.

**[Figure 6 삽입 위치]**
- 넣을 것: Video-LLaMA 전체 아키텍처 도식
- 원문 캡션: Figure 6: Overall architecture of Video-LLaMA. The figure is copied from Zhang et al. (2023b).

#### InstructBLIP (1.2B)

InstructBLIP(Dai et al., 2023)는 BLIP-2를 출발점으로 하는 vision-language instruction tuning 프레임워크다. 이미지 인코더, LLM(FlanT5 또는 Vicuna), 그리고 이 둘을 연결하는 Query Transformer(Q-Former)로 구성된다.

핵심은 Q-Former가 instruction-aware visual feature를 추출해 이를 frozen LLM에 soft prompt처럼 넣는다는 점이다. 논문은 InstructBLIP를 이미지 분류, 이미지 캡셔닝, 시각 질의응답, 시각 추론 등 다양한 과제에서 평가했으며, 26개의 공개 데이터셋을 held-in과 held-out으로 나누어 훈련·평가한다. 결과적으로 여러 vision-language 과제에서 강한 zero-shot 성능을 보인다.

**[Figure 7 삽입 위치]**
- 넣을 것: InstructBLIP 전체 아키텍처 도식
- 원문 캡션: Figure 7: Overall architecture of InstructBLIP. The figure is copied from Dai et al. (2023).

#### Otter

Otter(Li et al., 2023b)는 OpenFlamingo(9B)를 기반으로 fine-tuning한 멀티모달 모델이다. 언어·비전 인코더는 고정하고, Perceiver resampler, cross-attention layer, input/output embedding만 미세조정한다.

저자들은 11개 범주의 다양한 멀티모달 과제를 구성하고, 2.8M 규모의 MIMIC-IT multimodal instruction-response pair를 구축했다. 각 샘플은 이미지에 맞춘 instruction-answer triplet과, 그것과 문맥적으로 연관된 여러 개의 context triplet을 포함한다. 논문은 Otter가 OpenFlamingo보다 사용자의 지시를 더 정확하게 따르고, 이미지에 대한 더 자세한 설명을 제공한다고 설명한다.

#### MultiModal-GPT

MultiModal-GPT(Gong et al., 2023)는 다양한 instruction을 따르고, 상세 캡션을 생성하며, 특정 객체를 세고, 일반 질의에 응답할 수 있는 멀티모달 instruction-tuned 모델이다. OpenFlamingo(9B)를 VQA, Image Captioning, Visual Reasoning, Text OCR, Visual Dialogue 같은 여러 시각 instruction 데이터로 fine-tuning한다.

논문은 실험 결과를 통해 MultiModal-GPT가 사람과 연속적인 대화를 유지할 수 있는 능력을 보인다고 정리한다.

## 6 Domain-specific Instruction Finetuning

이 절에서는 특정 도메인과 특정 응용에 맞춘 instruction tuning 사례를 정리한다.

**[Table 7 삽입 위치]**
- 넣을 것: domain-specific instruction fine-tuned LLM 개요 표
- 표 핵심 열: Domain Type / Domain-specific Instruction Fine-tuned LLMs / Model Name / Base Model / # Params / Trainset Size
- 원문 표 하단 링크/각주:
  1. https://github.com/prakharguptaz/Instructdial
  2. https://github.com/BeyonderXX/InstructUIE
  3. https://github.com/amazon-science/instruction-tuning-for-absa
  4. https://github.com/facebookresearch/EditEval
  5. https://github.com/vipulraheja/coedit
  6. https://github.com/vishakhpk/creative-instructions
  7. https://huggingface.co/spaces/allen-eric/radiology-gpt
  8. https://github.com/Kent0n-Li/ChatDoctor
  9. https://github.com/SCIR-HI/Med-ChatGLM
  10. https://github.com/liutiedong/goat
  11. https://github.com/nlpxucan/WizardLM
- 원문 캡션: Table 7: An overview of domain-specific instruction fine-tuned LLMs.

### 6.1 Dialogue

#### InstructDial

InstructDial(Gupta et al., 2022)는 대화 과제를 위해 설계된 instruction tuning 프레임워크다. 59개의 dialogue 데이터셋을 기반으로 48개의 dialogue task를 일관된 text-to-text 형식으로 재구성한다. 각 인스턴스는 task description, instance inputs, constraints, instructions, output으로 이뤄진다.

instruction 준수를 강화하기 위해 두 가지 메타 과제도 도입한다. 하나는 주어진 input-output pair에 맞는 instruction을 고르는 instruction selection task이고, 다른 하나는 특정 instruction이 주어진 input에서 해당 output을 만들 수 있는지를 yes/no로 판별하는 instruction binary task다. 논문은 T0-3B와 BART0를 이 데이터로 fine-tuning했을 때 unseen dialogue dataset과 few-shot setting에서도 좋은 결과를 보였다고 설명한다.

### 6.2 Intent Classification and Slot Tagging

#### LINGUIST

LINGUIST(Rosenbaum et al., 2022)는 AlexaTM 5B라는 50억 파라미터 규모의 다국어 모델을 intent classification과 slot tagging용 instruction 데이터셋으로 fine-tuning한 방법이다.

각 instruction은 다섯 블록으로 구성된다. 출력 언어, 의도 정보, 출력에 포함해야 할 slot type과 value, slot type label을 숫자에 매핑하는 규칙, 그리고 최대 10개의 형식 예시가 그것이다. 논문은 LINGUIST가 SNIPS의 novel intent 설정과 mATIS++의 zero-shot cross-lingual 설정에서 강한 baseline을 앞선다고 정리한다.

**[Figure 8 삽입 위치]**
- 넣을 것: InstructUIE 전체 프레임워크 도식
- 원문 캡션: Figure 8: The overview framework of InstructUIE. The figure is copied from Wang et al. (2023b).

### 6.3 Information Extraction

#### InstructUIE

InstructUIE(Wang et al., 2023b)는 정보 추출 과제를 seq2seq 형식으로 통일해 instruction tuning으로 해결하는 프레임워크다. 11B FlanT5를 기반으로 하며, 32개의 다양한 정보 추출 데이터셋을 text-to-text 형식으로 통일한 IE INSTRUCTIONS benchmark를 도입한다.

각 인스턴스는 task instruction, options, text, output의 네 속성으로 구성된다. task instruction에는 추출해야 할 정보의 종류, 출력 구조, 추가 제약이 들어가고, options는 출력 라벨 제약을 나타내며, text는 입력 문장, output은 원래 태그를 문장형 출력으로 바꾼 결과다. 논문은 supervised setting에서는 BERT와 비슷한 수준을 보이고, zero-shot setting에서는 기존 SOTA와 GPT-3.5보다 더 나은 결과를 보였다고 설명한다.

### 6.4 Aspect-based Sentiment Analysis

#### Varia et al. (2022)

이 연구는 T5(220M)를 fine-tuning해 Aspect-based Sentiment Analysis(ABSA)를 해결하는 통합 instruction tuning 프레임워크를 제안한다. ABSA를 구성하는 Aspect Term, Aspect Category, Opinion Term, Sentiment를 여러 하위 과제로 분해한 뒤, 이를 다섯 개의 Question Answering 과제로 바꿔 instruction template 기반으로 학습한다.

예를 들어 `"What are the aspect terms in the text: $TEXT?"` 같은 템플릿이 사용된다. 논문은 few-shot learning 상황에서 평균 8.29 F1만큼 기존 최고 성능을 넘어서고, full fine-tuning 상황에서도 경쟁력 있는 결과를 유지한다고 정리한다.

### 6.5 Writing

#### Zhang et al. (2023d)

이 연구는 Writing-Alpaca-7B를 제안한다. LLaMA-7B를 writing instruction 데이터셋으로 fine-tuning해 writing assistance에 활용하는 방식이다. 데이터셋은 EDITEVAL을 instruction 형식으로 확장한 것으로, Updating 과제는 제거하고 grammaticality 과제를 새롭게 추가했다.

instruction 형식은 Stanford Alpaca 프로젝트의 스키마를 그대로 따른다. 공통 preface, 과제 수행을 지시하는 instruction field, 편집할 텍스트를 제공하는 input field, 모델이 채워야 하는 response field로 구성된다. 논문은 Writing-Alpaca-7B가 기본 LLaMA보다 모든 writing 과제에서 더 좋고, 더 큰 일반 LLM들보다도 우수한 결과를 낸다고 설명한다.

**[Figure 9 삽입 위치]**
- 넣을 것: COEDIT 전체 프레임워크 도식
- 원문 캡션: Figure 9: The overview framework of COEDIT. The figure is copied from Raheja et al. (2023).

#### CoEdIT

CoEdIT(Raheja et al., 2023)는 FLAN-T5의 770M, 3B, 11B 버전을 text editing instruction 데이터셋으로 fine-tuning한 writing assistance 모델이다. 데이터셋은 약 82K개의 `<instruction: source, target>` pair로 구성된다.

사용자는 `"Make the sentence simpler"`처럼 원하는 편집 특성을 instruction으로 지정하고, 모델은 그에 맞는 편집 결과를 출력한다. 논문은 CoEdIT가 grammatical error correction, text simplification, iterative text editing, formality style transfer, neutralization, paraphrasing 등 여러 과제에서 state-of-the-art를 달성하고, 학습하지 않은 인접 과제로도 잘 일반화한다고 설명한다.

#### CoPoet

CoPoet(Chakrabarty et al., 2022)은 협업 시 쓰기 도구로, 시 창작 instruction에 대해 학습된 T5-3B, T5-11B, T0-3B 계열 모델을 활용한다. 각 샘플은 `<instruction, poem_line>` pair로 이루어지며, instruction 유형은 Continuation, Lexical Constraints, Rhetorical Techniques의 세 가지가 핵심이다.

사용자는 `"love"`에 관한 문장을 쓰라거나 `"fly"`로 끝나는 행을 쓰라는 식으로 시적 제약을 instruction으로 제공할 수 있다. 논문은 CoPoet이 InstructGPT 같은 instruction-tuned LLM과 경쟁력 있는 수준을 보이면서도, 학습 시 보지 못한 compositional instruction도 만족시킬 수 있다고 본다.

### 6.6 Medical

#### Radiology-GPT

Radiology-GPT(Liu et al., 2023c)는 Alpaca-7B를 방사선학 도메인으로 fine-tuning한 모델이다. 방사선학 보고서는 보통 `Findings`와 `Impression` 두 부분으로 나뉘는데, Findings에는 상세 관찰 내용이, Impression에는 최종 해석이 들어간다.

Radiology-GPT는 Findings 텍스트에 대해 `"Derive the impression from findings in the radiology report"`라는 간단한 instruction을 주고, 동일 보고서의 Impression을 target output으로 삼아 학습한다. 논문은 Radiology-GPT가 일반 언어 모델들보다 방사선학 진단, 연구, 커뮤니케이션에 더 잘 적응한다고 설명한다.

#### ChatDoctor

ChatDoctor(Li et al., 2023g)는 LLaMA-7B를 Alpaca instruction 데이터와 HealthCareMagic100k 환자-의사 대화 데이터로 fine-tuning한 의료 대화 모델이다. 여기에 질병 데이터베이스나 Wikipedia 같은 외부 지식 검색용 prompt template도 설계해, doctor-patient conversation 중 더 정확한 출력을 유도한다.

논문은 ChatDoctor가 환자의 요구를 이해하고 조언을 제공하는 능력을 크게 향상시키며, 신뢰 가능한 온라인·오프라인 지식원으로부터 자기주도적으로 정보를 검색하게 했을 때 응답 정확도도 높아진다고 정리한다.

#### ChatGLM-Med

ChatGLM-Med는 ChatGLM-6B를 중국어 의료 instruction 데이터셋으로 fine-tuning한 모델이다. 이 instruction 데이터셋은 GPT-3.5 API와 Medical Knowledge Graph를 이용해 만든 의료 질의응답 쌍으로 구성된다.

논문은 이 모델이 의료 분야에서 ChatGLM의 질의응답 성능을 향상시킨다고 설명한다.

### 6.7 Arithmetic

#### Goat

Goat(Liu and Low, 2023)는 LLaMA-7B를 arithmetic 문제 해결용 instruction 데이터로 fine-tuning한 모델이다. 산술 문제를 `"What is 8914/64?"`처럼 자연어 질문응답 형태로 표현하고, ChatGPT를 이용해 수백 개의 instruction template를 생성한다.

또한 숫자와 기호 사이 공백을 무작위로 제거하거나, `*`를 `x` 또는 `times`로 바꾸는 식으로 질문 형식 다양성에 대한 적응력을 높인다. 논문은 Goat가 BIG-bench arithmetic subtask에서 state-of-the-art를 달성하고, zero-shot Goat-7B가 few-shot PaLM-540B와 맞먹거나 더 나은 정확도를 보였다고 설명한다.

### 6.8 Code

#### WizardCoder

WizardCoder(Luo et al., 2023)는 StarCoder 15B를 기반으로, Evol-Instruct 방식을 코드 도메인에 맞게 변형한 code instruction fine-tuning 모델이다. 학습 데이터는 Code Alpaca 데이터셋에 Evol-Instruct를 반복 적용해 만든다. 각 샘플은 instruction, input, expected output을 포함한다.

예를 들어 instruction이 `"Amend the following SQL query to select distinct elements"`라면, input은 SQL query이고 expected output은 수정된 정답 쿼리다. 논문은 WizardCoder가 다른 오픈소스 코드 LLM을 앞서고, HumanEval과 HumanEval+에서는 일부 대형 폐쇄형 LLM보다도 더 좋은 결과를 냈다고 정리한다.

## 7 Efficient Tuning Techniques

이 절은 LLM을 instruction 데이터에 적응시키되 전체 파라미터를 모두 학습하지 않거나, 메모리와 계산 비용을 더 줄이는 방향의 효율적 튜닝 기법들을 정리한다. 논문은 이를 addition-based, specification-based, reparameterization-based로 구분한다. addition-based는 원래 모델에 없던 trainable module이나 parameter를 추가하는 방식이고, specification-based는 특정 파라미터만 선택적으로 학습하는 방식이며, reparameterization-based는 가중치를 더 효율적인 형태로 다시 표현해 학습하는 방식이다.

### 7.1 LoRA

LoRA(Hu et al., 2021)는 저랭크 업데이트를 이용해 LLM을 효율적으로 적응시키는 방법이다. 핵심 가정은 새로운 과제에 적응할 때 실제로 필요한 가중치 변화가 저차원 부분공간에 존재한다는 것이다. 따라서 전체 가중치를 직접 업데이트하는 대신, 작은 두 행렬로 분해된 low-rank update만 학습한다.

논문은 LoRA가 DeepSpeed를 학습 백본으로 사용하며, GPT-3 기준으로 full fine-tuning 대비 학습해야 하는 파라미터 수를 10,000배 줄이고 메모리 사용량도 3배 절감한다고 설명한다. 즉, 성능을 크게 해치지 않으면서 instruction tuning 비용을 크게 줄일 수 있는 대표 기법으로 제시된다.

### 7.2 HINT

HINT(Ivison et al., 2022)는 instruction tuning의 일반화 이점과 on-demand efficient tuning을 결합하는 방법이다. 긴 instruction을 매번 반복 처리하지 않기 위해 hypernetwork를 사용한다는 점이 핵심이다.

hypernetwork는 자연어 instruction과 few-shot example을 받아 encoded instruction을 만들고, 이를 바탕으로 adapter와 prefix parameter를 생성한다. 생성된 parameter-efficient module은 backbone 모델에 삽입된다. 추론 시에는 과제마다 한 번만 hypernetwork를 실행해 적응 모듈을 만들면 되므로, 긴 instruction이나 추가 few-shot 예시를 포함하더라도 계산량 증가를 억제할 수 있다.

### 7.3 QLoRA

QLoRA(Dettmers et al., 2023)는 양자화와 메모리 최적화를 결합해 instruction tuning을 더 효율적으로 만드는 기법이다. 핵심은 4-bit NormalFloat(NF4) 양자화인데, 이는 LLM 가중치가 대체로 정규분포를 따른다는 점을 이용해 설계된 4비트 표현 방식이다.

또한 양자화 상수 자체도 다시 8비트로 양자화해 메모리를 더 줄이고, unified memory를 이용해 optimizer state를 CPU RAM으로 페이지 아웃함으로써 OOM을 방지한다. QLoRA는 4비트로 양자화된 base LLM은 고정하고, 그 위에 소수의 16비트 low-rank adapter만 학습한다. 논문은 이를 통해 65B 모델도 단일 48GB GPU에서 full 16-bit fine-tuning과 큰 성능 차이 없이 학습할 수 있다고 설명한다.

### 7.4 LOMO

LOMO(Lv et al., 2023)는 제한된 계산 자원에서도 LLM full-parameter fine-tuning을 가능하게 하려는 low-memory optimization 방법이다. 핵심 아이디어는 역전파 중 gradient 계산과 parameter update를 하나의 단계로 융합해, 전체 gradient tensor를 저장하지 않도록 만드는 것이다.

각 parameter tensor의 gradient를 계산하자마자 즉시 업데이트하면 한 번에 한 tensor의 gradient만 저장하면 되므로 메모리 사용량이 크게 줄어든다. 논문은 여기에 gradient clipping, gradient norm 분리 계산, dynamic loss scaling, activation checkpointing, ZeRO 최적화 같은 기법을 함께 결합해 안정성과 메모리 절약을 확보한다고 설명한다.

### 7.5 Delta-tuning

Delta-tuning(Ding et al., 2023b)은 optimization과 optimal control의 관점에서 효율적 fine-tuning을 해석하는 접근이다. 직관적으로는 학습을 저차원 manifold로 제한해 subspace optimization을 수행한다.

논문은 tuning된 parameter를 downstream task에서 모델 행동을 유도하는 optimal controller처럼 바라본다. 즉, 전체 모델을 크게 바꾸는 대신 작은 delta만 잘 조정해도 instruction-tuned 적응이 가능하다는 관점이다.

## 8 Evaluation, Analysis and Criticism

### 8.1 HELM Evaluation

HELM(Liang et al., 2022)은 language model의 능력, 위험, 한계를 더 투명하게 이해하기 위한 holistic evaluation 프레임워크다. 논문은 HELM이 instruction-tuned 모델 평가에 중요한 이유를 세 가지로 설명한다.

첫째는 broad coverage다. HELM은 ACL 2022에 등장한 과제를 기반으로 top-down taxonomy를 구축하고, 각 과제를 시나리오와 metric의 조합으로 분해해 더 넓은 평가 공간을 다룬다. 이를 통해 language model 평가의 시나리오 포괄 범위를 크게 늘린다. 둘째는 multi-metric measurement다. HELM은 하나의 점수 대신 여러 관점에서 모델을 평가하도록 설계되었다. 셋째는 standardization이다. HELM은 다양한 기관의 대표 language model을 같은 틀 안에 넣어 비교할 수 있게 한다. 논문은 이처럼 IT 모델도 다양한 과제와 지표 위에서 비교되어야 한다고 본다.

### 8.2 Low-resource Instruction Tuning

Gupta et al. (2023)은 IT 모델이 SOTA supervised model과 비슷해지기 위해 필요한 최소한의 downstream training data가 얼마인지 추정하려 했다. 이를 위해 Super-Natural Instructions의 119개 과제를 single-task learning과 multi-task learning 설정에서 실험했다.

논문은 STL에서는 전체 downstream training data의 25%만 있어도 IT 모델이 기존 SOTA를 넘는 경우가 많고, MTL에서는 6% 수준만으로도 SOTA에 도달할 수 있다고 정리한다. 즉, instruction tuning은 적은 데이터에서도 새로운 과제를 빠르게 익히는 데 도움이 될 수 있다는 것이다. 다만 더 큰 LLM과 더 큰 데이터셋으로는 아직 추가 검증이 필요하다고 본다.

### 8.3 Smaller Instruction Dataset

논문은 instruction tuning이 대규모 specialized instruction 데이터를 꼭 필요로 하는지에 대해서도 논의한다. Zhou et al. (2023)은 LLM이 실제로는 사용자와 상호작용하는 style이나 format만 배우면 충분할 수 있다고 가정하고, 1,000개의 carefully selected example만으로도 강한 성능을 내는 LIMA를 제안했다.

LIMA는 수작업으로 선별한 1,000개의 고품질 demonstration으로 LLaMA 65B를 fine-tuning한다. 논문은 LIMA가 300개 이상의 까다로운 과제에서 GPT-davinci-003보다 낫고, demonstration 수를 절반으로 줄여도 GPT-4, Claude, Bard 수준에 근접하는 결과를 낸다고 소개한다. 이를 통해 강력한 기반 모델의 지식이 잘 정제된 소수 instruction만으로도 충분히 드러날 수 있다는 해석이 가능하다고 본다.

### 8.4 Evaluating Instruction-tuning Datasets

IT 모델 성능은 어떤 instruction 데이터셋으로 학습했는지에 크게 좌우된다. 그러나 기존에는 개방형·주관적 측면까지 포함하는 instruction dataset 평가가 충분하지 않았다.

Wang et al. (2023c)은 여러 오픈 instruction 데이터셋으로 LLaMA를 각각 fine-tuning하고, 자동 평가와 사람 평가를 함께 수행했다. 또 데이터셋들을 수작업으로 조합한 혼합 모델도 만들었다. 논문은 모든 과제에서 항상 가장 좋은 단일 데이터셋은 없지만, 데이터셋을 잘 조합하면 전체적으로 가장 좋은 성능을 낼 수 있다고 정리한다. 또한 instruction tuning의 이득은 모델 크기 전반에서 나타나지만, 특히 작은 모델과 base quality가 높은 모델이 더 큰 이득을 얻는 경향이 있으며, 사람 평가에서는 더 큰 모델이 acceptability 점수를 더 잘 받는다고 설명한다.

### 8.5 Do IT just learn Pattern Copying?

Kung and Peng (2023)은 instruction tuning 동안 모델이 실제로 무엇을 배우는지를 분석한다. 저자들은 의미 정보는 제거하고 출력 형식 정보만 남긴 단순화된 task definition, 그리고 잘못된 input-output mapping을 담은 delusive example을 사용해 원래 instruction과 비교 실험을 수행했다.

놀랍게도 모델은 이런 단순화된 instruction이나 잘못된 예시로 학습해도 원래 instruction으로 학습한 모델과 비슷한 성능을 내는 경우가 있었다. 또 분류 과제에서는 zero-shot baseline이 low-resource IT와 비슷한 성능을 보이기도 했다. 논문은 이를 근거로, 현재 IT 모델의 성능 향상 중 일부는 과제 자체의 이해보다 출력 형식 복사와 추측 같은 표면 패턴 학습에서 올 수 있다고 정리한다.

### 8.6 Proprietary LLMs Imitation

논문은 더 강한 proprietary model의 출력을 수집해 open model을 fine-tuning하는 imitation 접근도 비판적으로 다룬다. Gudibande et al. (2023)은 ChatGPT 출력을 모아 1.5B부터 13B까지 다양한 크기의 GPT-2 및 LLaMA 계열 모델을 fine-tuning하고, 데이터 양도 0.3M token부터 150M token까지 바꿔가며 실험했다.

실험 결과, imitation dataset이 잘 지원하는 과제에서는 모델이 훨씬 좋아지고 출력도 ChatGPT처럼 보이지만, imitation 데이터가 없는 과제에서는 거의 개선이 없거나 오히려 정확도가 떨어지는 경우가 있었다. 논문은 이 현상이 open model이 ChatGPT의 문체, 자신감 있는 어조, 구조화된 답변 양식은 잘 모방하지만 일반 능력 자체가 획기적으로 올라간 것은 아닐 수 있음을 시사한다고 본다. 따라서 단순 imitation보다 base model 품질과 instruction example 품질을 높이는 쪽이 더 중요할 수 있다고 정리한다.

## 9 Conclusion

논문은 빠르게 성장 중인 instruction tuning 분야의 최근 진전을 폭넓게 정리한다. 방법론, 데이터셋 구축, instruction-tuned 모델, 멀티모달 적용, 도메인 특화 적용, 효율적 튜닝, 평가와 한계까지 한 흐름으로 묶어 보여 준다는 점이 핵심이다.

저자들은 IT가 LLM의 능력과 제어 가능성을 높이는 강력한 수단이지만, 데이터 품질과 다양성, 실제 일반화 능력, 표면 패턴 학습에 대한 비판 등 아직 해결해야 할 문제가 많다고 본다. 이 survey가 앞으로의 연구를 자극하고, 현재 IT 전략의 부족한 점을 보완하는 후속 연구로 이어지기를 기대한다.

## References

원문 References 전체를 붙일 위치.

## 추가 설명

- 이 문서는 사용자가 제공한 arXiv abs 링크가 현재 가리키는 버전(arXiv v10)을 기준으로 재작성했다.
- Figure와 Table은 실제 이미지·표를 직접 가져오지 않고, 원문 위치에 맞춰 삽입 위치와 원문 캡션만 표시했다.
- 현재 arXiv HTML에는 도입부 개요 bullet의 section 번호가 실제 본문 구조와 일부 어긋나는 부분이 있다. 이 문서에서는 본문 실제 전개 순서를 기준으로 정리했다.
- arXiv HTML 변환본에는 일부 고유명사의 철자나 표기가 거칠게 보이는 부분이 있어, 본문 재작성에서는 통상적으로 알려진 모델명·데이터셋명 표기를 사용했다.
