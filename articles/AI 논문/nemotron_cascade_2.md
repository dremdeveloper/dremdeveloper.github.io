---
title: nemotron_cascade_2
math: true
---

# Nemotron-Cascade 2: Post-Training LLMs with Cascade RL and Multi-Domain On-Policy Distillation

## Abstract

이 논문은 **Nemotron-Cascade 2**를 소개한다. 모델은 총 30B 규모의 Mixture-of-Experts 구조이지만 실제 추론 시 활성화되는 파라미터는 3B에 불과하다. 저자들의 핵심 주장은 단순히 “작고 빠른 모델”을 제시하는 데 있지 않다. 이 모델이 수학 추론, 코드 추론, 장문 문맥 처리, 에이전트형 작업, 인간 선호 정렬까지 아우르는 폭넓은 사후학습 과정을 통해, 훨씬 큰 공개 모델과 맞설 수 있을 만큼 높은 지능 밀도를 달성했다는 점이 논문의 출발점이다.

저자들은 특히 세 가지 성과를 전면에 내세운다. 첫째, Nemotron-Cascade 2는 2025년 IMO, IOI, ICPC World Finals에서 모두 골드 메달 수준 결과를 낸 두 번째 오픈 웨이트 모델이다. 둘째, 그 과정에서 이전 Nemotron-Cascade 1보다 훨씬 넓은 영역으로 확장된 Cascade RL을 사용한다. 셋째, 각 단계에서 가장 강했던 중간 정책을 교사로 삼아 **Multi-Domain On-Policy Distillation(MOPD)**을 수행함으로써, RL 단계 사이에 생기는 성능 회귀를 효율적으로 복구한다.

이 논문이 전달하려는 요지는 분명하다. 강한 추론 모델을 만드는 일은 더 큰 기저 모델을 마련하는 것만으로 끝나지 않는다. 어떤 순서로 어떤 능력을 올리고, 그 과정에서 생기는 간섭과 망각을 어떻게 제어하며, 이미 잘하는 능력을 잃지 않으면서 새 능력을 더할 것인지가 성패를 가른다. Nemotron-Cascade 2는 바로 그 “사후학습 설계”를 논문의 중심 문제로 삼는다.

## 1. Introduction

도입부는 최근의 오픈 모델 생태계가 단순 질의응답을 넘어, 수학 문제 풀이, 경쟁 프로그래밍, 장문 문맥 추론, 다중 턴 도구 사용, 소프트웨어 엔지니어링 같은 복합 작업으로 빠르게 확장되고 있다는 문제의식에서 시작한다. 이때 모델의 품질을 결정하는 것은 파라미터 수 그 자체보다, **사후학습 단계가 어떤 능력을 어떤 순서로 얼마나 정교하게 끌어올렸는가**라는 점이라고 저자들은 본다.

Nemotron-Cascade 1이 이미 Cascade RL이라는 아이디어를 통해, 특정 영역에서의 강화학습을 순차적으로 쌓아 올리는 방식이 효과적임을 보여 주었다면, Nemotron-Cascade 2는 그 개념을 한 단계 더 밀어붙인다. 이번 논문에서 Cascade RL은 단순한 “여러 RL 단계의 나열”이 아니다. 각 능력 영역이 서로를 도와주는지, 방해하는지, 먼저 올려야 할 능력과 나중에 다듬어야 할 능력이 무엇인지를 하나의 설계 문제로 다룬다. 또한 RL 단계 사이에서 일부 벤치마크가 떨어지는 현상이 실제로 관찰되기 때문에, 그 하락을 다시 회복시키기 위한 별도의 안정화 기법으로 MOPD를 도입한다.

도입부의 또 다른 강조점은 **지능 밀도(intelligence density)**다. DeepSeek-V3.2-Speciale 같은 훨씬 큰 공개 모델과 비교할 때, Nemotron-Cascade 2는 활성 파라미터 수를 크게 줄인 상태에서도 수학·코딩·에이전트 성능을 높게 유지한다. 즉, 저자들이 설계한 사후학습 절차는 “작은 활성 모델이 어떻게 큰 모델의 추론력을 압축해서 흡수할 수 있는가”라는 질문에 대한 실험적 답변으로 제시된다.

논문은 마지막으로 세 가지 공개물을 함께 내놓는다고 밝힌다. 첫째는 최종 post-trained 모델 체크포인트, 둘째는 SFT 데이터 모음, 셋째는 RL 데이터 모음이다. 이는 단순 성능 보고를 넘어, 사후학습 파이프라인 자체를 재현 가능한 자산으로 공개한다는 의미를 갖는다.

## 2. Main Results

<!-- Table 1 -->

> [Table 1 삽입 위치]

Section 2는 논문의 전체 주장을 압축해서 보여 주는 결과 요약부다. Table 1은 Nemotron-3-Nano, Nemotron-3-Super, Qwen3.5 계열, 그리고 Nemotron-Cascade 2를 한 표에 두고, 수학, 코드 추론, 지식·STEM, 정렬·지시수행, 장문 문맥, 에이전트, 다국어 영역을 모두 묶어서 비교한다. 이 표의 핵심은 Nemotron-Cascade 2가 특정 한두 벤치마크에서만 강한 모델이 아니라는 점이다. IMO 계열 수학 벤치마크, LiveCodeBench와 Codeforces 계열의 경쟁 코딩, ArenaHard·IFBench 같은 정렬/지시 수행, BFCL·τ2-Bench·SWE Verified 같은 에이전트 작업까지, 매우 다른 평가 셋에서 상단권 성능을 낸다.

특히 표에서 괄호로 병기된 수치는 Tool-Integrated Reasoning, 즉 상태를 유지하는 Python 실행기를 사용할 수 있는 설정의 결과다. 이 표를 읽는 데서 중요한 점은, Nemotron-Cascade 2가 **도구 없이도 이미 강한 모델**이지만, 도구를 붙였을 때 특정 수학·코딩 과제에서 추가 상승 여지가 분명하다는 사실이다. 저자들은 이 차이를 통해 모델 자체의 추론력과 도구 결합형 추론의 시너지를 분리해서 보여 준다.

또한 Table 1은 Nemotron-Cascade 2가 단순히 “최종 모델 하나가 세다”는 주장으로 끝나지 않게 한다. 같은 base 계열에서 출발한 Nemotron-3-Nano와 비교했을 때도 거의 전 영역에서 후속 사후학습의 이득이 드러나고, Nemotron-3-Super처럼 더 큰 모델과의 비교에서도 일부 영역에서는 근접하거나 앞서는 모습을 보인다. 이 때문에 Table 1은 모델 크기보다 **post-training recipe**가 더 결정적인 차이를 만들 수 있다는 논문의 핵심 논지를 실증적으로 지탱한다.

<!-- Table 2 -->

> [Table 2 삽입 위치]

Table 2는 논문의 가장 눈에 띄는 메시지, 즉 “골드 메달 수준”을 정량적으로 보여 준다. IMO 2025에서는 42점 만점에 35점을 얻어 골드, IOI 2025에서는 600점 만점에 439.28점을 얻어 골드, ICPC World Finals 2025에서는 12문제 중 10문제를 해결해 골드 수준 성과를 보고한다. 저자들은 IMO 2025의 Problem 2는 풀이가 장문의 해석기하 성격을 띠기 때문에, ProofBench 계열의 마킹 스키마를 적용한 LLM 채점과 인간 전문가 검토를 함께 사용했다고 밝힌다.

이 표가 중요한 이유는 세 대회의 성격이 완전히 다르기 때문이다. IMO는 자연어 수학 증명, IOI는 정답 프로그램과 부분점수 구조를 가진 알고리즘 코딩, ICPC World Finals는 제한된 제출과 다문제 라운드 운영을 가진 팀형 경쟁 프로그래밍에 가깝다. Nemotron-Cascade 2는 이 셋 모두에서 높은 성능을 보였고, 이는 한 종류의 보상 신호나 한 종류의 프롬프트만으로 얻은 성과가 아니라, 논문 전체가 설명하는 SFT–Cascade RL–MOPD–RLHF–Code RL–SWE RL의 연결된 설계 덕분이라는 것이 저자들의 주장이다.

Section 2 말미의 서술은 이 수치를 과장 없이 해석한다. 저자들은 Nemotron-Cascade 2가 30B-A3B 규모임에도 매우 강한 모델이지만, 성능은 “한 번의 비법”으로 나온 것이 아니라 각 능력 영역을 단계적으로 쌓은 결과라고 정리한다. 그래서 이후 섹션 3과 4는 단순한 데이터 목록이 아니라, 이 최종 결과가 어떤 순서의 학습과 안정화 과정을 거쳐 형성되었는지를 설명하는 본론 역할을 하게 된다.

## 3. Supervised Fine-Tuning

Section 3은 전체 post-training의 첫 단계인 SFT를 다룬다. 여기서 저자들의 기본 입장은 명확하다. 강화학습이 아무리 강하더라도, 그 앞단에 놓인 SFT가 형식, 응답 습관, 추론 모드, 도구 호출 방식, 에이전트 작업의 기본 골격을 제대로 심어 주지 못하면 이후 RL 단계가 불안정해지거나 비싼 비용을 치르게 된다. 따라서 SFT는 단순한 “워밍업”이 아니라, 전체 Cascade RL의 토대를 만드는 단계다.

### 3.1. Training Framework

#### 3.1.1. Overview

Nemotron-Cascade 2의 SFT 데이터는 수학, 코드 추론, 과학, 장문 문맥, 일반 채팅, 지시수행, 안전, 대화형 도구 사용, 소프트웨어 엔지니어링 에이전트, 터미널 에이전트까지 매우 넓은 영역을 포괄한다. 저자들은 모델이 실제로 마주칠 사용 환경이 이미 단일 도메인이 아니라고 보고, SFT 단계에서부터 다양한 형식의 문제를 한 모델 안에 동시에 주입한다.

학습은 한 번에 여러 단계로 나누지 않고 **단일 단계(single-stage)**로 수행된다. 모든 샘플을 최대 256K 토큰 길이의 packed sequence로 묶어 학습하며, 경험적으로 약 1.5 epoch 부근에서 SFT 성능이 가장 좋았다고 보고한다. 이는 데이터를 여러 차례 오래 보는 것보다, 넓은 분포를 한 번 잘 정리해서 보여 주는 편이 이후 RL과의 연결성 측면에서 유리하다는 판단으로 읽을 수 있다.

Appendix B의 Table 7이 이 단계의 구체적 하이퍼파라미터를 정리하지만, Section 3의 서술만 보더라도 저자들의 관점은 뚜렷하다. SFT는 “최종 응답 스타일을 예쁘게 만드는 미세조정”이 아니라, 앞으로의 RL이 어느 능력을 기반으로 출발할지를 결정하는 초기 정책 구축 단계다.

#### 3.1.2. Chat Template

<!-- Figure 1 -->

> [Figure 1 삽입 위치]

Figure 1은 Nemotron-Cascade 2의 채팅 템플릿을 설명한다. 이번 버전에서 가장 중요한 변경점은 **thinking mode와 non-thinking mode의 표시 방식 단순화**다. 이전 Nemotron-Cascade 계열에서 사용하던 `/think`, `/no_think` 태그를 제거하고, non-thinking 응답은 인접한 `<think></think>`로, thinking 응답은 단일 `<think>` 뒤에 실제 추론 내용을 두는 방식으로 정리했다. 저자들은 이 단순화가 추론 모드 제어를 더 일관되게 만들고, 학습 시퀀스 안에서 모드 전환을 더 명확히 표현하게 해 준다고 본다.

도구 사용에서도 같은 철학이 적용된다. 사용 가능한 도구 목록은 system prompt 안의 `<tools> ... </tools>` 블록에 넣고, 실제 함수 호출은 `<tool_call> ... </tool_call>` 태그로 감싼다. 이 구조는 “모델이 언제 자연어로 대답하고, 언제 구조화된 호출을 내놓는가”를 학습 단계에서 분리해서 보여 주는 장치다. 곧이어 나오는 conversational agent, code reasoning, terminal agent 데이터도 모두 이 템플릿을 토대로 생성되므로, Figure 1은 이후 여러 데이터 묶음이 공통적으로 기대하는 입출력 형식을 정의한다고 볼 수 있다.

### 3.2. SFT Data Curation

#### 3.2.1. Math

수학 데이터는 크게 **비증명형 수학**과 **자연어 증명형 수학**으로 나뉜다. 비증명형 프롬프트는 주로 Nemotron-Cascade와 Nemotron-Math-v2에서 가져온다. 저자들은 이 데이터 위에 reasoning trace를 더 정교하게 입히고, 필요할 때는 Python 도구를 사용하는 풀이 형식을 추가해, 계산형·추론형·도구보조형 수학을 함께 학습시킨다.

자연어 증명형 수학에서는 AOPS split의 Nemotron-Math-Proofs-v1로부터 98K개의 수학 증명 문제를 수집한다. 여기에 증명 생성 410K, 증명 검증 400K 규모의 데이터를 추가로 만들고, 교사 모델로는 DeepSeek-V3.2-Speciale를 사용한다. 최종적으로 이 묶음은 약 816K 규모가 된다. 중요한 점은 단순히 “정답 증명”만 보여 주는 것이 아니라, 생성–검증–재검토 흐름을 함께 노출해, 이후 IMO-ProofBench와 실제 IMO 풀이에서 쓰이는 generate-verify-refine 스타일의 기반을 SFT 단계부터 깔아 둔다는 것이다.

저자들이 이 절에서 노리는 것은 고난도 증명 그 자체보다도, 모델이 수학 문제를 볼 때 **정답형 응답, 도구를 곁들인 풀이, 장문의 정리된 증명**을 상황에 따라 구분해서 낼 수 있게 만드는 일이다. 이후 RL 단계가 정답률을 더 밀어 올린다고 해도, 그 출발점이 되는 서술 방식과 사고 틀은 이미 이 데이터에서 형성된다.

#### 3.2.2. Code Reasoning

코드 추론 데이터는 OpenCode-Stage2, OpenCodeReasoning, HardTests를 포함한 여러 공개 소스에서 약 165K개의 고유 프롬프트를 추린 뒤 정제해 만든다. 저자들은 이 영역에서 **중복 제거를 매우 공격적으로 수행**한다. 샘플 I/O fingerprinting과 n-gram 기반 텍스트 분석을 함께 써서, 겉보기에는 다른 문제처럼 보여도 본질적으로 같은 문제인 경우를 제거한다. 이 과정으로 약 24.2%가 걸러졌다고 보고한다.

교사 모델은 GPT-OSS-120B다. 검증 가능한 테스트 케이스가 있는 문제는 정답 프로그램만 채택하고, 테스트가 없거나 약한 문제는 더 긴 reasoning trace를 허용하는 식으로 데이터 성격을 나눈다. 최종 SFT 데이터에는 약 1.9M개의 Python trace, 1.0M개의 C++14 trace, 그리고 1.3M개의 Python tool-calling trace가 포함된다. 즉, Nemotron-Cascade 2는 단순한 “정답 코드 생성기”가 아니라, 문제 해석, 계획, 코드 작성, 필요 시 Python 실험까지 이어지는 연쇄적 행동 패턴을 SFT 단계에서부터 익힌다.

또한 과학 계산형 코딩 문제도 별도로 모은다. 생물, 재료, 물리, 화학, 수학에 걸친 scientific coding 프롬프트를 수집하고, 응답 역시 강한 교사 모델로부터 생성한다. 이는 이후 Code RL이 경쟁 프로그래밍 성능을 끌어올리더라도, 모델이 더 넓은 의미의 연구형 계산 문제를 다루는 능력을 잃지 않도록 하는 안전장치다.

#### 3.2.3. Science

과학 데이터는 물리, 화학, 생물 전반을 아우른다. Nemotron-Cascade로부터 1.4M, Nemotron-3-Nano로부터 1.3M의 science SFT 샘플을 가져오고, 응답 생성에는 GPT-OSS-120B를 활용한다. 이 구성은 과학 영역을 하나의 독립된 “지식 질의응답”으로 취급하지 않고, 길게 설명하고 이유를 붙이는 reasoning-heavy 도메인으로 본다는 점에서 특징적이다.

Section 4의 multi-domain RL에서 STEM MCQA가 별도 학습 영역으로 들어가는데, 이 science SFT는 그보다 앞서 **과학 문제를 읽고, 필요한 배경을 호출하고, 정답을 설명형으로 쓰는 습관**을 미리 형성해 준다. 즉, RL이 객관식 정답률을 다듬는다면, SFT는 그 정답으로 향하는 사고 구조와 서술 방식을 제공한다.

#### 3.2.4. Long Context

장문 문맥 데이터는 Nemotron-3-Nano에서 가져온 160K개 샘플과 추가로 수집한 74K개 샘플로 구성된다. 전자는 평균 길이가 128K 토큰에 달하고, 후자는 ChatQA-2 중심으로 평균 29K 토큰 수준이다. 저자들은 장문 능력을 단순히 “긴 입력을 읽을 수 있다”는 말로 이해하지 않는다. 긴 문서를 끝까지 유지한 채 문제를 푸는 것, 필요한 조각을 찾아 다시 조합하는 것, 긴 맥락에서도 형식 지시를 놓치지 않는 것이 모두 포함된다.

이 절은 나중의 long-context RL이 왜 별도 단계로 분리되는지의 배경도 제공한다. 장문 문제는 입력 길이 자체가 훈련 비용과 보상 지연을 키우기 때문에, SFT에서 기본 습관을 먼저 만들어 놓지 않으면 RL 단계가 지나치게 비싸지고 불안정해질 수 있다.

#### 3.2.5. General Chat

일반 채팅 데이터는 Nemotron-Cascade 1에서 가져온 프롬프트를 바탕으로, reasoning-on 4.9M과 reasoning-off 372K 샘플을 구성한다. reasoning-on 응답은 GPT-OSS-120B가 생성하고, reasoning-off에서는 데이터 안의 고품질 짧은 정답 300K를 그대로 사용하며, 추가로 DeepSeek-V3-0324가 생성한 330K를 넣어 응답 질을 높인다.

멀티턴 대화 능력 강화를 위해서는 약 700K개의 합성 대화 데이터를 만든다. 여기서는 두 개의 GPT-OSS-120B 인스턴스가 역할놀이 방식으로 서로 user와 assistant 역할을 나눠 대화를 진행한다. user 쪽 모델은 필요하면 대화를 종료할 수 있어, 무의미한 장기 반복을 줄인다. 이러한 self-play형 대화 데이터는 모델이 질문 응답뿐 아니라, **대화 흐름 유지, 화제 전환, 추가 질문 유도, 맥락 회수**를 익히게 한다.

또한 Nemotron-3-Nano에서 4.6M개의 reasoning-on 채팅 샘플을 더 가져오며, 프롬프트는 LMSYS와 WildChat에서 뽑는다. 응답 생성에는 GPT-OSS-120B, Qwen3-235B-A22B-Thinking-2507, Qwen3-235B-A22B-Instruct-2507을 혼합 사용한다. 일반 채팅은 겉으로는 가장 평범한 데이터처럼 보이지만, 사실 RLHF와 ArenaHard 같은 인간 선호 중심 평가의 기반 분포를 형성한다는 점에서 뒤 단계와 강하게 연결된다.

#### 3.2.6. Instruction Following

지시수행 데이터는 Nemotron-Cascade 1 기반 프롬프트에서 reasoning-on 230K, reasoning-off 64K를 만들고, 여기에 Nemotron-3-Nano에서 가져온 497K 샘플을 더해 총량을 키운다. Nano 쪽 샘플은 457K reasoning-on, 40K reasoning-off로 나뉜다. 응답 생성에는 GPT-OSS-120B, Qwen3-235B-A22B-Thinking-2507, Qwen3-235B-A22B-Instruct-2507이 사용된다.

이 데이터의 목적은 모델이 “정답을 아는가”보다 “주어진 형식과 제약을 정확히 지키는가”에 있다. 길이 제한, 키워드 포함, 금지 표현 회피, 특정 포맷 준수 같은 과제는 뒤의 IF-RL에서 검증 가능한 보상으로 강화되지만, 그 전에 SFT가 포맷 준수 습관을 충분히 심어 두어야 RL이 빠르게 수렴한다.

#### 3.2.7. Safety

안전 SFT 샘플은 4K 규모로 비교적 작지만, 역할은 분명하다. Nemotron Content Safety v2, Gretel Safety Alignment v1, Harmful Tasks, Red-Team-2K 등의 소스에서 가져온 프롬프트를 통해, 모델이 위험한 입력에 대해 적절히 거절하고, 가능한 경우에는 안전한 대안으로 유도하는 반응 양식을 익힌다.

여기서 중요한 점은 안전이 독립된 “거절 데이터”로만 존재하는 것이 아니라는 것이다. 이 작은 안전 묶음은 일반 채팅, instruction following, conversational agent 데이터와 함께 들어가며, 그래서 모델은 위험 여부 판단과 일반 유용성 사이를 하나의 정책 안에서 조정하게 된다. 이후 RLHF에서 synthetic safety blend가 추가되는 것도 같은 이유다.

#### 3.2.8. Conversational Agent

대화형 에이전트 데이터는 Python 도구 사용 데이터를 넘어, 여러 도구가 동시에 제공되는 멀티턴 대화 환경을 다룬다. 모델은 어떤 도구를 언제 호출해야 하는지, 도구 결과를 어떻게 읽고 다음 행동으로 연결할지를 학습한다. 이 묶음은 Nemotron-3-Nano에서 가져온 822K개의 conversational tool-use 샘플로 구성되며, 응답 생성에는 Qwen3-235B-A22B-Thinking-2507, Qwen3-32B, Qwen3-235B-A22B-Instruct-2507, GPT-OSS-120B가 사용된다.

이 데이터의 본질은 “도구 호출 문법”이 아니라 **도구 선택과 상태 유지**다. 여러 턴에 걸쳐 이전 결과를 참고하고, 필요하면 새로운 도구를 부르며, 자연어 응답과 구조화된 호출을 오가는 패턴이 학습된다. 이는 BFCL, τ2-Bench, terminal agent 데이터와 직접 연결된다.

#### 3.2.9. Software Engineering Agent

SWE 에이전트 데이터는 OpenHands, SWE-Agent, Mini-SWE-Agent, 그리고 agentless scaffold를 모두 포괄한다. 먼저 Nemotron-3-Nano와 Super의 데이터를 활용해, Qwen3-Coder-480B-A35B-Instruct가 생성한 SWE agentic trajectory를 수집한다. 문제 원천은 SWE-Gym, SWE-rebench, R2E-Subset이다. 여기에 Nemotron-Cascade 1에서 온 agentless 데이터, 즉 buggy code localization, code repair, test case generation을 더한다. code repair 데이터는 DeepSeek-V3.2를 사용해 पुन구성한다.

저자들은 이 절에서 중요한 관찰 하나를 제시한다. **agentless 데이터가 agentic 성능도 높여 준다**는 것이다. 오직 agentic 데이터만으로 미세조정한 경우보다, agentic과 agentless를 함께 섞었을 때 SWE-bench Verified에서 더 좋은 Pass@1, Pass@4가 나왔다고 보고한다. 이 관찰은 Section 4.8의 agentless RL과 Table 4로 이어진다.

최종적으로는 125K agentic 샘플과 389K agentless 샘플을 함께 사용한다. 또한 저자들은 agentic 데이터에는 non-thinking 모드, agentless 데이터에는 thinking 모드를 주로 붙이는 식으로, 작업 형태에 따라 추론 표현 방식을 나눈다. 이는 “도구와 환경 자체가 이미 외부 추론 구조를 제공하는가” 여부에 따라 내부 reasoning trace의 필요성이 달라질 수 있음을 시사한다.

#### 3.2.10. Terminal Agent

터미널 에이전트 데이터는 Terminal-Task-Gen 방법론을 이용해 구성된다. 전체 규모는 490K이며, 이 안에는 수학 162K, 코드 32K, SWE 32K를 기존 데이터에서 터미널 작업으로 변환한 부분과, seed-based task 120K, skill-based task 140K가 포함된다. 실제 trajectory는 DeepSeek-V3.2를 이용해 격리된 Docker 환경 안에서 생성하고, Terminus 2 scaffold를 사용해 명령 실행, 파일 조작, 상태 추적을 기록한다.

이 데이터의 목적은 모델이 단순히 “터미널 명령을 아는 것”이 아니라, 긴 명령 시퀀스의 중간 상태를 기억하고, 실패를 보고 수정하고, 다음 명령을 선택하는 **장기적 행동 정책**을 배우게 하는 데 있다. 이후 τ2-Bench와 Terminal Bench 2.0 평가에서 필요한 능력이 바로 이 데이터에서 형성된다.

## 4. Cascade RL and Multi-Domain On-Policy Distillation

Section 4는 Nemotron-Cascade 2의 핵심 기술 섹션이다. SFT가 초기 정책을 만든 뒤, 저자들은 이를 곧바로 단일 RLHF 단계로 마감하지 않는다. 대신 서로 다른 능력 영역을 순차적으로 혹은 묶어서 최적화하는 **Cascade RL**을 적용하고, 그 중간에서 생기는 성능 회귀를 회복하기 위해 **MOPD**를 삽입한다. 이 절의 중요한 메시지는, RL 단계의 “존재 여부”보다 “배치 순서와 상호작용”이 더 중요하다는 것이다.

### 4.1. Training Framework

<!-- Figure 2 -->

> [Figure 2 삽입 위치]

Figure 2는 전체 post-training 파이프라인을 그림으로 요약한다. base model에서 SFT를 거친 뒤, 먼저 IF-RL을 수행해 지시 준수 능력을 끌어올리고, 다음으로 multi-domain RL을 통해 STEM MCQA·에이전트 도구 사용·복합 지시수행 능력을 공동 강화한다. 그 다음에 MOPD로 이전 단계들에서 잃은 능력을 교사 정책으로부터 다시 끌어오고, 이후 RLHF, long-context RL, code RL, SWE RL을 क्रम차적으로 쌓아 최종 Nemotron-Cascade 2에 이른다.

저자들은 이 순서를 임의로 고른 것이 아니라고 강조한다. 모델이 아직 형식을 잘 지키지 못하는 상태에서 인간 선호만 먼저 맞추면, 창의적이지만 지시를 놓치는 응답이 강화될 수 있다. 반대로 지시수행을 먼저 강하게 올리면, 그 정책이 이후 단계들에서 일종의 “기초 체력” 역할을 하게 된다. Figure 2는 따라서 학습 파이프라인의 구성도이면서, 논문 전체의 인과관계를 압축한 그림이기도 하다.

#### 4.1.1. What determines the ordering of Cascade RL

이 절에서 저자들은 Cascade RL의 순서를 정하는 기준을 세 가지로 설명한다.

첫째는 **inter-domain interference 완화**다. 어떤 RL 단계는 다른 능력을 심하게 손상시킬 수 있다. 예를 들어 IF-RL은 지시 준수에는 매우 좋지만, 지나치게 길이를 통제하거나 포맷 중심으로 학습하면 ArenaHard 같은 인간 선호 정렬 점수가 떨어질 수 있다. RLHF는 그 반대 방향의 보정을 제공한다. 따라서 중요한 것은 “각 단계가 무슨 점수를 얼마나 올리느냐”뿐 아니라, “무엇을 대가로 무엇을 올리느냐”다.

둘째는 **multi-domain stage의 가능성**이다. 서로 큰 충돌을 일으키지 않거나, 오히려 함께 묶었을 때 효율이 좋은 영역은 하나의 RL 단계로 묶을 수 있다. Nemotron-Cascade 2에서는 STEM MCQA, agentic tool calling, 장문의 복합 지시수행이 이런 경우에 해당한다고 본다. 이들을 한꺼번에 올리면 별도 단계로 쪼갤 때보다 학습 비용을 줄이면서도 폭넓은 성능 프로파일을 유지할 수 있다.

셋째는 **on-policy distillation을 통한 안정화**다. 저자들은 아무리 순서를 잘 짜도, 특정 영역을 강하게 올린 뒤 다른 영역 일부가 내려가는 현상을 완전히 제거할 수 없다고 인정한다. 그래서 MOPD를 중간 안정화 단계로 넣어, 이미 특정 영역에서 좋았던 정책의 토큰 분포를 다시 학생 정책 안으로 회수한다. 이 아이디어 덕분에 후반 단계는 단순히 “앞선 능력을 희생하면서 새 능력을 얻는 과정”이 아니라, 이전 이득을 최대한 보존하면서 새 영역을 확장하는 과정으로 바뀐다.

#### 4.1.2. RL Training Configuration

Cascade RL 전반에는 GRPO가 사용된다. 저자들은 strict on-policy 설정을 채택해, 데이터를 수집한 정책과 업데이트되는 정책이 항상 같도록 만든다. 이렇게 하면 importance sampling ratio가 1이 되어 학습이 단순해지고, entropy collapse를 줄이며, KL 항 없이도 안정적인 업데이트가 가능하다고 설명한다. 사용 프레임워크는 Nemo-RL이다.

논문은 이때의 GRPO 목적을 다음과 같이 적는다.

\[
\mathcal{J}_{\mathrm{GRPO}}(\theta)
=
\mathbb{E}_{(q,a)\sim\mathcal{D},\ \{o_i\}_{i=1}^{G}\sim\pi_\theta(\cdot\mid q)}
\left[
\frac{1}{\sum_{i=1}^{G}|o_i|}
\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}\hat{A}_{i,t}
\right],
\qquad
\hat{A}_{i,t}
=
\frac{r_i-\mathrm{mean}(\{r_i\}_{i=1}^{G})}
{\mathrm{std}(\{r_i\}_{i=1}^{G})}.
\tag{1}
\]

여기서 \(r_i\)는 한 질문에 대해 샘플된 각 응답의 보상이고, \(\hat{A}_{i,t}\)는 그 보상을 그룹 내부에서 정규화한 토큰 수준 advantage다. 핵심은 “한 응답 전체에 하나의 보상만 주는” 단순 시퀀스 보상보다, 그룹 안에서 상대적으로 얼마나 좋은 응답인지 표준화해 보는 것이다. RLVR에서는 이 보상이 정답 검증으로부터 오고, RLHF에서는 GenRM의 선호 점수로부터 온다.

저자들이 이 설정을 선호하는 이유는 명확하다. on-policy 구성은 안정적이고, KL 항 제거는 구현을 단순하게 하며, 그룹 정규화는 서로 다른 난도의 문제를 함께 학습할 때 보상 스케일 차이를 줄인다. 이후 각 하위 절에서는 같은 기본 뼈대 위에 도메인별 보상과 데이터가 어떻게 달라지는지를 설명한다.

### 4.2. Instruction-Following Reinforcement Learning (IF-RL)

IF-RL은 전체 Cascade RL의 첫 단계다. 저자들은 이 선택을 매우 의식적으로 설명한다. 모델이 복잡한 reasoning이나 human preference를 배우기 전에, **주어진 지시를 정확히 따른다**는 능력을 먼저 극대화해야 한다는 것이다. 논문은 이 단계만으로 IFBench에서 83.13%라는 state-of-the-art 수준 정확도를 얻었다고 보고한다.

#### 4.2.1. Dataset

데이터는 NVIDIA Nano-v3 post-training에 사용된 instruction-following RL 데이터와 동일 계열을 쓴다. 이 데이터의 특징은 정답 검증이 가능한 지시들로 이루어져 있다는 점이다. 예를 들어 “200단어 이하로 대답하라”처럼 명시적으로 채점 가능한 규칙을 담고 있다. 따라서 인간 선호처럼 애매한 보상 대신, 비교적 분명한 RLVR 신호를 줄 수 있다.

이 데이터는 지시수행을 “좋은 대답처럼 보이는가”가 아니라 “정확히 조건을 만족하는가”로 정의한다. 그래서 후반 RLHF가 유창성과 인간 선호를 다듬는다면, IF-RL은 먼저 정책의 규칙 준수 능력을 단단히 만든다.

#### 4.2.2. Training recipe

IF-RL에서는 dynamic filtering을 적용한다. 모든 rollout이 전부 정답이거나 전부 오답인 샘플은 gradient를 거의 주지 못하므로 제거한다. 이렇게 하면 각 batch가 실제로 학습 신호를 주는 프롬프트로 채워져 학습이 더 안정된다. 또한 너무 긴 응답이 자주 나와 token usage가 폭증하는 문제를 막기 위해 overlong penalty를 적용한다. 최대 길이 안에서 답을 마치지 못하면 보상을 0으로 주는 방식이다.

저자들은 IF-RL을 맨 앞에 놓은 이유를 두 가지로 정리한다. 첫째, IF-RL은 ArenaHard 같은 인간 선호 지표를 약간 손상시킬 수 있지만, 뒤의 RLHF가 그 회귀를 상당 부분 되돌릴 수 있다. 둘째, instruction-following이 강한 모델은 이후 multi-domain stage와 MOPD에서 더 좋은 교사 역할을 한다. 즉, IF-RL은 그 자체가 목표인 동시에 다음 단계들을 위한 기초 정책 개선이기도 하다.

훈련은 batch size 128, prompt당 16 response, temperature 1.0, top-p 1.0, AdamW 학습률 3e-6, entropy 계수 0, KL 계수 0의 설정으로 약 180 step 수행된다. Thinking mode만 사용한다는 점도 중요한데, 저자들은 non-thinking 응답을 섞을 때 instruction-following 점수가 오히려 떨어진다고 보고한다.

### 4.3. Multi-domain RL

IF-RL 뒤에는 multi-domain RL 단계가 이어진다. 여기서는 세 가지 능력을 함께 묶는다. 첫째는 STEM 영역의 multi-choice question answering, 둘째는 채팅 도메인의 agentic tool calling, 셋째는 길고 복잡한 instruction-following이다. 저자들은 이 세 영역이 서로 큰 충돌 없이 함께 올릴 수 있는 축이라고 판단한다.

이 단계를 별도로 둔 이유는, 모든 능력을 미세하게 나눠 여러 RL 단계로 처리하면 지나치게 비용이 커지고, 각 단계마다 다른 능력들이 흔들릴 수 있기 때문이다. 반대로 서로 잘 맞는 영역은 묶어서 올리면, 넓은 분포를 다루면서도 효율적인 학습이 가능하다. 저자들은 실제로 이 구성이 MMLU-Pro, τ2-Bench, IFBench에서 동시에 개선을 가져왔다고 설명한다.

학습은 batch 128, prompt당 16 sample, temperature 1.0, top-p 1.0, Adam 최적화, 학습률 3e-6으로 약 70 step 진행된다. entropy와 KL은 0으로 둔다. 여기서 중요한 것은 성능의 절대 상승값보다도, **한 단계 안에서 여러 능력을 동시에 올릴 수 있었다**는 설계적 결과다. 이것이 이후 MOPD와 결합되면서, Cascade RL이 단순 순차학습보다 훨씬 유연한 형태로 작동하게 된다.

### 4.4. Multi-domain On-Policy Distillation (MOPD)

MOPD는 이 논문의 가장 핵심적인 신규 요소다. 저자들은 잘 설계된 Cascade RL이 catastrophic forgetting을 줄여 주긴 하지만, 특정 도메인을 강하게 올린 뒤 다른 벤치마크가 내려가는 현상을 완전히 없애지는 못한다고 본다. 따라서 RL 단계 사이에서 **가장 좋았던 중간 정책을 각 도메인의 교사로 재활용**해, 학생 정책이 그 토큰 분포를 다시 따라가도록 만든다.

MOPD의 출발점은, 각 도메인마다 “가장 강했던 순간”이 서로 다를 수 있다는 관찰이다. 어떤 정책은 수학에서 최고였고, 다른 정책은 tool use에서 최고였을 수 있다. 최종 정책 하나만을 교사로 쓰면 이런 국소 최적점을 잃어버린다. 그래서 MOPD는 훈련 샘플의 도메인에 따라 서로 다른 teacher를 선택한다. 학생 정책은 on-policy로 응답을 생성하지만, 그 응답 토큰 하나하나에 대해 해당 도메인의 teacher가 더 선호하는 방향으로 probability를 보정받는다.

논문은 MOPD의 토큰 수준 distillation advantage를 다음과 같이 정의한다.

\[
a_t^{\mathrm{MOPD}}
=
\log \pi_{\mathrm{domain}_i}(y_t\mid s_t)
-
\log \pi_{\mathrm{train}}(y_t\mid s_t).
\tag{2}
\]

이 항은 domain teacher가 학생보다 샘플된 토큰에 더 높은 확률을 줄수록 양수가 된다. 즉, teacher가 “그 토큰을 더 그럴듯하게 본다”면 학생은 그 토큰을 더 밀어 주고, 반대면 덜 밀게 된다. 중요한 점은 전체 vocabulary 분포를 전부 맞추는 것이 아니라, 실제 학생이 샘플한 토큰에 대해 teacher와 학생의 log-prob 차이만 본다는 것이다. 덕분에 계산량이 줄고, RL 중간에 삽입하기 쉬운 distillation이 된다.

하지만 응답은 \(\pi_{\mathrm{inf}}\)로 샘플되고, 실제 최적화 대상은 \(\pi_{\mathrm{train}}\)이다. 이 train–inference mismatch를 다루기 위해 논문은 truncated importance weighting을 사용한다.

\[
r_t
=
\frac{\pi_{\mathrm{train}}(y_t\mid s_t)}
{\pi_{\mathrm{inf}}(y_t\mid s_t)},
\qquad
w_t
=
\mathrm{sg}[r_t]\mathbf{1}\!\left[\epsilon_{\mathrm{low}}\le r_t\le \epsilon_{\mathrm{high}}\right].
\tag{3}
\]

여기서 \(\mathrm{sg}[\cdot]\)는 stop-gradient다. 너무 큰 비율이나 너무 작은 비율은 잘라 내어 불안정한 업데이트를 막는다. 결국 surrogate objective는 다음과 같이 적힌다.

\[
\mathcal{L}_{\mathrm{MOPD}}
=
-
\mathbb{E}_{x\sim\mathcal{D},\ y\sim\pi_{\mathrm{inf}}(\cdot\mid x)}
\left[
\frac{1}{|\mathcal{V}(y)|}
\sum_{t\in\mathcal{V}(y)}
w_t\,\mathrm{sg}[a_t^{\mathrm{MOPD}}]\log\pi_{\mathrm{train}}(y_t\mid s_t)
\right].
\tag{4}
\]

\(\mathcal{V}(y)\)는 token mask를 통과한 유효 응답 토큰 집합이다. 즉, MOPD는 teacher가 잘하는 토큰 분포를 학생의 실제 샘플 위에 얹어 주는 방식으로 작동하며, RL이 희생한 영역을 다시 되살리는 복구 단계로 해석할 수 있다.

<!-- Figure 3 -->

> [Figure 3 삽입 위치]

Figure 3은 이 아이디어가 실제로 어떻게 작동하는지를 보여 준다. reverse KL, grad norm, AIME25 결과를 함께 그려, MOPD가 GRPO보다 훨씬 빠르게 교사 수준 성능을 회복하거나 넘어서며, 학습 동역학도 더 부드럽게 유지된다는 점을 시각화한다. 저자들의 해석은 단순하다. sequence-level sparse reward만 주는 RL보다, teacher가 어느 토큰이 좋은지 직접 알려 주는 dense token-level advantage가 훨씬 빠르고 안정적인 학습 신호를 제공한다는 것이다.

<!-- Table 3 -->

> [Table 3 삽입 위치]

Table 3은 ArenaHard V2.0에서 RLHF와 MOPD를 같은 평가 지점 기준으로 비교한다. Initial 모델은 Hard Prompt 71.5, Creative Writing 40.6에서 시작한다. RLHF는 100 step에 Hard Prompt 81.7, Creative Writing 68.6까지 올리고, 160 step에는 80.7 / 71.2를 기록한다. 반면 MOPD는 52 step만에 85.5 / 71.0에 도달한다. 즉, creative writing은 RLHF와 비슷한 수준까지 빠르게 올라가고, hard prompt 쪽은 더 높은 수치에 도달한다.

여기서 저자들이 강조하는 바는, MOPD가 RLHF를 완전히 대체한다는 뜻이 아니다. MOPD는 이미 존재하는 좋은 intermediate teacher를 다시 활용해 회귀를 회복하는 데 특히 강하다. 반면 RLHF는 인간 선호에 맞는 표현, 창의적 글쓰기, 비검증형 추론처럼 teacher checkpoint만으로는 충분히 규정되지 않는 영역을 정교하게 다듬는 역할을 맡는다. 따라서 Nemotron-Cascade 2에서는 두 방법이 경쟁 관계가 아니라, 서로 다른 시점과 목적을 가진 상보적 단계로 배치된다.

하이퍼파라미터 측면에서 MOPD는 rollout size 4, prompt 128, 효과적 batch 512 response 구성으로 진행되고, Table 8에 요약된 설정을 따른다. max response length를 98K까지 늘린 것도 눈에 띄는데, 이는 회귀가 자주 일어나는 긴 reasoning trace를 충분히 받아들이기 위한 선택으로 볼 수 있다.

### 4.5. Reinforcement Learning from Human Feedback (RLHF)

MOPD 뒤의 RLHF는 다시 인간 선호를 정면으로 다루는 단계다. 저자들은 이 시점을 매우 신중하게 잡는다. IF-RL과 multi-domain RL, MOPD를 거친 뒤에야 모델이 충분히 지시를 따르고 도구를 쓰며 여러 능력을 일정 수준 유지하게 되었고, 이제 그 위에서 인간이 선호하는 응답 스타일과 비검증형 창의성을 다듬는 것이 효과적이라는 것이다.

#### 4.5.1. Dataset

RLHF 데이터는 NVIDIA Nano-v3 계열 데이터 블렌드를 사용한다. HelpSteer3, arena-human-preference-140k의 상업 친화적 부분집합, synthetic safety blend가 포함된다. 생성 보상 모델(GenRM)로는 HelpSteer3 프레임워크로 학습된 **Qwen3-235B-A22B-Thinking-2507**을 사용한다. 이 GenRM은 대화 기록, 사용자 요청, 두 개의 candidate response를 받아 각각의 장단점을 비교하고, 최종 선호 순위를 산출한다.

#### 4.5.2. Training recipe

훈련은 각 prompt에 대해 모든 rollout 쌍을 pairwise 비교하는 방식으로 진행된다. 보상 집계는 Nano-v3 RLHF와 같은 규칙을 따르며, 여기에 **length-normalized reward adjustment**와 **quality-gated conciseness bonus**를 그대로 사용한다. 이 두 장치는 짧은 응답을 무조건 장려하는 것이 아니라, 품질을 유지하는 범위 안에서 토큰 사용량의 폭주를 억제하는 역할을 한다.

저자들은 특히 RLHF를 **thinking mode 전용**으로 수행한다. thinking과 non-thinking을 함께 넣으면 일부 평가 점수는 약간 좋아질 수 있지만, instruction-following이 크게 무너지고 그 손실을 나중 단계에서 완전히 회복하기 어렵다고 말한다. 즉, RLHF는 인간 선호를 다듬되, 앞서 세워 놓은 지시 준수와 사고 구조를 망가뜨리지 않는 방향으로 매우 제한적으로 적용된다.

#### 4.5.3. Hyper-parameters

RLHF의 하이퍼파라미터는 batch size 128, prompt당 16 rollout, temperature 1.0, top-p 1.0, max response length 16K, AdamW 학습률 3e-6이다. entropy 계수는 0, KL 계수는 0.03으로 둔다. overlong filtering은 적용하지 않는다. 학습은 약 25–30 step 정도 진행된다. Table 9는 이 설정을 long-context RL, code RL과 나란히 보여 주며, RLHF가 상대적으로 짧은 max response length를 쓰는 반면 KL을 유일하게 남겨 두는 단계라는 점을 드러낸다.

### 4.6. Long-context RL

RLHF 뒤에는 long-context RL이 온다. 저자들은 long-context 능력을 강화하고 싶더라도 모든 도메인을 함께 섞어 RL을 돌리는 것은 오히려 해가 된다고 보고한다. 실험상 long-context RL 단계에 다른 도메인을 섞으면 관련 없는 벤치마크 점수가 떨어졌기 때문에, 이 단계는 long-context 데이터셋 전용으로 제한된다.

보상 환경에는 Nemo-Gym이 쓰이고, 롤아웃 평가는 Qwen3-235B-A22B-Instruct-2507을 judge로 사용해 question answering 형태로 수행된다. 학습 시 입력은 32K로 제한하되 최대 시퀀스 길이는 49K까지 허용하며, overlength filtering은 두지 않는다. batch 128, rollout 16, temperature 1.0, top-p 1.0, AdamW 3e-6, entropy 0, KL 0으로 약 30 step 훈련한다.

이 단계의 핵심은 장문 추론을 “한 번에 다 읽는 능력”이 아니라, 긴 문맥에서도 핵심 정보를 유지하고 필요한 부분을 다시 호출하는 정책으로 보는 데 있다. 토큰 길이가 빠르게 늘어나기 때문에 학습을 오래 끌면 비용이 급증하므로, 저자들은 성능 상승이 확인되는 짧은 구간에서 훈련을 멈춘다.

### 4.7. Code RL

#### 4.7.1. Data Curation

Code RL 데이터는 Nemotron-Cascade 코딩 코퍼스에서 출발하며, AtCoder, Codeforces, AIZU처럼 강한 테스트케이스를 가진 경쟁 프로그래밍 문제 위주로 구성된다. 저자들은 이 단계의 효율을 높이기 위해 GPT-OSS-120B가 8회 중 8회 모두 정답을 낸 쉬운 문제를 과감히 제거한다. 이렇게 남은 최종 데이터는 3.5K 정도로 매우 작아지지만, 저자들은 오히려 이 작은 고난도 세트가 모델의 심층 추론력을 더 잘 밀어 올린다고 본다.

즉, Code RL은 데이터 수를 늘리는 방향이 아니라, **정말 어렵고 보상 신호가 날카로운 문제만 남기는 방향**으로 설계된다. 강한 테스트케이스와 높은 난도가 결합되어야 RL이 reward hacking 없이 진짜 추론 능력을 끌어올릴 수 있다는 것이 이 절의 메시지다.

#### 4.7.2. Training Details

Code RL은 batch 128, AdamW 3e-6으로 수행된다. Nemotron-Cascade 1보다 max response length를 118K까지 늘리고, prompt당 rollout도 16개로 늘린다. 이는 경쟁 프로그래밍에서 긴 사고 과정을 거쳐야만 희소한 정답 보상에 도달할 수 있기 때문이다. 보상 함수는 strict binary reward를 채택해, 부분적 패턴 맞추기나 보상 해킹을 줄이고 완전히 on-policy 상태를 유지한다.

학습 인프라도 논문이 강조하는 부분이다. 한 RL step마다 128×16=2,048개의 코드 실행이 필요하므로, 비동기 reward verification 서버를 두고 384 CPU core에서 한 batch를 약 427.2초에 처리한다. 이 숫자는 Code RL이 단순한 텍스트 RL이 아니라, 실제 실행 기반 검증을 대규모로 돌리는 인프라 문제이기도 하다는 점을 보여 준다.

### 4.8. Software Engineering Reinforcement Learning (SWE RL)

#### 4.8.1. Agentless RL

Agentless RL은 코드 수리 능력 자체를 강화하는 단계다. 데이터는 Wang et al. 계열의 agentless code repair RL 소스를 사용한다. 다만 많은 인스턴스가 실행 가능한 Docker 환경을 제공하지 않기 때문에, 저자들은 GPT-OSS-120B를 reward model로 써서 생성된 패치의 품질을 평가한다. 각 인스턴스에 대해서는 golden localization과 top-5 retrieved localization을 함께 사용해 프롬프트를 만들고, 너무 쉬운 샘플은 걸러 낸다.

학습은 batch 128×16=2,048, max sequence length 98,304, AdamW 3e-6, temperature 1.0, top-p 1.0으로 진행된다. 어느 rollout도 reward 0.5를 넘지 못한 prompt는 loss를 masking해, 지나치게 어려운 사례가 학습을 망치지 않게 한다. 학습은 대개 40–50 step 내에 수렴한다.

<!-- Table 4 -->

> [Table 4 삽입 위치]

Table 4는 agentless RL의 효과를 SWE-bench Verified 위에서 보여 준다. Agentless Mini와 OpenHands 두 scaffold 모두에서 avg@4, pass@4가 소폭이지만 일관되게 상승한다. 이 결과는 중요한 함의를 갖는다. 에이전트형 SWE는 도구 호출, 파일 탐색, 상태 관리 같은 scaffold 전체의 능력도 필요하지만, **기저 모델의 코드 수리 능력 자체를 강화하는 것만으로도 agentic 환경 성능이 좋아질 수 있다**는 것이다. 저자들은 이를 “code repair capability의 scaffold 간 일반화”라고 해석한다.

#### 4.8.2. Execution-based RL for Agentic SWE Scaffold

마지막 SWE RL 단계는 실행 기반 agentic RL이다. 여기서는 OpenHands류의 scaffold 안에서 실제로 파일을 열고, 검색하고, 코드를 수정하고, 테스트를 돌리는 전체 trajectory를 강화학습 대상으로 삼는다. 각 episode는 SWE-bench 계열 issue 하나를 해결하는 과정에 해당하며, 컴파일 결과와 unit test 결과가 deterministic reward를 제공한다.

훈련은 16 prompt × 64 rollout = batch 1024 규모로 진행되며, max context length는 256K, agent는 최대 200 turn까지 상호작용할 수 있다. 데이터는 SWE-Gym과 R2E-Subset에서 가져오고, intermediate model로 16 rollout을 생성해 검증한다. 여기서 100% pass인 너무 쉬운 인스턴스는 제거하고, 0% pass인 너무 어려운 인스턴스는 90%를 무작위로 버린다. 즉, 학습 분포를 “어느 정도 노력하면 신호가 나오는 문제”에 맞춰 재조정한다.

이 단계는 소프트웨어 엔지니어링을 단순 코드 생성이 아니라, **환경과 상호작용하며 문제를 진단하고 패치를 검증하는 정책**으로 본다. 그래서 Nemotron-Cascade 2의 SWE 능력은 하나의 정답 스니펫을 생성하는 능력보다, 장기 궤적을 운영하는 능력에 가깝다.

## 5. International Mathematical Olympiad (IMO)

### 5.1. IMO 2025

IMO 2025 평가는 self-improving test-time scaling 프레임워크로 수행된다. 모델은 풀이를 생성하고, 스스로 혹은 judge를 통해 검증하며, 그 결과를 바탕으로 다시 정제한다. 이 generate-verify-refine 흐름은 Section 3의 proof SFT와 Section 4 이후의 reasoning 강화가 inference-time에서 결합된 사례다.

저자들이 가장 강조하는 사실은, Nemotron-Cascade 2가 첫 다섯 문제를 풀어 총 35/42점에 도달했다는 점이다. 이는 30B-A3B 규모를 고려하면 매우 인상적이다. 하지만 저자들은 여기서 만족하지 않고, Appendix E에 전체 모델 풀이와 인간 전문가 코멘트를 함께 넣는다. 즉, 단순 점수 보고가 아니라 실제 풀이의 질과 형태를 검토 대상으로 올린다.

Section 5.1의 해석은 균형 잡혀 있다. 전문가 리뷰에 따르면 일부 증명은 필요 이상으로 길고, 쓸모없는 중간 정의나 단계를 포함하며, 내부 사고 흔적이 드러나거나 사소한 오타가 섞여 있다. Problem 2의 경우 모델은 보다 기하적인 해법보다 해석기하식 접근을 택한다. 저자들은 이를 약점으로 숨기지 않고, compact model에서도 강한 inference-time scaling이 olympiad 수준 추론을 낳을 수 있음을 보여 주는 증거로 제시한다.

### 5.2. IMO-ProofBench

<!-- Table 5 -->

> [Table 5 삽입 위치]

Table 5는 IMO-ProofBench 결과를 Basic 30문제, Advanced 30문제, Overall 60문제로 나눠 보여 준다. Nemotron-Cascade 2는 generate-verify-refine을 적용했을 때 Basic 92.5, Advanced 53.4, Overall 72.9를 기록한다. 저자들은 DeepSeek-Math-V2-671B-A37B와 비교하면 활성 파라미터 수는 10배 적지만 점수 격차는 8점 이내라고 강조한다. 이 표에서 읽을 수 있는 중요한 사실은, 모델이 쉬운 증명 문제에서만 강한 것이 아니라 Advanced split에서도 상당한 점수를 낸다는 점이다.

또한 이 표는 모델이 단순히 “한 번 잘 뽑은 proof”에 의존하지 않는다는 점을 보여 준다. generate-verify-refine을 여러 라운드 돌린 결과가 표에 반영되어 있고, 이는 증명 생성 능력 자체와 더불어 **자기 검토와 수정 능력**이 함께 모델 성능을 이끈다는 뜻이다.

<!-- Figure 4 -->

> [Figure 4 삽입 위치]

Figure 4는 Advanced split 성능이 generate-verify-refine 라운드 수에 따라 어떻게 상승하는지를 그린다. Nemotron-Cascade 2는 1라운드 40.7에서 시작해 5라운드 53.4까지 올라가고, 저자들이 재현한 DeepSeek-Math-V2도 함께 그려 비교한다. 이 그림의 메시지는, 증명 능력이 단발 generation만으로 결정되지 않고 **자기 검증과 정제 루프에서 크게 올라간다**는 점이다. 이 때문에 저자들은 proof SFT와 proof evaluation을 inference-time scaling 관점에서 다시 해석한다.

## 6. Competitive Coding

### 6.1. IOI 2025 and ICPC World Finals 2025

IOI 2025에서는 Nemotron-Cascade 계열의 IOI test-time scaling 파이프라인을 계승해 사용한다. 핵심 구조는 multi-round generate-select-submit이다. 각 subtask마다 최대 50라운드를 수행할 수 있고, 한 라운드 안에서는 최대 40개의 candidate solution을 생성해 이전 제출 기록과 다른 subtask에서 얻은 통찰까지 함께 반영한다. 이 파이프라인 덕분에 Problem 3과 4는 만점을 얻고, 전체적으로 439.28/600으로 골드 수준 점수를 얻는다.

저자들은 여기서 잠재 성능도 함께 언급한다. generation 수를 충분히 늘리면 507.66/600까지 도달 가능하다고 추정하며, 특히 Problem 2에서 휴리스틱 전략이 적은 generation budget 안에서도 86점 이상을 얻는다고 설명한다. 이는 코딩 능력이 단순 1-pass 정답률만이 아니라, **후보 생성과 선택을 반복하는 test-time computation budget**의 함수이기도 하다는 점을 보여 준다.

ICPC World Finals 2025에서는 문제당 최대 1000개의 솔루션을 생성하고, 초기 필터링 뒤 공식 제출 평가에 넣는다. 그 결과 12문제 중 10문제를 풀어 #4 Gold 수준에 해당하는 성과를 냈고, Problems A와 I를 제외한 8문제는 100 submissions 이내에 해결했다. IOI가 세밀한 부분점수 구조를 가진다면, ICPC는 제한된 시도 속에서 문제 집합 전체를 관리해야 한다는 점에서 성격이 다르다. Nemotron-Cascade 2가 दोनों를 모두 잘했다는 점은 code RL과 test-time scaling의 결합 효과를 보여 준다.

<!-- Table 6 -->

> [Table 6 삽입 위치]

### 6.2. Competitive Coding Benchmark Results

Table 6은 LiveCodeBench v6, LiveCodeBench Pro 25Q1/25Q2, Codeforces ELO와 percentile을 통합해 보여 준다. 이 표를 보면 Nemotron-Cascade 2는 frontier open model들과 비교해도 매우 높은 pass@1과 ELO를 기록한다. 특히 TIR 설정에서는 stateful Python executor를 최대 100회 호출할 수 있어, difficult reasoning을 외부 계산과 결합하는 능력을 평가한다.

저자들이 강조하는 대목은 두 가지다. 첫째, 모델이 LiveCodeBench Pro hard split 같은 매우 어려운 과제에서도 0이 아닌 성과를 낸다. 둘째, Codeforces ELO 추정에서 2300대 수준을 기록하며, Python 도구를 붙였을 때는 추가 상승한다. 이는 모델이 단순 알고리즘 암기보다, 긴 사고와 시행착오를 거쳐 문제를 해결하는 경쟁 코딩형 정책으로 정렬되었음을 보여 준다.

## 7. Acknowledgments

Acknowlegments에서는 NVIDIA NeMo 팀을 포함해 reasoning model 구축, 데이터 큐레이션, 지식집약형 SFT, 소프트웨어 엔지니어링 데이터, 인프라 설계에 기여한 다수의 협업자를 언급한다. 이 절은 논문의 실질적 성격을 다시 한 번 드러낸다. Nemotron-Cascade 2는 단순 모델 아키텍처 논문이 아니라, 데이터·학습 프레임워크·검증 환경·에이전트 scaffold가 결합된 대규모 시스템 작업이며, 그만큼 다양한 하위 팀의 협력이 필요했다.

## Appendix

## A. Benchmarks and Evaluation Setups

### A.1. Math

#### A.1.1. Non-proof Math

비증명형 수학 평가는 AIME 2025, AIME 2026, HMMT February 2025, IMO AnswerBench로 구성된다. 저자들은 Nemotron-Cascade 2를 이 벤치마크들에서 131K thinking budget, temperature 1.0, top-p 1.0으로 평가한다. with-tool 설정에서는 system prompt 뒤에 postfix를 붙여 stateful Python executor를 최대 100회 호출하게 하며, 최대 응답 길이도 131K로 둔다.

또한 baseline 모델은 공식 수치를 우선 사용하고, 공식 수치가 없을 때만 권장 추론 설정으로 직접 재평가한다. Appendix A의 관점은 단순히 benchmark 이름을 나열하는 데 있지 않다. 각 과제가 “tool use를 허용하는가”, “답안 추출은 어떻게 하는가”, “몇 번 샘플링을 평균내는가”가 결과 해석에 큰 영향을 주기 때문에, 평가 설정을 본문과 분리해 상세히 밝힌다.

#### A.1.2. Math Proof

수학 증명 평가는 IMO 2025와 IMO-ProofBench가 중심이다. Nemotron-Cascade 2는 DeepSeek-Math-V2의 generate-verify-refine 파이프라인을 따르되, 같은 지시를 사용해 비교 가능성을 확보한다. 일반 설정에서는 proof generation 128개, verification 64개, 상위 32개 proof를 대상으로 refinement 4개를 생성하는 구성이 쓰인다.

다만 모든 문제에 같은 compute budget을 쓰지는 않는다. IMO-ProofBench Basic과 일부 Advanced 문제들에서는 비용을 줄인 소형 구성도 적용한다. 또한 자동 채점이 인간 채점보다 후한 경우가 생길 수 있어, 저자들은 judge 결과를 그대로 믿지 않고 집계 규칙을 따로 분석해 사용한다. 이 점은 proof benchmark가 단순 문자열 정답 문제보다 훨씬 더 어려운 평가 문제라는 사실을 보여 준다.

### A.2. Code Reasoning

코드 추론 평가는 LiveCodeBench v6, LiveCodeBench Pro, Codeforces 시뮬레이션 등으로 구성된다. Nemotron-Cascade 2는 128K thinking budget, temperature 1.0, top-p 0.95로 평가되고, with-tool 설정에서는 Python executor를 최대 100회 호출할 수 있다. baseline 모델도 가능하면 권장 budget을 지키되, 최소 128K 수준의 thinking budget을 확보한다.

이 부록이 강조하는 것은 경쟁 코딩 평가가 단일 스칼라 점수로 환원되기 어렵다는 점이다. 제출 횟수, 부분점수, 다중 샘플 평균, 도구 허용 여부, 추정 ELO 방식이 결과를 크게 좌우하므로, 본문 Table 6만 보면 놓치기 쉬운 평가 맥락을 이 절이 보완한다.

### A.3. Knowledge and STEM

지식·STEM 평가는 MMLU-Redux, MMLU-Pro, GPQA-Diamond, Humanity’s Last Exam(HLE)로 구성된다. 저자들은 HLE에서 정답 추출을 위해 `\boxed{}` 형식을 붙이고, Appendix C.2에 제공한 judge prompt를 이용해 최종 답과 정오를 판정한다. 이를 통해 자유서술형 응답에서도 비교적 일관된 자동 채점을 수행한다.

평가 설정은 thinking mode, temperature 1.0, top-p 0.95, 128K token budget이 기본이다. 이 절은 지식·STEM 과제가 단순 상식 답변이 아니라, 어려운 질문에 대해 reasoning trace를 충분히 허용해야 제대로 평가된다는 점을 보여 준다.

### A.4. Alignment and Instruction-Following

정렬 및 지시수행 평가는 ArenaHard 2.0, IFEval, IFBench 등을 포함한다. Nemotron-Cascade 계열은 IFEval에서는 non-thinking mode, IFBench와 ArenaHard에서는 thinking mode를 사용하며, temperature 0.6, top-p 0.95, 128K budget으로 평가된다. 이는 각 benchmark가 측정하는 능력이 다르기 때문이다. IFEval은 포맷·규칙 준수에 가깝고, ArenaHard는 보다 실제 사용자형 프롬프트에서의 선호를 본다.

부록의 설명을 통해, 본문에서 IF-RL과 RLHF가 왜 अलग 단계로 분리되는지 다시 이해할 수 있다. 같은 “alignment” 범주 안에서도 objective adherence와 human preference는 서로 다른 평가 규칙을 가지며, 한쪽을 올리는 것이 다른 쪽을 항상 같이 올려 주지는 않는다.

### A.5. Long Context and Context Learning

장문 평가는 AA-LCR, LongBench v2, NIAH@1M(RULER subset), CL-Bench 등을 포함한다. 이 과제들은 단순히 긴 문서를 넣는 것이 아니라, 실제 기업 문서·정부 보고서·법률 문서처럼 구조가 복잡한 텍스트를 읽고 질문에 답하거나, 거대한 distractor 사이에서 needle을 찾아내는 능력을 측정한다.

이 절을 통해 Section 4.6의 long-context RL이 왜 domain-specific하게 분리되었는지가 더 분명해진다. 장문 문제는 입력이 길 뿐 아니라, 검색·정렬·증거 유지·정답 압축까지 요구하므로, 다른 도메인과 묶어 학습하면 손해가 커질 수 있다.

### A.6. Agentic Tasks

에이전트 평가는 BFCL v4, τ2-Bench, Terminal Bench 2.0, SWE-bench Verified(OpenHands)를 포함한다. BFCL은 함수 호출과 메모리 조작, 웹 검색 같은 일반 agentic tool use를 본다. τ2-Bench는 멀티턴 상호작용 속 reasoning trace와 도구 사용을 본다. Terminal Bench 2.0은 터미널 환경에서의 작업 수행을 본다. SWE-bench Verified는 장기적인 코드 수정 에이전트 성능을 본다.

이 절의 중요한 기술적 포인트는 **thought retention policy**다. 예를 들어 τ2-Bench 평가에서는 최신 턴의 reasoning만 유지하는 정책을 사용한다. 반면 SWE-bench OpenHands 평가는 파일 보기, 검색 결과, 실행된 명령, 중간 패치까지 모두 긴 문맥으로 유지한다. 즉, 에이전트 평가는 “모델이 생각을 잘하나”만이 아니라, **어떤 상태를 얼마만큼 보존하며 다음 턴으로 넘기는가**가 함께 평가된다.

### A.7. Multilingual

다국어 평가는 MMLU-ProX와 WMT24++를 사용한다. MMLU-ProX는 MMLU-Pro를 여러 언어로 확장한 벤치마크이고, WMT24++는 영어에서 독일어·스페인어·프랑스어·이탈리아어·일본어로의 번역을 XCOMET-XXL로 평가한다. 이 절은 Nemotron-Cascade 2가 मुख्य 영어 reasoning 모델이면서도, 사후학습이 다국어 성능을 완전히 희생하지 않았음을 보여 준다.

## B. Training Hyperparameters

<!-- Table 7 -->

> [Table 7 삽입 위치]

Table 7은 SFT 하이퍼파라미터를 정리한다. global batch size 64, packed sequence length 256K, max learning rate \(5\times10^{-5}\), min learning rate \(5\times10^{-6}\), warmup 200 step, cosine scheduler, max step 40,000, AdamW(\(\beta_1=0.9,\beta_2=0.98\)), weight decay 0.1, 실제 training step 33,000이다. Section 3에서 설명한 “single-stage, 256K packing, 약 1.5 epoch”가 이 표에서 구체적인 숫자로 고정된다.

<!-- Table 8 -->

> [Table 8 삽입 위치]

Table 8은 IF-RL, multi-domain RL, MOPD의 설정을 나란히 보여 준다. 세 단계 모두 batch size는 128이지만, max response length와 rollout size, optimizer가 다르다. IF-RL과 multi-domain RL은 49K max response를 쓰고, MOPD는 98K까지 늘린다. rollout size는 IF-RL과 multi-domain RL이 16, MOPD가 4다. step 수는 180, 70, 52로, Cascade RL이 도메인별로 얼마나 다른 budget을 배정받는지 한눈에 드러난다.

<!-- Table 9 -->

> [Table 9 삽입 위치]

Table 9는 RLHF, long-context RL, code RL을 비교한다. RLHF는 16K max response로 가장 짧고, long-context RL은 49K, code RL은 118K까지 사용한다. 세 단계 모두 batch 128, rollout 16, learning rate \(3\times10^{-6}\)이지만, top-p는 code RL만 0.95이고, overlong filtering은 세 단계 모두 true다. 표만 봐도 저자들이 도메인 특성에 따라 “필요한 사고 길이”를 얼마나 다르게 보고 있는지가 분명해진다.

<!-- Table 10 -->

> [Table 10 삽입 위치]

Table 10은 execution-based agentic SWE-RL의 별도 설정을 보여 준다. prompts per step 16, rollout 64, temperature 0.8, max sequence length 256K, max turn 200, max learning rate \(3\times10^{-6}\), min learning rate 0, warmup 10 step이다. 다른 RL 단계보다 rollout 수와 상호작용 횟수가 크다는 점이 눈에 띄며, 이는 agentic SWE가 단발 응답보다 긴 궤적 제어를 필요로 하기 때문이다.

## C. Prompt Templates

### C.1. Prompt Templates for Test-Time Scaling on IOI 2025

Appendix C.1은 IOI test-time scaling에 쓰인 실제 프롬프트 템플릿을 제시한다. 이 템플릿은 문제 설명, 제출 형식, 이전 제출 결과, subtask별 피드백, 다른 subtask에서 얻은 통찰을 모델 입력 안에 체계적으로 넣는다. 모델은 단순히 “정답 코드를 한 번에 써라”가 아니라, 현재까지의 시도와 남은 오류를 참고해 다음 candidate를 생성해야 한다.

이 프롬프트가 의미하는 바는, IOI 성능이 단지 base coding ability만의 결과가 아니라는 점이다. test-time scaling 파이프라인 전체가 모델에게 일종의 외부 작업 기억과 자기개선 루프를 제공하고, 그 위에서 모델이 reasoning trace와 코드 생성을 반복한다.

### C.2. HLE Judge Prompt

Appendix C.2는 HLE 채점용 judge prompt를 제시한다. 입력으로 question, response, correct_answer를 주고, 출력으로는 extracted_final_answer, reasoning, correct 여부, confidence를 구조적으로 생성하게 한다. 즉, 자유서술형 응답에서도 최종 답안과 정오를 분리해 채점할 수 있도록 설계한다.

이 템플릿은 벤치마크 평가가 단순히 모델 출력과 정답 문자열을 직접 비교하는 일이 아니라, 별도의 judge 설계 문제이기도 하다는 점을 잘 보여 준다. 특히 HLE처럼 길고 복잡한 응답에서는 최종 답안 추출 단계가 성능 수치에 큰 영향을 준다.

## D. ELO Rating Analysis

Appendix D는 2501–2507 구간의 최근 Codeforces Div.1/Div.2 40개 대회에 대한 ELO 분석을 제공한다. 문제는 LiveCodeBench Pro 계열에서 가져오고, 모델은 각 대회 문제에 최대 \(N=8\)번까지 제출을 시도하는 방식으로 시뮬레이션된다. 응답 생성에는 temperature 1.0, top-p 0.95, 128K token budget이 쓰인다.

이 절의 핵심은 단일 benchmark average가 아니라 **대회 단위 성능 분포**다. 어떤 라운드에서는 Div.1에서도 매우 높은 순위를 추정하고, 어떤 라운드에서는 특정 난이도 구간에서만 약점을 드러낸다. 따라서 Codeforces ELO는 단순 pass@1보다 모델의 실제 경쟁력과 변동성을 더 잘 보여 준다.

<!-- Table 11 -->

> [Table 11 삽입 위치]

Table 11은 Python tool 없이 각 대회별 점수, 패널티, 추정 순위, 문제별 해결 상태를 정리한다. 이 표를 보면 Nemotron-Cascade 2가 여러 Div.2 라운드에서 1위권 추정치를 기록할 만큼 강하지만, Div.1의 일부 어려운 문제군에서는 여전히 해결률 변동이 크다는 점도 드러난다.

<!-- Table 12 -->

> [Table 12 삽입 위치]

Table 12는 같은 분석을 Python tool 사용 설정으로 반복한다. tool use는 모든 문제를 일률적으로 해결해 주는 만능 열쇠는 아니지만, 긴 계산이나 상태 검증이 필요한 문제에서는 분명한 이득을 준다. Appendix D의 역할은 바로 이 점, 즉 **모델 자체의 알고리즘 추론력**과 **외부 계산 보조의 시너지**를 contest-by-contest 수준에서 분리해 보여 주는 데 있다.

## E. IMO 2025 Model Solutions

Appendix E는 모델이 실제로 생성한 IMO 2025 풀이와 인간 전문가 코멘트를 모아 둔 절이다. 여기에는 Problem 1부터 Problem 5까지의 해설이 포함되며, 이는 본문 Section 5.1에서 언급한 “첫 다섯 문제 해결”과 정확히 대응한다. 단순 점수만 보면 놓치기 쉬운 점, 즉 어떤 풀이가 간결했고 어떤 풀이는 장황했는지, 어디서 내부 사고 흔적이 비치고 어디서 정교한 정리가 이뤄졌는지를 이 절이 보여 준다.

### Problem 1

문제 1의 short answer는 \(k=0,1,3\)이다. 모델 풀이의 구조는 크게 두 부분으로 나뉜다. 먼저 \(S_n=\{(a,b)\in\mathbb Z^2\mid a\ge1,b\ge1,a+b\le n+1\}\)를 정의하고, 실제로 \(k=0,1,3\)이 가능함을 구성적으로 보인다. \(k=0\)은 수직선들만으로 덮는 가장 단순한 구성이고, \(k=1\)과 \(k=3\)은 일부 남는 점들을 sunny line으로 조합해 처리한다.

그다음은 귀납적 불가능성 증명이다. 어떤 \(n\)에 대해서도 \(n\)개의 선으로 \(S_n\)을 덮을 때 가능한 sunny line 개수가 \(\{0,1,3\}\)밖에 아님을 보이기 위해, 경계 집합을 분석하고 \(n-1\)로 줄이는 절차를 사용한다. 특히 \(x=1\)과 같은 특수한 선을 제거한 뒤 나머지를 평행이동해 \(S_{n-1}\) 문제로 환원하는 아이디어가 핵심이다. 마지막에는 \(n=3\) 기저 사례를 직접 확인해 귀납을 닫는다.

인간 전문가 평가는 7/7이다. 코멘트의 요지는 풀이가 참신한 표현을 쓰더라도 논리 구조가 완결되어 있고, 구성과 불가능성의 두 축이 모두 제대로 세워져 있다는 것이다.

### Problem 2

문제 2는 short answer가 따로 없는 기하 증명 문제다. 모델은 순수 기하적 해법 대신 **좌표기하와 대수 계산**을 선택한다. 원 \(\Omega,\Gamma\)의 중심을 각각 \(M=(0,0)\), \(N=(d,0)\)로 두고, 교점 \(A,B\), 보조점 \(P,E,F,S,H,O\)를 좌표로 정의한 뒤, 접선 조건과 원의 중심 조건을 식으로 밀어붙여 결론에 도달한다.

풀이의 흐름은 다음과 같다. 먼저 원들의 방정식과 교점 좌표를 정리하고, \(P\)를 삼각형의 외심으로 놓은 뒤 \(AP\) 방향 벡터를 사용해 \(E,F\)의 위치를 파라미터로 나타낸다. 다음으로 \(EF\)의 중점 \(S\), \(PMN\)의 orthocenter \(H\), 그리고 \(BEF\)의 circumcenter \(O\)를 차례로 계산한다. 마지막에는 거리 조건과 접선 조건을 대수식으로 바꾸고, 여러 보조량 \(U,V,K,L,T\)를 도입해 동일식을 확인한다.

전문가 코멘트는 이 풀이가 우아하다고 보기는 어렵지만, 좌표 설정, 도출, 대입, 최종 정리까지 모두 닫혀 있어 full-score에 해당한다고 평가한다. 즉, 모델이 기하 문제에서 “가장 아름다운 해법” 대신 “계산량이 많더라도 틀리지 않는 해법”을 택할 수 있음을 보여 준다.

### Problem 3

문제 3의 short answer는 \(c=4\)다. 모델은 먼저 bonza 함수의 정의에서 출발해 기본 성질을 끌어낸다. \(f(a)\mid a^a\)와 같은 형태를 얻어, \(f(a)\)의 소인수가 \(a\)의 소인수에 의해 강하게 제한된다는 사실을 확인한다. 이후 prime 입력에서 \(f(p)\)의 형태를 분석하고, \(S=\{p\text{ prime}\mid f(p)>1\}\) 같은 집합을 두어 경우를 나눈다.

증명의 중심은 \(S\)가 무한한 경우와 그렇지 않은 경우를 가르는 데 있다. odd prime이 \(S\)에 하나라도 들어가면 특정 합동 조건 때문에 \(S\)가 무한해져 버리고, 결국 가능한 경우가 크게 제한된다. 이후 \(S=\varnothing\)이면 상수 1 함수로 붕괴하고, \(S=\{2\}\)인 경우에는 \(2\)-진 valuation을 세밀하게 분석해 \(f(n)\le 4n\)을 얻는다.

마지막에는 실제로 equality를 달성하는 bonza 함수 예시를 만들어, 상한 \(4\)가 최적임을 증명한다. 전문가 평가는 7/7이며, 핵심 아이디어 전개와 sharpness 증명까지 모두 갖췄다고 본다.

### Problem 4

문제 4는 가능한 초기값 \(a_1\)의 정확한 형태를 묻는다. 모델의 short answer는 이를 특정 인수분해 형태로 완전히 분류하는 것이다. 본문에서는 최종적으로 \(a_1 = 6\cdot 12^k\cdot d\) (\(k\in\mathbb N_0\), \(d\) odd, \(5\nmid d\))와 동치인 형태로 정리된다.

모델 풀이의 초반은 약수 구조 분석이다. 어떤 \(N\)의 proper divisor들을 \(d_1<d_2<d_3<d_4<\cdots\)로 놓고, 가장 큰 proper divisor들이 \(N/d_2,N/d_3,N/d_4\)라는 점을 이용해 점화식의 성질을 파고든다. 이후 \(f(N)\)이 어떤 경우에는 \(N\)보다 작아지고, 어떤 경우에는 6의 배수 꼴로 환원된다는 사실을 보이며, 결국 모든 문제가 \(N=6M\)의 형태를 분석하는 것으로 줄어든다.

그다음은 \(M\)의 짝홀성과 5의 배수 여부에 따라 \(d_4\)가 4, 5, 6 중 무엇이 되는지를 나누어 계산한다. 이 계산을 통해 수열이 어떻게 12의 거듭제곱을 줄여 가며 13의 인수를 늘리는지, 그리고 왜 마지막 remainder \(d\)가 odd이면서 5로 나누어떨어지지 않아야 하는지를 증명한다. 전문가 평가는 이 전체 분류가 완결되었다고 보고 7/7을 부여한다.

### Problem 5

문제 5의 short answer는 \(\lambda\)의 임계값이 \(\sqrt{2}/2\)라는 것이다. 정확히는,
- \(\lambda>\sqrt{2}/2\)이면 Alice에게 winning strategy가 있고,
- \(\lambda<\sqrt{2}/2\)이면 Bazza에게 winning strategy가 있으며,
- \(\lambda=\sqrt{2}/2\)이면 누구도 강제 승리를 만들 수 없어 무승부가 가능하다.

모델은 먼저 \(S_n=\sum_{i=1}^n x_i\), \(Q_n=\sum_{i=1}^n x_i^2\)를 정의하고, 짝수 턴 뒤에는 Cauchy–Schwarz로 \(S_{2k}\le 2k\)를 얻는다. 이어서 even turn 뒤의 slack
\[
d_k=\lambda(2k+1)-S_{2k}
\]
를 도입해, Alice가 다음 odd turn에서 합 제약을 만족하며 움직일 수 있는지를 추적한다. Bazza의 핵심 전략은 가능한 한 \(Q_{2k+2}=2k+2\)를 꽉 채우는 **maximal strategy**다. 이 전략을 쓰면 \(u+v\)의 최소치가 \(\sqrt2\) 이상이 되어, slack의 증감이 사실상 \(2\lambda-\sqrt2\)의 부호에 의해 결정된다.

- \(\lambda<\sqrt2/2\)에서는 \(2\lambda-\sqrt2<0\)이므로 \(d_k\)가 결국 음수가 되어 Alice가 합 제약 때문에 더 이상 움직일 수 없게 된다.
- \(\lambda=\sqrt2/2\)에서는 \(2\lambda-\sqrt2=0\)이라 slack이 유지되거나 줄어들 뿐이어서, Alice는 0을 고르는 식으로 패배를 피할 수 있지만 동시에 즉시 승리도 만들지 못한다. 결과적으로 draw가 가능하다.
- \(\lambda>\sqrt2/2\)에서는 \(d_k\)가 결국 충분히 커져, Alice가 어떤 odd turn에서 \(Q_{2k+1}>2k+2\)를 강제로 만드는 \(u\)를 고를 수 있고, 다음 even turn에서 Bazza가 움직일 수 없게 된다.

전문가 코멘트는 논증 중 일부 문장이 사고 흔적을 그대로 드러내는 듯한 표현을 포함한다고 지적한다. 그럼에도 불구하고 전략의 구분, 임계값 도출, 세 경우의 완결된 결론은 모두 옳다고 보아 7/7을 준다.

## References

참고문헌은 크게 다섯 갈래로 기능한다. 첫째, Nemotron-Cascade, Nemotron-3-Nano, Nano-v3 같은 선행 NVIDIA 계열 작업이 본 논문의 데이터와 학습 레시피의 직접적 기반을 이룬다. 둘째, DeepSeek-Math-V2, GPT-OSS, Qwen3, Gemini Deep Think, OpenAI 계열 모델이 교사·judge·비교 기준으로 사용된다. 셋째, LiveCodeBench, ArenaHard 2.0, BFCL v4, τ2-Bench, LongBench v2, AA-LCR, HLE, MMLU-ProX 같은 평가 프레임워크가 모델의 다면적 성능을 측정한다. 넷째, OpenHands, SWE-Agent, SWE-Gym, R2E-Subset, Terminal-Task-Gen 같은 agentic scaffold와 환경이 도구 사용 및 소프트웨어 엔지니어링 학습의 바탕을 이룬다. 다섯째, GRPO, RLVR, on-policy distillation, reward modeling, proof generation-evaluation 파이프라인에 관한 선행 연구들이 Section 4의 핵심 설계를 이론적으로 뒷받침한다.
