# Demystifying Instruction Mixing for Fine-tuning Large Language Models

## Abstract

이 논문은 instruction tuning에 사용되는 데이터 조합이 대규모 언어모델의 성능을 어떻게 바꾸는지를 체계적으로 분석한다. 저자들은 instruction 데이터를 크게 세 종류로 나눈다. 첫째는 NLP downstream task를 지시문 형태로 바꾼 데이터, 둘째는 코드 생성 중심 데이터, 셋째는 일반 대화형 instruction 데이터다. 논문은 이 세 부류를 대표하는 P3, CodeAlpaca, Alpaca를 선택하고, 이들을 어떻게 섞느냐에 따라 NLP 벤치마크 성능, 코드 생성 성능, 그리고 alignment 능력, 즉 채팅 능력이 어떻게 달라지는지를 비교한다. 핵심 결론은 단순히 데이터를 많이 섞는다고 모든 능력이 함께 좋아지지는 않는다는 점이다. 특정 instruction 데이터는 자기와 가까운 영역에서는 분명한 이득을 주지만, 다른 영역에서는 오히려 손실을 만들 수도 있다. 특히 NLP task 중심 instruction은 벤치마크 성능을 끌어올리는 반면 대화형 정렬 능력에는 부정적일 수 있고, 코드 instruction은 코딩 능력뿐 아니라 일부 alignment 능력에도 도움을 준다.

## 1. Introduction

Instruction tuning은 대규모 언어모델이 사람의 지시를 더 잘 따르도록 만드는 데 매우 효과적이라고 알려져 왔다. 다만 instruction tuning이라고 해서 하나의 동일한 효과만 있는 것은 아니다. 일반 대화형 instruction은 챗봇 성능을 개선하고, NLP task를 instruction으로 다시 표현한 데이터는 각종 NLP 벤치마크 점수를 끌어올리며, 코드 instruction은 코드 생성 성능을 강화한다. 문제는 이 서로 다른 성격의 instruction 데이터를 실제 fine-tuning에서 어떻게 섞어야 전체적으로 좋은 모델이 되는지 아직 잘 알려져 있지 않다는 데 있다.

이 논문은 바로 그 질문을 정면으로 다룬다. 저자들은 instruction mixing의 영향을 세 가지 핵심 영역에서 본다. NLP downstream task, coding, 그리고 chat이다. 이를 위해 P3를 NLP task instruction의 대표로, CodeAlpaca를 code generation instruction의 대표로, Alpaca를 general-purpose instruction의 대표로 선택한다. 이후 세 데이터를 단독으로 쓰거나, 둘씩 섞거나, 셋을 모두 섞는 식으로 총 여덟 가지 조합을 만들고, 각 조합이 어떤 능력을 강화하고 어떤 능력을 희생시키는지 비교한다.

![Figure 1. P3와 Alpaca의 instruction 유형 분포.](/assets/images/for%20Fine-tuning%20Large%20Language%20Models/figure_01_instruction_type_distribution_p3_alpaca.png)

*Figure 1. P3와 Alpaca의 instruction 유형 분포. P3는 원본 데이터의 통계를 사용했고, Alpaca는 각 instruction의 root verb를 dependency parsing으로 추출해 집계했다.*

Figure 1은 두 데이터의 성격 차이를 단번에 보여준다. P3는 질문응답, 분류, 요약, 구조-텍스트 변환, 패러프레이즈 판별처럼 비교적 제한된 NLP task 축에 집중되어 있다. 반대로 Alpaca는 훨씬 다양한 instruction을 포함하며, 저자들은 dependency parser를 사용해 1천 개가 넘는 고유 root verb를 식별한다. 그중 generate, create, describe가 가장 자주 등장한다. CodeAlpaca는 이 둘과 달리 코드 생성에 거의 전적으로 집중되어 있고 변이도도 더 작다. 이 차이는 뒤의 Table 3 예시에서도 다시 확인된다.

논문의 기여는 세 가지로 정리된다.

- 특정 instruction 데이터는 자신이 설계된 영역에서 분명한 이득을 준다. 다만 모든 종류를 한꺼번에 섞는다고 모든 과제에서 성능이 고르게 좋아지지는 않는다.
- NLP downstream task를 instruction 형태로 바꾼 데이터, 대표적으로 P3는 NLP 벤치마크에는 유리하지만 모델의 대화 능력에는 악영향을 줄 수 있다. 반면 코드 중심 instruction은 코딩 성능뿐 아니라 일부 채팅 정렬 능력도 함께 끌어올린다.
- 더 큰 모델은 더 많은 용량을 바탕으로 다양한 instruction을 작은 모델보다 더 효과적으로 활용할 수 있다.

## 2. Related Work

관련 연구는 크게 두 흐름으로 정리된다. 첫째는 NLP 데이터셋을 자연어 지시문 형태로 바꾸어 instruction tuning 데이터로 만든 연구다. P3는 대표적인 예로, 다양한 supervised NLP dataset을 여러 프롬프트와 템플릿으로 다시 써서 하나의 instructional corpus로 만든다. 이런 접근은 NLP 벤치마크 성능에서는 강력하지만, 챗봇처럼 사람과 자연스럽게 상호작용하는 설정에서는 한계가 있었다.

둘째는 사람과의 상호작용에 더 가까운 general-purpose instruction 데이터를 구축한 연구다. Dolly, Self-Instruct, Alpaca 같은 작업은 사람 주석이나 자동 생성 방식을 통해 일반적인 질문과 요청에 답하는 데이터를 만들었고, 이후 데이터 규모, 언어 범위, 과제 종류가 계속 확장되었다. 여기에 CodeAlpaca나 MAmmoTH처럼 특정 능력에 특화된 instruction 데이터도 늘어나면서, 하나의 모델에 서로 다른 능력을 동시에 심는 시도가 활발해졌다.

하지만 기존 연구는 instruction mixture의 효과를 체계적으로 분리해서 보지 못했다. 어떤 연구는 여러 데이터를 섞었지만 데이터 볼륨과 task type을 엄밀히 통제하지 않았고, 어떤 연구는 alignment를 위해 아예 P3를 제외했다. 또 동시기 연구로 Wang et al.은 12개 instruction tuning dataset을 각각 따로 fine-tuning해 7개 task에서 비교하고, 최적의 조합을 찾으려 했다. 이에 비해 이 논문은 instruction과 모델 skill을 NLP, code, chat 세 축으로 정리하고, mixture가 각 축에 미치는 영향을 더 깊이 분석한다는 점에서 차별화된다.

## 3. Experimental Setup

### Datasets

저자들은 Alpaca를 general instruction dataset으로 사용한다. Alpaca는 52K instruction-response pair로 구성되어 있으며, 사람과 대화하듯 일반적인 요청을 처리하는 데 초점을 둔다. P3는 NLP downstream task를 다양한 human-written template으로 재구성한 데이터다. 각 세부 task의 샘플 수 차이가 매우 크기 때문에, 저자들은 각 subtask에서 1천 개 예시를 여러 프롬프트 형식으로 추출해 총 660K 샘플을 만든다. CodeAlpaca는 코드 생성 중심 instruction dataset이며, 서로 다른 프로그래밍 언어에 걸쳐 20K 샘플을 포함한다.

비교를 공정하게 맞추기 위해 실제 fine-tuning에서는 각 데이터에서 20K subset만 사용한다. 따라서 이 논문의 비교는 데이터의 절대량이 아니라 데이터의 성격과 조합 방식에 더 초점을 둔다. 세 데이터의 실제 예시는 Appendix A의 Table 3에서 제시된다.

### Evaluation

평가는 세 갈래로 나뉜다. 첫째는 NLP benchmark 성능이다. ARC, Winogrande, PIQA, MMLU, RACE, HellaSwag를 사용한다. 둘째는 code generation 평가다. 여기서는 HumanEval을 사용해 생성한 코드의 통과율을 본다. 셋째는 alignment evaluation, 즉 채팅 능력 평가다. 이를 위해 FLASK 프레임워크를 사용하고, 원래 평가 세트에서 가장 자주 등장하는 8개 alignment skill만 남겨 총 1,180개 샘플을 만든다. 이후 GPT-4가 human-written principle에 따라 각 응답을 평가한다. 각 skill의 의미는 Appendix B에서 자세히 설명된다.

### Models

실험 모델은 LLaMA-2 7B와 13B다. 두 모델 모두 generative 방식으로 2 epoch fine-tuning하며, linear scheduler와 3% warmup rate를 사용한다. batch size는 64이고, 최대 learning rate는 5×10^-5다. 학습 자원과 평가 비용은 Appendix C에서 별도로 정리된다.

## 4. Results

이 절에서 저자들은 Alpaca, CodeAlpaca, P3를 각각 A, C, P로 줄여 부른다. 그리고 데이터 혼합 전략은 None, A, C, P, AC, AP, CP, ACP의 여덟 가지로 표기한다. None은 fine-tuning을 하지 않은 원래 모델이고, 나머지는 표기된 데이터셋으로 fine-tuning한 모델이다. 예를 들어 AC는 Alpaca와 CodeAlpaca를 함께 사용한 경우다.

![Table 1. NLP와 code generation benchmark 결과.](/assets/images/for%20Fine-tuning%20Large%20Language%20Models/table_01_nlp_and_code_generation_results.png)

*Table 1. NLP와 code generation benchmark 결과. 모든 실험은 zero-shot 설정에서 수행되었고, 굵은 값은 최고 성능, 밑줄은 두 번째 성능을 뜻한다.*

![Figure 2. LLaMA-2-7B에서 mixing ratio와 데이터 수에 따른 NLP 및 code benchmark 변화.](/assets/images/for%20Fine-tuning%20Large%20Language%20Models/figure_02_nlp_and_code_benchmark_scores.png)

*Figure 2. LLaMA-2-7B에서 mixing ratio와 데이터 수에 따른 NLP benchmark 평균 점수와 code benchmark, 즉 HumanEval 점수의 변화. Alpaca는 20K로 고정하고 P3와 CodeAlpaca의 양을 바꾸어 비율을 조절했다.*

### 4.1 NLP Tasks and Code Benchmark Results

Table 1은 먼저 각 전문화된 instruction 데이터가 자기 영역에서 강점을 가진다는 점을 보여준다. 단일 데이터만 사용하는 no-mixture setting에서 P3로 학습한 모델은 NLP 과제 평균이 가장 높고, CodeAlpaca로 학습한 모델은 code generation에서 가장 강하다. 왜 이런 차이가 나는지는 각 benchmark의 형식과 fine-tuning 데이터의 형식이 얼마나 닮아 있는지를 보면 이해하기 쉽다. Alpaca로 fine-tuning한 모델은 story completion과 닮은 형식의 RACE와 HellaSwag에서 강한 반면, P3로 fine-tuning한 모델은 multiple-choice QA와 cloze test 성격이 강한 ARC와 Winogrande에서 더 잘 나온다.

혼합 setting에서는 specialized data를 포함하는지가 해당 benchmark에서 중요한 차이를 만든다. 예를 들어 NLP benchmark만 놓고 보면 P, AP, CP, ACP처럼 P3가 들어간 조합이 None, A, C, AC보다 일관되게 유리하다. 반대로 code benchmark에서는 CodeAlpaca가 없는 조합보다 CodeAlpaca가 포함된 조합이 낫다. 특히 흥미로운 점은 general instruction을 함께 넣으면 coding 성능이 더 좋아지는 경우가 많다는 것이다. 7B에서 C의 HumanEval 점수는 @1 16.2, @10 24.4인데 AC는 @1 17.5, @10 25.0으로 오른다. 13B에서는 C가 @1 17.9, @10 24.4이고 AC가 @1 17.1, @10 27.4라서 @1은 예외적으로 조금 낮아졌지만 @10은 크게 상승한다.

모델 크기에 따른 차이도 중요하다. 7B에서는 AC가 code benchmark에서 가장 좋은 조합으로 나타나고, 13B에서는 ACP가 HumanEval @1 20.2, @10 32.9로 가장 강하다. 반면 13B의 NLP 평균은 CP가 61.9로 최고다. 즉 더 큰 모델일수록 다양한 instruction을 함께 넣었을 때 충돌보다 분업의 이점을 더 잘 흡수하는 경향이 있다. 논문은 이를 larger models can better learn from varied instructions라는 메시지로 정리한다.

이 결과는 단순하지만 실용적인 결론으로 이어진다. instruction mixture는 데이터 다양성을 최대화하는 문제가 아니라, 목표로 하는 모델 행동을 어떻게 조합할 것인가의 문제다. NLP benchmark를 올리고 싶은지, coding 능력을 키우고 싶은지, 혹은 대화형 assistant로 더 정렬시키고 싶은지에 따라 최적 조합이 달라진다.

#### Mixing with Different Ratios

Figure 2의 위쪽 그래프는 Alpaca를 20K로 고정한 채 specialized instruction의 비율을 늘려 갈 때 성능이 어떻게 바뀌는지를 보여준다. 결과는 단조 증가가 아니다. NLP 평균과 code benchmark 모두 비율이 낮을 때는 잠깐 떨어졌다가, specialized instruction 비율이 커질수록 다시 올라간다. 두 곡선은 ratio 1.5에서 정점을 찍고, 2.0에서는 약간 되돌아간다. 저자들은 이를 specialized instruction이 너무 많아지면 모델이 그 형식에 과적합하기 때문이라고 해석한다.

#### Number of Instances

Figure 2의 아래쪽 그래프는 세 종류 instruction을 같은 수로 섞되, 총 데이터 수를 늘렸을 때의 변화를 보여준다. 이 설정에서는 NLP benchmark와 code benchmark 모두 대략 10K를 넘어가면 증가 폭이 눈에 띄게 줄어든다. 즉 이 논문의 실험 범위에서는 무작정 더 많은 instruction을 넣기보다, 어떤 성격의 instruction을 얼마나 섞느냐가 더 본질적인 문제로 드러난다.

### 4.2 Alignment Skill Results

alignment 평가는 FLASK의 설정을 따르며, GPT-4-0613이 각 skill 점수를 0에서 100 사이로 부여한다. 원래 모델, 즉 None은 instruction을 제대로 따르지 못하기 때문에 Table 2에서는 제외된다.

![Table 2. alignment skill assessment 결과.](/assets/images/for%20Fine-tuning%20Large%20Language%20Models/table_02_alignment_skill_assessment_results.png)

*Table 2. GPT-4를 이용한 alignment skill assessment 결과. 논리적 정확성, 사실성, 상식 이해, 이해도, 완전성, 통찰성, 가독성, 간결성과 평균 점수를 함께 보고한다. 굵은 값은 최고 성능, 밑줄은 두 번째 성능이다.*

Table 2에서 첫 번째로 눈에 띄는 점은 single-dataset setting에서 Alpaca가 가장 alignment 친화적이라는 사실이다. 7B에서 평균 점수는 A가 60.6, C가 57.4, P가 45.8이고, 13B에서는 A가 64.0, C가 62.4, P가 47.8이다. Alpaca는 일반적인 요청과 사람 같은 응답 형식을 담고 있기 때문에, 사람과 상호작용하는 assistant를 정렬하는 데 가장 잘 맞는 데이터로 해석된다.

두 번째 관찰은 CodeAlpaca의 역할이다. CodeAlpaca만 단독으로 쓰면 alignment가 크게 오르지는 않지만, Alpaca와 함께 쓰면 가장 높은 평균 점수가 나온다. 7B에서는 A의 60.6이 AC에서 61.2로, 13B에서는 A의 64.0이 AC에서 65.6으로 올라간다. 특히 commonsense understanding, comprehension, completeness, conciseness 쪽에서 개선이 두드러진다. 논문은 이를 코드 instruction이 단지 코딩 능력만 주는 것이 아니라, 보다 정밀하고 조건을 잘 지키는 응답 패턴을 학습시키는 효과가 있을 수 있음을 시사하는 결과로 읽는다.

세 번째 관찰은 P3의 부작용이다. 저자들은 P3를 섞으면 alignment 평균이 7B에서 약 2.8점, 13B에서 약 3.6점 떨어진다고 정리한다. 실제 표를 봐도 P와 P가 들어간 여러 조합은 completeness, insightfulness, readability 같은 항목에서 특히 약하다. 이는 P3가 나쁜 데이터라기보다, 정답 중심의 task-format instruction이 사람과의 자연스러운 대화 응답 형식과는 다르기 때문이다. 다시 말해 benchmark score를 올리는 데이터와 chat assistant를 잘 만드는 데이터가 항상 같은 것은 아니다.

결국 이 절은 instruction tuning의 중요한 역설을 보여준다. 어떤 데이터는 벤치마크 점수를 올리지만 대화형 정렬을 해칠 수 있고, 어떤 데이터는 채팅 품질을 높이면서 다른 능력까지 조금씩 끌어줄 수 있다. 따라서 mixture 설계는 평균 점수 하나만 보고 결정할 수 없다.

## 5. Conclusion

이 논문은 instruction fine-tuning에서 서로 다른 데이터 혼합 전략을 비교하고, 그 효과를 NLP benchmark, code generation, alignment skill이라는 세 축에서 측정했다. 결론은 분명하다. general instruction은 alignment와 일부 NLP benchmark에서 좋고, code instruction은 coding 능력과 일부 alignment 능력을 개선하며, NLP task instruction은 NLP benchmark에는 강하지만 다른 instruction과 섞였을 때 alignment를 해칠 수 있다. 즉 fine-tuning의 성패는 단순한 데이터 양보다도 어떤 능력을 목표로 어떤 성격의 데이터를 섞느냐에 달려 있다.

## Limitations

저자들은 두 가지 한계를 분명히 적는다.

1. 실험 모델이 LLaMA-2 7B와 13B 두 종류뿐이다. 모델 크기와 구조가 달라지면 instruction mixture의 효과도 달라질 수 있으므로, 더 다양한 모델에서 검증이 필요하다.
2. 각 instruction 데이터는 20K로 제한했고, 주로 1:1 비율을 중심으로 비교했다. 따라서 더 많은 데이터, 더 다양한 mixing ratio, 더 극단적인 조합에서 어떤 현상이 나타나는지는 후속 연구가 필요하다.

논문은 이런 한계를 인정하면서도, 바로 그 지점이 다음 단계 연구의 핵심이라고 본다. instruction mixture의 영향을 더 넓은 범위에서 측정해야 community가 truly useful한 tuning recipe를 만들 수 있다는 것이다.

## References

- Anand, Y., Nussbaum, Z., Duderstadt, B., Schmidt, B., and Mulyar, A. 2023. GPT4All: Training an assistant-style chatbot with large scale data distillation from GPT-3.5-Turbo.
- Bisk, Y., Zellers, R., Gao, J., and Choi, Y. 2020. PIQA: Reasoning about physical commonsense in natural language.
- Chaudhary, S. 2023. Code Alpaca: An instruction-following LLaMA model for code generation.
- Chen, M., Tworek, J., Jun, H., Yuan, Q., Ponde de Oliveira Pinto, H., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., et al. 2021. Evaluating large language models trained on code.
- Chiang, W.-L., Li, Z., Lin, Z., Sheng, Y., Wu, Z., Zhang, H., Zheng, L., Zhuang, S., Zhuang, Y., Gonzalez, J. E., Stoica, I., and Xing, E. P. 2023. Vicuna: An open-source chatbot impressing GPT-4 with 90%* ChatGPT quality.
- Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, Y., Wang, X., Dehghani, M., Brahma, S., et al. 2022. Scaling instruction-finetuned language models.
- Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., and Tafjord, O. 2018. Think you have solved question answering? Try ARC, the AI2 reasoning challenge.
- Conover, M., Hayes, M., Mathur, A., Xie, J., Wan, J., Shah, S., Ghodsi, A., Wendell, P., Zaharia, M., and Xin, R. 2023. Free Dolly: Introducing the world’s first truly open instruction-tuned LLM.
- Ding, N., Chen, Y., Xu, B., Qin, Y., Zheng, Z., Hu, S., Liu, Z., Sun, M., and Zhou, B. 2023. Enhancing chat language models by scaling high-quality instructional conversations.
- Fu, H. and Khot, T. 2022. How does GPT obtain its ability? Tracing emergent abilities of language models to their sources.
- Gunasekar, S., Zhang, Y., Aneja, J., Mendes, C. C. T., Del Giorno, A., Gopi, S., Javaheripi, M., Kauffmann, P., de Rosa, G., Saarikivi, O., Salim, A., Shah, S., Behl, H. S., Wang, X., Bubeck, S., Eldan, R., Kalai, A. T., Lee, Y. T., and Li, Y. 2023. Textbooks are all you need.
- Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J. 2020. Measuring massive multitask language understanding.
- Lai, G., Xie, Q., Liu, H., Yang, Y., and Hovy, E. 2017. RACE: Large-scale ReAding comprehension dataset from examinations.
- Li, H., Koto, F., Wu, M., Aji, A. F., and Baldwin, T. 2023. Bactrian-X: Multilingual replicable instruction-following models with low-rank adaptation.
- Longpre, S., Hou, L., Vu, T., Webson, A., Chung, H. W., Tay, Y., Zhou, D., Le, Q. V., Zoph, B., Wei, J., and Roberts, A. 2023. The Flan collection: Designing data and methods for effective instruction tuning.
- Mishra, S., Khashabi, D., Baral, C., and Hajishirzi, H. 2022. Cross-task generalization via natural language crowdsourcing instructions.
- Muennighoff, N., Wang, T., Sutawika, L., Roberts, A., Biderman, S., Le Scao, T., Bari, M. S., Shen, S., Yong, Z. X., Schoelkopf, H., Tang, X., Radev, D., Aji, A. F., Almubarak, K., Albanie, S., Alyafeai, Z., Webson, A., Raff, E., and Raffel, C. 2023a. Crosslingual generalization through multitask finetuning.
- Muennighoff, N., Wang, T., Sutawika, L., Roberts, A., Biderman, S., Le Scao, T., Bari, M. S., Shen, S., Yong, Z.-X., Schoelkopf, H., Tang, X., Radev, D., Aji, A. F., Almubarak, K., Albanie, S., Alyafeai, Z., Webson, A., Raff, E., and Raffel, C. 2023b. Crosslingual generalization through multitask finetuning.
- Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. 2022. Training language models to follow instructions with human feedback.
- Radford, A., Narasimhan, K., Salimans, T., and Sutskever, I. 2018. Improving language understanding by generative pre-training.
- Sakaguchi, K., Le Bras, R., Bhagavatula, C., and Choi, Y. 2021. Winogrande: An adversarial Winograd schema challenge at scale.
- Sanh, V., Webson, A., Raffel, C., Bach, S. H., Sutawika, L., Alyafeai, Z., Chaffin, A., Stiegler, A., Le Scao, T., Raja, A., et al. 2022. Multitask prompted training enables zero-shot task generalization.
- Sengupta, N., Sahu, S. K., Jia, B., Katipomu, S., Li, H., Koto, F., Marshall, W., Gosal, G., Liu, C., Chen, Z., Afzal, O. M., Kamboj, S., Pandit, O., Pal, R., Pradhan, L., Mujahid, Z. M., Baali, M., Han, X., Bsharat, S. M., Aji, A. F., Shen, Z., Liu, Z., Vassilieva, N., Hestness, J., Hock, A., Feldman, A., Lee, J., Jackson, A., Ren, H. X., Nakov, P., Baldwin, T., and Xing, E. 2023. Jais and Jais-chat: Arabic-centric foundation and instruction-tuned open generative large language models.
- Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin, C., Liang, P., and Hashimoto, T. B. 2023. Stanford Alpaca: An instruction-following LLaMA model.
- Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. 2023. LLaMA 2: Open foundation and fine-tuned chat models.
- Wang, Y., Ivison, H., Dasigi, P., Hessel, J., Khot, T., Chandu, K. R., Wadden, D., MacMillan, K., Smith, N. A., Beltagy, I., and Hajishirzi, H. 2023a. How far can camels go? Exploring the state of instruction tuning on open resources.
- Wang, Y., Kordi, Y., Mishra, S., Liu, A., Smith, N. A., Khashabi, D., and Hajishirzi, H. 2023b. Self-Instruct: Aligning language models with self-generated instructions.
- Wang, Y., Mishra, S., Alipoormolabashi, P., Kordi, Y., Mirzaei, A., Naik, A., Ashok, A., Dhanasekaran, A. S., Arunkumar, A., Stap, D., Pathak, E., Karamanolakis, G., Lai, H., Purohit, I., Mondal, I., Anderson, J., Kuznia, K., Doshi, K., Pal, K. K., Patel, M., Moradshahi, M., Parmar, M., Purohit, M., Varshney, N., Kaza, P. R., Verma, P., Puri, R. S., Karia, R., Doshi, S., Sampat, S. K., Mishra, S., Reddy, S. A., Patro, S., Dixit, T., and Shen, X. 2022. Super-NaturalInstructions: Generalization via declarative instructions on 1600+ NLP tasks.
- Wu, M., Waheed, A., Zhang, C., Abdul-Mageed, M., and Aji, A. F. 2023. LaMini-LM: A diverse herd of distilled models from large-scale instructions.
- Xu, C., Sun, Q., Zheng, K., Geng, X., Zhao, P., Feng, J., Tao, C., and Jiang, D. 2023. WizardLM: Empowering large language models to follow complex instructions.
- Ye, S., Kim, D., Kim, S., Hwang, H., Kim, S., Jo, Y., Thorne, J., Kim, J., and Seo, M. 2023. FLASK: Fine-grained language model evaluation based on alignment skill sets.
- Yue, X., Qu, X., Zhang, G., Fu, Y., Huang, W., Sun, H., Su, Y., and Chen, W. 2023. MAmmoTH: Building math generalist models through hybrid instruction tuning.
- Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., and Choi, Y. 2019. HellaSwag: Can a machine really finish your sentence?

## Appendix A. Examples of Instruction Types

Appendix A는 세 instruction 데이터가 실제로 어떻게 다른지를 매우 직관적으로 보여준다. 같은 instruction tuning이라고 해도 데이터의 표면 형식과 응답 방식이 크게 다르며, 이 차이가 후반부 결과 해석의 핵심이 된다.

![Table 3. Alpaca, CodeAlpaca, P3의 예시.](/assets/images/for%20Fine-tuning%20Large%20Language%20Models/table_03_instruction_type_examples.png)

*Table 3. Alpaca, CodeAlpaca, P3에서 각각 하나씩 뽑은 예시.*

Alpaca 예시는 “왜 코딩을 배워야 하는가”처럼 열린 질문에 대해 여러 이유를 자연어로 길게 나열한다. 이는 일반 대화형 assistant가 사람의 질문에 설명식으로 답하는 형식과 가깝다. CodeAlpaca 예시는 상태 수를 세는 함수를 작성하라는 요청과 함께 입력이 제시되고, 응답은 바로 파이썬 함수 코드로 이어진다. P3 예시는 “Anna Kournikova, Michelangelo, ILOVEYOU, Melissa, Stuxnet은 무엇의 예인가”처럼 정답이 명확한 질문에 짧은 정답을 내놓는다. 즉 세 데이터의 차이는 단지 주제 차이가 아니라, 요구되는 출력 구조와 정답 형식의 차이이기도 하다.

## Appendix B. Alignment Skills Demonstration

FLASK는 각 instruction이 답변되기 위해 필요한 skill을 세 개씩 붙여 두는 방식으로 동작한다. 이 논문은 그중 가장 자주 등장하는 8개 skill만 남기고, 다른 skill이 붙은 instruction은 제외해 총 1,180개 평가 샘플을 구성한다. 아래의 정의들은 저자들이 실제 평가에 사용한 skill의 의미를 정리한 것이다.

### Logical Correctness

응답의 최종 답이 논리적으로 정확한지, 그리고 정답이 결정적인 instruction에 대해 올바른 결과를 냈는지를 본다. 사실을 알고 있느냐와는 별개로, 추론 구조와 최종 결론이 맞는지가 핵심이다.

### Factuality

배경지식이 필요한 instruction에서 응답이 관련 사실을 정확하게 끌어왔는지, 허위 정보가 없는지, 그리고 가능하다면 근거가 믿을 만한지를 본다. 잘 정리된 답변이라도 사실이 틀리면 낮게 평가된다.

### Commonsense Understanding

상식이나 공간 감각, 세계 지식에 기반한 이해가 필요한 요청을 제대로 해석하는지를 평가한다. 단순 정보 회상이 아니라 현실 개념을 자연스럽게 적용하는 능력에 가깝다.

### Comprehension

instruction이 요구하는 바를 실제로 충족했는지를 본다. 특히 여러 조건이 동시에 달린 복합 instruction에서, 명시적 요구뿐 아니라 암묵적 목적까지 읽어냈는지가 중요하다.

### Completeness

설명이 충분히 완전한지, 다뤄야 할 항목을 빠뜨리지 않았는지, 그리고 각 항목의 깊이가 적절한지를 본다. 짧고 예쁜 답변보다 필요한 내용을 다 담았는지가 더 중요하다.

### Insightfulness

응답이 단순 반복이 아니라 창의적이거나 새로운 시각을 주는지 평가한다. 기존 정보를 재배열하더라도 새로운 해석이나 관점을 만들면 높은 점수를 받을 수 있다.

### Readability

문장이 잘 조직되어 있고 읽기 쉬운지, 응답 전체가 논리적 흐름을 갖추는지를 본다. 구조화와 coherence가 핵심이다.

### Conciseness

불필요하게 장황하지 않고, 독자에게 필요한 정보만 효율적으로 전달하는지를 평가한다. 짧기만 한 것이 아니라 군더더기 없이 핵심을 전달해야 한다.

저자들은 FLASK repository에 있는 prompt를 바탕으로 GPT-4 평가를 수행한다. 평가 prompt는 한 번에 세 개 skill에 대한 rubric을 주고, 각 항목에 대해 먼저 코멘트를 쓰고 그다음 점수를 1에서 5 사이로 매기도록 요구한다. 또한 skill 간 점수는 서로 직교적이어야 하며, 한 항목의 판단 기준이 다른 항목에 끌려가지 않도록 설계되어 있다.

![Figure 3. alignment skill assessment prompt.](/assets/images/for%20Fine-tuning%20Large%20Language%20Models/figure_03_alignment_skill_assessment_prompt.png)

*Figure 3. FLASK에서 가져온 alignment skill assessment prompt. 파란색으로 표시된 부분은 실제 평가할 instruction과 정답, 모델 응답에 따라 채워지는 자리다.*

Figure 3를 보면 평가 prompt는 단순 점수 채점표가 아니라 비교적 엄격한 rubric 기반 평가 템플릿이라는 점을 알 수 있다. 시스템 메시지 안에 각 skill의 정의와 scoring principle이 들어가고, instruction, ground truth answer, assistant response를 채운 뒤 GPT-4가 세부 코멘트와 숫자 점수를 함께 반환한다. 마지막에는 skill 이름을 key로 하는 Python dictionary 객체를 반환하도록 요구해 후처리도 쉽게 만든다.

## Appendix C. Resources

자원 사용량도 별도로 공개한다. LLaMA-2-7B 학습에는 4×A100, LLaMA-2-13B 학습에는 8×A100을 사용한다. 20K 데이터 기준으로 한 번의 학습에는 약 2시간이 걸리고, 전체 실험은 총 288 A100 GPU hours 정도가 든다. 평가 단계에서는 GPT-4를 사용하며, 입력 길이는 평균 950 token, 출력 길이는 평균 293 token이다. 전체 평가 비용은 약 760달러로 보고된다.

## Additional Explanation

### 1. 이 논문이 정말 묻는 질문

겉으로 보면 이 논문은 instruction dataset 세 개를 섞어 본 단순한 ablation처럼 보일 수 있다. 하지만 실제 질문은 훨씬 더 근본적이다. 모델이 어떤 행동을 하길 원하는가에 따라, 학습 데이터의 성격을 어떻게 조합해야 하는가를 묻고 있다. 즉 instruction tuning을 하나의 만능 recipe가 아니라 capability composition 문제로 본다.

### 2. 왜 P3는 benchmark에는 좋고 chat에는 나쁠 수 있는가

P3는 원래 supervised NLP task를 자연어 프롬프트 형식으로 바꾼 데이터다. 그래서 정답이 분명하고, 출력 형식도 짧고 직접적이며, 문제 해결 중심이다. 이런 특성은 ARC나 Winogrande 같은 benchmark에서 큰 강점을 만든다. 반면 사람과 상호작용하는 assistant는 정답만 맞히는 것보다도 설명의 흐름, 응답의 자연스러움, 요구 조건 해석, 정중한 표현 같은 요소가 중요하다. 논문이 관찰한 alignment 저하는 바로 이 형식 차이에서 비롯된다고 이해할 수 있다.

### 3. 왜 code instruction은 alignment에도 도움을 줄 수 있는가

CodeAlpaca는 표면적으로는 코드 데이터지만, 실제로는 조건을 정확히 해석하고, 요구된 출력 형식을 엄격히 따르며, 중간 설명 없이도 목표를 정확히 달성하는 응답 습관을 학습시킨다. 그래서 Alpaca와 결합했을 때 comprehension, completeness, conciseness 같은 항목이 좋아진다. 논문은 이를 과장해서 일반화하지는 않지만, coding data가 단지 코딩만 가르치는 것은 아니라는 중요한 힌트를 준다.

### 4. 왜 큰 모델이 다양한 mixture를 더 잘 받는가

13B가 7B보다 더 풍부한 mixture에서 좋은 결과를 내는 것은 모델 용량이 커질수록 서로 다른 instruction 형식이 한 모델 안에서 충돌하기보다 분리될 여지가 커지기 때문이다. 작은 모델은 다양한 데이터가 들어오면 표현 공간이 쉽게 혼잡해질 수 있지만, 큰 모델은 서로 다른 요구를 더 잘 분산해서 흡수한다. 이 때문에 동일한 mixture라도 모델 크기에 따라 최적 조합이 달라진다.

### 5. 실무적으로 읽으면 어떤 recipe가 보이는가

이 논문을 실무 관점에서 읽으면 메시지는 의외로 분명하다. 대화형 assistant가 목표라면 Alpaca 같은 general instruction이 기본축이 되어야 하고, code ability도 필요하다면 CodeAlpaca를 함께 넣는 것이 유리하다. 반대로 NLP benchmark 점수를 극대화하려면 P3를 넣는 것이 좋지만, 그 대가로 chat alignment가 흔들릴 수 있다는 점을 감수해야 한다. 즉 데이터 혼합은 무조건 많이 넣는 문제가 아니라, 어떤 사용자 경험을 만들 것인지에 맞춰 설계해야 하는 문제다.
