window.siteData = {
  articles: {
    owner: 'dremdeveloper',
    repo: 'dremdeveloper.github.io',
    branch: 'main',
    articlesPath: 'articles',
    categoryOrder: ['AI 논문', '생각', '지식 정리', '트러블 슈팅'],
    pinnedFirstFiles: [
      'articles/AI 논문/AI_논문_용어집.md'
    ],
    defaultFile: 'articles/AI 논문/attention_is_all_you_need.md',
    ignoredFiles: [
      'articles/AI 논문/sample.md'
    ],
    files: [
      {
        name: 'articles/AI 논문/AI_논문_용어집.md',
        category: 'AI 논문',
        title: '용어집'
      },
      {
        name: 'articles/AI 논문/attention_is_all_you_need.md',
        category: 'AI 논문',
        title: 'attention is all you need'
      },
      {
        name: 'articles/AI 논문/Direct Preference Optimization.md',
        category: 'AI 논문',
        title: 'Direct Preference Optimization'
      },
      {
        name: 'articles/AI 논문/Proximal_Policy_Optimization_Algorithms.md',
        category: 'AI 논문',
        title: 'Proximal Policy Optimization Algorithms'
      },
      {
        name: 'articles/AI 논문/Rate_or_Fate_RLVeR.md',
        category: 'AI 논문',
        title: 'Rate or Fate RLVeR'
      },
      {
        name: 'articles/AI 논문/Training language models to follow instructions with human feedback.md',
        category: 'AI 논문',
        title: 'Training language models to follow instructions with human feedback'
      },
      {
        name: 'articles/AI 논문/mixture_of_experts_in_large_language_models.md',
        category: 'AI 논문',
        title: 'mixture of experts in large language models'
      },
      {
        name: 'articles/생각정리/2026-03-21-sample-article.md',
        category: '생각',
        title: 'sample article'
      },
      {
        name: 'articles/유용한 지식 및 팁/markdown_guide_ko.md',
        category: '지식 정리',
        title: 'Markdown Guide KO'
      },
      {
        name: 'articles/유용한 지식 및 팁/유용한_지식_용어집.md',
        category: '지식 정리',
        title: '용어집'
      },
      {
        name: 'articles/트러블 슈팅/2026-03-21-temp-troubleshooting-note.md',
        category: '트러블 슈팅',
        title: 'temp troubleshooting note'
      },
      {
        name: 'articles/instruction_tuning_survey_section6_mathjax.md',
        category: '미분류',
        title: 'instruction tuning survey section6 mathjax'
      }
    ]
  },
  lectures: {
    python: {
      key: 'python',
      label: '코딩 테스트 합격자 되기 - 파이썬편',
      playlistId: 'PLrfS7Czu1oUd_6CqnTrznFZgg3M9aVTVK',
      bookUrl: 'https://product.kyobobook.co.kr/detail/S000210881884',
      totalLabel: '33 videos',
      items: [
        { value: 'hXDygHLvwPQ', title: '알고리즘의 효율 분석', summary: '알고리즘은 결국 문제를 푸는 순서입니다. 이 강의에서는 같은 문제라도 풀이를 어떻게 짜느냐에 따라 걸리는 시간이 크게 달라진다는 점을 먼저 짚어 줍니다. 그래서 입력 크기와 제한 시간을 보고 허용될 만한 시간 복잡도를 먼저 가늠한 뒤, 그에 맞는 풀이를 고르는 습관으로 연결합니다.' },
        { value: 'Ybndhuvbyf0', title: '스택', summary: '스택은 가장 나중에 넣은 값을 가장 먼저 꺼내는 구조입니다. 괄호 짝 맞추기나 실행 취소처럼 방금 처리한 상태를 다시 꺼내야 할 때 왜 스택이 잘 맞는지 예시와 함께 설명합니다. 문제를 읽다가 순서가 뒤집히거나 최근 상태를 기억해야 한다면 스택부터 떠올리면 된다는 감각을 잡아 줍니다.' },
        { value: 'Le6JaXi5whM', title: '큐', summary: '큐는 먼저 들어온 값이 먼저 나가는 자료구조입니다. 줄 서서 처리하는 상황이나 BFS처럼 가까운 것부터 차례대로 확장해야 하는 상황에서 큐가 왜 자연스럽게 쓰이는지 차근차근 풀어 줍니다. 덕분에 문제 속에 대기열, 차례 처리, 레벨 순회가 보이면 바로 큐를 연결해 생각할 수 있게 됩니다.' },
        { value: 'DReB0IKu550', title: '해시(개념)', summary: '해시는 값을 빨리 찾고 싶을 때 가장 먼저 떠올릴 만한 도구입니다. 이 강의에서는 키를 특정 위치에 매핑해 두면 존재 여부 확인이나 빈도 계산을 아주 빠르게 할 수 있다는 점을 쉽게 설명합니다. 그래서 코딩 테스트에서 찾기와 집계가 반복되면 왜 해시를 우선 후보로 봐야 하는지 자연스럽게 이해하게 됩니다.' },
        { value: '5gbsCLXQI48', title: '해시(해시함수및충돌처리)', summary: '해시를 쓸 때는 키를 어디에 저장할지 정해 주는 해시 함수가 필요합니다. 그런데 서로 다른 키가 같은 위치로 몰릴 수 있어서 충돌 처리를 함께 생각해야 하고, 여기서 체이닝이나 개방 주소법 같은 방법이 등장합니다. 결국 해시는 빠르기만 한 구조가 아니라, 충돌을 어떻게 다루느냐까지 봐야 제대로 쓸 수 있다는 점을 짚어 줍니다.' },
        { value: 'imXnLratBCE', title: '트리(개념/배열로 구축하기)', summary: '트리는 부모와 자식으로 이어지는 계층 구조를 표현할 때 쓰입니다. 루트, 리프, 서브트리 같은 기본 용어를 먼저 정리한 뒤, 완전 이진트리처럼 규칙이 분명한 경우에는 배열 인덱스로도 충분히 표현할 수 있음을 보여 줍니다. 그림으로만 보던 트리를 실제 코드 속 구조로 옮기는 감을 여기서 잡게 됩니다.' },
        { value: 'j4szC6JQl1g', title: '트리(포인터와인접리스트로 구축하기/순회)', summary: '배열만으로는 표현하기 어려운 트리는 노드가 자식을 가리키는 방식이나 인접 리스트 방식으로 더 유연하게 다룰 수 있습니다. 이 강의에서는 그런 표현 방법을 비교해 보고, 전위·중위·후위 순회가 실제 코드에서 어떻게 돌아가는지도 함께 짚습니다. 트리를 저장하는 방식과 방문하는 방식이 따로 노는 것이 아니라 한 흐름으로 이어진다는 점이 자연스럽게 정리됩니다.' },
        { value: 'mqrE9ZcOBhw', title: '트리(이진탐색트리및최종정리)', summary: '이진 탐색 트리는 왼쪽에는 더 작은 값, 오른쪽에는 더 큰 값이 온다는 규칙이 핵심입니다. 그래서 원하는 값을 찾을 때 매번 절반씩 범위를 좁혀 가는 느낌으로 탐색할 수 있고, 삽입 위치도 비교를 따라가며 정할 수 있습니다. 다만 트리가 한쪽으로 치우치면 장점이 줄어든다는 점까지 함께 짚어 주면서 일반 트리와 BST를 구분해 보게 합니다.' },
        { value: '4Ttl35GIIuw', title: '집합(개념및표현방법)', summary: '서로소 집합은 겹치지 않는 여러 그룹을 관리할 때 쓰는 구조입니다. 이 강의에서는 각 원소가 부모를 가리키도록 두고, 결국 같은 대표를 따라가면 같은 집합이라고 판단하는 방식을 차근차근 설명합니다. 연결 여부를 빠르게 묻는 문제에서 왜 이 구조가 강력한지 감을 잡기 좋은 출발점입니다.' },
        { value: 'tweOtNqfvHI', title: '집합(파인드연산)', summary: 'find 연산은 어떤 원소가 어느 집합에 속해 있는지 대표 원소를 찾아 확인하는 과정입니다. 부모를 따라 루트까지 올라가면 같은 그룹인지 바로 판단할 수 있고, 이후 union도 이 결과를 바탕으로 움직입니다. 겉보기엔 단순한 탐색 같지만 서로소 집합 전체를 떠받치는 핵심 연산이라는 점을 분명하게 보여 줍니다.' },
        { value: 'gRNpxHw1KLg', title: '집합(경로압축)', summary: '경로 압축은 find를 한 번 수행할 때마다 구조를 더 납작하게 만드는 기법입니다. 지나가는 노드들의 부모를 대표 원소로 바로 연결해 두면, 다음 번에는 훨씬 짧은 경로로 루트를 찾을 수 있습니다. 구현은 짧지만 성능 차이는 꽤 크기 때문에 실전에서는 사실상 기본처럼 붙는 최적화라는 점을 알려 줍니다.' },
        { value: 'lvA_Qzll4QI', title: '집합(유니온연산)', summary: 'union은 두 원소가 속한 집합을 하나로 합치는 연산입니다. 먼저 find로 각 집합의 대표를 확인하고, 이미 같은 집합이 아니라면 그때 루트를 연결해 병합하는 순서로 진행됩니다. 연결 관계가 점점 넓어지는 문제를 볼 때 union과 find가 늘 함께 움직인다는 감각을 여기서 익히게 됩니다.' },
        { value: 'nObTu0gqIh4', title: '집합(랭크)', summary: '랭크는 union을 할 때 트리가 쓸데없이 길어지지 않게 잡아 주는 장치입니다. 더 얕은 쪽을 더 깊은 쪽 아래로 붙이면 루트를 찾으러 올라가는 경로가 길어지는 일을 줄일 수 있습니다. 경로 압축과 함께 쓰였을 때 왜 서로소 집합이 거의 상수 시간처럼 빠르게 동작하는지도 이 강의에서 연결해 줍니다.' },
        { value: 'HedY-mtg850', title: '집합(마무리)', summary: '여기서는 서로소 집합 파트를 한 번에 다시 묶어 줍니다. 배열 표현부터 find, 경로 압축, union, rank까지가 각각 따로 있는 기술이 아니라 대표를 빨리 찾고 안정적으로 합치기 위한 한 세트라는 점이 또렷해집니다. 그래서 연결 요소 판별이나 그룹 병합 문제를 만났을 때 어느 부분을 꺼내 써야 할지 정리가 됩니다.' },
        { value: 'fBmIcN_eQAA', title: '그래프(기본 개념)', summary: '그래프는 정점과 간선으로 관계를 표현하는 가장 기본적인 모델입니다. 이 강의에서는 방향이 있는지, 가중치가 있는지 같은 성질을 먼저 구분해야 이후 알고리즘 선택이 쉬워진다는 점을 짚습니다. 사람 관계, 길 찾기, 네트워크 문제를 그래프로 바꿔 보는 시선도 함께 열어 줍니다.' },
        { value: 'NGOeJY1B7qk', title: '그래프(인접행렬과인접리스트)', summary: '그래프를 코드로 옮길 때 가장 많이 쓰는 표현이 인접 행렬과 인접 리스트입니다. 인접 행렬은 연결 여부를 빠르게 확인할 수 있지만 공간을 많이 쓰고, 인접 리스트는 간선이 적은 그래프에서 훨씬 경제적입니다. 정점 수와 간선 수를 보고 어떤 표현이 더 자연스러운지 판단하는 기준을 여기서 잡게 됩니다.' },
        { value: 'sORP6QtUp_s', title: '그래프(DFS와BFS의개념)', summary: 'DFS와 BFS는 둘 다 그래프를 방문하는 방법이지만 움직이는 방식이 꽤 다릅니다. DFS는 한 갈래를 깊게 파고들다가 돌아오고, BFS는 가까운 정점부터 층층이 넓혀 갑니다. 이 차이를 분명하게 이해해 두면 이후 구현과 문제 선택이 훨씬 수월해집니다.' },
        { value: 'MhhORyIWYHk', title: '그래프(스택을활용한DFS)', summary: '스택으로 DFS를 구현하면 깊이 우선 탐색이 자료구조 관점에서 더 선명하게 보입니다. 마지막에 넣은 정점부터 다시 꺼내며 다음 경로를 이어 가기 때문에, 왜 DFS가 깊게 들어갔다가 돌아오는 탐색인지 코드로 바로 느낄 수 있습니다. 방문 처리 시점과 이웃을 넣는 순서가 결과에 어떤 차이를 만드는지도 함께 짚어 줍니다.' },
        { value: 'RaDeHg10NLs', title: '그래프(재귀를활용한DFS)', summary: '재귀 DFS는 함수 호출 스택이 곧 탐색 스택 역할을 한다는 점이 핵심입니다. 현재 정점을 방문한 뒤 아직 가지 않은 이웃으로 재귀 호출을 이어 가고, 더 내려갈 곳이 없으면 자연스럽게 되돌아옵니다. 그래서 코드가 간결해지는 대신 종료 조건과 방문 처리를 정확히 넣어야 한다는 점을 분명히 잡아 줍니다.' },
        { value: 'iIW-7X58Wb4', title: '그래프(큐를활용한BFS)', summary: '큐로 구현한 BFS는 먼저 발견한 정점부터 차례대로 처리하면서 탐색 범위를 넓혀 갑니다. 그래서 출발점에서 가까운 정점이 먼저 방문되고, 가중치가 없는 그래프에서는 최단 거리 문제와도 자연스럽게 연결됩니다. 같은 정점이 중복으로 들어가지 않도록 언제 방문 표시를 해야 하는지도 함께 정리해 줍니다.' },
        { value: 'scGPmqpsSAk', title: '그래프(DFS의구현)', summary: '개념으로 배운 DFS를 실제 문제 풀이 코드로 옮기면 어떤 모습이 되는지 보여 주는 강의입니다. 그래프 표현, 방문 배열, 재귀나 스택을 이용한 순회 흐름이 한 코드 안에서 어떻게 맞물리는지 차분히 따라갑니다. 덕분에 DFS를 아는 수준을 넘어서, 바로 템플릿처럼 꺼내 쓸 수 있는 구현 감각까지 챙기게 됩니다.' },
        { value: '4tV-dz7v7ao', title: '그래프(BFS구현)', summary: '이 강의에서는 BFS를 실제 코드 순서대로 구현해 봅니다. 큐 초기화, 시작 정점 처리, 반복문 안에서의 pop과 인접 정점 확장이 어떤 흐름으로 이어지는지 따라가다 보면 BFS의 뼈대가 한눈에 잡힙니다. 필요하면 거리 정보까지 함께 관리하는 패턴도 자연스럽게 연결돼서 실전 문제 풀이에 바로 써먹기 좋습니다.' },
        { value: 'l6SdqywBpw0', title: '그래프(문제풀이시DFS와BFS선택하기)', summary: '그래프 문제를 풀 때 늘 고민하게 되는 게 DFS를 쓸지 BFS를 쓸지입니다. 이 강의는 최단 거리나 레벨 단위 확장이 중요하면 BFS, 모든 경우를 깊게 살피거나 되돌아오는 탐색이면 DFS가 더 자연스럽다는 식으로 판단 기준을 정리해 줍니다. 문제를 읽는 순간 탐색 방향부터 잡을 수 있게 도와주는 비교 정리입니다.' },
        { value: '7ZhxvLomajw', title: '최단경로(다익스트라의 개념)', summary: '다익스트라는 가중치가 음수가 아닌 그래프에서 최단 거리를 구할 때 가장 먼저 떠올릴 알고리즘입니다. 아직 확정되지 않은 정점 가운데 현재 가장 가까운 곳을 하나씩 골라 거리를 굳혀 가는 흐름이 핵심입니다. 왜 이런 문제에서는 단순 BFS만으로 부족한지, 가중치를 따로 관리해야 하는 이유까지 함께 이해하게 됩니다.' },
        { value: 'Wo6trI4RXlA', title: '최단경로(다익스트라의 동작)', summary: '다익스트라가 실제로 어떻게 움직이는지는 거리 테이블과 간선 완화 과정을 따라가 보면 분명해집니다. 가장 짧은 거리 후보를 가진 정점을 뽑고, 그 정점을 거쳐 가는 편이 더 짧다면 주변 정점의 거리를 바로 갱신합니다. 이름만 외우는 데서 끝나지 않고 손으로 추적하고 코드로 옮길 수 있을 만큼 절차를 잡아 주는 강의입니다.' },
        { value: '3SLsupzyzpM', title: '최단경로(다익스트라의 한계)', summary: '다익스트라가 항상 통하는 것은 아닙니다. 한 번 최단 거리라고 확정한 값이 이후에도 바뀌지 않는다는 전제가 있어야 하는데, 음수 간선이 있으면 이 가정이 무너집니다. 그래서 언제 다익스트라를 쓰면 안 되는지까지 알아 두는 것이 알고리즘 자체를 아는 것만큼 중요하다는 점을 짚어 줍니다.' },
        { value: 'Fql3epqo5k4', title: '최단경로(밸만포드 알고리즘의 개념)', summary: '벨만포드는 음수 간선이 있는 그래프에서도 최단 거리를 구할 수 있게 해 주는 알고리즘입니다. 특정 정점 하나를 확정하는 대신 모든 간선을 여러 번 훑으면서 거리 정보를 조금씩 갱신해 나간다는 발상이 핵심입니다. 다익스트라가 막히는 조건에서 어떤 대안을 써야 하는지 자연스럽게 이어서 배울 수 있습니다.' },
        { value: '6PTCch3nONk', title: '최단경로(밸만포드 알고리즘의 실제 동작)', summary: '벨만포드의 동작은 간선을 한 번 훑고 끝나는 방식이 아니라, 같은 완화 작업을 여러 라운드 반복한다는 점에 특징이 있습니다. 정점 수가 V개라면 최대 V-1번 반복하면 최단 거리 정보가 전체 그래프로 퍼져 나간다는 이유를 예시와 함께 설명합니다. 여러 번의 갱신이 쌓여 결과를 만든다는 흐름을 이해하는 데 초점을 둔 강의입니다.' },
        { value: 'Oil_ol0DXMo', title: '최단경로(밸만포드 알고리즘 - 음의 순환 탐지)', summary: '벨만포드는 최단 거리 계산뿐 아니라 음의 순환이 있는지도 찾아낼 수 있습니다. V-1번 완화가 끝난 뒤에도 어떤 값이 더 줄어든다면, 그 그래프에는 돌수록 비용이 낮아지는 순환이 남아 있다는 뜻입니다. 그래서 이 강의에서는 벨만포드를 단순 계산 도구가 아니라 그래프 상태를 판별하는 방법으로도 보게 됩니다.' },
        { value: 'l993FqfBrHM', title: '최단경로(밸만포드 알고리즘의 한계)', summary: '벨만포드는 적용 범위가 넓은 대신 계산량이 꽤 큽니다. 모든 간선을 여러 번 확인해야 하니 정점과 간선 수가 커질수록 부담이 빠르게 늘어납니다. 그래서 음수 간선을 다뤄야 할 때만 신중하게 선택해야 한다는 현실적인 기준도 함께 잡아 줍니다.' },
        { value: 'Hj3sh12lwoU', title: '최단경로(다익스트라 vs 밸만포드)', summary: '두 알고리즘은 모두 최단 경로를 구하지만 쓰이는 장면은 분명히 다릅니다. 음수 간선이 없고 빠른 처리가 중요하면 다익스트라가 맞고, 음수 간선이나 음의 순환 검사까지 필요하면 벨만포드를 봐야 합니다. 문제 조건을 읽고 둘 중 무엇을 꺼낼지 결정하는 기준을 한 번에 정리해 주는 비교 강의입니다.' },
        { value: 'etRpwAKCvCc', title: '백트래킹(개념)', summary: '백트래킹은 가능한 경우를 모두 보되, 답이 될 수 없는 가지는 일찍 잘라 내는 탐색 방식입니다. 그래서 완전 탐색보다 훨씬 효율적으로 움직일 수 있고, 선택했다가 틀리면 다시 돌아오는 흐름이 자연스럽게 생깁니다. 문제를 결정의 나무로 바라보는 시선을 익히기에 좋은 입문 강의입니다.' },
        { value: 'nDFrhvdlH98', title: '백트래킹(활용)', summary: '이 강의에서는 백트래킹이 실제 문제에서 어떻게 쓰이는지를 상태 선택, 재귀 호출, 복원이라는 순서로 보여 줍니다. 부분합이나 N-Queen 같은 예시를 따라가다 보면 가지치기가 탐색량을 얼마나 줄여 주는지 체감하게 됩니다. 정답 후보를 만들다가 조건이 어긋나면 바로 돌아오고, 바꿔 둔 상태를 다시 원래대로 돌리는 구현 감각까지 함께 익히게 됩니다.' }
      ]
    },
    cpp: {
      key: 'cpp',
      label: '코딩 테스트 합격자 되기 - C++편',
      playlistId: 'PLrfS7Czu1oUd8lKwNkc9rarTuXn_Qw6ie',
      bookUrl: 'https://product.kyobobook.co.kr/detail/S000213087020',
      totalLabel: '13 videos',
      items: [
        { value: 'CLXFgptB81M', title: 'C++ 책 소개', summary: '이 영상은 책 전체를 어떤 순서로 공부하면 좋은지 먼저 안내해 주는 소개 강의입니다. 시간 복잡도부터 C++ 문법, 자료구조, 알고리즘, 문제 풀이까지가 따로 흩어진 주제가 아니라 한 흐름으로 이어진다는 점을 보여 줍니다. 처음 시작하는 사람이라면 어디서부터 어떻게 들어가야 할지 감을 잡는 데 도움이 됩니다.' },
        { value: '_D_LTv-kyF4', title: '효율적으로 공부하기', summary: '코딩 테스트 공부는 많이 푸는 것만으로 잘되지 않는다는 이야기를 이 강의는 꽤 현실적으로 풀어 줍니다. 남의 풀이를 분석하고, 내 언어로 다시 정리하고, 테스트 케이스와 의사코드까지 스스로 만들어 보는 복습 습관이 왜 중요한지 구체적으로 짚습니다. 결국 실력을 올리는 공부는 문제 수보다 복기와 설계의 밀도에 달려 있다는 메시지로 이어집니다.' },
        { value: '6oDZgi_7ao0', title: '03장 시간 복잡도', summary: '시간 복잡도는 알고리즘을 감으로 고르지 않게 해 주는 가장 기본적인 기준입니다. 이 강의에서는 연산 횟수로 성능을 바라보는 법과 빅오 표기법을 차근차근 정리한 뒤, 입력 크기와 제한 시간을 어떻게 연결해서 읽어야 하는지 설명합니다. 그래서 문제를 받자마자 어느 정도 수준의 풀이가 가능한지 먼저 가늠하는 습관을 들이게 됩니다.' },
        { value: 'xc0HZiqh8Fs', title: '04/05장 반드시 알아야 할 C++ 문법', summary: '코딩 테스트용 C++ 문법을 한꺼번에 정리해 주는 강의입니다. 기본 자료형, 배열, 문자열부터 STL 컨테이너와 반복자, 자주 쓰는 알고리즘까지 실전에서 바로 꺼내 쓰는 것들 위주로 묶어 줍니다. 문법 전체를 넓게 훑기보다는 문제 풀이에 필요한 도구를 빠르게 손에 익히는 데 초점을 둔 구성입니다.' },
        { value: '-TGCT74wFeg', title: '06/07장 스택과 큐', summary: '스택과 큐는 이름보다 처리 순서를 이해하는 게 더 중요합니다. 하나는 나중에 넣은 것이 먼저 나오고, 다른 하나는 먼저 들어온 것이 먼저 나간다는 차이를 예시와 함께 풀어 줍니다. 그래서 괄호, 실행 취소, 대기열, BFS 같은 문제를 볼 때 어떤 자료구조가 자연스러운지 바로 연결해 볼 수 있습니다.' },
        { value: 'KsfmSyIYX3g', title: '08장 해시', summary: '해시는 빠르게 찾고 세고 대응시켜야 할 때 정말 자주 등장합니다. 이 강의에서는 해시 함수가 어떤 역할을 하는지, 충돌은 왜 생기고 어떻게 다뤄야 하는지까지 함께 정리해 줍니다. 덕분에 존재 여부 확인이나 빈도 집계 문제를 만났을 때 해시를 왜 먼저 떠올려야 하는지 분명해집니다.' },
        { value: 'CXiVJgUjL6o', title: '09장 트리', summary: '트리는 계층 구조를 표현할 때 가장 자주 쓰는 자료구조입니다. 이 강의에서는 배열 기반 표현과 인접 리스트 기반 표현을 나눠 보고, 전위 순회 같은 기본 방문 방식까지 자연스럽게 연결합니다. 나중에 이진 탐색 트리로 넘어가기 전에 트리를 저장하고 따라가는 감을 먼저 단단히 잡아 주는 구성입니다.' },
        { value: 'pQ4fcGEG-PY', title: '10장 집합', summary: '서로소 집합은 연결된 그룹을 빠르게 관리해야 할 때 빛을 발합니다. 부모 배열로 집합을 표현하고, find와 union, 경로 압축, rank 최적화가 어떻게 맞물리는지 차례대로 설명해 줍니다. 개념만 배우고 끝나는 것이 아니라 실제 구현과 활용 장면까지 이어져서 실전 감각을 잡기 좋습니다.' },
        { value: 'OmYQsxreXNo', title: '11장 그래프', summary: '그래프 파트를 넓게 훑어 보는 입문 강의에 가깝습니다. 인접 행렬과 인접 리스트, DFS와 BFS, 그리고 최단 경로 문제까지 이어 보면서 그래프에서 자주 만나는 주제들이 어떻게 연결되는지 큰 그림을 잡아 줍니다. 표현 방법과 탐색 방식, 거리 계산이 각각 따로 떨어진 개념이 아니라는 점이 자연스럽게 정리됩니다.' },
        { value: 'VvcEx75Bgvk', title: '12장 백트래킹', summary: '백트래킹은 무작정 다 보는 탐색이 아니라, 필요 없는 가지를 빨리 잘라 내는 탐색입니다. 이 강의에서는 선택하고, 가능성을 검사하고, 더 들어가고, 아니면 되돌아오는 흐름을 구조적으로 설명합니다. 부분합이나 N-Queen 같은 예시와 함께 보면서 가지치기 감각과 상태 복원 감각을 같이 익히게 됩니다.' },
        { value: 'j4ExZH1r3jE', title: '13장 정렬', summary: '정렬은 결과를 예쁘게 나열하는 데만 쓰이는 것이 아니라 많은 문제의 전처리 도구로도 중요합니다. 삽입 정렬, 병합 정렬, 힙 정렬의 핵심 아이디어를 비교하고, 우선순위 큐가 정렬과 어떻게 이어지는지도 함께 보여 줍니다. 그래서 어떤 정렬을 외우기보다 상황에 맞는 도구를 고르는 기준을 세우게 됩니다.' },
        { value: '_K2WH4VhdkQ', title: '15장 동적 계획법', summary: '동적 계획법은 이전에 구한 답을 다시 쓸 수 있을 때 비로소 힘을 발휘합니다. 이 강의에서는 피보나치처럼 익숙한 예제로 출발해 계단 오르기와 LIS 문제까지 확장하면서 상태와 점화식을 세우는 과정을 차근차근 보여 줍니다. DP를 단순 암기 주제가 아니라 문제를 잘게 나누고 저장해 푸는 사고법으로 이해하게 됩니다.' },
        { value: 'k3ctQPulBW8', title: '16장 그리디', summary: '그리디는 지금 가장 좋아 보이는 선택을 해도 전체 답이 깨지지 않을 때만 통합니다. 거스름돈이나 최소 신장 트리 같은 예시를 통해 탐욕적 선택이 언제 안전한지, 그 판단 근거가 무엇인지 차분히 짚어 줍니다. 요령처럼 외우기보다 문제 구조를 보고 그리디가 되는지부터 먼저 판단하게 만드는 마무리 강의입니다.' }
      ]
    }
  },
  studyPlans: [
    {
      key: 'learning-plan',
      label: '학습 플랜',
      typeLabel: 'Embedded HTML',
      title: '파이썬 학습 플랜 문서 모음',
      summary: '코테자료 메뉴 안에서 파이썬 학습 플랜 파일을 왼쪽 메뉴로 고르고, 오른쪽에서 바로 열어볼 수 있도록 구성한 섹션입니다.',
      files: [
        {
          id: 'python-weekly-docs',
          label: '파이썬 학습 플랜',
          description: '1주차부터 4주차까지 이어지는 파이썬 코딩 테스트 학습 플랜 문서를 주차별 과제 목록으로 볼 수 있습니다.',
          path: 'python-weekly-docs.html?embed=1',
          metaLabel: '학습 플랜 · HTML 문서',
          lessons: [
            {
              group: '1주차',
              items: [
                { id: 'w1-a01', label: '01. 코테 기본기 세팅 문서' },
                { id: 'w1-a02', label: '02. 배열 패턴 노트' },
                { id: 'w1-a03', label: '03. 스택 템플릿 과제' },
                { id: 'w1-a04', label: '04. 큐 덱 선택 기준 과제' },
                { id: 'w1-a05', label: '05. 해시 문자열 해싱 과제' }
              ]
            },
            {
              group: '2주차',
              items: [
                { id: 'w2-a06', label: '06. 트리 재귀 패턴 과제' },
                { id: 'w2-a07', label: '07. 유니온-파인드 MST 과제' },
                { id: 'w2-a08', label: '08. 그래프 템플릿 과제' },
                { id: 'w2-a09', label: '09. 힙 이분탐색 보충 과제' }
              ]
            },
            {
              group: '3주차',
              items: [
                { id: 'w3-a10', label: '10. 백트래킹 완전탐색 과제' },
                { id: 'w3-a11', label: '11. 정렬 커스텀 정렬 과제' },
                { id: 'w3-a12', label: '12. 시뮬레이션 구현 과제' },
                { id: 'w3-a13', label: '13. DP 점화식 노트 과제' }
              ]
            },
            {
              group: '4주차',
              items: [
                { id: 'w4-a14', label: '14. DP 심화 + 그리디 판단 과제' },
                { id: 'w4-a15', label: '15. 최종 모의고사 과제' }
              ]
            }
          ]
        }
      ]
    }
  ],
  studyMaterials: [
    {
      key: 'python-flow-visualizer',
      label: '시각화',
      typeLabel: 'Interactive HTML',
      title: '파이썬 알고리즘 실행 흐름 시각화',
      summary: '강의 페이지의 기존 서브메뉴는 유지한 채, 코테자료에서만 정렬, 그래프 탐색, 트리 탐색, 트리 순회, 재귀, 스택, 큐, 힙 정렬, N-Queen을 단계별로 볼 수 있게 구성한 자료입니다.',
      detailA: '형식',
      valueA: 'HTML / CSS / JavaScript',
      detailB: '주제',
      valueB: '정렬 · 탐색 · 순회 · 재귀 · 힙',
      path: 'projects/python-flow-visualizer.html',
      groups: [
        {
          key: 'sorting',
          label: '정렬',
          description: '대표 정렬 알고리즘의 비교, 교환, 분할 과정을 단계별로 따라갑니다.',
          items: [
            { id: 'bubble-sort', label: '버블 정렬' },
            { id: 'insertion-sort', label: '삽입 정렬' },
            { id: 'merge-sort', label: '머지 정렬' },
            { id: 'quick-sort', label: '퀵 정렬' },
            { id: 'counting-sort', label: '계수 정렬' },
            { id: 'heap-sort', label: '힙 정렬' }
          ]
        },
        {
          key: 'structures',
          label: '스택·큐',
          description: '자료구조 내부 상태가 push/pop, enqueue/dequeue에 따라 어떻게 바뀌는지 확인합니다.',
          items: [
            { id: 'stack-basic', label: '스택' },
            { id: 'queue-basic', label: '큐' }
          ]
        },
        {
          key: 'graph',
          label: '그래프 탐색',
          description: '탐색 순서와 방문 상태가 그래프 위에서 어떻게 확장되는지 살펴봅니다.',
          items: [
            { id: 'dfs', label: '깊이 우선 탐색' },
            { id: 'bfs', label: '너비 우선 탐색' }
          ]
        },
        {
          key: 'tree-search',
          label: '트리 탐색',
          description: 'BST에서 탐색 경로가 조건문과 함께 어떻게 결정되는지 보여줍니다.',
          items: [
            { id: 'bst-search', label: 'BST 탐색' }
          ]
        },
        {
          key: 'tree-traversal',
          label: '트리 순회',
          description: '전위·중위·후위 순회가 재귀 호출 스택과 함께 어떻게 이동하는지 설명합니다.',
          items: [
            { id: 'preorder', label: '전위 순회' },
            { id: 'inorder', label: '중위 순회' },
            { id: 'postorder', label: '후위 순회' }
          ]
        },
        {
          key: 'dynamic',
          label: '동적 계획법·백트래킹',
          description: '가지치기와 상태 전이를 시각적으로 비교하며 문제 해결 흐름을 따라갑니다.',
          items: [
            { id: 'n-queen', label: 'N-Queen' },
            { id: 'subset-sum-pruning', label: '부분집합 합 가지치기' }
          ]
        },
        {
          key: 'recursion',
          label: '재귀',
          description: '재귀 호출의 전개와 복귀 시점이 코드 흐름에 맞춰 어떻게 변하는지 확인합니다.',
          items: [
            { id: 'factorial', label: '팩토리얼' },
            { id: 'fibonacci', label: '피보나치' },
            { id: 'combination', label: '조합' }
          ]
        }
      ]
    }
  ]
};
