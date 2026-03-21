window.siteData = {
  articles: {
    owner: 'dremdeveloper',
    repo: 'dremdeveloper.github.io',
    branch: 'main',
    articlesPath: 'articles',
    categoryOrder: ['AI 논문', '트러블 슈팅', '코딩 테스트 팁', '생각정리'],
    defaultFile: 'articles/생각정리/2026-03-21-sample-article.md',
    files: [
      {
        name: 'articles/생각정리/2026-03-21-sample-article.md',
        title: 'sample article'
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
        { value: 'hXDygHLvwPQ', title: '알고리즘의 효율 분석', summary: '알고리즘이 무엇인지부터 시작해, 알고리즘의 성능을 어떻게 분석하는지 설명합니다. 시간 복잡도를 연산 횟수 기준으로 해석하고, 빅오 표기법을 실제 코딩 테스트에서 어떻게 활용할지까지 연결합니다.' },
        { value: 'Ybndhuvbyf0', title: '스택', summary: '스택의 개념과 동작 원리, ADT 관점을 먼저 정리한 뒤 실제 코딩 테스트에서 스택이 필요한 상황을 설명합니다. push/pop 같은 기본 연산을 이해하는 데서 끝나지 않고, 어떤 문제에서 스택을 떠올려야 하는지도 짚습니다.' },
        { value: 'Le6JaXi5whM', title: '큐', summary: '큐의 FIFO 구조와 핵심 연산을 중심으로 정리하는 강의입니다. 앞뒤가 분리된 자료구조라는 점을 바탕으로, 코딩 테스트에서 큐가 필요한 대표 상황과 사고 방식을 설명합니다.' },
        { value: 'DReB0IKu550', title: '해시(개념)', summary: '해시의 기본 개념을 소개하는 강의입니다. 해시의 정의와 핵심 용어를 먼저 정리하고, 해시를 사용했을 때 얻을 수 있는 장점과 문제 해결 속도 측면의 이점을 설명합니다.' },
        { value: '5gbsCLXQI48', title: '해시(해시함수및충돌처리)', summary: '해시 함수가 키를 어떤 방식으로 주소에 대응시키는지 설명하고, 충돌이 왜 발생하는지 다룹니다. 충돌 처리라는 관점에서 해시를 실제 자료구조로 사용할 때 무엇을 고민해야 하는지 정리합니다.' },
        { value: 'imXnLratBCE', title: '트리(개념/배열로 구축하기)', summary: '트리의 개념과 기본 용어를 먼저 잡아 준 뒤, 배열로 트리를 표현하는 방법을 설명합니다. 트리 구조를 머릿속 그림이 아니라 실제 코드 표현으로 옮기는 첫 단계에 초점을 맞춥니다.' },
        { value: 'j4szC6JQl1g', title: '트리(포인터와인접리스트로 구축하기/순회)', summary: '배열 방식 외에 포인터와 인접 리스트로 트리를 표현하는 방법을 보여 줍니다. 이어서 트리 순회의 기본 흐름을 함께 다루며, 트리 구조와 순회가 코드에서 어떻게 연결되는지 설명합니다.' },
        { value: 'mqrE9ZcOBhw', title: '트리(이진탐색트리및최종정리)', summary: '이진 탐색 트리의 핵심 성질을 설명하고, 탐색 기준이 어떻게 만들어지는지 정리합니다. 앞선 트리 내용을 다시 묶어 코딩 테스트 관점에서 트리 파트를 마무리하는 강의입니다.' },
        { value: '4Ttl35GIIuw', title: '집합(개념및표현방법)', summary: '서로소 집합의 개념과, 집합을 배열 등으로 표현하는 기본 방법을 설명합니다. 어떤 문제를 집합 구조로 바라봐야 하는지 감을 잡게 해 주는 도입 강의입니다.' },
        { value: 'tweOtNqfvHI', title: '집합(파인드연산)', summary: '서로소 집합의 핵심 연산인 find가 어떤 식으로 대표 원소를 찾는지 설명합니다. parent를 따라 올라가 루트를 찾는 기본 흐름을 짧고 명확하게 보여 줍니다.' },
        { value: 'gRNpxHw1KLg', title: '집합(경로압축)', summary: 'find 연산에 경로 압축을 적용해 탐색 경로를 짧게 만드는 아이디어를 설명합니다. 왜 이 기법이 성능을 크게 개선하는지, 구현 시 어떤 변화가 생기는지도 함께 다룹니다.' },
        { value: 'lvA_Qzll4QI', title: '집합(유니온연산)', summary: '상호 배타적 집합의 주 연산인 union을 설명하는 강의입니다. 두 집합을 하나로 합칠 때 대표 원소를 어떻게 연결하는지와, union이 find와 어떻게 함께 쓰이는지 정리합니다.' },
        { value: 'nObTu0gqIh4', title: '집합(랭크)', summary: 'union을 수행할 때 트리 높이가 불필요하게 커지지 않도록 랭크를 사용하는 이유를 설명합니다. 어떤 루트를 부모로 삼는지가 성능에 어떤 영향을 주는지 보여 줍니다.' },
        { value: 'HedY-mtg850', title: '집합(마무리)', summary: '서로소 집합 파트 전체를 다시 정리하는 마무리 강의입니다. 개념, 표현, find, 경로 압축, union, rank가 한 구조 안에서 어떻게 맞물리는지 복습 중심으로 정리합니다.' },
        { value: 'fBmIcN_eQAA', title: '그래프(기본 개념)', summary: '그래프가 무엇인지, 정점과 간선으로 어떤 관계를 표현하는지 소개합니다. 방향성, 가중치 등 그래프 문제를 읽을 때 먼저 구분해야 하는 기본 개념을 정리하는 강의입니다.' },
        { value: 'NGOeJY1B7qk', title: '그래프(인접행렬과인접리스트)', summary: '그래프를 코드로 표현하는 대표 방법인 인접 행렬과 인접 리스트를 비교해 설명합니다. 각 표현 방식의 장단점과, 문제 조건에 따라 어떤 방식을 선택할지 판단하는 기준을 다룹니다.' },
        { value: 'sORP6QtUp_s', title: '그래프(DFS와BFS의개념)', summary: 'DFS와 BFS의 기본 개념을 설명하는 강의입니다. 두 탐색이 어떤 순서로 그래프를 훑는지와, 각각의 탐색 철학이 어떻게 다른지 기초부터 정리합니다.' },
        { value: 'MhhORyIWYHk', title: '그래프(스택을활용한DFS)', summary: '스택을 이용해 DFS를 구현하는 흐름을 설명합니다. 재귀 없이도 깊이 우선 탐색을 구현할 수 있다는 점과, 스택에 무엇을 넣고 꺼내야 하는지에 초점을 맞춥니다.' },
        { value: 'RaDeHg10NLs', title: '그래프(재귀를활용한DFS)', summary: '재귀 호출을 이용한 DFS를 설명합니다. 호출 스택 기준으로 탐색이 깊어졌다가 되돌아오는 흐름을 이해하도록 돕는 강의입니다.' },
        { value: 'iIW-7X58Wb4', title: '그래프(큐를활용한BFS)', summary: '큐를 이용해 BFS가 어떻게 진행되는지 짧고 집중적으로 설명합니다. 가까운 정점부터 차례대로 확장하는 BFS의 핵심 동작을 큐 관점에서 보여 줍니다.' },
        { value: 'scGPmqpsSAk', title: '그래프(DFS의구현)', summary: 'DFS를 실제 코드로 구현하는 강의입니다. 방문 배열, 그래프 표현, 순회 로직이 한 코드 안에서 어떻게 맞물리는지 구현 관점에서 정리합니다.' },
        { value: '4tV-dz7v7ao', title: '그래프(BFS구현)', summary: 'BFS를 실제 코드로 구현하는 과정을 다룹니다. 큐 초기화, 방문 처리, 인접 정점 순회 등 BFS 코드의 필수 골격을 정리합니다.' },
        { value: 'l6SdqywBpw0', title: '그래프(문제풀이시DFS와BFS선택하기)', summary: '문제를 풀 때 DFS와 BFS 중 무엇을 선택할지 판단하는 기준을 설명합니다. 탐색의 목적, 최단 거리 필요 여부, 상태 공간의 성격에 따라 어떤 접근이 더 자연스러운지 비교합니다.' },
        { value: '7ZhxvLomajw', title: '최단경로(다익스트라의 개념)', summary: '최단 경로 문제에서 다익스트라 알고리즘이 어떤 역할을 하는지 설명합니다. 가중치 그래프에서 가장 짧은 후보를 확정해 나가는 핵심 아이디어를 중심으로 개념을 잡습니다.' },
        { value: 'Wo6trI4RXlA', title: '최단경로(다익스트라의 동작)', summary: '다익스트라 알고리즘이 실제로 동작하는 과정을 단계적으로 설명합니다. 거리 테이블 갱신과 다음 후보 선택이 어떤 순서로 진행되는지 이해하는 데 초점을 둡니다.' },
        { value: '3SLsupzyzpM', title: '최단경로(다익스트라의 한계)', summary: '다익스트라 알고리즘의 적용 한계를 설명합니다. 특히 음수 간선이 존재할 때 왜 문제가 생기는지, 언제 안전하게 쓸 수 없는지를 짚습니다.' },
        { value: 'Fql3epqo5k4', title: '최단경로(밸만포드 알고리즘의 개념)', summary: '벨만-포드 알고리즘의 개념을 설명하는 강의입니다. 다익스트라와 달리 음수 간선을 다룰 수 있다는 점을 배경으로, 어떤 상황에서 벨만-포드를 고려해야 하는지 소개합니다.' },
        { value: '6PTCch3nONk', title: '최단경로(밸만포드 알고리즘의 실제 동작)', summary: '벨만-포드 알고리즘이 실제로 어떻게 거리 값을 갱신하는지 설명합니다. 모든 간선을 반복 완화하는 방식이 여러 라운드에 걸쳐 어떤 의미를 가지는지 보여 줍니다.' },
        { value: 'Oil_ol0DXMo', title: '최단경로(밸만포드 알고리즘 - 음의 순환 탐지)', summary: '벨만-포드 알고리즘으로 음의 순환을 탐지하는 원리를 설명합니다. 마지막 단계에서도 값이 더 줄어든다면 음의 순환이 있다고 판단하는 기준을 다룹니다.' },
        { value: 'l993FqfBrHM', title: '최단경로(밸만포드 알고리즘의 한계)', summary: '벨만-포드 알고리즘의 한계를 짚는 강의입니다. 음수 간선은 처리할 수 있지만, 그만큼 계산량이 커지는 점과 실전 적용 시 주의할 점을 설명합니다.' },
        { value: 'Hj3sh12lwoU', title: '최단경로(다익스트라 vs 밸만포드)', summary: '다익스트라와 벨만-포드를 비교하는 강의입니다. 적용 조건, 시간 복잡도, 음수 간선 처리 가능 여부를 기준으로 두 알고리즘을 나란히 정리합니다.' },
        { value: 'etRpwAKCvCc', title: '백트래킹(개념)', summary: '백트래킹의 기본 개념을 설명합니다. 완전 탐색과의 차이, 왜 가지치기가 필요한지, 유망 함수를 사용하면 탐색을 어떻게 줄일 수 있는지를 중심으로 정리합니다.' },
        { value: 'nDFrhvdlH98', title: '백트래킹(활용)', summary: '백트래킹을 실제 문제에 적용하는 과정을 보여 주는 강의입니다. 부분합이나 N-Queen처럼 상태를 선택·되돌리며 푸는 전형 문제에 백트래킹이 어떻게 쓰이는지 설명합니다.' }
      ]
    },
    cpp: {
      key: 'cpp',
      label: '코딩 테스트 합격자 되기 - C++편',
      playlistId: 'PLrfS7Czu1oUd8lKwNkc9rarTuXn_Qw6ie',
      bookUrl: 'https://product.kyobobook.co.kr/detail/S000213087020',
      totalLabel: '13 videos',
      items: [
        { value: 'CLXFgptB81M', title: 'C++ 책 소개', summary: '책 전체가 어떤 흐름으로 구성되어 있는지, 어떤 독자를 대상으로 하는지 소개하는 영상입니다. 자료구조·알고리즘·문제 풀이를 어떻게 연결해 학습할지 큰 그림을 먼저 잡아 줍니다.' },
        { value: '_D_LTv-kyF4', title: '효율적으로 공부하기', summary: '코딩 테스트를 효율적으로 공부하는 방법에 초점을 둔 강의입니다. 타인의 풀이를 보는 방법, 아는 것만 반복하지 않는 법, 나만의 용어로 정리하는 습관, 테스트 케이스 설계, 의사코드 작성까지 학습 전략을 구체적으로 다룹니다.' },
        { value: '6oDZgi_7ao0', title: '03장 시간 복잡도', summary: '알고리즘이 무엇인지에서 출발해 성능을 어떻게 측정할지 설명합니다. 연산 횟수 기반 분석, 점근적 표기법, 자주 나오는 복잡도, 실제 코딩 테스트에서 시간 복잡도를 해석하는 법까지 연결합니다.' },
        { value: 'xc0HZiqh8Fs', title: '04/05장 반드시 알아야 할 C++ 문법', summary: '코딩 테스트 전에 반드시 알아야 할 C++ 문법을 한 번에 정리하는 강의입니다. 빌트인 데이터 타입, 배열, 문자열, STL 개념, 반복자, 컨테이너, 알고리즘을 폭넓게 다룹니다.' },
        { value: '-TGCT74wFeg', title: '06/07장 스택과 큐', summary: '스택과 큐의 개념을 각각 설명하고, 언제 쓰이는지 예시 중심으로 정리합니다. 자료구조를 외우는 수준이 아니라 문제 상황에서 어떤 구조를 떠올려야 하는지 이해하는 데 초점을 둡니다.' },
        { value: 'KsfmSyIYX3g', title: '08장 해시', summary: '해시의 개념과 빠른 탐색 구조로서의 의미를 설명하는 강의입니다. 키를 저장·검색하는 방식과 충돌 처리 관점을 함께 이해하도록 구성된 해시 입문 파트입니다.' },
        { value: 'CXiVJgUjL6o', title: '09장 트리', summary: '트리의 기본 개념과 구조를 설명하고, 코딩 테스트에서 자주 마주치는 트리 문제의 핵심 포인트를 정리합니다. 트리 순회와 이진 탐색 트리 같은 중요한 하위 주제로 자연스럽게 이어지는 강의입니다.' },
        { value: 'pQ4fcGEG-PY', title: '10장 집합', summary: '집합 파트 전체를 체계적으로 정리하는 강의입니다. 상호 배타적 집합의 개념, 배열 기반 표현, find, 경로 압축, union, 랭크 기반 최적화, 실제 구현과 사용 예시까지 순서대로 설명합니다.' },
        { value: 'OmYQsxreXNo', title: '11장 그래프', summary: '그래프의 개념과 종류를 소개한 뒤 인접 행렬과 인접 리스트 구현을 비교합니다. 이어서 DFS와 BFS, 둘의 차이, 최단 경로 아이디어, 다익스트라 알고리즘까지 그래프 파트를 넓게 다루는 강의입니다.' },
        { value: 'VvcEx75Bgvk', title: '12장 백트래킹', summary: '백트래킹의 개념과 필요성을 설명하고, 문제를 푸는 과정 자체를 구조적으로 보여 줍니다. 부분합과 N-Queen 예시를 통해 가지치기와 상태 복원이 실제로 어떻게 쓰이는지 설명합니다.' },
        { value: 'j4ExZH1r3jE', title: '13장 정렬', summary: '정렬이 왜 필요한지에서 시작해 삽입 정렬, 병합 정렬, 힙 정렬을 설명합니다. 이어서 우선순위 큐와 실제 구현까지 연결해, 정렬 알고리즘을 문제 풀이 관점에서 정리합니다.' },
        { value: '_K2WH4VhdkQ', title: '15장 동적 계획법', summary: '동적 계획법이 어떤 문제에서 필요한지 개념부터 설명합니다. 팩토리얼과 피보나치로 아이디어를 소개한 뒤, 계단 오르기, 정사각형 개수, 최장 증가 부분 수열 같은 예제로 DP 적용 과정을 보여 줍니다.' },
        { value: 'k3ctQPulBW8', title: '16장 그리디', summary: '그리디 알고리즘의 개념과, 최적해를 보장하려면 어떤 특성이 필요한지 설명합니다. 거스름돈 문제와 최소 신장 트리, 그리고 코딩 테스트에서 자주 나오는 그리디 패턴을 함께 정리합니다.' }
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
