(() => {
  const data = window.BOJ_ARCHIVE_DATA;
  if (!data || !Array.isArray(data.problems)) return;

  const extraProblems = [
    {
      id: 21918,
      title: '전구',
      url: 'https://www.acmicpc.net/problem/21918',
      tier: '',
      time_limit: '-',
      memory_limit: '-',
      submissions: 0,
      accepted: 0,
      solved: 0,
      acceptance_rate: null,
      algorithms: ['구현'],
      topics: ['코딩테스트 대비 문제집', '구현 (수정 : 2021-05-06)'],
      in_chat: false,
      chat_title: null
    },
    {
      id: 21922,
      title: '학부 연구생 민상',
      url: 'https://www.acmicpc.net/problem/21922',
      tier: '',
      time_limit: '-',
      memory_limit: '-',
      submissions: 0,
      accepted: 0,
      solved: 0,
      acceptance_rate: null,
      algorithms: ['구현', '시뮬레이션'],
      topics: ['코딩테스트 대비 문제집', '시뮬레이션 (수정 : 2021-04-20)'],
      in_chat: false,
      chat_title: null
    },
    {
      id: 22860,
      title: '폴더 정리 (small)',
      url: 'https://www.acmicpc.net/problem/22860',
      tier: '',
      time_limit: '-',
      memory_limit: '-',
      submissions: 0,
      accepted: 0,
      solved: 0,
      acceptance_rate: null,
      algorithms: ['구현'],
      topics: ['코딩테스트 대비 문제집', '구현 (수정 : 2021-05-06)'],
      in_chat: false,
      chat_title: null
    },
    {
      id: 22861,
      title: '폴더 정리 (large)',
      url: 'https://www.acmicpc.net/problem/22861',
      tier: '',
      time_limit: '-',
      memory_limit: '-',
      submissions: 0,
      accepted: 0,
      solved: 0,
      acceptance_rate: null,
      algorithms: ['구현', '시뮬레이션'],
      topics: ['코딩테스트 대비 문제집', '시뮬레이션 (수정 : 2021-04-20)'],
      in_chat: false,
      chat_title: null
    }
  ];

  const existingIds = new Set(data.problems.map((problem) => problem.id));
  const additions = extraProblems.filter((problem) => !existingIds.has(problem.id));
  if (!additions.length) return;

  data.problems.push(...additions);

  const incrementCount = (items, name, amount = 1) => {
    const entry = items.find((item) => item.name === name);
    if (entry) {
      entry.count += amount;
      return;
    }
    items.push({ name, count: amount });
  };

  additions.forEach((problem) => {
    (problem.topics || []).forEach((topic) => incrementCount(data.allTopics, topic));
    (problem.algorithms || []).forEach((algorithm) => incrementCount(data.allAlgorithms, algorithm));
  });

  if (Array.isArray(data.topTopics)) {
    data.topTopics = data.allTopics
      .slice()
      .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name, 'ko'))
      .slice(0, data.topTopics.length);
  }

  if (Array.isArray(data.topAlgorithms)) {
    data.topAlgorithms = data.allAlgorithms
      .slice()
      .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name, 'ko'))
      .slice(0, data.topAlgorithms.length);
  }

  data.allTopics.sort((a, b) => b.count - a.count || a.name.localeCompare(b.name, 'ko'));
  data.allAlgorithms.sort((a, b) => b.count - a.count || a.name.localeCompare(b.name, 'ko'));

  if (data.stats) {
    const nextTotal = (Number(data.stats.finalUniqueProblems) || data.problems.length) + additions.length;
    data.stats.finalUniqueProblems = nextTotal;
  }

  data.researchSources = [
    {
      label: 'Tony9402 코딩테스트 대비 문제집 메인',
      url: 'https://github.com/tony9402/baekjoon'
    },
    {
      label: 'Tony9402 구현 추천 문제',
      url: 'https://raw.githubusercontent.com/tony9402/baekjoon/main/algorithms/implementation/README.md'
    },
    {
      label: 'Tony9402 시뮬레이션 추천 문제',
      url: 'https://raw.githubusercontent.com/tony9402/baekjoon/main/algorithms/simulation/README.md'
    }
  ];
})();
