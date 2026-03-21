
(() => {
  const frame = document.getElementById('study-frame');
  const rootList = document.getElementById('study-root-list');
  const groupList = document.getElementById('study-group-list');
  const titleNode = document.getElementById('study-viewer-title');
  const breadcrumbNode = document.getElementById('study-breadcrumb');

  if (!frame || !rootList || !groupList) return;

  const studyTree = [
    {
      key: 'visualizer',
      label: '시각화',
      path: 'projects/python-flow-visualizer.html',
      groups: [
        {
          key: 'sorting',
          label: '정렬',
          items: [
            { id: 'bubble-sort', label: '버블 정렬' },
            { id: 'insertion-sort', label: '삽입 정렬' },
            { id: 'merge-sort', label: '머지 정렬' },
            { id: 'quick-sort', label: '퀵 정렬' },
            { id: 'counting-sort', label: '계수 정렬' },
            { id: 'heap-sort', label: '힙 정렬' },
          ],
        },
        {
          key: 'structures',
          label: '스택·큐',
          items: [
            { id: 'stack-basic', label: '스택' },
            { id: 'queue-basic', label: '큐' },
          ],
        },
        {
          key: 'graph',
          label: '그래프 탐색',
          items: [
            { id: 'dfs', label: '깊이 우선 탐색' },
            { id: 'bfs', label: '너비 우선 탐색' },
          ],
        },
        {
          key: 'tree-search',
          label: '트리 탐색',
          items: [
            { id: 'bst-search', label: 'BST 탐색' },
          ],
        },
        {
          key: 'tree-traversal',
          label: '트리 순회',
          items: [
            { id: 'preorder', label: '전위 순회' },
            { id: 'inorder', label: '중위 순회' },
            { id: 'postorder', label: '후위 순회' },
          ],
        },
        {
          key: 'dynamic',
          label: '동적 계획법·백트래킹',
          items: [
            { id: 'n-queen', label: 'N-Queen' },
            { id: 'subset-sum-pruning', label: '부분집합 합 가지치기' },
          ],
        },
        {
          key: 'recursion',
          label: '재귀',
          items: [
            { id: 'factorial', label: '팩토리얼' },
            { id: 'fibonacci', label: '피보나치' },
            { id: 'combination', label: '조합' },
          ],
        },
      ],
    },
  ];

  let currentRootKey = 'visualizer';
  let currentScenarioId = 'bubble-sort';
  let resizeObserver = null;
  let mutationObserver = null;
  let rafId = null;

  function getCurrentRoot() {
    return studyTree.find((item) => item.key === currentRootKey) || studyTree[0];
  }

  function findScenarioInfo(id) {
    for (const root of studyTree) {
      for (const group of root.groups) {
        for (const item of group.items) {
          if (item.id === id) {
            return { root, group, item };
          }
        }
      }
    }
    return null;
  }

  function buildFrameUrl(root, scenarioId) {
    return `${root.path}?embed=1&scenario=${encodeURIComponent(scenarioId)}`;
  }

  function renderRootMenu() {
    rootList.innerHTML = studyTree.map((root) => `
      <button type="button" class="study-root-button ${root.key === currentRootKey ? 'is-active' : ''}" data-root="${root.key}">
        ${root.label}
      </button>
    `).join('');

    rootList.querySelectorAll('[data-root]').forEach((button) => {
      button.addEventListener('click', () => {
        const nextRootKey = button.dataset.root;
        if (nextRootKey === currentRootKey) return;
        currentRootKey = nextRootKey;
        const root = getCurrentRoot();
        currentScenarioId = root.groups[0].items[0].id;
        renderRootMenu();
        renderGroupMenu();
        loadScenario(true);
      });
    });
  }

  function renderGroupMenu() {
    const root = getCurrentRoot();
    groupList.innerHTML = root.groups.map((group) => `
      <section class="study-group">
        <h3 class="study-group-title">${group.label}</h3>
        <div class="study-submenu">
          ${group.items.map((item) => `
            <button type="button" class="study-subitem ${item.id === currentScenarioId ? 'is-active' : ''}" data-scenario="${item.id}">
              ${item.label}
            </button>
          `).join('')}
        </div>
      </section>
    `).join('');

    groupList.querySelectorAll('[data-scenario]').forEach((button) => {
      button.addEventListener('click', () => {
        const nextScenarioId = button.dataset.scenario;
        if (nextScenarioId === currentScenarioId) return;
        currentScenarioId = nextScenarioId;
        renderGroupMenu();
        loadScenario(false);
      });
    });
  }

  function updateViewerHeader() {
    const info = findScenarioInfo(currentScenarioId);
    if (!info) return;
    breadcrumbNode.textContent = `${info.root.label} · ${info.group.label}`;
    titleNode.textContent = info.item.label;
  }

  function injectViewerTheme() {
    try {
      const doc = frame.contentDocument || frame.contentWindow?.document;
      if (!doc) return;

      let style = doc.getElementById('kp-study-theme-override');
      if (!style) {
        style = doc.createElement('style');
        style.id = 'kp-study-theme-override';
        doc.head.appendChild(style);
      }

      style.textContent = `
        :root {
          --bg: #f6f8fa !important;
          --bg-2: #f6f8fa !important;
          --panel: #ffffff !important;
          --panel-strong: #ffffff !important;
          --panel-soft: #f6f8fa !important;
          --line: #d0d7de !important;
          --line-strong: #c4ccd5 !important;
          --text: #1f2328 !important;
          --muted: #59636e !important;
          --muted-2: #6e7781 !important;
          --accent: #24292f !important;
          --accent-2: #0969da !important;
          --accent-soft: #eef2f7 !important;
          --good: #1a7f37 !important;
          --warn: #9a6700 !important;
          --danger: #cf222e !important;
          --cyan: #0969da !important;
          --shadow: 0 1px 2px rgba(31, 35, 40, 0.08) !important;
        }
        html, body {
          background: #f6f8fa !important;
          color: #1f2328 !important;
        }
        body {
          padding: 0 !important;
        }
        .glass,
        .sidebar-card,
        .stat-card,
        .panel-card,
        .visual-panel,
        .hero,
        .toolbar,
        .section-button,
        .algorithm-tab,
        .control-btn,
        .workspace-card,
        .state-card,
        .legend-item,
        .array-table,
        .array-cell,
        .board-frame,
        .structure-card,
        .description-card,
        .code-block {
          background: #ffffff !important;
          border-color: #d0d7de !important;
          box-shadow: 0 1px 2px rgba(31, 35, 40, 0.08) !important;
          backdrop-filter: none !important;
        }
        .section-button.active,
        .algorithm-tab.active,
        .control-btn.active {
          background: #eef2f7 !important;
          border-color: #b6c3d1 !important;
          color: #1f2328 !important;
        }
      `;
    } catch (error) {
      // same-origin access only
    }
  }

  function syncFrameHeight() {
    try {
      const doc = frame.contentDocument || frame.contentWindow?.document;
      if (!doc) return;
      const nextHeight = Math.max(
        doc.body ? doc.body.scrollHeight : 0,
        doc.documentElement ? doc.documentElement.scrollHeight : 0,
      );
      if (nextHeight) {
        frame.style.height = `${nextHeight + 12}px`;
      }
    } catch (error) {
      // same-origin access only
    }
  }

  function queueHeightSync() {
    if (rafId) cancelAnimationFrame(rafId);
    rafId = requestAnimationFrame(syncFrameHeight);
  }

  function bindFrameObservers() {
    try {
      const win = frame.contentWindow;
      const doc = frame.contentDocument || win?.document;
      if (!win || !doc || !doc.body) return;

      if (resizeObserver) resizeObserver.disconnect();
      if (mutationObserver) mutationObserver.disconnect();

      if ('ResizeObserver' in window) {
        resizeObserver = new ResizeObserver(() => queueHeightSync());
        resizeObserver.observe(doc.body);
        if (doc.documentElement) resizeObserver.observe(doc.documentElement);
      }

      mutationObserver = new MutationObserver(() => queueHeightSync());
      mutationObserver.observe(doc.body, {
        childList: true,
        subtree: true,
        attributes: true,
        characterData: true,
      });

      win.addEventListener('resize', queueHeightSync);
      queueHeightSync();
      setTimeout(queueHeightSync, 60);
      setTimeout(queueHeightSync, 220);
      setTimeout(queueHeightSync, 500);
    } catch (error) {
      // ignore observer errors
    }
  }

  function loadScenario(forceReload) {
    const root = getCurrentRoot();
    updateViewerHeader();

    const samePageLoaded = frame.dataset.loadedRoot === root.key;
    if (forceReload || !samePageLoaded) {
      frame.src = buildFrameUrl(root, currentScenarioId);
      frame.dataset.loadedRoot = root.key;
      return;
    }

    try {
      if (frame.contentWindow?.visualizerApi?.setScenario) {
        frame.contentWindow.visualizerApi.setScenario(currentScenarioId);
        queueHeightSync();
      } else {
        frame.src = buildFrameUrl(root, currentScenarioId);
      }
    } catch (error) {
      frame.src = buildFrameUrl(root, currentScenarioId);
    }
  }

  frame.addEventListener('load', () => {
    injectViewerTheme();
    bindFrameObservers();
    try {
      if (frame.contentWindow?.visualizerApi?.setScenario) {
        frame.contentWindow.visualizerApi.setScenario(currentScenarioId);
      }
    } catch (error) {
      // ignore api call errors
    }
    queueHeightSync();
  });

  window.addEventListener('message', (event) => {
    const data = event.data;
    if (!data || data.type !== 'kp-visualizer-height') return;
    if (typeof data.height === 'number' && data.height > 0) {
      frame.style.height = `${Math.round(data.height + 12)}px`;
    }
  });

  renderRootMenu();
  renderGroupMenu();
  updateViewerHeader();
  loadScenario(true);
})();
