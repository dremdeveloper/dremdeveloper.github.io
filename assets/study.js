(() => {
  const frame = document.getElementById('study-frame');
  const scenarioSelect = document.getElementById('study-scenario-select');
  const titleNode = document.getElementById('study-viewer-title');
  const breadcrumbNode = document.getElementById('study-breadcrumb');
  const submenuRootsNode = document.getElementById('study-submenu-roots');
  const submenuSelectorNode = document.getElementById('study-submenu-selector');
  const selectorTitleNode = document.getElementById('study-selector-title');
  const selectorDescriptionNode = document.getElementById('study-selector-description');
  const currentGroupNode = document.getElementById('study-current-group');
  const totalCountNode = document.getElementById('study-total-count');
  const groupCountNode = document.getElementById('study-group-count');
  const currentIndexNode = document.getElementById('study-current-index');
  const currentDescriptionNode = document.getElementById('study-current-description');
  const currentStatusNode = document.getElementById('study-viewer-status');
  const quickChipsNode = document.getElementById('study-quick-chips');
  const toolbarTitleNode = document.getElementById('study-toolbar-title');
  const toolbarDescriptionNode = document.getElementById('study-toolbar-description');
  const prevButton = document.getElementById('study-prev-button');
  const nextButton = document.getElementById('study-next-button');
  const submenuPanel = document.getElementById('study-top-submenu');
  const submenuToggle = document.getElementById('study-nav-toggle');

  if (!frame || !scenarioSelect || !submenuRootsNode || !submenuSelectorNode) return;

  const studyTree = [
    {
      key: 'visualizer',
      label: '시각화',
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
            { id: 'heap-sort', label: '힙 정렬' },
          ],
        },
        {
          key: 'structures',
          label: '스택·큐',
          description: '자료구조 내부 상태가 push/pop, enqueue/dequeue에 따라 어떻게 바뀌는지 확인합니다.',
          items: [
            { id: 'stack-basic', label: '스택' },
            { id: 'queue-basic', label: '큐' },
          ],
        },
        {
          key: 'graph',
          label: '그래프 탐색',
          description: '탐색 순서와 방문 상태가 그래프 위에서 어떻게 확장되는지 살펴봅니다.',
          items: [
            { id: 'dfs', label: '깊이 우선 탐색' },
            { id: 'bfs', label: '너비 우선 탐색' },
          ],
        },
        {
          key: 'tree-search',
          label: '트리 탐색',
          description: 'BST에서 탐색 경로가 조건문과 함께 어떻게 결정되는지 보여줍니다.',
          items: [
            { id: 'bst-search', label: 'BST 탐색' },
          ],
        },
        {
          key: 'tree-traversal',
          label: '트리 순회',
          description: '전위·중위·후위 순회가 재귀 호출 스택과 함께 어떻게 이동하는지 설명합니다.',
          items: [
            { id: 'preorder', label: '전위 순회' },
            { id: 'inorder', label: '중위 순회' },
            { id: 'postorder', label: '후위 순회' },
          ],
        },
        {
          key: 'dynamic',
          label: '동적 계획법·백트래킹',
          description: '가지치기와 상태 전이를 시각적으로 비교하며 문제 해결 흐름을 따라갑니다.',
          items: [
            { id: 'n-queen', label: 'N-Queen' },
            { id: 'subset-sum-pruning', label: '부분집합 합 가지치기' },
          ],
        },
        {
          key: 'recursion',
          label: '재귀',
          description: '재귀 호출의 전개와 복귀 시점이 코드 흐름에 맞춰 어떻게 변하는지 확인합니다.',
          items: [
            { id: 'factorial', label: '팩토리얼' },
            { id: 'fibonacci', label: '피보나치' },
            { id: 'combination', label: '조합' },
          ],
        },
      ],
    },
  ];

  const totalScenarioCount = studyTree.reduce(
    (sum, root) => sum + root.groups.reduce((groupSum, group) => groupSum + group.items.length, 0),
    0,
  );

  let currentRootKey = 'visualizer';
  let currentScenarioId = 'bubble-sort';
  let isRootSelectorVisible = false;
  let resizeObserver = null;
  let mutationObserver = null;
  let rafId = null;
  let boundFrameWindow = null;

  function getCurrentRoot() {
    return studyTree.find((item) => item.key === currentRootKey) || studyTree[0];
  }

  function getRootScenarioItems(root = getCurrentRoot()) {
    return root.groups.flatMap((group) => group.items.map((item) => ({ ...item, group })));
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

  function getScenarioPosition(root, scenarioId) {
    const items = getRootScenarioItems(root);
    return items.findIndex((item) => item.id === scenarioId);
  }

  function buildFrameUrl(root, scenarioId) {
    return `${root.path}?embed=1&scenario=${encodeURIComponent(scenarioId)}`;
  }

  function escapeHtml(value) {
    return String(value)
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;')
      .replaceAll("'", '&#39;');
  }

  function renderQuickChips() {
    if (!quickChipsNode) return;
    const info = findScenarioInfo(currentScenarioId);
    if (!info) {
      quickChipsNode.hidden = true;
      quickChipsNode.setAttribute('aria-hidden', 'true');
      quickChipsNode.innerHTML = '';
      return;
    }

    quickChipsNode.hidden = false;
    quickChipsNode.setAttribute('aria-hidden', 'false');
    quickChipsNode.innerHTML = info.group.items.map((item) => `
      <button
        type="button"
        class="study-quick-chip ${item.id === currentScenarioId ? 'is-active' : ''}"
        data-scenario-chip="${item.id}"
        aria-pressed="${item.id === currentScenarioId ? 'true' : 'false'}"
      >
        ${escapeHtml(item.label)}
      </button>
    `).join('');

    quickChipsNode.querySelectorAll('[data-scenario-chip]').forEach((button) => {
      button.addEventListener('click', () => {
        setScenario(button.dataset.scenarioChip);
      });
    });
  }

  function updateStepperControls() {
    const root = getCurrentRoot();
    const items = getRootScenarioItems(root);
    const currentIndex = items.findIndex((item) => item.id === currentScenarioId);
    const currentItem = items[currentIndex];

    if (prevButton) prevButton.disabled = currentIndex <= 0;
    if (nextButton) nextButton.disabled = currentIndex < 0 || currentIndex >= items.length - 1;

    if (toolbarTitleNode && currentItem) {
      toolbarTitleNode.textContent = `${currentItem.group.label} · ${currentItem.label}`;
    }

    if (toolbarDescriptionNode) {
      const positionLabel = currentIndex >= 0 ? `${currentIndex + 1} / ${items.length}` : '0 / 0';
      toolbarDescriptionNode.textContent = `같은 주제 안에서 ${positionLabel} 순서로 이동하고 있습니다. 빠른 이동 칩이나 이전·다음 버튼으로 흐름을 이어서 볼 수 있습니다.`;
    }
  }

  function setScenario(nextScenarioId, forceReload = false) {
    if (!nextScenarioId) return;
    const nextInfo = findScenarioInfo(nextScenarioId);
    if (!nextInfo) return;

    currentRootKey = nextInfo.root.key;
    const isSameScenario = nextScenarioId === currentScenarioId;
    currentScenarioId = nextScenarioId;

    renderScenarioSelect();
    renderSubmenu();
    updateViewerHeader();
    renderQuickChips();
    updateStepperControls();

    loadScenario(forceReload || isSameScenario);
  }

  function renderSubmenu() {
    submenuRootsNode.innerHTML = studyTree.map((root) => `
      <button
        type="button"
        class="study-submenu-root ${root.key === currentRootKey ? 'is-active' : ''}"
        data-root="${root.key}"
        aria-pressed="${root.key === currentRootKey && isRootSelectorVisible ? 'true' : 'false'}"
      >
        <span class="study-submenu-root-label">${escapeHtml(root.label)}</span>
        <span class="study-submenu-root-meta">${root.groups.length}개 카테고리</span>
      </button>
    `).join('');

    submenuRootsNode.querySelectorAll('[data-root]').forEach((button) => {
      button.addEventListener('click', () => {
        const nextRoot = studyTree.find((root) => root.key === button.dataset.root);
        if (!nextRoot) return;

        currentRootKey = nextRoot.key;
        isRootSelectorVisible = true;

        const nextRootItems = getRootScenarioItems(nextRoot);
        if (!nextRootItems.some((item) => item.id === currentScenarioId) && nextRootItems[0]) {
          currentScenarioId = nextRootItems[0].id;
        }

        renderSubmenu();
        renderScenarioSelect();
        updateViewerHeader();
        renderQuickChips();
        updateStepperControls();
      });
    });

    submenuSelectorNode.hidden = !isRootSelectorVisible;
  }

  function renderScenarioSelect() {
    const root = getCurrentRoot();
    if (selectorTitleNode) selectorTitleNode.textContent = root.label;
    if (selectorDescriptionNode) {
      const scenarioCount = root.groups.reduce((sum, group) => sum + group.items.length, 0);
      selectorDescriptionNode.textContent = `${scenarioCount}개의 시나리오를 카테고리별로 골라 바로 확인할 수 있습니다.`;
    }

    scenarioSelect.innerHTML = root.groups.map((group) => `
      <optgroup label="${escapeHtml(group.label)}">
        ${group.items.map((item) => `
          <option value="${item.id}" ${item.id === currentScenarioId ? 'selected' : ''}>${escapeHtml(item.label)}</option>
        `).join('')}
      </optgroup>
    `).join('');

    scenarioSelect.value = currentScenarioId;
  }

  function updateViewerHeader() {
    const info = findScenarioInfo(currentScenarioId);
    if (!info) return;

    const itemIndex = info.group.items.findIndex((item) => item.id === currentScenarioId) + 1;
    const rootScenarioCount = getRootScenarioItems(info.root).length;

    breadcrumbNode.textContent = `${info.root.label} · ${info.group.label}`;
    titleNode.textContent = info.item.label;

    if (currentGroupNode) currentGroupNode.textContent = info.group.label;
    if (totalCountNode) totalCountNode.textContent = `${totalScenarioCount}개`;
    if (groupCountNode) groupCountNode.textContent = `${info.group.items.length}개 시나리오`;
    if (currentIndexNode) currentIndexNode.textContent = `${itemIndex} / ${info.group.items.length}`;
    if (currentDescriptionNode) currentDescriptionNode.textContent = info.group.description;
    if (currentStatusNode) {
      currentStatusNode.textContent = `${info.group.label} 카테고리의 ${itemIndex}번째 시나리오입니다. 현재 주제에는 총 ${rootScenarioCount}개의 시나리오가 준비되어 있어 흐름을 이어서 보기 좋습니다.`;
    }
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
          --bg: #f4f7fb !important;
          --bg-2: #f4f7fb !important;
          --panel: #ffffff !important;
          --panel-strong: #ffffff !important;
          --panel-soft: #f5f8fc !important;
          --line: #d0d7de !important;
          --line-strong: #c4ccd5 !important;
          --text: #111827 !important;
          --muted: #44505c !important;
          --muted-2: #5b6773 !important;
          --accent: #1f2937 !important;
          --accent-2: #0969da !important;
          --accent-soft: #eaf1f8 !important;
          --good: #1a7f37 !important;
          --warn: #9a6700 !important;
          --danger: #cf222e !important;
          --cyan: #0b63ce !important;
          --shadow: 0 8px 24px rgba(15, 23, 42, 0.08) !important;
        }
        html {
          color-scheme: light !important;
          scroll-behavior: auto !important;
        }
        html, body {
          background: linear-gradient(180deg, #f8fbff 0%, #f4f7fb 100%) !important;
          color: #111827 !important;
        }
        body {
          padding: 0 !important;
        }
        *, *::before, *::after {
          animation-duration: 0.12s !important;
          animation-delay: 0s !important;
          transition-duration: 0.12s !important;
          transition-delay: 0s !important;
          caret-color: #111827 !important;
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
        .code-block,
        .note-chip,
        .badge,
        .stack-token,
        .structure-token,
        .queue-lane,
        .stack-column,
        .graph-stage,
        .tree-stage,
        .recursion-stage,
        .pruning-stage {
          background: #ffffff !important;
          border-color: #d0d7de !important;
          box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06) !important;
          backdrop-filter: none !important;
        }
        .ambient,
        .ambient-a,
        .ambient-b {
          display: none !important;
        }
        .section-button.active,
        .algorithm-tab.active,
        .control-btn.active,
        .control-btn-primary,
        .badge-highlight,
        .legend-item.highlight {
          background: #eaf1f8 !important;
          border-color: #b8c4d0 !important;
          color: #111827 !important;
          box-shadow: none !important;
        }
        .panel-title,
        .workspace-head,
        .description-title,
        .state-value,
        .graph-node text,
        .tree-node text,
        .call-node text,
        .brand h1,
        .hero h2,
        .visual-guide-title,
        .stat-value,
        .section-title,
        .algorithm-tab,
        .control-btn,
        .factorial-result,
        .array-value,
        .code-block,
        strong {
          color: #111827 !important;
          fill: #111827 !important;
        }
        .description-summary,
        .description-detail,
        .state-label,
        .meta-line,
        label,
        .bar-footer,
        .panel-subtitle,
        .brand-copy,
        .hero p,
        .project-meta p,
        .note-block p,
        .section-desc,
        .sidebar-list,
        .stat-label,
        .visual-guide-detail,
        .token-hint,
        .factorial-result.muted,
        .array-value.muted,
        .factorial-note,
        .factorial-expression,
        .queue-caption,
        .stack-caption {
          color: #44505c !important;
          fill: #44505c !important;
        }
        .eyebrow,
        .visual-guide-tag {
          color: #0b63ce !important;
        }
        .description-card {
          background: linear-gradient(180deg, #f8fbff, #eef4fb) !important;
          border-color: #c8d7e6 !important;
        }
        .state-card,
        .workspace-card,
        .graph-stage,
        .tree-stage,
        .recursion-stage,
        .board-frame,
        .array-table-panel {
          background: #ffffff !important;
          border-color: #d0d7de !important;
        }
        .bar-track,
        .stack-column,
        .queue-lane,
        .token-row,
        .board-grid {
          background: #f4f7fb !important;
        }
        .bar-track {
          background: #e5e7eb !important;
          border: 1px solid #c4ccd5 !important;
        }
        .bar-fill {
          color: #ffffff !important;
          background: linear-gradient(180deg, #1f2937, #111827) !important;
          border: 1px solid #111827 !important;
          box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.16), 0 6px 14px rgba(17, 24, 39, 0.16) !important;
        }
        .bar-card.comparing .bar-fill,
        .bar-card.key .bar-fill,
        .bar-card.shifted .bar-fill,
        .bar-card.sorted .bar-fill,
        .bar-card.fixed .bar-fill,
        .bar-card.current-write .bar-fill {
          background: linear-gradient(180deg, #1f2937, #111827) !important;
          border-color: #111827 !important;
        }
        .pulse-ring,
        .current-node,
        .call-stack-item.active,
        .stack-frame.is-active,
        .tree-node.active,
        .graph-node.active,
        [class*="pulse"],
        [class*="highlight"][style*="animation"] {
          animation: none !important;
        }
        @media (prefers-reduced-motion: reduce) {
          *, *::before, *::after {
            animation: none !important;
            transition: none !important;
          }
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

      if (boundFrameWindow && boundFrameWindow !== win) {
        boundFrameWindow.removeEventListener('resize', queueHeightSync);
      }
      boundFrameWindow = win;

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

  function stepScenario(direction) {
    const root = getCurrentRoot();
    const items = getRootScenarioItems(root);
    const currentIndex = getScenarioPosition(root, currentScenarioId);
    const nextIndex = currentIndex + direction;
    if (nextIndex < 0 || nextIndex >= items.length) return;
    setScenario(items[nextIndex].id);
  }

  function toggleSubmenu(forceOpen) {
    if (!submenuPanel || !submenuToggle) return;
    const shouldOpen = typeof forceOpen === 'boolean'
      ? forceOpen
      : submenuToggle.getAttribute('aria-expanded') !== 'true';

    isRootSelectorVisible = false;
    renderSubmenu();

    submenuToggle.setAttribute('aria-expanded', String(shouldOpen));
    submenuPanel.hidden = !shouldOpen;
  }

  scenarioSelect.addEventListener('change', () => {
    const nextScenarioId = scenarioSelect.value;
    if (!nextScenarioId) return;
    setScenario(nextScenarioId);
  });

  prevButton?.addEventListener('click', () => stepScenario(-1));
  nextButton?.addEventListener('click', () => stepScenario(1));

  if (submenuToggle && submenuPanel) {
    submenuToggle.addEventListener('click', () => toggleSubmenu());

    document.addEventListener('click', (event) => {
      if (submenuPanel.hidden) return;
      const target = event.target;
      if (!(target instanceof Node)) return;
      if (submenuPanel.contains(target) || submenuToggle.contains(target)) return;
      toggleSubmenu(false);
    });

    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') toggleSubmenu(false);
    });
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

  renderSubmenu();
  renderScenarioSelect();
  updateViewerHeader();
  renderQuickChips();
  updateStepperControls();
  loadScenario(true);
})();
