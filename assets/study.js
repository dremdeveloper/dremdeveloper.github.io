(() => {
  const frame = document.getElementById('study-frame');
  const groupSelect = document.getElementById('study-group-select');
  const scenarioSelect = document.getElementById('study-scenario-select');
  const visualizationToggle = document.getElementById('study-visualization-toggle');
  const learningPlanToggle = document.getElementById('study-learning-plan-toggle');
  const toolbarTitleNode = document.getElementById('study-toolbar-title');
  const visualizationPanel = document.getElementById('study-visualization-panel');
  const learningPlanPanel = document.getElementById('study-learning-plan-panel');
  const selectorCard = document.getElementById('study-selector-card');
  const planMenuRoot = document.getElementById('study-plan-menu');
  const planFrame = document.getElementById('study-plan-frame');
  const submenuPanel = document.getElementById('study-top-submenu');
  const submenuToggle = document.getElementById('study-nav-toggle');

  const studyMaterials = Array.isArray(window.siteData?.studyMaterials)
    ? window.siteData.studyMaterials.filter((material) => Array.isArray(material.groups) && material.groups.length)
    : [];
  const studyPlans = Array.isArray(window.siteData?.studyPlans)
    ? window.siteData.studyPlans.filter((plan) => Array.isArray(plan.files) && plan.files.length)
    : [];

  if (!studyMaterials.length && !studyPlans.length) return;

  let currentMaterialKey = '';
  let currentGroupKey = '';
  let currentScenarioId = '';
  let currentPlanKey = '';
  let currentPlanFileId = '';
  let expandedPlanWeek = '';
  let resizeObserver = null;
  let mutationObserver = null;
  let rafId = null;
  let boundFrameWindow = null;
  let hasLoadedVisualization = false;
  let hasLoadedPlan = false;
  let pendingScrollAnchorTop = null;
  let pendingScrollAnchorTimeout = null;
  let submenuCloseTimer = null;
  const requestedView = new URLSearchParams(window.location.search).get('view');

  function normalizeRequestedView(value) {
    if (!value) return '';
    const normalizedValue = String(value).trim().toLowerCase();
    if (['visualization', 'visual', 'viz'].includes(normalizedValue)) return 'visualization';
    if (['plan', 'learning-plan', 'learning_plan'].includes(normalizedValue)) return 'plan';
    return '';
  }

  function updateViewQuery(view) {
    const normalizedView = normalizeRequestedView(view);
    const url = new URL(window.location.href);

    if (normalizedView) {
      url.searchParams.set('view', normalizedView);
    } else {
      url.searchParams.delete('view');
    }

    window.history.replaceState({}, '', url);
  }

  function ensureInitialSelection() {
    if (currentMaterialKey && currentGroupKey && currentScenarioId) return;

    const defaultMaterial = studyMaterials[0];
    const defaultGroup = defaultMaterial?.groups[0];
    const defaultScenario = defaultGroup?.items[0];

    if (!defaultMaterial || !defaultGroup || !defaultScenario) return;

    currentMaterialKey = defaultMaterial.key;
    currentGroupKey = defaultGroup.key;
    currentScenarioId = defaultScenario.id;
  }

  function getCurrentMaterial() {
    ensureInitialSelection();
    return studyMaterials.find((item) => item.key === currentMaterialKey) || studyMaterials[0];
  }

  function getCurrentGroup(material = getCurrentMaterial()) {
    return material.groups.find((group) => group.key === currentGroupKey) || material.groups[0];
  }

  function ensureInitialPlanSelection() {
    if (currentPlanKey && currentPlanFileId) return;

    const defaultPlan = studyPlans[0];
    const defaultFile = defaultPlan?.files[0];
    const defaultGroup = defaultFile?.lessons?.[0];
    const defaultLesson = defaultGroup?.items?.[0];

    if (!defaultPlan || !defaultFile) return;

    currentPlanKey = defaultPlan.key;
    currentPlanFileId = defaultLesson?.id || defaultFile.id;
    expandedPlanWeek = defaultGroup?.group || '';
  }

  function getCurrentPlan() {
    ensureInitialPlanSelection();
    return studyPlans.find((plan) => plan.key === currentPlanKey) || studyPlans[0];
  }

  function findPlanWeekByFileId(fileId) {
    for (const plan of studyPlans) {
      for (const file of plan.files) {
        if (!Array.isArray(file.lessons)) continue;
        for (const group of file.lessons) {
          if (group.items.some((item) => item.id === fileId)) {
            return group.group;
          }
        }
      }
    }
    return '';
  }

  function findPlanFileInfo(fileId) {
    for (const plan of studyPlans) {
      for (const file of plan.files) {
        if (file.id === fileId) {
          return { plan, file, lesson: null };
        }

        if (Array.isArray(file.lessons)) {
          for (const group of file.lessons) {
            const lesson = group.items.find((item) => item.id === fileId);
            if (lesson) {
              return { plan, file, lesson };
            }
          }
        }
      }
    }
    return null;
  }

  function findScenarioInfo(id) {
    for (const material of studyMaterials) {
      for (const group of material.groups) {
        for (const item of group.items) {
          if (item.id === id) {
            return { material, group, item };
          }
        }
      }
    }
    return null;
  }

  function buildFrameUrl(material, scenarioId) {
    return `${material.path}?embed=1&scenario=${encodeURIComponent(scenarioId)}`;
  }

  function escapeHtml(value) {
    return String(value)
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;')
      .replaceAll("'", '&#39;');
  }

  function updateViewerHeader() {
    const currentInfo = findScenarioInfo(currentScenarioId);
    if (!toolbarTitleNode || !currentInfo) return;

    toolbarTitleNode.textContent = `${currentInfo.material.label} · ${currentInfo.group.label} · ${currentInfo.item.label}`;
  }


  function setVisualizationExpanded(isExpanded) {
    if (!visualizationToggle || !visualizationPanel) return;

    visualizationToggle.classList.toggle('is-active', isExpanded);
    visualizationToggle.setAttribute('aria-expanded', String(isExpanded));
    visualizationPanel.hidden = !isExpanded;

    if (learningPlanToggle && learningPlanPanel && isExpanded) {
      learningPlanToggle.classList.remove('is-active');
      learningPlanToggle.setAttribute('aria-expanded', 'false');
      learningPlanPanel.hidden = true;
    }

    if (!isExpanded) return;
    if (!studyMaterials.length) return;

    updateViewQuery('visualization');
    ensureInitialSelection();
    renderSelectors();
    updateViewerHeader();

    if (!hasLoadedVisualization) {
      loadScenario(true);
      hasLoadedVisualization = true;
    }
  }

  function setLearningPlanExpanded(isExpanded) {
    if (!learningPlanToggle || !learningPlanPanel) return;

    learningPlanToggle.classList.toggle('is-active', isExpanded);
    learningPlanToggle.setAttribute('aria-expanded', String(isExpanded));
    learningPlanPanel.hidden = !isExpanded;

    if (visualizationToggle && visualizationPanel && isExpanded) {
      visualizationToggle.classList.remove('is-active');
      visualizationToggle.setAttribute('aria-expanded', 'false');
      visualizationPanel.hidden = true;
    }

    if (!isExpanded) return;
    if (!studyPlans.length) return;

    updateViewQuery('plan');
    ensureInitialPlanSelection();
    renderPlanMenu();
    if (!hasLoadedPlan) {
      loadPlanFile(true);
      hasLoadedPlan = true;
    }
  }

  function renderSelectors() {
    if (!groupSelect || !scenarioSelect || !studyMaterials.length) return;

    const material = getCurrentMaterial();
    const group = getCurrentGroup(material);

    groupSelect.innerHTML = material.groups.map((item) => `
      <option value="${item.key}" ${item.key === currentGroupKey ? 'selected' : ''}>${escapeHtml(item.label)}</option>
    `).join('');
    groupSelect.value = group.key;

    scenarioSelect.innerHTML = group.items.map((item) => `
      <option value="${item.id}" ${item.id === currentScenarioId ? 'selected' : ''}>${escapeHtml(item.label)}</option>
    `).join('');
    scenarioSelect.value = currentScenarioId;
  }

  function renderPlanMenu() {
    if (!planMenuRoot || !studyPlans.length) return;

    if (!expandedPlanWeek) {
      expandedPlanWeek = findPlanWeekByFileId(currentPlanFileId);
    }

    planMenuRoot.innerHTML = studyPlans.map((plan) => {
      const file = plan.files[0];
      return Array.isArray(file.lessons) && file.lessons.length ? `
        <section class="study-plan-group">
          <div class="study-plan-list">
            ${file.lessons.map((group, index) => {
              const isExpandedWeek = group.group === expandedPlanWeek;
              const weekCountLabel = `${group.items.length}개 과제`;
              return `
                <section class="study-plan-group-block">
                  <button
                    class="study-plan-link study-plan-link-week-${index + 1} ${isExpandedWeek ? 'is-active' : ''}"
                    type="button"
                    data-plan-week="${escapeHtml(group.group)}"
                    data-plan-week-order="${index + 1}"
                    role="tab"
                    aria-selected="${String(isExpandedWeek)}"
                    aria-expanded="${String(isExpandedWeek)}"
                  >
                    <span class="study-plan-link-copy">
                      <span class="study-plan-link-title">${escapeHtml(group.group)}</span>
                      <span class="study-plan-link-description">${escapeHtml(weekCountLabel)}</span>
                    </span>
                  </button>
                  <div class="study-plan-submenu" ${isExpandedWeek ? '' : 'hidden'}>
                    <div class="study-plan-list study-plan-list-nested">
                      ${group.items.map((item) => `
                        <button
                          class="study-plan-sub-link ${item.id === currentPlanFileId ? 'is-active' : ''}"
                          type="button"
                          data-plan-file="${escapeHtml(item.id)}"
                          role="tab"
                          aria-selected="${String(item.id === currentPlanFileId)}"
                        >
                          <span class="study-plan-link-title">${escapeHtml(item.label)}</span>
                        </button>
                      `).join('')}
                    </div>
                  </div>
                </section>
              `;
            }).join('')}
          </div>
        </section>
      ` : '';
    }).join('');

    planMenuRoot.querySelectorAll('[data-plan-week]').forEach((button) => {
      button.addEventListener('click', () => {
        const weekLabel = button.getAttribute('data-plan-week');
        if (!weekLabel || weekLabel === expandedPlanWeek) return;

        const nextWeekGroup = studyPlans
          .flatMap((plan) => plan.files)
          .flatMap((file) => file.lessons || [])
          .find((group) => group.group === weekLabel);

        expandedPlanWeek = weekLabel;

        const nextLessonId = nextWeekGroup?.items?.[0]?.id;
        if (nextLessonId) {
          setPlanFile(nextLessonId);
          return;
        }

        renderPlanMenu();
      });
    });

    planMenuRoot.querySelectorAll('[data-plan-file]').forEach((button) => {
      button.addEventListener('click', () => {
        const nextFileId = button.getAttribute('data-plan-file');
        if (!nextFileId) return;
        setPlanFile(nextFileId);
      });
    });
  }

  function loadPlanFile(forceReload = false) {
    if (!planFrame) return;

    const currentInfo = findPlanFileInfo(currentPlanFileId);
    if (!currentInfo) return;

    const lessonId = currentInfo.file.lessons ? currentPlanFileId : '';
    const frameUrl = lessonId
      ? `${currentInfo.file.path}${currentInfo.file.path.includes('?') ? '&' : '?'}lesson=${encodeURIComponent(lessonId)}`
      : currentInfo.file.path;

    if (forceReload || planFrame.dataset.loadedFile !== currentPlanFileId) {
      planFrame.src = frameUrl;
      planFrame.dataset.loadedFile = currentPlanFileId;
    }
  }

  function setPlanFile(nextFileId, forceReload = false) {
    if (!nextFileId) return;

    const currentInfo = findPlanFileInfo(nextFileId);
    if (!currentInfo) return;

    currentPlanKey = currentInfo.plan.key;
    currentPlanFileId = currentInfo.lesson?.id || currentInfo.file.id;
    expandedPlanWeek = findPlanWeekByFileId(currentPlanFileId) || expandedPlanWeek;

    renderPlanMenu();
    if (!learningPlanPanel?.hidden) {
      loadPlanFile(forceReload);
    }
  }

  function setScenario(nextScenarioId, forceReload = false) {
    if (!nextScenarioId) return;

    const nextInfo = findScenarioInfo(nextScenarioId);
    if (!nextInfo) return;

    const isSameScenario = nextScenarioId === currentScenarioId;
    currentMaterialKey = nextInfo.material.key;
    currentGroupKey = nextInfo.group.key;
    currentScenarioId = nextScenarioId;

    renderSelectors();
    updateViewerHeader();
    if (!visualizationPanel.hidden) {
      loadScenario(forceReload || isSameScenario);
    }
  }

  function setGroup(nextGroupKey) {
    if (!nextGroupKey) return;

    const material = getCurrentMaterial();
    const nextGroup = material.groups.find((group) => group.key === nextGroupKey);
    if (!nextGroup) return;

    currentGroupKey = nextGroup.key;
    currentScenarioId = nextGroup.items[0]?.id || '';

    renderSelectors();
    updateViewerHeader();
    if (!visualizationPanel.hidden) {
      loadScenario(false);
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
          background: transparent !important;
          border: 0 !important;
        }
        .bar-fill {
          color: #ffffff !important;
          background: linear-gradient(180deg, #1f2937, #111827) !important;
          border: 1px solid #111827 !important;
          box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.16), 0 6px 14px rgba(17, 24, 39, 0.16) !important;
          font-size: 14px !important;
          line-height: 1 !important;
          align-items: center !important;
          padding: 8px 6px !important;
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
        restoreScrollAnchor();
      }
    } catch (error) {
      // same-origin access only
    }
  }

  function captureScrollAnchor() {
    if (visualizationPanel.hidden) return;
    pendingScrollAnchorTop = selectorCard.getBoundingClientRect().top;
    if (pendingScrollAnchorTimeout) clearTimeout(pendingScrollAnchorTimeout);
    pendingScrollAnchorTimeout = setTimeout(() => {
      pendingScrollAnchorTop = null;
      pendingScrollAnchorTimeout = null;
    }, 400);
  }

  function restoreScrollAnchor() {
    if (pendingScrollAnchorTop === null) return;
    const delta = selectorCard.getBoundingClientRect().top - pendingScrollAnchorTop;
    if (Math.abs(delta) > 1) {
      window.scrollBy(0, delta);
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
    if (!frame || !studyMaterials.length) return;

    const material = getCurrentMaterial();
    const samePageLoaded = frame.dataset.loadedMaterial === material.key;
    if (forceReload || !samePageLoaded) {
      frame.src = buildFrameUrl(material, currentScenarioId);
      frame.dataset.loadedMaterial = material.key;
      return;
    }

    try {
      if (frame.contentWindow?.visualizerApi?.setScenario) {
        frame.contentWindow.visualizerApi.setScenario(currentScenarioId);
        queueHeightSync();
      } else {
        frame.src = buildFrameUrl(material, currentScenarioId);
      }
    } catch (error) {
      frame.src = buildFrameUrl(material, currentScenarioId);
    }
  }

  function syncPlanFrameHeight() {
    if (!planFrame) return;

    try {
      const doc = planFrame.contentDocument || planFrame.contentWindow?.document;
      if (!doc) return;
      const nextHeight = Math.max(
        doc.body ? doc.body.scrollHeight : 0,
        doc.documentElement ? doc.documentElement.scrollHeight : 0,
      );
      if (nextHeight) {
        planFrame.style.height = `${nextHeight + 12}px`;
      }
    } catch (error) {
      // same-origin access only
    }
  }

  function toggleSubmenu(forceOpen) {
    if (!submenuToggle) return;
    const shouldOpen = typeof forceOpen === 'boolean'
      ? forceOpen
      : submenuToggle.getAttribute('aria-expanded') !== 'true';

    submenuToggle.setAttribute('aria-expanded', String(shouldOpen));
    submenuToggle.parentElement?.classList.toggle('is-open', shouldOpen);
  }

  function clearSubmenuCloseTimer() {
    if (!submenuCloseTimer) return;
    window.clearTimeout(submenuCloseTimer);
    submenuCloseTimer = null;
  }

  function openSubmenu() {
    clearSubmenuCloseTimer();
    toggleSubmenu(true);
  }

  function scheduleSubmenuClose() {
    clearSubmenuCloseTimer();
    submenuCloseTimer = window.setTimeout(() => {
      toggleSubmenu(false);
      submenuCloseTimer = null;
    }, 220);
  }

  if (groupSelect) {
    groupSelect.addEventListener('change', () => {
      captureScrollAnchor();
      setGroup(groupSelect.value);
    });
  }

  if (scenarioSelect) {
    scenarioSelect.addEventListener('change', () => {
      const nextScenarioId = scenarioSelect.value;
      if (!nextScenarioId) return;
      captureScrollAnchor();
      setScenario(nextScenarioId);
    });
  }

  if (visualizationToggle) {
    visualizationToggle.addEventListener('click', () => {
      const shouldExpand = visualizationToggle.getAttribute('aria-expanded') !== 'true';
      setVisualizationExpanded(shouldExpand);
      if (shouldExpand) toggleSubmenu(false);
    });
  }

  if (learningPlanToggle) {
    learningPlanToggle.addEventListener('click', () => {
      const shouldExpand = learningPlanToggle.getAttribute('aria-expanded') !== 'true';
      setLearningPlanExpanded(shouldExpand);
      if (shouldExpand) toggleSubmenu(false);
    });
  }

  if (submenuToggle && submenuPanel) {
    const submenuContainer = submenuToggle.parentElement;

    submenuToggle.addEventListener('click', () => {
      clearSubmenuCloseTimer();
      toggleSubmenu();
    });
    submenuContainer?.addEventListener('mouseenter', openSubmenu);
    submenuContainer?.addEventListener('mouseleave', scheduleSubmenuClose);
    submenuToggle.addEventListener('focus', openSubmenu);
    submenuPanel.addEventListener('focusin', openSubmenu);
    submenuPanel.addEventListener('focusout', () => {
      window.setTimeout(() => {
        if (!submenuContainer?.contains(document.activeElement)) {
          scheduleSubmenuClose();
        }
      }, 0);
    });

    document.addEventListener('click', (event) => {
      const target = event.target;
      if (!(target instanceof Node)) return;
      if (submenuContainer?.contains(target)) return;
      clearSubmenuCloseTimer();
      toggleSubmenu(false);
    });

    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') {
        clearSubmenuCloseTimer();
        toggleSubmenu(false);
      }
    });
  }

  if (frame) {
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
  }

  if (planFrame) {
    planFrame.addEventListener('load', () => {
      syncPlanFrameHeight();
      window.setTimeout(syncPlanFrameHeight, 80);
      window.setTimeout(syncPlanFrameHeight, 220);
    });
  }

  window.addEventListener('message', (event) => {
    const data = event.data;
    if (!data || typeof data !== 'object') return;

    if (data.type === 'kp-visualizer-height' && frame && typeof data.height === 'number' && data.height > 0) {
      frame.style.height = `${Math.round(data.height + 12)}px`;
      restoreScrollAnchor();
      return;
    }

    if (data.type === 'kp-study-docs-height' && planFrame && typeof data.height === 'number' && data.height > 0) {
      planFrame.style.height = `${Math.round(data.height + 12)}px`;
    }
  });

  window.addEventListener('resize', () => {
    if (hasLoadedPlan) syncPlanFrameHeight();
  });

  if (!studyMaterials.length && visualizationToggle) {
    visualizationToggle.hidden = true;
  }

  if (!studyPlans.length && learningPlanToggle) {
    learningPlanToggle.hidden = true;
  }

  const initialView = normalizeRequestedView(requestedView);
  const defaultView = studyMaterials.length ? 'visualization' : (studyPlans.length ? 'plan' : '');
  const activeInitialView = initialView || defaultView;

  if (activeInitialView === 'plan' && studyPlans.length) {
    setVisualizationExpanded(false);
    setLearningPlanExpanded(true);
  } else if (activeInitialView === 'visualization' && studyMaterials.length) {
    setLearningPlanExpanded(false);
    setVisualizationExpanded(true);
  } else {
    setVisualizationExpanded(false);
    setLearningPlanExpanded(false);
  }
})();
