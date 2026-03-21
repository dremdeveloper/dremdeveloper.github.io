(() => {
  const frame = document.getElementById('study-frame');
  const groupSelect = document.getElementById('study-group-select');
  const scenarioSelect = document.getElementById('study-scenario-select');
  const visualizationToggle = document.getElementById('study-visualization-toggle');
  const toolbarTitleNode = document.getElementById('study-toolbar-title');
  const selectorCard = document.getElementById('study-selector-card');
  const submenuPanel = document.getElementById('study-top-submenu');
  const submenuToggle = document.getElementById('study-nav-toggle');

  if (!frame || !groupSelect || !scenarioSelect || !visualizationToggle || !selectorCard) return;

  const studyMaterials = Array.isArray(window.siteData?.studyMaterials)
    ? window.siteData.studyMaterials.filter((material) => Array.isArray(material.groups) && material.groups.length)
    : [];

  if (!studyMaterials.length) return;

  let currentMaterialKey = studyMaterials[0].key;
  let currentGroupKey = studyMaterials[0].groups[0]?.key || '';
  let currentScenarioId = studyMaterials[0].groups[0]?.items[0]?.id || '';
  let resizeObserver = null;
  let mutationObserver = null;
  let rafId = null;
  let boundFrameWindow = null;

  function getCurrentMaterial() {
    return studyMaterials.find((item) => item.key === currentMaterialKey) || studyMaterials[0];
  }

  function getCurrentGroup(material = getCurrentMaterial()) {
    return material.groups.find((group) => group.key === currentGroupKey) || material.groups[0];
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

  function syncCurrentSelection() {
    const material = getCurrentMaterial();
    const group = getCurrentGroup(material);
    const groupItems = group?.items || [];

    if (!group) return;

    currentGroupKey = group.key;
    if (!groupItems.some((item) => item.id === currentScenarioId)) {
      currentScenarioId = groupItems[0]?.id || '';
    }
  }


  function updateViewerHeader() {
    const currentInfo = findScenarioInfo(currentScenarioId);
    if (!toolbarTitleNode || !currentInfo) return;

    toolbarTitleNode.textContent = `${currentInfo.material.label} · ${currentInfo.group.label} · ${currentInfo.item.label}`;
  }

  function setVisualizationExpanded(isExpanded) {
    visualizationToggle.classList.toggle('is-active', isExpanded);
    visualizationToggle.setAttribute('aria-expanded', String(isExpanded));
    selectorCard.hidden = !isExpanded;
  }

  function renderSelectors() {
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
    loadScenario(forceReload || isSameScenario);
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
    loadScenario(false);
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

  function toggleSubmenu(forceOpen) {
    if (!submenuPanel || !submenuToggle) return;
    const shouldOpen = typeof forceOpen === 'boolean'
      ? forceOpen
      : submenuToggle.getAttribute('aria-expanded') !== 'true';

    submenuToggle.setAttribute('aria-expanded', String(shouldOpen));
    submenuPanel.hidden = !shouldOpen;
  }

  groupSelect.addEventListener('change', () => {
    setGroup(groupSelect.value);
  });

  scenarioSelect.addEventListener('change', () => {
    const nextScenarioId = scenarioSelect.value;
    if (!nextScenarioId) return;
    setScenario(nextScenarioId);
  });

  visualizationToggle.addEventListener('click', () => {
    const shouldExpand = visualizationToggle.getAttribute('aria-expanded') !== 'true';
    setVisualizationExpanded(shouldExpand);
    if (shouldExpand) toggleSubmenu(false);
  });

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

  syncCurrentSelection();
  setVisualizationExpanded(false);
  renderSelectors();
  renderQuickChips();
  updateViewerHeader();
  loadScenario(true);
})();
