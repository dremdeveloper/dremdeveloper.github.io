(() => {
  const config = window.siteData?.articles;
  if (!config) return;

  const listNode = document.getElementById('article-list');
  const countNode = document.getElementById('article-count-label');
  const statusNode = document.getElementById('article-status');
  const viewerNode = document.getElementById('article-viewer');
  const sidebarNode = document.querySelector('.article-sidebar');
  const sidebarToggle = document.getElementById('article-sidebar-toggle');
  const sidebarCurrentNode = document.getElementById('article-sidebar-current');
  const searchInput = document.getElementById('article-search-input');
  const searchResultsNode = document.getElementById('article-search-results');
  const viewerCardNode = document.querySelector('.article-viewer-card');
  const mobileMediaQuery = window.matchMedia('(max-width: 860px)');
  const SIDEBAR_CURRENT_LABEL = '아티클 선택';

  if (!listNode || !countNode || !statusNode || !viewerNode) return;

  const searchParams = new URLSearchParams(window.location.search);
  const requestedFile = decodeURIComponent(searchParams.get('file') || '');
  const owner = config.owner;
  const repo = config.repo;
  const branch = config.branch || 'main';
  const articlesPath = config.articlesPath || 'articles';
  const defaultFile = config.defaultFile || '';
  const configuredFiles = Array.isArray(config.files) ? config.files : [];
  const ignoredFiles = new Set(
    (Array.isArray(config.ignoredFiles) ? config.ignoredFiles : [])
      .map((item) => normalizeFileKey(item))
      .filter(Boolean)
  );
  const configuredCategoryOrder = Array.isArray(config.categoryOrder) ? config.categoryOrder : [];
  const pinnedFirstFiles = Array.isArray(config.pinnedFirstFiles)
    ? config.pinnedFirstFiles.map((item) => normalizeFileKey(item)).filter(Boolean)
    : [];
  const pinnedFirstFileOrder = new Map(
    pinnedFirstFiles.map((file, index) => [file, index])
  );
  const categoryLabelMap = {
    'AI논문': 'AI 논문',
    'AI 논문': 'AI 논문',
    '생각정리': '생각',
    '생각 정리': '생각',
    '생각': '생각',
    '지식정리': '지식 정리',
    '유용한 지식 및 팁': '지식 정리',
    '지식 정리': '지식 정리',
    '트러블슈팅': '트러블 슈팅',
    '트러블 슈팅': '트러블 슈팅'
  };
  const requestedCategory = normalizeCategoryLabel(decodeURIComponent(searchParams.get('category') || ''));

  const contentsApiBaseUrl = `https://api.github.com/repos/${owner}/${repo}/contents`;
  const rawBaseUrl = `https://raw.githubusercontent.com/${owner}/${repo}/${encodeURIComponent(branch)}`;
  const mathJaxSrc = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js';
  const articleIndexCacheKey = `article-index-cache:${owner}/${repo}/${branch}/${articlesPath}`;
  const articleIndexCacheTtlMs = 1000 * 60 * 60 * 6;
  let articleFiles = [];
  let allArticleFiles = [];
  let currentFile = '';
  let mathJaxLoader = null;
  let searchMatches = [];
  let currentArticleTitle = '';

  window.MathJax = window.MathJax || {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']],
      processEscapes: true,
      tags: 'ams'
    },
    options: {
      skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
    },
    svg: { fontCache: 'global' }
  };

  function escapeHtml(value) {
    return String(value)
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;')
      .replaceAll("'", '&#39;');
  }

  function slugToTitle(fileName) {
    return fileName
      .replace(/\.md$/i, '')
      .replace(/^\d{4}-\d{2}-\d{2}[-_]?/, '')
      .replace(/[-_]+/g, ' ')
      .trim() || fileName;
  }

  function extractTitle(markdown, fileName) {
    const heading = markdown.match(/^#\s+(.+)$/m);
    return heading?.[1]?.trim() || slugToTitle(fileName);
  }

  function encodePath(path) {
    return String(path || '')
      .split('/')
      .filter(Boolean)
      .map((segment) => encodeURIComponent(segment))
      .join('/');
  }

  function normalizeFileKey(value) {
    return String(value || '')
      .replace(/^\/+|\/+$/g, '')
      .trim();
  }

  function normalizeCategoryLabel(category) {
    const normalizedCategory = String(category || '').trim();
    return categoryLabelMap[normalizedCategory] || normalizedCategory || '미분류';
  }

  function buildArticleEntry(item) {
    const fileKey = normalizeFileKey(item.path || item.name || item.file || '');
    const safeFileName = fileKey.split('/').pop() || '';
    const categoryParts = fileKey.split('/').slice(1, -1);
    const category = normalizeCategoryLabel(item.category || categoryParts.join(' / ') || '미분류');
    const encodedPath = encodePath(fileKey);
    const localPath = item.localPath || encodeURI(fileKey);
    const githubRawUrl = item.githubRawUrl || item.download_url || `${rawBaseUrl}/${encodedPath}`;

    return {
      name: fileKey,
      fileName: safeFileName,
      title: item.title || slugToTitle(safeFileName),
      category,
      localPath,
      downloadUrl: githubRawUrl
    };
  }

  function setSidebarState(isOpen) {
    if (!sidebarNode || !sidebarToggle) return;
    sidebarNode.classList.toggle('is-mobile-open', isOpen);
    sidebarToggle.setAttribute('aria-expanded', String(isOpen));
  }

  function closeSidebarOnMobile() {
    if (!mobileMediaQuery.matches) return;
    setSidebarState(false);
  }

  function updateSidebarCurrentLabel() {
    if (!sidebarCurrentNode) return;
    sidebarCurrentNode.textContent = currentArticleTitle
      || (requestedCategory ? `${requestedCategory} 목록` : SIDEBAR_CURRENT_LABEL);
  }

  function focusViewerOnMobile() {
    if (!mobileMediaQuery.matches || !viewerCardNode) return;

    const topbarHeight = document.querySelector('.topbar')?.offsetHeight || 0;
    const targetTop = window.scrollY + viewerCardNode.getBoundingClientRect().top - topbarHeight - 16;
    window.scrollTo({ top: Math.max(0, targetTop), behavior: 'smooth' });
  }

  function applyCategoryFilter(files) {
    if (!requestedCategory) return files;
    return files.filter((item) => normalizeCategoryLabel(item.category) === requestedCategory);
  }

  function updateCountLabel() {
    const prefix = requestedCategory ? `${requestedCategory} · ` : '';
    countNode.textContent = `${prefix}${articleFiles.length}개의 md 파일`;
  }

  function setArticleFiles(files) {
    allArticleFiles = files
      .filter((item) => {
        const fileKey = normalizeFileKey(item?.path || item?.name);
        return fileKey && !ignoredFiles.has(fileKey);
      })
      .map(buildArticleEntry)
      .sort((a, b) => {
        const aPinnedOrder = pinnedFirstFileOrder.get(normalizeFileKey(a.name));
        const bPinnedOrder = pinnedFirstFileOrder.get(normalizeFileKey(b.name));
        const aPinned = Number.isInteger(aPinnedOrder);
        const bPinned = Number.isInteger(bPinnedOrder);

        if (aPinned && bPinned) return aPinnedOrder - bPinnedOrder;
        if (aPinned) return -1;
        if (bPinned) return 1;
        return b.name.localeCompare(a.name, 'ko');
      });

    articleFiles = applyCategoryFilter(allArticleFiles);
    updateCountLabel();
    updateSidebarCurrentLabel();
    renderList();
    updateSearchResults();
  }

  function mergeArticleFiles(primaryFiles, secondaryFiles = []) {
    const merged = new Map();

    primaryFiles.forEach((item) => {
      const key = normalizeFileKey(item?.path || item?.name);
      if (!key) return;
      merged.set(key, { ...item, path: key, name: key });
    });

    secondaryFiles.forEach((item) => {
      const key = normalizeFileKey(item?.path || item?.name);
      if (!key) return;

      const existing = merged.get(key);
      if (existing) {
        merged.set(key, { ...item, ...existing, path: key, name: key });
        return;
      }

      merged.set(key, { ...item, path: key, name: key });
    });

    return Array.from(merged.values());
  }

  function loadCachedArticleFiles() {
    try {
      const cachedValue = window.localStorage?.getItem(articleIndexCacheKey);
      if (!cachedValue) return [];

      const parsed = JSON.parse(cachedValue);
      const cachedAt = Number(parsed?.cachedAt || 0);
      const files = Array.isArray(parsed?.files) ? parsed.files : [];
      if (!cachedAt || Date.now() - cachedAt > articleIndexCacheTtlMs) {
        window.localStorage?.removeItem(articleIndexCacheKey);
        return [];
      }

      return files;
    } catch (error) {
      console.warn('Unable to read cached article index.', error);
      return [];
    }
  }

  function saveCachedArticleFiles(files) {
    try {
      if (!Array.isArray(files) || !files.length) return;
      window.localStorage?.setItem(
        articleIndexCacheKey,
        JSON.stringify({
          cachedAt: Date.now(),
          files
        })
      );
    } catch (error) {
      console.warn('Unable to cache article index.', error);
    }
  }

  function areSameFileSets(leftFiles, rightFiles) {
    const leftKeys = leftFiles
      .map((item) => normalizeFileKey(item?.path || item?.name))
      .filter(Boolean)
      .sort();
    const rightKeys = rightFiles
      .map((item) => normalizeFileKey(item?.path || item?.name))
      .filter(Boolean)
      .sort();

    if (leftKeys.length !== rightKeys.length) return false;
    return leftKeys.every((value, index) => value === rightKeys[index]);
  }

  async function fetchDirectoryEntries(path = articlesPath) {
    const encodedPath = encodePath(path);
    const response = await fetch(`${contentsApiBaseUrl}/${encodedPath}?ref=${encodeURIComponent(branch)}`, {
      headers: {
        Accept: 'application/vnd.github+json'
      }
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const items = await response.json();
    const files = [];

    for (const item of items) {
      if (item.type === 'dir') {
        files.push(...await fetchDirectoryEntries(item.path));
        continue;
      }

      if (item.type === 'file' && /\.md$/i.test(item.name)) {
        files.push(item);
      }
    }

    return files;
  }

  function setStatus(message, isError = false) {
    statusNode.hidden = false;
    statusNode.textContent = message;
    statusNode.classList.toggle('is-error', isError);
  }

  function upsertMeta(selector, attributeName, value) {
    const node = document.head.querySelector(selector);
    if (node) {
      node.setAttribute(attributeName, value);
      return node;
    }

    const meta = document.createElement('meta');
    const match = selector.match(/\[(name|property)="([^"]+)"\]/);
    if (match) {
      meta.setAttribute(match[1], match[2]);
    }
    meta.setAttribute(attributeName, value);
    document.head.appendChild(meta);
    return meta;
  }

  function updateArticleSeo(title) {
    const articleTitle = title || 'Article';
    document.title = `${articleTitle} | DremDeveloper`;

    const metaDesc = document.querySelector('meta[name="description"]');
    if (metaDesc) {
      metaDesc.setAttribute('content', articleTitle);
    }

    upsertMeta('meta[property="og:title"]', 'content', articleTitle);
    upsertMeta('meta[property="og:description"]', 'content', articleTitle);
    upsertMeta('meta[name="twitter:title"]', 'content', articleTitle);
    upsertMeta('meta[name="twitter:description"]', 'content', articleTitle);
    upsertMeta('meta[property="og:url"]', 'content', window.location.href);

    const canonicalLink = document.querySelector('link[rel="canonical"]') || document.createElement('link');
    canonicalLink.setAttribute('rel', 'canonical');
    canonicalLink.setAttribute('href', window.location.href);
    if (!canonicalLink.parentNode) {
      document.head.appendChild(canonicalLink);
    }

    document.querySelectorAll('script[data-article-structured-data="true"]').forEach((node) => node.remove());

    const structuredData = {
      '@context': 'https://schema.org',
      '@type': 'Article',
      headline: articleTitle,
      author: {
        '@type': 'Person',
        name: '박경록'
      }
    };

    const script = document.createElement('script');
    script.type = 'application/ld+json';
    script.dataset.articleStructuredData = 'true';
    script.text = JSON.stringify(structuredData);
    document.head.appendChild(script);
  }

  function stripFrontMatter(markdown) {
    if (!markdown.startsWith('---')) return markdown;
    const match = markdown.match(/^---\n[\s\S]*?\n---\n?/);
    return match ? markdown.slice(match[0].length) : markdown;
  }

  function stripMathJaxSnippets(markdown) {
    return markdown
      .replace(/<script\b[^>]*>[\s\S]*?<\/script>\s*/gi, '')
      .trim();
  }

  function sanitizeMarkdown(markdown) {
    return stripMathJaxSnippets(stripFrontMatter(markdown));
  }

  function ensureMathJax() {
    if (window.MathJax?.typesetPromise) {
      return Promise.resolve(window.MathJax);
    }

    if (mathJaxLoader) return mathJaxLoader;

    mathJaxLoader = new Promise((resolve, reject) => {
      const existingScript = document.querySelector(`script[src="${mathJaxSrc}"]`);
      if (existingScript) {
        existingScript.addEventListener('load', () => resolve(window.MathJax), { once: true });
        existingScript.addEventListener('error', reject, { once: true });
        return;
      }

      const script = document.createElement('script');
      script.src = mathJaxSrc;
      script.defer = true;
      script.addEventListener('load', () => resolve(window.MathJax), { once: true });
      script.addEventListener('error', reject, { once: true });
      document.head.appendChild(script);
    }).catch((error) => {
      mathJaxLoader = null;
      throw error;
    });

    return mathJaxLoader;
  }

  async function typesetMath() {
    try {
      const mathJax = await ensureMathJax();
      if (!mathJax?.typesetPromise) return;
      if (typeof mathJax.typesetClear === 'function') {
        mathJax.typesetClear([viewerNode]);
      }
      await mathJax.typesetPromise([viewerNode]);
    } catch (error) {
      console.error('MathJax failed to load.', error);
    }
  }

  function updateQuery(fileName) {
    const url = new URL(window.location.href);
    if (fileName) {
      url.searchParams.set('file', fileName);
    } else {
      url.searchParams.delete('file');
    }
    window.history.replaceState({}, '', url);
  }

  marked.setOptions({
    gfm: true,
    breaks: true
  });

  function protectMathSegments(markdown) {
    const placeholders = [];
    let placeholderIndex = 0;

    const createPlaceholder = (mathSource, isBlock = false) => {
      const placeholderId = String(placeholderIndex);
      const token = `CODEXMATHPLACEHOLDER${isBlock ? 'BLOCK' : 'INLINE'}${placeholderId}TOKEN`;
      placeholders.push({ id: placeholderId, token, mathSource, isBlock });
      placeholderIndex += 1;
      return isBlock ? `\n\n${token}\n\n` : token;
    };

    const processSegment = (segment) => segment
      .replace(/\$\$[\s\S]+?\$\$/g, (match) => createPlaceholder(match, true))
      .replace(/\\\[[\s\S]+?\\\]/g, (match) => createPlaceholder(match, true))
      .replace(/\\\([\s\S]+?\\\)/g, (match) => createPlaceholder(match, false));

    const protectedMarkdown = segmentMarkdownByCodeFence(markdown)
      .map((segment) => (segment.type === 'code' ? segment.content : processSegment(segment.content)))
      .join('');

    return { protectedMarkdown, placeholders };
  }

  function segmentMarkdownByCodeFence(markdown) {
    const fencePattern = /(```[\s\S]*?```|~~~[\s\S]*?~~~)/g;
    const segments = [];
    let lastIndex = 0;
    let match;

    while ((match = fencePattern.exec(markdown)) !== null) {
      if (match.index > lastIndex) {
        segments.push({ type: 'text', content: markdown.slice(lastIndex, match.index) });
      }
      segments.push({ type: 'code', content: match[0] });
      lastIndex = match.index + match[0].length;
    }

    if (lastIndex < markdown.length) {
      segments.push({ type: 'text', content: markdown.slice(lastIndex) });
    }

    return segments;
  }

  function normalizeStandaloneMathBlocks(markdown) {
    const isStandaloneMathLine = (line) => {
      const trimmed = line.trim();
      if (!trimmed) return false;
      if (/^(#{1,6}\s|[-*+]\s|>\s|\d+\.\s|```|~~~|!\[|<)/.test(trimmed)) return false;
      if (/[가-힣]/.test(trimmed)) return false;

      const hasMathSignal = /\\[A-Za-z]+|[_^][A-Za-z{(]|[{}]|=/.test(trimmed);
      if (!hasMathSignal) return false;

      const plainText = trimmed
        .replace(/\\[A-Za-z]+/g, '')
        .replace(/[{}[\]()<>=_^\\|,&;.:+\-/*~`]/g, '')
        .replace(/\d+/g, '')
        .replace(/\s+/g, '');

      return plainText.length <= 6;
    };

    const normalizeTextSegment = (segment) => {
      const lines = segment.split('\n');
      const normalized = [];
      let index = 0;
      let inDisplayMath = false;

      while (index < lines.length) {
        const line = lines[index];
        const trimmed = line.trim();

        if (/^(\$\$|\\\[)\s*$/.test(trimmed)) {
          inDisplayMath = true;
          normalized.push(line);
          index += 1;
          continue;
        }

        if (inDisplayMath) {
          normalized.push(line);
          if (/^(\$\$|\\\])\s*$/.test(trimmed)) {
            inDisplayMath = false;
          }
          index += 1;
          continue;
        }

        if (!isStandaloneMathLine(line)) {
          normalized.push(line);
          index += 1;
          continue;
        }

        const blockLines = [];
        let cursor = index;
        while (cursor < lines.length && isStandaloneMathLine(lines[cursor])) {
          blockLines.push(lines[cursor].trim());
          cursor += 1;
        }

        const previousLine = normalized.length ? normalized[normalized.length - 1].trim() : '';
        const nextLine = cursor < lines.length ? lines[cursor].trim() : '';
        const shouldWrap =
          blockLines.length >= 2 ||
          (blockLines.length === 1 && !previousLine && !nextLine);

        if (shouldWrap) {
          normalized.push('\\[');
          normalized.push(...blockLines);
          normalized.push('\\]');
        } else {
          normalized.push(...blockLines);
        }
        index = cursor;
      }

      return normalized.join('\n');
    };

    return segmentMarkdownByCodeFence(markdown)
      .map((segment) => (segment.type === 'code' ? segment.content : normalizeTextSegment(segment.content)))
      .join('');
  }

  function restoreMathPlaceholdersInHtml(html, placeholders) {
    return placeholders.reduce((restoredHtml, entry) => {
      const placeholderMarkup = entry.isBlock
        ? `<div class="math-placeholder" data-math-placeholder="${entry.id}"></div>`
        : `<span class="math-placeholder" data-math-placeholder="${entry.id}"></span>`;

      if (entry.isBlock) {
        const wrappedTokenPattern = new RegExp(`<p>\\s*${entry.token}\\s*<\\/p>`, 'g');
        return restoredHtml
          .replace(wrappedTokenPattern, placeholderMarkup)
          .replaceAll(entry.token, placeholderMarkup);
      }

      return restoredHtml.replaceAll(entry.token, placeholderMarkup);
    }, html);
  }

  function restoreMathPlaceholders(rootNode, placeholders) {
    if (!rootNode || !placeholders.length) return;

    const placeholderMap = new Map(
      placeholders.map((entry) => [entry.id, entry])
    );

    rootNode.querySelectorAll('[data-math-placeholder]').forEach((node) => {
      const entry = placeholderMap.get(node.getAttribute('data-math-placeholder'));
      if (!entry) return;
      const mathNode = document.createElement(entry.isBlock ? 'div' : 'span');
      mathNode.className = entry.isBlock ? 'math-render-source math-render-source-block' : 'math-render-source';
      mathNode.textContent = entry.mathSource;
      node.replaceWith(mathNode);
    });
  }

  function renderMarkdown(md) {
    const normalizedMarkdown = normalizeStandaloneMathBlocks(md);
    const { protectedMarkdown, placeholders } = protectMathSegments(normalizedMarkdown);
    const sanitizedHtml = DOMPurify.sanitize(marked.parse(protectedMarkdown));
    return { html: restoreMathPlaceholdersInHtml(sanitizedHtml, placeholders), placeholders };
  }

  function setSearchResultsVisibility(isVisible) {
    if (!searchResultsNode) return;
    searchResultsNode.hidden = !isVisible;
  }

  function getSearchMatches(query) {
    const normalizedQuery = String(query || '').trim().toLowerCase();
    if (!normalizedQuery) return [];

    return articleFiles
      .filter((entry) => {
        const haystack = `${entry.title} ${entry.category} ${entry.fileName}`.toLowerCase();
        return haystack.includes(normalizedQuery);
      })
      .slice(0, 8);
  }

  function renderSearchResults() {
    if (!searchResultsNode) return;

    if (!searchMatches.length) {
      searchResultsNode.innerHTML = '<div class="search-item" aria-disabled="true"><strong>검색 결과가 없습니다</strong><span>다른 키워드로 다시 찾아보세요.</span></div>';
      setSearchResultsVisibility(Boolean(searchInput?.value.trim()));
      return;
    }

    searchResultsNode.innerHTML = '';
    searchMatches.forEach((file) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = `search-item${file.name === currentFile ? ' is-active' : ''}`;
      button.innerHTML = `
        <strong>${escapeHtml(file.title)}</strong>
        <span>${escapeHtml(file.category)}</span>
      `;
      button.addEventListener('click', () => {
        loadArticle(file.name, { focusContent: true });
        if (searchInput) {
          searchInput.value = '';
        }
        searchMatches = [];
        renderSearchResults();
        setSearchResultsVisibility(false);
      });
      searchResultsNode.appendChild(button);
    });

    setSearchResultsVisibility(true);
  }

  function updateSearchResults() {
    if (!searchInput || !searchResultsNode) return;
    searchMatches = getSearchMatches(searchInput.value);
    renderSearchResults();
  }

  function resolveInitialFileName() {
    return articleFiles.find((file) => file.name === requestedFile)?.name
      || articleFiles.find((file) => file.name === defaultFile)?.name
      || articleFiles[0]?.name
      || allArticleFiles.find((file) => file.name === requestedFile)?.name
      || '';
  }

  function renderList() {
    listNode.innerHTML = '';
    updateSidebarCurrentLabel();

    if (!articleFiles.length) {
      const emptyState = document.createElement('div');
      emptyState.className = 'menu-empty-state';
      emptyState.textContent = requestedCategory
        ? `${requestedCategory} 카테고리에 표시할 md 파일이 없습니다.`
        : '표시할 md 파일이 없습니다.';
      listNode.appendChild(emptyState);
      return;
    }

    const groupedFiles = articleFiles.reduce((groups, file) => {
      const key = file.category || '미분류';
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(file);
      return groups;
    }, new Map());

    const orderedCategories = [
      ...configuredCategoryOrder,
      ...Array.from(groupedFiles.keys()).filter((category) => !configuredCategoryOrder.includes(category))
    ].filter((category) => (groupedFiles.get(category) || []).length > 0);

    let articleIndex = 0;

    orderedCategories.forEach((category) => {
      const files = groupedFiles.get(category) || [];
      const section = document.createElement('section');
      section.className = 'article-category-group';

      const heading = document.createElement('h3');
      heading.className = 'article-category-title';
      heading.textContent = category;
      section.appendChild(heading);

      const groupList = document.createElement('div');
      groupList.className = 'article-category-items';

      files.forEach((file) => {
        articleIndex += 1;
        const button = document.createElement('button');
        button.type = 'button';
        button.className = `menu-item${file.name === currentFile ? ' is-active' : ''}`;
        button.innerHTML = `
          <span class="menu-item-index">${String(articleIndex).padStart(2, '0')}</span>
          <span class="menu-item-copy">
            <strong>${escapeHtml(file.title)}</strong>
          </span>
        `;
        button.addEventListener('click', () => {
          loadArticle(file.name, { focusContent: true });
        });
        groupList.appendChild(button);
      });

      section.appendChild(groupList);
      listNode.appendChild(section);
    });
  }

  async function fetchMarkdown(file) {
    const targets = [file.localPath, file.downloadUrl].filter(Boolean);
    let lastError = null;

    for (const target of targets) {
      try {
        const response = await fetch(target, { headers: { Accept: 'text/plain' } });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.text();
      } catch (error) {
        lastError = error;
      }
    }

    throw lastError || new Error('Unable to fetch markdown');
  }

  async function loadArticle(fileName, { focusContent = false } = {}) {
    const file = allArticleFiles.find((entry) => entry.name === fileName);
    if (!file) return;

    currentFile = file.name;
    currentArticleTitle = file.title || '';
    renderList();
    updateSearchResults();
    updateSidebarCurrentLabel();
    setStatus('아티클 내용을 불러오는 중입니다.');
    updateQuery(file.name);

    try {
      const markdown = await fetchMarkdown(file);
      const sanitizedMarkdown = sanitizeMarkdown(markdown);
      const renderedTitle = extractTitle(sanitizedMarkdown, file.fileName || file.name);
      const { html, placeholders } = renderMarkdown(sanitizedMarkdown);
      file.title = renderedTitle;
      currentArticleTitle = renderedTitle;
      renderList();
      updateSearchResults();
      updateSidebarCurrentLabel();
      viewerNode.innerHTML = `<div id="article-content" class="article-container">${html}</div>`;
      restoreMathPlaceholders(viewerNode, placeholders);
      viewerNode.hidden = false;
      statusNode.hidden = true;
      const title = document.querySelector('#article-viewer h1')?.innerText || renderedTitle || 'Article';
      updateArticleSeo(title);
      closeSidebarOnMobile();
      if (focusContent) {
        focusViewerOnMobile();
      }
      await typesetMath();
    } catch (error) {
      currentArticleTitle = requestedCategory ? `${requestedCategory} 목록` : '';
      updateSidebarCurrentLabel();
      setStatus('md 파일을 불러오지 못했습니다. 배포된 articles 경로와 파일명을 확인해 주세요.', true);
      viewerNode.hidden = true;
      console.error(error);
    }
  }

  async function loadArticleIndex() {
    const cachedFiles = loadCachedArticleFiles();
    const bootstrapFiles = mergeArticleFiles(configuredFiles, cachedFiles);
    const hasBootstrapFiles = bootstrapFiles.length > 0;

    if (hasBootstrapFiles) {
      setArticleFiles(bootstrapFiles);

      if (!articleFiles.length) {
        const emptyMessage = requestedCategory
          ? `${requestedCategory} 카테고리에는 아직 md 파일이 없습니다.`
          : 'articles 폴더에 아직 표시할 md 파일이 없습니다.';
        setStatus(emptyMessage);
        viewerNode.hidden = true;
      } else {
        setStatus('아티클 목록을 먼저 표시했습니다. 최신 목록을 백그라운드에서 확인하는 중입니다.');
        const initialFile = resolveInitialFileName();
        if (initialFile) {
          loadArticle(initialFile);
        }
      }
    } else {
      setStatus('GitHub 저장소에서 article 목록을 불러오는 중입니다.');
    }

    try {
      const items = await fetchDirectoryEntries();
      const mergedFiles = mergeArticleFiles(configuredFiles, items);
      saveCachedArticleFiles(mergedFiles);

      const shouldRefreshList = !hasBootstrapFiles || !areSameFileSets(bootstrapFiles, mergedFiles);
      if (shouldRefreshList) {
        setArticleFiles(mergedFiles);
      }

      if (!articleFiles.length) {
        const emptyMessage = requestedCategory
          ? `${requestedCategory} 카테고리에는 아직 md 파일이 없습니다.`
          : 'articles 폴더에 md 파일이 아직 없습니다. 새 파일을 올리면 여기에 자동으로 나타납니다.';
        setStatus(emptyMessage);
        viewerNode.hidden = true;
        return;
      }

      const initialFile = resolveInitialFileName();

      if (!currentFile || !allArticleFiles.some((file) => file.name === currentFile)) {
        loadArticle(initialFile);
      }
    } catch (error) {
      if (hasBootstrapFiles) {
        setStatus('최신 목록 확인에 실패해 미리 준비된 article 목록을 계속 표시합니다.', true);
        if (!currentFile) {
          const initialFile = resolveInitialFileName();
          if (initialFile) loadArticle(initialFile);
        }
        return;
      }

      if (configuredFiles.length) {
        setArticleFiles(configuredFiles);
        setStatus('GitHub 목록을 불러오지 못해 assets/data.js에 등록된 article 목록만 표시합니다.', true);

        const initialFile = resolveInitialFileName();

        if (initialFile) loadArticle(initialFile);
        return;
      }

      countNode.textContent = '목록을 불러오지 못했습니다';
      setStatus('article 목록을 불러오지 못했습니다. assets/data.js의 article 파일 목록 또는 GitHub 저장소 설정을 확인해 주세요.', true);
      console.error(error);
    }
  }

  sidebarToggle?.addEventListener('click', () => {
    const isOpen = sidebarToggle.getAttribute('aria-expanded') !== 'true';
    setSidebarState(isOpen);
  });

  searchInput?.addEventListener('input', () => {
    updateSearchResults();
  });

  searchInput?.addEventListener('focus', () => {
    updateSearchResults();
  });

  searchInput?.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      setSearchResultsVisibility(false);
      searchInput.blur();
    }
  });

  document.addEventListener('click', (event) => {
    if (!searchResultsNode || !searchInput) return;
    if (searchResultsNode.contains(event.target) || searchInput.contains(event.target)) return;
    setSearchResultsVisibility(false);
  });

  mobileMediaQuery.addEventListener?.('change', (event) => {
    if (!event.matches) {
      setSidebarState(false);
    }
  });

  updateSidebarCurrentLabel();
  loadArticleIndex();
})();
