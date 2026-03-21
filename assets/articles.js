(() => {
  const config = window.siteData?.articles;
  if (!config) return;

  const listNode = document.getElementById('article-list');
  const countNode = document.getElementById('article-count-label');
  const statusNode = document.getElementById('article-status');
  const viewerNode = document.getElementById('article-viewer');

  if (!listNode || !countNode || !statusNode || !viewerNode) return;

  const searchParams = new URLSearchParams(window.location.search);
  const requestedFile = decodeURIComponent(searchParams.get('file') || '');
  const owner = config.owner;
  const repo = config.repo;
  const branch = config.branch || 'main';
  const articlesPath = config.articlesPath || 'articles';
  const defaultFile = config.defaultFile || '';
  const configuredFiles = Array.isArray(config.files) ? config.files : [];
  const configuredCategoryOrder = Array.isArray(config.categoryOrder) ? config.categoryOrder : [];

  const contentsApiBaseUrl = `https://api.github.com/repos/${owner}/${repo}/contents`;
  const rawBaseUrl = `https://raw.githubusercontent.com/${owner}/${repo}/${encodeURIComponent(branch)}`;
  const mathJaxSrc = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js';
  let articleFiles = [];
  let currentFile = '';
  let mathJaxLoader = null;

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

  function buildArticleEntry(item) {
    const fileKey = normalizeFileKey(item.path || item.name || item.file || '');
    const safeFileName = fileKey.split('/').pop() || '';
    const categoryParts = fileKey.split('/').slice(1, -1);
    const category = item.category || categoryParts.join(' / ') || '미분류';
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

  function setArticleFiles(files) {
    articleFiles = files
      .filter((item) => normalizeFileKey(item?.path || item?.name))
      .map(buildArticleEntry)
      .sort((a, b) => b.name.localeCompare(a.name, 'ko'));

    countNode.textContent = `${articleFiles.length}개의 md 파일`;
    renderList();
  }

  function mergeArticleFiles(primaryFiles, secondaryFiles = []) {
    const merged = new Map();

    [...primaryFiles, ...secondaryFiles].forEach((item) => {
      const key = normalizeFileKey(item?.path || item?.name);
      if (!key) return;
      const existing = merged.get(key) || {};
      merged.set(key, { ...item, ...existing, path: key, name: key });
    });

    return Array.from(merged.values());
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

  function showViewer(html) {
    viewerNode.innerHTML = html;
    viewerNode.hidden = false;
    statusNode.hidden = true;
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

  function renderInline(text) {
    return escapeHtml(text)
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/\*([^*]+)\*/g, '<em>$1</em>')
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  }

  function renderMarkdown(markdown) {
    const lines = markdown.replace(/\r\n/g, '\n').split('\n');
    const html = [];
    let inCodeBlock = false;
    let codeLines = [];
    let listType = '';
    let quoteLines = [];
    let paragraphLines = [];

    const flushParagraph = () => {
      if (!paragraphLines.length) return;
      html.push(`<p>${renderInline(paragraphLines.join(' '))}</p>`);
      paragraphLines = [];
    };

    const flushList = () => {
      if (!listType) return;
      html.push(`</${listType}>`);
      listType = '';
    };

    const flushQuote = () => {
      if (!quoteLines.length) return;
      html.push(`<blockquote><p>${renderInline(quoteLines.join('<br />'))}</p></blockquote>`);
      quoteLines = [];
    };

    const flushTextBlocks = () => {
      flushParagraph();
      flushList();
      flushQuote();
    };

    const flushCode = () => {
      if (!inCodeBlock) return;
      html.push(`<pre><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`);
      codeLines = [];
      inCodeBlock = false;
    };

    for (const line of lines) {
      if (line.trim().startsWith('```')) {
        flushTextBlocks();
        if (inCodeBlock) {
          flushCode();
        } else {
          inCodeBlock = true;
          codeLines = [];
        }
        continue;
      }

      if (inCodeBlock) {
        codeLines.push(line);
        continue;
      }

      if (!line.trim()) {
        flushTextBlocks();
        continue;
      }

      if (/^---+$/.test(line.trim())) {
        flushTextBlocks();
        html.push('<hr />');
        continue;
      }

      const headingMatch = line.match(/^(#{1,6})\s+(.+)$/);
      if (headingMatch) {
        flushTextBlocks();
        const level = Math.min(6, headingMatch[1].length);
        html.push(`<h${level}>${renderInline(headingMatch[2].trim())}</h${level}>`);
        continue;
      }

      const quoteMatch = line.match(/^>\s?(.*)$/);
      if (quoteMatch) {
        flushParagraph();
        flushList();
        quoteLines.push(quoteMatch[1]);
        continue;
      }

      const orderedMatch = line.match(/^\d+\.\s+(.+)$/);
      if (orderedMatch) {
        flushParagraph();
        flushQuote();
        if (listType !== 'ol') {
          flushList();
          listType = 'ol';
          html.push('<ol>');
        }
        html.push(`<li>${renderInline(orderedMatch[1])}</li>`);
        continue;
      }

      const unorderedMatch = line.match(/^[-*]\s+(.+)$/);
      if (unorderedMatch) {
        flushParagraph();
        flushQuote();
        if (listType !== 'ul') {
          flushList();
          listType = 'ul';
          html.push('<ul>');
        }
        html.push(`<li>${renderInline(unorderedMatch[1])}</li>`);
        continue;
      }

      paragraphLines.push(line.trim());
    }

    flushTextBlocks();
    flushCode();

    return html.join('');
  }

  function resolveInitialFileName() {
    return articleFiles.find((file) => file.name === requestedFile)?.name
      || articleFiles.find((file) => file.name === defaultFile)?.name
      || articleFiles[0]?.name
      || '';
  }

  function renderList() {
    listNode.innerHTML = '';

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
          loadArticle(file.name);
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

  async function loadArticle(fileName) {
    const file = articleFiles.find((entry) => entry.name === fileName);
    if (!file) return;

    currentFile = file.name;
    renderList();
    setStatus('아티클 내용을 불러오는 중입니다.');
    updateQuery(file.name);

    try {
      const markdown = await fetchMarkdown(file);
      const sanitizedMarkdown = sanitizeMarkdown(markdown);
      const renderedTitle = extractTitle(sanitizedMarkdown, file.fileName || file.name);
      file.title = renderedTitle;
      renderList();
      showViewer(renderMarkdown(sanitizedMarkdown));
      await typesetMath();
    } catch (error) {
      setStatus('md 파일을 불러오지 못했습니다. 배포된 articles 경로와 파일명을 확인해 주세요.', true);
      viewerNode.hidden = true;
      console.error(error);
    }
  }

  async function loadArticleIndex() {
    setStatus('GitHub 저장소에서 article 목록을 불러오는 중입니다.');

    try {
      const items = await fetchDirectoryEntries();
      setArticleFiles(
        mergeArticleFiles(
          configuredFiles,
          items
        )
      );

      if (!articleFiles.length) {
        setStatus('articles 폴더에 md 파일이 아직 없습니다. 새 파일을 올리면 여기에 자동으로 나타납니다.');
        return;
      }

      const initialFile = resolveInitialFileName();

      loadArticle(initialFile);
    } catch (error) {
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

  loadArticleIndex();
})();
