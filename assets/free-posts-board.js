(() => {
  const STORAGE_KEY = 'studyBoardPosts';
  const form = document.getElementById('study-board-form');
  const listNode = document.getElementById('study-board-list');
  const emptyNode = document.getElementById('study-board-empty');
  const detailNode = document.getElementById('study-post-detail');

  if (!form || !listNode || !emptyNode) {
    return;
  }

  const fields = {
    title: document.getElementById('study-title'),
    scheduleDate: document.getElementById('study-schedule-date'),
    capacity: document.getElementById('study-capacity'),
    field: document.getElementById('study-field'),
    mode: document.getElementById('study-mode'),
    apply: document.getElementById('study-apply'),
    detail: document.getElementById('study-detail')
  };
  const filters = {
    field: document.getElementById('filter-field'),
    mode: document.getElementById('filter-mode')
  };

  function readPosts() {
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      const parsed = raw ? JSON.parse(raw) : [];
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }

  function savePosts(posts) {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(posts));
  }

  function isExpired(post) {
    const scheduleTime = new Date(post.schedule).getTime();
    return Number.isNaN(scheduleTime) || scheduleTime < Date.now();
  }

  function parseScheduleDate(dateValue) {
    if (!dateValue) {
      return null;
    }

    const [yearRaw, monthRaw, dayRaw] = String(dateValue).split('-');
    const year = Number(yearRaw);
    const month = Number(monthRaw);
    const day = Number(dayRaw);

    if (!year || !month || !day || month < 1 || month > 12 || day < 1 || day > 31) {
      return null;
    }

    const date = new Date(year, month - 1, day, 23, 59, 59, 999);
    if (
      Number.isNaN(date.getTime()) ||
      date.getFullYear() !== year ||
      date.getMonth() !== month - 1 ||
      date.getDate() !== day
    ) {
      return null;
    }

    return date;
  }

  function initScheduleFields() {
    if (!fields.scheduleDate) return;

    const today = new Date();
    const minDate = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}-${String(today.getDate()).padStart(2, '0')}`;
    fields.scheduleDate.min = minDate;

    if (!fields.scheduleDate.value) {
      const nextDay = new Date(today);
      nextDay.setDate(today.getDate() + 1);
      fields.scheduleDate.value = `${nextDay.getFullYear()}-${String(nextDay.getMonth() + 1).padStart(2, '0')}-${String(nextDay.getDate()).padStart(2, '0')}`;
    }
  }

  function getPostPermalink(postId) {
    const url = new URL(window.location.href);
    url.searchParams.set('post', postId);
    return url.toString();
  }

  function pruneExpiredPosts() {
    const storedPosts = readPosts().filter((post) => post && post.id);
    const posts = storedPosts.filter((post) => !isExpired(post));
    const shouldSave = posts.length !== storedPosts.length;
    if (shouldSave) {
      savePosts(posts);
    }
    return posts;
  }

  function escapeHtml(value) {
    return String(value)
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;')
      .replaceAll("'", '&#39;');
  }

  function formatSchedule(value) {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
      return '일정 미정';
    }

    return new Intl.DateTimeFormat('ko-KR', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit'
    }).format(date);
  }

  function normalizeApplyText(value) {
    const trimmed = value.trim();
    if (/^https?:\/\//i.test(trimmed)) {
      return `<a class="text-link" href="${escapeHtml(trimmed)}" target="_blank" rel="noopener noreferrer">${escapeHtml(trimmed)}</a>`;
    }

    return escapeHtml(trimmed);
  }

  function getFilteredPosts(posts) {
    return posts.filter((post) => {
      const fieldValue = filters.field?.value || 'all';
      const modeValue = filters.mode?.value || 'all';

      if (fieldValue !== 'all' && post.field !== fieldValue) return false;
      if (modeValue !== 'all' && post.mode !== modeValue) return false;

      return true;
    });
  }

  function renderPostDetail(post) {
    if (!detailNode) return;

    if (!post) {
      detailNode.hidden = true;
      detailNode.innerHTML = '';
      return;
    }

    detailNode.hidden = false;
    detailNode.innerHTML = `
      <h3>게시글 상세</h3>
      <div class="study-post-detail-head">
        <strong>${escapeHtml(post.title)}</strong>
      </div>
      <div class="study-post-detail-meta">
        <p><b>모집 마감</b> ${formatSchedule(post.schedule)}</p>
        <p><b>모집인원</b> ${escapeHtml(post.capacity)}명</p>
        <p><b>분야</b> ${escapeHtml(post.field || '-')}</p>
        <p><b>진행 방식</b> ${escapeHtml(post.mode || '-')}</p>
        <p><b>신청방법</b> ${normalizeApplyText(post.apply || '')}</p>
      </div>
      <p class="study-post-detail-content">${escapeHtml(post.detail)}</p>
      <div class="study-post-detail-share">
        <a class="text-link" href="free-posts.html">목록으로</a>
        <a class="text-link" href="${escapeHtml(getPostPermalink(post.id))}">상세 링크</a>
      </div>
    `;
  }

  function renderPosts() {
    const posts = pruneExpiredPosts().sort((a, b) => {
      const scheduleDiff = new Date(a.schedule).getTime() - new Date(b.schedule).getTime();
      if (scheduleDiff !== 0) return scheduleDiff;

      return new Date(b.createdAt || 0).getTime() - new Date(a.createdAt || 0).getTime();
    });
    const filteredPosts = getFilteredPosts(posts);

    listNode.innerHTML = filteredPosts
      .map((post) => {
        return `
          <li class="study-board-item" id="study-post-${escapeHtml(post.id)}">
            <div class="study-board-item-head">
              <strong><a class="text-link" href="${escapeHtml(getPostPermalink(post.id))}">${escapeHtml(post.title)}</a></strong>
              <span class="study-board-item-date">${formatSchedule(post.schedule)}</span>
            </div>
            <div class="study-board-item-meta">
              <p><b>분야</b> ${escapeHtml(post.field || '-')}</p>
              <p><b>모집 마감</b> ${formatSchedule(post.schedule)}</p>
              <p><b>진행</b> ${escapeHtml(post.mode || '-')}</p>
              <p><b>모집인원</b> ${escapeHtml(post.capacity)}명</p>
              <p><b>신청방법</b> ${normalizeApplyText(post.apply)}</p>
            </div>
            <div class="study-board-item-share">
              <a class="text-link" href="${escapeHtml(getPostPermalink(post.id))}">게시글 보기</a>
            </div>
            <p class="study-board-item-detail">${escapeHtml(String(post.detail).slice(0, 120))}${String(post.detail).length > 120 ? '...' : ''}</p>
          </li>
        `;
      })
      .join('');

    emptyNode.hidden = filteredPosts.length > 0;

    const selectedPostId = new URLSearchParams(window.location.search).get('post');
    if (!selectedPostId) {
      renderPostDetail(null);
      return;
    }

    const selectedPost = posts.find((post) => post.id === selectedPostId) || null;
    renderPostDetail(selectedPost);
    const target = document.getElementById(`study-post-${selectedPostId}`);
    if (!target) {
      return;
    }

    target.classList.add('is-focused');
    target.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  function resetForm() {
    form.reset();
    initScheduleFields();
    fields.title?.focus();
  }

  form.addEventListener('submit', (event) => {
    event.preventDefault();

    const nextPost = {
      id: crypto.randomUUID(),
      createdAt: new Date().toISOString(),
      title: fields.title?.value.trim() || '',
      schedule: '',
      capacity: fields.capacity?.value || '',
      field: fields.field?.value || '',
      mode: fields.mode?.value || '',
      apply: fields.apply?.value.trim() || '',
      detail: fields.detail?.value.trim() || ''
    };

    const parsedSchedule = parseScheduleDate(fields.scheduleDate?.value || '');
    if (!parsedSchedule) {
      window.alert('모집 마감 날짜를 선택해주세요.');
      fields.scheduleDate?.focus();
      return;
    }
    nextPost.schedule = parsedSchedule.toISOString();

    if (!nextPost.title || !nextPost.capacity || !nextPost.apply || !nextPost.detail) {
      return;
    }

    const posts = pruneExpiredPosts();
    posts.push(nextPost);
    savePosts(posts);

    renderPosts();
    resetForm();
  });

  Object.values(filters).forEach((node) => {
    node?.addEventListener('input', renderPosts);
    node?.addEventListener('change', renderPosts);
  });

  initScheduleFields();
  renderPosts();
  window.setInterval(renderPosts, 60 * 1000);
})();
