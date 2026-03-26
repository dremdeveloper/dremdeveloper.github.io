(() => {
  const STORAGE_KEY = 'studyBoardPosts';
  const form = document.getElementById('study-board-form');
  const listNode = document.getElementById('study-board-list');
  const emptyNode = document.getElementById('study-board-empty');

  if (!form || !listNode || !emptyNode) {
    return;
  }

  const fields = {
    title: document.getElementById('study-title'),
    schedule: document.getElementById('study-schedule'),
    capacity: document.getElementById('study-capacity'),
    apply: document.getElementById('study-apply'),
    detail: document.getElementById('study-detail')
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
    return Number.isNaN(scheduleTime) || scheduleTime <= Date.now();
  }

  function pruneExpiredPosts() {
    const posts = readPosts();
    const activePosts = posts.filter((post) => !isExpired(post));

    if (activePosts.length !== posts.length) {
      savePosts(activePosts);
    }

    return activePosts;
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
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false
    }).format(date);
  }

  function normalizeApplyText(value) {
    const trimmed = value.trim();
    if (/^https?:\/\//i.test(trimmed)) {
      return `<a class="text-link" href="${escapeHtml(trimmed)}" target="_blank" rel="noopener noreferrer">${escapeHtml(trimmed)}</a>`;
    }

    return escapeHtml(trimmed);
  }

  function renderPosts() {
    const posts = pruneExpiredPosts().sort((a, b) => {
      const scheduleDiff = new Date(a.schedule).getTime() - new Date(b.schedule).getTime();
      if (scheduleDiff !== 0) return scheduleDiff;

      return new Date(b.createdAt || 0).getTime() - new Date(a.createdAt || 0).getTime();
    });

    listNode.innerHTML = posts
      .map((post) => {
        return `
          <li class="study-board-item">
            <div class="study-board-item-head">
              <strong>${escapeHtml(post.title)}</strong>
              <span class="study-board-item-date">${formatSchedule(post.schedule)}</span>
            </div>
            <div class="study-board-item-meta">
              <p><b>모집인원</b> ${escapeHtml(post.capacity)}명</p>
              <p><b>신청방법</b> ${normalizeApplyText(post.apply)}</p>
            </div>
            <p class="study-board-item-detail">${escapeHtml(post.detail)}</p>
          </li>
        `;
      })
      .join('');

    emptyNode.hidden = posts.length > 0;
  }

  function resetForm() {
    form.reset();
    fields.title?.focus();
  }

  form.addEventListener('submit', (event) => {
    event.preventDefault();

    const nextPost = {
      id: crypto.randomUUID(),
      createdAt: new Date().toISOString(),
      title: fields.title?.value.trim() || '',
      schedule: fields.schedule?.value || '',
      capacity: fields.capacity?.value || '',
      apply: fields.apply?.value.trim() || '',
      detail: fields.detail?.value.trim() || ''
    };

    if (!nextPost.title || !nextPost.schedule || !nextPost.capacity || !nextPost.apply || !nextPost.detail) {
      return;
    }

    if (isExpired(nextPost)) {
      window.alert('지난 일정은 등록할 수 없습니다. 미래 일정을 입력해주세요.');
      return;
    }

    const posts = pruneExpiredPosts();
    posts.push(nextPost);
    savePosts(posts);

    renderPosts();
    resetForm();
  });

  renderPosts();
  window.setInterval(renderPosts, 60 * 1000);
})();
