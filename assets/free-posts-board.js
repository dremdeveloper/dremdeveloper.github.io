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
    scheduleDate: document.getElementById('study-schedule-date'),
    capacity: document.getElementById('study-capacity'),
    field: document.getElementById('study-field'),
    mode: document.getElementById('study-mode'),
    apply: document.getElementById('study-apply'),
    password: document.getElementById('study-password'),
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

  async function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return;
    }

    const tempTextArea = document.createElement('textarea');
    tempTextArea.value = text;
    tempTextArea.setAttribute('readonly', '');
    tempTextArea.style.position = 'absolute';
    tempTextArea.style.left = '-9999px';
    document.body.append(tempTextArea);
    tempTextArea.select();
    const success = document.execCommand('copy');
    tempTextArea.remove();
    if (!success) {
      throw new Error('copy-failed');
    }
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
              <strong>${escapeHtml(post.title)}</strong>
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
              <button class="button ghost-button study-copy-link-btn" type="button" data-copy-link="${escapeHtml(getPostPermalink(post.id))}">링크 복사</button>
              <input class="study-delete-password" type="password" placeholder="비밀번호 입력" aria-label="삭제 비밀번호 입력" data-delete-password="${escapeHtml(post.id)}" />
              <button class="button ghost-button study-delete-button" type="button" data-delete-post="${escapeHtml(post.id)}">글 삭제</button>
            </div>
            <p class="study-board-item-detail">${escapeHtml(String(post.detail).slice(0, 120))}${String(post.detail).length > 120 ? '...' : ''}</p>
          </li>
        `;
      })
      .join('');

    emptyNode.hidden = filteredPosts.length > 0;

    const selectedPostId = new URLSearchParams(window.location.search).get('post');
    if (!selectedPostId) return;
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
      password: fields.password?.value || '',
      detail: fields.detail?.value.trim() || ''
    };

    const parsedSchedule = parseScheduleDate(fields.scheduleDate?.value || '');
    if (!parsedSchedule) {
      window.alert('모집 마감 날짜를 선택해주세요.');
      fields.scheduleDate?.focus();
      return;
    }
    nextPost.schedule = parsedSchedule.toISOString();

    if (!nextPost.title || !nextPost.capacity || !nextPost.apply || !nextPost.password || !nextPost.detail) {
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

  listNode.addEventListener('click', async (event) => {
    const copyButton = event.target.closest('[data-copy-link]');
    if (copyButton) {
      const link = copyButton.getAttribute('data-copy-link');
      if (!link) return;
      try {
        await copyText(link);
        window.alert('링크를 복사했어요.');
      } catch {
        window.alert('링크 복사에 실패했어요. 직접 복사해주세요.');
      }
      return;
    }

    const deleteButton = event.target.closest('[data-delete-post]');
    if (!deleteButton) return;

    const postId = deleteButton.getAttribute('data-delete-post');
    if (!postId) return;

    const passwordInput = listNode.querySelector(`[data-delete-password="${postId}"]`);
    const passwordValue = passwordInput instanceof HTMLInputElement ? passwordInput.value : '';
    if (!passwordValue) {
      window.alert('비밀번호를 입력해주세요.');
      passwordInput?.focus();
      return;
    }

    const posts = pruneExpiredPosts();
    const targetPost = posts.find((post) => post.id === postId);
    if (!targetPost) {
      window.alert('이미 삭제되었거나 존재하지 않는 글입니다.');
      renderPosts();
      return;
    }

    if (targetPost.password !== passwordValue) {
      window.alert('비밀번호가 일치하지 않습니다.');
      passwordInput?.focus();
      return;
    }

    const nextPosts = posts.filter((post) => post.id !== postId);
    savePosts(nextPosts);
    renderPosts();
  });

  initScheduleFields();
  renderPosts();
  window.setInterval(renderPosts, 60 * 1000);
})();
