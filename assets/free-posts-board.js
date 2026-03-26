(() => {
  const STORAGE_KEY = 'studyBoardPosts';
  const STORAGE_RESET_VERSION_KEY = 'studyBoardPostsResetVersion';
  const STORAGE_RESET_VERSION = '2026-03-26-clear-all';
  const form = document.getElementById('study-board-form');
  const listNode = document.getElementById('study-board-list');
  const emptyNode = document.getElementById('study-board-empty');
  const countNode = document.getElementById('study-board-count');

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
    passwordConfirm: document.getElementById('study-password-confirm'),
    detail: document.getElementById('study-detail')
  };
  const submitButton = document.getElementById('study-submit-button');
  const editIdField = document.getElementById('study-edit-id');
  const choiceButtons = Array.from(document.querySelectorAll('[data-choice-target][data-choice-value]'));
  const filters = {
    keyword: document.getElementById('filter-keyword'),
    sort: document.getElementById('filter-sort'),
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

  function applyStorageResetIfNeeded() {
    try {
      const appliedVersion = window.localStorage.getItem(STORAGE_RESET_VERSION_KEY);
      if (appliedVersion === STORAGE_RESET_VERSION) {
        return;
      }

      window.localStorage.setItem(STORAGE_KEY, JSON.stringify([]));
      window.localStorage.setItem(STORAGE_RESET_VERSION_KEY, STORAGE_RESET_VERSION);
    } catch {
      // ignore storage access errors
    }
  }

  function savePosts(posts) {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(posts));
  }

  function getScheduleDeadlineTime(post) {
    const parsedDate = parseScheduleDate(post.scheduleDate || '');
    if (parsedDate) {
      return parsedDate.getTime();
    }

    const legacyScheduleTime = new Date(post.schedule).getTime();
    return Number.isNaN(legacyScheduleTime) ? Number.NaN : legacyScheduleTime;
  }

  function isExpired(post) {
    const scheduleTime = getScheduleDeadlineTime(post);
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

  function formatSchedule(value, scheduleDate = '') {
    const parsedDate = parseScheduleDate(scheduleDate);
    const date = parsedDate || new Date(value);
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

  function getModeLabel(value) {
    if (value === '혼합') {
      return '온라인/오프라인';
    }
    return value || '-';
  }

  function getTagClass(type, value) {
    const normalized = String(value || '')
      .trim()
      .toLowerCase()
      .replaceAll('/', '-')
      .replaceAll(' ', '-');
    return `study-tag study-tag-${type} study-tag-${type}-${normalized}`;
  }

  function getFilteredPosts(posts) {
    const keyword = String(filters.keyword?.value || '').trim().toLowerCase();
    return posts.filter((post) => {
      const fieldValue = filters.field?.value || 'all';
      const modeValue = filters.mode?.value || 'all';
      const keywordTargets = [post.title, post.detail, post.apply].map((value) => String(value || '').toLowerCase()).join(' ');

      if (fieldValue !== 'all' && post.field !== fieldValue) return false;
      if (modeValue !== 'all' && post.mode !== modeValue) return false;
      if (keyword && !keywordTargets.includes(keyword)) return false;

      return true;
    });
  }

  function getDaysLeftText(post) {
    const scheduleTime = getScheduleDeadlineTime(post);
    if (Number.isNaN(scheduleTime)) {
      return '마감일 미정';
    }

    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const deadline = new Date(scheduleTime);
    deadline.setHours(0, 0, 0, 0);
    const daysLeft = Math.ceil((deadline.getTime() - today.getTime()) / (24 * 60 * 60 * 1000));

    if (daysLeft <= 0) {
      return 'D-day';
    }
    return `D-${daysLeft}`;
  }

  function sortPosts(posts) {
    const sortValue = filters.sort?.value || 'deadline-asc';
    const sorted = [...posts];

    if (sortValue === 'created-desc') {
      return sorted.sort((a, b) => new Date(b.createdAt || 0).getTime() - new Date(a.createdAt || 0).getTime());
    }

    if (sortValue === 'capacity-desc') {
      return sorted.sort((a, b) => Number(b.capacity || 0) - Number(a.capacity || 0));
    }

    return sorted.sort((a, b) => {
      const aTime = getScheduleDeadlineTime(a);
      const bTime = getScheduleDeadlineTime(b);
      if (aTime !== bTime) return aTime - bTime;
      return new Date(b.createdAt || 0).getTime() - new Date(a.createdAt || 0).getTime();
    });
  }

  function renderPosts() {
    const posts = pruneExpiredPosts();
    const filteredPosts = sortPosts(getFilteredPosts(posts));
    if (countNode) {
      countNode.textContent = `전체 ${posts.length}개 · 검색 결과 ${filteredPosts.length}개`;
    }

    listNode.innerHTML = filteredPosts
      .map((post) => {
        return `
          <li class="study-board-item" id="study-post-${escapeHtml(post.id)}">
            <div class="study-board-item-head">
              <strong>${escapeHtml(post.title)}</strong>
              <span class="study-board-item-date">${formatSchedule(post.schedule, post.scheduleDate)} · ${escapeHtml(getDaysLeftText(post))}</span>
            </div>
            <div class="study-board-item-meta">
              <p><b>분야</b> <span class="${escapeHtml(getTagClass('field', post.field))}">${escapeHtml(post.field || '-')}</span></p>
              <p><b>모집 마감</b> ${formatSchedule(post.schedule, post.scheduleDate)}</p>
              <p><b>진행</b> <span class="${escapeHtml(getTagClass('mode', post.mode))}">${escapeHtml(getModeLabel(post.mode))}</span></p>
              <p><b>모집인원</b> ${escapeHtml(post.capacity)}명</p>
              <p><b>신청방법</b> ${normalizeApplyText(post.apply)}</p>
            </div>
            <div class="study-board-item-share">
              <button class="button ghost-button study-copy-link-btn" type="button" data-copy-link="${escapeHtml(getPostPermalink(post.id))}">링크 복사</button>
              <button class="button ghost-button" type="button" data-edit-post="${escapeHtml(post.id)}">글 수정</button>
              <button class="button ghost-button study-delete-button" type="button" data-delete-post="${escapeHtml(post.id)}">글 삭제</button>
            </div>
            <div class="study-board-item-auth">
              <label class="study-delete-password-wrap">
                <span>수정/삭제 비밀번호</span>
                <input class="study-delete-password" type="password" placeholder="비밀번호 입력" aria-label="수정 또는 삭제 비밀번호 입력" data-post-password="${escapeHtml(post.id)}" />
              </label>
            </div>
            <p class="study-board-item-detail">${escapeHtml(String(post.detail).slice(0, 120))}${String(post.detail).length > 120 ? '...' : ''}</p>
          </li>
        `;
      })
      .join('');

    emptyNode.hidden = filteredPosts.length > 0;
    if (!emptyNode.hidden) {
      if (posts.length > 0) {
        emptyNode.textContent = '조건에 맞는 모집글이 없습니다. 검색어나 필터를 조정해보세요.';
      } else {
        emptyNode.textContent = '등록된 모집글이 없습니다.';
      }
    }

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
    updateChoiceSelection('study-field', '알고리즘');
    updateChoiceSelection('study-mode', '온라인');
    if (editIdField) {
      editIdField.value = '';
    }
    if (submitButton) {
      submitButton.textContent = '등록하기';
    }
    initScheduleFields();
    fields.title?.focus();
  }

  function updateChoiceSelection(targetId, value) {
    choiceButtons.forEach((button) => {
      if (button.dataset.choiceTarget !== targetId) {
        return;
      }
      const isActive = button.dataset.choiceValue === value;
      button.classList.toggle('is-active', isActive);
      button.setAttribute('aria-pressed', String(isActive));
    });

    const hiddenInput = document.getElementById(targetId);
    if (hiddenInput instanceof HTMLInputElement) {
      hiddenInput.value = value;
    }
  }

  function startEditMode(post) {
    if (!post) return;

    fields.title.value = post.title || '';
    if (fields.scheduleDate) {
      fields.scheduleDate.value = post.scheduleDate || new Date(post.schedule).toISOString().slice(0, 10);
    }
    fields.capacity.value = post.capacity || '';
    fields.apply.value = post.apply || '';
    fields.detail.value = post.detail || '';
    fields.password.value = post.password || '';
    fields.passwordConfirm.value = post.password || '';
    updateChoiceSelection('study-field', post.field || '알고리즘');
    updateChoiceSelection('study-mode', post.mode || '온라인');

    if (editIdField) {
      editIdField.value = post.id;
    }
    if (submitButton) {
      submitButton.textContent = '수정 저장';
    }
    form.scrollIntoView({ behavior: 'smooth', block: 'start' });
    fields.title?.focus();
  }

  form.addEventListener('submit', (event) => {
    event.preventDefault();

    const nextPost = {
      id: crypto.randomUUID(),
      createdAt: new Date().toISOString(),
      title: fields.title?.value.trim() || '',
      schedule: '',
      scheduleDate: fields.scheduleDate?.value || '',
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
    nextPost.scheduleDate = fields.scheduleDate?.value || '';

    const passwordConfirm = fields.passwordConfirm?.value || '';
    if (nextPost.password !== passwordConfirm) {
      window.alert('비밀번호와 비밀번호 확인 값이 일치하지 않습니다.');
      fields.passwordConfirm?.focus();
      return;
    }

    if (!nextPost.title || !nextPost.capacity || !nextPost.apply || !nextPost.password || !nextPost.detail || !nextPost.field || !nextPost.mode) {
      return;
    }

    const posts = pruneExpiredPosts();
    const editId = editIdField?.value || '';
    if (editId) {
      const editIndex = posts.findIndex((post) => post.id === editId);
      if (editIndex < 0) {
        window.alert('수정할 게시글을 찾을 수 없습니다. 다시 시도해주세요.');
        resetForm();
        renderPosts();
        return;
      }
      nextPost.id = editId;
      nextPost.createdAt = posts[editIndex].createdAt || nextPost.createdAt;
      posts[editIndex] = nextPost;
    } else {
      posts.push(nextPost);
    }
    savePosts(posts);

    renderPosts();
    resetForm();
  });

  Object.values(filters).forEach((node) => {
    node?.addEventListener('input', renderPosts);
    node?.addEventListener('change', renderPosts);
  });

  choiceButtons.forEach((button) => {
    button.addEventListener('click', () => {
      const targetId = button.dataset.choiceTarget;
      const selectedValue = button.dataset.choiceValue;
      if (!targetId || !selectedValue) return;
      updateChoiceSelection(targetId, selectedValue);
    });
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

    const editButton = event.target.closest('[data-edit-post]');
    if (editButton) {
      const postId = editButton.getAttribute('data-edit-post');
      if (!postId) return;

      const passwordInput = listNode.querySelector(`[data-post-password="${postId}"]`);
      const passwordValue = passwordInput instanceof HTMLInputElement ? passwordInput.value : '';
      if (!passwordValue) {
        window.alert('수정 비밀번호를 입력해주세요.');
        passwordInput?.focus();
        return;
      }

      const posts = pruneExpiredPosts();
      const targetPost = posts.find((post) => post.id === postId);
      if (!targetPost) {
        window.alert('존재하지 않는 글입니다.');
        renderPosts();
        return;
      }
      if (targetPost.password !== passwordValue) {
        window.alert('비밀번호가 일치하지 않습니다.');
        passwordInput?.focus();
        return;
      }

      startEditMode(targetPost);
      return;
    }

    const deleteButton = event.target.closest('[data-delete-post]');
    if (!deleteButton) return;

    const postId = deleteButton.getAttribute('data-delete-post');
    if (!postId) return;

    const passwordInput = listNode.querySelector(`[data-post-password="${postId}"]`);
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

  applyStorageResetIfNeeded();
  initScheduleFields();
  renderPosts();
  window.setInterval(renderPosts, 60 * 1000);
})();
