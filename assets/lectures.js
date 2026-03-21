(() => {
  const data = window.siteData?.lectures;
  if (!data) return;

  const listNode = document.getElementById('lecture-list');
  const trackMeta = document.getElementById('track-meta');
  const trackButtons = [...document.querySelectorAll('.track-button')];
  const courseLabel = document.getElementById('lecture-course-label');
  const titleNode = document.getElementById('lecture-title');
  const detailGrid = document.getElementById('lecture-detail-grid');
  const player = document.getElementById('lecture-player');
  const bookLink = document.getElementById('lecture-book-link');

  let currentTrack = 'python';
  let currentValue = 'playlist';

  const buildEmbed = (track, item, index) => {
    const params = new URLSearchParams({
      rel: '0',
      playsinline: '1',
      enablejsapi: '1'
    });

    if (window.location.protocol === 'http:' || window.location.protocol === 'https:') {
      params.set('origin', window.location.origin);
      params.set('widget_referrer', window.location.href);
    }

    if (item.type === 'playlist') {
      params.set('listType', 'playlist');
      params.set('list', track.playlistId);
      return `https://www.youtube.com/embed?${params.toString()}`;
    }

    params.set('list', track.playlistId);
    if (index > 0) {
      params.set('index', String(index - 1));
    }
    return `https://www.youtube.com/embed/${item.value}?${params.toString()}`;
  };

  const renderTrackMeta = (track) => {
    trackMeta.innerHTML = `
      <h2>${track.label}</h2>
      <p>${track.totalLabel}</p>
    `;
  };

  const renderList = () => {
    const track = data[currentTrack];
    listNode.innerHTML = '';

    track.items.forEach((item, index) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = `menu-item${item.value === currentValue ? ' is-active' : ''}`;
      button.dataset.value = item.value;
      button.innerHTML = `
        <span class="menu-item-index">${String(index).padStart(2, '0')}</span>
        <span class="menu-item-copy">
          <strong>${item.title}</strong>
          <span>${item.topic}</span>
        </span>
      `;
      button.addEventListener('click', () => {
        currentValue = item.value;
        renderList();
        renderPlayer();
      });
      listNode.appendChild(button);
    });
  };

  const renderPlayer = () => {
    const track = data[currentTrack];
    const item = track.items.find((entry) => entry.value === currentValue) || track.items[0];
    const index = track.items.findIndex((entry) => entry.value === item.value);

    courseLabel.textContent = track.label;
    titleNode.textContent = item.title;
    bookLink.href = track.bookUrl;
    player.src = buildEmbed(track, item, index);

    detailGrid.innerHTML = `
      <div class="meta-card"><span>Playlist</span><strong>${track.label}</strong></div>
      <div class="meta-card"><span>Order</span><strong>${String(index).padStart(2, '0')}</strong></div>
      <div class="meta-card"><span>Topic</span><strong>${item.topic}</strong></div>
      <div class="meta-card"><span>Summary</span><strong>${item.note}</strong></div>
    `;
  };

  const switchTrack = (trackKey) => {
    currentTrack = trackKey;
    currentValue = 'playlist';
    trackButtons.forEach((button) => {
      button.classList.toggle('is-active', button.dataset.track === trackKey);
    });
    renderTrackMeta(data[trackKey]);
    renderList();
    renderPlayer();
  };

  trackButtons.forEach((button) => {
    button.addEventListener('click', () => switchTrack(button.dataset.track));
  });

  switchTrack('python');
})();
