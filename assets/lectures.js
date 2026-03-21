(() => {
  const data = window.siteData?.lectures;
  if (!data) return;

  const listNode = document.getElementById('lecture-list');
  const trackMeta = document.getElementById('track-meta');
  const trackButtons = [...document.querySelectorAll('.track-button')];
  const courseLabel = document.getElementById('lecture-course-label');
  const titleNode = document.getElementById('lecture-title');
  const detailGrid = document.getElementById('lecture-detail-grid');
  const bookLink = document.getElementById('lecture-book-link');
  const playToggle = document.getElementById('lecture-play-toggle');
  const progressInput = document.getElementById('lecture-progress');
  const currentTimeNode = document.getElementById('lecture-current-time');
  const durationNode = document.getElementById('lecture-duration');

  let currentTrack = 'python';
  let currentValue = 'playlist';
  let ytPlayer;
  let playerReady = false;
  let pendingSelection = null;
  let syncTimer = null;
  let isSeeking = false;

  const syncProgressVisual = () => {
    const min = Number(progressInput.min || 0);
    const max = Number(progressInput.max || 1000);
    const value = Number(progressInput.value || 0);
    const ratio = max > min ? ((value - min) / (max - min)) * 100 : 0;
    progressInput.style.setProperty('--progress-percent', `${Math.min(100, Math.max(0, ratio))}%`);
  };

  const formatTime = (seconds) => {
    if (!Number.isFinite(seconds) || seconds <= 0) {
      return '00:00';
    }

    const totalSeconds = Math.floor(seconds);
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const secs = totalSeconds % 60;

    if (hours > 0) {
      return [hours, minutes, secs].map((value, index) => String(value).padStart(index === 0 ? 1 : 2, '0')).join(':');
    }

    return `${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  };

  const setControlState = ({ ready = playerReady, playing = false, currentTime = 0, duration = 0 } = {}) => {
    playToggle.disabled = !ready;
    progressInput.disabled = !ready;
    playToggle.textContent = playing ? '일시정지' : '재생';
    playToggle.setAttribute('aria-label', playing ? '일시정지' : '재생');
    currentTimeNode.textContent = formatTime(currentTime);
    durationNode.textContent = formatTime(duration);

    if (!isSeeking) {
      const progressValue = duration > 0 ? Math.min(1000, Math.round((currentTime / duration) * 1000)) : 0;
      progressInput.value = String(progressValue);
    }

    syncProgressVisual();
  };

  const getCurrentSelection = () => {
    const track = data[currentTrack];
    const item = track.items.find((entry) => entry.value === currentValue) || track.items[0];
    const index = track.items.findIndex((entry) => entry.value === item.value);
    return { track, item, index };
  };

  const updateProgress = () => {
    if (!playerReady || !ytPlayer?.getDuration) return;

    const playerState = ytPlayer.getPlayerState?.();
    const duration = ytPlayer.getDuration();
    const currentTime = ytPlayer.getCurrentTime?.() || 0;

    setControlState({
      ready: true,
      playing: playerState === window.YT?.PlayerState?.PLAYING,
      currentTime,
      duration
    });
  };

  const startSync = () => {
    if (syncTimer) return;
    syncTimer = window.setInterval(updateProgress, 250);
  };

  const stopSync = () => {
    if (!syncTimer) return;
    window.clearInterval(syncTimer);
    syncTimer = null;
  };

  const cueSelection = ({ track, item, index }) => {
    if (!ytPlayer?.cuePlaylist) return;

    ytPlayer.cuePlaylist({
      listType: 'playlist',
      list: track.playlistId,
      index: item.type === 'playlist' ? 0 : Math.max(index - 1, 0)
    });
  };

  const ensureYouTubeApi = () => {
    if (window.YT?.Player) {
      return Promise.resolve(window.YT);
    }

    if (!window.youtubeIframeApiPromise) {
      window.youtubeIframeApiPromise = new Promise((resolve) => {
        const previousCallback = window.onYouTubeIframeAPIReady;
        window.onYouTubeIframeAPIReady = () => {
          previousCallback?.();
          resolve(window.YT);
        };

        const script = document.createElement('script');
        script.src = 'https://www.youtube.com/iframe_api';
        script.async = true;
        document.head.appendChild(script);
      });
    }

    return window.youtubeIframeApiPromise;
  };

  const mountPlayer = async (selection) => {
    pendingSelection = selection;
    await ensureYouTubeApi();

    if (!ytPlayer) {
      ytPlayer = new window.YT.Player('lecture-player', {
        width: '100%',
        height: '100%',
        playerVars: {
          rel: 0,
          playsinline: 1,
          controls: 1,
          enablejsapi: 1,
          origin: window.location.origin
        },
        events: {
          onReady: () => {
            playerReady = true;
            if (pendingSelection) {
              cueSelection(pendingSelection);
            }
            setControlState({ ready: true, playing: false, currentTime: 0, duration: ytPlayer.getDuration?.() || 0 });
          },
          onStateChange: (event) => {
            const isPlaying = event.data === window.YT.PlayerState.PLAYING;
            if (isPlaying) {
              startSync();
            } else {
              stopSync();
            }
            updateProgress();
          }
        }
      });
      return;
    }

    if (playerReady) {
      cueSelection(selection);
      setControlState({ ready: true, playing: false, currentTime: 0, duration: 0 });
    }
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
    const { track, item, index } = getCurrentSelection();

    courseLabel.textContent = track.label;
    titleNode.textContent = item.title;
    bookLink.href = track.bookUrl;

    detailGrid.innerHTML = `
      <div class="meta-card"><span>Playlist</span><strong>${track.label}</strong></div>
      <div class="meta-card"><span>Order</span><strong>${String(index).padStart(2, '0')}</strong></div>
      <div class="meta-card"><span>Topic</span><strong>${item.topic}</strong></div>
      <div class="meta-card"><span>Summary</span><strong>${item.note}</strong></div>
    `;

    setControlState({ ready: playerReady, playing: false, currentTime: 0, duration: 0 });
    mountPlayer({ track, item, index });
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

  playToggle.addEventListener('click', () => {
    if (!playerReady || !ytPlayer?.getPlayerState) return;

    const playerState = ytPlayer.getPlayerState();
    if (playerState === window.YT.PlayerState.PLAYING) {
      ytPlayer.pauseVideo();
      return;
    }

    ytPlayer.playVideo();
  });

  progressInput.addEventListener('input', () => {
    if (!playerReady || !ytPlayer?.getDuration) return;

    isSeeking = true;
    syncProgressVisual();
    const duration = ytPlayer.getDuration();
    const targetTime = duration > 0 ? (Number(progressInput.value) / 1000) * duration : 0;
    currentTimeNode.textContent = formatTime(targetTime);
    durationNode.textContent = formatTime(duration);
  });

  progressInput.addEventListener('change', () => {
    if (!playerReady || !ytPlayer?.getDuration) {
      isSeeking = false;
      return;
    }

    const duration = ytPlayer.getDuration();
    const targetTime = duration > 0 ? (Number(progressInput.value) / 1000) * duration : 0;
    ytPlayer.seekTo(targetTime, true);
    isSeeking = false;
    updateProgress();
  });

  progressInput.addEventListener('pointerdown', () => {
    isSeeking = true;
  });

  progressInput.addEventListener('pointerup', () => {
    isSeeking = false;
  });

  trackButtons.forEach((button) => {
    button.addEventListener('click', () => switchTrack(button.dataset.track));
  });

  syncProgressVisual();
  switchTrack('python');
})();
