(() => {
  const yearNode = document.getElementById('year');
  if (yearNode) yearNode.textContent = new Date().getFullYear();

  const menuToggle = document.querySelector('.menu-toggle');
  const siteNav = document.querySelector('.site-nav');
  const studyNavItem = document.getElementById('study-nav-item');
  const studyNavToggle = document.getElementById('study-nav-toggle');
  const studyTopSubmenu = document.getElementById('study-top-submenu');
  const visualizationToggle = document.getElementById('study-visualization-toggle');
  const desktopMediaQuery = window.matchMedia('(min-width: 861px)');

  const setStudyMenuOpen = (isOpen) => {
    if (!studyNavItem || !studyNavToggle) return;
    studyNavItem.classList.toggle('is-open', isOpen);
    studyNavToggle.setAttribute('aria-expanded', String(isOpen));
  };

  if (menuToggle && siteNav) {
    menuToggle.addEventListener('click', () => {
      const isOpen = siteNav.classList.toggle('is-open');
      menuToggle.setAttribute('aria-expanded', String(isOpen));
      if (!isOpen) setStudyMenuOpen(false);
    });

    siteNav.querySelectorAll('a').forEach((link) => {
      link.addEventListener('click', () => {
        siteNav.classList.remove('is-open');
        menuToggle.setAttribute('aria-expanded', 'false');
        setStudyMenuOpen(false);
      });
    });
  }

  if (visualizationToggle || !studyNavItem || !studyNavToggle || !studyTopSubmenu) return;

  let closeTimer = null;

  const clearCloseTimer = () => {
    if (closeTimer) {
      window.clearTimeout(closeTimer);
      closeTimer = null;
    }
  };

  const openStudyMenu = () => {
    clearCloseTimer();
    setStudyMenuOpen(true);
  };

  const scheduleCloseStudyMenu = () => {
    clearCloseTimer();
    closeTimer = window.setTimeout(() => {
      setStudyMenuOpen(false);
      closeTimer = null;
    }, 280);
  };

  studyNavToggle.addEventListener('click', (event) => {
    event.preventDefault();
    const nextOpen = !studyNavItem.classList.contains('is-open');
    clearCloseTimer();
    setStudyMenuOpen(nextOpen);
  });

  if (desktopMediaQuery.matches) {
    studyNavItem.addEventListener('mouseenter', openStudyMenu);
    studyNavItem.addEventListener('mouseleave', scheduleCloseStudyMenu);
  }

  studyNavItem.addEventListener('focusin', openStudyMenu);
  studyNavItem.addEventListener('focusout', () => {
    window.setTimeout(() => {
      if (!studyNavItem.contains(document.activeElement)) {
        scheduleCloseStudyMenu();
      }
    }, 0);
  });

  document.addEventListener('click', (event) => {
    if (!studyNavItem.contains(event.target)) {
      setStudyMenuOpen(false);
    }
  });

  desktopMediaQuery.addEventListener('change', () => {
    setStudyMenuOpen(false);
  });
})();
