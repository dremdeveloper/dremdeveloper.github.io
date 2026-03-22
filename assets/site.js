(() => {
  const yearNode = document.getElementById('year');
  if (yearNode) yearNode.textContent = new Date().getFullYear();

  const menuToggle = document.querySelector('.menu-toggle');
  const siteNav = document.querySelector('.site-nav');
  const visualizationToggle = document.getElementById('study-visualization-toggle');
  const desktopMediaQuery = window.matchMedia('(min-width: 861px)');
  const dropdownConfigs = [
    {
      item: document.getElementById('article-nav-item'),
      toggle: document.getElementById('article-nav-toggle')
    },
    {
      item: document.getElementById('study-nav-item'),
      toggle: document.getElementById('study-nav-toggle')
    }
  ].filter((entry) => entry.item && entry.toggle);

  const setDropdownOpen = (targetItem, isOpen) => {
    dropdownConfigs.forEach(({ item, toggle }) => {
      const shouldOpen = item === targetItem && isOpen;
      item.classList.toggle('is-open', shouldOpen);
      toggle.setAttribute('aria-expanded', String(shouldOpen));
    });
  };

  const closeAllDropdowns = () => {
    dropdownConfigs.forEach(({ item }) => setDropdownOpen(item, false));
  };

  if (menuToggle && siteNav) {
    menuToggle.addEventListener('click', () => {
      const isOpen = siteNav.classList.toggle('is-open');
      menuToggle.setAttribute('aria-expanded', String(isOpen));
      if (!isOpen) closeAllDropdowns();
    });

    siteNav.querySelectorAll('a').forEach((link) => {
      link.addEventListener('click', () => {
        siteNav.classList.remove('is-open');
        menuToggle.setAttribute('aria-expanded', 'false');
        closeAllDropdowns();
      });
    });
  }

  if (visualizationToggle || dropdownConfigs.length === 0) return;

  const closeTimers = new Map();

  const clearCloseTimer = (item) => {
    const timer = closeTimers.get(item);
    if (timer) {
      window.clearTimeout(timer);
      closeTimers.delete(item);
    }
  };

  const openDropdown = (item) => {
    clearCloseTimer(item);
    setDropdownOpen(item, true);
  };

  const scheduleCloseDropdown = (item) => {
    clearCloseTimer(item);
    const timer = window.setTimeout(() => {
      setDropdownOpen(item, false);
      closeTimers.delete(item);
    }, 280);
    closeTimers.set(item, timer);
  };

  dropdownConfigs.forEach(({ item, toggle }) => {
    toggle.addEventListener('click', (event) => {
      event.preventDefault();
      const nextOpen = !item.classList.contains('is-open');
      clearCloseTimer(item);
      setDropdownOpen(item, nextOpen);
    });

    if (desktopMediaQuery.matches) {
      item.addEventListener('mouseenter', () => openDropdown(item));
      item.addEventListener('mouseleave', () => scheduleCloseDropdown(item));
    }

    item.addEventListener('focusin', () => openDropdown(item));
    item.addEventListener('focusout', () => {
      window.setTimeout(() => {
        if (!item.contains(document.activeElement)) {
          scheduleCloseDropdown(item);
        }
      }, 0);
    });
  });

  document.addEventListener('click', (event) => {
    if (!dropdownConfigs.some(({ item }) => item.contains(event.target))) {
      closeAllDropdowns();
    }
  });

  desktopMediaQuery.addEventListener('change', () => {
    closeAllDropdowns();
  });
})();
