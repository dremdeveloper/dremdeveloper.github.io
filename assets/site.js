(() => {
  const CLOSE_DELAY_MS = 280;
  const DESKTOP_MEDIA_QUERY = '(min-width: 861px)';
  const NAV_DROPDOWN_SELECTORS = [
    { itemId: 'article-nav-item', toggleId: 'article-nav-toggle' },
    { itemId: 'study-nav-item', toggleId: 'study-nav-toggle' }
  ];

  const yearNode = document.getElementById('year');
  if (yearNode) {
    yearNode.textContent = String(new Date().getFullYear());
  }

  const menuToggle = document.querySelector('.menu-toggle');
  const siteNav = document.querySelector('.site-nav');
  const desktopMediaQuery = window.matchMedia(DESKTOP_MEDIA_QUERY);
  const dropdownConfigs = NAV_DROPDOWN_SELECTORS
    .map(({ itemId, toggleId }) => {
      const item = document.getElementById(itemId);
      const toggle = document.getElementById(toggleId);

      if (!item || !toggle) {
        return null;
      }

      return {
        item,
        toggle,
        detachDesktopListeners: null
      };
    })
    .filter(Boolean);

  const closeTimers = new Map();

  const setExpandedState = (element, isExpanded) => {
    element.setAttribute('aria-expanded', String(isExpanded));
  };

  const setDropdownOpen = (targetItem, isOpen) => {
    dropdownConfigs.forEach(({ item, toggle }) => {
      const shouldOpen = item === targetItem && isOpen;
      item.classList.toggle('is-open', shouldOpen);
      setExpandedState(toggle, shouldOpen);
    });
  };

  const closeAllDropdowns = () => {
    dropdownConfigs.forEach(({ item }) => {
      clearCloseTimer(item);
      setDropdownOpen(item, false);
    });
  };

  const setMenuOpen = (isOpen) => {
    if (!menuToggle || !siteNav) {
      return;
    }

    siteNav.classList.toggle('is-open', isOpen);
    setExpandedState(menuToggle, isOpen);

    if (!isOpen) {
      closeAllDropdowns();
    }
  };

  function clearCloseTimer(item) {
    const timer = closeTimers.get(item);
    if (timer) {
      window.clearTimeout(timer);
      closeTimers.delete(item);
    }
  }

  const openDropdown = (item) => {
    clearCloseTimer(item);
    setDropdownOpen(item, true);
  };

  const scheduleCloseDropdown = (item) => {
    clearCloseTimer(item);
    const timer = window.setTimeout(() => {
      setDropdownOpen(item, false);
      closeTimers.delete(item);
    }, CLOSE_DELAY_MS);
    closeTimers.set(item, timer);
  };

  const bindDesktopHoverListeners = (config) => {
    const handleMouseEnter = () => openDropdown(config.item);
    const handleMouseLeave = () => scheduleCloseDropdown(config.item);

    config.item.addEventListener('mouseenter', handleMouseEnter);
    config.item.addEventListener('mouseleave', handleMouseLeave);

    config.detachDesktopListeners = () => {
      config.item.removeEventListener('mouseenter', handleMouseEnter);
      config.item.removeEventListener('mouseleave', handleMouseLeave);
      config.detachDesktopListeners = null;
    };
  };

  const syncDesktopHoverListeners = () => {
    dropdownConfigs.forEach((config) => {
      if (desktopMediaQuery.matches) {
        if (!config.detachDesktopListeners) {
          bindDesktopHoverListeners(config);
        }
        return;
      }

      config.detachDesktopListeners?.();
    });
  };

  if (menuToggle && siteNav) {
    menuToggle.addEventListener('click', () => {
      const nextOpen = !siteNav.classList.contains('is-open');
      setMenuOpen(nextOpen);
    });

    siteNav.querySelectorAll('a').forEach((link) => {
      link.addEventListener('click', () => {
        setMenuOpen(false);
      });
    });
  }

  if (dropdownConfigs.length === 0) {
    return;
  }

  dropdownConfigs.forEach((config) => {
    const { item, toggle } = config;

    toggle.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      const nextOpen = !item.classList.contains('is-open');
      clearCloseTimer(item);
      setDropdownOpen(item, nextOpen);
    });

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

  syncDesktopHoverListeners();
  desktopMediaQuery.addEventListener('change', () => {
    closeAllDropdowns();
    syncDesktopHoverListeners();
  });
})();
