(() => {
  const yearNode = document.getElementById('year');
  if (yearNode) yearNode.textContent = new Date().getFullYear();

  const menuToggle = document.querySelector('.menu-toggle');
  const siteNav = document.querySelector('.site-nav');
  if (menuToggle && siteNav) {
    menuToggle.addEventListener('click', () => {
      const isOpen = siteNav.classList.toggle('is-open');
      menuToggle.setAttribute('aria-expanded', String(isOpen));
    });

    siteNav.querySelectorAll('a').forEach((link) => {
      link.addEventListener('click', () => {
        siteNav.classList.remove('is-open');
        menuToggle.setAttribute('aria-expanded', 'false');
      });
    });
  }

  const studyNavItem = document.getElementById('study-nav-item');
  const studyNavToggle = document.getElementById('study-nav-toggle');
  if (studyNavItem && studyNavToggle && window.matchMedia('(min-width: 861px)').matches) {
    let closeTimer = null;

    const openStudyMenu = () => {
      if (closeTimer) {
        window.clearTimeout(closeTimer);
        closeTimer = null;
      }
      studyNavItem.classList.add('is-open');
      studyNavToggle.setAttribute('aria-expanded', 'true');
    };

    const scheduleCloseStudyMenu = () => {
      if (closeTimer) window.clearTimeout(closeTimer);
      closeTimer = window.setTimeout(() => {
        studyNavItem.classList.remove('is-open');
        studyNavToggle.setAttribute('aria-expanded', 'false');
        closeTimer = null;
      }, 280);
    };

    studyNavItem.addEventListener('mouseenter', openStudyMenu);
    studyNavItem.addEventListener('mouseleave', scheduleCloseStudyMenu);
    studyNavItem.addEventListener('focusin', openStudyMenu);
    studyNavItem.addEventListener('focusout', () => {
      if (!studyNavItem.contains(document.activeElement)) scheduleCloseStudyMenu();
    });
  }
})();
