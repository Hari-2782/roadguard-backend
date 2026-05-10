(() => {
  const storageKey = 'roadguard-theme';
  const prefersLight = window.matchMedia('(prefers-color-scheme: light)');

  function getPreferredTheme() {
    const saved = localStorage.getItem(storageKey);
    if (saved === 'light' || saved === 'dark') {
      return saved;
    }
    return prefersLight.matches ? 'light' : 'dark';
  }

  function getThemeMeta(theme) {
    return theme === 'light'
      ? {
          label: 'Light mode',
          nextLabel: 'Dark mode',
          icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="4"></circle><path d="M12 2v2.5M12 19.5V22M4.93 4.93l1.77 1.77M17.3 17.3l1.77 1.77M2 12h2.5M19.5 12H22M4.93 19.07l1.77-1.77M17.3 6.7l1.77-1.77"></path></svg>`
        }
      : {
          label: 'Dark mode',
          nextLabel: 'Light mode',
          icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.8A9 9 0 1111.2 3a7 7 0 009.8 9.8z"></path></svg>`
        };
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    const toggle = document.querySelector('[data-theme-toggle]');
    if (toggle) {
      const meta = getThemeMeta(theme);
      toggle.setAttribute('aria-label', `Switch to ${meta.nextLabel.toLowerCase()}`);
      toggle.innerHTML = `${meta.icon}<span>${meta.label}</span>`;
    }
  }

  function setTheme(theme) {
    localStorage.setItem(storageKey, theme);
    applyTheme(theme);
  }

  applyTheme(getPreferredTheme());

  document.addEventListener('DOMContentLoaded', () => {
    if (document.querySelector('[data-theme-toggle]')) {
      return;
    }

    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'theme-toggle';
    button.setAttribute('data-theme-toggle', 'true');
    document.body.appendChild(button);
    applyTheme(document.documentElement.getAttribute('data-theme') || getPreferredTheme());

    button.addEventListener('click', () => {
      const nextTheme = document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
      setTheme(nextTheme);
    });
  });

  prefersLight.addEventListener('change', (event) => {
    if (!localStorage.getItem(storageKey)) {
      applyTheme(event.matches ? 'light' : 'dark');
    }
  });
})();
