/* Light/dark override (design review 2026-10-08, R4). Loaded in <head> so a
   stored choice applies before first paint. No choice stored = follow the
   device (the CSS media query). Nothing is sent anywhere. */
(function () {
  var KEY = 'theme', root = document.documentElement, stored = null;
  try { stored = localStorage.getItem(KEY); } catch (e) {}
  if (stored === 'light' || stored === 'dark') root.setAttribute('data-theme', stored);

  function isDark() {
    var t = root.getAttribute('data-theme');
    return t ? t === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  }
  function paintMeta(theme) {
    var metas = document.querySelectorAll('meta[name="theme-color"]');
    for (var i = 0; i < metas.length; i++) {
      var m = metas[i];
      if (!m.dataset.orig) m.dataset.orig = m.content;
      m.content = theme === 'dark' ? '#161614' : '#fafaf8';
    }
  }
  function label(btn) {
    btn.setAttribute('aria-label', isDark() ? 'Switch to the light theme' : 'Switch to the dark theme');
  }
  document.addEventListener('DOMContentLoaded', function () {
    var btn = document.querySelector('[data-theme-toggle]');
    if (!btn) return;
    label(btn);
    if (stored) paintMeta(stored);
    btn.addEventListener('click', function () {
      var next = isDark() ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem(KEY, next); } catch (e) {}
      paintMeta(next);
      label(btn);
    });
  });
})();
