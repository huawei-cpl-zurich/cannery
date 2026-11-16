/* Delegated so copied commands and disclosure controls survive HTMX refreshes. */
document.addEventListener('click', async event => {
  document.querySelectorAll('.download-menu[open]').forEach(menu => {
    if (!menu.contains(event.target)) menu.open = false;
  });
  const button = event.target.closest('button[data-copy]');
  if (!button) return;
  const source = document.getElementById(button.dataset.copy);
  const status = button.closest('.download-panel, main').querySelector('.copy-status');
  try {
    await navigator.clipboard.writeText(source.textContent);
    status.textContent = 'Copied to clipboard.';
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(source); selection.removeAllRanges(); selection.addRange(range);
    status.textContent = 'Text selected. Press Ctrl+C or ⌘C to copy.';
  }
});
document.addEventListener('keydown', event => {
  if (event.key !== 'Escape') return;
  document.querySelectorAll('.download-menu[open]').forEach(menu => {
    menu.open = false; menu.querySelector('summary').focus();
  });
});
