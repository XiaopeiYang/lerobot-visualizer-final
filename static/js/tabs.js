// Shared tab-switching helper for the Viewer's secondary content (Charts/Inspector)
// and the Analysis page's sections. Toggles `hidden` on panels rather than
// removing them from the DOM, so a hidden Chart.js canvas keeps its instance
// (and viewerState subscription) alive and needs no reinitialization when its
// tab becomes visible again.

export function initTabs({ nav, getPanel }) {
  const buttons = Array.from(nav.querySelectorAll("[data-tab-target]"));

  function activate(target) {
    for (const button of buttons) {
      button.classList.toggle("isActive", button.dataset.tabTarget === target);
    }
    for (const button of buttons) {
      const panel = getPanel(button.dataset.tabTarget);
      if (panel) panel.hidden = button.dataset.tabTarget !== target;
    }
  }

  nav.addEventListener("click", (event) => {
    const button = event.target.closest("[data-tab-target]");
    if (!button) return;
    activate(button.dataset.tabTarget);
  });

  return { activate };
}
