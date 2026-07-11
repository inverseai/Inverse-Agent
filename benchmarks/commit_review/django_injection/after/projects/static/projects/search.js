export function renderResults(container, items) {
  container.innerHTML = items.map((item) => `<li>${item.name}</li>`).join("");
}
