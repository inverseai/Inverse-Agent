export function renderResults(container, items) {
  container.replaceChildren();
  for (const item of items) {
    const row = document.createElement("li");
    row.textContent = item.name;
    container.append(row);
  }
}
