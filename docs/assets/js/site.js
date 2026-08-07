(() => {
  const sidebar = document.getElementById("sidebar");
  const menu = document.getElementById("menu-button");
  const scrim = document.getElementById("scrim");
  const search = document.getElementById("docs-search");
  const links = [...document.querySelectorAll("[data-nav]")];

  const normalize = (path) => path.replace(/index\.html$/, "").replace(/\/+$/, "/");
  const current = normalize(window.location.pathname);
  links.forEach((link) => {
    if (normalize(new URL(link.href).pathname) === current) link.classList.add("active");
  });

  const closeMenu = () => {
    sidebar.classList.remove("open");
    menu.setAttribute("aria-expanded", "false");
    scrim.hidden = true;
  };

  menu.addEventListener("click", () => {
    const open = !sidebar.classList.contains("open");
    sidebar.classList.toggle("open", open);
    menu.setAttribute("aria-expanded", String(open));
    scrim.hidden = !open;
  });
  scrim.addEventListener("click", closeMenu);
  links.forEach((link) => link.addEventListener("click", closeMenu));

  search.addEventListener("input", () => {
    const query = search.value.trim().toLowerCase();
    links.forEach((link) => { link.hidden = query !== "" && !link.textContent.toLowerCase().includes(query); });
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "/" && document.activeElement !== search) {
      event.preventDefault();
      search.focus();
    }
    if (event.key === "Escape") {
      search.value = "";
      search.dispatchEvent(new Event("input"));
      search.blur();
      closeMenu();
    }
  });
})();
