# doctorofdata.github.io — Data Science Portfolio (with receipts)

Welcome to the repository behind my personal site: **`doctorofdata.github.io`**  
Think: *data science + storytelling + shipping things that work.*

## TL;DR
- **Live site:** `https://doctorofdata.github.io`
- **What’s here:** the code + content that powers the portfolio
- **What it’s not:** a museum exhibit of the original Academic Pages README

---

## What you’ll find (human-friendly map)

### Content (the stuff you actually came for)
- Blog posts / writing: usually in `_posts/`
- Pages (about, projects, etc.): typically in `_pages/` (or root pages, depending on setup)
- Static assets (images, PDFs, etc.): commonly `images/`, `files/`

### Site plumbing (necessary, but not the vibe)
- Jekyll configuration: `_config.yml`
- Theme/layout guts: `_layouts/`, `_includes/`, `_sass/`, `assets/`
- Styling: SCSS/CSS (because typography is a feature)

### Data/automation helpers
- `markdown_generator/` — scripts/notebooks to generate markdown (e.g., publications/talks) from structured files.

> If you’re looking for the “how do I run this locally?” section, it’s below.  
> If you’re looking for enlightenment, try the projects page.

---

## Run locally (so you can preview before you break the internet)

### Option A — Ruby/Jekyll (classic)
1. Install Ruby + Bundler (and Node if your setup needs it)
2. Install dependencies:
   ```bash
   bundle install
   ```
3. Serve locally:
   ```bash
   bundle exec jekyll serve -l -H localhost
   ```
4. Open:
   - `http://localhost:4000`

### Option B — Docker (less “works on my machine”, more “works.”)
If this repo includes a Docker setup (e.g., `Dockerfile` / `docker-compose.yml`):
```bash
docker compose up
```

---

## Update workflow (how this site changes)
Typical edits:
- Add a post → commit → push → GitHub Pages rebuilds → site updates.
- Change styling → tweak SCSS → refresh locally → commit when it looks good everywhere.

---

## Credits (because we stand on shoulders)
This site is built on the **Academic Pages** template (itself based on **Minimal Mistakes**).  
Template credit is real; template README copy-paste is optional.

---

## Contact / questions
If you found a bug, typo, or broken link: open an issue or PR.  
If you found a plot hole in my analysis: definitely open an issue.

*(Now go read the portfolio—this README*