# Publishing the joint portfolio dashboard on doctorofdata.github.io

Files are already committed to this repo (from the earlier `traders-folder` session's kit):

```
.github/workflows/refresh_joint_dashboard.yml
scripts/build_joint_platform.py
scripts/sector_map.py
scripts/template.html
scripts/.gitignore
```

## One correction from the original plan

The original kit wrote to `docs/trading/...` and told you to point GitHub
Pages at `/docs`. This repo is already a full Jekyll site published from the
**root** of `main` (it has `_config.yml`, `_posts`, `_layouts`, etc.) —
switching Pages to `/docs` would take your whole blog/portfolio/resume
offline, since only the `/docs` subtree would build. So this build now
writes to **`trading/joint_portfolio_platform.html`** at the repo root
instead. Jekyll copies plain `.html` files with no front matter straight
through to `_site` untouched, so the URL comes out exactly the same and
**no Pages setting needs to change**:

```
https://doctorofdata.github.io/trading/joint_portfolio_platform.html
```

## 1. Add 6 repository secrets

Repo → Settings → Secrets and variables → Actions → New repository secret.
Use the real Alpaca paper-trading keys from each book's `.env`
(`/Users/anon/Documents/traders/<book>/.env`):

| Secret name | Comes from |
|---|---|
| `ALPACA_TRADER_KEY_ID` | `alpaca_trader/.env` — `ALPACA_API_KEY_ID` |
| `ALPACA_TRADER_SECRET` | `alpaca_trader/.env` — `ALPACA_API_SECRET_KEY` |
| `FINTECH_TRADER_KEY_ID` | `fintech_trader/.env` — `ALPACA_API_KEY_ID` |
| `FINTECH_TRADER_SECRET` | `fintech_trader/.env` — `ALPACA_API_SECRET_KEY` |
| `AGGRESSIVE_CHALLENGER_TRADER_KEY_ID` | `aggressive_challenger_trader/.env` — `ALPACA_API_KEY_ID` |
| `AGGRESSIVE_CHALLENGER_TRADER_SECRET` | `aggressive_challenger_trader/.env` — `ALPACA_API_SECRET_KEY` |

Paper-trading keys — no real money movable — but treat them as real
credentials: secrets only, never committed.

## 2. Trigger the first run manually

Repo → Actions tab → "Refresh joint portfolio dashboard" → Run workflow.
That first run creates `trading/joint_portfolio_platform.html` and commits
it — you don't create that file yourself. Confirm the run succeeds, then
check the URL above. After that it refreshes automatically every 30 minutes,
9:30am–4:30pm ET, Monday–Friday (cron line in the workflow file; GitHub's
scheduler is best-effort and can slip a few minutes under load).

## 3. If you add `assigned_asset_trader` back later, or a 5th book

Edit `BOOKS` and `ENV_PREFIX` at the top of `scripts/build_joint_platform.py`,
add the matching secrets, done.

## What's public once this is live

Current positions, sector/category weights, and full fill-by-fill trade
timing history for all 3 books, refreshed every 30 minutes. Paper money, not
real capital — but the strategy and its timing are visible to anyone with
the URL, and it'll be crawlable/indexable like any other page on the site
unless you add `robots.txt`/`noindex` rules yourself.

## Housekeeping

Two harmless `.bak` files (`scripts/build_joint_platform.py.bak`,
`.github/workflows/refresh_joint_dashboard.yml.bak`) were created as a
side effect of a sed edit and couldn't be auto-deleted (file-delete
permission was declined) — they're excluded from git via `.gitignore`
(`*.bak`) so they won't get committed, but feel free to delete them by
hand whenever convenient.
