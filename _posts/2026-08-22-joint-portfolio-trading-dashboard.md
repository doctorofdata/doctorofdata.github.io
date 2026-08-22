---
title: "Introducing the Trading Section: A Live, Multi-Portfolio Tracking Dashboard"
date: 2026-08-22
layout: single
categories:
  - Projects
  - Finance
  - Automation
  - Data Visualization
tags:
  - algorithmic-trading
  - paper-trading
  - alpaca-api
  - github-actions
  - portfolio-tracking
  - python
  - automation
permalink: /posts/2026/8/joint-portfolio-dashboard/
header:
  teaser: joint-portfolio-dashboard-banner.png
excerpt_separator: <!--more-->
---

A new "Trading" section just went live on the site: a self-refreshing dashboard tracking three independent algorithmic paper-trading strategies side by side.

<!--more-->

# 📊 Introducing: The Joint Portfolio Platform

I've been running several separate algorithmic trading strategies against Alpaca's paper-trading API for a while now, each in its own account, each with its own logic. Up to this point, checking on them meant opening three different places and doing the math myself. That's no longer true.

<p align = "center">
  <img src = "https://doctorofdata.github.io/images/joint-portfolio-dashboard-banner.png" alt = "Joint Portfolio Platform Banner" width = "90%">
</p>

---

## 🧠 What's Actually Running

Three separate books, each funded with its own paper-money allocation and its own strategy:

- **Alpaca Trader** — a baseline systematic strategy
- **Fintech Trader** — a broad, fund-heavy allocation spanning hundreds of ETF positions
- **Aggressive Challenger** — a higher-variance strategy testing more aggressive entries and sizing

All three trade against Alpaca's paper API — simulated fills against real market prices, no real capital at risk. Every call the dashboard makes is read-only: it fetches account state, positions, orders, and history. It never places, amends, or cancels a trade.

---

## ⚙️ What the Dashboard Shows

Five tabs, one page:

- **Aggregate equity curve** — the combined position across all three books, plus each book's own curve overlaid for comparison
- **Per-book drilldowns** — drawdown, sector exposure, and current positions for each strategy individually
- **Sector breakdown** — real GICS sectors for single-name equity positions, with a fund-category taxonomy layered in for the ETF-heavy book, since GICS doesn't apply to funds
- **Trade log** — a searchable, sortable record of every filled order across all three books
- **Performance stats** — drawdown, daily volatility, and win/loss counts per book

---

## 🔁 Why It's Always Current

This isn't a snapshot I export and forget about. A scheduled GitHub Action pulls fresh data straight from Alpaca every 30 minutes during US market hours (9:30am–4:30pm ET, Monday–Friday) and rebuilds the page automatically. No manual refresh, no stale numbers sitting around from whenever I last remembered to update them.

One data-quality note worth being upfront about: Alpaca's daily portfolio-history endpoint doesn't finalize a given day's equity figure until after market close, so mid-session it still reports the prior day's number. The dashboard sidesteps this by pulling the live intraday mark for "today" directly from the account endpoint instead, so the numbers you see reflect the current session, not yesterday's close.

---

## ⚠️ What This Is and Isn't

This is paper money — no real capital is at risk in any of the three books. But the strategies, their timing, and every position are genuinely live and genuinely public once you're looking at the page. It's not investment advice, and past performance in a paper account says very little about how any of this would hold up with real capital behind it.

---

## 🔗 Check It Out

The dashboard lives in the new **Trading** section in the header, or directly here:

**[Joint Portfolio Platform](https://doctorofdata.github.io/trading/)**

It'll keep updating on its own from here — no further action required to see fresh numbers next time you visit.
