---
title: "Microsoft MINDS Data: A Machine Learning Recommendation Engine"
date: 2026-04-02
layout: single
categories:
  - Projects
  - Machine Learning
  - Market-Basket
  - EDA
  - Recommendations
tags:
  - Machine Learning
  - Market-Basket
  - EDA
  - Recommendations
permalink: /posts/2026/4/minds-recommendation-engine/
header:
  teaser: recommendations-generator.png
excerpt_separator: <!--more-->
---

In this post, build a series of recommendation engines for the Microsoft MINDS dataset using popular heuristic strategies and a combination of machine learning algorithms.

<!--more-->

<a class = "anchor" id = "top"></a>

<center><img src = "/images/recommendations-generator.png"></center>

# 📰 News Data: Microsoft MIND — Two-Stage Generate-&-Rerank News Recommendation Engine

The **MIND** dataset is the standard benchmark for neural news recommendation, released by Microsoft Research. It contains ~160 K users, ~65 K articles, and 1 M+ click-through logs collected from MSN News in October 2019.

In this post we build a **two-stage generate-and-rerank** paradigm from large-scale recommendation systems:

| Stage | What it does |
|-------|-------------|
| **Stage 1 — Retrieval** | Cast a wide net: merge candidates from popularity, category-affinity, item-CF, and recency signals |
| **Stage 2 — Ranking** | Re-score every candidate with a LightGBM meta-ranker that sees retriever membership, base scores, and rich user/article features |

<a class="anchor" id="top"></a>

---

## 📋 Table of Contents

| # | Section |
|---|---------|
| **1** | [Setup & data loading](#sec-1) |
| **2** | [Exploratory data analysis](#sec-2) |
| **3** | [Feature engineering](#sec-3) |
| **4** | [Article-based collaborative filtering](#sec-4) |
| **5** | [Temporal sequence modelling](#sec-5) |
| **6** | [Evaluation harness & S1–S5 strategies](#sec-6) |
| **7** | [S6 architecture & cold-start gate](#sec-7) |
| **8** | [Stage 1 — Expanded candidate pool](#sec-8) |
| **9** | [Stage 2 — Meta-ranker training](#sec-9) |
| **10** | [Full benchmark: S1 → S7](#sec-10) |
| **11** | [Visualisations](#sec-11) |
| **12** | [Leaderboard & takeaways](#sec-12) |


---

## 🗺️ System Blueprint — How It All Fits Together

Before diving into the code, here's a bird's-eye view of the **entire two-stage pipeline** you'll build in this notebook:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     MIND News Recommendation Engine                             │
│                                                                                 │
│   ┌──────────┐    ┌──────────┐    ┌─────────────────────────────────────────┐  │
│   │  Raw     │    │  EDA &   │    │         FEATURE STORE                   │  │
│   │  Data    │───▶│  Stats   │───▶│  user_stats · article_feat              │  │
│   │(MIND TSV)│    │(Sec 2)   │    │  user_cat_affinity · TF-IDF centroids   │  │
│   └──────────┘    └──────────┘    └───────────────┬─────────────────────────┘  │
│                                                   │                             │
│              ┌────────────────────────────────────▼─────────────────────────┐  │
│              │                 STAGE 1 — RETRIEVAL (Sec 8)                  │  │
│              │                                                               │  │
│              │  S1 Popularity  S2 Category  S3 Item-CF  S4 Temporal Taste   │  │
│              │       ↓               ↓            ↓            ↓            │  │
│              │              MERGE & DEDUPLICATE                              │  │
│              │              200-candidate pool  (Recall@200 ~diagnostic)    │  │
│              └────────────────────────┬──────────────────────────────────────┘  │
│                                       │                                         │
│              ┌────────────────────────▼──────────────────────────────────────┐  │
│              │                 STAGE 2 — RERANKING (Sec 9)                   │  │
│              │                                                               │  │
│              │   Base LightGBM (LambdaMART, SET_A) ──▶ OOF scores           │  │
│              │   Meta-LGB (extended features, SET_B) ──▶ S6                 │  │
│              │   XGBoost ensemble blend ──────────────────▶ S7              │  │
│              └────────────────────────┬──────────────────────────────────────┘  │
│                                       │                                         │
│              ┌────────────────────────▼──────────────────────────────────────┐  │
│              │         EVALUATION (Sec 10–12)                                │  │
│              │   Precision · Recall · F1 · NDCG · Hit-Rate  @ K=5 & K=10   │  │
│              └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Notebook Roadmap

| Section | Focus | Key Output |
|---------|-------|------------|
| §1 Setup | Load MIND-small ZIPs | `all_interactions`, `news` DataFrames |
| §2 EDA | Statistical exploration | 8 visualisations, CTR & sparsity stats |
| §3 Features | Engineer 4 feature tables | `user_stats`, `article_feat`, `user_cat_affinity`, `imp_train_df` |
| §4 Item-CF | Sparse co-click similarity | `item_sim_lookup` (top-50 neighbours per article) |
| §5 Temporal | Recency-weighted taste | `temporal_taste_matrix` with 7-day half-life decay |
| §6 Eval + S1–S5 | Baseline strategies | Metrics for 5 retrieval/simple-rank methods |
| §7 Cold-start gate | Handle zero-history users | Binary cold/warm routing logic |
| §8 Stage 1 | Candidate pool fusion | 200 candidates, Recall@200 diagnostic |
| §9 Stage 2 | Meta-ranker training | `meta_lgb` model with enriched features |
| §10–12 | Full benchmark | S1→S7 leaderboard, lift metrics |

> **Reading tip:** Each section opens with a `📖` callout explaining the *why* before the code shows the *how*.


---

## <a id="sec-1"></a>1. Setup & data loading

### 📖 Dataset & Problem Framing

**The data:** MIND-small contains **~1 M impression logs** from 50,000 users over six weeks (Oct 12–Nov 22, 2019). Each impression records a user session: the articles shown, which ones were clicked (label=1) or ignored (label=0), and the user's recent click history.

| File | Key columns | Role |
|------|-------------|------|
| `behaviors.tsv` | `ImpressionId, UserId, Time, History, Impressions` | Primary signal — click/no-click |
| `news.tsv` | `NewsId, Category, SubCategory, Title, Abstract` | Article metadata |

**Task framing.** Given a user's click history, rank candidate news articles so that clicked articles appear at the top. We evaluate with ranking metrics (Precision@K, Recall@K, NDCG@K, Hit-Rate@K).

**Train/test split strategy.** MIND provides an official train split and a dev (validation) split. We use train behaviors for all model fitting and dev behaviors as the held-out test set, preserving the temporal ordering of the original benchmark.

**Implicit feedback.** Unlike star-ratings, every click is a positive signal (label = 1); every article shown but not clicked is a negative (label = 0). We treat clicks as our "liked" items throughout.



### 🔍 Data Schema at a Glance

**`behaviors.tsv`** — one row per user session (impression):

```
ImpressionId | UserId | Time                  | History              | Impressions
─────────────┼────────┼───────────────────────┼──────────────────────┼──────────────────────────────
imp-1234     | U5678  | 10/15/2019 8:32:01 AM | N1001 N1087 N2334 …  | N3301-1 N2201-0 N4412-0 …
                                                 ↑ past click IDs       ↑ candidate-label pairs
```

Each entry in `Impressions` is `newsId-label` where label=1 means clicked, label=0 means skipped. **This is the core supervision signal.**

**`news.tsv`** — one row per article:

```
newsId | category  | subCategory | title                          | abstract
───────┼───────────┼─────────────┼────────────────────────────────┼──────────────
N1001  | Sports    | NFL         | "Eagles defeat Cowboys 31-14"  | "The Philadelphia Eagles …"
N1087  | Finance   | Stocks      | "Apple earnings beat Q3"       | "Apple Inc. reported …"
```

> **Key insight:** The recommendation task is *session-level re-ranking*, not global ranking. For each impression, you rank the ~10–20 candidate articles shown in that session, using the user's click history as context.



```python
# Import libraries
import subprocess, sys

for pkg in ['lightgbm', 'xgboost', 'scikit-learn']:

    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check = True)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import zipfile, gc, time, warnings, os, re
from datetime import datetime
from collections import defaultdict, Counter
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import lightgbm as lgb
from xgboost import XGBClassifier
from joblib import Parallel, delayed
from google.colab import drive
import zipfile

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
pd.set_option('display.float_format', '{:.4f}'.format)
pd.set_option('display.max_columns', None)

print('✅  Libraries loaded')
```

    ✅  Libraries loaded



```python
# Connect to data
drive.mount('/content/drive')
```

    Mounted at /content/drive



```python
# Function to perform parsing
def parse_behaviors_from_zip(zip_path, inner_path):

    with zipfile.ZipFile(zip_path, 'r') as z:

        with z.open(inner_path) as f:

            raw = pd.read_csv(f, sep = '\t', header = None, names = BEH_COLS)

    raw['time'] = pd.to_datetime(raw['time'], format='%m/%d/%Y %I:%M:%S %p')
    raw['ts']   = raw['time'].astype('int64') // 10**9

    rows = []

    for _, r in raw.iterrows():

        uid = r['userId']
        ts  = r['ts']

        if pd.notna(r['impressions']):

            for pair in str(r['impressions']).split():

                nid, lbl = pair.rsplit('-', 1)
                rows.append((uid, nid, int(lbl), ts))

    df = pd.DataFrame(rows, columns=['userId','newsId','clicked','timestamp'])

    return df, raw

# Function to perform metrics for ranking
def precision_at_k(recs, true_set, k):

    return len(set(recs[:k]) & true_set) / k if k else 0.0

def recall_at_k(recs, true_set, k):

    return len(set(recs[:k]) & true_set) / len(true_set) if true_set else 0.0

def f1_at_k(recs, true_set, k):

    p = precision_at_k(recs, true_set, k)
    r = recall_at_k(recs,    true_set, k)

    return 2*p*r/(p+r) if (p+r) > 0 else 0.0

def ndcg_at_k(recs, true_set, k):

    dcg   = sum(1 / np.log2(i + 2) for i, m in enumerate(recs[:k]) if m in true_set)
    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(true_set), k)))

    return dcg/ideal if ideal else 0.0

def score_recs(recs, true_set, K):

    return {'precision': precision_at_k(recs, true_set, K),
            'recall'   : recall_at_k(recs,    true_set, K),
            'f1'       : f1_at_k(recs,        true_set, K),
            'ndcg'     : ndcg_at_k(recs,      true_set, K),
            'hit_rate' : 1 if any(m in true_set for m in recs[:K]) else 0,}

def evaluate_strategy(score_fn, eval_df, K = 10, n = None):

    # score_fn(uid, candidates) -> candidates sorted best-first
    rows = eval_df if n is None else eval_df.sample(n=n, random_state=100)
    m = {k: [] for k in ('precision','recall','f1','ndcg','hit_rate')}

    for _, row in rows.iterrows():

        recs = score_fn(row['userId'], row['imp_candidates'])
        s    = score_recs(recs, row['true_items'], K)

        for k in m:

            m[k].append(s[k])

    result = {k: float(np.mean(v)) for k, v in m.items()}

    # FIX 4: composite = mean(NDCG, Hit-Rate) — avoids double-counting P/R via F1
    result['composite'] = float(np.mean([result['ndcg'], result['hit_rate']]))

    return result

metric_keys = ['precision','recall','f1','ndcg','hit_rate']

def parse_history_length(raw_df):

    raw_df = raw_df.copy()
    raw_df['history_len'] = raw_df['history'].fillna('').apply(lambda h: len(str(h).split()) if str(h).strip() else 0)

    return raw_df.groupby('userId')['history_len'].max()

def daily_agg(df, split_label):

    tmp = df.copy()
    tmp['date']  = pd.to_datetime(tmp['timestamp'], unit='s').dt.date
    tmp['split'] = split_label

    return (tmp.groupby(['date','split']).agg(impressions=('clicked','count'), clicks=('clicked','sum')).reset_index().assign(ctr=lambda d: d['clicks']/d['impressions']))

# Function to filter previously seen articles
def _filter_seen(article_list, uid):

    seen = _seen_cache.get(uid, set())

    return [a for a in article_list if a not in seen]

# Ranking metrics
def s1_popularity(uid, N = 50):

    return _filter_seen(POPULARITY_POOL, uid)[:N]

def s2_category(uid, N = 50):

    if uid not in user_cat_affinity.index:

        return s1_popularity(uid, N)

    uvec = user_cat_affinity.loc[uid].values.astype('float32')
    uvec_n = uvec / (np.linalg.norm(uvec) + 1e-9)
    scores = article_cat_norm @ uvec_n
    ranking = np.argsort(-scores)
    ordered = [article_cat_idx[i] for i in ranking]

    return _filter_seen(ordered, uid)[:N]

def s3_itemcf(uid, N = 50):

    clicked = list(user_click_sets.get(uid, []))

    if not clicked:

        return s1_popularity(uid, N)

    score_acc = defaultdict(float)

    for aid in clicked[-20:]:

        for n_aid, sim in item_sim_lookup.get(aid, [])[:30]:

            score_acc[n_aid] += sim

    seen = _seen_cache.get(uid, set())
    ranked = sorted(score_acc.items(), key=lambda x: -x[1])
    filtered = [a for a, _ in ranked if a not in seen]

    if len(filtered) < N:

        filtered += _filter_seen(POPULARITY_POOL, uid)[:N]

    return filtered[:N]

def s4_temporal(uid, N = 50):

    if uid not in user_taste_norm.index:

        return s1_popularity(uid, N)

    tvec = user_taste_norm.loc[uid].values.astype('float32')
    scores = article_cat_taste_norm @ tvec
    ranking = np.argsort(-scores)
    ordered = [taste_article_idx[i] for i in ranking]

    return _filter_seen(ordered, uid)[:N]

# Compute tfidf centroids and resulting article affinity
def tfidf_affinity(uid, aid):

    '''Cosine sim: user click-history TF-IDF centroid vs article.'''
    centroid = user_tfidf_centroids.get(uid)
    if centroid is None:
        return 0.0
    i = tfidf_idx.get(aid, -1)
    if i < 0:
        return 0.0
    return float(tfidf_mat[i].dot(centroid))

def recent_tfidf_affinity(uid, aid):

    '''Cosine sim using centroid of user recent 20 clicks only.'''
    centroid = user_recent_tfidf_centroids.get(uid)
    if centroid is None:
        return 0.0
    i = tfidf_idx.get(aid, -1)
    if i < 0:
        return 0.0
    return float(tfidf_mat[i].dot(centroid))

# Evaluation scoring functions
def s1_score(uid, candidates):

    return sorted(candidates, key = lambda a: -float(pop_stats.loc[a,'bayesian_ctr'] if a in pop_stats.index else 0))

def s2_score(uid, candidates):

    if uid not in user_cat_affinity.index:

        return s1_score(uid, candidates)

    uvec = user_cat_affinity.loc[uid].values.astype('float32')
    uvec /= np.linalg.norm(uvec) + 1e-9

    def _s(a):

        i = art_pos.get(a, -1)
        return float(article_cat_norm[i] @ uvec) if i >= 0 else 0.0

    return sorted(candidates, key=lambda a: -_s(a))

def s3_score(uid, candidates):

    clicked = list(user_click_sets.get(uid, []))

    if not clicked:

        return s1_score(uid, candidates)

    score_acc = defaultdict(float)

    for aid in clicked[-20:]:

        for n_aid, sim in item_sim_lookup.get(aid, [])[:30]:

            score_acc[n_aid] += sim

    return sorted(candidates, key=lambda a: -score_acc.get(a, 0))

def s4_score(uid, candidates):

    if uid not in user_taste_norm.index:

        return s1_score(uid, candidates)

    tvec = user_taste_norm.loc[uid].values.astype('float32')

    # Use taste_pos dict (O(1)) instead of list.index() (O(n))
    def _s(a):

        i = taste_pos.get(a, -1)

        return float(article_cat_taste_norm[i] @ tvec) if i >= 0 else 0.0

    return sorted(candidates, key=lambda a: -_s(a))

def _build_feature_matrix(uid, candidates, s2_vec, s4_vec):

    '''Build the full FEATURE_COLS-aligned matrix for all impression candidates.
    Includes within-impression context signals (ctr_norm_rank, imp_size).'''
    n    = len(candidates)
    u_cc = float(us_click_count.get(uid, 0))
    u_cf = float(us_click_freq.get(uid,  0))
    ctrs = np.array([af_bayesian_ctr.get(a, 0) for a in candidates], dtype='float32')
    ctr_norm_rank = np.argsort(np.argsort(-ctrs)).astype('float32') / max(1, n - 1)
    rows = []

    for k, a in enumerate(candidates):

        ai = art_pos.get(a, -1)
        ti = taste_pos.get(a, -1)
        subc = newsid_to_subcat.get(a)
        rows.append([
            u_cc,
            u_cf,
            float(af_log_clicks.get(a,   0)),
            float(af_log_impr.get(a,     0)),
            float(af_article_len.get(a,  0)),
            float(s2_vec[ai]) if ai >= 0 else 0.0,
            float(s4_vec[ti]) if ti >= 0 else 0.0,
            tfidf_affinity(uid, a),
            recent_tfidf_affinity(uid, a),
            float(af_article_age.get(a,  0)),
            float(ctr_norm_rank[k]),
            float(n),
            float(user_subcat_clicks.get((uid, subc), 0)) if subc else 0.0,])

    return np.array(rows, dtype='float32')

def s5_score(uid, candidates):

    if uid in user_cat_affinity.index:

        uvec   = user_cat_affinity.loc[uid].values.astype('float32')
        s2_vec = article_cat_norm @ (uvec / (np.linalg.norm(uvec) + 1e-9))

    else:

        s2_vec = np.zeros(len(article_cat_idx))

    if uid in user_taste_norm.index:

        tvec   = user_taste_norm.loc[uid].values.astype('float32')
        s4_vec = article_cat_taste_norm @ tvec

    else:

        s4_vec = np.zeros(len(taste_article_idx))

    X = _build_feature_matrix(uid, candidates, s2_vec, s4_vec)
    probs = lgb_model.predict(X)

    return [candidates[i] for i in np.argsort(-probs)]

def s6_score(uid, candidates):

    if is_cold(uid):

        return s1_score(uid, candidates)

    if uid in user_cat_affinity.index:

        uvec   = user_cat_affinity.loc[uid].values.astype('float32')
        s2_vec = article_cat_norm @ (uvec / (np.linalg.norm(uvec) + 1e-9))

    else:

        s2_vec = np.zeros(len(article_cat_idx))

    if uid in user_taste_norm.index:

        tvec   = user_taste_norm.loc[uid].values.astype('float32')
        s4_vec = article_cat_taste_norm @ tvec

    else:

        s4_vec = np.zeros(len(taste_article_idx))

    X_base      = _build_feature_matrix(uid, candidates, s2_vec, s4_vec)
    base_scores = lgb_model.predict(X_base)
    cands_s2    = s2_category(uid, N_STAGE1)
    cands_s3    = s3_itemcf(uid,   N_STAGE1)
    cands_s4    = s4_temporal(uid, N_STAGE1)
    X_meta      = _build_meta_features(uid, candidates, cands_s2, cands_s3, cands_s4, s2_vec, s4_vec, base_scores)
    scores = meta_lgb.predict(X_meta)

    return [candidates[i] for i in np.argsort(-scores)]

def s7_score(uid, candidates):

    if is_cold(uid):

        return s1_score(uid, candidates)

    if uid in user_cat_affinity.index:

        uvec   = user_cat_affinity.loc[uid].values.astype('float32')
        s2_vec = article_cat_norm @ (uvec / (np.linalg.norm(uvec) + 1e-9))

    else:

        s2_vec = np.zeros(len(article_cat_idx))

    if uid in user_taste_norm.index:

        tvec   = user_taste_norm.loc[uid].values.astype('float32')
        s4_vec = article_cat_taste_norm @ tvec

    else:

        s4_vec = np.zeros(len(taste_article_idx))

    X_base      = _build_feature_matrix(uid, candidates, s2_vec, s4_vec)
    base_scores = lgb_model.predict(X_base)
    cands_s2    = s2_category(uid, N_STAGE1)
    cands_s3    = s3_itemcf(uid,   N_STAGE1)
    cands_s4    = s4_temporal(uid, N_STAGE1)
    X_meta      = _build_meta_features(uid, candidates, cands_s2, cands_s3, cands_s4, s2_vec, s4_vec, base_scores)
    lgb_probs = meta_lgb.predict(X_meta)
    xgb_probs = xgb_meta.predict_proba(X_meta)[:, 1]
    scores    = 0.6 * lgb_probs + 0.4 * xgb_probs
    return [candidates[i] for i in np.argsort(-scores)]

def _build_feature_row(uid, aid, s2_scores_dict, s4_scores_dict):

    '''Used by s5_lgb retriever (not evaluation path).'''
    ai      = art_pos.get(aid, -1)
    ti      = taste_pos.get(aid, -1)
    cat_aff = float(s2_scores_dict.get(ai, 0))
    tst_aff = float(s4_scores_dict.get(ti, 0))
    return [
        float(us_click_count.get(uid, 0)),
        float(us_click_freq.get(uid,  0)),
        float(af_log_clicks.get(aid,   0)),
        float(af_log_impr.get(aid,     0)),
        float(af_bayesian_ctr.get(aid, 0)),
        float(af_article_len.get(aid,  0)),
        cat_aff,
        tst_aff,
        tfidf_affinity(uid, aid),
        float(af_article_age.get(aid, 0)),]

def _build_meta_features(uid, candidates, cands_s2, cands_s3, cands_s4, s2_vec, s4_vec, lgb_base_scores):

    s2_rank = {a: r for r, a in enumerate(cands_s2)}
    s3_rank = {a: r for r, a in enumerate(cands_s3)}
    s4_rank = {a: r for r, a in enumerate(cands_s4)}
    n    = len(candidates)
    ctrs = np.array([af_bayesian_ctr.get(a, 0) for a in candidates], dtype='float32')
    ctr_norm_rank = np.argsort(np.argsort(-ctrs)).astype('float32') / max(1, n - 1)
    rows = []

    for k, aid in enumerate(candidates):

        ai      = art_pos.get(aid, -1)
        ti      = taste_pos.get(aid, -1)
        cat_aff = float(s2_vec[ai]) if ai >= 0 else 0.0
        tst_aff = float(s4_vec[ti]) if ti >= 0 else 0.0
        in_s2   = int(aid in s2_rank)
        in_s3   = int(aid in s3_rank)
        in_s4   = int(aid in s4_rank)
        rows.append([
            float(us_click_count.get(uid, 0)),
            float(us_click_freq.get(uid,  0)),
            float(af_log_clicks.get(aid,   0)),
            float(af_log_impr.get(aid,     0)),
            float(af_article_len.get(aid,  0)),
            cat_aff, tst_aff,
            tfidf_affinity(uid, aid),
            recent_tfidf_affinity(uid, aid),
            float(af_article_age.get(aid, 0)),
            float(ctr_norm_rank[k]),
            float(n),
            float(user_subcat_clicks.get((uid, newsid_to_subcat.get(aid)), 0))
            if newsid_to_subcat.get(aid) else 0.0,
            in_s2, in_s3, in_s4,
            s2_rank.get(aid, N_STAGE1),
            s3_rank.get(aid, N_STAGE1),
            s4_rank.get(aid, N_STAGE1),
            in_s2 + in_s3 + in_s4,
            float(lgb_base_scores[k]),
        ])

    return np.array(rows, dtype='float32')

def s5_lgb(uid, N = 50):

    candidates = list(dict.fromkeys(s2_category(uid, K_CAND) + s3_itemcf(uid, K_CAND) + s4_temporal(uid, K_CAND)))[:K_CAND]

    if not candidates:

        return s1_popularity(uid, N)

    if uid in user_cat_affinity.index:

        uvec   = user_cat_affinity.loc[uid].values.astype('float32')
        s2_vec = article_cat_norm @ (uvec / (np.linalg.norm(uvec) + 1e-9))

    else:

        s2_vec = np.zeros(len(article_cat_idx))

    if uid in user_taste_norm.index:

        tvec   = user_taste_norm.loc[uid].values.astype('float32')
        s4_vec = article_cat_taste_norm @ tvec

    else:

        s4_vec = np.zeros(len(taste_article_idx))

    X     = _build_feature_matrix(uid, candidates, s2_vec, s4_vec)
    probs = lgb_model.predict(X)

    return [candidates[i] for i in np.argsort(-probs)][:N]

def s6_meta_lgb(uid, N = 50):

    if is_cold(uid):

        return s1_popularity(uid, N)

    cands_s2   = s2_category(uid, N_STAGE1)
    cands_s3   = s3_itemcf(uid,   N_STAGE1)
    cands_s4   = s4_temporal(uid, N_STAGE1)
    candidates = list(dict.fromkeys(cands_s2 + cands_s3 + cands_s4))[:N_STAGE1]

    if uid in user_cat_affinity.index:

        uvec   = user_cat_affinity.loc[uid].values.astype('float32')
        s2_vec = article_cat_norm @ (uvec / (np.linalg.norm(uvec) + 1e-9))

    else:

        s2_vec = np.zeros(len(article_cat_idx))

    if uid in user_taste_norm.index:

        tvec   = user_taste_norm.loc[uid].values.astype('float32')
        s4_vec = article_cat_taste_norm @ tvec

    else:

        s4_vec = np.zeros(len(taste_article_idx))

    X_base      = _build_feature_matrix(uid, candidates, s2_vec, s4_vec)
    base_scores = lgb_model.predict(X_base)
    X_meta      = _build_meta_features(uid, candidates, cands_s2, cands_s3, cands_s4, s2_vec, s4_vec, base_scores)
    scores = meta_lgb.predict(X_meta)

    return [candidates[i] for i in np.argsort(-scores)][:N]

def s7_ensemble(uid, N = 50):

    if is_cold(uid):

        return s1_popularity(uid, N)

    cands_s2   = s2_category(uid, N_STAGE1)
    cands_s3   = s3_itemcf(uid,   N_STAGE1)
    cands_s4   = s4_temporal(uid, N_STAGE1)
    candidates = list(dict.fromkeys(cands_s2 + cands_s3 + cands_s4))[:N_STAGE1]

    if uid in user_cat_affinity.index:

        uvec   = user_cat_affinity.loc[uid].values.astype('float32')
        s2_vec = article_cat_norm @ (uvec / (np.linalg.norm(uvec) + 1e-9))

    else:

        s2_vec = np.zeros(len(article_cat_idx))

    if uid in user_taste_norm.index:

        tvec   = user_taste_norm.loc[uid].values.astype('float32')
        s4_vec = article_cat_taste_norm @ tvec

    else:

        s4_vec = np.zeros(len(taste_article_idx))

    X_base      = _build_feature_matrix(uid, candidates, s2_vec, s4_vec)
    base_scores = lgb_model.predict(X_base)
    X_meta      = _build_meta_features(uid, candidates, cands_s2, cands_s3, cands_s4,
                                       s2_vec, s4_vec, base_scores)
    lgb_probs = meta_lgb.predict(X_meta)
    xgb_probs = xgb_meta.predict_proba(X_meta)[:, 1]
    scores    = 0.6 * lgb_probs + 0.4 * xgb_probs

    return [candidates[i] for i in np.argsort(-scores)][:N]

COLD_THRESHOLD = 2

def is_cold(uid):

    if uid not in user_stats.index:

        return True

    return user_stats.loc[uid, 'click_count'] < COLD_THRESHOLD

def _raw_s2(uid, N):

    if uid not in user_cat_affinity.index:

        return POPULARITY_POOL[:N]

    uvec = user_cat_affinity.loc[uid].values.astype('float32')
    uvec = uvec / (np.linalg.norm(uvec) + 1e-9)

    return [article_cat_idx[j] for j in np.argsort(-(article_cat_norm @ uvec))[:N]]

def _raw_s3(uid, N):

    clicked = list(user_click_sets.get(uid, []))

    if not clicked:

        return POPULARITY_POOL[:N]

    score_acc = defaultdict(float)

    for aid in clicked[-20:]:

        for n_aid, sim in item_sim_lookup.get(aid, [])[:30]:

            score_acc[n_aid] += sim

    ranked = [a for a, _ in sorted(score_acc.items(), key=lambda x: -x[1])]

    return (ranked + POPULARITY_POOL)[:N]

def _raw_s4(uid, N):

    if uid not in user_taste_norm.index:

        return POPULARITY_POOL[:N]
    tvec = user_taste_norm.loc[uid].values.astype('float32')

    return [taste_article_idx[j] for j in np.argsort(-(article_cat_taste_norm @ tvec))[:N]]

def chunked_topn(A_norm, U_mat, article_idx_arr, n_top, rank_col):

    parts = []

    for start in range(0, n_users, CHUNK_SIZE):

        end       = min(start + CHUNK_SIZE, n_users)
        u_batch   = unique_users[start:end]
        scores    = A_norm @ U_mat[start:end].T
        top_idx   = np.argsort(-scores, axis=0)[:n_top]
        chunk_len = end - start
        parts.append(pd.DataFrame({
            'userId': np.repeat(u_batch, n_top),
            'newsId': article_idx_arr[top_idx.T.ravel()],
            rank_col: np.tile(np.arange(n_top), chunk_len),
        }))
        del scores, top_idx

    gc.collect()
    return pd.concat(parts, ignore_index = True)
```


```python
# Load the data
TRAIN_ZIP = 'drive/MyDrive/MINDsmall_train.zip'
DEV_ZIP   = 'drive/MyDrive/MINDsmall_dev.zip'

# Quick sanity-check: list contents of each archive
for label, path in [('TRAIN', TRAIN_ZIP), ('DEV', DEV_ZIP)]:

    with zipfile.ZipFile(path, 'r') as z:

        print(f'{label} ZIP contents: {z.namelist()}')
```

    TRAIN ZIP contents: ['MINDsmall_train/', 'MINDsmall_train/behaviors.tsv', 'MINDsmall_train/news.tsv', 'MINDsmall_train/entity_embedding.vec', 'MINDsmall_train/relation_embedding.vec']
    DEV ZIP contents: ['MINDsmall_dev/', 'MINDsmall_dev/behaviors.tsv', 'MINDsmall_dev/news.tsv', 'MINDsmall_dev/entity_embedding.vec', 'MINDsmall_dev/relation_embedding.vec']



```python
# Define columns of interest
NEWS_COLS = ['newsId','category','subCategory','title','abstract','url', 'titleEntities','abstractEntities']
BEH_COLS = ['impressionId', 'userId', 'time', 'history', 'impressions']

print('Loading train news...', end=' ', flush = True)

# Load the data from file
with zipfile.ZipFile(TRAIN_ZIP, 'r') as z:

    with z.open('MINDsmall_train/news.tsv') as f:

        news_train = pd.read_csv(f, sep = '\t', header = None, names = NEWS_COLS, usecols = ['newsId', 'category', 'subCategory', 'title', 'abstract'])

print(f'done  ({len(news_train):,} articles)')

print('Loading dev news...  ', end = ' ', flush = True)

# Load the data from file
with zipfile.ZipFile(DEV_ZIP, 'r') as z:

    with z.open('MINDsmall_dev/news.tsv') as f:

        news_dev = pd.read_csv(f, sep = '\t', header = None, names = NEWS_COLS, usecols = ['newsId','category','subCategory','title','abstract'])

print(f'done  ({len(news_dev):,} articles)')
```

    Loading train news... done  (51,282 articles)
    Loading dev news...   done  (42,416 articles)



```python
# Merge ther files together
news = pd.concat([news_train, news_dev]).drop_duplicates('newsId').reset_index(drop = True)

# Fill empty cells
news['abstract'] = news['abstract'].fillna('')
news['text']     = news['title'] + ' ' + news['abstract']

print(f'\nUnique articles : {len(news):,}')
print(f'Categories      : {news["category"].nunique()}')
print(f'Sub-categories  : {news["subCategory"].nunique()}')
news.head()
```

    
    Unique articles : 65,238
    Categories      : 18
    Sub-categories  : 270






  <div id="df-df71511c-f805-48dc-bf5c-851a889b9da6" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>newsId</th>
      <th>category</th>
      <th>subCategory</th>
      <th>title</th>
      <th>abstract</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>N55528</td>
      <td>lifestyle</td>
      <td>lifestyleroyals</td>
      <td>The Brands Queen Elizabeth, Prince Charles, an...</td>
      <td>Shop the notebooks, jackets, and more that the...</td>
      <td>The Brands Queen Elizabeth, Prince Charles, an...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>N19639</td>
      <td>health</td>
      <td>weightloss</td>
      <td>50 Worst Habits For Belly Fat</td>
      <td>These seemingly harmless habits are holding yo...</td>
      <td>50 Worst Habits For Belly Fat These seemingly ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>N61837</td>
      <td>news</td>
      <td>newsworld</td>
      <td>The Cost of Trump's Aid Freeze in the Trenches...</td>
      <td>Lt. Ivan Molchanets peeked over a parapet of s...</td>
      <td>The Cost of Trump's Aid Freeze in the Trenches...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>N53526</td>
      <td>health</td>
      <td>voices</td>
      <td>I Was An NBA Wife. Here's How It Affected My M...</td>
      <td>I felt like I was a fraud, and being an NBA wi...</td>
      <td>I Was An NBA Wife. Here's How It Affected My M...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>N38324</td>
      <td>health</td>
      <td>medical</td>
      <td>How to Get Rid of Skin Tags, According to a De...</td>
      <td>They seem harmless, but there's a very good re...</td>
      <td>How to Get Rid of Skin Tags, According to a De...</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-df71511c-f805-48dc-bf5c-851a889b9da6')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-df71511c-f805-48dc-bf5c-851a889b9da6 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-df71511c-f805-48dc-bf5c-851a889b9da6');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>





```python
# Expand each impression list into one row per (user, article, label) from the behavioral data
print('Parsing train behaviors...', end = ' ', flush = True)
interactions_train, raw_train = parse_behaviors_from_zip(TRAIN_ZIP, 'MINDsmall_train/behaviors.tsv')
print(f'done  ({len(interactions_train):,} rows)')

print('Parsing dev behaviors...  ', end = ' ', flush = True)
interactions_dev, raw_dev = parse_behaviors_from_zip(DEV_ZIP, 'MINDsmall_dev/behaviors.tsv')
print(f'done  ({len(interactions_dev):,} rows)')

# Tag splits and combine
interactions_train['split'] = 'train'
interactions_dev['split']   = 'dev'

all_interactions = pd.concat([interactions_train, interactions_dev], ignore_index = True)

print(f'\nTotal interactions : {len(all_interactions):,}')
print(f'  Train            : {len(interactions_train):,}')
print(f'  Dev              : {len(interactions_dev):,}')
```

    Parsing train behaviors... done  (5,843,444 rows)
    Parsing dev behaviors...   done  (2,740,998 rows)
    
    Total interactions : 8,584,442
      Train            : 5,843,444
      Dev              : 2,740,998



```python
all_interactions.head()
```





  <div id="df-717072fa-fc55-4610-a635-d3495748e74a" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>newsId</th>
      <th>clicked</th>
      <th>timestamp</th>
      <th>split</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>U13740</td>
      <td>N55689</td>
      <td>1</td>
      <td>1573463158</td>
      <td>train</td>
    </tr>
    <tr>
      <th>1</th>
      <td>U13740</td>
      <td>N35729</td>
      <td>0</td>
      <td>1573463158</td>
      <td>train</td>
    </tr>
    <tr>
      <th>2</th>
      <td>U91836</td>
      <td>N20678</td>
      <td>0</td>
      <td>1573582290</td>
      <td>train</td>
    </tr>
    <tr>
      <th>3</th>
      <td>U91836</td>
      <td>N39317</td>
      <td>0</td>
      <td>1573582290</td>
      <td>train</td>
    </tr>
    <tr>
      <th>4</th>
      <td>U91836</td>
      <td>N58114</td>
      <td>0</td>
      <td>1573582290</td>
      <td>train</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-717072fa-fc55-4610-a635-d3495748e74a')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-717072fa-fc55-4610-a635-d3495748e74a button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-717072fa-fc55-4610-a635-d3495748e74a');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>





```python
all_interactions['split'].value_counts()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>split</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>train</th>
      <td>5843444</td>
    </tr>
    <tr>
      <th>dev</th>
      <td>2740998</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>




```python
all_interactions['userId'].nunique()
```




    94057




```python
# Split the data for training
train_clicks = interactions_train[interactions_train['clicked'] == 1]
train_clicks['newsId'] = train_clicks['newsId'].astype(str)

test_clicks = interactions_dev[interactions_dev['clicked'] == 1]
test_clicks['newsId'] = test_clicks['newsId'].astype(str)
```


```python
# Compile the ground truths
_seen_cache = train_clicks.groupby('userId')['newsId'].apply(set).to_dict()
ground_truth = (test_clicks.groupby('userId')['newsId'].apply(set).rename('true_items'))
```


```python
# Gather the users
train_users   = set(train_clicks['userId'].unique())
test_users    = set(ground_truth.index)
warm_users    = train_users & test_users
cold_users    = test_users - train_users

print(f'Train positive clicks  : {len(train_clicks):,}')
print(f'Dev   positive clicks  : {len(test_clicks):,}')
print(f'Unique train users     : {len(train_users):,}')
print(f'Unique test  users     : {len(test_users):,}')
print(f'Warm users (train test): {len(warm_users):,}')
print(f'Cold users (test only) : {len(cold_users):,}')
```

    Train positive clicks  : 236,344
    Dev   positive clicks  : 111,383
    Unique train users     : 50,000
    Unique test  users     : 50,000
    Warm users (train test): 5,943
    Cold users (test only) : 44,057



```python
# Parse raw_dev into per-impression evaluation rows.
# Each impression is one independent ranking query: candidates = articles shown
# in that session, true_items = what was clicked. Keeping sessions separate
# prevents global popularity from dominating via cross-session aggregation.
eval_rows = []

for _, r in raw_dev.iterrows():

    uid = r['userId']

    if uid not in warm_users or pd.isna(r['impressions']):

        continue

    pairs   = str(r['impressions']).split()
    cands   = [p.split('-')[0] for p in pairs]
    clicked = {p.split('-')[0] for p in pairs if p.endswith('-1')}

    if not clicked:

        continue

    eval_rows.append({'userId'        : uid,
                      'impressionId'  : r['impressionId'],
                      'imp_candidates': cands,
                      'true_items'    : clicked})

eval_df   = pd.DataFrame(eval_rows)
eval_warm = eval_df.reset_index(drop = True)

print(f'Eval impressions          : {len(eval_warm):,}')
print(f'Unique warm users         : {eval_warm["userId"].nunique():,}')
print(f'Avg candidates/impression : {eval_warm["imp_candidates"].apply(len).mean():.1f}')
print(f'Avg clicks/impression     : {eval_warm["true_items"].apply(len).mean():.2f}')
eval_warm.head()
```

    Eval impressions          : 8,959
    Unique warm users         : 5,943
    Avg candidates/impression : 37.6
    Avg clicks/impression     : 1.52






  <div id="df-6f442c1c-fcea-4f2c-b9de-ee6684b71a27" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>impressionId</th>
      <th>imp_candidates</th>
      <th>true_items</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>U44035</td>
      <td>24</td>
      <td>[N37204, N48487, N59933, N512, N51776, N64077,...</td>
      <td>{N37204, N496}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>U88867</td>
      <td>66</td>
      <td>[N20036, N36786, N50055, N2960, N5940, N32536,...</td>
      <td>{N31958, N23513}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>U80349</td>
      <td>69</td>
      <td>[N31958, N5472, N36779, N29393, N34130, N23513...</td>
      <td>{N29393}</td>
    </tr>
    <tr>
      <th>3</th>
      <td>U61801</td>
      <td>70</td>
      <td>[N20036, N53242, N6916, N48487, N36940, N46917...</td>
      <td>{N5940}</td>
    </tr>
    <tr>
      <th>4</th>
      <td>U54826</td>
      <td>82</td>
      <td>[N29363, N44289, N7344, N6340, N4610, N40943, ...</td>
      <td>{N7344}</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-6f442c1c-fcea-4f2c-b9de-ee6684b71a27')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-6f442c1c-fcea-4f2c-b9de-ee6684b71a27 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-6f442c1c-fcea-4f2c-b9de-ee6684b71a27');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>





```python
# Count clicks per article across the training split
pop_counts = train_clicks.groupby('newsId')['clicked'].count().rename('click_count')

# Bayesian-smoothed score: (clicks + C*global_rate) / (impressions + C)
total_impressions = interactions_train.groupby('newsId')['clicked'].count().rename('impressions')

# Global click-through rate
GLOBAL_CTR = train_clicks.shape[0] / len(interactions_train)

# Smoothing constant
C = 50

pop_stats = (pop_counts.to_frame().join(total_impressions).fillna(0))
pop_stats['bayesian_ctr'] = ((pop_stats['click_count'] + C * GLOBAL_CTR) / (pop_stats['impressions'] + C)).astype('float32')

# Articles ranked by training CTR
#train_ranked = pop_stats.sort_values('bayesian_ctr', ascending=False).index.tolist()

# Dev articles not seen in training are appended so they are still reachable by every retriever
#train_pool_set  = set(train_ranked)
#unseen_articles = [a for a in news['newsId'].astype(str) if a not in train_pool_set]
#POPULARITY_POOL = train_ranked + unseen_articles

POPULARITY_POOL = pop_stats.sort_values('bayesian_ctr', ascending = False).index.tolist()

print(f'Popularity pool  : {len(POPULARITY_POOL):,} training articles')
print(f'Global CTR       : {GLOBAL_CTR:.4f}')
pop_stats.sort_values('impressions', ascending=False).head(6)
```

    Popularity pool  : 7,713 training articles
    Global CTR       : 0.0404






  <div id="df-df1be03f-d38b-4e11-a187-87be1a79c0be" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>click_count</th>
      <th>impressions</th>
      <th>bayesian_ctr</th>
    </tr>
    <tr>
      <th>newsId</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>N47061</th>
      <td>820</td>
      <td>23037</td>
      <td>0.0356</td>
    </tr>
    <tr>
      <th>N51048</th>
      <td>1875</td>
      <td>19242</td>
      <td>0.0973</td>
    </tr>
    <tr>
      <th>N26262</th>
      <td>1139</td>
      <td>19106</td>
      <td>0.0596</td>
    </tr>
    <tr>
      <th>N50872</th>
      <td>279</td>
      <td>18702</td>
      <td>0.0150</td>
    </tr>
    <tr>
      <th>N55689</th>
      <td>4316</td>
      <td>18315</td>
      <td>0.2351</td>
    </tr>
    <tr>
      <th>N38779</th>
      <td>1490</td>
      <td>18101</td>
      <td>0.0822</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-df1be03f-d38b-4e11-a187-87be1a79c0be')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-df1be03f-d38b-4e11-a187-87be1a79c0be button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-df1be03f-d38b-4e11-a187-87be1a79c0be');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>




[Back to top](#top)

---

## <a id="sec-2"></a>2. Exploratory data analysis

### 📖 Understanding the data before modelling

This section answers eight key questions before building any model:

1. How are clicks distributed across articles? (power law expected)
2. How active are individual users?
3. Which categories dominate the corpus?
4. How does CTR vary by category?
5. What is the article title-length distribution?
6. How do click volumes trend over time?
7. What fraction of users have very thin histories (cold-start risk)?
8. How much overlap exists between train and dev article pools?



```python
# Compile high-level stats
n_users   = all_interactions['userId'].nunique()
n_articles= all_interactions['newsId'].nunique()
n_impr    = len(all_interactions)
n_clicks  = all_interactions['clicked'].sum()
overall_ctr = n_clicks / n_impr

print(f'{"Users":<30} {n_users:>10,}')
print(f'{"Articles":<30} {n_articles:>10,}')
print(f'{"Total impressions":<30} {n_impr:>10,}')
print(f'{"Total clicks":<30} {n_clicks:>10,}')
print(f'{"Overall CTR":<30} {overall_ctr:>10.4f}')
print(f'{"Sparsity":<30} {1 - n_clicks/(n_users*n_articles):>10.6f}')
```

    Users                              94,057
    Articles                           22,771
    Total impressions               8,584,442
    Total clicks                      347,727
    Overall CTR                        0.0405
    Sparsity                         0.999838



> **Interpreting the headline numbers:**
> - **~3–5% CTR** is typical for editorial news feeds. Random chance would yield ~10% (1 click in 10 shown), so position bias and user selectivity drive CTR well below that.
> - **Matrix sparsity > 99.9%** means collaborative filtering on raw co-clicks alone is brittle — content and temporal signals are essential complements.
> - The gap between unique *articles* and unique *users* (~65K vs ~50K) tells you the article space is only slightly larger than the user space in this small subset, which is atypically dense for a real-world recommender.



```python
# Compile the clicks distribution and user activity
article_clicks = train_clicks.groupby('newsId')['clicked'].count()
user_clicks    = train_clicks.groupby('userId')['clicked'].count()

fig, axes = plt.subplots(1, 3, figsize=(21, 5))
fig.suptitle('MIND - Small: Click distributions', fontsize = 14, fontweight = 'bold')

# (a) Article click histogram (log scale)
ax = axes[0]
ax.hist(np.log1p(article_clicks.values), bins = 60, color = 'steelblue', edgecolor = 'white', lw = 0.4)
ax.set_xlabel('log(1 + clicks per article)')
ax.set_ylabel('Number of articles')
ax.set_title('(a) Article popularity (log scale)')
top5 = article_clicks.nlargest(5)

# Iterate
for nid, cnt in top5.items():

    title = news.set_index('newsId').loc[nid, 'title'] if nid in news['newsId'].values else nid
    ax.axvline(np.log1p(cnt), color='red', lw=0.8, alpha=0.5)

# (b) User activity histogram
ax = axes[1]
ax.hist(np.log1p(user_clicks.values), bins=60, color='darkorange', edgecolor='white', lw=0.4)
ax.set_xlabel('log(1 + clicks per user)')
ax.set_ylabel('Number of users')
ax.set_title('(b) User activity (log scale)')

# (c) Click count CDF for articles
ax = axes[2]
sorted_clicks = np.sort(article_clicks.values)
cdf = np.arange(1, len(sorted_clicks)+1) / len(sorted_clicks)
ax.plot(np.log1p(sorted_clicks), cdf, color='purple', lw=2)
ax.axhline(0.8, color='grey', ls='--', lw=1)
ax.set_xlabel('log(1 + clicks)')
ax.set_ylabel('CDF')
ax.set_title('(c) Article popularity CDF')

# Find where 80% of articles have fewer than X clicks
p80_idx = np.searchsorted(cdf, 0.8)
ax.annotate(f'80% articles ≤ {sorted_clicks[p80_idx]} clicks',
            xy=(np.log1p(sorted_clicks[p80_idx]), 0.8),
            xytext=(np.log1p(sorted_clicks[p80_idx])+0.5, 0.65),
            arrowprops=dict(arrowstyle='->', color='black'), fontsize=9)

plt.tight_layout()
plt.savefig('eda_click_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
```


    
![png](data:image/png;base64,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m2++qRo1apjH6IsXL7aoc/jwYfXs2VO1atVShQoV9Nprr+nvv/82lydPDbtx40a1atVKNWvWTPN4N2/e1Pvvv6+aNWsqICBADRs21OTJkxUbG2tRb+fOnXrppZcUFBSk+vXr66OPPlJkZKQuXbokf39/xcTEaPjw4fLz85NkufRDp06d1KFDhxTHnjdvnvz9/XX79m3GqYxTAQDAQ5bW2PfGjRsaOnSoqlWrpsqVK6tbt24KDg42l3N/mPvDUvrj7tjYWH3//fdq166dypUrl6Lc3d1dixYt0oIFC9I8n2RWVlYaNmyYnJycMlUfwKOHRAUAFrZs2SJ3d3dVrVrVvC0yMlLdu3eXra2tfvzxR23YsEEdO3bUmDFj9Ouvv1rs37BhQx05ckTXr19P9zgrVqxQ5cqV9eyzz6pmzZoqUqRImgO/uXPnauLEiZozZ45GjhypRo0aqVChQtq9e7d69OiRon54eLhef/11mUwmLV68WN9++63u3r2rHj16KCYmJkX9+Ph4de3aVQcOHNDYsWO1bt06tWnTRqNGjdKaNWsy7LPRo0frtdde008//aTevXtrwYIF+vrrr83lffr00a+//qqxY8dq48aNGjZsmNauXav33nvPop2TJ09qzZo1mjFjhlasWKFChQqpf//+unz5coYxpOWvv/7Su+++q3r16mnDhg1avny5SpQood69e6f4O/r666/10ksvafPmzXJxcbEou7/fhw4dqiZNmmjNmjUymUzmeqGhofrzzz/TvQG8ZcsWVa9eXa6urmnWWbFihYYPH67GjRtrzZo1WrhwoRISEvT666/r2rVrkqRTp05p4MCBeuaZZ7R8+XKNGjVK06ZNy3Ct3S+//FI///yzJkyYoM2bN+uzzz7Tv//+q3fffddcZ9CgQfr99981depUrVmzRoGBgerdu7f+/fdfSdKcOXM0f/58vfvuu9q6datmz56ty5cva8CAAWked9++ferdu7cKFCig77//Xt9++63i4+PVuXNnhYaGmutVrFhRXl5e2rp1a7rnAQAAclZq4+Rkn376qapUqaJVq1bpgw8+0KZNmyzeALKyslKDBg20c+dOxcfHZ1tMffr0kaurqxYvXqwNGzaoW7dumjJlinnq07Nnz6pr164yGAyaP3++li1bpkKFCqlHjx4pxkxz5szRwIEDtXr16jSPN3ToUO3f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D3JSk1EfpvZgLtl/6cPkh5GpyUof3j55W9vHbtfV41e1L2Gf9P+zoNu72cuzhKcc3B1UrX81BbwSkKKtH374QS1btpSdnZ0SExP12muv6f3331fZsmVT1E2vD11cXNJ9cza969vLyytF8lIyJyenB+5DT09PeXp6mj8bDAbdvHlTNjY2j813xL1y6jvi3n56XL4jkmXUh9n1HXH/tfSof0fcL73r283NzTxTRGoy24fJfeTh4SEbG5vH4jviXjn1HZHcT/b29rKxsXksviPuld717eDgkOZ+97POdE0AAAAAAAAAADJQrlw5BQYGatq0aYqMjFRISIgWLFigjh07SpKaNWtmnga6YcOGWrFihY4fP67ExETt3r1bf/zxhxo1apSbpwA8FnaO26mjS48q9GCobv7zvynxg7oEqe+Rvuqxu4cCOqRMUpCS/h1++umniouLU8eOHbVlyxb5+/vrlVde0cGDB3PqFAAATzFmVAAAAAAAAAAAZKvPP/9co0ePVu3ateXq6qoOHTqoU6dOkqSzZ8+ap3Lu3bu3EhMT1a9fP4WGhqpo0aIaP358iqnBAUiGeIMOzD+g64evKzEmUWe2/m9qbStrK9k52ylPqTyq0rtKptrr37+//vjjD61evVq//fab9u/frylTpqhSpUpq3ry5RowYoTp16jys0wEAPOVIVAAAAAAAAAAAZKtChQpp/vz5qZadOHHC/LudnZ0GDRqkQYMG5VBkwOPrn2X/aGP/jSkLrKT3It6To6tjltqzsrLS/PnzFRwcrH79+umPP/5Qjx499OOPP2rixIl67rnn9Nxzz2nEiBFq2rSprKyssulMAABg6QcAAADggezatUu1atXS4MGDU5Rt2LBBrVq1UsWKFdW+fXvt3r3bXGY0GjVjxgw1atRIVatWVc+ePXXx4kVzeXh4uAYNGqRatWqpTp06GjlypGJjY83lx44dU+fOnVW5cmU1adJE33zzzcM9UQAAAABArgkNCdWxVccU/H2wjq86nmqdZ9o+Izsnuwdq38XFRRs3btTw4cMlSba2turUqZOOHDmin376SXFxcWrevLkqV66sFStWyGAwPPC5AABwLxIVAAAAgCyaP3++xo8fL29v7xRlx44d07Bhw/TOO+/ozz//VLdu3dS/f39du3ZNkrRkyRKtW7dO8+bN0/bt21WiRAn169dPJpNJ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AACgJLjiQgUAAAAgUFWqVEn9+vWTJNlsNj3xxBPGBgIAAAAClMlkMjoCAAAAShAKFQAAAFAiDB06VD///LPWrl2r1NRUSVL16tV19913q1q1aganAwAAAAAAAICSw+xJp6NHjzpeF7QEGAAAABDoPvvsM3Xr1k1btmxRbm6ucnNz9eWXX6pTp0768ccfjY4HAAAAAAAAACWGRysqtG/fXt99952CgoLUvn177dq1q6hzAQAAAD41d+5cvfjii+rYsaPT9hUrVujFF1/UBx98YFAyAAAAwHg2m00ffPCB7HZ7odt69eplRDwAAAAUMx4VKlx33XVq166doqOjlZubq969e7vs+/777/ssHAAAAOArf/zxh9q1a5dve5cuXTRlyhQDEgEAAACBo1KlSpo/f36h20wmE4UKAAAA8AmPChXmz5+vzz77TJmZmbJarWrZsqXPAnz99dcaPXq0mjVrppdeesmxfcWKFXr22WdVunRpp/5Lly5V/fr1lZeXp5dffllr1qzRqVOnVL9+fU2YMMHxfOGMjAxNmDBB33//vcxms1q3bq1x48YpJCTEZ9kBAABw9ahSpYp27dqlxo0bO21PTExUhQoVDEoFAAAABIaNGzcaHQEAAAAliEeFCpUqVVK/fv0knV/u64knnvDJwRctWqQPP/xQNWrUKLC9SZMmeueddwpsW7p0qVavXq1FixYpOjpaL730kh5//HGtXLlSJpNJ48aNU25urtasWaOzZ89q6NChmjFjhsaOHeuT7AAAALi6PPjgg3r00UfVuXNn1apVS5L0yy+/aPXq1RowYIDB6QAAAIDAYLfbtXv3bh06dEgWi0W1atVyzJ8BAAAAX/GoUOFiQ4cO1c8//6y1a9cqNTVVklS9enXdfffdjtUMPBUcHKwPP/xQkydPVk5OjlfvXbZsmfr16+eYJA8fPlzNmjXTrl27VLVqVa1fv14ff/yxoqKiJEmDBg3S0KFDNXr06HyrNAAAAKD46927typVqqSPPvpI27dvV25urqpXr66JEyeqY8eORscDAAAADPfdd99pzJgxOnTokMLDw3Xu3DllZWUpNjZWkydPVnx8vNERS4wT+08o+2S27Hl22fPskl3nX9vt+bddtL2gbfY8u84cP2P0KQEAADjxulDhs88+01NPPaW6deuqevXqkqQvv/xS8+bN05tvvqmbb77Z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AAAAgGvMq8+73Hn1iRMnmHsC8DkKFQBclSpUqKCWLVvq5ZdfVkZGhk6ePKnZs2erWbNmuvbaa9WkSROdOHFC77//vnJzc/XRRx/pt99+c7y/du3aSktL08mTJ/2S9+6779bhw4f1wQcfKDc3V59++qlmzZqlsmXLunxP3bp1FRcXp9mzZ+v06dPat2+fxowZo+zsbEcfs9msunXrauDAgRo9erQyMzP15ptv6h//+IeOHDkiSTpw4IBOnjzpWMrsUkFBQXr11VeVnZ2t/fv369NPP1Xbtm0lSb169dJnn32mzZs369y5c/r222919913a9euXR6dd0xMjI4cOaLDhw/rzz//1IQJE1SpUiUdPXrU5XuCg4N1+PBhnTp1SufOnXNqu/fee/Xee+/pl19+0enTpzV9+nT9+OOP+fbhSe6ff/5ZderU8eg8AAAAjMK897xAn/dWrVpVFotFa9eu1blz5/Txxx/r8OHDjvbevXtr0aJFysnJ0dmzZ7Vr1y7VqFHjio7NfBYAAMBzzKvPu9x59YEDB5h7AvA5ChUAXLVeeOEFhYaGqkOHDurYsaPCwsL08ssvSzpfPTp58mS98soruvXWW7V371517drV8dvzCQkJCg0NdaomHTt2rBISEvTII49Ikm6++WYlJCTok08+KTTHtGnTCl2mS5IqVqyoN954Q2+99ZaaNGmihQsX6tVXX3VbhTp//nylpqaqRYsWGjhwoAYNGqRWrVrl6zdgwABFRUVp6tSpeuihh1SnTh3dc889atiwoYYNG6ZRo0apbt26BR4jPDxcderU0Z133qkePXrojjvuUO/evSVJt956q0aPHq3nnntOjRs31nPPPacJEyaoYcOGhea+oF27dmrVqpU6duyoXr16qU2bNnrssce0fv16TZ8+vcD3dO7cWQcPHtTtt9/uVP0rSQ888IDuuece9enTR7fffrssFovGjRuXbx+e5P7+++91yy23eHQeAAAARmLe+z+BOu+tWLGiRo0apdmzZ+uWW25RUlKSOnbs6GifPXu2Nm3apFtuuUUtWrTQtm3bNGPGjMs+dmZmpvbs2cN8FgAAwAvMq//Hm3n1/v37deLECeaeAHzOZOehMgCKqdzcXJUuXdoxmRw9erTy8vIcPyCfNm2afvnlFy1cuNDImIZasWKFZs6cqX//+99GR/GrpKQk9ezZUxs2bFClSpWMjgMAAHBFmPe6V9zmvW+99ZZWrFihVatWGR0FAACg2GBeXbDJkycrJSVF8+fPNzoKgGKGFRUAFEtZWVlq3ry5/vWvfykvL0+JiYnasGGDWrdu7ejzyCOPaNeuXU7PGkPJMHfuXPXp04ciBQAAcNVj3lvynD59Wm+99ZaGDBlidBQAAIBig3l1wY4ePapPPvlETzzxhNFRABRDFCoAKJZCQ0P18ssva/ny5WrcuLEGDx6shx9+WJ06dXL0ueaaazR58mQ99dRTTs/pQvG2atUqpaSkaOTIkUZHAQAAuGLMe0ueqVOnqlWrVvrb3/5mdBQAAIBig3l1fna7Xc8884z69++v+Ph4o+MAKIZ49AMAAAAAAAAAAAAAAPAbVlQAAAAAAAAAAAAAAAB+Q6ECAAAAAAAAAAAAAADwGwoVAAAAAAAAAAAAAACA31CoAAAAAAAAAAAAAAAA/IZCBQAAAAAAAAAAAAAA4DcUKgAAAAAAAAAAAAAAAL+hUAEAAAAAAAAAAAAAAPgNhQoAAAAAAAAAAAAAAMBvKFQAAAAAAAAAAAAAAAB+8/9HdKlu2b6DlwAAAABJRU5ErkJggg==
)
    



> **📊 What to look for in these plots:**
>
> | Plot | Expected shape | Why it matters |
> |------|---------------|----------------|
> | Article click histogram | Long-tailed / power law | A few viral articles capture most clicks — popularity bias is strong |
> | User activity histogram | Right-skewed | Most users click < 10 articles; a handful click 100+. Heavy-tail users dominate training signal |
> | Sparsity heatmap | Nearly all-zero | Collaborative filtering must handle extreme sparsity — motivates CF via item similarity rather than direct user–user CF |
>
> A **power-law click distribution** is the single most important structural property of the dataset. It means:
> 1. A popularity baseline (S1) is a surprisingly strong competitor.
> 2. Personalisation gains are concentrated on *heavy users* who have rich histories.
> 3. Cold-start users (zero history) must fall back to popularity.



```python
# Analysis by category
news_lookup = news.set_index('newsId')[['category','subCategory','title']]
train_with_cat = train_clicks.join(news_lookup, on = 'newsId')
all_with_cat   = all_interactions.join(news_lookup, on = 'newsId')

# Compile the stats by cat
cat_stats = (all_with_cat.groupby('category').agg(impressions = ('clicked','count'), clicks = ('clicked','sum')).assign(ctr = lambda d: d['clicks']/d['impressions']).sort_values('impressions', ascending = False))

fig, axes = plt.subplots(1, 2, figsize = (20, 6))

# (a) Volume per category
ax = axes[0]
palette = sns.color_palette('husl', len(cat_stats))
bars = ax.barh(cat_stats.index, cat_stats['impressions'], color=palette)
ax.set_xlabel('Total impressions')
ax.set_title('(a) Impressions per category')
ax.invert_yaxis()
for bar, (_, row) in zip(bars, cat_stats.iterrows()):

    ax.text(bar.get_width()*1.01, bar.get_y()+bar.get_height()/2,
            f'{row["ctr"]:.2%} CTR', va='center', fontsize=8)

# (b) CTR per category (sorted)
ax = axes[1]
cat_ctr = cat_stats.sort_values('ctr', ascending=False)
bars2   = ax.barh(cat_ctr.index, cat_ctr['ctr']*100, color=palette)
ax.set_xlabel('CTR (%)')
ax.set_title('(b) Click-through rate by category')
ax.invert_yaxis()
ax.axvline(GLOBAL_CTR*100, color = 'red', ls = '--', lw = 1.5, label = f'Global CTR {GLOBAL_CTR:.2%}')
ax.legend()

plt.tight_layout()
plt.savefig('eda_categories.png', dpi=150, bbox_inches='tight')
plt.show()

print(cat_stats.to_string())
```


    
![png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAB8UAAAJOCAYAAAAu69ZBAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQABAABJREFUeJzs3Xd8Tvf7x/F3ckeWEHuP1ootiRExitBSu3SIhirVovaKTe0atUdjVmyltra01VKjVipq71HEJiLrzvn94Zf721uGEYTk9Xw8+uh9zvmcc65z7o8k17nO+RwbwzAMAQAAAAAAAAAAAACQAtkmdwAAAAAAAAAAAAAAALwoFMUBAAAAAAAAAAAAACkWRXEAAAAAAAAAAAAAQIpFURwAAAAAAAAAAAAAkGJRFAcAAAAAAAAAAAAApFgUxQEAAAAAAAAAAAAAKRZFcQAAAAAAAAAAAABAikVRHAAAAAAAAAAAAACQYlEUBwAAAAAAAAAAAACkWBTFAQCvrfXr16tcuXI6d+7cE7UfPXq0GjRooPv37z+27e7du+Xm5qY//vgjqWG+8qZMmSI3NzdFREQkdygAAAAAgFfIf/PuJ8mTO3furFatWik6OvqJ93Hw4EF17dpVVapUUcmSJVW1alV9/vnn+u2336zatWjRQh9++KFl2s3NTePGjXvi/fj4+Khbt25P3D4hq1atkpubmy5evJjkbb1osfn+6+jixYtyc3PTkiVLkjsUAEAKQVEcAPBaOnr0qPr376+RI0cqf/78T7ROz549lTZtWvn7+7/g6F4vrVu31vbt2+Xg4JDcobzWdu3aJR8fn+QOAwAAAACei2fJu0eOHKlLly5p7NixT9R+2bJl+uijj+Tg4KCJEyfqp59+0jfffCNXV1e1a9dOEydOTHDd7du3q3379k+0n6SYPHmy+vTp88L3k1JcuHDhtS3EPy1/f39NmTIlucMAADwhiuIAgNfS8OHDVaZMGb3zzjtPvI6dnZ369u2rzZs3p4onwJ9U2rRplTVr1uQO47V34MCB5A4BAAAAAJ6bZ8m7XVxc1K1bNy1YsEAnTpxItO3Ro0c1dOhQtWrVSl9//bXKlSun3Llzq3z58ho7dqw+/fRTzZo1S2fOnIl3/axZsypt2rRPdUzPglzv6aSm85WajhUAUgKK4gCA186uXbu0Z88edejQwWr+wYMH1aZNG3l6eqp06dKqW7euli5datWmTJkyqly58jPdyevm5qZZs2bp66+/VsWKFeXh4SF/f39FRERowoQJqly5ssqXL6++ffsqMjJS0v+G+1q5cqUGDRqkChUqyN3dXV9++aVu3Lhh2baPj4+GDx+uvn37qkyZMvr1118lSdeuXVPv3r3l4+OjUqVKqV69evr++++t4tq8ebOaNm0qT09PeXp6qlmzZtqxY4dl+dGjR9W2bVtVrFjRcl4CAwMty+MbPn3VqlVq0KCBSpUqpbJly6pNmzY6dOiQ1XI3NzcdP35cbdu2lYeHh6pUqaKRI0cqJibG0m7ZsmVq0KCB3N3dVb58ebVu3Vr//PNPguc4dji+rVu3qkuXLvL09FTZsmXVt29fhYWFWdpFRkZq0qRJqlevnkqXLq1q1app3LhxlvMuPRxar0OHDpo4caI8PDy0cOHCBPf7999/q0WLFnJ3d1eVKlXUu3dvXbt2zbL8zJkz6tSpkypUqKCSJUvq7bff1owZMyzH2qdPH02cOFGXLl2Sm5ubpX+FhoZq2LBhql27tkqVKqVatWopICBAhmFYtn3v3j317t1bZcuWtRzrn3/+KTc3N+3evdvS7sCBA/rkk0/k4eGh0qVL67333tOGDRssy2P72ooVK9SsWTOVLFlS3333nYoWLaoLFy5YHe/Vq1dVrFgxLV68OMFzAgAAACD1Sijvlh7mMN27d7fkoL169bLK1959913lz59f06ZNS3QfCxYskJOTkzp37hzv8i5duuj333/Xm2++Ge/yR4dPDwkJUY8ePVShQgWVLVtWrVq1UnBwcIL7v3XrlmrXrq2WLVta5ZL/5ePjox07duiHH36Ik6PdunVLHTt2lIeHh8qWLauhQ4fGuRbw3/zs3r17kp48t3t02PA+ffqocuXKluknzSWlh09vf/LJJypTpoy8vLw0derUBM9L7L4aNWqkJUuWqEKFCvr6668lPbxG0adPH3l7e6tkyZLy8fHR6NGjFR4eLunh9YVevXpJevj9xD5h/yQ5fEIiIyM1ZMgQeXl5qUyZMmrXrp0lX2/SpIlatGgRZ52AgACVLl1ad+7ciXebMTExmjt3rt555x2VLl1aderU0YIFC6zarF27Vu+9957luoivr6/++usvy3I3NzedO3dOU6dOtRpO/++//1abNm1UqVIlubu76+OPP9b+/futtr1v3z41adLEcp1g1apVGjRokNXoc4ZhaPbs2apdu7ZKliypChUqqFOnTlavEJwyZYrKlSunLVu2qEqVKurcufMznxMASA0oigMAXjubN29W+vTpVb58ecu80NBQffrpp7Kzs9Py5cu1ceNG+fr6avDgwZYCcywfHx8dPHhQV69efep9L1u2TOnSpdOyZcvUrVs3rV69Wp988omio6O1aNEide7cWatWrbJKaKWHiUr+/Pm1fPlyffPNN9qzZ0+cYdx///13pU2bVuvWrVPFihUVGRmpTz75RPv27dOQIUO0bt06NWrUSAMGDNDq1aslPSzWdu3aVbVr19aaNWu0YsUKlSxZUp9//rkuX74sSWrXrp1cXFwUGBiojRs3Wu7C37hxY7zH+P3336tv376qVauWVq9erfnz5ysqKkotW7bUlStXrNoOGTJEH3zwgdauXauPPvpI3333nTZt2iRJ2rlzp4YMGaJPP/1UGzZsUGBgoFxdXdW6dWs9ePAg0fM8YsQIVatWTT/88IMGDhyo9evXW5JwSfrqq680Z84cffLJJ1q/fr38/f21YsUKDR482Go7x48f17lz57Ry5Uo1atQo3n2dPXtWrVq1Ut68ebV8+XJNnTpVhw8ftgzDZxiG5XzOnz9fP/30k7p06aJp06Zp0aJFkqT+/furZs2aypEjh7Zv367WrVtLkjp27Kj169erS5cu2rBhg9q2baupU6daXRwaMmSIfv75Zw0aNEgrVqxQtmzZNHToUKsYT548qU8++UTOzs5auHChfvjhB5UtW1bdu3fXli1brNrOmTNH77//vn7++Wc1adJEjo6Olv4Sa9OmTXJwcFCDBg0S/R4AAAAApE7x5d2xJk6cqHLlylkKeT/++KPGjBljWW5jY6MaNWro999/T7To+ddff8nLy0tOTk7xLndyclKWLFmeKN7IyEi1adNG58+f17fffqvly5db8s/4cv/w8HC1b99eadOm1fTp02Vvbx/vdr///ntlypRJ7777rrZv3y4PDw/LstGjR6t+/fpavXq1WrdurUWLFmnNmjVW6/83P0ubNu1T5XaP8yS5ZKzhw4frs88+0+rVq/Xuu+9qypQp2rlzZ6Lbv3XrlrZs2aLAwEB98cUXkqQePXpo7969mj59ujZv3qzBgwdr5cqVlmHuW7duLT8/P0kPh7fv37+/pCfP4eMz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qY4dO2rChAmaNWuWVZvjx4/L399fEyZMUNGiRZNy6gE8IbPZbDXdo0cPderUSX379tXs2bP16aefWpbFxMTIMAzLOmazWTY2NpZpwzC0ePFirVy50rKsXr168vPzU+XKlbVjxw4VLlxYZ86c0RtvvKH3339fHTp0UNmyZa1icHJyUlRUVJzYHvVoPLExjB07VgEBAQoPD9eVK1fUtWtXVahQ4bHbw8vz3z4EpFT0c6QG9PPUje/9+fLy8tKBAwfk4+OjiIgInT9/XiaTSQ8ePJCTk5P27t0rLy8vdejQQZ6enpowYYLlBvavv/5aw4YN04ULF+Tv7685c+bI29tbu3btUuvWreXp6al3331XJ0+e1LZt2yyvTJIeFtvfeustbdu2Td9//71GjBihBg0aKHPmzOrTp4/s7Oy0YcMGmUwmDR48WH379tXcuXMt6//111/69ddflTZt2uQ4bQAAAHhFkS88G/JsJEVC/edJ+1OS3yke648//tCDBw/UqVMn2dvbK1++fPr44481e/Zs9e7dWytXrpS3t7dq1aolSfL29lb16tW1ceNGVatWTZs3b1bz5s2VPXt2SVLbtm31448/WrYfEBAgGxsb2dvbK1OmTKpatar2799vFYOvr69l/SpVqlje+71582a99dZblsLUF198YfWU+uP2LUlVq1ZV/vz5Ezz+Ro0aqUCBApIePlk+Z84cq+U3b95U+/bt1atXL1WpUuUJzyqApAoKCpL08EkPs9ls+Xfs5uamzZs3y8PDw9I2NDRUZ86csVonU6ZMlunjx4/L1dVVZ8+etaxTuHBhde/eXffv39fYsWPVr18/bdq0SQUKFNChQ4d069Yty/qxsmTJovXr18cZKv3EiRMqXLiwZfrKlSuKioqyWj80NFTvvfeeypQpo/DwcPXt21cmkynOPvBqiP09BKRk9HOkBvRzIOkqVqyoVatWSZL+/vtvFS5cWHZ2dvr7779VsWJF7d27V5UrV9aWLVu0ZMkSOTk5ycnJSZ06dVKbNm00dOhQ5cmTR7t27bKMsOTt7a3MmTPrn3/+SfDG8zx58ui9996TJNWrV09fffWVzp8/L+nhSG4bN260bK9nz56qXr26rl27Zlm/cePGcnFxeWHnBQAAAK8nrscmDXk2kuJZ+89zK4pfvHhRefPmlb29vWVe/vz59e+//yomJkYXL15UwYIFrdbJnz+/pbB95coV5cmTx7LsjTfesGp76NAhjR8/XseOHbM8YVmyZEmrNv9d38nJyTK8+dWrV/Xmm29alqVJk8aq7eP2LUm5c+dO9PgT2rckRUdHq3PnzsqWLZvV8HAAXjx3d3dJD/+dL126VPPnz5ednZ1++ukneXp6WpZLUvHixRUQEKAMGTLojTfe0K+//qq6deta2uzfv18VK1a0WifWkCFD5O/vrwoVKuj8+fMyDEOlSpWSvb19nPa9evVSmzZt1LRpUxUvXlyStHLlSn3//fdatGiR7Owe/mjevn27IiMjrdZ3cXFRgQIFLPMGDx6sKVOmaNmyZZb1kPzMZrOCg4NVqlQpmUym5A4HeCHo50gN6OepW1hYmI4fP57cYaQYXl5eGjp0qKKjo7Vnzx55enrKxsZG+/btU9myZS3FcbPZLC8vL6t1zWazbt26pUyZMmnJkiX6/vvvFRISIsMwFBkZqcjIyAT3+99c3dHRUZIUGRlpGV69cePGVu1NJpMuX76sTJkySZJy5cr1PA4fAAAAKUx814jxeOTZSIqE+s+T5u/PrYKSUBIa+yTk45Y/OpSwYRiWz3fu3NHnn38uX19fzZo1Sy4uLpo4caJ27NgR77biiy06OtpqXkzM/95rkdi+Yz3uH2dC+5Yevqcsa9as2rp1q3799Vf5+Pgkui0Az0/sv926devqyJEj8vPzk8lkUpYsWTRixAiZTCbLEIqlS5fW6NGjLU9fZ82aVSNHjrRs4+rVq8qZM2ecnwcHDhxQdHS0ZRSIunXrqkOHDlq5cqU+/vjjOO0LFy6sadOmadSoUQoPD5e9vb0KFCigOXPmyMHBwdLO1tZWNjY2Vuvb2NjI1tbWMq9mzZpat26dZs2apY4dOz7/E4gkMZlM/HGHFI9+jtSAfp468Z0/X7lz51aWLFl0+PBh7d27V23btpUkzZkzR4cPH1amTJlUuHBhOTs768CBA/FuY8WKFQoICND06dNVvnx5mUwmVatWLdH9JpSrxxbI//jjD2XMmDHO8osXL0qiHwAAACB+/J2YNOTZSIpH+8+T9qXnVhTPmzevLly4oMjISMvT4qdPn1aePHlka2urfPny6fTp01brnD59Wnnz5pUkZcuWTZcvX7YsO3nypFW7+/fvq02bNpZhyw4fPvzEsWXLlk1XrlyxTP/3rvDH7ft5yJQpkyZMmKBFixZp4MCBcnd3t9x1DuDlsLGxUc+ePeNd1r9/f8tnb29veXt7x9tuwIAB8c738PCwGoY9Xbp0CgwMTDSeUqVKafHixYm26dSpU5x58W134sSJiW4HAAAAeBV4eXlpz549Cg4Olru7u2JiYhQcHKw9e/bI29tb+fLlU1hYmC5cuGC5VhAaGqqoqChlzJhRwcHBKleunCpWrChJunbtmkJCQp4plty5c8vW1lbHjh2zbC8qKko3b960vFoNAAAAAJBy2D6vDb311luys7PTtGnTFBkZqdOnT2vBggWWocgaNmyoP//8U7/99puio6O1bds2bd261bK8atWqWr58ua5du6abN29q9uzZlm3nypVLtra2OnDggMLCwjR//nxdv35d169fj/MEeEKxbd++XQcPHlR4eLimTp1q9aR4Yvt+HmxtH57m5s2bq3DhwhoyZMhz3T4AAAAAAK+6ihUrasWKFXrjjTfk7OwsFxcX5cyZUz/88IO8vb1VpEgReXh4aMSIEbp586bu3r2rwYMHq3fv3pIeFrJPnz6tO3fu6NKlSxo+fLhy5cqlq1evSpIcHBx07do13b59O9Eh1aWHN7LWrVtX48aN05UrVxQeHq5vvvlGrVu3jnf0OAAAAADA6+25FcXTpk2rgIAAyx3ebdu2VaNGjdSuXTtJsiS248ePV/ny5TVmzBiNGzdOFSpUkPTwHbtvvvmm6tSpo/fff1/vvfee5f242bNnV/fu3dWvXz/VqFFDd+7c0bhx4xQZGanmzZs/NrZ3331XLVu2VLt27VStWrU47/hNbN/Pk42NjUaOHKk///xTa9asee7bBwAAAADgVeXl5aUzZ86obNmylnmenp46deqUZbSm8ePHyzAM1axZU2+//bbMZrNGjx4tSfL19VX+/PlVrVo1ff755/Lz85Ofn5/mzZunRYsWqVatWjIMQ9WrV9ehQ4ceG8/AgQOVP39+1atXT1WrVtXJkyc1ffr0RF+PBgAAgLgMWxudqpBBpypkkGHL31IAXk02BrdAvzLCwsJ05MgR/XNmmMIizid3OECK0O7j/ckdAlIhs9msoKAgubu7824cpFj0c6QG9PPULTY/K1asmJydnZM7HLxiyN8BAABSN647PxvybCRFQv3nSfP35/akOAAAAAAAAAAAAAAArxqK4gAAAAAAAAAAAACAFIuiOAAA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6M0335Svr6+qVq2qyZMnKzk5WZI0bdo0de3aVb1791a5cuU0cuRIhYWFad68eapbt65lG7GxserZs6el77tv3z7Lsl27dqlFixby9fVVtWrVNHXqVMuyNWvWqHHjxlqxYoWqVKkiPz8/hYWF6bvvvtNrr72mcuXKadiwYZb28fHxGjFihGrWrCkfHx+1a9dOv/32m9WxLliwQFWrVtXs2bNT9fF/+eUXvfXWW/Lx8VG9evW0efPm+4oTAAAAAJB12XVRvE+fPqpcubL27dunVatWafv27Vq+fPld12nQoIG6deumMmXKKDIyUsWKFdPSpUv15Zdfavr06Tpw4IAaN26s4OBgXb16VRcvXtSIESM0depURUREaNq0afriiy+sLj4AD4vRaFTu3LklSWfPntWOHTvUsGFDqza3h0zfvHmz2rVrp44dO1ouIJUuXVpbt26VJP3444/6559/dP36dSUnJyshIUGnTp1S0aJFNWHCBHXv3l1z5szRoEGDtGXLllSxHD16VN7e3qnm582b94GPKy4uTlu2bFG5cuUeeF0AAAAAKUJDQzVq1Cjt2bNHBQsW1OTJkyVJf/75pzp27KimTZtq3759mj17tlatWmX1jPCIiAj5+flp//79GjJkiCpWrKgOHTpYRp2SpFWrVqlTp07at2+fKlSooJCQEEkpn+d79OihNm3a6ODBgwoNDdX8+fP17bffWtaNjo7WpUuXtH37dr3zzjsaP368NmzYoP/973+aNWuWli9friNHjkiSJkyYoKNHj2rFihXau3evvL291b17d5nNZsv2tm3bprVr16pz585W5+DmzZvq2rWrXnvtNf34448aOnSoBgwYoFOnTt1XnAAAAEiDq6t0+XLKj6urraMBgDTZ9TPF165dKycnJxmNRhUpUkQVK1bUkSNHVKlSpQfazqpVq9S2bVt5eXlJkjp06KDQ0FDt2LFDZcuWVXJyslz//z+CV155RXv27JGDg11/HwEPmclksnp9/Phx9ezZUyEhISpcuLDVcpPJpJs3b8rT01Nt2rTR+vXr1b9/f61atUohISEaO3as/ve//6lixYp67rnn5OjoqPbt2+vdd9/VCy+8oFu3bik2NlZ58uRRQkKCQkJC1LlzZ9WpU8cqr11cXJSYmJgqttuvb/+bnJwss9ls1c5sNmv+/PnatGmTkpKSdOrUKXXp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27Ztq/bt28toNMpkMql9+/YKDw9X3bp1H+h8AAAAAMiaFi5cKB8fH02ZMkUODgxiBwAAkKU5OkqNGtk6CgC4K3qegJ3o2rWrunTpIoPBYJl348YNFShQQJJUqFAhXb16VefOndPu3bsVGBiY7rZmzZqlDh06WAriUkqhe9GiRVbbvx9Go1He3t46ceLEAx4RAAAAgKwqJiZG7u7uFMQBAAAAAI+F3fU+vby8tGnTJrVo0UJlypRRly5ddPHiRXXs2FG+vr5q0aKFzp8/LynlLtc333zTav0qVapozZo1kqSff/5Zb775pnx9fVWpUiV99NFHio+PlyQFBARo2bJlkiSTyaQJEyaoSpUqqlixonr16qUbN248voNGthEdHa0ePXrorbfe0vjx45WUlJSqTUxMjPr27avWrVtr6NChiomJkSTlyZMnVdv8+fPr0qVLMpvNOnPmjIoUKaKQkBANGDBAY8aM0cCBA/Xjjz+mWu/o0aPy9vZONT9v3rwPfEzXrl3T9u3bVa5cuQdeFwAAAEDWExQUpP3792vevHmqV6+evLy8dOvWLUkpffatW7eqTZs28vHxUePGjXX06FHLuuvXr1fDhg3l6+urgIAAhYWFWZZNmzZN3bp105w5cyz975CQEMvymzdvasiQIapUqZIqV66sIUOGWB4HFR8frxEjRqhmzZry8fFRu3bt9Ntvvz2mMwIAAJDFJSZKCxak/CQm2joaAEiTXQ6fvnz5cs2aNUtxcXFq3LixOnfurLFjx8rd3V2BgYGaP3++hgwZcs/tfPjhh+rUqZNatmypP//8U8HBwVqxYoXat29v1W7x4sUKDw/XihUrlC9fPvXp00cjR47UxIkTH9UhIpsxmUwqVqyYunfvrgYNGshsNqt79+5atmyZ2rZta2n31FNPqU+fPmrUqJFy5sypoUOHavr06erfv7+ljdlsVnJyskwmkwwGgxo1aqSgoCBVqVJFu3fvlqenp86cOaPixYurVatWCg4OVvny5a3icXFxUWJiokwm013jTk5OltlstmpnNpsVFhamb775Rrdu3dLFixfVu3dv+fn53XN7QFZxO5fJaWRn5DnsAXlu3/i9PzpLlixRu3btVLZsWVWrVk1vv/221fLQ0FB9+umnKly4sLp3767Jkydrzpw5ioqK0oABAzR37lz5+/tr79696tChg8qVK6cX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OHBAf//9d7oF8Vu3bqlz58768MMPVbduXUkpRfeuXbtq7dq1mjx5siRp3759mjhxohYvXmxZ9+jRo3r99dfVr18/y7zJkydr+vTp+vjjjzN/ogAAAABkeTExMSpbtqwcHBg0DgAAAABgX7JdT9jLy0ubNm1SixYtVKZMGXXp0kUXL15Ux44d5evrqxYtWuj8+fOW9tu2bVOTJk3k4+OjgIAALVq0SJIUFhamgIAAq20fPXpUpUuX1qVLl9SuXTtNmDDBsmzJkiVq0KCBypYtq0aNGmnbtm2WZTt27FDjxo3l6+urqlWravz48UpOTn7EZwL3o0KFCpo4caIkKSEhQZcvX1bhwoVTtevatau6dOkig8FgNX/q1Kny8fGRJJ0/f16FCxeWwWBQcnKycubMqUKFCunPP/9UYmKiJkyYoI8++ijdWNauXavSpUtbCuKSVKZMGX311Vd67rnnHvjYfH19dfLkyQdeDwAAAED2ExQUpP3792vevHmqV6+eTp48qbffflsVKlRQpUqVNGzYMN26dcvSPr2+siTFxcWpb9++qlChgurUqaNvv/3Wal/nzp1Tx44dValSJVWqVEl9+/bV33//LSml3+Tl5aUffvhBzZo1k4+Pj1q3bm3VT58xY4aqVq2qSpUqacaMGVbbvte+AwIC9Pnnn6t27doaNmyY9u3bJy8vL8uxeXl5aevWrWrTpo18fHzUuHFjHT16NM1ztn37dlWsWFG//vprBs44AAAAAOBJku2K4pK0fPlyzZo1S+vXr9eePXvUuXNnffDBB/r+++9lMpk0f/58SdKvv/6qXr16qWfPntq/f79GjRqliRMn6rvvvtNrr72mixcvWnV+w8PDVaFChVQFyq1bt2r69OmWYbh79eql3r17648//lBiYqL69OmjQYMG6eDBg1qyZIm+/vrrVB132Na4ceMUEBCgUqVKpfl8vdvDn6flt99+U8uWLfXRRx9ZCuwuLi66ceOGTp06pWLFimnu3Llq0qSJvv76a3300UdaunRpqu0cPXpUr7zySqr5efPmfeDjSUpK0rp161SuXLkHXhcAAABA9rNkyRJVrFhRHTp00IYNG9ShQweVLVtWu3bt0pdffqn9+/frs88+k3T3vrIkzZo1S7/++qs2bdqkVatW6auvvrLa18cff6yCBQvq+++/15YtW3TmzBnNnDnTqs2i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ysvLS6dOnZIk/f7772rdurV8fX31xhtv6OzZs1br/Prrr2rfvr0qVKigypUrKyQkRImJiZbl27ZtU5MmTeTj46OAgAAtWrTo0R9cNjRu3DgFBASoVKlSatiwodWy4sWL6+LFi5bf2c6dO3X58mVdvnzZUvyWJCcnJ+XMmVOSdOnSJZnNZp05c0ZFihRRSEiIBgwYoDFjxmjgwIH68ccfU8Vw9OhReXt7p5qfN2/eBz6ea9euafv27SpXrtwDrwsAAAAA9xIdHa333ntPlSpVUsWKFfXhhx8qJiZG+/btU/ny5bVz507Vr19fPj4+6tixo/766y9t2bJFffv21eHDh+Xt7a2oqCglJydr6tSpqlOnjsqWLauWLVvqp59+suwnICBAn3/+uWrXrq1hw4bJz89P//zzj5o2bWrpc69fv14NGzaUr6+vAgICFBYWZll/4MCB6tOnjyRpzZo1atKkidauXauAgAD5+vqqT58+lj72wIED9cknn2jo0KHy9fVV7dq1dfDgQc2ePVv+/v7y9/fXmjVr7nkOJD3weQAAAAAAZG12+0zx+vXra8OGDerevbtl3jfffKMePXpo165dlnkDBw6Um5ub5s2bpwsXLqhXr16WZTdv3lSnTp3Url07zZkzR5cuXVJwcLDmzp2r9957T7/++qt69eqlzz77TDVq1NCBAwf03nvvycPDQzVq1Hisx5tVmUwmSdIHH3ygHj16aNCgQQoNDdW7775raZMrVy59+umn+vjjj5UzZ05VrlxZTk5OSk5OttqGJJnNZhkMBjVq1EhBQUGqUqWKdu/eLU9PT505c0bFixdXq1atFBwcrPLly1vF4uLiosTERKvtpSU5OVlmsznVfsePH6/Zs2crPj5eFy9eVO/eveXn53fP7WU3t4/X3o4b9oU8hz0gz2EPyHP7lpV/72azWcHBwSpXrpwmT56suLg49e3bV2PHjtXrr7+umzdvatOmTVqxYoVu3rypVq1aaeXKlercubN+++03ff/991q5cqUkaf78+dq0aZNCQ0NVpEgRrVixQt26ddOOHTvk6uoqSdq0aZPmzZsnd3d3de7cWbVr19a6detUsmRJRUVFacCAAZo7d678/f21d+9edejQQeXKldOLL76YKvbo6GgdOXJEGzduVHR0tFq0aKHw8HDLl6M3b96sTz/9VB999JG6d++uvn37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
)
    


                   impressions  clicks    ctr
    category                                 
    news               2232125   95172 0.0426
    lifestyle          1016267   45431 0.0447
    sports              942187   54220 0.0575
    finance             789133   24610 0.0312
    foodanddrink        572554   17579 0.0307
    entertainment       464494   13362 0.0288
    travel              446318   10858 0.0243
    health              441673   15331 0.0347
    autos               382055   10282 0.0269
    tv                  374229   20176 0.0539
    music               358613   19776 0.0551
    movies              243102    7604 0.0313
    video               181367    7076 0.0390
    weather             140130    6246 0.0446
    kids                   166       3 0.0181
    northamerica            29       1 0.0345



> **📊 Category plots — what they tell you:**
>
> The left panel (impression counts) reveals the **supply** of content per category. The right panel (CTR per category) reveals **demand quality** — which categories users *actually engage with* vs. merely see. Gaps between supply and CTR (e.g. high-impression, low-CTR categories) point to editorial over-representation and motivate category-affinity personalisation (S2).



```python
all_interactions.head()
```





  <div id="df-b4fb0f15-e069-4153-877b-04f4059c8de5" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>newsId</th>
      <th>clicked</th>
      <th>timestamp</th>
      <th>split</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>U13740</td>
      <td>N55689</td>
      <td>1</td>
      <td>1573463158</td>
      <td>train</td>
    </tr>
    <tr>
      <th>1</th>
      <td>U13740</td>
      <td>N35729</td>
      <td>0</td>
      <td>1573463158</td>
      <td>train</td>
    </tr>
    <tr>
      <th>2</th>
      <td>U91836</td>
      <td>N20678</td>
      <td>0</td>
      <td>1573582290</td>
      <td>train</td>
    </tr>
    <tr>
      <th>3</th>
      <td>U91836</td>
      <td>N39317</td>
      <td>0</td>
      <td>1573582290</td>
      <td>train</td>
    </tr>
    <tr>
      <th>4</th>
      <td>U91836</td>
      <td>N58114</td>
      <td>0</td>
      <td>1573582290</td>
      <td>train</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-b4fb0f15-e069-4153-877b-04f4059c8de5')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-b4fb0f15-e069-4153-877b-04f4059c8de5 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-b4fb0f15-e069-4153-877b-04f4059c8de5');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>





```python
all_interactions['split'].value_counts()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>split</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>train</th>
      <td>5843444</td>
    </tr>
    <tr>
      <th>dev</th>
      <td>2740998</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>




```python
# Analysis of cold start data
hist_train = parse_history_length(raw_train)
hist_dev   = parse_history_length(raw_dev)
```


```python
# Visualize the cold start ratios
fig, axes = plt.subplots(1, 2, figsize=(20, 5))

ax = axes[0]
ax.hist(hist_train.clip(upper=100), bins=50, color='teal', edgecolor='white', lw=0.4)
ax.set_xlabel('History length (clicks, capped at 100)')
ax.set_ylabel('Users')
ax.set_title('Train: history length distribution')
cold_frac = (hist_train == 0).mean()
ax.axvline(0, color='red', lw=1.5, label=f'Cold ({cold_frac:.1%})')
ax.legend()

ax = axes[1]
thresholds = [0, 1, 3, 5, 10, 20]
fracs = [(hist_train <= t).mean() for t in thresholds]
ax.plot(thresholds, [f*100 for f in fracs], 'o-', color='darkorange', lw=2)
ax.set_xlabel('History length threshold')
ax.set_ylabel('% users at or below threshold')
ax.set_title('Cumulative cold-start risk')
ax.axhline(50, color='grey', ls='--', lw=1, label='50%')
ax.legend()

plt.tight_layout()
plt.savefig('eda_coldstart.png', dpi=150, bbox_inches='tight')
plt.show()

print(f'Train users with zero history : {(hist_train==0).sum():,}  ({cold_frac:.2%})')
print(f'Train users with ≤5 history   : {(hist_train<=5).sum():,}  ({(hist_train<=5).mean():.2%})')
```


    
![png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAB8YAAAHqCAYAAAB2uSQnAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAA2uJJREFUeJzs3Xlc1NX+x/H3MLKIyOaCKCpoihvimonmgppKoWlqafW7XW9lenPJbqlXvWppZYbZZmq7ZjevZWnllrllLplpIpKWW4q7AqKgwDC/P0YGR8YFRGYGXs/Hw0fMOWe+8/nOOUx8v5855xjMZrNZAAAAAAAAAAAAAACUUG6ODgAAAAAAAAAAAAAAgNuJxDgAAAAAAAAAAAAAoEQjMQ4AAAAAAAAAAAAAKNFIjAMAAAAAAAAAAAAASjQS4wAAAAAAAAAAAACAEo3EOAAAAAAAAAAAAACgRCMxDgAAAAAAAAAAAAAo0UiMAwAAAAAAAAAAAABKNBLjAAAAAAAAAAAAAIASjcQ4AKDYjR49WuHh4df99+ijj97SayxatEjh4eHat29fEUVtsWXLFoWHh2vLli3XbRceHq7XXnutSF/7drpd71dhHTlyROHh4Vq0aFGRHO+ZZ55RdHS09XF0dLSeeeaZIjn2lUaPHq02bdrc9teRpEcffVT9+vW7LccGAAAAgFwbNmzQU089pdatW6tRo0bq0KGDhg8frl9++cXRodl46623FB4erkuXLt3ScVztWutm7j9cfU3s6q6+9i7q9gCAkquMowMAAJQ+Y8eO1bPPPmt9PGHCBCUkJOiLL76wlrm7u9/Sa8TExOjuu+9WYGDgLR2nsDZs2CBvb++bbv/mm2/q6NGjeuWVV25jVM5r8+bN+ve//63Vq1cXy+t98cUXBRpjX375pb7++mvNmzfvuu3Gjh2rrKysWw3Pro4dO+qVV15Rq1atJFlu+gAAAADA7TRjxgzNmTNHDz/8sIYMGaLAwEAdOnRIH3/8sR599FFNnDhRDz74oKPDvCVca928UaNGKSQkREOHDr3lY2VmZqpZs2Zavny5QkJCCvTc23ntDQAo2UiMAwCKXfny5VW+fHnrY09PTxmNRlWqVKnIXsPLy0teXl5FdryCKui5bN++XUFBQbcpGue3ffv2Yn29gn5h4mbju3JcF6UTJ07o6NGjNmX+/v635bUAAAAAQJLWrVund999V//5z3/08MMPW8tDQkIUFRWl4cOH67XXXlO3bt3k5+fnwEgLj2utgtm+fXuBk9jXEh8fX+jk9u269gYAlHwspQ4AcFq5y3uvW7dOnTp10gMPPCBJys7O1htvvKFOnTqpYcOGatOmjYYNG6YjR47ke27u0uCjR49Wz549tWXLFvXu3VuRkZHq0qWLvvrqK5vXvNkl0C9evKixY8eqRYsWatKkiUaOHKnz58/bPY7ZbNasWbPUtWtXNW7cWHfddZeefvppHT58WJJlue2NGzfqq6++slmmfd++fXrqqafUokULNWrUSDExMflmLIeHh2vOnDkaNGiQIiIiNG/ePLtLvWdmZqpFixaaNm3aTb33krR+/Xo98sgjuvPOO9WsWTM98cQTNkut577He/fu1RNPPKGmTZuqbdu2eumll5STk2Nt9+eff+qRRx5R48aNdffdd+u9997T7NmzFR4eLsnSNzNmzFBSUpLCw8Ntvp1vMpn02muv6a677lJERIT+8Y9/6MSJE9eNe9u2berVq5ciIiIUHR2tTz/9NF+bq5c4X7BggWJjY9WkSRO1bNlSAwcOVEJCgiTLMnoLFy7Uzz//bF3ePXdJ/WXLlik2NlatW7e2nou95dnmz5+vjh07qlGjRurdu7d27NhhrbP3nNyl5P/73/9qy5YtateunSTp//7v/6zL3129vF9mZqbi4uIUHR2tRo0aKSoqSqNHj9aZM2dsXutmfg8AAAAA4MMPP1RoaKgGDBiQr85gMOiFF17QDz/8YE2K29tKyt61+X333ad169YpJiZGERERuv/++5WYmKhNmzapZ8+eioyM1AMPPKDff//dehx7y5vnXpetX7/ebvw3undwo2sts9msjh072qx4l+s///mPWrVqZU3s3uj6+Vq++uorxcbGqnHjxurcubPeeOMNZWdnW+tv5r7A1W7mmtiezMxMvfLKK4qOjlZERITatGmjUaNGKTk5WZLl/sOhQ4f09ttvKzw83Po+LlmyxPp6zZs3V//+/fXzzz9bj5t7fbtw4UI99NBDatSokT755BPruOrUqdM1t9O72Wvvn3/+WY888ohatmypJk2aqFevXvruu++uea45OTkaNmyY7r77bpt7SQCAko/EOADA6c2ePVsvvfSSZs2aJUmaNWuW3nvvPT333HNatWqV3n33XSUlJWnYsGHXPc7Zs2f19ttva9y4cfr6669Vu3ZtjR8/XseOHbO22bBhgwYPHnzDmN544w01adJEX375pcaOHavvvvtO77//vt22X3zxhWbPnq3nnntOy5cv15w5c3Tu3DkNGjTIWh8YGKju3btrw4YNatq0qc6cOaOHH35YKSkpmjNnjr799lv17NlTU6ZM0dy5c22Ov3DhQjVv3lzLli3Tgw8+qODg4HyJzvXr1ystLU19+/a94blJlovKQYMGqXLlyvrss8/0ySefKDMzU4888ojOnj1r03bixInq27evlixZogcffFCffPKJli1bJslyYf3kk0/qxIkTev/99/Xee+9p27Zt+vLLL63PHzt2rDp16qQqVapow4YNGjhwoLVu7ty58vPz0+eff664uDht3br1ul9cSElJ0VNPPSVPT099/vnneuedd/Tzzz9r69at13zOpk2bNHHiRP3973/Xd999p3nz5snPz08DBw5URkaG3nrrLTVs2FBNmzbVhg0bFBMTY33urFmzNHz48Osmln/99Vdt2bJF7777rv773//KbDZr8ODBSk9Pv3YHXKFp06aKi4uTZFnS78otB640btw4ffbZZxo2bJiWLl2ql19+WVu2bNETTzwhs9lsbXczvwcAAAAASrfs7Gz9+uuvat++vQwGg902/v7+8vX1LfCxk5OTNW/ePMXFxenTTz/V2bNn9fzzz2vmzJmaPHmy5s2bp1OnTmnKlCm3dA43undwo2stg8Gge++9V2vWrLHZtzw7O1srV67UvffeK3d39wJdP1/pm2++0dixY/XAAw/om2++0ejRo/Xxxx9r+vTpklSg+wK5CnNNnGvmzJn67rvvNGXKFK1cuVJvvPGGdu/ereeee06SrFufDRw4UBs2bFBwcLC2bt2q5557Tu3bt9fSpUu1cOFChYaGatCgQfm+1P7BBx+oT58+Wrlypfr27at//etfkiz3NG60fP31rr3T0tI0aNAg1atXT//73/+0ZMkSde3aVc8++6zNl9Kv9PLLL2vz5s364IMPimwGPADANZAYBwA4vZiYGLVq1cq6PPmAAQO0ZMkSdevWTcHBwWrcuLH69OmjhISE6150njx5UuPHj1ezZs0UFhamf/zjH8rKytLu3butbSpVqqRy5crdMKZWrVqpb9++qlmzpvr27avatWtr586ddtsmJCQoODhYnTt3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qfkhQ4Zo5cqVSkpK0v79+zVr1ix5eXlZE8q+vr46ePCg4uPj7X5j/3q8vLw0duxY7d27Vzt37tSUKVMUFBSkqKgoSZabHVlZWZo1a5YOHz6sVatWWZfBz5W7595PP/2k3bt357txUKlSJfXq1Utz5szRt99+q8OHD+uHH37Qyy+/rFatWqlx48YFihkAAAAAWrduraFDh+qdd97RqFGj9OuvvyopKUlbtmzRk08+qe+//17Tpk1TcHCwJMu1za+//qpVq1bpr7/+0scff2z9YvCtaty4sY4cOaL//e9/Onz4sBYtWnTda9ObvXdwo2utXLGxsdqwYYPWrVunnj172tQV9vr5ySef1KZNmzRr1iwlJSVp9erVmjFjhnUWc2HuCxTmmliS3Nzc9P7772vEiBH65ZdfdOzYMSUkJOjzzz9X3bp15e/vLw8PD3l5eWnHjh36/fffde7cOTVp0kRbtmzRxo0bdejQIU2bNk05OTkyGo3auXPndWeP577369at0549e64b3/X89ddfevrpp/XRRx/p4MGDSkpK0qJFi3TgwAG1bNkyX3tPT09Nnz5dBw4c0Kuvvlro1wUAuCYS4wAAlzNt2jS5u7urb9++GjZsmLp06aJx48apWbNmmjx5spYvX17oY7dt21bvvvtuEUYrvfjii2rdurXGjh2re+65R4899pjOnz+vDz/80Lpk2aBBg3T8+HH1799fW7duVWBgoObNm6fy5cvr73//u2JjY7Vq1SpNnTpV999//029bkxMjLKzs/Xggw8WOOYWLVro/fff1549e/Tggw8qNjZWK1eu1Ouvv65OnTrd9HHKlSund999V0ajUY888oj++c9/ql27drr77rttlpTr37+/goKC9Nhjj2nu3LkFjjdXxYoV9c477yglJUV9+/bVkCFDdNddd6l9+/bXfM7w4cPVp08fTZ06Vd26dVP//v2VmJio9957z3qD5+9//7vMZrMGDBhQ4PHVtm1bRURE6IknntCAAQPk6empWbNmydPTU5Klnx599FF99tlnio2N1fz58/Xiiy/aHCMiIkKdOnXSRx99pMcff9zu8n8TJ07UQw89pNdee03dunXThAkT1LlzZ7uzOwAAAADgZjz99NP6+OOPlZqaqiFDhqhbt27697//rYoVK2rRokXq3Lmzte2wYcPUpk0bjRo1Sn369NG+ffv0zDPPFEkcjz76qGJjY/Xaa6+pZ8+eWrt2rcaNG3fd59zMvYObudaSpG7duuns2bM6f/68unXrZlNX2Ovn+++/X1OmTNGSJUvUrVs3TZkyRY888oieffZZSSrUfYHCXBPneuedd1S9enUNHz5cXbp00VNPPSV/f3/rPRKDwaAhQ4Zo+/btevjhh7Vv3z6NGDFCrVq10tNPP62HHnpI2dnZmjBhgh555BF99913mjZt2jVfr3379mrWrJleeeUVjRkz5obxXUu7du300ksvafHixbr//vt13333ae7cuRo3bpy6du1q9zl16tTR6NGjNW/evGsuEw8AKJkM5qJYtxQAADidjz76SHPmzNGaNWsKtL94UcudwX/l/mdDhgzRoUOH9N133zkqLAAAAAAAAABAKVLG0QEAAICidfz4cf388896/fXXNW7cOIcmxbOzs9WjRw8FBgZq/PjxCgwM1I8//qg1a9Zo1KhRDosLAAAAAAAAAFC6MGMcAIASpmHDhgoMDNQjjzyiQYMGOTocHTx4UNOmTdO2bduUkZGhkJAQ9e3bV48++qiMRqOjwwMAAAAAAAAAlAIkxgEAAAAAAAAAAAAAJZqbowMAAAAAAAAAAAAAAOB2IjEOAAAAAAAAAAAAACjRSIwDAAAAAAAAAAAAAEq0Mo4OwBlkZ2crNTVVnp6ecnPjuwIAAAAAgNsnJydHly5dkp+fn8qUKVmX5VxfAwAAAACKU0GusUvWFXghpaam6uDBg44OAwAAAABQioSGhqpChQqODqNIcX0NAAAAAHCEm7nGJjEuydPTU5LlDStbtqyDo7mGjAypTRtJkmn9ehl9fBwcEJCfyWTS3r17VbduXRmNRkeHA+TDGIWzY4zC2TFG4excZYxmZGTo4MGD1mvRksQlrq8vc5XxUprRR66BfnJ+9JFroJ9cA/3k/Ogj10A/uQZX6aeCXGOTGJesy7uVLVtW3t7eDo7mGsxmac8eSZLJy0tGZ40TpZrJZJIkeXt7O/WHJEovxiicHWMUzo4xCmfnamO0JC417hLX15e52ngpjegj10A/OT/6yDXQT66BfnJ+9JFroJ9cg6v1081cY5e8q3AAAAAAAAAAAAAAAK5AYhwAAAAAAAAAAAAAUKKRGAcAAAAAAAAAAAAAlGgOS4xv3bpVERERNv8aNWqk8PBwSdKmTZvUp08fNWvWTPfee6+WLFli8/y5c+eqa9euatasmfr3769du3ZZ6y5duqT//Oc/ateunVq1aqVhw4YpOTm5WM8PAAAAAAAAAAAAAOAcHJYYb9mypeLj423+Pf300+revbtOnjypIUOG6KGHHtKmTZs0duxYjR8/XvHx8ZKk1atX66233tKrr76qjRs3qmPHjnrqqaeUnp4uSXr99deVkJCgBQsWaMWKFTKbzRozZoyjThUAAAAAAAAAAAAA4EBOs5T60aNH9dFHH+n555/XN998o9DQUPXp00eenp6KiopSdHS0Fi5cKElasGCBevfurcjISHl5eenxxx+XJK1Zs0bZ2dn64osvNGTIEAUHB8vf318jRozQ2rVrdeLECUeeIgAAAAAAAAAAAADAAco4OoBcb7zxhh544AFVrVpVCQkJatCggU19gwYNtGzZMklSQkKCYmJirHVubm6qX7++4uPjVb9+faWlpalhw4bW+tq1a8vLy0sJCQkKCgq6Zgwmk0kmk6mIz6yImEwyWn80Sc4aJ0q13N8fp/09QqnHGIWzY4zC2TFG4excZYw6e3wAAAAAAJRETpEYP3LkiFauXKmVK1dKklJSUvIlsP39/a37hKekpMjPz8+m3s/PT8nJyUpJSZEk+fr62tT7+vrecJ/xvXv33spp3FZuGRlqevnnhIQE5ZQt69B4gOvJ3fYAcFaMUTg7xiicHWMUzo4xCgAAAAAAruYUifH58+frnnvuUaVKlW76OWaz+Zbq7albt668vb0L/LxiceGC9ceGDRvKeFXiH3AGJpNJ8fHxioiIkNFovPETgGLGGIWzY4zC2TFG4excZYymp6c79RezAQAAAAAoiZwiMb5ixQqNGjXK+jggIMA68ztXcnKyAgMDr1mfkpKiOnXqWNukpKSoXLly1vrU1FRVqFDhunEYjUbnvXlyRVxOHScgxiicH2MUzo4xCmfHGIWzc/Yx6syxAQAAAABQUrk5OoDExEQlJSWpTZs21rKIiAjt2rXLpt2uXbsUGRkpSWrUqJESEhKsdSaTSbt371ZkZKSqV68uPz8/m/q9e/cqMzNTjRo1us1nAwAAAACwZ9GiRTbXfVd77bXX9Oijj173GJ988okeffRR5eTkFHV4Ny0zM1P33Xefvv32W4fFAAAAAAAACs7hifHdu3fL399fPj4+1rLY2FglJSVp4cKFunTpktatW6d169apX79+kqT+/fvr66+/1o4dO5SRkaF3331XHh4e6tChg4xGo/r166dZs2bp2LFjSk5O1vTp09WlSxdVrFjRUacJAAAAAC5t//79evbZZxUVFaXIyEhFR0dr8uTJ+Vbzul327Nmjt956S6+88orc3CyXsocOHVLv3r2vm3DPNX/+fHXt2lVNmzZV165dNW/ePGvd8uXL1bZtW7Vt21bff/+9zfN27typbt266dKlS5IkDw8PvfLKK5o4caKOHTtWhGcIZxYeHq5GjRopIiLC+u/FF1+UJG3atEl9+vRRs2bNdO+992rJkiXW523dulWdOnVSq1atNH/+fJtjJiUlqUOHDjp79myxngsAAAAA3FCOSTq8VgEnlkuH11oelwAOX0r99OnT+fYWr1ChgmbPnq3Jkydr0qRJqlatmqZNm6Z69epJktq1a6eRI0dqxIgROnPmjCIiIjRnzhx5eXlJkoYNG6YLFy6oZ8+eys7OVseOHTVx4sTiPjUAAAAAKBESExP18MMPq3///lqyZIkCAgK0d+9evfTSS+rfv7+++uor6/XY7fL222/r3nvvVbVq1SRZkpHPP/+8mjZtqhMnTlz3uevWrdO0adP0ySefKCIiQvHx8frb3/6m6tWrq127dnrxxRf1/vvvy2Aw6IknnlDnzp1lMBiUnZ2t//znP5owYYI8PT2tx2vUqJFatGihDz74QOPGjbut5w3nsXz5coWEhNiUnTx5UkOGDNHYsWMVGxurbdu2afDgwQoLC1NERIReeeUVjRkzRo0bN1ZsbKxiY2Pl6+srSXrxxRc1dOhQ65ZwAAAAAOAU/lgkrR4u4/kjqiVJiZJ8QqToN6Q6vR0c3K1xeGJ80KBBGjRoUL7yli1bavHixdd83oABAzRgwAC7dR4eHpowYYImTJhQZHECAAAAQGn1wgsvqG3btnruueesZfXr19e7776rKVOm6OTJk6pRo4aOHz+uSZMm6ddff1V2drbatWunCRMmyN/fP98xV69eralTp+rkyZNq3779dVf4OnXqlFatWqUvvvjCWpaSkqKPP/5Yv/32m7Zt23bd+Hft2qU6depYt+eKjIxU3bp1tXv3bjVo0MB6PpKUnZ1t/QL33LlzVa9ePbVu3TrfMR966CE988wzev755+Xh4XHd10fJ9c033yg0NFR9+vSRJEVFRSk6OloLFy5URESE9uzZo7vvvluenp6qXr269u/fryZNmmjFihW6cOGCHnjgAQefAQAAAABc4Y9F0pI+ksy25eeTLOU9vnDp5LjDl1IHAAAAgFLNbJYuXCjef2bzjeO67MyZM/r111/1yCOP5Kvz8fHRyy+/rBo1akiShgwZovLly+uHH37QihUrdPLkSbtfWD537pyeeeYZPfLII9qyZYt69eqlr7/++poxbNmyReXLl7cmryWpe/fuql279k2dw913360///xTW7ZsUWZmprZv3659+/apbdu2MhgMNnuWm81mGQwGHT16VJ9++qm6du2qAQMG6MEHH9TatWut7Vq0aKFLly5p586dNxUDXF9cXJw6dOigFi1aaPz48bpw4YISEhKsX67I1aBBA+3atUuSbMZX7tg6f/68pk2bpoEDB+qxxx5T3759bb70AQAAAAAOkWOSVg9XvqS4lFe2ZoRLL6vu8BnjAAAAAFBqmc1S27bSxo3F+7pt2kg//igZDDdsevjwYUlSWFjYddslJiYqISFBs2fPlo+Pj3x8fPTkk0/qn//8pzIzM23abtiwQd7e3nr44Yfl5uam9u3bq0WLFrpw4YLdY//xxx+64447rHuLF1Tjxo01ZswYDRw4UNnZ2SpTpoxGjx6txo0by2w2y93dXb/99ptMJpO8vb1VsWJFPfXUUxo2bJji4uI0ceJEVatWTX379tWaNWvk7u4uHx8fValSRX/88YdatGhRqLiQJy0tTefPn7cp8/LyUkBAgLKzs3Xq1Kl8zwkODpZk2aItKyvLps7f319ly5bVhQsXdO7cOZs6Dw8PVahQoUDxNWnSRFFRUZo6daoOHz6sESNGaNKkSUpJSVFQUFC+105OTpYkNWzYUGvWrFFERISSkpJUu3ZtzZgxQ/fff78+//xz9erVS506dVJMTIw6duxY4LgAAAAAoEiYc6T496XzR67XSEo7LCX9KFXvUFyRFSkS4wAAAADgSDeRnHYkw+X4rpxVbc+RI0fk5+enSpUqWctq1KihrKysfHuAHz9+XMHBwTaJ7tDQUCUkJNg9dkpKivz8/Ap7Ctq8ebPi4uL0/vvvq1mzZoqPj9fw4cMVHByszp07a8KECRo6dKgMBoNeeOEFrVy5UhcvXlSnTp300ksvWRPflSpV0v79+xUeHi5JCggI0NmzZwsdF/Js27ZN69atsymLiIhQ7969de7cOc2ZMyffc3JXI1i8eLGOHLG9edOrVy81btxYCQkJWrZsmU1d7dq17a6AcD0LFiywef6//vUvDR48WM2bN7/u80aPHq3nn39eaWlpeu6553TgwAFt2bJFX375pdq2bau4uDj5+PiocePG+u233xQdHV2guAAAAACg0DLOSodWSgeWSQeXS+knb+5554/d3rhuIxLjKBRTTo6MhZytUZTHAAAAAFyawWCZuZ2eXryv6+190wn53GXS//jjj3wzY6909azwKxmueq3MzEyZTLZLr90o8X71MQriv//9r+655x7rXuEtWrTQvffeqy+++EKdO3dWp06d1KlTJ0nS+fPn1bt3b7333ns6f/68vL29rccpW7as0tLSbGIyF2BZelxb8+bNrV84yOXl5SVJ8vX11ZNPPnnN5/bs2dPujHHJMmO7evXqNnVFsSd8SEiITCaT3NzclJKSYlOXnJyswMBASZaZ5itXrpQkmUwm9e3bVxMmTJCHh4fS0tKs4+vqsQUAAAAARc6cI53cYUmEH1gqHdtsKSson+AiD624kBhHoRjd3PTwokVKtLOc3c2oX6mS5vfuXcRRAQAAAC7IYJDKlXN0FNcUEBCgO++8Ux999JHatm1rU5eRkaGHH35YY8eOVfXq1ZWamqrTp0+rYsWKkqT9+/fL09MzX0K9cuXKOnHihHXPZUnat2/fNWPw9/fXH3/8UehzyMnJyZeIv1Yif8aMGerdu7dq1qyptLQ0m2W4U1JS5OPjY3189uxZawIUt6Z8+fIqX7683boyZcpYl023J3e82VOuXDmVu8Xfr927d2vJkiUaPXq0tWzfvn3y8PBQ+/bt9dVXX9m037VrlyIjI/MdZ968eWrYsKF1BQIfHx+lpqYqICBAKSkptxwnAAAAAORzMUU69P3lWeHLpAvH7bdzLydVj7Ysk34pVfb3GTdI5UOkanffxoBvL6brotAST53S9uPHC/WvsAl1AAAAAMVv7Nix2rFjh0aOHKnjx48rJydHiYmJevzxx+Xl5aXGjRsrIiJCtWvXVlxcnNLT03XixAm9++67uvfee+Xu7m5zvKioKJ0/f16ff/65MjMztWrVKv3222/XfP06depo3759BZqd/be//U1Lly6VJEVHR2vFihX65ZdflJ2drZ07d2rZsmXq0qWLzXN27dqln3/+Wf/4xz8kWZK1QUFB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DIVxAAAAAABcyKuvvqrff/9dU6ZMUe3atfXSSy/pyJEjunTpkqpUqaJ///vfqlevntExARSEs/ul1V2kS8czHnv4SF1XSPc8ZGwuAAAAwAm5G/nkMTExev7559WsWTO1aNFCL730ki5evChJioqK0uOPP65GjRqpY8eOWrp0qcO269evV7du3dSwYUP17t1b27dvty+zWq2aM2eO2rdvryZNmujpp5/WiRMnCvTYAAAAAADIL/Xq1dOqVatUo0YNTZ48Wb6+vho+fLgGDRp020Xx3JyjAzDAXxulz1tkFsVLVpYG7KAoDgAAAGTD0ML4s88+q1KlSmnr1q1avXq1Dh06pBkzZujy5csaNmyYmjdvrh9//FFz5szRokWLtGnTJkkZJ+Tjx4/XuHHjtHv3bg0ePFgjRozQmTNnJEnLly/XunXrtHjxYn3//feqVq2ann/+edlsNiMPFwAAAACAPOPu7q6nnnpKn376qTZs2KCnn35aJ0+evO393e45OgAD/LYg40rx1EsZj8s3kh79SSrHbBEAAABAdgwrjF+8eFHBwcEaO3asSpQooQoVKqhXr1763//+p/DwcKWlpem5556Tt7e36tatq759+2rFihWSpJUrVyo0NFShoaHy9PRU9+7dVatWLa1du1aStGLFCg0ePFjVq1eXj4+PwsLCFB0drX379hl1uAAAAAAA5Inz589rzpw5euaZZzR06FB9+eWXeuutt/TII4/o6aef1pIlS2SxWHK0z9ycowMoQFaL9H2YtGW4ZPvndV6jl/TINsmnorHZAAAAACdnWGG8VKlSmj59usqWLWtvO336tO68805FRkYqMDBQJpPJviwoKEj79++XJEVGRiooKMhhf0FBQYqIiNDly5d1+PBhh+U+Pj6qWrWqIiIi8vmoAAAAAADIXyNGjFBKSooef/xxPfbYYzp79qxGjRqljh076ssvv9Tff/+tPn365GifuTlHB1BAUhOkNb2kX97JbGvyotR9leRRwrBYAAAAgKsoZnSAKyIiIvTZZ59pwYIF+vbbb1WqVCmH5X5+fjKbzbJarTKbzfL19XVY7uvrq8OHD+vChQuy2WxZLo+Pj79hBovFkuNv1RcYi0Um+48WKRc5r/4wA4WL0eP3yvMbnQPIDmMUzo4xCmfHGIWzc5Uxmtt8f/zxhz766CMVL15cktS0aVO1bt1a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sWNH29KlS42OhSKua9eutpdeesmWkJBgO336tK137962V155xZacnGy7//77bfPmzbMlJiba9u/fb2vatKlt48aNRkdGEXXgwAFb06ZNbbVq1bLZbDZbbGysrUGDBraVK1faLl++bNuxY4etXr16tt9//93gpChqPvvsM9uDDz5oi46Otl26dMn2+uuv215//XXGKJxCWlqarWXLlrbZs2fbUlJSbOfPn7c9+eSTtpEjRzJGYYjFixfbOnbsaOvfv7/thRdecFh24MABW3BwsC08PNx2+fJl25o1a2z169e3nT592maz2WyffPKJrW3btrbDhw/bLl26ZHvttdds3bp1s1mtViMOBQXoVs+lt2zZYmvcuLHtt99+syUnJ9sWLVpka9mypS0xMdGA1EVPdueW1/ryyy9tjz/+uAEJYbPZbLVq1bKdOHHipuvxenIuCxYssI0ePfq69nfffdc2fvz4gg9UBOXmPcy1RowYYRs6dKjt3LlztjNnztgeeeQR2+uvv14Qh1Go3aiP9u3bZwsJCbF99913trS0NFt4eLitbt26tj179mS5r8cff9z25ZdfFkTsIudG/ZTT9wi8lvLPjfrpWmlpabYuXbrYwsPDs1zetm1b2+7du/MjZpGXV7UdVz3XLrRXjBcWEREROnjwoMaNG6eSJUuqWrVqGjx4sFasWGF0NBRhFy9eVHBwsMaOHasSJUqoQoUK6tWrl/73v/8pPDxcaWlpeu655+Tt7a26deuqb9++jFkYwmq1avLkyRo8eLC9bd26dapWrZr69OkjT09PtWjRQu3atdPKlSuNC4oiaenSpQoLC9M999wjHx8fTZw4URMnTmSMwinExcUpLi5OPXr0UPHixVW6dGk98MADioqKYozCEJ6enlq1apWqVq163bKVK1cqNDRUoaGh8vT0VPfu3VWrVi37TAYrVqzQ4MGDVb16dfn4+CgsLEzR0dHat29fQR8GClBOzqVXrFih3r17q379+vLy8tKQIUMkSd9//31Bxy5ybnRuCdfE68l5nDp1SsuWLdOLL75odJQiLTfvYa529uxZbd68WWFhYfL391f58uU1fPhwffnll0pLSyuIQym0btRHZrNZw4YNU4cOHVSsWDGFhoaqVq1a/D9lgBv1U07wWspfOemnjz/+WHfddZdCQ0MLIBmuyMvajquea1MYd3KRkZEKCAiQr6+vva1u3bo6evSoEhISDEyGoqxUqVKaPn26ypYta287ffq07rzzTkVGRiowMFAmk8m+LCgoiKnTYIj//Oc/8vT0VLdu3extkZGRCgoKcliPMYqCFhsbq5MnT+rChQt66KGH1KxZM40aNUrnz59njMIplC9fXnXq1NGKFSuUmJioc+fOadOmTWrTpg1jFIYYNGiQSpYsmeWy7MZkRESELl++rMOHDzss9/HxUdWqVRUREZGvmWGsnJxLXzuG3N3dVadOHcZIAbjRuWVWTp8+rSeffFJNmjRR+/bttWbNmoKKCkmzZ89WmzZt1LhxY02aNEmJiYnXrcPryXnMnTtXDz/8sO66664sl//xxx/q37+//dY4N5rCG7fvdt/DXCsqKkomk8lhavy6desqKSlJR44cydvQRcyN+qh169Z6/vnn7Y/T09MVFxen8uXLZ7u/9evX66GHHlLDhg01ePBgHT9+PM8zF0U36ifp1t8j8FrKXzfrpysuXryohQsX6l//+tcN1/vkk0/UoUMHNWzYUKNGjdK5c+fyKmqRlVe1HVc+16Yw7uTMZrNKlSrl0HblxD4+Pt6ISMB1IiIi9Nlnn+m5557Lcsz6+fnJbDZnez8/ID+cPXtW8+bN0+TJkx3asxuj/JuKgnTmzBlJ0oYNG7Rs2TKtWbNGZ86c0cSJExmjcAru7u6aN2+etmzZonvvvVctWrRQenq6xo4dyxiF0zGbzQ7FTynjnCk+Pl4XLlyQzWbLdjkKr5ycS99oDKFgXX1ueS1/f39Vq1ZN//rXv7Rjxw6NGTNGr7zyinbt2mVA0qKnQYMGatGihTZt2qQVK1bot99+09SpU69bj9eTczh58qQ2bdqkJ598MsvlFSpUUOXKlTVjxgzt2LFDffv21bPPPktRqIDl5PViNpvl4+MjNzc3h3UlPiMuSLNmzZK3t7ceeuihLJdXr15dNWvW1P/93/9py5Yt8vf315AhQ5SamlrASYuWnLxH4LXkHD777DM1adJENWvWzHadOnXqqF69elqzZo3Wr18vs9ms0aNHF2DKouF2azuufK5NYdwF2Gw2oyMA2dq7d6+efvppjR07Vi1atMh2vavfbAAFYfr06erdu7dq1KhhdBTgOlf+bx8yZIjKly+vChUqaOTIkdq6davByYAMqampevbZZ/Xggw/qf//7n3744QeVLFlS48aNMzoakKWbnTNxTlU05aTfGSPGu9m5ZZs2bfTBBx8oKChIxYsXV5cuXfTAAw9o9erVBqQtelasWKG+ffuqePHiql69usaNG6dvvvkmy2IPryfjLV++XB07dlS5cuWyXN63b1+9++67qlq1qu644w4NHjxYderUyXIKb+Qv/q9yDTabTTNnztQ333yjBQsWyNPTM8v1pkyZovHjx8vPz0/+/v567bXXFBMTo7179xZw4qIlp+8ReC0Zy2KxaPny5Ro0aNAN15s/f76GDRumEiVKqGLFipo8ebL27NnDLAx5KC9qO674eqIw7uT8/f1lNpsd2sxms9zc3OTv729MKOAfW7du1dChQ/XKK6/Y/yPz9/fP8goMPz8/ubvzTw4Kxq5du/Trr786THd1RenSpa/7dzU+Pp5/U1GgrkxXdPW3MAMCAmSz2ZSWlsYYheF27dqlkydPasyYMSpZsqTKly+vUaNG6bvvvpO7uztjFE4lq//bzWaz/P397e9Bs1pepkyZgguJApeTc+kbjSEUjKzOLW9FQECA/v7773xMhuxUqlRJFovluilNeT05h40bN6pdu3Y52obXU8HLyevF399fCQkJslgsDutK4j1NPrNarXrppZe0detWff7557rnnntueVsfHx/5+voqNjY2HxMiK9n9m8ZryXh79uxRamqqGjdunKPtAgICJIn/q/JIbms7rnyuTZXKyQUHB+v06dM6f/68vS0iIkI1atRQiRIlDEyGou6XX37R+PHjNXfuXPXs2dPeHhwcrD/++EPp6en2toiICNWvX9+AlCiq1q5dq3Pnzqlt27Zq1qyZevfuLUlq1qyZatWqdd19Ufbv388YRYGqUKGCfHx8FBUVZW+LiYmRh4eHQkNDGaMwnMVikdVqdfjm75Urwlq0aMEYhVMJDg6+bkxeef/p6empmjVrKjIy0r7s4sWLOn78uOrVq1fQUVGAcnIuHRwc7DBGLBaLDhw4wL9rBSS7c8trff7551q/fr1DW3R0tCpXrpzPCXHgwAG99dZbDm3R0dEqXrz4dfeD5/VkvKioKMXExKhly5bZrvP+++9fN8Uwr6eCd6P3MNeqU6eObDabDh486LBuqVKldPfdd+d71qLszTff1KFDh/T555/f8DWSkJCgKVOmOBTBz58/r/Pnz/Paymc5eY/Aa8l4W7ZsUfPmzVWsWLFs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+PDhevnllyVJhw4d0qBBg9S4cWM1a9ZMkydPVkpKin0/Xbt21VtvvaUGDRroq6++UosWLWSxWOzPferUKdWuXfum07dZrVa9++676tChg+rXr6+HH35Ye/futS9v166dVq5cqaFDh6phw4bq0KGDtm/fbl8eHh6uNm3aqGHDhnr55Zc1d+5cDRw4UL///rv69++vs2fPKiQkRLt377Zv8/nnn6tVq1Zq0KCBZsyYkW221atXq0qVKqpfv769LTIyUo888ogaNGigTp06af369Tf9vZ8/f16jRo1So0aN1KpVK7399tv2AuzVLl68qE6dOtmnFs+uv27FmjVr1KlTJzVs2FD9+/dXVFSUfdlHH32kDh06qGHDhurcubM2bdpkXzZw4EC9/fbbeuGFF9SgQQOFhobqu+++sy8PDAzU6tWr1adPH9WrV089e/bUkSNH7MsPHjyoJ554Qo0bN1bz5s01bdo0paWl2ZfPnz9frVq1UrNmzTR//vwbHsOV39t9992nxo0b65lnntHp06clZT1mr9WgQQMtXrxYXl5e1y1bsWKFBg8erOrVq8vHx0dhYWGKjo7Wvn37dPbsWW3evFlhYWHy9/dX+fLlNXz4cH355Zf2Y3nkkUcUHh6u2NjYGx4DAAAAABQkPge4/c8Bnn32WYWHh2vatGl64okn7O1xcXF64oknVK9ePT300EP6888/JWXcUq1hw4b66KOPdO+99+rXX3+VJH322Wfq3Lmz6tevry5dumjz5s0O2bt166aGDRuqVatWmjlzpqxWq315YmKiRo0aZe+vn376yb7sRv1zrS+++ELt2rVTo0aNNHXqVIfnAADgdlEYBwC4jDNnzui1117Tu+++q99++03z5s3TokWLdODAAa1du1aS9P7772v69OlKTU3VU089pfr162v79u1auXKl9uzZo7lz59r39/fff8vT01N79uxRp06dlJycrB07dtiXb9q0ScHBwTedvu3jjz/Wf//7X33wwQfas2ePevbsqeeee05JSUn2dT788EONGDFCP/30k5o2bao333zTnmHkyJEaPHiwfvrpJzVq1EjLly+XJNWrV0+vv/66ypYtq4iICDVv3lySdOzYMV24cEFbt27V3LlztXTpUkVGRmaZbffu3fbtJCk5OVnDhg1Tx44d9fPPP+vVV1/V+PHj7R84ZGfixImSpG3b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Hi8vL40dO1ZPPfVUluuOGDFCffv21ebNm7V582Z98MEH+vjjj1WvXr0bPtet9M/V697quTcAADnBVOoAAJdhtVp18eJFVa1aVU8//bS++OIL1ahRQ999991161auXFknTpxwOPE6cuSIKlWqlO0Jp7u7u7p166YNGzbo22+/vaXp064817Xf7M7u6utrlSlTRmfOnHEoTkZERNzStrfCz8/PYersypUrKyYmxuH38vXXXysqKirbfQQEBMjd3d1hqrXdu3c73ENswoQJmj17tnbv3q2vv/5aUs7661qVK1d2eL7U1FR9+OGHio+PV0REhNq3b6+goCC5u7tneV+1EydO2H+2Wq06c+aMKlSoYG87fvy4/edTp05JkipUqKAqVaroxIkTSkxMtC+Pj49XQkKCJOnOO+/UmTNnHHJd/VxXO3v2rGJiYjRw4ED5+/tL0i3dX/1WXCngX33sFy9e1PHjx1WvXj3VqVNHNptNBw8etC+PiIhQqVKlHK54P3/+vD0bAAAAADgbPgcoeFWqVLnu2E6dOmXPazabVb58eT322GNatmyZHnzwQfvtx24kJ/1z7bm3JEVHR9/uIQEAYEdhHADgMtavX6++ffvqyJEjkqSYmBjFxsbap9z29PTUsWPHlJCQoNatW6tYsWKaP3++UlNTdeTIEX3yySfq2bPnDZ+jZ8+e+vHHH/X777/rgQceuKVc/fv31/Lly/Xbb7/JYrFo/fr16tq1q73geiNNmjTR+fPn9Z///Eepqan68ssvdezYMftyLy8vXbp0SbGxsbp8+fIt5blazZo1He4h3bp1a3l7e2vhwoVKSUnRzz//rMmTJztM4XYtPz8/tW/fXvPnz5fZbNapU6c0adIkxcbG2tdxd3dX+fLlNWHCBL3xxhs6c+bMTfvrxRdf1LJly7J8zt69e+unn37S999/r7S0NH300Uf65JNP5OPjo4CAAB08eFDJyck6fPiwPvjgA5UsWdIhz6+//qqdO3cqNTVVn332mRITE9WyZUv78jVr1ujYsWNKTEzUkiVLFBwcrHLlyqlVq1by9/fXjBkzlJCQoLi4OI0ePVqzZs2y//62b99unxr9vffey/Zb6/7+/vL29tZvv/2mlJQUrVu3TlFRUUpISLAX3q8eszk1YMAAffLJJ4qOjlZCQoJmzZqlOnXqKCQkRP7+/urUqZPeeecdnT9/XmfOnNH8+fPVp08fhysBoqOjHe5BDwAAAADOhM8Bcv45gKenp44fP65Lly7leFtJeuSRR7R+/XqFh4crPT1du3fvVteuXbVv3z79+uuv6ty5s37//XfZbDadO3dOR48edbgVWnZy0j+tW7fWgQMHFB4ertTUVC1fvtzhnB8AgNvFVOoAAJfRpUsXHTp0SE888YQuXryosmXLqm/fvurQoYOkjBPTf//739q5c6cWLFigxYsX66233tJ9990nPz8/9ezZU88+++wNn6N69eqqXr26qlatqhIlStxSrj59+uj06dMaMWKEEhISdM899+i9995zuEdXdipXrqw33nhDM2fO1Ntvv62ePXuqR48e9iuBmzdvrkqVKqlDhw6aMWPGLeW5WvPmzR2mBS9evLiWLVuml156SR988IEqVqyoN998U7Vq1dJPP/2U7X6mT5+uCRMmqG3btvLx8VHPnj31yCOPXLdez549tXnzZr3yyiv68MMPb9hfp0+fdriK+2p16tTRrFmz9Prrr+v8+fOqXbu2FixYIA8PDw0bNkxhYWFq3ry5atasqenTp6t8+fKaNm2a/ern7t27a8WKFRo+fLhKlSqluXPnOtzbu0+fPho7dqz+/PNP3X333fZ7yHl4eOj999/XtGnT1LJlS/n4+Kh9+/YaP368JKlz5876448/9Oyzz8pisWjgwIFq0KBBlsdQrFgxTZkyRTNnztTcuXPVpUsXzZs3T48//rg6duyoHTt2XDdmr7Znzx771HWpqamaNm2a3nzzTTVp0kRLly5V//79FRcXp4EDB9qn/Hvvvffs27/22muaPHmy2rdvLw8PD3Xt2lVhYWH25YcPH9b58+fVvHnzbPsdAAAAAIzE5wA5/xygX79+euedd7Rz585bupL7Wi1bttT48eP12muv6ezZs6pUqZKmTJliP/d97rnn9MILL+js2bPy8/NT586d9dhjj910vyVKlLjl/qlfv74mTpyoKVOm6OLFi+rWrZsefPBBbgUGAMg1Nxv/mwAAYGexWNSxY0dNmzZN9913X4E8Z2pqqjw8POz31Ro/frysVqtmzpyZ632fO3dO7dq106effmq/35ez2LVrl/bv369nnnkmT/c7cOBA1a9fX+PGjctyeWBgoJYsWaLWrVvn6fO6mjfeeEMnTpzQwoULjY4CAAAAAIYpbJ8DAACA7DGVOgAA/0hPT9fcuXPl7+9fYFfRJiUl6b777tP//d//yWq1KjIyUlu2bFFoaGie7L9MmTIaNGiQ3n///TzZX17asmWLmjRpYnSMIik2NlZff/21RowYYXQUAAAAADBMYfwcAAAAZI/COAAAkk6dOqWGDRtq9+7dmj17tv1b2/nN29tbc+fO1cqVK3Xvvfdq5MiReuqpp9SlS5c8e45Ro0bp3LlzWrVqVZ7tMy9MnDgx22nIkX9sNptefvllDRkyRMHBwUbHAQAAAABDFObPAQAAQNaYSh0AAAAAAAAAAAAAUKhxxTgAAAAAAAAAAAAAoFCjMA4AAAAAAAAAAAAAKNQojAMAAAAAAAAAAAAACjUK4wAAAAAAAAAAAACAQo3COAAAAAAAAAAAAACgUKMwDgAAAAAAAAAAAAAo1CiMAwAAAAAAAAAAAAAKNQrjAAAAAAAAAAAAAIBCjcI4AAAAAAAAAAAAAKBQ+3+5sZd8XUj+uAAAAABJRU5ErkJggg==
)
    


    Train users with zero history : 892  (1.78%)
    Train users with ≤5 history   : 12,979  (25.96%)



> **❄️ Cold-start implications:**
>
> The history-length distribution directly sets your cold-start strategy. Users with **zero history** cannot benefit from personalised retrieval (no clicks to aggregate into a taste vector or to look up similar articles from). The pipeline handles this with a **binary gate** in §7:
>
> ```
> is_cold(user) → True   ➜  return top-N global popularity articles
> is_cold(user) → False  ➜  run full personalised pipeline (S2 + S3 + S4)
> ```
>
> Even "warm" users with only 1–2 clicks have very noisy taste signals. The Bayesian smoothing in `bayesian_ctr` and the normalised affinity vectors are designed to degrade gracefully in this sparse regime.


[Back to top](#top)

---

## <a id="sec-3"></a>3. Feature engineering

We construct four reusable feature tables:

- **`user_stats`** — per-user: click count, active days, click frequency, favourite category
- **`article_feat`** — per-article: click count (log), Bayesian CTR, category one-hot, TF-IDF centroid
- **`user_cat_affinity`** — (user × category) matrix of normalised click preferences
- **`imp_train_df`** — impression-level (userId, newsId, label) frame with query groups for LambdaRank (Fix 1 & 2)

The TF-IDF vectoriser is fit on training article titles+abstracts only and transforms both train and dev articles, preventing feature leakage from future text.

`tfidf_sim` (cosine similarity between user click-history TF-IDF centroid and each candidate article) and `article_age_days` (log-scaled age since first impression, capturing news recency).



### 🛠️ Feature Engineering Map

Four complementary feature tables are constructed — each captures a different signal about users and articles:

```
                         FEATURE ENGINEERING
                         ═══════════════════

   ┌──────────────────────────────────────────────────────────────────────┐
   │  SOURCE: train_clicks (positive interactions only)                   │
   └───────────┬──────────────────────┬───────────────────────────────────┘
               │                      │
               ▼                      ▼
   ┌────────────────────┐   ┌──────────────────────────────────────────┐
   │   USER SIDE        │   │   ARTICLE SIDE                           │
   │                    │   │                                          │
   │  user_stats        │   │  article_feat                            │
   │  ─────────────     │   │  ─────────────                           │
   │  click_count       │   │  log_clicks  (log(1+n))                  │
   │  active_days       │   │  log_impr                                │
   │  click_freq        │   │  bayesian_ctr  ← smoothed CTR            │
   │  fav_category      │   │  article_len   ← title+abstract words    │
   │                    │   │  article_age_days                        │
   │  user_cat_affinity │   │  category one-hot (18 categories)        │
   │  ─────────────     │   │                                          │
   │  18-dim L2-norm    │   │  TF-IDF centroid (10k-dim, reduced)      │
   │  click distribution│   │                                          │
   └────────────────────┘   └──────────────────────────────────────────┘
               │                      │
               └──────────┬───────────┘
                          │ cross-signals
                          ▼
              ┌───────────────────────┐
              │  INTERACTION FEATURES │
              │                       │
              │  cat_affinity    ←  user_cat · article_cat (dot product)   │
              │  taste_affinity  ←  temporal_taste · article_cat           │
              │  tfidf_sim       ←  user_centroid · article_tfidf          │
              │  recent_tfidf_sim ←  recent-click centroid similarity      │
              └───────────────────────┘
```

> **Design principle:** Each feature is normalized to a comparable scale before being passed to LightGBM. Tree models are invariant to monotonic transforms, but consistent scaling improves interpretability of feature importances.



```python
# Compile user features
user_stats = train_clicks.groupby('userId').agg(click_count = ('newsId', 'count'),
                                                first_ts    = ('timestamp', 'min'),
                                                last_ts     = ('timestamp', 'max'),)

user_stats['active_days'] = ((user_stats['last_ts'] - user_stats['first_ts']) / 86400).clip(lower = 1).astype('float32')
user_stats['click_freq']  = (user_stats['click_count'] / user_stats['active_days']).astype('float32')

fav_cat = (train_with_cat.groupby(['userId','category'])['clicked'].count().reset_index().sort_values('clicked', ascending = False).drop_duplicates('userId').set_index('userId')['category'])

user_stats['fav_category'] = fav_cat
user_stats = user_stats.fillna({'fav_category': 'unknown'})

print(f'user_stats: {user_stats.shape}')

# Compile article features
article_feat = (pop_stats[['click_count','impressions','bayesian_ctr']].rename(columns={'click_count':'global_clicks','impressions':'global_impressions'}))
article_feat['log_clicks'] = np.log1p(article_feat['global_clicks']).astype('float32')
article_feat['log_impr']   = np.log1p(article_feat['global_impressions']).astype('float32')
article_feat = article_feat.join(news.set_index('newsId')[['category','subCategory','text']], how='left')
article_feat['article_len'] = article_feat['text'].fillna('').apply(len).astype('float32')

# Aticle recency — use earliest training impression as proxy for publish time
EVAL_TS = int(interactions_train['timestamp'].max())
article_first_seen = interactions_train.groupby('newsId')['timestamp'].min()
article_feat['article_age_days'] = (np.log1p((EVAL_TS - article_first_seen) / 86_400).clip(lower = 0).astype('float32').reindex(article_feat.index).fillna(article_feat['log_impr']))
print(f'article_feat: {article_feat.shape}')

# Sub-category click counts per user — finer-grained than category affinity
user_subcat_clicks = (train_with_cat.groupby(['userId', 'subCategory'])['clicked'].count().to_dict())
print(f'user_subcat_clicks entries: {len(user_subcat_clicks):,}')
```

    user_stats: (50000, 6)
    article_feat: (7713, 10)
    user_subcat_clicks entries: 188,670



```python
train_cat_vocab = pd.get_dummies(article_feat['category'].dropna(), prefix = 'cat').columns

all_news_cat    = news.set_index('newsId')['category'].dropna()
article_cat     = (pd.get_dummies(all_news_cat, prefix = 'cat').astype('float32').reindex(columns = train_cat_vocab, fill_value = 0))

cat_cols = article_cat.columns.tolist()
print(f'Category columns ({len(cat_cols)}): {cat_cols}')
print(f'article_cat covers {len(article_cat):,} articles  '
      f'(train: {len(article_feat):,}  dev-only: {len(article_cat)-len(article_feat):,})')
```

    Category columns (16): ['cat_autos', 'cat_entertainment', 'cat_finance', 'cat_foodanddrink', 'cat_health', 'cat_kids', 'cat_lifestyle', 'cat_movies', 'cat_music', 'cat_news', 'cat_northamerica', 'cat_sports', 'cat_travel', 'cat_tv', 'cat_video', 'cat_weather']
    article_cat covers 65,238 articles  (train: 7,713  dev-only: 57,525)



```python
user_stats.head()
```





  <div id="df-9b5fdc2f-ee4d-4ec3-b034-6d7e2e28bf8a" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>click_count</th>
      <th>first_ts</th>
      <th>last_ts</th>
      <th>active_days</th>
      <th>click_freq</th>
      <th>fav_category</th>
    </tr>
    <tr>
      <th>userId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>U100</th>
      <td>1</td>
      <td>1573544052</td>
      <td>1573544052</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>news</td>
    </tr>
    <tr>
      <th>U1000</th>
      <td>4</td>
      <td>1573686978</td>
      <td>1573771041</td>
      <td>1.0000</td>
      <td>4.0000</td>
      <td>news</td>
    </tr>
    <tr>
      <th>U10001</th>
      <td>3</td>
      <td>1573450221</td>
      <td>1573710414</td>
      <td>3.0115</td>
      <td>0.9962</td>
      <td>autos</td>
    </tr>
    <tr>
      <th>U10003</th>
      <td>3</td>
      <td>1573455962</td>
      <td>1573481638</td>
      <td>1.0000</td>
      <td>3.0000</td>
      <td>sports</td>
    </tr>
    <tr>
      <th>U10008</th>
      <td>1</td>
      <td>1573308813</td>
      <td>1573308813</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>weather</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-9b5fdc2f-ee4d-4ec3-b034-6d7e2e28bf8a')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-9b5fdc2f-ee4d-4ec3-b034-6d7e2e28bf8a button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-9b5fdc2f-ee4d-4ec3-b034-6d7e2e28bf8a');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>





```python
article_feat.head()
```





  <div id="df-7545a8f0-a48d-45f9-bc80-d4800e76ca5d" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>global_clicks</th>
      <th>global_impressions</th>
      <th>bayesian_ctr</th>
      <th>log_clicks</th>
      <th>log_impr</th>
      <th>category</th>
      <th>subCategory</th>
      <th>text</th>
      <th>article_len</th>
      <th>article_age_days</th>
    </tr>
    <tr>
      <th>newsId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>N10032</th>
      <td>1</td>
      <td>190</td>
      <td>0.0126</td>
      <td>0.6931</td>
      <td>5.2523</td>
      <td>foodanddrink</td>
      <td>recipes</td>
      <td>14 butternut squash recipes for delightfully c...</td>
      <td>172.0000</td>
      <td>0.2827</td>
    </tr>
    <tr>
      <th>N10051</th>
      <td>1</td>
      <td>370</td>
      <td>0.0072</td>
      <td>0.6931</td>
      <td>5.9162</td>
      <td>autos</td>
      <td>autosenthusiasts</td>
      <td>VW ID.3 Electric Motor Is So Compact That Fits...</td>
      <td>160.0000</td>
      <td>0.2640</td>
    </tr>
    <tr>
      <th>N10056</th>
      <td>6</td>
      <td>38</td>
      <td>0.0912</td>
      <td>1.9459</td>
      <td>3.6636</td>
      <td>sports</td>
      <td>football_nfl</td>
      <td>Russell Wilson, Richard Sherman swap jerseys d...</td>
      <td>176.0000</td>
      <td>1.4091</td>
    </tr>
    <tr>
      <th>N10057</th>
      <td>2</td>
      <td>41</td>
      <td>0.0442</td>
      <td>1.0986</td>
      <td>3.7377</td>
      <td>weather</td>
      <td>weathertopstories</td>
      <td>Venice swamped by highest tide in more than 50...</td>
      <td>243.0000</td>
      <td>1.0215</td>
    </tr>
    <tr>
      <th>N1006</th>
      <td>1</td>
      <td>2</td>
      <td>0.0581</td>
      <td>0.6931</td>
      <td>1.0986</td>
      <td>sports</td>
      <td>football_nfl</td>
      <td>Jaguars vs. Colts: A.J. Cann, Will Richardson ...</td>
      <td>487.0000</td>
      <td>0.2862</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-7545a8f0-a48d-45f9-bc80-d4800e76ca5d')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-7545a8f0-a48d-45f9-bc80-d4800e76ca5d button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-7545a8f0-a48d-45f9-bc80-d4800e76ca5d');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>





```python
article_cat.head()
```





  <div id="df-ee193fd1-80d0-46b6-a132-aea6a8070d55" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cat_autos</th>
      <th>cat_entertainment</th>
      <th>cat_finance</th>
      <th>cat_foodanddrink</th>
      <th>cat_health</th>
      <th>cat_kids</th>
      <th>cat_lifestyle</th>
      <th>cat_movies</th>
      <th>cat_music</th>
      <th>cat_news</th>
      <th>cat_northamerica</th>
      <th>cat_sports</th>
      <th>cat_travel</th>
      <th>cat_tv</th>
      <th>cat_video</th>
      <th>cat_weather</th>
    </tr>
    <tr>
      <th>newsId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>N55528</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>N19639</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>N61837</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>N53526</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>N38324</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-ee193fd1-80d0-46b6-a132-aea6a8070d55')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-ee193fd1-80d0-46b6-a132-aea6a8070d55 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-ee193fd1-80d0-46b6-a132-aea6a8070d55');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>





```python
# Perform TF-IDF
train_news_ids = set(train_clicks['newsId'].unique())

news_indexed  = news.set_index('newsId')
train_texts   = news_indexed.loc[news_indexed.index.isin(train_news_ids), 'text'].fillna('')

print('Fitting TF-IDF on train articles...', end = ' ', flush = True)
tfidf = TfidfVectorizer(max_features = 50000, sublinear_tf = True, min_df = 2, ngram_range = (1,2))
tfidf.fit(train_texts)
print('done.')

# Transform all articles (train + dev)
all_texts  = news_indexed['text'].fillna('')
tfidf_mat  = tfidf.transform(all_texts)    # sparse (n_articles, 5000)
tfidf_idx  = {nid: i for i, nid in enumerate(news_indexed.index)}
print(f'TF-IDF matrix: {tfidf_mat.shape}  nnz={tfidf_mat.nnz:,}')
```

    Fitting TF-IDF on train articles... done.
    TF-IDF matrix: (65238, 46525)  nnz=3,181,345



```python
%%time

# Build per-user TF-IDF centroids (click-history text profile)
# The centroid is the mean of the TF-IDF vectors of all articles a user has clicked,
# normalised to unit L2 so dot products equal cosine similarity at scoring time.
print('Building user TF-IDF centroids...', end = ' ', flush = True)

user_tfidf_centroids = {}

for uid, group in train_clicks.groupby('userId'):

    idxs = [tfidf_idx[nid] for nid in group['newsId'] if nid in tfidf_idx]

    if not idxs:

        continue

    centroid = np.asarray(tfidf_mat[idxs].mean(axis=0)).ravel()  # (10000,)
    norm = np.linalg.norm(centroid)

    if norm > 1e-9:

        user_tfidf_centroids[uid] = centroid / norm

print(f'done  ({len(user_tfidf_centroids):,} users have centroids)')

# Centroid of only the last 20 clicks — captures recent vs lifetime interest
print('Building recent TF-IDF centroids (last 20 clicks)...', end = ' ', flush = True)
user_recent_tfidf_centroids = {}

for uid, group in train_clicks.sort_values('timestamp').groupby('userId'):

    recent_nids = group['newsId'].tolist()[-20:]
    idxs = [tfidf_idx[nid] for nid in recent_nids if nid in tfidf_idx]

    if not idxs:

        continue

    centroid = np.asarray(tfidf_mat[idxs].mean(axis=0)).ravel()
    norm = np.linalg.norm(centroid)

    if norm > 1e-9:

        user_recent_tfidf_centroids[uid] = centroid / norm

print(f'done  ({len(user_recent_tfidf_centroids):,} users)')
```

    Building user TF-IDF centroids... done  (50,000 users have centroids)
    Building recent TF-IDF centroids (last 20 clicks)... done  (50,000 users)
    CPU times: user 17min 48s, sys: 1min 15s, total: 19min 3s
    Wall time: 2min 26s



```python
# Create an affinity matrix for user-category: compute normalised click counts per category
user_cat = (train_with_cat.groupby(['userId','category'])['clicked'].count().unstack(fill_value = 0).astype('float32'))

# Normalise rows to unit L2 norm
norms = np.linalg.norm(user_cat.values, axis = 1, keepdims = True).clip(min = 1e-9)
user_cat_affinity = pd.DataFrame(user_cat.values / norms, index   = user_cat.index, columns = user_cat.columns)

# Align article-category matrix columns with user-category matrix
article_cat_aligned = article_cat.reindex(columns = user_cat.columns, fill_value = 0)
article_cat_norm    = normalize(article_cat_aligned.values.astype('float32'), norm = 'l2', axis = 1)
article_cat_idx     = article_cat_aligned.index.tolist()

print(f'user_cat_affinity : {user_cat_affinity.shape}')
print(f'article_cat_norm  : {article_cat_norm.shape}')
user_cat_affinity.head(3)
```

    user_cat_affinity : (50000, 16)
    article_cat_norm  : (65238, 16)






  <div id="df-6a46972a-64c9-455c-a750-796a32981aa1" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>category</th>
      <th>autos</th>
      <th>entertainment</th>
      <th>finance</th>
      <th>foodanddrink</th>
      <th>health</th>
      <th>kids</th>
      <th>lifestyle</th>
      <th>movies</th>
      <th>music</th>
      <th>news</th>
      <th>northamerica</th>
      <th>sports</th>
      <th>travel</th>
      <th>tv</th>
      <th>video</th>
      <th>weather</th>
    </tr>
    <tr>
      <th>userId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>U100</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>U1000</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.4082</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.4082</td>
      <td>0.0000</td>
      <td>0.8165</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>U10001</th>
      <td>0.5774</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.5774</td>
      <td>0.5774</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-6a46972a-64c9-455c-a750-796a32981aa1')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-6a46972a-64c9-455c-a750-796a32981aa1 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-6a46972a-64c9-455c-a750-796a32981aa1');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>




Good to Know-

> **📐 Why L2-normalise the affinity vectors?**
>
> After normalising both `user_cat_affinity` (user rows) and `article_cat` (article rows) to unit L2 norm, their **dot product equals cosine similarity** — a value in [–1, 1] that measures directional agreement, independent of how many clicks a user has. This prevents heavy users (who click 200+ articles) from dominating the ranking signal purely because their affinity magnitudes are large.
>
> The same logic applies to TF-IDF centroids: unit-norm centroids mean that a user with 3 clicks and a user with 300 clicks are compared on the same scale when scoring article relevance.

[Back to top](#top)

---

## <a id="sec-4"></a>4. Article-based collaborative filtering

### 📖 Batched sparse co-click similarity

We build a **article × user click matrix** from positive training interactions, normalise rows (articles) to unit L2 norm, and compute cosine similarities between articles in batches to avoid OOM. The result is an `item_sim_lookup` dict mapping `newsId → [(newsId, similarity), …]` for the top-50 nearest neighbours.

This creates the **S3 retriever**: for a given user, find all articles they clicked, look up each article's nearest neighbours, aggregate scores (weighted by similarity × recency), and surface the top-N unseen articles.



```python
%%time

# Build article x user_click matrices
article_ids_cf = train_clicks['newsId'].unique()
user_ids_cf    = train_clicks['userId'].unique()

a_idx = {a: i for i, a in enumerate(article_ids_cf)}
u_idx = {u: i for i, u in enumerate(user_ids_cf)}
idx_a = {i: a for a, i in a_idx.items()}

R_cf = csr_matrix((np.ones(len(train_clicks), dtype='float32'), (train_clicks['newsId'].map(a_idx).values, train_clicks['userId'].map(u_idx).values)), shape = (len(article_ids_cf), len(user_ids_cf)))

R_norm_cf = normalize(R_cf, norm = 'l2', axis = 1)
print(f'Click matrix: {R_cf.shape}  nnz={R_cf.nnz:,}')
print(f'Memory: R={R_cf.data.nbytes/1e6:.0f} MB  R_norm={R_norm_cf.data.nbytes/1e6:.0f} MB')
```

    Click matrix: (7713, 50000)  nnz=234,468
    Memory: R=1 MB  R_norm=1 MB
    CPU times: user 149 ms, sys: 80 µs, total: 149 ms
    Wall time: 148 ms



```python
# Perform a batched knn to get similar articles
item_sim_lookup = {}
n_articles_cf   = R_norm_cf.shape[0]
t0              = time.time()

for start in range(0, n_articles_cf, 1000):

    batch = R_norm_cf[start : start + 1000]
    sims  = (batch @ R_norm_cf.T).toarray()

    for local_i, sim_row in enumerate(sims):

        global_i          = start + local_i
        sim_row[global_i] = 0.0
        top_k = np.argpartition(sim_row, -50)[-50:]
        top_k = top_k[np.argsort(sim_row[top_k])[::-1]]
        aid   = idx_a[global_i]
        item_sim_lookup[aid] = [(idx_a[j], float(sim_row[j])) for j in top_k]

    if start % 1000 == 0:
        print(f'  {start:>6}/{n_articles_cf}  {time.time()-t0:.0f}s')

del R_cf, R_norm_cf; gc.collect()
print(f'\nItem-sim lookup: {len(item_sim_lookup):,} articles in {time.time()-t0:.0f}s')
```

           0/7713  0s
        1000/7713  1s
        2000/7713  1s
        3000/7713  1s
        4000/7713  1s
        5000/7713  2s
        6000/7713  2s
        7000/7713  2s
    
    Item-sim lookup: 7,713 articles in 2s



> **🔗 How item-based CF works here:**
>
> The similarity lookup captures the intuition: *"users who clicked article A also tended to click article B."*
>
> ```
>   Article × User click matrix  R  (shape: 65K articles × 50K users)
>
>   R[i, u] = 1  if user u clicked article i, else 0
>
>   Normalise rows to unit L2:   R_norm = R / ||R||₂  (row-wise)
>
>   Similarity matrix:  S = R_norm · R_normᵀ   → cosine similarity between articles
>
>   item_sim_lookup[A] = top-50 articles by S[A, :]
> ```
>
> **Why batch the computation?** A full 65K × 65K similarity matrix would require ~17 GB of float32 memory. Processing in batches of 1,000 articles keeps peak memory under 2 GB by materialising only one slice at a time.
>
> **Retriever score for a user:** sum the similarity scores of all articles in the user's click history toward each candidate article — the more co-clicked history overlaps with the candidate, the higher its S3 score.


[Back to top](#top)

---

## <a id="sec-5"></a>5. Temporal sequence modelling

### 📖 Recency-weighted taste vectors

Recent clicks should dominate a user's preference profile — an article clicked yesterday matters more than one from three weeks ago. We compute per-user **category taste vectors** using exponential decay (half-life = 7 days, matching news freshness intuition). The resulting matrix enables fast batch dot-products at inference time.



```python
# Compute recency weighted taste vectors -  one week
DECAY_HALF_LIFE = 7
DECAY_K         = np.log(2) / DECAY_HALF_LIFE
now_ts          = int(train_clicks['timestamp'].max())

clicks_ts = train_clicks[['userId','newsId','timestamp']].copy()
clicks_ts['weight'] = np.exp(-DECAY_K * (now_ts - clicks_ts['timestamp'].values.astype('float64')) / 86400).astype('float32')

# Join category info for each click
clicks_ts = clicks_ts.join(news.set_index('newsId')[['category']], on='newsId')
clicks_ts = clicks_ts.dropna(subset=['category'])

# Aggregate: user × category, weighted by recency
user_taste = (clicks_ts.groupby(['userId','category'])['weight'].sum().unstack(fill_value=0).astype('float32'))

# Normalise to unit L2 so dot-products equal cosine similarity
taste_norms  = np.linalg.norm(user_taste.values, axis=1, keepdims=True).clip(min=1e-9)
user_taste_norm = pd.DataFrame(user_taste.values / taste_norms, index   = user_taste.index, columns = user_taste.columns)

# Align with article-category matrix
article_cat_taste = article_cat.reindex(columns=user_taste.columns, fill_value=0)
article_cat_taste_norm = normalize(article_cat_taste.values.astype('float32'), norm='l2', axis=1)
taste_article_idx = article_cat_taste.index.tolist()

print(f'user_taste_norm   : {user_taste_norm.shape}')
print(f'article_cat_taste : {article_cat_taste_norm.shape}')
user_taste_norm.head(3)
```

    user_taste_norm   : (50000, 16)
    article_cat_taste : (65238, 16)






  <div id="df-0fc10f57-01d6-43fa-82c2-c92b288869df" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>category</th>
      <th>autos</th>
      <th>entertainment</th>
      <th>finance</th>
      <th>foodanddrink</th>
      <th>health</th>
      <th>kids</th>
      <th>lifestyle</th>
      <th>movies</th>
      <th>music</th>
      <th>news</th>
      <th>northamerica</th>
      <th>sports</th>
      <th>travel</th>
      <th>tv</th>
      <th>video</th>
      <th>weather</th>
    </tr>
    <tr>
      <th>userId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>U100</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>U1000</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.4027</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.4402</td>
      <td>0.0000</td>
      <td>0.8025</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>U10001</th>
      <td>0.6261</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.4647</td>
      <td>0.6261</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-0fc10f57-01d6-43fa-82c2-c92b288869df')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-0fc10f57-01d6-43fa-82c2-c92b288869df button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-0fc10f57-01d6-43fa-82c2-c92b288869df');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>





### ⏱️ Exponential Decay — The Intuition

The recency weight for a click is:

$$w(t) = e^{-k \cdot \Delta t_{days}}, \quad k = \frac{\ln 2}{\text{half-life}}$$

With **half-life = 7 days**, a click from:

| Days ago | Weight |
|----------|--------|
| 0 (today) | 1.000 |
| 3.5 days | ~0.707 |
| 7 days | **0.500** ← half-life |
| 14 days | 0.250 |
| 28 days | 0.063 |
| 42 days | 0.016 |

The six-week MIND window (Oct 12–Nov 22) means clicks from the *start* of the window receive weight ≈ 0.016 relative to the most recent clicks — effectively negligible. This mirrors real editorial news consumption where interests shift week-to-week.

> **Alternative half-lives to consider:** Shorter (3 days) captures breaking-news spikes; longer (14 days) suits evergreen topic interests (e.g. a user researching a health condition over two weeks). The 7-day default is a reasonable starting point for general news.



```python
# Feature column list — all base LGB and meta-ranker base features
FEATURE_COLS = ['u_click_count', 'u_click_freq',      # user engagement
                'm_log_clicks',  'm_log_impr',        # article global popularity
                'm_article_len',                       # article length
                'cat_affinity',  'taste_affinity',    # collaborative signals
                'tfidf_sim',                           # content similarity (full history centroid)
                'recent_tfidf_sim',                    # content similarity (last-20 clicks centroid)
                'article_age_days',                    # news recency
                'ctr_norm_rank',                       # rank by CTR within impression (0=most popular)
                'imp_size',                            # number of candidates in impression
                'subcat_clicks']                       # user click count for this sub-category

# Select num candidates for training
K_CAND  = 200
rng_feat = np.random.default_rng(100)
print(f'FEATURE_COLS ({len(FEATURE_COLS)}): {FEATURE_COLS}')
```

    FEATURE_COLS (13): ['u_click_count', 'u_click_freq', 'm_log_clicks', 'm_log_impr', 'm_article_len', 'cat_affinity', 'taste_affinity', 'tfidf_sim', 'recent_tfidf_sim', 'article_age_days', 'ctr_norm_rank', 'imp_size', 'subcat_clicks']



```python
# Split the data for training / testing (70-30)
rng_oof    = np.random.default_rng(42)
all_train_users = np.array(list(train_users & set(user_stats.index)))
rng_oof.shuffle(all_train_users)

split_idx  = int(len(all_train_users) * 0.70)
SET_A_users = set(all_train_users[:split_idx])
SET_B_users = set(all_train_users[split_idx:])

print(f'Training users total : {len(all_train_users):,}')
print(f'  SET_A (base LGB)   : {len(SET_A_users):,}')
print(f'  SET_B (meta OOF)   : {len(SET_B_users):,}')

# user_click_sets is needed by s3_itemcf and training loops
user_click_sets = train_clicks.groupby('userId')['newsId'].apply(set).to_dict()
```

    Training users total : 50,000
      SET_A (base LGB)   : 35,000
      SET_B (meta OOF)   : 15,000



```python
# Pre-compile feature dicts (O(1) lookups at scoring time)
art_pos   = {a: i for i, a in enumerate(article_cat_idx)}
taste_pos = {a: i for i, a in enumerate(taste_article_idx)}

af_log_clicks   = article_feat['log_clicks'].to_dict()
af_log_impr     = article_feat['log_impr'].to_dict()
af_bayesian_ctr = article_feat['bayesian_ctr'].to_dict()
af_article_len  = article_feat['article_len'].to_dict()
af_article_age  = article_feat['article_age_days'].to_dict()

us_click_count  = user_stats['click_count'].to_dict()
us_click_freq   = user_stats['click_freq'].to_dict()

newsid_to_subcat = news.set_index('newsId')['subCategory'].to_dict()
```


```python
%%time

# Build training pairs from actual MIND impression rows (SET_A users only).
# Each impression is one ranking query; every article shown is a candidate;
# the click label is the ground truth. This aligns train and eval distributions.

print('Parsing training impressions for SET_A users...', end = ' ', flush = True)

# Init
imp_rows = []

# Iterate
for _, r in raw_train.iterrows():

    uid = r['userId']

    if uid not in SET_A_users:

        continue

    imp_id = r['impressionId']

    if pd.notna(r['impressions']):

        for pair in str(r['impressions']).split():

            nid, lbl = pair.rsplit('-', 1)
            imp_rows.append((imp_id, uid, str(nid), int(lbl)))

# Compile the iterations
imp_train_df = pd.DataFrame(imp_rows, columns = ['impressionId','userId','newsId','label'])
del imp_rows; gc.collect()

n_pos = int(imp_train_df['label'].sum())
print(f'done  ({len(imp_train_df):,} rows | {imp_train_df["impressionId"].nunique():,} impressions | '
      f'pos={n_pos:,} neg={len(imp_train_df)-n_pos:,})')

# Merge user fts
imp_train_df = imp_train_df.join(user_stats[['click_count','click_freq']].rename(columns = {'click_count':'u_click_count','click_freq':'u_click_freq'}), on = 'userId')
imp_train_df = imp_train_df.join(article_feat[['log_clicks','log_impr','bayesian_ctr','article_len','article_age_days']].rename(columns = {'log_clicks':'m_log_clicks','log_impr':'m_log_impr', 'bayesian_ctr':'m_bayesian_ctr','article_len':'m_article_len', 'article_age_days':'article_age_days'}), on = 'newsId')

# Merge category and taste affinity
newsid_to_cat       = news.set_index('newsId')['category'].to_dict()
imp_train_df['category'] = imp_train_df['newsId'].map(newsid_to_cat)

relevant_users = imp_train_df['userId'].unique()

uca_long = (user_cat_affinity.reindex(index=relevant_users).stack().reset_index().rename(columns={'level_0':'userId','level_1':'category',0:'cat_affinity'}))
imp_train_df = imp_train_df.merge(uca_long, on = ['userId','category'], how = 'left')

del uca_long

uta_long = (user_taste_norm.reindex(index=relevant_users).stack().reset_index().rename(columns = {'level_0':'userId','level_1':'category',0:'taste_affinity'}))
imp_train_df = imp_train_df.merge(uta_long, on = ['userId','category'], how = 'left')

del uta_long; gc.collect()

# Compute the tf-idf similarities
print('Computing TF-IDF affinities...', end = ' ', flush = True)

uid_nid_sim = {}

for uid, grp in imp_train_df.groupby('userId'):

    centroid = user_tfidf_centroids.get(uid)

    if centroid is None:

        continue

    nids  = grp['newsId'].unique()
    idxs  = [tfidf_idx[nid] for nid in nids if nid in tfidf_idx]
    valid = [(nid, tfidf_idx[nid]) for nid in nids if nid in tfidf_idx]

    if not valid:

        continue

    v_nids, v_idxs = zip(*valid)
    sims = np.asarray(tfidf_mat[list(v_idxs)].dot(centroid)).ravel()

    for nid, sim in zip(v_nids, sims):

        uid_nid_sim[(uid, nid)] = float(sim)

imp_train_df['tfidf_sim'] = [uid_nid_sim.get((r.userId, r.newsId), 0.0) for r in imp_train_df.itertuples()]
del uid_nid_sim; gc.collect()
print('done.')

# recent_tfidf_sim — centroid of user's last 20 clicks
print('Computing recent TF-IDF affinities...', end = ' ', flush = True)
uid_nid_recent_sim = {}

for uid, grp in imp_train_df.groupby('userId'):

    centroid = user_recent_tfidf_centroids.get(uid)

    if centroid is None:

        continue

    valid = [(nid, tfidf_idx[nid]) for nid in grp['newsId'].unique() if nid in tfidf_idx]

    if not valid:

        continue

    v_nids, v_idxs = zip(*valid)
    sims = np.asarray(tfidf_mat[list(v_idxs)].dot(centroid)).ravel()

    for nid, sim in zip(v_nids, sims):

        uid_nid_recent_sim[(uid, nid)] = float(sim)

imp_train_df['recent_tfidf_sim'] = [uid_nid_recent_sim.get((r.userId, r.newsId), 0.0) for r in imp_train_df.itertuples()]
del uid_nid_recent_sim; gc.collect()
print('done.')

# subcat_clicks — user click count for candidate's specific sub-category
imp_train_df['_subcat'] = imp_train_df['newsId'].map(newsid_to_subcat)
_subcat_lkp = pd.DataFrame([(u, sc, cnt) for (u, sc), cnt in user_subcat_clicks.items()], columns = ['userId', '_subcat', 'subcat_clicks'])
imp_train_df = imp_train_df.merge(_subcat_lkp, on=['userId', '_subcat'], how='left')
imp_train_df['subcat_clicks'] = imp_train_df['subcat_clicks'].fillna(0).astype('float32')
imp_train_df.drop(columns=['_subcat'], inplace=True)
del _subcat_lkp

# Within-impression context features
imp_train_df['imp_size'] = (imp_train_df.groupby('impressionId')['newsId'].transform('count').astype('float32'))
imp_train_df['ctr_norm_rank'] = (imp_train_df.groupby('impressionId')['m_bayesian_ctr'].transform(lambda x: (x.rank(ascending=False, method='average') - 1).div(max(1, len(x) - 1))).astype('float32'))
imp_train_df[FEATURE_COLS] = imp_train_df[FEATURE_COLS].fillna(0).astype('float32')
print(f'imp_train_df shape: {imp_train_df.shape}')
```

    Parsing training impressions for SET_A users... done  (4,090,484 rows | 110,162 impressions | pos=165,852 neg=3,924,632)
    Computing TF-IDF affinities... done.
    Computing recent TF-IDF affinities... done.
    imp_train_df shape: (4090484, 19)
    CPU times: user 1min 24s, sys: 1.27 s, total: 1min 25s
    Wall time: 1min 25s



```python
imp_train_df.head()
```





  <div id="df-610ad2b8-dedb-4458-a9f6-b6555a839af1" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>impressionId</th>
      <th>userId</th>
      <th>newsId</th>
      <th>label</th>
      <th>u_click_count</th>
      <th>u_click_freq</th>
      <th>m_log_clicks</th>
      <th>m_log_impr</th>
      <th>m_bayesian_ctr</th>
      <th>m_article_len</th>
      <th>article_age_days</th>
      <th>category</th>
      <th>cat_affinity</th>
      <th>taste_affinity</th>
      <th>tfidf_sim</th>
      <th>recent_tfidf_sim</th>
      <th>subcat_clicks</th>
      <th>imp_size</th>
      <th>ctr_norm_rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>U73700</td>
      <td>N50014</td>
      <td>0</td>
      <td>3.0000</td>
      <td>1.8087</td>
      <td>3.8067</td>
      <td>8.2895</td>
      <td>0.0114</td>
      <td>163.0000</td>
      <td>1.1812</td>
      <td>sports</td>
      <td>0.8944</td>
      <td>0.8616</td>
      <td>0.0236</td>
      <td>0.0236</td>
      <td>0.0000</td>
      <td>36.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>U73700</td>
      <td>N23877</td>
      <td>0</td>
      <td>3.0000</td>
      <td>1.8087</td>
      <td>6.4232</td>
      <td>9.3310</td>
      <td>0.0545</td>
      <td>340.0000</td>
      <td>0.7478</td>
      <td>news</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0219</td>
      <td>0.0219</td>
      <td>0.0000</td>
      <td>36.0000</td>
      <td>0.2857</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>U73700</td>
      <td>N35389</td>
      <td>0</td>
      <td>3.0000</td>
      <td>1.8087</td>
      <td>5.5053</td>
      <td>8.1259</td>
      <td>0.0720</td>
      <td>244.0000</td>
      <td>1.1917</td>
      <td>finance</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0423</td>
      <td>0.0423</td>
      <td>0.0000</td>
      <td>36.0000</td>
      <td>0.0857</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>U73700</td>
      <td>N49712</td>
      <td>0</td>
      <td>3.0000</td>
      <td>1.8087</td>
      <td>6.2305</td>
      <td>8.9469</td>
      <td>0.0658</td>
      <td>290.0000</td>
      <td>0.8228</td>
      <td>news</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0161</td>
      <td>0.0161</td>
      <td>0.0000</td>
      <td>36.0000</td>
      <td>0.1429</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>U73700</td>
      <td>N16844</td>
      <td>0</td>
      <td>3.0000</td>
      <td>1.8087</td>
      <td>5.5294</td>
      <td>8.4845</td>
      <td>0.0518</td>
      <td>278.0000</td>
      <td>1.1625</td>
      <td>autos</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0219</td>
      <td>0.0219</td>
      <td>0.0000</td>
      <td>36.0000</td>
      <td>0.3143</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-610ad2b8-dedb-4458-a9f6-b6555a839af1')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-610ad2b8-dedb-4458-a9f6-b6555a839af1 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-610ad2b8-dedb-4458-a9f6-b6555a839af1');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>





```python
# Training data summary
print(imp_train_df.dtypes)
print(f'\nLabel distribution:\n{imp_train_df["label"].value_counts()}')
```

    impressionId          int64
    userId               object
    newsId               object
    label                 int64
    u_click_count       float32
    u_click_freq        float32
    m_log_clicks        float32
    m_log_impr          float32
    m_bayesian_ctr      float32
    m_article_len       float32
    article_age_days    float32
    category             object
    cat_affinity        float32
    taste_affinity      float32
    tfidf_sim           float32
    recent_tfidf_sim    float32
    subcat_clicks       float32
    imp_size            float32
    ctr_norm_rank       float32
    dtype: object
    
    Label distribution:
    label
    0    3924632
    1     165852
    Name: count, dtype: int64



> **📋 `imp_train_df` — the learning-to-rank training table:**
>
> Each row is one **(user, article, impression)** triple from an actual MIND impression session. The `impressionId` groups rows so the ranker knows which candidates competed against each other in the same session:
>
> ```
> impressionId | userId | newsId | label | u_click_count | m_log_clicks | cat_affinity | … | tfidf_sim
> ─────────────┼────────┼────────┼───────┼───────────────┼──────────────┼──────────────┼───┼──────────
> imp-001      | U1234  | N5001  |   1   |    42         |    2.30      |    0.81      | … |   0.67
> imp-001      | U1234  | N5002  |   0   |    42         |    1.10      |    0.23      | … |   0.12
> imp-001      | U1234  | N5003  |   0   |    42         |    3.45      |    0.61      | … |   0.44
> imp-002      | U9876  | N1001  |   0   |     8         |    2.30      |    0.05      | … |   0.31
> …
> ```
>
> The `impressionId` column becomes the **LightGBM query group** — the model is told "these rows compete against each other, optimise their relative ordering" via LambdaRank.


[Back to top](#top)

---

## <a id="sec-6"></a>6. Evaluation harness & S1–S5 strategies

Five metrics evaluated at K = 5 and K = 10.
**Composite score = mean(NDCG@K, Hit-Rate@K)** — avoids double-counting
Precision and Recall through F1.

| Strategy | Description |
|----------|-------------|
| **S1** | Global popularity — Bayesian CTR ranking |
| **S2** | Category affinity — dot product of user preferences with article categories |
| **S3** | Item-based CF — aggregate neighbour scores from clicked articles |
| **S4** | Temporal taste — recency-weighted category preference |
| **S5** | LightGBM LambdaRank ranker |



### 📏 Evaluation Metrics — Quick Reference

All metrics are computed **per impression** (one ranking query = one session), then averaged across users. K ∈ {5, 10} controls the *cutoff* — only the top-K predicted articles count.

| Metric | Formula (simplified) | Interpretation |
|--------|----------------------|----------------|
| **Precision@K** | (# clicked in top-K) / K | Of K articles shown, how many did the user click? |
| **Recall@K** | (# clicked in top-K) / (# total clicks in session) | Of all clicked articles, how many were in top-K? |
| **F1@K** | 2 · P · R / (P + R) | Harmonic mean of precision and recall |
| **NDCG@K** | DCG@K / IDCG@K | Position-weighted relevance; clicked articles ranked first score highest |
| **Hit-Rate@K** | 1 if ≥ 1 clicked article in top-K else 0 | Did the user find at least one article they liked? |
| **Composite** | mean(NDCG@K, HR@K) | Summary score used for leaderboard ranking |

> **Why Composite = mean(NDCG, HR)?** Using their mean avoids double-counting the Precision and Recall components that are already captured by F1, while still rewarding both ranked quality (NDCG) and binary coverage (HR).

> **Why per-impression, not global?** Evaluating globally would mix impressions from different sessions and let popular articles dominate. Per-impression evaluation mirrors deployment: the model ranks *a specific set of candidates* for *one user at one moment*.



```python
%%time

# LambdaRank objective with per-impression query groups- LambdaMART directly optimises NDCG within each impression list

# Sort by impressionId so groups are contiguous
imp_train_df = imp_train_df.sort_values('impressionId').reset_index(drop = True)

# 85 / 15 impression-level split (no leakage across impression boundaries)
all_imp_ids = imp_train_df['impressionId'].unique()
rng_ltr     = np.random.default_rng(100)
val_imp_ids = set(rng_ltr.choice(all_imp_ids, size=int(len(all_imp_ids) * 0.15), replace=False))

tr_mask  = ~imp_train_df['impressionId'].isin(val_imp_ids)
val_mask =  imp_train_df['impressionId'].isin(val_imp_ids)

# Recompute bayesian_ctr from train-fold impressions only, then apply to both splits
tr_imp_df = imp_train_df[tr_mask]
fold_pop  = (tr_imp_df.groupby('newsId')['label'].agg(['sum', 'count']).rename(columns = {'sum': 'clicks', 'count': 'impr'}))
fold_ctr  = ((fold_pop['clicks'] + C * GLOBAL_CTR) / (fold_pop['impr'] + C))

# Unseen articles keep global estimate
imp_train_df['m_bayesian_ctr'] = (imp_train_df['newsId'].map(fold_ctr).fillna(imp_train_df['m_bayesian_ctr']).astype('float32'))
del tr_imp_df, fold_pop, fold_ctr

# Refresh ctr_norm_rank using fold-corrected CTR values
imp_train_df['ctr_norm_rank'] = (imp_train_df.groupby('impressionId')['m_bayesian_ctr'].transform(lambda x: (x.rank(ascending=False, method='average') - 1).div(max(1, len(x) - 1))).astype('float32'))

x_tr  = imp_train_df.loc[tr_mask,  FEATURE_COLS].values.astype('float32')
y_tr  = imp_train_df.loc[tr_mask,  'label'].values.astype('int')
g_tr  = imp_train_df.loc[tr_mask].groupby('impressionId', sort=True).size().values

x_val = imp_train_df.loc[val_mask, FEATURE_COLS].values.astype('float32')
y_val = imp_train_df.loc[val_mask, 'label'].values.astype('int')
g_val = imp_train_df.loc[val_mask].groupby('impressionId', sort=True).size().values

lgb_params = {'objective'        : 'lambdarank',
            'metric'           : 'ndcg',
            'ndcg_eval_at'     : [5, 10],
            'label_gain'       : [0, 1],
            'learning_rate'    : 0.05,
            'feature_fraction' : 0.8,
            'bagging_fraction' : 0.8,
            'bagging_freq'     : 5,
            'min_child_samples': 5,
            'verbose'          : -1,
            'n_jobs'           : -1,}

lgb_model = lgb.train(lgb_params, lgb.Dataset(x_tr, label = y_tr, group = g_tr), num_boost_round = 800, valid_sets      = [lgb.Dataset(x_val, label = y_val, group = g_val)], callbacks       = [lgb.early_stopping(50, verbose = False), lgb.log_evaluation(100)],)

del x_tr, x_val, y_tr, y_val; gc.collect()
print(f'\nBase LGB trees: {lgb_model.num_trees()}')
print(f'Features used : {FEATURE_COLS}')
```

    [100]	valid_0's ndcg@5: 0.96686	valid_0's ndcg@10: 0.969242
    [200]	valid_0's ndcg@5: 0.967218	valid_0's ndcg@10: 0.969574
    
    Base LGB trees: 233
    Features used : ['u_click_count', 'u_click_freq', 'm_log_clicks', 'm_log_impr', 'm_article_len', 'cat_affinity', 'taste_affinity', 'tfidf_sim', 'recent_tfidf_sim', 'article_age_days', 'ctr_norm_rank', 'imp_size', 'subcat_clicks']
    CPU times: user 4min 27s, sys: 598 ms, total: 4min 28s
    Wall time: 58.5 s



### 🌲 LambdaMART — Why It's the Right Objective Here

Standard classification loss (binary cross-entropy) treats every mis-ranked pair equally. But in news recommendation, **the rank matters**: predicting a click at position 1 is far more valuable than at position 10.

**LambdaRank** (implemented via LightGBM's `lambdarank` objective) directly optimises NDCG by computing *lambda gradients* — pair-wise adjustment weights that scale each gradient by the NDCG improvement that would result from swapping that pair's positions:

```
   λᵢⱼ = |ΔNDCG(swap i ↔ j)| · σ(sⱼ - sᵢ)
              ↑                      ↑
   how much the swap helps     logistic margin
```

The `query_group` parameter tells LightGBM which rows belong to the same ranking query (same impression), so pairwise comparisons are made *within sessions only* — exactly matching the evaluation setup.

> **Practical consequence:** LambdaMART generally outperforms pointwise (logistic regression, XGBoost on binary labels) and pairwise (BPR) methods by 2–5 NDCG points on standard LTR benchmarks. The gain compounds in §9 when the meta-ranker uses the base LGB's OOF scores as a feature.

[Back to top](#top)

---

## <a id="sec-7"></a>7. S6 architecture & cold-start gate

### Architecture

![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABaAAAAOKCAYAAABtR/eiAAAEDmlDQ1BrQ0dDb2xvclNwYWNlR2VuZXJpY1JHQgAAOI2NVV1oHFUUPpu5syskzoPUpqaSDv41lLRsUtGE2uj+ZbNt3CyTbLRBkMns3Z1pJjPj/KRpKT4UQRDBqOCT4P9bwSchaqvtiy2itFCiBIMo+ND6R6HSFwnruTOzu5O4a73L3PnmnO9+595z7t4LkLgsW5beJQIsGq4t5dPis8fmxMQ6dMF90A190C0rjpUqlSYBG+PCv9rt7yDG3tf2t/f/Z+uuUEcBiN2F2Kw4yiLiZQD+FcWyXYAEQfvICddi+AnEO2ycIOISw7UAVxieD/Cyz5mRMohfRSwoqoz+xNuIB+cj9loEB3Pw2448NaitKSLLRck2q5pOI9O9g/t/tkXda8Tbg0+PszB9FN8DuPaXKnKW4YcQn1Xk3HSIry5ps8UQ/2W5aQnxIwBdu7yFcgrxPsRjVXu8HOh0qao30cArp9SZZxDfg3h1wTzKxu5E/LUxX5wKdX5SnAzmDx4A4OIqLbB69yMesE1pKojLjVdoNsfyiPi45hZmAn3uLWdpOtfQOaVmikEs7ovj8hFWpz7EV6mel0L9Xy23FMYlPYZenAx0yDB1/PX6dledmQjikjkXCxqMJS9WtfFCyH9XtSekEF+2dH+P4tzITduTygGfv58a5VCTH5PtXD7EFZiNyUDBhHnsFTBgE0SQIA9pfFtgo6cKGuhooeilaKH41eDs38Ip+f4At1Rq/sjr6NEwQqb/I/DQqsLvaFUjvAx+eWirddAJZnAj1DFJL0mSg/gcIpPkMBkhoyCSJ8lTZIxk0TpKDjXHliJzZPO50dR5ASNSnzeLvIvod0HG/mdkmOC0z8VKnzcQ2M/Yz2vKldduXjp9bleLu0ZWn7vWc+l0JGcaai10yNrUnXLP/8Jf59ewX+c3Wgz+B34Df+vbVrc16zTMVgp9um9bxEfzPU5kPqUtVWxhs6OiWTVW+gIfywB9uXi7CGcGW/zk98k/kmvJ95IfJn/j3uQ+4c5zn3Kfcd+AyF3gLnJfcl9xH3OfR2rUee80a+6vo7EK5mmXUdyfQlrYLTwoZIU9wsPCZEtP6BWGhAlhL3p2N6sTjRdduwbHsG9kq32sgBepc+xurLPW4T9URpYGJ3ym4+8zA05u44QjST8ZIoVtu3qE7fWmdn5LPdqvgcZz8Ww8BWJ8X3w0PhQ/wnCDGd+LvlHs8dRy6bLLDuKMaZ20tZrqisPJ5ONiCq8yKhYM5cCgKOu66Lsc0aYOtZdo5QCwezI4wm9J/v0X23mlZXOfBjj8Jzv3WrY5D+CsA9D7aMs2gGfjve8ArD6mePZSeCfEYt8CONWDw8FXTxrPqx/r9Vt4biXeANh8vV7/+/16ffMD1N8AuKD/A/8leAvFY9bLAAAAXGVYSWZNTQAqAAAACAAEAQYAAwAAAAEAAgAAARIAAwAAAAEAAQAAASgAAwAAAAEAAgAAh2kABAAAAAEAAAA+AAAAAAACoAIABAAAAAEAAAWgoAMABAAAAAEAAAOKAAAAAJydkswAAAILaVRYdFhNTDpjb20uYWRvYmUueG1wAAAAAAA8eDp4bXBtZXRhIHhtbG5zOng9ImFkb2JlOm5zOm1ldGEvIiB4OnhtcHRrPSJYTVAgQ29yZSA2LjAuMCI+CiAgIDxyZGY6UkRGIHhtbG5zOnJkZj0iaHR0cDovL3d3dy53My5vcmcvMTk5OS8wMi8yMi1yZGYtc3ludGF4LW5zIyI+CiAgICAgIDxyZGY6RGVzY3JpcHRpb24gcmRmOmFib3V0PSIiCiAgICAgICAgICAgIHhtbG5zOnRpZmY9Imh0dHA6Ly9ucy5hZG9iZS5jb20vdGlmZi8xLjAvIj4KICAgICAgICAgPHRpZmY6UmVzb2x1dGlvblVuaXQ+MjwvdGlmZjpSZXNvbHV0aW9uVW5pdD4KICAgICAgICAgPHRpZmY6T3JpZW50YXRpb24+MTwvdGlmZjpPcmllbnRhdGlvbj4KICAgICAgICAgPHRpZmY6Q29tcHJlc3Npb24+MTwvdGlmZjpDb21wcmVzc2lvbj4KICAgICAgICAgPHRpZmY6UGhvdG9tZXRyaWNJbnRlcnByZXRhdGlvbj4yPC90aWZmOlBob3RvbWV0cmljSW50ZXJwcmV0YXRpb24+CiAgICAgIDwvcmRmOkRlc2NyaXB0aW9uPgogICA8L3JkZjpSREY+CjwveDp4bXBtZXRhPgqWqErQAABAAElEQVR4AezdB5xU1b048DO7CyxdQBEbKvbeEyv2XqLJi0lMeSbxJfpP8l6Sl8T4YuqL0ZeYxBhTFLuIgIAgTbBQ1dg1RqVXpfe2hd2Z/z2Ds8wuICALzO5+z+ezzJ17zz33nO+ZGPjN2d8JQSFAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIENhAILXBGScIECBAgAABAgQIbAeBW4fP3S8Tmu+RTlftGULRbtvhEZokQIBAgxUoLgnpTHHxnKry6rlt27Wa873urec22MHoOAECBAgQyBMQgM7DcEiAAAECBAgQIFB/AncOW9yuPJW6dFXFqi+UNGt+fro6XbR6TcXalatWpFavLitJAtHN6+9pWiJAgEDDFkgVFVW1b9euqm2bNpnWLVs2X7u2cvna6qrB7XZp2/Om83Z9pmGPTu8JECBAoCkLCEA35dk3dgIECBAgQIDAdhAYNSpTMn7NgptDqugn8+bNL58waVKbKdOmh8VLlm6Hp2mSAAECjVNgn732DAd02z8cevABlS1KS5cWp5rd9Ksr93igcY7WqAgQIECgMQsIQDfm2TU2AgQIECBAgMAOFrhl5JJvVVeW/++sWXObjX7++Tbz5i/YwT3wOAIECDQ+gcMOOSiccerJlc1bNJu5S9s21/34ws5jG98ojYgAAQIEGquAAHRjnVnjIkCAAAECBAjsYIEf9ZnWs6RZ8RWDh41sO33GzB38dI8jQIBA4xc4/pijw4XnnR3mz593853XHnVL4x+xERIgQIBAYxAQgG4Ms2gMBAgQIECAAIGdKNC3b6b566kZY1etXH1Q34FPdlyzZs1O7I1HEyBAoHEL7LfvPuHKyy+pWLFsRb87rz3iS417tEZHgAABAo1BoKgxDMIYCBAgQIAAAQIEdp7Aa5lpT30wd/4BD/bqLfi886bBkwkQaCICM2bODg8+8liLlq1aXvXtB969o4kM2zAJECBAoAELCEA34MnTdQIECBAgQIDAzha4sc/Uv5dXVB79xJNDdt3ZffF8AgQINBWBZctXhP6DhrTqvNuu37nhnjevbyrjNk4CBAgQaJgCAtANc970mgABAgQIECCw0wV+PXLRVS1KW17Tb+CTnXZ6Z3SAAAECTUwgbvL65LDhRV327PLn20et8CVgE5t/wyVAgEBDEhCAbkizpa8ECBAgQIAAgQISqFxT9tvnRo9rvnzFygLqla4QIECg6QhMmDQ5JD+ZmbNm39d0Rm2kBAgQINDQBASgG9qM6S8BAgQIECBAoAAEfjVk4TfWrC7r8Obbb7cogO7oAgECBJqswLjn/9GsS+fOV/xm2OLDmyyCgRMgQIBAQQsIQBf09OgcAQIECBAgQKBQBdI3Pjt2vNQbhTo9+kWAQJMRWLZ8efjHK6+FxUuX/F+TGbSBEiBAgECDEhCAblDTpbMECBAgQIAAgZ0v8Pun5h+TTld3mTx12s7vjB4QIECAQHhv4qTQuk3rM1EQIECAAIFCFBCALsRZ0ScCBAgQIECAQAELrKkOV06bMWN1AXdR1wgQINCkBOKGhJWVa0tue3rJ6U1q4AZLgAABAg1CQAC6QUyTThIgQIAAAQIECkcglcpc9s57k3YrnB7pCQECBAhMmDSp5dLlK75CggABAgQIFJqAAHShzYj+ECBAgAABAgQKXKA6k+m2aPGSAu+l7hEgQKBpCSxZujSkq9P7N61RGy0BAgQINAQBAeiGMEv6SIAAAQIECBAoIIGSopKOq1bLwFFAU6IrBAgQCKtWrQ4lqaI9URAgQIAAgUITEIAutBnRHwIECBAgQIBAAQv8cezqPdLpdGVFRUUB91LXCBAg0PQE4heDJSUluza9kRsxAQIECBS6gAB0oc+Q/hEgQIAAAQIECkigurq6ffXa6soC6pKuECBAgEAiEL8YLCkubgmDAAECBAgUmoAAdKHNiP4QIECAAAECBAgQIECAAAECBAgQIECgkQgIQDeSiTQMAgQIECBAgAABAgQIECBAgAABAgQIFJqAAHShzYj+ECBAgAABAgQIECBAgAABAgQIECBAoJEICEA3kok0DAIECBAgQIAAAQIECBAgQIAAAQIECBSagAB0oc2I/hAgQIAAAQIECBAgQIAAAQIECBAgQKCRCAhAN5KJNAwCBAgQIECAAAECBAgQIECAAAECBAgUmoAAdKHNiP4QIECAAAECBAgQIECAAAECBAgQIECgkQgIQDeSiTQMAgQIECBAgAABAgQIECBAgAABAgQIFJqAAHShzYj+ECBAgAABAgQIECBAgAABAgQIECBAoJEICEA3kok0DAIECBAgQIAAAQIECBAgQIAAAQIECBSagAB0oc2I/hAgQIAAAQIECBAgQIAAAQIECBAgQKCRCAhAN5KJNAwCBAgQIECAAAECBAgQIECAAAECBAgUmoAAdKHNiP4QIECAAAECBAgQIECAAAECBAgQIECgkQgIQDeSiTQMAgQIECBAgAABAgQIECBAgAABAgQIFJqAAHShzYj+ECBAgAABAgQIECBAgAABAgQIECBAoJEICEA3kok0DAIECBAgQIAAAQIECBAgQIAAAQIECBSagAB0oc2I/hAgQIAAAQIECBAgQIAAgY8hkMlkUh/jNrcQIECAAIHtKiAAvV15NU6AAAECBAgQIECAAAECBAgQIECAAIGmKyAA3XTn3sgJECBAgAABAgQIECBAgAABAgQIECCwXQUEoLcrr8YJECBAgAABAgQIECBAgAABAgQIECDQdAUEoJvu3Bs5AQIECBAgQIAAAQIECBAgQIAAAQIEtquAAPR25dU4AQIECBAgQIAAAQIECBAgQIAAAQIEmq6AAHTTnXsjJ0CAAAECBAgQIECAAAECBAgQIECAwHYVEIDerrwaJ0CAAAECBAgQIECAAAECBAgQIECAQNMVEIBuunNv5AQIECBAgAABAgQIECBAgAABAgQIENiuAgLQ25VX4wQIECBAgAABAgQIECBAgAABAgQIEGi6AgLQTXfujZwAAQIECBAgQIAAAQIECBAgQIAAAQLbVUAAervyapwAAQIECBAgQIAAAQIECBAgQIAAAQJNV0AAuunOvZETIECAAAECBAgQIECAAAECBAgQIEBguwoIQG9XXo0TIECAAAECBAgQIECAAAECBAgQIECg6QoIQDfduTdyAgQIECBAgAABAgQIECBAgAABAgQIbFcBAejtyqtxAgQIECBAgAABAgQIECBAgAABAgQINF2BkqY7dCMnQIAAAQIECBAgQIBA4xE4oOse4XOXdA8LlywPvYeOCStXlzWawZW2aB5aJj+xVFZVhdVryhvN2AyEAAECBAg0dgEB6MY+w8ZHgAABAgQIECCwUwX27rJrOP7wA8JhB3YNLZqVhMkz54Q33p0aJk7/IGQymZ3at8bw8JalLULb1i0/HEomLF62MlRXpxvD0LZqDM2Tz9Yt3/tKSKVS2fsO3m/P8MPf3r9VbRRq5TikO396fWjfplW2i+/PWxT++7Z7C7W7+kWAAAECBAjUERCArgPiLQECBAgQIECAAIH6EPjKleeGS848sSYgmGvziIP2DVeed0o2+PyvyTPDb/7eJ6TTAtE5n615veD048O1nz4vFBetyywY4/n/+9de4Z3Js7ammUZR9xNHH1Lrs7Z3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AAEEEHALEIB2e7CHAAIIIIAAAggggAACCCCAAAIIIIAAAgggECIBAtAhgmQYBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAbcAAWi3B3sIIIAAAggggAACCCCAAAIIIIAAAggggAACIRIgAB0iSIZBAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQcAsQgHZ7sIcAAggggAACCCCAAAIIIIAAAggggAACCCAQIgEC0CGCZBgEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABtwABaLcHewgggAACCCCAAAIIIIAAAggggAACCCCAAAIhEiAAHSJIhkEAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBwCxCAdnuwhwACCCCAAAIIIIAAAggggAACCCCAAAIIIBAiAQLQIYJkGAQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAG3AAFotwd7CCCAAAIIIIAAAggggAACCCCAAAIIIIAAAiESIAAdIkiGQQABBBBAAAEEEEAAAQQQQAABBBBAAAEEEHALEIB2e7CHAAIIIIAAAggggAACCCCAAAIIIIAAAgggECIBAtAhgmQYBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAbcAAWi3B3sIIIAAAggggAACCCCAAAIIIIAAAggggAACIRIgAB0iSIZBAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQcAsQgHZ7sIcAAggggAACCCCAAAIIIIAAAggggAACCCAQIgEC0CGCZBgEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABtwABaLcHewgggAACCCCAAAIIIIAAAggggAACCCCAAAIhEiAAHSJIhkEAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBwCxCAdnuwhwACCCCAAAIIIIAAAggggAACCCCAAAIIIBAiAQLQIYJkGAQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAG3AAFotwd7CCCAAAIIIIAAAggggAACCCCAAAIIIIAAAiESIAAdIkiGQQABBBBAAAEEEEAAAQQQQAABBBBAAAEEEHALEIB2e7CHAAIIIIAAAggggAACCCCAAAIIIIAAAgggECIBAtAhgmQYBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAbcAAWi3B3sIIIAAAggggAACCCCAAAIIIIAAAggggAACIRIgAB0iSIZBAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQcAsQgHZ7sIcAAggggAACCCCAAAIIIIAAAggggAACCCAQIgEC0CGCZBgEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABtwABaLcHewgggAACCCCAAAIIIIAAAggggAACCCCAAAIhEiAAHSJIhkEAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBwCxCAdnuwhwACCCCAAAIIIIAAAggggAACCCCAAAIIIBAiAQLQIYJkGAQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAG3AAFotwd7CCCAAAIIIIAAAggggAACCCCAAAIIIIAAAiESIAAdIkiGQQABBBBAAAEEEEAAAQQQQAABBBBAAAEEEHALEIB2e7CHAAIIIIAAAggggAACCCCAAAIIIIAAAgggECIBAtAhgmQYBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAbcAAWi3B3sIIIAAAggggAACCCCAAAIIIIAAAggggAACIRIgAB0iSIZBAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQcAsQgHZ7sIcAAggggAACCCCAAAIIIIAAAggggAACCCAQIgEC0CGCZBgEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABtwABaLcHewgggAACCCCAAAIIIIAAAggggAACCCCAAAIhEiAAHSJIhkEAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBwCxCAdnuwhwACCCCAAAIIIIAAAggggAACCCCAAAIIIBAiAQLQIYJkGAQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAG3AAFotwd7CCCAAAIIIIAAAggggAACCCCAAAIIIIAAAiESIAAdIkiGQQABBBBAAAEEEEAAAQQQQAABBBBAAAEEEHALEIB2e7CHAAIIIIAAAggggAACCCCAAAIIIIAAAgggECIBAtAhgmQYBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAbcAAWi3B3sIIIAAAggggAACCCCAAAIIIIAAAggggAACIRIgAB0iSIZBAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQcAsQgHZ7sIcAAggggAACCCCAAAIIIIAAAggggAACCCAQIgEC0CGCZBgEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABtwABaLcHewgggAACCCCAAAIIIIAAAggggAAC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)

### Cold-start gate
A user is **cold** if they have fewer than 2 training clicks. Cold users skip the two-stage pipeline entirely and fall back to global popularity ranking.



```python
# Cold start gate
eval_warm = eval_df
cold_in_eval = sum(is_cold(uid) for uid in eval_warm['userId'])
print(f'Cold users in eval fold: {cold_in_eval:,}  '
      f'({100*cold_in_eval/len(eval_warm):.1f}%)')
```

    Cold users in eval fold: 1,226  (13.7%)


[Back to top..](#top)

---

## <a id="sec-8"></a>8. Stage 1 — Expanded candidate pool

Stage 1 merges four retrievers to maximise recall before the expensive re-ranking step. We measure **Stage-1 Recall@200** on a diagnostic sample: what fraction of the user's ground-truth articles appear anywhere in the 200-candidate pool?



### 🎯 Stage 1 — Retriever Fusion Strategy

The four retrievers are complementary by design — each catches a different class of relevant articles:

```
                        USER QUERY
                            │
          ┌─────────────────┼─────────────────┐------------------|
          │                 │                 │                  |
          ▼                 ▼                 ▼                  ▼
   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
   │ S1 Popular  │  │ S2 Category │  │ S3 Item-CF  │  │ S4 Temporal │
   │             │  │             │  │             │  │             │
   │ Bayesian    │  │ user_cat ·  │  │ co-click    │  │ recency-    │
   │ CTR rank    │  │ article_cat │  │ neighbours  │  │ weighted    │
   │ (global)    │  │ dot product │  │ aggregation │  │ taste vec   │
   │             │  │             │  │             │  │             │
   │ Best for:   │  │ Best for:   │  │ Best for:   │  │ Best for:   │
   │ cold users  │  │ category    │  │ warm users  │  │ trend-      │
   │ new articles│  │ loyal users │  │ with many   │  │ sensitive   │
   │             │  │             │  │ clicks      │  │ users       │
   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
          │  N/4           │  N/4           │  N/2           │  N/4
          └────────────────┴────-───────────┴─-──────────────┘
                                    │
                           dict.fromkeys()  ← preserves order, deduplicates
                                    │
                            ┌───────▼────────┐
                            │  200 candidates│
                            │  (Recall@200   │
                            │   diagnostic)  │
                            └───────┬────────┘
                                    │
                            STAGE 2 RERANKING
```

**Budget split:** S3 (Item-CF) gets half the budget because it produces the most personalised candidates for warm users. S1–S2–S4 each get a quarter. For cold users, the gate bypasses S2–S4 entirely and returns pure popularity.

**Why `dict.fromkeys()` for deduplication?** It preserves insertion order (unlike `set()`), so the highest-priority retriever's candidates remain first when the total pool is truncated to 200.



```python
# Generate candidates in stage 1
N_STAGE1 = 200

def stage1_candidates(uid):

    if is_cold(uid):

        return _filter_seen(POPULARITY_POOL, uid)[:N_STAGE1]

    pool = list(dict.fromkeys(
        s1_popularity(uid, N_STAGE1//4) +
        s2_category(uid,   N_STAGE1//4) +
        s3_itemcf(uid,     N_STAGE1//2) +
        s4_temporal(uid,   N_STAGE1//4)
    ))
    return pool[:N_STAGE1]

# Recall diagnostic
DIAG_N = 500
diag_users = eval_warm.sample(n = min(DIAG_N, len(eval_warm)), random_state = 100)
recalls = []

for _, row in diag_users.iterrows():

    pool = set(stage1_candidates(row['userId']))
    true = row['true_items']
    recalls.append(len(pool & true) / len(true) if true else 0.0)

print(f'Stage-1 Recall@{N_STAGE1} (n={DIAG_N}): {np.mean(recalls):.4f}')
print(f'  Min: {np.min(recalls):.4f}  Max: {np.max(recalls):.4f}  Std: {np.std(recalls):.4f}')
```

    Stage-1 Recall@200 (n=500): 0.0470
      Min: 0.0000  Max: 1.0000  Std: 0.1945


[back to top..](#top)

---

## <a id="sec-9"></a>9. Stage 2 — Meta-ranker training

The meta-ranker sees **enriched features** beyond what the base LightGBM sees:

| Feature group | Features |
|---------------|----------|
| Base ranker features | All 9 features from Section 5 |
| Retriever membership | `in_s2`, `in_s3`, `in_s4` (binary flags) |
| Retriever ranks | `rank_s2`, `rank_s3`, `rank_s4` (position in each retriever's list) |
| Ensemble depth | `n_retrievers` (how many retrievers surfaced this candidate) |
| Base LGB score | `s5_score` (predicted probability from the base model) |

This lets the meta-ranker learn which retrievers are reliable for which users and articles.



```python
STAGE2_FEATURE_COLS = FEATURE_COLS + ['in_s2','in_s3','in_s4','rank_s2','rank_s3','rank_s4','n_retrievers','s5_score']
print(f'Stage-2 features: {len(STAGE2_FEATURE_COLS)}')
```

    Stage-2 features: 21



```python
%%time

# Meta-ranker training data uses SET_B users (OOF).
# The base LGB was trained only on SET_A; scoring SET_B gives
# true out-of-fold predictions — no in-sample leakage.

CHUNK_SIZE = 500

user_gt_clicks = train_clicks.groupby('userId')['newsId'].apply(set).to_dict()

# Sample up to 5000 SET_B users
rng_meta      = np.random.default_rng(100)
set_b_pool    = np.array(list(SET_B_users & set(user_stats.index)))
sample_meta   = rng_meta.choice(set_b_pool, size = min(5000, len(set_b_pool)), replace = False)

print(f'Meta-ranker training users (SET_B OOF): {len(sample_meta):,}')

# Stage-1 candidate generation without seen-filter (positives must stay in pool)
print('Compiling stage 1 candidates..', end = ' ', flush = True)

_orig_seen = dict(_seen_cache)
_seen_cache.clear()

meta_pair_rows = []

for uid in sample_meta:

    candidates = stage1_candidates(uid)
    gt = user_gt_clicks.get(uid, set())

    for nid in candidates:

        meta_pair_rows.append((uid, str(nid), int(str(nid) in gt)))

_seen_cache.update(_orig_seen)
del _orig_seen

meta_df = pd.DataFrame(meta_pair_rows, columns=['userId', 'newsId', 'label'])
del meta_pair_rows; gc.collect()
print('done.')

n_pos_raw = int(meta_df['label'].sum())
print(f'Stage-1 pairs: {len(meta_df):,}  pos={n_pos_raw:,}  neg={len(meta_df)-n_pos_raw:,}')

if n_pos_raw == 0:

    raise RuntimeError('No positives found in SET_B meta-ranker pairs. '
                       'Check SET_B_users and stage1_candidates().')

# Merge fts
meta_df = meta_df.join(user_stats[['click_count','click_freq']].rename(columns = {'click_count':'u_click_count','click_freq':'u_click_freq'}), on = 'userId')
meta_df = meta_df.join(article_feat[['log_clicks','log_impr','bayesian_ctr','article_len','article_age_days']].rename(columns = {'log_clicks':'m_log_clicks','log_impr':'m_log_impr', 'bayesian_ctr':'m_bayesian_ctr','article_len':'m_article_len', 'article_age_days':'article_age_days'}), on = 'newsId')

# Add category ft
newsid_to_cat        = news.set_index('newsId')['category'].to_dict()
meta_df['category']  = meta_df['newsId'].map(newsid_to_cat)
relevant_users_meta  = meta_df['userId'].unique()

uca_long = (user_cat_affinity.reindex(index = relevant_users_meta).stack().reset_index().rename(columns = {'level_0':'userId','level_1':'category',0:'cat_affinity'}))
meta_df  = meta_df.merge(uca_long, on=['userId','category'], how='left')
del uca_long; gc.collect()

uta_long = (user_taste_norm.reindex(index = relevant_users_meta).stack().reset_index().rename(columns = {'level_0':'userId','level_1':'category',0:'taste_affinity'}))
meta_df  = meta_df.merge(uta_long, on = ['userId','category'], how='left')
del uta_long; gc.collect()

# TF-IDF affinities for meta pairs
print('Computing TF-IDF affinities for meta-ranker pairs...', end = ' ', flush = True)

uid_nid_sim_meta = {}

for uid, grp in meta_df.groupby('userId'):

    centroid = user_tfidf_centroids.get(uid)

    if centroid is None:

        continue

    valid = [(nid, tfidf_idx[nid]) for nid in grp['newsId'].unique() if nid in tfidf_idx]

    if not valid:

        continue

    v_nids, v_idxs = zip(*valid)
    sims = np.asarray(tfidf_mat[list(v_idxs)].dot(centroid)).ravel()

    for nid, sim in zip(v_nids, sims):

        uid_nid_sim_meta[(uid, nid)] = float(sim)

meta_df['tfidf_sim'] = [uid_nid_sim_meta.get((r.userId, r.newsId), 0.0) for r in meta_df.itertuples()]
del uid_nid_sim_meta; gc.collect()
print('done.')

# recent_tfidf_sim for meta-ranker pairs
print('Computing recent TF-IDF affinities for meta pairs...', end = ' ', flush = True)

uid_nid_recent_meta = {}

for uid, grp in meta_df.groupby('userId'):

    centroid = user_recent_tfidf_centroids.get(uid)

    if centroid is None:

        continue

    valid = [(nid, tfidf_idx[nid]) for nid in grp['newsId'].unique() if nid in tfidf_idx]

    if not valid:

        continue

    v_nids, v_idxs = zip(*valid)
    sims = np.asarray(tfidf_mat[list(v_idxs)].dot(centroid)).ravel()

    for nid, sim in zip(v_nids, sims):

        uid_nid_recent_meta[(uid, nid)] = float(sim)

meta_df['recent_tfidf_sim'] = [uid_nid_recent_meta.get((r.userId, r.newsId), 0.0) for r in meta_df.itertuples()]
del uid_nid_recent_meta; gc.collect()
print('done.')

# subcat_clicks for meta-ranker pairs
meta_df['_subcat'] = meta_df['newsId'].map(newsid_to_subcat)
_subcat_lkp_meta = pd.DataFrame([(u, sc, cnt) for (u, sc), cnt in user_subcat_clicks.items()], columns = ['userId', '_subcat', 'subcat_clicks'])
meta_df = meta_df.merge(_subcat_lkp_meta, on=['userId', '_subcat'], how='left')
meta_df['subcat_clicks'] = meta_df['subcat_clicks'].fillna(0).astype('float32')
meta_df.drop(columns=['_subcat'], inplace=True)
del _subcat_lkp_meta

base_feature_cols = [c for c in FEATURE_COLS if c not in ('ctr_norm_rank', 'imp_size')]
meta_df[base_feature_cols] = meta_df[base_feature_cols].fillna(0)

meta_df['imp_size'] = (meta_df.groupby('userId')['newsId'].transform('count').astype('float32'))
meta_df['ctr_norm_rank'] = (meta_df.groupby('userId')['m_bayesian_ctr'].transform(lambda x: (x.rank(ascending=False, method='average') - 1).div(max(1, len(x) - 1))).astype('float32'))

# Get fts from the other retrievers
unique_users = np.array(meta_df['userId'].unique())
n_users           = len(unique_users)

print(f'Building retriever membership for {n_users:,} users...', end = ' ', flush = True)

article_cat_idx_arr   = np.array(article_cat_idx)
taste_article_idx_arr = np.array(taste_article_idx)

_uca   = user_cat_affinity.reindex(unique_users).fillna(0).values.astype('float32')
_uca_n = _uca / (np.linalg.norm(_uca, axis = 1, keepdims = True).clip(min = 1e-9))
s2_top = chunked_topn(article_cat_norm, _uca_n, article_cat_idx_arr, N_STAGE1, 'rank_s2')

del _uca, _uca_n

_taste   = user_taste_norm.reindex(unique_users).fillna(0).values.astype('float32')
s4_top   = chunked_topn(article_cat_taste_norm, _taste, taste_article_idx_arr, N_STAGE1, 'rank_s4')

del _taste; gc.collect()

# Collaborative filte ranking
s3_rows = []

for uid in unique_users:

    for rank, nid in enumerate(s3_itemcf(uid, N_STAGE1)):

        s3_rows.append((uid, str(nid), rank))

s3_top = pd.DataFrame(s3_rows, columns=['userId','newsId','rank_s3'])

del s3_rows

s2_top['newsId'] = s2_top['newsId'].astype(str)
s4_top['newsId'] = s4_top['newsId'].astype(str)
s3_top['newsId'] = s3_top['newsId'].astype(str)

# Merge fts
meta_df = meta_df.merge(s2_top[['userId','newsId','rank_s2']], on = ['userId','newsId'], how = 'left')
meta_df = meta_df.merge(s3_top[['userId','newsId','rank_s3']], on = ['userId','newsId'], how = 'left')
meta_df = meta_df.merge(s4_top[['userId','newsId','rank_s4']], on = ['userId','newsId'], how = 'left')

del s2_top, s3_top, s4_top; gc.collect()
print('done.')

# Compile flags
meta_df['in_s2'] = meta_df['rank_s2'].notna().astype(int)
meta_df['in_s3'] = meta_df['rank_s3'].notna().astype(int)
meta_df['in_s4'] = meta_df['rank_s4'].notna().astype(int)
meta_df[['rank_s2','rank_s3','rank_s4']] = meta_df[['rank_s2','rank_s3','rank_s4']].fillna(N_STAGE1)
meta_df['n_retrievers'] = meta_df[['in_s2','in_s3','in_s4']].sum(axis = 1)

meta_train_df = meta_df.copy()
del meta_df; gc.collect()
print(f'meta_train_df: {meta_train_df.shape}')
```

    Meta-ranker training users (SET_B OOF): 5,000
    Compiling stage 1 candidates.. done.
    Stage-1 pairs: 917,992  pos=10,098  neg=907,894
    Computing TF-IDF affinities for meta-ranker pairs... done.
    Computing recent TF-IDF affinities for meta pairs... done.
    Building retriever membership for 5,000 users... done.
    meta_train_df: (951509, 25)
    CPU times: user 14min 44s, sys: 1.12 s, total: 14min 45s
    Wall time: 2min 11s



```python
meta_train_df.head()
```





  <div id="df-0239e2b0-2f2f-463e-8adb-b70f6b83f950" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>newsId</th>
      <th>label</th>
      <th>u_click_count</th>
      <th>u_click_freq</th>
      <th>m_log_clicks</th>
      <th>m_log_impr</th>
      <th>m_bayesian_ctr</th>
      <th>m_article_len</th>
      <th>article_age_days</th>
      <th>category</th>
      <th>cat_affinity</th>
      <th>taste_affinity</th>
      <th>tfidf_sim</th>
      <th>recent_tfidf_sim</th>
      <th>subcat_clicks</th>
      <th>imp_size</th>
      <th>ctr_norm_rank</th>
      <th>rank_s2</th>
      <th>rank_s3</th>
      <th>rank_s4</th>
      <th>in_s2</th>
      <th>in_s3</th>
      <th>in_s4</th>
      <th>n_retrievers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>U19087</td>
      <td>N49279</td>
      <td>0</td>
      <td>1</td>
      <td>1.0000</td>
      <td>7.7280</td>
      <td>8.7371</td>
      <td>0.3618</td>
      <td>126.0000</td>
      <td>1.7416</td>
      <td>music</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0121</td>
      <td>0.0121</td>
      <td>0.0000</td>
      <td>200.0000</td>
      <td>0.0000</td>
      <td>200.0000</td>
      <td>30.0000</td>
      <td>200.0000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>U19087</td>
      <td>N49685</td>
      <td>0</td>
      <td>1</td>
      <td>1.0000</td>
      <td>7.7385</td>
      <td>8.8860</td>
      <td>0.3154</td>
      <td>187.0000</td>
      <td>1.7186</td>
      <td>music</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0289</td>
      <td>0.0289</td>
      <td>0.0000</td>
      <td>200.0000</td>
      <td>0.0050</td>
      <td>200.0000</td>
      <td>31.0000</td>
      <td>200.0000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>U19087</td>
      <td>N60750</td>
      <td>0</td>
      <td>1</td>
      <td>1.0000</td>
      <td>4.8363</td>
      <td>5.8693</td>
      <td>0.3152</td>
      <td>303.0000</td>
      <td>0.2710</td>
      <td>sports</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0058</td>
      <td>0.0058</td>
      <td>0.0000</td>
      <td>200.0000</td>
      <td>0.0101</td>
      <td>200.0000</td>
      <td>32.0000</td>
      <td>200.0000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>U19087</td>
      <td>N53585</td>
      <td>0</td>
      <td>1</td>
      <td>1.0000</td>
      <td>7.9502</td>
      <td>9.2012</td>
      <td>0.2849</td>
      <td>132.0000</td>
      <td>1.5704</td>
      <td>tv</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>200.0000</td>
      <td>0.0151</td>
      <td>200.0000</td>
      <td>33.0000</td>
      <td>200.0000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>U19087</td>
      <td>N25791</td>
      <td>0</td>
      <td>1</td>
      <td>1.0000</td>
      <td>5.0938</td>
      <td>6.4489</td>
      <td>0.2409</td>
      <td>175.0000</td>
      <td>1.3492</td>
      <td>news</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>0.0164</td>
      <td>0.0164</td>
      <td>0.0000</td>
      <td>200.0000</td>
      <td>0.0201</td>
      <td>200.0000</td>
      <td>34.0000</td>
      <td>200.0000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-0239e2b0-2f2f-463e-8adb-b70f6b83f950')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-0239e2b0-2f2f-463e-8adb-b70f6b83f950 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-0239e2b0-2f2f-463e-8adb-b70f6b83f950');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>





```python
%%time

# s5_score on SET_B users: base LGB has not seen these users during training (trained on SET_A only), so the meta-ranker learns from true OOF scores.
xmeta_base = meta_train_df[FEATURE_COLS].values.astype('float32')
meta_train_df['s5_score'] = lgb_model.predict(xmeta_base)
del xmeta_base; gc.collect()

xmeta = meta_train_df[STAGE2_FEATURE_COLS].values
ymeta = meta_train_df['label'].values

# Split for training
xm_tr, xm_val, ym_tr, ym_val = train_test_split(xmeta, ymeta, test_size = 0.15, random_state = 100, stratify = ymeta)

meta_lgb_params = {'objective'        : 'binary',
                    'metric'           : 'auc',
                    'learning_rate'    : 0.03,
                    'feature_fraction' : 0.8,
                    'bagging_fraction' : 0.8,
                    'bagging_freq'     : 5,
                    'verbose'          : -1,
                    'n_jobs'           : -1,}

meta_lgb = lgb.train(meta_lgb_params, lgb.Dataset(xm_tr, label=ym_tr), num_boost_round = 800, valid_sets      = [lgb.Dataset(xm_val, label=ym_val)], callbacks       = [lgb.early_stopping(40, verbose=False), lgb.log_evaluation(100)],)

xgb_meta = XGBClassifier(n_estimators          = 1000,
                        learning_rate         = 0.05,
                        max_depth             = 6,
                        subsample             = 0.8,
                        colsample_bytree      = 0.8,
                        eval_metric           = 'auc',
                        early_stopping_rounds = 30,
                        verbosity             = 0,)

xgb_meta.fit(xm_tr, ym_tr, eval_set=[(xm_val, ym_val)], verbose=False)

print(f'Meta-LGB trees : {meta_lgb.num_trees()}')
print(f'Meta-XGB trees : {xgb_meta.best_iteration}')
print(f'STAGE2_FEATURE_COLS ({len(STAGE2_FEATURE_COLS)}): {STAGE2_FEATURE_COLS}')
```

    [100]	valid_0's auc: 1
    Meta-LGB trees : 72
    Meta-XGB trees : 36
    STAGE2_FEATURE_COLS (21): ['u_click_count', 'u_click_freq', 'm_log_clicks', 'm_log_impr', 'm_article_len', 'cat_affinity', 'taste_affinity', 'tfidf_sim', 'recent_tfidf_sim', 'article_age_days', 'ctr_norm_rank', 'imp_size', 'subcat_clicks', 'in_s2', 'in_s3', 'in_s4', 'rank_s2', 'rank_s3', 'rank_s4', 'n_retrievers', 's5_score']
    CPU times: user 45.7 s, sys: 141 ms, total: 45.8 s
    Wall time: 7.04 s



```python
del xmeta, xm_tr, xm_val, ym_tr, ym_val, meta_train_df; gc.collect()
```




    14



[Back to top](#top)

---

## <a id="sec-10"></a>10. Full benchmark: S1 → S7

We evaluate all seven strategies on the held-out eval fold. Each strategy is given the same `eval_warm` users and the same ground-truth sets.



```python
# Run the benchmark to compare all strategies
%%time

strategies = [('S1: Popularity',         s1_score),
              ('S2: Category Affinity',  s2_score),
              ('S3: Item-CF',            s3_score),
              ('S4: Temporal Taste',     s4_score),
              ('S5: LightGBM Base',      s5_score),
              ('S6: Meta-LGB (2-Stage)', s6_score),
              ('S7: Ensemble (LGB + XGB)', s7_score),]

all_results = {}
EVAL_N = min(1000, len(eval_warm))

for name, fn in strategies:

    for K in [5, 10]:

        print(f'  {name}  @K={K}...', end = ' ', flush = True)
        t0 = time.time()
        res = evaluate_strategy(fn, eval_warm, K=K, n=EVAL_N)
        print(f'{time.time()-t0:.0f}s  composite={res["composite"]:.4f}')
        all_results[(name, K)] = res
```

      S1: Popularity  @K=5... 0s  composite=0.3385
      S1: Popularity  @K=10... 0s  composite=0.4794
      S2: Category Affinity  @K=5... 0s  composite=0.2964
      S2: Category Affinity  @K=10... 0s  composite=0.4295
      S3: Item-CF  @K=5... 0s  composite=0.3049
      S3: Item-CF  @K=10... 0s  composite=0.4344
      S4: Temporal Taste  @K=5... 0s  composite=0.2964
      S4: Temporal Taste  @K=10... 0s  composite=0.4295
      S5: LightGBM Base  @K=5... 21s  composite=0.3942
      S5: LightGBM Base  @K=10... 20s  composite=0.5078
      S6: Meta-LGB (2-Stage)  @K=5... 61s  composite=0.3138
      S6: Meta-LGB (2-Stage)  @K=10... 60s  composite=0.4477
      S7: Ensemble (LGB + XGB)  @K=5... 97s  composite=0.3137
      S7: Ensemble (LGB + XGB)  @K=10... 96s  composite=0.4470
    CPU times: user 46min 7s, sys: 1.8 s, total: 46min 9s
    Wall time: 5min 56s



```python
# Compile the leaderboard
records = []

for (name, K), res in all_results.items():

    records.append({'strategy': name, 'K': K, **res})

leaderboard = (pd.DataFrame(records).sort_values(['K','composite'], ascending = [True, False]).reset_index(drop = True))

for k_val in [5, 10]:

    print(f'\n{"="*65}')
    print(f'  LEADERBOARD  @  K = {k_val}')
    print('='*65)
    lb = leaderboard[leaderboard['K'] == k_val][['strategy'] + metric_keys + ['composite']]
    print(lb.to_string(index=False))
```

    
    =================================================================
      LEADERBOARD  @  K = 5
    =================================================================
                    strategy  precision  recall     f1   ndcg  hit_rate  composite
           S5: LightGBM Base     0.1070  0.4262 0.1647 0.2934    0.4950     0.3942
              S1: Popularity     0.0908  0.3658 0.1407 0.2540    0.4230     0.3385
      S6: Meta-LGB (2-Stage)     0.0836  0.3376 0.1292 0.2335    0.3940     0.3138
    S7: Ensemble (LGB + XGB)     0.0836  0.3376 0.1292 0.2335    0.3940     0.3137
                 S3: Item-CF     0.0806  0.3317 0.1255 0.2258    0.3840     0.3049
       S2: Category Affinity     0.0780  0.3236 0.1219 0.2197    0.3730     0.2964
          S4: Temporal Taste     0.0780  0.3236 0.1219 0.2197    0.3730     0.2964
    
    =================================================================
      LEADERBOARD  @  K = 10
    =================================================================
                    strategy  precision  recall     f1   ndcg  hit_rate  composite
           S5: LightGBM Base     0.0782  0.5887 0.1338 0.3506    0.6650     0.5078
              S1: Popularity     0.0719  0.5579 0.1239 0.3209    0.6380     0.4794
      S6: Meta-LGB (2-Stage)     0.0671  0.5247 0.1158 0.2983    0.5970     0.4477
    S7: Ensemble (LGB + XGB)     0.0670  0.5242 0.1156 0.2981    0.5960     0.4470
                 S3: Item-CF     0.0639  0.5114 0.1110 0.2878    0.5810     0.4344
       S2: Category Affinity     0.0632  0.5059 0.1098 0.2830    0.5760     0.4295
          S4: Temporal Taste     0.0632  0.5059 0.1098 0.2830    0.5760     0.4295



### 📊 How to Read the Leaderboard

Before the visualisations, here's the analytical lens to apply:

| What to look for | What it means |
|-----------------|---------------|
| **Gap between S1 and S2–S4** | Size of *personalisation lift* — how much history helps vs. pure popularity |
| **S5 vs S2–S4** | Value added by learning feature interactions (LambdaMART) over hand-crafted dot-products |
| **S6 vs S5** | Value of two-stage architecture: does meta-learning on OOF scores help? |
| **S7 vs S6** | Value of model ensembling (LGB + XGB diversity) |
| **K=5 vs K=10 patterns** | If gains are larger at K=5, the model is especially good at surfacing the *single best* article — valuable for mobile one-article layouts |
| **NDCG vs HR gap** | Large HR with low NDCG means the model finds *some* relevant article in top-K but ranks it poorly; focus tuning on the ranking objective |

> **General expectation:** S1 < S2 ≈ S3 ≈ S4 < S5 < S6 ≤ S7. Deviations from this ordering reveal where personalisation is breaking down (e.g. if S3 < S1, the CF graph is too sparse to be useful at this sample size).

[Back to top](#top)

---

## <a id="sec-11"></a>11. Benchmark visualisations



```python
# Visualize the composite scores comparison
fig, ax = plt.subplots(figsize = (20, 5))
lb10 = leaderboard[leaderboard['K'] == 5].sort_values('composite')
palette = sns.color_palette('husl', len(lb10))
bars = ax.barh(lb10['strategy'], lb10['composite']*100, color=palette)
ax.set_xlabel('Composite score (%) — mean of P@10, R@10, F1@10, NDCG@10, HR@10')
ax.set_title('News Recommendation Benchmark  |  Composite @ K=5')

for bar, val in zip(bars, lb10['composite']):

    ax.text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2, f'{val*100:.2f}%', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('benchmark_composite.png', dpi=150, bbox_inches='tight')
plt.show()
```


    
![png](data:image/png;base64,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a8qUKcqZM6eaN2/+1BqyZcumLFmyPHZa8sd9CePRkd/J1/Bo8J+QkKBZs2bp9ddfl42NjZKSkrRx40az5Qsex87OzniPhgwZohMnTqh3795avXq1ChcuLOnhGuO+vr5P7Ofnn39+7L6yZctKejjFPgAAAGAJrJ/eBAAAAACQUd566y1Vr15dY8aMMQtUSpQooWzZsikyMlLFihUzfgoXLqwHDx4YIWLlypVlZWWlFStWKC4uzpgK29PTUwcPHjSmWZekmzdvav369TKZTJIerrXbokUL+fn5pZheOi169uwpJycnjRgxwqjdwcFBZcqU0dmzZ83qLlasmOLj45UrVy5JUpkyZRQeHm4WDoeFhalVq1a6cOGCsZ7w4cOHzc555MgRWVtbG2ukv0guLi6ysbHR5cuXza4rT548sra2fuqI+0eVLFlSx48fN94L6eH04PHx8cbr5JHEjwbGR48e1blz51KMun3S9Nhubm6KjIxMEX4dOXJEJUuWTHUk7PNy5swZY3r053n/nvb5SU+pvS8XL17UgQMH0jxNec2aNVWyZEkFBwdr//79mjdvXrrU+neBgYEymUwaN25cqvunT5+uzz77TKdPn5ajo6NKlSplzEiR7Pr164qMjHym6fyf5ODBg2avT5w4YXwRxM3NTffv30+xLvyRI0fk6Oio4sWL69y5c9q5c6exr0iRIurbt69Kly6t33777bHnffS9cnd317179xQXF2f22cySJYuyZcumzJkzp9pHkSJFFBMToz///FOOjo6qUKGCli1bpps3b+rMmTPq3bu3ypYtq7i4ON26dUvjxo1TUlKS/P39n+keZcqUyViHvnv37saXMJLXGH/Sj/Tw92j//v118+ZNs37Dw8MlyfhiEwAAAPBfRzAOAAAAAC+5jz/+WL/++qvCwsKMbba2tvrggw+0bNkyffnllzp37pxOnjypoUOHqkWLFkbIaWdnJ09PTy1atEgVKlQwAhxPT08dOnRIv/76q2rWrCnpYRAUFBSk4cOH67ffftOVK1e0d+9e7dy5M8Xaumnh4OCgYcOG6eDBg1q1apWxvUuXLtqxY4dmzJihiIgInT59WhMmTFCTJk2MdYmTA7pBgwbp7NmzOnbsmEaNGqX4+HgVKVJEbm5uqlatmsaPH68ffvhBkZGRWr9+vebMmaOAgIAU62W/CLlz51bz5s0V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d+/eFyoY//3333Xq1CnVrl1b//vf/9Ktj4uLU+nSpY2v8+XL99T/g7dmzRqCcQAAAAAAAACPLSUlRePGjVOtWrUk3Zup0sbGRqtWrVKRIkWM5S1bttTdu3d14sSJdMdYvXq13nnnHeXPn185cuTQ1KlTVbduXWP99evXNXXqVA0aNMhYdvXqVb3yyivKkyeP7O3tdfbsWUnSwYMHde7cObVt2/Z5NhsAAOglCsZ3796tli1bqkKFCrK0tJSNjY18fX0VFBQkKysrSZKNjY2GDx+uYsWKKXfu3OrRo4f+/PNPXb58WZIUGBio4cOHZ3j8mzdvavz48Ro2bJg8PT2VK1cuWVtbq27dupoxY4ZsbGyM6XTWr1+vFi1ayNnZWQ0bNtTKlSslSd9++62GDh2qI0eOyN7eXufPn1dqaqpmzJihxo0by9HRUe3atdPBgweN8547d06enp5ycHBQly5dtHHjRlWoUMFYf+rUKXXv3l3Vq1eXi4uLRo8erbt370qSQkND1apVK02cOFFOTk763//+J1dXV6WkpBj7//XXX6pYsaLxP2p/FxISogYNGqhVq1YKDQ01licmJsre3l6S5OfnpwoVKigwMFBXr16Vvb299u/fr4YNG+qrr76SJI0YMUJjx47VhAkT9NZbb6lmzZpasGCBcbwKFSpo165dGjt2rFauXKnFixerSZMmGjlypAYOHGhS09q1a+Xm5pbpUfoAAAAAAAAAXg758+dX06ZNjdc///yz7ty5oxIlSpj8XVSScuXKpT///DPdMY4fP66YmBh5e3urWbNmmjx5spKSkoz148ePV48ePfTaa68Zy7Jly2aMPk9LS5OFhYWSkpI0btw4DRo0SIMHD9Y777yjPXv2POsmAwCA/++lCcbLlCmj//3vfybT10hS69atVbRoUUmSv7+/atasaay7ePGirK2tVaBAAUlSUFCQJk2alOHxw8PDlZycnOHU4A4ODho1apSsrKx0/vx5vf/++woMDNShQ4c0btw4jR07Vr/99puaN28uX19fOTg4KDIyUiVLltTSpUu1adMmLVy4UPv375eHh4d8fX0VHx8vSRowYIBKlSpljE6fPn26cd7ExET5+PjI0dFR4eHhWr16tfbv32+yTXR0tKytrbV//341a9ZMCQkJJs/y/u6771S1alWVKVMmXbsSExO1bt06tW7dWo0b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tXTpUpOfBatXr1b//v3VokUL7dixQwcPHtTMmTN16tQpdejQId0ND//F65ZZ969bZGSkDhw4oPfff1+TJ0/Wxo0bTbYbP368YmNjFRISopYtW6po0aKysLBQoUKF1KVLFy1YsECBgYGKj4839vn444+1fPlyFStWLN15w8LCNHPmTE2aNEkRERFq0KCB+vXrZ7I/AAAA8CwRjAMAAADAI3zyySeqU6eOAgICVLhwYVlaWqpSpUqaO3eunJycjMDl0qVL8vX1lYuLi6pVqyZ/f39jpOK+ffv05ptvavv27WrYsKGcnZ01bdo0RUZGqnXr1nJ2dtaAAQOUlJQkSerWrZs+++wzDRkyRE5OTnJzc9P3339v1HTjxg0NHz5cderUkbOzs/r06aOoqChJ90YZTpw4UXXq1JGTk5Nat25tBEqhoaGqXbu2JOmtt97SrVu31KZNG82aNUvSvdGRHTt2lLOzs+rWrWsSIv/dL7/8og4dOsjZ2VkuLi4mI1sTEhI0atQoubi4qGbNmho1apQSExMlSXfv3lVQUJDq168vR0dHdenSRcePHzeOW6FCBX3xxReqU6eO5s+f/9h1HTlyRMeOHVP79u0l3QtRHR0dlS9fPtWuXdtk1O/YsWM1cOBA2dramhzD2tpabdq00apVqx56nmfhcd8XqampmjFjhho3bixHR0e1a9dOBw8eNI537tw59erVSy4uLnJxcdHQoUONIPf+yNDdu3fLw8NDTk5O8vb2Nt43GVm1apWaN28uR0dHvf3229q8ebOkezcezJkzR0eOHJG9vb0uXLjwj23NlSuXfHx8VKRIEe3atctYPmbMGPXr108uLi46cOCAWrVqJWdnZw0cOFArVqzQ0KFDNXz4cIWEhBj7xMTE6LPPPlOHDh3SnScyMlK//fabhg0bprx588rOzk49e/bU119//Y81Pky+fPk0YsQI3bp1Sz///LOxfMqUKXr77bfVsmVLnT59Wl5eXnJyclL37t21efNmdejQQX5+fgoLCzPe/9evX1dAQIAGDhyY7jxXr17Vtm3b5O/vL1tbWxUtWlR+fn5as2aN8R54WtbW1goJCXloML5x40bFxMToww8/1O3btzV48GA5Ozvr7bff1r59++To6Kg333xTr776qsl7z8nJSfPnz8/wRomvv/5aPXv2VLly5WRjYyN/f3+dPn1av/zyS6br9vf3V1BQkHEd/0mePHnk5OSkuXPnqmjRogoODpZ072aG8ePHa9iwYfL09FSuXLlkbW2tunXrasaMGbKxsVFiYqKx3fDhw+Xp6ancuXMre/bsqlKliubOnav27dubhLf/1ev2JLJnz6569eqpRYsWJp87hw4d0o4dOzRp0iRZWFhozJgxqlatmho0aKDt27erQYMGypEjh9zc3PTdd98Z+7322mtauXJlup+z99t4/5EMOXPmVO/evSVJP/zww3NtIwAAAF5eBOMAAAAA8BDXrl3ToUOH1LVr13TrbGxsNGHCBJUqVUqS5Ofnp7x582r79u3aunWroqOjNXr0aGP7hIQE7dmzR5s2bdLo0aP1+eefa86cOfriiy8UGhqqnTt3mkxzvmrVKnl4eOinn37Su+++K39/f2Pa88DAQF25ckXr16/Xjz/+qJw5c2rIkCGSpE2bNikiIkLr16/XwYMH1aNHD73//vvpgrV169YZ/x0wYIAuXbokPz8/derUSQcOHNDChQu1atUqbdiwIcNrM3z4cHl5eengwYPasGGDTpw4YYSPn332mX7//Xd9++232rx5s44ePWqE2VOnTtX+/fu1fPly7du3T5UrV1bfvn1NAq9t27Zp7dq1evfddx+7rj179qhChQpGCGNhYWFMj5yWliYLCwtJ0tatWxUfH6/k5GR5eXmpZ8+e+vPPP43juLi46NChQ5kO4p7U47wvli5dqk2bNmnhwoXav3+/PDw85OvrawR0gYGBKlKkiH788Ud9++23Onv2rObMmWNyvi+//FLz5s3Tjh07FB8fr4ULF2ZYV1hYmCZPnqyxY8fqwIEDGjRokAICAnTixAkNGTJEvr6+xsjq4sWLZ7q9KSkpxmjp8+fP6+TJk+rUqZNOnDghX19f+fv7a//+/WrUqJE+++wzOTo6qkiRIrKysjJC/lq1aunNN9/M8PhHjx5V8eLFlT9/fmNZlSpVdPbs2XQjfB9HamqqyUjd27dva8uWLfL19dXVq1fVuXNn40aFvn376pNPPpGjo6Osra1VuXJlY0rrihUrqnHjxhme4/jx48Zo8wdrj4+P15kzZ5649gd1795defPmfej6r776SgMGDJCFhYXeffdd2djYKCIiQvPnz9fEiRNlZ2en3Llzq1atWiZTXvv5+Rl960F37tzR77//rsqVKxvLbGxsVLp0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
)
    



```python
# Visualize the per metric breakdown at 5
lb10   = leaderboard[leaderboard['K'] == 5].set_index('strategy')[metric_keys]
fig, ax = plt.subplots(figsize=(20, 6))
x     = np.arange(len(lb10))
width = 0.15
colors = sns.color_palette('husl', len(metric_keys))

for i, (metric, col) in enumerate(zip(metric_keys, colors)):

    ax.bar(x + i*width, lb10[metric]*100, width, label=metric, color=col, alpha=0.85)

ax.set_xticks(x + width*2)
ax.set_xticklabels(lb10.index, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('Score (%)')
ax.set_title('Per-metric breakdown @ K=10')
ax.legend(loc='upper left', ncol=5)
plt.tight_layout()
plt.savefig('benchmark_per_metric.png', dpi=150, bbox_inches='tight')
plt.show()
```


    
![png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAB8YAAAJOCAYAAADF3G1CAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAt2NJREFUeJzs3Xm8l3P+P/7HqZRok12WhBqUyj6mZKxDSFkztmEwtvDJvjPZCZN9+zRjGzIkJss04ROyi2SnSGNJi7TX6fz+6HfO15mUOk69T+/u99vN7eZ9Xdf7dT3f1/Xu2ek8rut1lZSVlZUFAAAAAAAAAIpUrUIXAAAAAAAAAACLk2AcAAAAAAAAgKImGAcAAAAAAACgqAnGAQAAAAAAAChqgnEAAAAAAAAAippgHAAAAAAAAICiJhgHAAAAAAAAoKgJxgEAAAAAAAAoaoJxAAAAAAAAAIqaYBwAAGAZd9ZZZ6VVq1aV/mvdunV+97vf5aabbsrMmTMLXeJi16pVq1xzzTVVfv8rr7ySVq1a5f/+7/+qsapFd+ihh+aAAw5Y5PcdcMABOfTQQxdDRdVv5MiRueKKK7L33nunffv2ad26dbbffvv06NEjzz///EKPM79zfv/996dVq1Z54IEHqlTf6NGjs99++6VVq1b59NNP51k/c+bMXHnlldl+++3TunXr7L777vnHP/5RpX0BAACw8OoUugAAAAAKr2nTphkwYEDF60mTJmXo0KG55ppr8umnn6Z3794FrK7qDjnkkHTr1i3dunVb4HYvvPBCVlhhhSVUFVV1++2356abbsouu+ySnj17Zv3110+dOnXy9ddfZ8iQITn99NPTsWPHXHHFFVluueUWefyBAwfmz3/+c3r27Jnu3bsv8vufeuqpnHfeeVlttdXmu82FF16YZ599Npdddlk22GCDPPfccznvvPNSv3797LHHHou8TwAAABaOYBwAAIDUqlUrq666asXrVVddNRtssEHGjx+fm266KWeccUbWWGONAla46GbPnp133333Z0PxJJU+OzXTBRdckFdeeSX9+/fP+uuvX2ndWmutlc033zwHHnhgDj/88PTu3TtnnnnmIo3/4osv5owzzsiRRx6ZY445pko1XnHFFTnvvPMyZ86cnH322fOsHzNmTB599NFcfPHF2XHHHZMkhx9+eN5+++3ccMMNgnEAAIDFyFTqAAAAzNevfvWrJMl//vOfimWPPfZY9t9//2y++ebZeuutc+qpp+abb76pWN+nT59sueWWGTRoUDp06JAePXrMd/xHHnkkrVq1yttvv51DDz00bdu2zQ477JABAwbkP//5T4488si0b98+O+20UwYOHFjpvW+//XaOOuqobLfddmnXrl1+//vf580330ySfPnll9l0000zbdq0nH322WnVqlWSudPGd+nSJQ888EC23nrrXHnllUnmnVb722+/Tc+ePbP11ltniy22yBFHHJHhw4f/7PH64Ycf8j//8z/ZfPPNs/nmm+f000/P1KlTK9a3atUqt99+e4499ti0adMmH374YZK504OfdNJJ2X777bPZZpulW7duGTx4cKWxy7fZeuut07p16+yyyy655ZZbMmfOnPnWM2PGjHTv3j177rlnvv/++yTJoEGD8rvf/a5iuvz/Pq7J3Om+r7322uy4445p3bp1tttuu5x11lkZN25ckqRnz57Zd999K72nZ8+eadWqVT7++OOKZS+//HLFlOLlx/6VV15Jt27d0rZt2+yyyy559NFHf/a4/vOf/8zgwYPzwAMPZP31188zzzyTLl26pE2bNtlzzz3zyiuv5LDDDsvDDz+cq6++Ovfcc0++++67nx233DvvvJMTTzwxXbt2zemnn77Q7/tvf/3rX7PPPvvMd/2LL76YsrKy7LDDDpWWb7/99hk1alRGjx5d5X0DAACwYIJxAAAA5mvUqFFJkjXXXDPJ3FD8jDPOSLt27fLII4/k5ptvzmeffZYjjjii0rPIS0tLc8899+SWW27JRRdd9LP7ufLKK3PMMcdU3A18wQUX5JxzzskhhxySRx55JOuuu27OO++8TJkyJcnckPjwww9PaWlp7rjjjjz44INZY401cuSRR+bTTz/Nmmuumfvuuy9Jcs455+SFF16o2NeECRMyaNCg3HPPPTn22GPnqWXmzJk56qij8sUXX+S2227LQw89lMaNG+fII4+sdAHAT7n++uuz5ZZb5pFHHskFF1yQp556KldddVWlbfr165ctttgiTz75ZNZff/1MmDAhhxxySEaPHp3evXvn0UcfzZZbbpkTTjghL7/8cpKkrKwsxxxzTL766qv07ds3Tz/9dE4++eTcdNNNFZ/zv82ZMyennXZavvnmm9x1111p3LhxPvvss5x88snZYIMN8sgjj+TKK6/Mgw8+mC+++KLSe88777zcf//96dGjRwYOHJjLL788r7zySo4++uiUlZWlQ4cOef/99yvORzL3OetrrrlmXn311UrLmjVrlg022CBJMn78+Nx4440577zz0r9//2ywwQY5//zz89VXX833mJaVleW6667Leeedl6ZNm2bgwIHp0aNHdttttwwcODBnnHFGrrjiirz77rtp3759Nttss7Ro0SJDhw5d4Lkq9+mnn+aYY45Jp06dcvHFF8+z/tZbb0379u3n+98FF1xQse166623wH2NHDkydevWzeqrr15p+brrrpsk+eyzzxaqZgAAABadqdQBAACYx6xZs/LKK6/k7rvvzq677loRjN96663Zaqutcu655yZJmjdvniuuuCL77LNPnn766ey1115JkqlTp+aII45ImzZtFmp/Xbp0SceOHZMkBx10UF566aVss802FdNNly/74osvsvHGG6dv376pVatW+vTpk4YNGyZJLrvssuy4447p27dv/vznP2ellVZKkjRs2LDSVOnffPNN7rzzzrRs2fIna3n22Wfz0UcfpX///tl4442TJBdffHEuueSSjB49ep5Q88e22267HHzwwRXH5vXXX88TTzyRCy+8MCUlJRX1/Hiq7r59+2bcuHF54IEHKgLSc845J6+++mpuv/32bLvttkmSu+++O/Xr188qq6ySJGnWrFn+9re/ZciQITn00EPnqeWyyy7LG2+8kfvvv7+i5sceeywlJSW54oorKo5b7969K459+fEZMGBAevbsWXH387rrrpuzzjorPXr0yBtvvJHf/OY3KS0tzZtvvpmOHTvm008/zQ8//JAjjzwyr776an7/+98nmXvHeIcOHSrG/vbbb3PXXXdVHPujjjoqzz77bN57772K79h/e/XVVzNt2rTssssumTFjRq688sr84Q9/yPHHH58kWWeddTJ69Oj06tUrbdu2TZJssMEGCwzby3311Vc56qijMmHChOy///6pVWve+wcOOuig7L777vMdo0GDBj+7n3KTJ0/OiiuuON8xfvjhh4UeCwAAgEUjGAcAACDjxo1L+/btK17PmDEjderUSZcuXXLWWWclmRvqffbZZ9l7770rvXfjjTdOkyZN8t5771UE40nSunXrhd7/pptuWvH/jRs3rhj3v5eVB4fvvPNO2rZtWxHuJkm9evWy+eabZ8SIEQvcV7169eYbipePvdxyy1Xaf5MmTdK7d++f/RxbbLFFpdetWrVKv379Mnbs2Ky22mpJ5j0u77zzTtZdd92KULzctttuWzHNeElJSSZNmpTevXvn7bffzsSJE1NWVpbp06f/5MUHd955Zx5++OHcd999ad68ecXyjz/+OOuuu26l47byyitX2ve7776bsrKybLnllpXGLP9+vPfee9lyyy3TsmXLvP766+nYsWNefvnltG3bNtttt10eeuihJHMvjhg+fHiOPPLIijFWWGGFSse+adOmSZJJkyb91OFMkrz22mvZZpttUrt27fz73//O119/nSOOOKLSNrVq1cpGG21UETBPmzbtJ0Pu//bEE0+ka9euGTt2bHr27Jl//OMfadasWaVtmjRpkiZNmvzsWAAAANRsgnEAAADSpEmTPPjggxWv69Spk1VXXTV169atWDZ58uQkyU033ZTbb7+90vunTZuWb7/9ttKyRo0aVfz/66+/nqOPPrri9RZbbJE777yz4nX9+vUr/r/8zuqfWlZWVlZRy4cfflgpzE/mToNeHrbOz49D4Z/yww8//ORdvQujPMAvV/4Zpk2bVrHsx8clmftZRo8ePc9nmTVrVmbNmpWZM2dm3LhxOeSQQ7LeeuvlggsuyDrrrJM6derktNNOm6eGTz75JCNGjMicOXPmCZynTJmSFVZYYZ73/Pjzlp/n/z5O5aFz+fTpHTp0yGuvvZZk7p3hW2+9dTbbbLNMmjQpn376af7zn/+krKwsv/71ryvG+Kl9J//vvP6Ub7/9tuKO91GjRqVx48bz3LU/ZMiQtGvXruL1yJEjs99++813zHJdunTJFVdckQkTJmTffffN8ccfnwceeGC+df5SDRs2rDT9fLnyCz7++7sBAABA9RGMAwAAkNq1a//s85HLg9Ijjjgi+++//zzrFxQmtm7dOv379694vfzyy1et0P9fo0aNssYaa6RXr17zrFuYO4UXpGnTppk8eXLKysoqAvmF9d+h59SpU5NkgUF7o0aNss466+SOO+74yfV16tTJoEGDMnXq1PTu3TstWrSoWDdp0qR5wvjatWvn3nvvzW233ZbTTz89AwYMqLhYoH79+hk3btw8+/jhhx8qzl95OPvf03r/d3j7m9/8Jvfee2+mTZuWV199NYceemjq1auXNm3a5NVXX82YMWPSrl27RZpq/KfUrl274v/r168/T4j+zjvv5LnnnquY2eCdd97JN998UzEF/YKU38W/0korpU+fPunevXvOOuus3HDDDRXn/tZbb81tt9023zH22muvXHLJJQv1WVq0aJGZM2fmq6++qjR1/KhRo5IkG2644UKNAwAAwKL7Zb8tAAAAYJmx4oorpmXLlhk5cmTWW2+9Sv/NnDkzK6+88nzfu/zyy1fafkHP6V4Y7dq1y8iRI7PmmmtWGresrKwi7Cy3oLuRf0rLli0ze/bsvPHGGxXLpk2blkMOOSRPPfXUAt/7yiuvVHr93nvvZaWVVqp4Lvj8PstXX32VBg0aVPostWvXzsorr5xatWpl1qxZSVLpbvg333wzo0aNmufzrb/++mnfvn0uu+yyJMmZZ55Zsc0GG2yQUaNG5fvvv6/Y/ptvvsno0aMrXrdu3Tq1atWquBu8XPnxKJ+6fauttkpJSUkefPDBTJs2reKO7S233DKvvvpqxTTrv9S6666bTz75JEny61//OpMnT87f//73TJ48Oc8++2yuu+66tG3bNt9//32+/PLLnHnmmfnTn/60yIH8pptumosuuihPP/10br755orlBx10UPr37z/f/04++eSF3kfHjh1Tq1atDB48uNLyQYMGpVWrVllrrbUWqWYAAAAWnmAcAACAhXbsscfm3//+d/r06ZNPP/00n3zySa688sp07do177333hKr47DDDsuUKVPSs2fPDB8+PKNHj85DDz2UffbZp2JK+PI7qV999dV88MEHmT59+kKNvfPOO6dFixa54IILMnz48Hz22We54IIL8sEHH6Rt27YLfO9LL72Ufv365Ysvvsg//vGPimdYL0i3bt3SuHHj9OjRI2+88Ua+/PLLDBw4MPvvv3/69OmTJBWh82233ZYvv/wygwYNyiWXXJLf/va3GT16dEaOHJk5c+ZUGrdp06a5+uqrM2TIkPTt2zfJ3LubS0tLc9FFF+WTTz7J22+/ndNOO63SRQ2rrrpqunbtmttvvz1PPPFERo8enX//+9+5/PLLs80222SzzTZLMvdZ7VtuuWX69u2btm3bVky7v+WWW+a1117LiBEj0qFDh4U65gvSqVOnvPzyy/nPf/6TDTfcMH/+859zxx13pFOnTunfv3+uuuqqHHjggfnb3/6Wgw46KPvss0+OPfbYKu2rW7du6d69e/r06ZNBgwYlmfuYgf++EOTH/5Ufu5kzZ2bs2LEZO3Zsxd31EyZMyNixYzN+/Pgkyeqrr56DDz44f/nLXzJ48OCMGTMmd9xxR5599tmceuqpv/hYAQAAMH+mUgcAAGCh7bnnnqlVq1buuOOO3HbbbalTp07atGmTO++8M61bt15iday33nq55557ct111+Wwww7LrFmz0rx585x55pnp3r17kmSVVVbJwQcfnH/84x957rnnKk3lviB169ZN3759c/nll+fII4/MnDlzsummm6Zv376Vpr/+KWeeeWYGDBiQyy67LLVq1UqXLl1yyimnLPA9TZo0yf33359rrrkmf/rTnzJ16tSsueaaOfzwwyuey7755punZ8+eueeee/L3v/89bdq0ybXXXpsJEybkxBNPzEEHHVQR5P7Yr3/96/zxj3/Mtddem6222iqtW7fO1VdfnRtuuCH77LNPmjVrlpNPPjn/+Mc/MnPmzIr3XXTRRWnatGmuueaajB07NiuttFJ22WWX9OzZs9L4HTp0yIsvvljped6bb755xo8fn8aNG2fTTTf9ucP9s1q0aJE99tgjp59+em6//fbst99+8zw/fJ999sk+++yT77//fp6p5RfVOeeck/fffz+nn356HnzwwbRs2XKh3vfWW2/lsMMOq7Ts97//fZKkWbNmFXeJn3322WnQoEEuuuiijB8/Puuvv36uu+66/Pa3v/1FdQMAALBgJWWLOqccAAAAwBI0derUHHfccfn6669z3HHHZfvtt0/Tpk1TWlqab7/9Nm+99VaefPLJDB8+PM8880zF3esAAABQTjAOAAAA1HilpaXp169fHnroobz33nupU6dOZs+enTp16qR169bZbbfdKqalBwAAgP8mGAcAAACWKtOnT8+ECROy3HLLpUmTJqlTx5PiAAAAWDDBOAAAAAAAAABFrVahCwAAAAAAAACAxUkwDgAAAAAAAEBRE4wDAAAAAAAAUNTqFLqA6jB79ux8//33qVevXmrVkvUDAAAAAAAAFLs5c+ZkxowZady4cerUWXD0XRTB+Pfff59Ro0YVugwAAAAAAAAAlrDmzZtn5ZVXXuA2RRGM16tXL8ncD1y/fv0CVwPVr7S0NB999FFatmyZ2rVrF7ocoID0A6CcfgCU0w+AcvoBUE4/AMrpBxS7adOmZdSoURV58YIURTBePn16/fr1s8IKKxS4Gqh+paWlSZIVVljBX1ywjNMPgHL6AVBOPwDK6QdAOf0AKKcfsKxYmMdteyA3AAAAAAAAAEVNMA4AAAAAAABAUROMAwAAAAAAAFDUBOMAAAAAAAAAFDXBOAAAAAAAAABFTTAOAAAAAAAAQFETjAMAAAAAAABQ1ATjAAAAAAAAABQ1wTgAAAAAAAAARU0wDgAAAAAAAEBRE4yT3XbbLf369au27ai5dtxxxzzwwANJkkMPPTTXXHNNgSuqeV5++eVsv/322WOPPQpdSlE79dRTc9ZZZxW6DAAAAAAAYBlRp5A7b9WqVZZbbrmUlJRULDvggANy/vnnZ+jQobn22mvz2WefZc0118yxxx6bvffeu1r3P+PyO6p1vAWpd/bRS2xfi+rpp5+u1u1qmn893HWJ7m+X/R5dovtb2nQbdN0S3d8jO5+6SNv/9a9/Tbt27XL99dcnSfr375+LL744v//973PaaacthgoXzoFPvrlE9/fg7psv0f0tDd69buoS3V/rU1dYpO1btWqVO+64I9tvv/086/r375+//OUvGTx4cHWVBwAAAAAAS5WCBuNJ8tRTT2XttdeutOzbb7/N8ccfn3PPPTd77bVX3njjjRx33HFZf/3106ZNmwJVCiwLJk+enLZt26ZWrVq5+OKLM3z48Ky11lqFLgt+kX322Sf77LNPxeuHH344O+64Y5o2bbpY9vfMM8+kVatWWW+99RbL+AAAAAAAsKhq5FTqjz/+eJo3b5799tsv9erVy3bbbZcdd9xxmZrG+8svv0yrVq3y9NNPp3Pnztlss81yyCGHZOzYsXnllVfSvn379O3bN5tvvnneeuutJMm9996b3XffPW3btk3nzp0zaNCgivGmTZuW888/P9tss0223XbbnH/++Zk5c2aSytNrv/322znggAPSvn37bLPNNjn33HMzffr0ebabM2dObrrppuyyyy7ZbLPN0rVr1wwdOrRif+Xn65hjjkn79u2z884754UXXlgix25p06pVq/Tt2zcdOnTI7bffniQZOnRoDjzwwLRv3z4dO3bMLbfcUuk9d999d377299m8803z1FHHZUvv/wySVJWVpZrrrkmnTp1Svv27dO1a9e89tprS/wzLa0OOeSQvPbaa7n77ruz2267Zc0118z999+/2MLDYtKqVas888wz6d69e9q1a5e99tor7733XsX6hx56KDvuuGO22GKLXHzxxZkzZ06l98/vO11aWppLLrkk7du3zw477JB//vOf2XXXXfPII48s0c9XTEpLS3PFFVdkwoQJi20ff/nLX/L5558vtvEBAAAAAGBRFTwYv/baa7PDDjtkyy23zPnnn58pU6ZkxIgR2WSTTSptt8kmm+Tdd98tUJWFc++99+buu+/OkCFDUlJSkosuuihJMmvWrHz++ed56aWX0q5duzzzzDO58cYbc/XVV+eNN97IySefnFNOOSX/+c9/kiS9e/fOJ598kieffDIDBw7MiBEjctNNN82zvzPOOCP7779/3njjjTz++OP58MMP8+CDD86z3X333Zd+/frlxhtvzOuvv5699torxx9/fMaNG1exzV133ZUTTzwxr7zySrbeeutcdtlli+cgFYFBgwalf//+Ofroo/P111/n+OOPT/fu3fP666/nzjvvzIMPPpgXX3yxYts77rgjt9xyS15++eWsueaaFVN8P/bYY+nfv38efPDBvP7669lpp53So0ePlJaWFvLjLTXuvffebLXVVjnyyCPz9NNP55hjjkndunULXdZS484778yll16aoUOHZrXVVst1182dNv+zzz7LBRdckHPOOSdDhw7Npptumueff77ifQv6Tt9zzz158skn89BDD2XAgAF58skn8+233xbk8y0Nxo4dm8MPPzybbbZZ9thjj3z00UdJkkceeSS/+c1vkiRbb711fvjhh3Tp0iU33njjz47Zp0+fHHvssTnllFOy+eZzp9gfP358evTokV//+tfZcsstc/TRR+err75Kkuy99975+OOPc/zxx+fss89OknzwwQc5/PDDs+WWW2bbbbdNr169MmvWrMVxCAAAAAAA4CcVdCr1du3aZbvttsuVV16Z0aNH55RTTsnFF1+ciRMnZvXVV6+0bZMmTX727rbS0tJFCgDLqlR11SxqMFl+N+VBBx2UVVZZJUly2GGH5X/+539y8MEHZ9asWTnwwAOz3HLLZc6cOenXr1/23XffbLzxxkmSnXbaKZtvvnkef/zxHHXUUenfv3969eqVxo0bJ0l69eqVSZMmpbS0NGVlZZkzZ05KS0szadKkLL/88ikrK8vKK6+cBx54ILVq1Zpnu379+qV79+7ZcMMNkySHH3547rzzzgwePDjdunVLWVlZdthhh2y66aZJkp133jn9+/fPrFmzUqvWkr0eo2yJnulFP9dJsttuu2WllVbKnDlzMmDAgGy44YbZa6+9kiQbbrhh9t9//7zwwgv505/+lIcffjh77LFHNtpooyRJjx498tprr2XWrFnZY489ssMOO6Rhw4ZJkt133z19+vTJl19+mbXXXrvSOSwrK0tZWVkBQvOafT5+6rgU7lj9uK4lu7+qfNa99tqrYurs3/72t7n77rtTWlqaf/3rX9l4443z29/+NknStWvX9O3bt+KYLug7/dxzz6Vz585p0aJFkuTUU0/Nv/71r4rv8RJVQ85B+fKfWv/3v/89V1xxRVZdddWceOKJ6d27d2666abMmTOn4ng/+uij2WWXXfLoo4+mRYsWP3sc58yZk2HDhqVHjx656qqrUlpamquuuiqTJ0/OM888k7KysvzP//xPLr300txwww159NFHs8kmm+TGG29Mx44dM3ny5Bx11FE55JBDcuutt+bbb7/NiSeemDvuuCPHHnvsLz9QsAxbUD8Ali36AVBOPwDK6QdAOf2AYrco3+2CBuM/vhN5gw02yGmnnZbjjjsuW2yxRZXGK78zbmFtOH1GlfZTFSOGDVuk7ceOHZskmTlzZob9/+/94YcfKr0eN25cpkyZkiT5+OOP8+KLL+avf/1rxRhz5sxJ48aNM2TIkEyaNCmTJ0+ueG+SLLfcchk2bFhmzpyZL7/8MsOGDcu+++6bc845JzfddFPatGmTjh07plmzZhW1lG/3xRdfpKSkpNJ4TZs2zZtvvpkWLVpUTNNevn7MmDEpLS3N66+/vsTvwJ0xY8md5ySVjsnCmj59esX7hg0bluHDh6dt27YV68vKyrLmmmtm+PDh+fjjj7P22mtX2s8aa6yRd955J1OnTs3f/va3vP322xXfjWTuFPnfffddpXM4efLkfPPNN1Wq95eo6efjp45LoY7Vj82YMXOJ7q8qn3XGjBkV7/v2228res67776bBg0aVBqzSZMmGT9+fIYNG7bA7/To0aOz4YYbVlpXv379fPHFF0v+fMxYf4nubtiw9xe4fvjw4fMs22KLLTJx4sRMnDgxG220Uf79739X9OzZs2dn2LBhFX+/vP/++5k0adLP1vH1119nzpw5adWqVcU+99lnn5SWllb8vduyZcs89thjlc7JZ599loYNG+bll1/OrFmzsvXWW1dMr7/TTjulX79+2WabbRbqWAAL9lP9AFg26QdAOf0AKKcfAOX0AyhwMP7f1l577ZSWlqZWrVqZOHFipXUTJkz42ef8tmzZMiussMJC72/WM29Wpcwqadeu3SJtP2bMmCRz7xYun1a+Xr16SZL1158bzrRv375iWePGjXPIIYfkD3/4wzxjlR/Lli1bzjNFfZLUrVs3a6+9dtq1a5d27drlD3/4QwYPHpzBgwfn3HPPzTXXXJOdd9650nalpaVZf/31K32uFVdcMWussUbatWuXunXrZp111qlYXx6Ut23btqLmJWXcyCW7v0U910my0UYbVbyvWbNm6dSpU26++eaK9aWlpRk+fHjatGmTFVZYIWuuueZP7ufMM8/MN998kwceeCDrrbdeRo8end/97nfZeOON06JFi0rnsEGDBll99dWrVO8vUe/Z539+o2q0qJ/vp45LoY7Vj9X75u0lur927dr+/Eb/ZcMNN6w4RiNHjkydOnXSrl27NG7cOLNnz650/Bo1apSGDRumXbt2C/xO161bN+utt16ldcstt1zWXXfdJX4+3n9+yV7UsfF8Pt+P+0Ht2rUrrdtmm20qjssHH3yQwYMHp127dpXOR/nfL+V94ee88MILWWedddK+ffuKZR9//HGuuuqqDB8+PNOnT6+4EOvH56RFixZp165dXnvttUyaNClHHHFExbqysrLUrVu3oH+moBgsqB8Ayxb9ACinHwDl9AOgnH5AsZs6depC3zxdsGD8vffey4ABA3LWWWdVLPv0009Tt27ddOrUKY8++mil7d99991Kd9D+lNq1ay/SH+rZi1byL7KozaZ8uvExY8akTZs2Sebetbf88stXXCDw48+73nrr5eOPP660n//85z9Zc801s/LKK6dRo0b5/PPPK8YaMWJEPvnkk3Tp0iUlJSWpVatWateunQkTJmTllVfO/vvvn/333z833nhjHn300ey2226Vtlt33XUzatSoiv3Nnj07X3zxRbp3757atWtX2vbHn2dRz1F1KEnJEt1fVT7fj4/Veuutl3//+9+pVatWSkrm1j527NjMmjUrtWvXzjrrrJPPP/+8Yvvx48enf//+OfTQQzN8+PDsv//+2WCDDZLMDcZ+PP6Pz0tJSUlKSkoK8BdhzT4fP3VcCnesflzXkt3fL/0el39/a9eundVXXz3vvfdepTE/++yztGvX7me/06usskq++uqrinWff/55Jk2aVGlfS0wNOwc/1U9/vOzH5+C//798/cIcw1q1aqVOnToV286ZMyfHH398tthiizz99NNp2rRp+vXrl+uvv77SeOXj169fPxtttFEef/zxRfr8wMIrxM9XQM2kHwDl9AOgnH4AlNMPKFaL8r1esg97/pGVV145Dz74YG6//fbMnDkzI0eOzA033JADDzwwXbp0yZgxY9KvX7/MmDEjzz//fJ5//vkccMABhSq3YB544IF89913mThxYv7617+mU6dOFWHpjx144IEZOHBgnnvuucyePTsvv/xy9txzz7z99ty7TLt165Y777wz33zzTSZMmJA///nP+fjjjyuN8fXXX2fHHXfMCy+8kDlz5uSHH37IRx99lHXXXXee/XXp0iX3339/Pv3008ycOTO33nprSktLs+OOOy6eA7EM6dy5cyZOnJibb74506dPz+jRo/PHP/4xTz31VJJk3333zT//+c+8/fbbmTlzZm666aY89dRTWW655bL22mtn+PDhFVPu//Of/0wyd1prKJTtt98+7733Xp577rnMnDkz9913X7755puK9Qv6Tm+zzTZ5/PHHM3LkyPzwww+57rrrFmlmEKrfd999lzFjxuTQQw+tuFCrfIr0n7Luuutm9OjRlR7vMGHChEyePHmx1woAAAAAAOUKdsf46quvnttvvz3XXnttbrnlltStWzddu3bNqaeemnr16uW2225Lr169cvHFF6dZs2a5+uqr86tf/apQ5RbM3nvvncMPPzxffPFF2rVrlwsvvDCffPLJPNv95je/yZlnnplLLrkk3333XdZee+1cdNFFFdPU9uzZM7169coee+yRunXrZuedd86JJ55YaYw11lgjl156aS699NL85z//SYMGDbL99tunR48e8+zvyCOPzIQJE3L00Udn0qRJ2XjjjfO3v/0tjRo1WizHYVmy0kor5eabb85VV12VW2+9NU2bNs1ee+2VTp06JZn7bN5TTz01J5xwQqZOnZr27dvn2muvTTL3PJ9xxhnZeuut07Zt21x11VVJkuOPPz733ntvwT7Topo5YU61jfXudVMXafspo0szduasDL7kk/T4e9ckyezSWXn9tTfS9+6/ptk6a+Xpp5+utvqWBW3bts15552Xiy66KJMmTcpee+2V3/3udykrK0uy4O/0H//4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NfeULjvnYCmlHxShmntKl51zsJTSD4pQzT2ly845WErpB0Wohp7SZeb4L6WWhV7w47GWie9jzT2ly8bxX4rpB0Wo5p7SZeP4L8X0gyJUc0/psnH8C2RRjm3BgvFyY8aMyW677ZbZs2fngAMOSI8ePXL00Udn9dVXr7RdkyZNMmHChAWO9dFHHy3OUqmhZsyYUegS5mvYsGHVOt4y8ziBGesXuoL5Gjbs/UKXwALoB0VIP6CK9IMipB9QRfpBEaqh/UAvqNmWpV6QLCP9oIb2gkQ/qOn0gyKkH1BF+kER0g/4GQUPxps1a5bhw4fn888/zwUXXJAzzjijymO1bNkyK6ywQjVWx9Jg3Mh6hS5hvtq1a1ct45SWlmb48OFp06ZNateuXS1j1mTvP19zfyDZuJrOKYuHflB89AOqSj8oPvoBVaUfFJ+a2g/0gpptWegFybLVD2pqL0j0g5pOPyg++gFVpR8UH/1g2TR16tSFvnm64MF4kpSUlKR58+Y59dRTc9BBB6VTp06ZOHFipW0mTJiQpk2bLnCc2rVrF/0fauZVkpJClzBf1f19XGa+4zX3lC4bx38pph8UoZp7SpeN478U0w+KUM09pcvG8V+K6QdFqIae0mXi2C/FlqVeUD5m0X8na+4pLf5jv5TTD4pQzT2lxX/sl3L6QRGquae0+I99AS3Ksa21GOtYoKFDh2a33XbLnDlz/l8xteaWs9lmm+Xdd9+ttP27776btm3bLtEaAQAAAAAAAFj6FSwYb926dSZPnpyrr74606ZNy/jx49OnT59sueWW6d69e8aMGZN+/fplxowZef755/P888/ngAMOKFS5AAAAAAAAACylChaMN2zYMHfffXfefffdbLvttuncuXMaNmyY3r17Z+WVV85tt92We++9N1tssUUuu+yyXH311fnVr35VqHIBAAAAAAAAWEoV9BnjrVq1yj333POT67baaqs89thjS7giAAAAAAAAAIpNQYNxoHAOfPLNQpcwX+fH7BAAAAAAAABUnyoF47Nnz87o0aMzfvz4lJWVpWnTpll33XVTp46cHQAAAAAAAICaZZGS7EGDBuWBBx7Im2++mWnTplVaV79+/Wy++ebp3r17dt5552otEgAAAAAAAACqaqGC8S+++CKnnnpqvvnmm+y999457LDDstFGG2WllVZKSUlJxo8fn48//jivvvpqLrrootxyyy25/vrrs8466yzu+gEAAAAAAABggRYqGO/evXuOOeaYdO/ePXXr1p1n/VprrZW11lornTp1ysknn5wHHngg3bt3zwsvvFDtBQMAAAAAAADAolioYPy+++5L8+bNF2rAunXr5vDDD88OO+zwC8oCAAAAAAAAgOqxUMH4f4fio0ePzhVXXJHXX389U6ZMyYorrpj27dvnrLPOqth2vfXWq+5aAQAAAAAAAGCR1arKmy688MJ07tw5//73v/POO+9k4MCB6dixY0455ZRqLg8AAAAAAAAAfpmFDsbPPPPMfP/990mSyZMnZ/vtt0+DBg1Sq1atrLzyytlll13y1VdfLbZCAQAAAAAAAKAqFmoq9SRp2bJlunbtmh49emT//ffP7rvvnvbt22eFFVbIxIkT88477+SEE05YnLUCAAAAAAAAwCJb6GD8qKOOyu9+97v8+c9/zvTp03Pddddl0qRJ+eGHH9KgQYNccsklWW211RZnrQAAAAAAAACwyBY6GE+SZs2a5dZbb83TTz+ds88+O926dcvRRx+dOnUWaRgAAAAAAAAAWGIW+hnjP7bbbrulf//+GTduXPbdd9+89dZb1V0XAAAAAAAAAFSLhb7Ve+TIkbnxxhvz/vvvp6SkJJtttllOPPHEdOnSJRdddFFat26d008/PQ0aNFic9QIAAAAAAADAIlnoO8ZPPvnktG3bNn369MkNN9yQ5s2b56STTkqbNm3Sr1+/rL/++tl3330XZ60AAAAAAAAAsMgW+o7xr776Kt26dau4I3yVVVbJ3XffnSSpVatWjjjiiOy+++6Lp0oAAAAAAAAAqKKFDsYPOOCA7LPPPmnbtm3mzJmTYcOG5dBDD620zeqrr17tBQIAAAAAAADAL7HQwfjpp5+erl275sMPP0xJSUl69OiR9ddff3HWBgAAAAAAAAC/2EI9Y/yss87KlClTsuGGG6Zz587ZY489FhiKT5kyJWeddVa1FQkAAAAAAAAAVbVQwXijRo2yxx575K677srEiRPnu93333+fu+++O507d06TJk2qqUQAAAAAAAAAqLqFmkr9nHPOSYcOHXLDDTfk2muvTatWrdKyZcs0btw4JSUlmThxYj7++ON8+OGH2XjjjfPnP/85HTt2XNy1AwAAAAAAAMDPWuhnjG+//fbZfvvt88477+Tll1/Oxx9/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kpKSQpcGAAAAAAU1adKkNGrUqNBlAADAUqFGB+MAsKwrLS1N7dq1C10GUEAfffRRWrZsWegygAIbNWpUGjRokFVWWSVlZWUuGIdl2IgRI9KvX7+8//77Ofzww7PHHnsUuiSgwPzuAJZtpaWlKSkpSa1aNXaSaKgxBONQQ/gBFpgfv/yGZdcLL7yQSy+9NIccckh+//vfF7ocYAmbPn16ll9++Xz88ce58847M23atJx++ulZZ511/HwAy4jyP+uTJk3K3/72twwZMiSlpaXZcccds88++2SttdYqdIlAAfzUzwGlpaUpKytLnTp1ClQVsKTM798CEyZMSKNGjeQMsAAuH4Eaovwvq6+++qrAlQBL2rRp0zJ16tSfXPf888+na9euSeb+0AsUt9LS0syZM6fidYcOHXL99denf//+ufnmm+fbK4DiMWnSpPz1r3/NIYccknPOOSeDBg3KRhttlCuvvDKrrbZaTj/99IwePVooDsuAH//S+9FHH819992XI444Ig8//HCOP/54oTgsY3744YeKfw/8988BL730Uo488shMnjy5EKUBS9DYsWPzf//3f0lS8fuDQYMG5cgjj8x+++2Xs846K2+99VYhS4QaTTAOS9icOXMq/cK7fNk999yTbt265ZhjjknPnj3zzTffFKhCYEkYP358xZ/zV199NR9++GGSZPTo0RkzZkzFdiUlJWnYsGFmzJjhF+CwDKhdu3Zq1aqV8ePHZ/z48UmSVq1a5cQTT8ywYcNy//33F7hCoLqVX/j21Vdf5fDDD8/BBx+c9957LyeeeGJat26dyy+/PG+++WaS5Lzzzssaa6yRq6++OhMnTixg1cDi9Nxzz+Xkk0/OhRdemH/84x8ZP358ttlmm2y11VaZMWNGpW2nTJmSzz77rECVAovbN998ky+++CLJ3NmkXnrppSRzf2544403KrZr0qRJPvjggzRp0qQQZQKLUVlZWaU84Ycffshjjz2WgQMHZty4cZk+fXoefvjh7Lbbbvn3v/+d5s2b5+abb86IESMKWDXUXIJxWMJq1aqVWrVqpbS0tGLZRx99lOeeey6nnXZaHn/88dSuXTt9+vTJ559/XsBKgcVl1KhRufXWWyt+oT1kyJBce+21ufjii3PyySfn+uuvrwjHR40alXXXXTezZ88uYMVAdZszZ06lWSDKfy4YOnRoDjzwwBx88MG55JJL8swzzySZe+f4zjvvnIEDB7p4DorIlClTKi58mzJlSt57772cfvrpufLKK7Ptttuma9eu2XTTTSv9Uuukk07K2LFj869//atQZQOLyfDhw3PggQfmrrvuyrbbbptf/epXefzxx/P222+nefPmWXPNNTNs2LAkyeDBg9OjR4/svffeeeONNzJr1qzCFg9Uu/Hjx+ehhx7Kl19+mWRuGH7ppZfm0EMPzX333Zczzzwz48aNSzL38Subb7653yVCESkPxP/72eH//Oc/M3DgwPTp0ydz5szJ008/nc8//zwHHnhgkrkX148YMSJvvPGG2SfhJwjGYTH57yu5yr344ovp0aNHjjzyyAwcODBJMmDAgKy44orZbrvtkiTrrbdehgwZUnEHKbD0+/HFMM2bN88555yT+vXr5/XXX89bb72VkSNHZqeddsptt92WyZMn56qrrkqSrLHGGhk2bFhWXHHFn+wpwNKl/B+ltWrVqgjDPvroo9SuXTuTJ0/OXXfdlYMOOihPPfVUdthhh9x55515++23U7t27Wy77bZp2LBhHnvssUJ+BOAX+vzzz9O7d+9069Yt5513Xm688caMGzcuG2ywQXbeeecMGDCgYtuysrJMmDAhO+ywQ8WyDTbYIJtsskneeeedfP311wX4BMDiMH369Nxzzz3ZZZddcs8996R79+45+OCDc+utt2arrbbK8ssvn1/96ld544030qlTpzz44IPp1KlTBg4cmP333z/LLbdcoT8CUA1+/LuDpk2b5qSTTkqjRo0yduzYDBw4MJMnT86+++6b0047Leutt16uuOKKfPfdd5k1a1amTZuW1VZbrYDVA9WpPBD/5ptvcscdd+TGG2/MZ599lm233Ta77rprtt9++6y++upZbbXVMmPGjPTq1StdunTJ//7v/+bss8/OwQcfbPZJ+AmCcahm87uSK0n69euX3r17Z7fddsvRRx+dKVOmZMKECVl77bXz0Ucf5dxzz80ee+yR4cOH59prr82uu+5aoE8B/FJlZWWVrsqsXbt2kuTNN9/M008/nb///e8577zzMm3atPTq1StbbLFFpkyZklVXXTWnnnpqPvvss9xxxx1Zd911s+qqq2bSpEnz9BRg6fDjXlD+j9J33303N910U+6777706NEj3333XYYPH55p06Zlr732SpKstNJK+fTTT/Pkk08mmXuhzG9+85s8//zzS/5DANXi1ltvzYknnpjZs2enV69e2W677TJ16tTUr18/SbLTTjvltddeywcffJDevXunS5cumTJlSm6//fb079+/Ypytt9463377rbvCoIhMnDgxL7zwQsWFMDNnzkySLL/88mnQoEGSpHXr1llnnXWy995757bbbsu+++6bevXquYAWlnI//jNc/ruD7777Lsncfzf8z//8T1577bX07ds3nTp1qphh7owzzkhpaWmuvPLKtGzZMp988kmWX375Jf8BgF/kpx69msz9XULfvn1z2GGHZeTIkalbt24GDBiQrbbaKh07dsxnn32Wzz//PGuvvXZWWWWVjBo1Kvfdd1/uu+++7LnnnrnzzjsrXWwDzFWn0AXA0m7WrFmVrswuKSlJSUlJvvnmmzz55JNZYYUV0rlz59StWzfvv/9+tthii3Tu3DnJ3OeBNGzYMBtttFFKS0vTsGHDirvIP//88/ztb3/LYYcdVpDPBVRN+Q+y/x1iv/TSS3nggQey4oorpnPnztloo43y0ksvZeTIkenYsWPq16+fN954Ix07dkzLli1z/PHH57777stLL72U1q1bm/oIljLlU6XXrl17niu0b7311jz00EM56KCD8vbbb+ebb77JiBEjUq9evUybNi0XXnhhhg8fnsaNG+e8887LnnvumSSpW7duWrRokSFDhuTLL7/M2muvXYiPBiyEsrKylJSU5JVXXsmmm26aBg0aZNiwYRk0aFDuuuuuiru5Ntlkk0rva9u2bZo2bZrDDz88Rx55ZP73f/83G264Yf7617/mvvvuy3rrrZf27dtnq622yi233CIMg6VI+UX05aHXf3vzzTfTqlWrip8b6tatO8826623Xlq2bJnRo0dnypQpWXHFFVNWVuYCWljKjB49Og0bNswf//jHXH/99RU/13/33Xf517/+lRtuuCEbbbRR9t9//+y99975zW9+kyFDhmSPPfZIy5Yt8/zzz+eEE05Iq1atctppp2W//fbLrbfemo033jijR4/OuuuuW+BPCPycd955J/fdd19KSkpy4YUXVlwo++abb2b11VdPs2bNMnv27DzxxBO5+uqrs9lmm1V6//rrr5/lllsur7zySg444IDsvPPOefLJJ1O/fv1MmDAhf/3rXzN58mQ/I8BP8KcCqmjw4ME55phj8sc//jF9+/bN999/n6TylVzvvPNOHnvssVx88cWpVatWNthggwwdOjQnnHBCOnTokK5du+b6669PSUlJdt1114o7Pj7//HO/6IKlxKxZs/Lmm29mwoQJSeYG4rVq1crYsWMzYMCAfPDBB0nm/hJr6NChqVOnTjp27JgWLVqkWbNmee+99zJlypRsueWW+frrryseobD77rvnwAMPzNChQ/P666+ncePGwnGo4X78bM9atWqldu3amTFjRp599tmMHDkyydxnCI8YMSJ//OMfc8wxx+Skk07KzjvvnMcffzxbbbVVJk+enOnTp+eGG27IPffck65du+bvf/97vvjii4pxGzZs6KpvqMHKZ49666230qNHj8yePTtJ8uyzz6ZFixZp2rRpxZ/hOXPm5PHHH8+f/vSn9O3bN6uuump+/etfZ9NNN82xxx6bDTfcMEly2GGH5dNPP60IzMrHKH/mqJ8RoGb671ljfioUL+8Ra665ZsaNG5cpU6b85FhPPvlkpkyZks033zyTJ0/OW2+9Nc8+gJpr6tSpmTx5cvr06ZPdd98906dPzx133JE111wzSdK3b98ceeSRmTNnTu6///7stddeueSSS5IkW221VT788MOMHz8+2267bWbPnp0XX3wxSbLWWmvl/PPPz7PPPpuvvvoqq6++esE+I7BgX375ZW688cbsuuuuOfvss9OiRYuceuqpqV+/fu68887svffeOe+883LKKafkqaeeynLLLZf69evn3HPPTZ8+ffKHP/whZ5xxRp577rlsueWWadasWYYNG5YkOeaYYyr+DfH73/8+Y8aMySGHHGIqdfgJgnFYBFOmTEmfPn3SqVOn9OvXL7/73e9y0kkn5cEHH8zDDz+cJJkwYUL+9re/5YYbbkjv3r1zxhln5D//+U9eeOGF/P73v8/111+f3/3ud7n22mtzxRVXZMiQIXn77bfzpz/9KU2aNMnBBx+cU045JSuvvHL22Wefwn5gYL7KL1x5/vnnc+GFF1YE4FOnTs0ZZ5yRP/zhD3nsscdy9NFH59VXX02zZs2y5ZZbpl69epk0aVKSZPPNN8+ECRMyfPjwbLvttpk2bVpGjBiRcePGJUn23HPPHH/88alVq1a+//57P8xCDTZ06NCKC1uSZNq0abnqqqvSuXPn3HTTTTnnnHMyaNCgigtnmjZtmtmzZ2edddZJx44dM2rUqEycODF77713kuT777/PnDlzcumll2bUqFFZa621ksy9yOb999+v+AUaUDNMmDAh9957b5L/N2vMv/71r3Tr1i1NmjRJknz88cdZfvnlU6dOndSuXTsjR47MwQcfnCeeeCJNmjTJP/7xjyTJjjvumA8//DDTpk1LMnc65QceeCCdOnXKeuutV7HPtm3bZvr06UniZwSoYcofq/TjP5tffvllLrvsslx22WX59ttvk8x9lnCdOnMnc2zfvn3q1KmTV199NZMnT07y/6ZT//zzz/PQQw/liy++yAYbbJCSkpK89tprSeadqQqoWWbOnFkRhH355ZeZPHly9ttvv8yePTuff/552rZtmyRp3rx5Pv/882y00UZp0aJF9thjj9SrVy+DBg3KtttumwYNGuTf//53Ntlkk6y11lp55plnKi7Q33nnnXPKKadU/C4BqFlmzZqVM888M8cdd1ymTJmSmTNn5vDDD8+xxx6b1VdfPe+8804GDx6cyy+/PAMHDszOO++c3r17J0kuv/zyHHXUUfn666/Trl27rLXWWjnuuOPy1VdfZZtttsl7772XHXfcMYMGDUqvXr1ywQUX5J///GeuvvrqNG/evLAfHGooU6nDIigtLc2wYcOy00475YILLqhYvu2221Y8+2fWrFm58MILs9FGG+XFF1/M3//+93z22WcZPHhwOnXqlA022CDrr79+xT9e11prrbRq1SoNGzbM5ZdfbmpUWEqU/xnu0KFD7rvvvnzyySfZaqut8vzzz2fcuHG5995706RJk5x99tnp27dvtt566+y2226555578v3336dRo0bZ4v9r777jsiz7x/+/LrayUVmiIiggIjhAJGUpiuAMt+KqXGWaC0vtLlNDUxumZlqO3CMH7j0RFURFVEhwMFRAQRRF9u8PHtcZOO67z/d330H1fv5TXNd1rh6d53kcx/t4v49WrThw4AAXL16kTZs2tG3blu+//54NGzawcuVKLC0tcXR0JCcnByMjoyq+YiGEWmpqKrt376ZevXo0bdoUOzs7MjIymD9/Pu3btycoKIinT59y4sQJ9u7di66uLnPnzmXPnj106NCBWrVqcf36ddq1a4eBgQHm5uY8fvyY48ePM2bMGNauXcuiRYtIT0/H09OT0NBQZdC8oKCApk2bUlhY+NoSq0KIqpGcnMzGjRu5dOkSffr0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
)
    



```python
# Ft imp for metaranker
imp_df = pd.DataFrame({'feature'   : STAGE2_FEATURE_COLS,
                       'importance': meta_lgb.feature_importance(importance_type='gain'),}).sort_values('importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=imp_df, x='importance', y='feature', palette='viridis', ax=ax)
ax.set_title('Meta-ranker feature importance (gain) — S6 LightGBM')
ax.set_xlabel('Information gain')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print(imp_df.to_string(index = False))
```


    
![png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA90AAAJOCAYAAACqS2TfAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAApQJJREFUeJzs3XdYFFffxvEbUETU2CuI7YmAEQSDIooNkxhExWhi59FgbNgTa4wlUWOP2IkFS2Jv2GOJGoO9RCWxBhvYjbGgIgj7/pHXfdwACsqK5fu5Lq9rd+bMmd+Mg3LvnDlrYTAYDAIAAAAAAOnOMqMLAAAAAADgdUXoBgAAAADATAjdAAAAAACYCaEbAAAAAAAzIXQDAAAAAGAmhG4AAAAAAMyE0A0AAAAAgJkQugEAAAAAMBNCNwAAAAAAZkLoBoCXTN++feXo6Ki+ffum2CYoKEiOjo6aOHHiC6zsxXN0dNSYMWNe2P4OHDig2rVrq2zZslqzZs0L2++LMnHiRDk6OurBgwcZXcprbeDAgWrYsKHi4uLStd+AgAA1btw41e1/+OEHeXt768qVK+laR1okJCRo4cKFatq0qSpVqqR33nlHVapUUadOnXT8+PEk7devX6/69evLxcVFPj4+mjRpkhITE1PsPzo6Wo6OjlqwYMFztUnO8/wbm5CQoCVLliggIEAVK1bUO++8Iy8vLwUFBenXX39N0t7Hx0eOjo7GP2XLllWNGjXUr18/nTt3zqRtQEDAU2vz9/eXo6Ojli9f/kz1A0hfhG4AeAnZ2tpqw4YNunv3bpJ1N27c0Pbt25U1a9Y097t79275+PikR4mvpZCQEN29e1crV65UzZo107XvZcuWKSAgIF37TKvAwECFh4crS5YsGVrH00RFRcnR0TGjy3gmixYt0tq1azV+/HhZW1una98TJ07UtGnTUt0+ICBA7777rjp37qyHDx+may2pNXDgQI0cOVJ169bVggULtGHDBn377be6dOmSAgICFBUVZWy7ceNG9ezZU40bN9b69evVtWtXff/992k65uQULlxY4eHh+uijj573cJJI7t/UuLg4tW3bVqNHj1b16tX1448/asOGDQoODlbWrFn12Wefafr06Un6qlWrlsLDwxUeHq4NGzbo66+/1sGDB9WiRQvdv3/fpK2tra1WrFghg8GQpJ/jx4/r/Pnz6XugAJ5LpowuAACQlLOzs06dOqX169fr448/Nlm3Zs0aFStWLMkvYanx22+/pVeJr6Xbt2+rRIkSKlWqVLr3/TKc+2zZsilbtmwZXcZTvQzn6lncunVLY8aM0aeffqqiRYume/+5cuVK8zZ9+vTRBx98oCVLlqhZs2bpXtOT3L17VytWrFC7du3UsmVL43J7e3uVLVtWn376qQ4fPmw8V6NHj1bz5s2Nbe3t7ZU7d27lzJnzueqwsrJS/vz5n6uPlCR3rQYHB2vfvn1asmSJnJycjMvt7e3l6emp3Llza8mSJWrcuLHJsWXJksWkTjs7O929e1c9evTQ77//rgoVKhjXVahQQdu3b9fu3bvl5eVlsv8VK1aoQoUK+uWXX9LzUAE8B+50A8BLyMrKStWrV092aGBYWFiyd6vj4uI0fvx4+fn5ydXVVdWrV9eYMWOMQ1z79u2r4OBgXbhwwWRo4pkzZ9SlSxdVrFhRZcuW1fvvv6+pU6c+cUinJO3Zs0eOjo5av3696tWrZ/KL3+zZs1WnTh2VLVtWnp6eatOmjclQ0kfb7tmzR1988YU8PDzk6empPn366N69eynu8/z586pcubJ69uxpvMOzfft2tWzZUhUrVlT58uXVtm1bRUZGGrdZvny5HB0d9csvv6hWrVpq1KhRsn07Ojrq0KFD2rt3r8mwzMOHD6tNmzaqXLmy3Nzc1KJFCx08eNBk2yNHjqhNmzYqX768XF1dVadOHS1cuNC4PiAgQEuWLDHp+9E52L59u0lf/x5C7OjoqGnTpql9+/ZycXHRiRMnJP3v761atWpydXVVw4YNtWXLlhTPnZR0eHlAQIDat2+vsLAw1apVS66urmrWrJkuXryodevWqXbt2nJ3d9d///tfXbx40diPj4+PBgwYoLlz56pGjRpycXFRo0aNdOTIEZP9bd26VY0bN5arq6vc3NzUrFkz7dixw7g+uWto4sSJ6tWrl/HYHz1mce3aNfXt21deXl4qW7asfHx8NGLECMXGxhr769u3r/z9/bVnzx41bNhQ5cqV0/vvv68VK1aY1HX69Gl16NBB5cuXl6enp4KCgnT27Fnj+qf9LKVkzpw5evjwoVq3bm2yfNGiRapVq5ZcXFz08ccf68iRI6pdu7bJIySp+TlM7tqYPXu2Jk6cqKpVqxr/rh4/liJFiuijjz7S5MmTlZCQ8MT601t8fLwSEhKSPW958+bVqlWrVLduXUnSsWPHdP78edWrV8+kXfXq1eXm5vZcdSQ3vHzz5s3y9fWVi4uL6tatq19++UVt2rRJdjRKaGioqlWrprJly6px48bGf1+S+zc1NjZWCxYsUKNGjUwC9+P69eunjRs3punDhH+3zZMnj9zd3ZP8H/Hw4UOtXr2aEU3AS4bQDQAvqXr16unAgQMmv0CfPHlSf/zxh/z8/JK0//rrrzVz5ky1atVKa9asUZ8+fbRkyRINGjRIktS/f3/VqlVLhQoVUnh4uAIDA2UwGNSuXTtdunRJs2fP1oYNG9StWzdNnjxZ8+bNS1WdISEh6tatmzHYhIWFafjw4WrRooU2btyoOXPmyNLSUu3atTMJSJI0YsQIeXl5acWKFfriiy8UFhamH3/8Mdn93LhxQ5999pnKli2rESNGyMLCQnv37lX79u1VoEABzZ8/X3PmzFFcXJxatmypGzdumGz//fff69tvv1VISEiy/YeHh+udd96Ru7u7wsPDVadOHZ05c0atWrVSQkKCpk+frkWLFqlQoUIKDAw0/uIdExOjTz/9VJkyZdLixYu1bt06NWvWTIMGDTKG4IkTJybpOy2WLFmid999V+vXr1eJEiX0999/q2XLloqKitJ3332nFStWyMPDQ506ddLu3bvT1PepU6e0detWff/99woJCdHx48fVrVs3rVy5UhMnTtSUKVMUERGR5PnR7du368iRI5o+fbrmzZunxMREtW/f3vhIxM6dO9WxY0c5OTlp6dKlWrRokQoWLKh27drpjz/+MOnr8WsoMDDQeKczPDxc/fv3lyR98cUX2r9/v6ZMmaJNmzZp0KBBWrZsmYKDg036unHjhiZNmqSvvvpKYWFhKlWqlAYMGKBLly5Jkm7evKn//ve/MhgM+uGHHzRnzhzduXNHgYGBxtEjT/tZSsmmTZvk6emp7NmzG5eFh4dr4MCBqlSpklasWKGOHTuqf//+Jtfn8/wcLly4UPfv39ecOXM0depUnThxQkOGDDFp4+Pjo2vXrunw4cNP7Cu95cqVS66urgoNDdXXX3+tI0eOpBj8jx07Znzdvn17VapUSR988IHmzJmT7BDq53Hq1Cl169ZNDg4OWrJkib766iuNHTvW5MO6R3766SddvXpVs2bN0rRp0xQdHa0BAwZISv7f1N9//1337t1T9erVU9y/lZXVU2s0GAw6deqUQkJC5O3trdKlSydpU7duXW3atEkxMTHGZb/++qtu376t2rVrp+ZUAHhBCN0A8JKqUqWK8ubNa3InY8WKFSpdunSSOyhXrlzR8uXL9dlnn6lx48ZycHBQnTp11LFjR61YsUJXrlxRjhw5lCVLFuNQy0fDjENDQxUSEqIyZcrIzs5OdevWVZkyZZKd7Cc5lStX1nvvvadChQpJ+ucX/NWrV6tFixYqUqSInJycFBAQoCtXrujkyZMm21aqVEkff/yxihYtqsaNG8ve3j7J3VJJun//vjp06KC8efNq/PjxypTpn6ejpk2bJjs7O40ePVr/+c9/5OLiorFjxyomJkaLFy826aNOnTry9PRMcZhp/vz5lSlTJmXOnFn58+eXjY2NZs+eLUtLS2NodnR01Lfffqts2bJp9uzZkiQbGxstW7ZMo0aN0n/+8x/Z29srICBA+fLlM57DXLlyJek7LXLkyKF27drJ3t5e1tbWWrJkif766y9NmDBBHh4eKlWqlL788kvjXfG0+OuvvzR06FD95z//UeXKleXp6akjR47o66+/VunSpeXl5SVPT08dPXrUZLt79+5p2LBhevvtt+Xq6qrevXvrxo0bxjvZM2fOVKlSpYz9ODo6atSoUcqePbvmz59v0tfj11C2bNmM8xXkz59fOXLkkPTPBzQ//PCD3N3dVbhwYVWvXl3e3t5JrtOrV69qwIABKl++vEqUKKE2bdooPj7eWP/y5cv1999/a/jw4XrnnXfk5OSkwYMHq3z58rp48WKqfpaSc/PmTZ08eVIeHh4my1esWKF8+fLp66+/1n/+8x/VqlVLPXr00O3bt03aPevPoa2trXr37q2SJUuqUqVK8vHxUUREhEmbR8OS9+3b98S+zGHSpEny8vLS/Pnz9cknn6hixYrq0KGDFi1aZPKIzPXr1yX98wx4vXr1FBoaKj8/Pw0fPlyhoaHpWtOjSRJHjRolJycnVapUSaNHjzZ+MPO4rFmzqm/fvipVqpQqV66sDz74wPhvVHL/pj66PgoXLpzmujZu3Ch3d3e5u7sb78DnzJlT3377bbLt69Spo4cPH2rt2rXGZStWrJC3t7dy586d5v0DMB+e6QaAl1SmTJlUp04dhYWFqXv37jIYDFq9erX++9//Jmn7+++/KzExUVWqVDFZ7uXlJYPBoKNHj6pgwYJJtrOwsNDt27f13Xff6fDhw7p586YMBoNiY2Pl4uKSqjrLli1r8j5r1qzavn27+vbtq4sXL+rBgwfGu1s3b940aVuuXDmT93ny5NGtW7dMliUkJOjzzz9XTEyMFixYYDKB3JEjR/TBBx+Y3DnKly+f3n777SQh8d91psaRI0dUrlw5Y/CT/nnusnz58sa7tZkyZdLly5c1YsQIHT9+3Fj//fv3kxzvs/p37UeOHJGDg4McHBxMlj+6m5oWDg4OJseXM2dO5c6d2/ghyqNl//7AxMXFxWRCtnfeeUeSdOHCBUlSRESEPvzwQ1lYWBjbWFtbq2zZss/0dxMfH69p06Zp7969unHjhhITExUXF5fkOWdbW1uTu4J58uSRJGPIPXLkiOzt7Y3LJalUqVLGWfJ//vnnZ/pZunbtmiQl+VAnKipKzs7Oxg+KJMnb21uZM2c2vn+en8N/D71O7mcoe/bsypo1q7HGF6lgwYKaNWuWTp8+re3bt2vfvn3at2+ftm7dqpCQEM2aNUvFixc3TvTWsmVL45DzMmXK6M8//9S0adPUunXrVN0hTo3z58/LwcHBZMi2o6OjihQpkqRtcuc3Pj5ecXFxyU6U9+h6//fjORs2bEjyjRTt27dXhw4djO+9vb315ZdfGre/cuWKli5dqvr16yskJETu7u4m2+fOnVtVq1bV8uXL1aRJE928eVNbtmzRyJEjU3EWALxIhG4AeInVr19fP/zwg8LDw2UwGHT9+nXjL6SPezS8MDAwUJaW/xvE9GhYZkq/bF+6dEktW7ZUsWLFNHDgQBUtWlSZMmVSz549jW0uXrxoMpy9SJEiJndWHg9skjRy5Ej9+OOPCgoKUq1atZQ9e3YdPnzY+Jzu42xtbU3ePx7QHlm8eLHu3btn/GX338cdFhZmUo8kPXjwIMkvxP+uMzViYmJ04sSJJL/sxsXFGUNbRESEAgMD5eHhoeHDh6tgwYKysrJK15nK33rrrSR1RUVFJakrPj7+iYEgOf+eBd/CwiJVfy//Pp+PtnkUbmNiYkyGWT+SLVs2kxmrk+vr3+7evauWLVsqc+bM6tWrl95++21lzpxZY8aMSfJ8/b9rf+TRz8KdO3eeOJncs/4sPTrufx/LzZs3k9z1tLa2NqkhNT+HKUnN39Wjuv59d/2RgQMHavXq1U/d15M8bfK7kiVLqmTJkmrdurXi4uK0bNkyDRs2TKNGjdKUKVOM5+3fH8B4eHho48aNunTpkuzt7Z+rxkdu3ryZ7DWQ3N3hf49KeXR+UxrybmdnJ+mfD1vKlCljXO7t7a2wsDDj+4CAgCT/ntna2qpYsWLG9yVKlFClSpXUvHlzffvtt1qyZEmS/dWvX1/du3dXZGSkdu/ercyZM/M8N/ASInQDwEvM1dVVJUqU0Lp16xQfH69333032bsxj+7YjBkzJtln/x6/q/e4zZs36969e/ruu+9UsmRJ4/Lbt28b+yxQoIDJL4uP37FLzurVq1WnTh117drVuOzfw13TomjRoho7dqzatGmj3r17a+bMmcZffN966y15e3urS5cuSbZLj69reuutt1SoUCENHTo0ybpHgWzt2rWytLTUlClTjCEzMTExyd3Gf0vpl/e7d+8+9Ry/9dZbKlq0aLJfOyQ9/e8oPfz76+wevX903eTIkcPkWdNHYmJi0vwByJ49e3T16lXNmDFDVatWNS5/0qR7KcmTJ0+S7z1+3LP+LD36YOTOnTsmy62trZPMZRAfH29y/lLzc/i87ty5k+TDm0e6deumNm3apMt+/u3WrVtJjsHa2lrNmjVTeHi4cYLF4sWLG9s/7tHPR3If4Dyr5P5OpJTDeFqUKVNGuXLl0saNG02eq/73Nwek5Wf0nXfeSfK4zCM+Pj7KkSOH1q1bpx07duj9999/pq+TBGBePNMNAC+5+vXrKzw8XNu3b08ys+8jZcuWlZWVlS5evKhixYoZ/+TPn1+WlpYmIefxkPfoTsvjQeLgwYM6e/assV2mTJlM+nx0Jyclj98FfuTRkOdnmRDJ29vbOPx39+7dJkHTzc1NkZGRJvUVK1ZMDx8+TJevCHJzc9OZM2dUuHBhk/4NBoMKFCgg6Z9zaG1tbRIK1q1bp9jY2CTH+/j7RwHo77//Ni67ffu2zpw5k6q6Ll26pOzZs5vUZWVlpbx585rcoTWXI0eOmASXR8PtS5QoIemfRwcOHDhgcswPHjzQ77//nupHFx5tm9x1Gh0drT179qT5mipdurSio6NNnt+Njo5Ws2bNtH///jT9LD3u0fV29epVk+XFihXTH3/8YTKB2JYtW0zucqbm5/B5xMTE6P79+yn+TOTNmzfJz1Ba/yRn9uzZqlSpUrLXtMFg0IULF4xD9T08PGRra6tNmzaZtNu/f78KFiz4TF+XlpJixYrp7NmzJgH/999/Nz4akVaP/x1lzpxZn376qdatW6ddu3Yl2/7KlStJPpx5ktOnTyf7SIP0z+MutWvX1rp163To0KEU/48AkLEI3QDwkqtfv76uX7+u+/fv68MPP0y2Tb58+fTxxx9r0qRJCgsLU1RUlA4fPqyuXbuqZcuWxgmL3nrrLV27dk379+9XVFSU8XnF77//XtHR0dq8ebO++eYb1axZU1FRUTpz5sxTvzrs39zd3bVx40YdPnxYkZGR6tu3r3FY6MGDB1Mc4vo0Hh4e6tChg8aPH2+chfmzzz7TiRMnNHjwYB0/flxnz57VtGnTVK9evXT5jtr//ve/unv3rr744gtFREQoKipKixcvVoMGDbRo0SJJ/wTgu3fvavbs2YqOjtby5cs1b948ubm56dSpU4qOjpb0z7k/e/asIiIidOnSJeMzpfPmzdPJkyd17Ngx9erVS/ny5XtqXQ0bNlTOnDnVtWtXHThwQNHR0Vq3bp0++eSTJLOMm4u1tbX69++vkydP6siRIxo1apQKFChgfBb6s88+0+nTpzV48GBFRkbq2LFj6tGjhx48ePDUofePPpDYvHmzTp8+rbJlyypTpkwKDQ1VVFSUdu3apU6dOsnX11c3b97U0aNHn/p1Xo80atRIuXPnVq9evXTy5EkdP35cgwYN0pUrV+Ts7Jzqn6V/y5Url0qXLq39+/ebLPf19dW1a9c0atQonTlzRlu3btX06dNNwrs5fg4ft3fvXkky+Z7nF8Hf318ODg769NNPtWTJEp04ccL4YUm3bt106tQpBQUFSfrnMYeOHTtq/vz5mjdvns6dO6fp06fr559/NrZ5kpiYGF27di3Jn0fPij/O19dX8fHx+uabb/Tnn39q7969GjRo0FM/UEzOv/9Nlf659t9//321b99e48aN09GjR3Xx4kUdOXJEkydPVv369ZUjR44kM4w/ePDApPYTJ05oxIgR2rFjh7p3755iDf7+/jp9+rTy5s2rypUrp/kYAJgfw8sB4CVnb2+vd999V2+99dYT7/YMHDhQBQoU0MSJE3X58mVly5ZN3t7e+vHHH43DDR8N6WzdurWaNWum/v3764svvtAPP/yghQsXGmf//vvvv9W5c2c1bdpUmzdvTtNw4EGDBumrr75Sq1atlDNnTjVr1kzt27fX33//rZkzZypTpkyqWLHiM52LTp06aefOnfr8888VFhYmDw8PzZgxQxMnTlSTJk2UmJgoR0dHjRs3TrVq1XqmfTyuWLFi+uGHHzRu3Dj997//VXx8vIoXL64+ffqoWbNmkiQ/Pz9FRETo+++/14QJE+Tp6ang4GAdOHBAX331lVq3bq3Nmzfr008/Ve/evdW8eXN9/vnn+vTTTzV69GiNGDFCjRo1UuHChdW5c2dt3br1qXfccuXKpfnz52vMmDHq0KGD7t27p8KFC6tVq1Zq27btcx93alSoUEEuLi5q3769rl27JkdHR02dOtU4bLZixYqaOnWqJk2apI8++khWVlYqV66c5s6dq1KlSj2x7/r162v16tXq3r27atasqUmTJmnYsGGaMGGC6tatq9KlS2vgwIHKnTu39u3bpxYtWiT7vGty8uTJox9++EEjRoxQkyZNZG1trfLly2vWrFnG4b+p+VlKznvvvafZs2ebPM9et25dnT9/XvPmzdPChQvl6uqq4cOHKyAgwPgIRPny5VP1c/istm7dqvz58yeZuNDccufOrQULFmju3LmaO3euLl++bJyfwd3dXfPnzzepqV27dsqSJYtmz56t4cOHq3DhwhoyZIg+/vjjp+5rzJgxxsnwHhcWFpbk3y93d3cNHTpUU6dOVcOGDfX222+rX79+Gj58eJofS0nu39RMmTJp/PjxWr9+vZYsWaKFCxfq7t27ypkzp0qXLq2uXbvq448/NpmIUPpnEr+ff/7Z5Pw5Ojrq+++/f+JXkFWoUEF2dnby8fFJt8nmAKQvC0N6f/khAAB4rfn4+KhcuXIaN25cRpfyUrl586bee+89BQYGGu/OGgwGXbt2Tfnz5zc+x3/r1i1VrFhRvXr10meffWbWmi5fvqz3339fffv2VYsWLcy6r1fJjRs3lCNHDuMs8g8fPlSVKlVUp06dp34fOwCkFcPLAQAA0kGuXLn0xRdfKDQ01DhaYdeuXapataq+++47nT9/XsePH1e/fv1ka2ub7DcRpLeRI0eqdOnSaty4sdn39aqIjIxU1apVNXDgQEVGRioyMlLffPONbt++naq76gCQVoRuAACAdNKsWTPj7P1xcXGqXLmyRo8erfDwcPn7+6tVq1a6d++eZs+ebfJd6Obw448/at++fZoyZYrJ94K/6UqVKqWQkBCdOXNGn3zyiZo0aaKTJ0/q+++/N37fPACkJ4aXAwAAAABgJtzpBgAAAADATAjdAAAAAACYCaEbAAAAAAAz4Xu6kWoPHz7UrVu3lCVLFlla8nkNAAAAgDdXYmKiHjx4oJw5cypTppSjNaEbqXbr1i2dPXs2o8sAAAAAgJdG8eLFlTdv3hTXE7qRalmyZJEkOTg4KFu2bBlcDV4HCQkJOnnypEqXLi0rK6uMLgevAa4pmAPXFdIb1xTSG9dUxrh//77Onj1rzEkpIXQj1R4NKbexsZGtrW0GV4PXQUJCgiTJ1taW/yCQLrimYA5cV0hvXFNIb1xTGetpj94SupFmnesO1MXTVzO6DAAAAACvuU1R8zK6hOfGbFgAAAAAAJgJoRsAAAAAADMhdAMAAAAAYCaEbgAAAAAAzITQDQAAAACAmTB7uRlER0erVq1asra2NlnevXt3tWnTJoOqAgAAAAC8aIRuM4qIiMjoEgAAAAAAGYjh5c9p2rRpqlmzpsqVK6fatWtr5cqVz93n4cOH1bhxY7m7u8vT01P9+/dXbGysJOn+/fsaMGCAPD09ValSJQ0YMEBxcXGSpAcPHmjo0KGqUaOGypUrpxYtWujYsWPGfh0dHTV79mx5e3tr2rRpkqRdu3apSZMmcnd3V9WqVTV58uTnrh8AAAAA8A9C93M4ePCg5s6dq3nz5unQoUMaMGCABg8erBs3bkiSevfuLW9vb1WqVEljx45VfHx8qvrt3bu3PvnkEx04cECrV6/WiRMntGjRIknSd999pz///FPr16/XunXr9McffxiD8rhx47Rv3z79+OOP2rNnj8qUKaP27dsbQ7kkbd68WWFhYWrbtq0uX76soKAgNWvWTPv379eMGTO0cOFCrV69Op3PFAAAAAC8mQjdz+HOnTuytLSUjY2NLCws5O3trQMHDqhQoUJyd3fX+++/r61bt2ratGlatWqVpkyZkqp+b9++LVtbW1laWqpAgQJavHixWrVqJYPBoLCwMAUGBipPnjzKkyePvv32W1WpUkWStHTpUrVv31729vaysbFR9+7dde3aNR08eNDYt6+vr/LlyycLCwutWbNGb7/9tho0aCArKys5OjqqadOm6XK3HgAAAADAM93PxcvLS2XKlJGPj4+8vLxUrVo1+fv7q0CBAlq4cKGxnaurq9q3b6/vv/9e3bp1e2q/n3/+ub788kvNnDlT3t7e8vf3V6lSpfT333/r9u3bsre3N7Z1cnKSJN26dUt37txRyZIljeuyZcumvHnz6sKFC8ZlRYoUMb4+f/68IiIi5OLiYlxmMBhUokSJZzshAAAAAAAT3Ol+DtbW1goJCdHChQtVtmxZzZs3T/7+/rpz506StnZ2drp+/boMBsNT+/3kk0+0bds2tWjRQn/++acaNGigzZs3y9Lyn7+uxMTEJNs8PoT83ywsLIyvraysjK9tbGxUvXp1RUREGP/8/vvvDC8HAAAAgHRC6H4O8fHxiomJkZOTkzp16qSwsDBZWFho586dmjp1qknb06dPy87OziQAp+Tvv/9W7ty51ahRI02ZMkXt27fX0qVLlStXLr311ls6c+aMse0ff/yhlStXKm/evMqWLZtOnz5tXHfr1i399ddfcnBwSHY/Dg4OOnnypMkHAdeuXXtigAcAAAAApB6h+zmEhoYaJySTpMjISN26dUt2dnaaPHmyVq5cqfj4eEVERGjmzJlq1qzZU/u8fPmyfHx8FB4ersTERN25c0cnT540BueGDRtqxowZunLliv7++28NGTJEp06dkqWlperWratp06bp8uXLunfvnsaMGaOiRYvK3d092X35+fnp5s2bmjJlimJjYxUVFaXAwEDNmTMn/U4SAAAAALzBeKb7OXz66ae6ePGiGjRooNjYWBUuXFg9e/ZU2bJlNW7cOE2aNEkDBw5Ujhw5FBAQoFatWj21z0KFCmnYsGEaNmyYLl68qOzZs6tatWrq2rWrJOmLL77Q0KFDVadOHVlbW+u9995T586dJUl9+/bVkCFD9MknnyguLk7u7u6aNWuWyZDyx+XOnVtTpkzRqFGjFBISojx58sjf31+BgYHpd5IAAAAA4A1mYUjNQ8aApHv37unYsWOa8PmPunj6akaXAwAAAOA1tylqXkaXkKJH+cjZ2Vm2trYptmN4OQAAAAAAZsLw8hdsyJAhWrx4cYrrO3bsqKCgoBdYEQAAAADAXBhejlR7NHyidOnSypEjR0aXg9dAQkKCDh06JDc3txTnHgDSgmsK5sB1hfTGNYX0xjWVMRheDgAAAABABiN0AwAAAABgJoRuAAAAAADMhNANAAAAAICZMHs50qx7s5G6dO6vjC7jjbAuYmpGlwAAAADgOXCnGwAAAAAAMyF0AwAAAABgJoRuAAAAAADMhNANAAAAAICZELrNaPXq1apUqZLatm2rsLAw+fj4pNh2zJgxCggIML4fNmyY3N3dNW3atOeqwcXFRTt27HiuPgAAAAAAz4bZy9PZxo0b5ejoqGLFimn69On66KOP1KdPH0lSgwYNUtXHzZs3NXfuXE2dOvWJQT01IiIinmt7AAAAAMCz4053OpswYYLOnTsnSYqJiZGDg0Oa+7h7964kqVixYulaGwAAAADgxSJ0p6P69evr1KlTCgoKkpOTky5cuKChQ4cqMDBQy5cvV5UqVYxtt2zZotq1a8vd3V3du3dXbGysJOnMmTOqXbu2JMnf319Tpkx56n63bdumevXqyd3dXd7e3ho9erQSExMlSY6Ojtq+fbskKSAgQFOmTFHnzp3l5uamunXr6vTp0xo6dKg8PDxUvXp1Y1sAAAAAwPMjdKejVatWSZKmTJmi48ePy87OTl999ZVCQ0NN2t2+fVs9evRQy5YttWfPHn300UcKCwuTJJUoUUI//fSTJGnlypUKCgp64j7j4+PVo0cP9evXTwcPHtSPP/6oDRs2aMuWLcm2X7x4sdq1a6fw8HBZWVkpMDBQZcqU0c6dO1WtWjWNHj36Oc8CAAAAAOARQncGCA8Pl62trVq0aCFra2tVr15dHh4ez9TXgwcPFBsbK1tbW1lYWKh48eLauHGj3nvvvWTbly9fXq6ursqePbsqVqyoTJkyqWHDhsY6zp49+xxHBgAAAAB4HKE7A1y+fFmFCxeWpeX/Tn/x4sWfqa/s2bOrU6dOatmypZo3b67JkyfrypUrKbYvVKiQ8XWWLFlUsGBB43tra2vFxcU9Ux0AAAAAgKQI3RkgLi5OCQkJJssePYP9LDp37qyff/5Zfn5+2r9/v+rUqaMjR44k2/bxoJ/cewAAAABA+iFxZYACBQroypUrMhgMxmWRkZHP3N/NmzdVsGBBtWjRQrNmzdKHH36olStXpkepAAAAAIDnQOhOZ1myZNG5c+cUExOTYpvKlSsrJiZGCxcuVFxcnDZv3qzDhw8/0/5+++03+fr66siRIzIYDPrrr7905syZZ/qqMgAAAABA+sqU0QW8bpo2bapRo0Zp586dKbYpVKiQxo4dqzFjxmjkyJGqVq2amjdvrt9++y3N+3N3d1fHjh3VvXt3Xb9+Xbly5ZKvr69atGjxPIcBAAAAAEgHFobHxzgDT3Dv3j0dO3ZMUwat0KVzf2V0OW+EdRFTM7oEs0pISNChQ4fk5uYmKyurjC4HrwGuKZgD1xXSG9cU0hvXVMZ4lI+cnZ1la2ubYjuGlwMAAAAAYCYML3/JdejQQTt27Ehx/ZAhQ9SgQYMXVxAAAAAAINUI3S+5kJCQjC4BAAAAAPCMCN1Is+AFfZQjR46MLgMAAAAAXno80w0AAAAAgJkQugEAAAAAMBNCNwAAAAAAZsIz3Uizz9tM0KXov826jzU7x5i1fwAAAAB4EbjTDQAAAACAmRC6AQAAAAAwE0I3AAAAAABmQugGAAAAAMBMCN0AAAAAAJgJofslMWXKFLVs2TKjywAAAAAApCNC9wuyceNGnTt3LsX1QUFB+vHHH19gRQAAAAAAcyN0vyATJkx4YugGAAAAALx+3tjQ7ejoqI0bN6pZs2Zyc3NTvXr1dPTo0VRtO3HiRLVv317du3dX+fLlJUmxsbH65ptvVKNGDbm5uSkgIEB//vmnJKl+/fo6deqUgoKC1K9fP0VHR8vR0VHz589XxYoVtWbNGk2cOFGNGzc27mPXrl1q0qSJ3N3dVbVqVU2ePFmS9Msvv8jNzU2xsbHGtjdu3FCZMmV06NAhSdKPP/4oX19flStXTn5+ftq8ebOxbUBAgEaPHq169eqpXbt2kqRp06apZs2aKleunGrXrq2VK1c++4kFAAAAABi9saFbkmbMmKFhw4Zp165dKlCggMaNG5fqbQ8dOqSKFStq3759kqQxY8bo6NGjWrRokXbv3i0XFxd17txZBoNBq1atkvTPc9vDhw839rF3715t2bJFfn5+Jn1fvnxZQUFBatasmfbv368ZM2Zo4cKFWr16tSpXrixra2uFh4cb22/ZskWFChWSm5ubNm7cqEmTJmn06NE6cOCAunXrpu7du+vixYvG9mvXrtWwYcP0/fff6+DBg5o7d67mzZunQ4cOacCAARo8eLD++uuvZzqnAAAAAID/eaNDt7+/v0qWLKmsWbPKx8dHkZGRqd7WyspKzZo1k5WVlRITE7V8+XIFBQWpYMGCsrGxMQbdI0eOpNhHgwYNlD17dllYWJgsX7Nmjd5++201aNBAVlZWcnR0VNOmTbVy5UplzpxZtWrV0s8//2xsv3nzZvn6+kqSli5dqo8//lhly5ZVpkyZ9MEHH+jdd9/VmjVrjO1dXV3l6uoqCwsL3blzR5aWlrKxsZGFhYW8vb114MAB5c2bN9XnAgAAAACQvEwZXUBGsre3N77OmjWrHjx4kOptCxUqZAzLf/31l+7evaugoCCTAJ2YmKhLly6pXLlyyfZRpEiRZJefP39eERERcnFxMS4zGAwqUaKEJOnDDz9Unz59lJCQoNjYWO3cuVNdu3Y1brtjxw7NmTPHZNv//Oc/xvd2dnbG115eXipTpox8fHzk5eWlatWqyd/fX7a2tqk+FwAAAACA5L3Rofvfd5jTIlOm/506GxsbSdLChQtVtmzZVPdhZWWV7HIbGxtVr15dISEhya6vXLmyEhMTdeDAAV2/fl2FCxdWmTJljNt+8cUXCgwMTNV+ra2tFRISouPHj+vnn3/WvHnzFBoaquXLlytHjhypPhYAAAAAQFJv9PDy9JIjRw7lypVLJ06cMFkeHR39TP05ODjo5MmTMhgMxmXXrl1TXFycJBmHmG/dulWbNm1SnTp1TLb9dx0XL1406etx8fHxiomJkZOTkzp16qSwsDBZWFho586dz1Q7AAAAAOB/CN3ppGnTppo6daoiIyMVHx+v2bNn6+OPP9b9+/clSVmyZNG5c+cUExPz1L78/Px08+ZNTZkyRbGxsYqKilJgYKDJkHFfX1+Fh4crPDzcJHQ3adJE69at07Zt2/Tw4UPt3r1bdevW1eHDh5PdV2hoqNq2bavLly9LkiIjI3Xr1i05ODg8z+kAAAAAAOgNH16enoKCgnT79m01b95c8fHxcnZ21vTp05U1a1ZJ/4TyUaNGaefOnerfv/8T+8qdO7emTJmiUaNGKSQkRHny5JG/v7/JkHEvLy9dvXpVhQoV0ttvv21cXqVKFfXp00fffPONrl+/Lnt7ew0ePFhubm7J7uvTTz/VxYsX1aBBA8XGxqpw4cLq2bOnnJ2dn/+kAAAAAMAbzsKQ0rhj4F/u3bunY8eOKWTkT7oU/bdZ97Vm5xiz9o+XQ0JCgg4dOiQ3N7cU5zgA0oJrCubAdYX0xjWF9MY1lTEe5SNnZ+cnTkTN8HIAAAAAAMyE4eX/MnPmTAUHB6e43t/fX0OHDn1xBQEAAAAAXlmE7n9p06aN2rRpk9FlvNS+m9mVrxMDAAAAgFRgeDkAAAAAAGZC6AYAAAAAwEwI3QAAAAAAmAmhGwAAAAAAM2EiNaRZj27f69LFW2nebu1PQ8xQDQAAAAC8vLjTDQAAAACAmRC6AQAAAAAwE0I3AAAAAABmQugGAAAAAMBMCN1mZjAY1LVrV7m5uWnNmjUKDAxUcHBwiu2rVKmi5cuXS5KuXLmihg0bqly5crp06dIz1xAWFiYfH59n3h4AAAAA8GyYvTwVNm7cKEdHRxUrVuypbRMSEjR37lx9+umnkqRjx45pw4YNWrVqlRwdHVW3bt1U73f9+vX666+/tGfPHtnY2Dxz/Q0aNFCDBg2eeXsAAAAAwLPhTncqTJgwQefOnUtV26NHj2rGjBnG9zExMZKk4sWLp3m/MTExKliw4HMFbgAAAABAxnlpQ7ejo6Nmz54tb29vTZs2TZK0a9cuNWnSRO7u7qpataomT55ssk1oaKhq1qyp8uXLq02bNoqOjjau+/HHH+Xr66ty5crJz89PmzdvNq4LCAhQSEiIevXqpfLly6tq1apauXKlJKl+/fo6deqUgoKC1K9fvyfWfOTIETVt2lTXr1+Xi4uLduzYocDAQEmSh4eHwsLCFBAQoDFjxkiSHj58qCFDhsjT01NVq1bVkiVLjH0FBwdrypQpOnLkiFxcXHThwoUn7jsxMVEjRoyQt7e33NzcVL9+ff3666+SpOXLl6tKlSqSpOjoaDk6Omrbtm3G89GvXz+dP39eTZs2lZubmwICAnTrVtq/hxsAAAAAYOqlDd2StHnzZoWFhalt27a6fPmygoKC1KxZM+3fv18zZszQwoULtXr1amPb6dOna+rUqdq9e7cKFy6snj17SvpnePikSZM0evRoHThwQN26dVP37t118eJF477mzZun+vXra8+ePWrcuLG++eYbxcfHa9WqVZKkKVOmaPjw4U+s19XVVUOGDFG+fPkUERGhKlWqaObMmZKk/fv3JxnivWzZMv3000+aP3++NmzYoN9//90Ydrt3766OHTvK1dVVERERsrOze+K+165dq507d2rVqlU6cOCAWrVqpT59+ig+Pj7Z9mFhYVq8eLFmzJih5cuXq2/fvho9erQ2bdqkM2fOaNmyZU/cHwAAAADg6V7q0O3r66t8+fLJwsJCa9as0dtvv60GDRrIyspKjo6Oatq0qfGO9LJly+Tn5ycnJydZW1urR48eatWqlRITE7V06VJ9/PHHKlu2rDJlyqQPPvhA7777rtasWWPc16O755kzZ5avr69iYmJ09epVsx7fpk2bVK9ePZUqVUq2trbq1q2bHj58+Ex93b59W5kyZVLWrFllZWWlRo0aKTw8XJkzZ062faNGjZQjRw5VqFBBOXLkUJUqVVS0aFHlz59frq6uOnv27HMcGQAAAABAesknUitSpIjx9fnz5xURESEXFxfjMoPBoBIlSkiSoqKi5OnpaVyXN29e+fr6GrfdsWOH5syZY7Ltf/7zH+N7e3t74+tHz1DHxsam8xGZunLlimrUqGF8nydPHuXMmfOZ+vLz89PKlStVrVo1ValSRTVq1JCfn58sLZP/XKVw4cLG11myZFHBggVN3sfFxT1THQAAAACA/3mpQ7eVlZXxtY2NjapXr66QkJBk21pYWMhgMCS7zsbGRl988YXx+erkpBROzSkuLi7Jne3ExMRn6itXrlxavHixDh48qK1bt2rChAlasGCB5s2bl2x7CwsLk/cZcfwAAAAA8Lp7ZZKWg4ODTp48aRKsr127ZrwjW7RoUZ05c8a47saNGwoNDVV8fLwcHBx04sQJk/4uXryYYkh/UQoUKKDLly8b31+9elW3b99+pr4ePHig+/fvq3z58vriiy+0Zs0anTx5UsePH0+vcgEAAAAAafTKhG4/Pz/dvHlTU6ZMUWxsrKKiohQYGGgcMt6oUSOtXbtWhw8fVlxcnCZPnqyffvpJmTNnVpMmTbRu3Tpt27ZNDx8+1O7du1W3bl0dPnw4VfvOkiWLzp07Z/z6ryexsbHRnTt3dOXKlacOT69atarWrFmjs2fPKiYmRuPGjVOWLFlSVdO/DRs2TH369NGNGzdkMBj0xx9/KDEx0WSIPgAAAADgxXqph5c/Lnfu3JoyZYpGjRqlkJAQ5cmTR/7+/sYh47Vq1VKPHj3UqVMn3bt3T+7u7ho7dqwkqUqVKurTp4+++eYbXb9+Xfb29ho8eLDc3NxSte+mTZtq1KhR2rlzp6ZOnfrEtpUqVZK9vb3ee+89jRw5Unnz5k2xbevWrRUVFaXGjRvL2tpaXbt21YEDB1J3Qv7liy++0KBBg1S7dm09fPhQxYoV09ixY5UnT55n6g8AAAAA8PwsDBk9xhqvjHv37unYsWOaOnmrLl1M+/d4r/1piBmqwqssISFBhw4dkpubm8kcDsCz4pqCOXBdIb1xTSG9cU1ljEf5yNnZWba2tim2e2WGlwMAAAAA8Kp5ZYaXvwzWr1+v3r17p7i+QoUKCg0NTff9Xr9+XTVr1nxim4iIiHTfLwAAAADg+RC608DX19f43d8vUr58+V6qUD1ufHvlyJEjo8sAAAAAgJcew8sBAAAAADATQjcAAAAAAGZC6AYAAAAAwEwI3QAAAAAAmAmhG2nWrc+MjC4BAAAAAF4JhG4AAAAAAMyE0A0AAAAAgJkQugEAAAAAMBNCNwAAAAAAZkLofolduHBBLi4uOnPmTEaXAgAAAAB4BoTul5idnZ0iIiJUokQJs+3DYDBo0qRJqlmzptzd3eXn56ewsDCz7Q8AAAAA3iSZMroAZKw5c+YoLCxMM2fOVLFixbRp0yb16NFDpUuXVpkyZTK6PAAAAAB4pXGn+yUWHR0tR0dHRUZGysfHR0uWLFG7du3k7u6u9957T+Hh4anq5/r16+rUqZM8PT1Vvnx5tW7dWlFRUZIkJycnjR07ViVLlpSVlZU+/PBD5ciRQ3/++ac5Dw0AAAAA3giE7lfIzJkz1blzZ+3Zs0cVK1bUt99+m6rtxo8fr5w5c2r79u0KDw+Xg4ODRo4cKUmqVKmSypUrJ0mKjY3Vjz/+KEtLS3l5eZntOAAAAADgTcHw8ldIzZo15erqKkmqXbu2wsLClJiYKEvLJ392cvv2beXKlUvW1taysLDQ4MGDk2zz1VdfaenSpSpSpIgmT56s/Pnzm+04AAAAAOBNwZ3uV4i9vb3xtY2NjRISEhQfH//U7T777DP9/PPPqlWrlgYOHKg9e/YkaTN06FAdOnRInTp1UocOHXT06NF0rR0AAAAA3kSE7lfI0+5op8TFxUVbtmxR//79ZTAY1LlzZ+Pw8sfZ2NioUaNGcnV11dKlS5+3XAAAAAB44xG63wA3b95U5syZVatWLQ0ZMkRTp07VwoULJUkdOnTQvHnzTNpbWFgoUyaePAAAAACA50XofgM0bdpU06dP14MHDxQfH6/Dhw+rWLFikqTy5ctr2rRpOnr0qB4+fKgtW7Zo165dqlmzZgZXDQAAAACvPm5nvgGCg4P19ddfa+rUqcqUKZNcXFw0ZswYSVKbNm0UHx+vdu3a6c6dO7K3t9fQoUOZvRwAAAAA0gGh+yVmb2+vEydOSJK2bNliss7T09O47mmcnJy0YMGCZNdZWVmpU6dO6tSp0/MVCwAAAABIguHlAAAAAACYCXe6X3Hr169X7969U1xfoUIFhYaGvsCKAAAAAACPELpfcb6+vvL19c3oMgAAAAAAyWB4OdJs/MjPMroEAAAAAHglELoBAAAAADATQjcAAAAAAGZC6AYAAAAAwEwI3QAAAAAAmAmhGwAAAAAAMyF0AwAAAABgJoRuAAAAAADMhNANAAAAAICZELoBAAAAADATQvcrwsfHRwsWLMjoMgAAAAAAaUDofsMdP35crVu3loeHh6pVq6Zhw4YpLi4uo8sCAAAAgNcCofsNdvfuXX322WcqV66cdu7cqVmzZunnn3/WjBkzMro0AAAAAHgtELrNxNHRUbNnz5a3t7emTZsmSVq1apXq1Kkjd3d3+fj4aP78+cb2EydOVMeOHTV9+nRVqVJFFSpU0NChQ5PtOy4uTs2bN1ffvn1TVcu0adNUs2ZNlStXTrVr19bKlSslSX/99ZeqVq2qLl26yNraWqVKlVLt2rW1f//+5zx6AAAAAIAkZcroAl5nmzdvVlhYmPLmzauoqCj16dNHM2fOlJeXl3bv3q3AwECVL19eTk5OkqSDBw/K1dVVW7du1YEDB9S6dWvVr19frq6uJv0OGjRI1tbWGjJkyFNrOHjwoObOnavFixercOHC2rFjh7p06SJvb285ODho+PDhJu0vXbqkggULpt9JAAAAAIA3GKHbjHx9fZUvXz5Jkr29vXbv3q2cOXNKkry8vJQ3b1798ccfxtBtZWWl9u3by9LSUl5eXsqTJ48iIyNNQvfMmTMVERGhBQsWKHPmzE+t4c6dO7K0tJSNjY0sLCzk7e2tAwcOyNIy6SCHn3/+WVu3btXSpUvT4/ABAAAA4I1H6DajIkWKGF9bWFhowYIFWrp0qa5evSqDwaC4uDiTScuKFCliEoazZs2q2NhY4/vt27dr27ZtmjlzpnLkyJGqGry8vFSmTBn5+PjIy8tL1apVk7+/v2xtbU3abdy4UX369NGoUaP09ttvP+shAwAAAAAewzPdZmRlZWV8vWTJEk2bNk1Dhw7Vb7/9poiICBUqVMikfXJ3nx/322+/qXr16ho3bpwSEhJSVYO1tbVCQkK0cOFClS1bVvPmzZO/v7/u3LljbLNo0SL1799fEydOVO3atdNwhAAAAACAJyF0vyARERHy8PBQpUqVZGVlpWvXrunq1atp6qNLly4aO3asbty4oZCQkFRtEx8fr5iYGDk5OalTp04KCwuThYWFdu7cKUn66aefNG7cOM2dO1fe3t5pPi4AAAAAQMoI3S+InZ2dTp8+rVu3bunChQsaOnSoihQpoitXrqS6D0tLS2XLlk3Dhw9XSEiIjh49+tRtQkND1bZtW12+fFmSFBkZqVu3bsnBwUF37tzR4MGDNXr0aDk7Oz/zsQEAAAAAkscz3S9Is2bNtHfvXlWvXl12dnYaPHiwfv/9dwUHByt//vxp6qtixYpq1qyZevfureXLl8va2jrFtp9++qkuXryoBg0aKDY2VoULF1bPnj3l7OyssLAw/f333woKCkqyXURERJqPEQAAAABgysJgMBgyugi8Gu7du6djx46pdOnSqZ7IDXiShIQEHTp0SG5ubiZzIADPimsK5sB1hfTGNYX0xjWVMR7lI2dn5yQTVT+O4eUAAAAAAJgJw8tfcR4eHnrw4EGK63/66SfZ2dm9wIoAAAAAAI8Qul9x+/fvz+gSAAAAAAApYHg5AAAAAABmQugGAAAAAMBMCN0AAAAAAJgJoRsAAAAAADMhdAMAAAAAYCaEbgAAAAAAzITQDQAAAACAmRC6AQAAAAAwE0I3AAAAAABmQuhOhb59+6pHjx6SpClTpqhly5ZP3Wb58uWqUqWKuUsDAAAAALzECN1pFBQUpB9//DGjyzCbWbNm6eHDhxldBgAAAAC8FgjdMLpx44ZGjhyphISEjC4FAAAAAF4Lr1Xojo6OlqOjoyIjI43LxowZo4CAgFRtHxoaqpo1a6p8+fJq06aNoqOjk7SZOHGiGjdubHwfHh6u+vXry83NTf7+/tq1a1eyfW/dulUVKlTQ8ePHn7uWhQsXytfXV+XKldOHH36odevWGdf5+PhowYIFxvfbt2+Xo6Oj8b2jo6M2btyoZs2ayc3NTfXq1dPRo0d1/fp1VatWTQaDQR4eHlq+fHmq6gQAAAAApOy1Ct3PY/PmzZo+fbqmTp2q3bt3q3DhwurZs+cTt7ly5Yq6dOmiDh06aN++fWrVqpU6deqkmzdvmrQ7efKk+vTpo3HjxsnJyem5atmyZYtGjx6tIUOGaP/+/eratat69eqlEydOpPpYZ8yYoWHDhmnXrl0qUKCAxo0bp3z58mnmzJmSpP3796thw4ap7g8AAAAAkLxMGV3Ay2LZsmXy8/MzhuIePXpo7969SkxMTHGb9evXq2jRoqpTp44kqWHDhsqSJYvJNjdu3FDHjh3Vq1cveXt7P3ctS5cuVd26deXh4SFJqlOnjkJDQ7VhwwaTO9pP4u/vr5IlS0r65874o7ANAAAAAEhf3On+f1FRUbK3tze+z5s3r3x9fWVpmfIpOn/+vMk2kuTn56c8efJIkh4+fKiuXbuqQIEC+uSTT9KllujoaJUqVcqkfbFixXThwoVU9/9431mzZtWDBw9SvS0AAAAAIPVe+9Cd2knBLCwsZDAY0tS3paXlE++E37x5U/nz59fx48e1ZcuWVPf7pFri4uJS3CY5ydWXUlsAAAAAQPp6rUJ3lixZJEmxsbHGZVFRUanatmjRojpz5ozx/Y0bNxQaGqr4+PgUt7G3tzfZRpJ+/PFH4z7z5MmjcePGqWfPnhowYIBu3Ljx3LU4ODjo9OnTJu1Pnz6tokWLSpKsra1Njv/8+fOp2icAAAAAIP29VqE7T548ypEjhzZu3KiEhASFh4fr0KFDqdq2UaNGWrt2rQ4fPqy4uDhNnjxZP/30kzJnzpziNnXr1tWlS5e0ePFixcXFae3atfruu++ULVs2STIOTW/evLnefvttDR48+Llr8ff31+rVq3Xo0CHFx8dr+fLlOnXqlPz8/CRJxYsX17Zt2xQbG6tz585p9erVqdqnJNnY2EiSzpw5o3v37qV6OwAAAABA8l6r0G1lZaVBgwZpxYoV8vDwUFhYmFq0aJGqbWvVqqUePXqoU6dOqlSpks6ePauxY8c+cZtHM37Pnj1bFSpU0LRp0zR58mTjM92PWFhY6Ntvv9WOHTu0cuXK56rFz89P7du3V+/eveXp6an58+crNDRUxYsXlyR1795dN27ckKenp/r06aM2bdqk6vglydnZWe7u7vr4449NvnYMAAAAAPBsLAxpfZAZb6x79+7p2LFjKl26tHLkyJHR5eA1kJCQoEOHDsnNzU1WVlYZXQ5eA1xTMAeuK6Q3rimkN66pjPEoHzk7O8vW1jbFdq/VnW4AAAAAAF4mb8T3dM+cOVPBwcEprvf399fQoUPfuFoAAAAAAOb1RoTuNm3apOnZZnN6mWoBAAAAAJgXw8sBAAAAADATQjcAAAAAAGZC6AYAAAAAwEwI3QAAAAAAmAmhGwAAAAAAMyF0AwAAAABgJoRuAAAAAADMhNANAAAAAICZELrTICAgQGPGjMnQGpYvX64qVapIkvbt2ycXFxfFxcU9cZvo6Gg5OjoqMjLyRZQIAAAAAPh/mTK6ADy7ChUqKCIiIqPLAAAAAACkgDvdAAAAAACYyWsfuh0dHbV27Vo1bNhQrq6uateunS5fvqw2bdrI3d1dDRs2VHR09DP1vXDhQvn6+qpcuXL68MMPtW7dOuO6GzduqFWrVnJ1dZW/v79++eUXOTo6pnpf4eHhql+/vtzc3OTv769du3YlabNnzx45OjrqwYMHkqSoqCgFBgbK3d1dNWvW1Ny5c5Pt+8KFC6pcubKWLVsmSZo2bZpq1qypcuXKqXbt2lq5cmVaTwUAAAAAIBmvfeiW/gnHISEhWrVqlXbt2qW2bdvqiy++0K+//qqEhATNmjUrzX1u2bJFo0eP1pAhQ7R//3517dpVvXr10okTJyRJ/fv3V3x8vLZv367g4GCNHz8+1X1fuXJFXbp0UYcOHbRv3z61atVKnTp10s2bN5+4XefOnVWqVCnt3LlTU6ZMUXBwsHbs2GHS5u7du+rQoYOaNGmiRo0a6eDBg5o7d67mzZunQ4cOacCAARo8eLD++uuvNJ8TAAAAAICpNyJ0+/n5qUCBAipevLhKliwpFxcXlSlTRtmzZ1fFihV19uzZNPe5dOlS1a1bVx4eHsqcObPq1KkjZ2dnbdiwQYmJifr1118VGBioXLlyqUSJEmrSpEmq+16/fr2KFi2qOnXqKHPmzGrYsKGGDBmixMTEFLc5evSoTpw4oU6dOilr1qxydnbWpEmTVKhQIWMbg8Ggnj17ysnJSd26dZMk3blzR5aWlrKxsZGFhYW8vb114MAB5c2bN83nBAAAAABg6o2YSK1w4cLG11myZFHBggVN3j9t9u/kREdHq1KlSibLihUrpgsXLujmzZuKj4+XnZ2dcZ2Li0uq+z5//rzs7e1Nlvn5+T11m+zZsytXrlzGZZUrVzbWKknBwcHauXOnyd1vLy8vlSlTRj4+PvLy8lK1atXk7+8vW1vbVNcLAAAAAEjeG3Gn28LCwuS9peXzH3ZKQd3CwkIGg0GSlCnT/z7TSMs+LS0tn3hX+1m3uXz5shwcHDRp0iTjMmtra4WEhGjhwoUqW7as5s2bJ39/f925cydN+wcAAAAAJPVGhG5zcHBw0OnTp02WnT59WkWLFlWuXLlkZWWlixcvGtel5au97O3tdebMGZNlP/74o6KiolLcpmjRorp7966uXr1qXLZ582bt3bvX+H748OEaNWqU5s2bp3379kmS4uPjFRMTIycnJ3Xq1ElhYWGysLDQzp07U10vAAAAACB5hO5n5O/vr9WrV+vQoUOKj4/X8uXLderUKfn5+cnKykoeHh6aNWuW7ty5ozNnzmjJkiWp7rtu3bq6dOmSFi9erLi4OK1du1bfffedsmXLluI2zs7OKlOmjIKDg3X37l2dPHlS/fv3V2xsrLGNpaWlnJ2d1aFDB/Xp00cxMTEKDQ1V27ZtdfnyZUlSZGSkbt26JQcHh2c/OQAAAAAASW/IM93m4OfnpwsXLqh37966fv26SpYsqdDQUBUvXlySNGzYMHXr1k1VqlRRmTJl1L59ewUFBaVqmHm+fPk0c+ZMDRo0SMOGDVPx4sU1efJk5cmT54nbhYSEqHfv3qpcubLy5s2roKAgVatWLcnXlLVv315bt27V8OHDNWjQIF28eFENGjRQbGysChcurJ49e8rZ2fmZzw0AAAAA4B8WhkcPICPdxcXFydraWpK0e/duffrppzp8+LBx2avm3r17OnbsmEqXLq0cOXJkdDl4DSQkJOjQoUNyc3OTlZVVRpeD1wDXFMyB6wrpjWsK6Y1rKmM8ykfOzs5PnIia4eVm8uWXX6pt27a6ffu27ty5o1mzZqly5cqvbOAGAAAAAKQdw8v/35AhQ7R48eIU13fs2FFBQUGp7q9Xr14aNGiQ3nvvPVlYWOjdd9/V119/rSNHjqhFixYpblekSBFt2LAhTbUDAAAAAF5OhO7/N2DAAA0YMCDd+sudO7cmTJiQZHmhQoXSNJM5AAAAAODVxfByAAAAAADMhNANAAAAAICZELoBAAAAADATQjcAAAAAAGZC6AYAAAAAwEwI3QAAAAAAmAmhGwAAAAAAMyF0AwAAAABgJoRuAAAAAADMhNANAAAAAICZELrTSUBAgMaMGWP2/YSFhcnHx8fs+wEAAAAAPD9C9yumQYMG2rJlS0aXAQAAAABIBUI3AAAAAABm8kaHbkdHR61du1YNGzaUq6ur2rVrp8uXL6tNmzZyd3dXw4YNFR0d/Ux9L1y4UL6+vipXrpw+/PBDrVu3zrjuxo0batWqlVxdXeXv769ffvlFjo6OqdrX8uXLVaVKFUlSdHS0HB0dtW3bNuO++vXrp/Pnz6tp06Zyc3NTQECAbt26JUnq27ev+vXrp2+++Ubly5dXpUqVNH/+fGPfAQEBGj16tOrVq6d27do903EDAAAAAP4nU0YXkNEWLlyokJAQ3bt3T/Xq1VPbtm01cuRIOTg4qEWLFpo1a5YGDBiQpj63bNmi0aNH6/vvv1e5cuW0adMm9erVS6VKlZKjo6P69++v+Ph4bd++XX///be++OKL5zqGsLAwLV68WMePH1fLli117tw5jR49WjY2Nvroo4+0bNkyBQYGSpJ++ukn9evXT7t379b27dvVuXNnlS9fXk5OTpKktWvXasKECXJxcXmumgAAAAAAb/idbkny8/NTgQIFVLx4cZUsWVIuLi4qU6aMsmfProoVK+rs2bNp7nPp0qWqW7euPDw8lDlzZtWpU0fOzs7asGGDEhMT9euvvyowMFC5cuVSiRIl1KRJk+c6hkaNGilHjhyqUKGCcuTIoSpVqqho0aLKnz+/XF1dTY6hSJEiaty4saytrfXee+/J2dlZW7duNa53dXWVq6urLCwsnqsmAAAAAAChW4ULFza+zpIliwoWLGjyPi4uLs19RkdHq1SpUibLihUrpgsXLujmzZuKj4+XnZ2dcd3z3lVOyzGUKFHCZNsiRYro6tWrxveP1wUAAAAAeD5vfOj+9x1dS8vnPyUpBXULCwsZDAZJUqZM/xvZ/7z7TMsxJCQkmLw3GAwm21tZWT1XLQAAAACA/3njQ7c5ODg46PTp0ybLTp8+raJFiypXrlyysrLSxYsXjesiIiJeWG1RUVEm7y9evKhChQq9sP0DAAAAwJuE0G0G/v7+Wr16tQ4dOqT4+HgtX75cp06dkp+fn6ysrOTh4aFZs2bpzp07OnPmjJYsWfLCartw4YLCwsIUHx+vTZs26fjx46pRo8YL2z8AAAAAvEne+NnLzcHPz08XLlxQ7969df36dZUsWVKhoaEqXry4JGnYsGHq1q2bqlSpojJlyqh9+/YKCgpKl6HtT1OtWjX99ttvGjJkiDJnzqzBgwerdOnSZt8vAAAAALyJLAyPHjLGCxUXFydra2tJ0u7du/Xpp5/q8OHDxmXm0LdvXz148EDjxo17pu3v3bunY8eOqXTp0sqRI0c6V4c3UUJCgg4dOiQ3NzfmE0C64JqCOXBdIb1xTSG9cU1ljEf5yNnZWba2tim2Y3h5Bvjyyy/Vtm1b3b59W3fu3NGsWbNUuXJlswZuAAAAAMCLx/DyVBgyZIgWL16c4vqOHTsqKCgo1f316tVLgwYN0nvvvScLCwu9++67+vrrr3XkyBG1aNEixe2KFCmiDRs2pKl2AAAAAEDGIXSnwoABAzRgwIB06y937tyaMGFCkuWFChUy60zmI0aMMFvfAAAAAICkGF4OAAAAAICZELoBAAAAADATQjcAAAAAAGZC6AYAAAAAwEwI3QAAAAAAmAmhGwAAAAAAMyF0AwAAAABgJoRuAAAAAADMhNANAAAAAICZELoBAAAAADATQvdL7MKFC3JxcdGZM2cyuhQAAAAAwDMgdL/E7OzsFBERoRIlSph1PwsWLFDt2rXl7u4uf39/bd682az7AwAAAIA3BaH7DbdhwwaNHTtW3377rfbu3auWLVuqe/fuioqKyujSAAAAAOCVR+h+iUVHR8vR0VGRkZHy8fHRkiVL1K5dO7m7u+u9995TeHh4qvq5fv26OnXqJE9PT5UvX16tW7c2hurY2Fh9/vnnevfdd5U5c2Z98sknypYtmw4dOmTGIwMAAACANwOh+xUyc+ZMde7cWXv27FHFihX17bffpmq78ePHK2fOnNq+fbvCw8Pl4OCgkSNHSpL8/f3VvHlzY9vbt2/r7t27KliwoFmOAQAAAADeJJkyugCkXs2aNeXq6ipJql27tsLCwpSYmChLyyd/dnL79m3lypVL1tbWsrCw0ODBg5PdxmAw6KuvvlK5cuVUsWJFsxwDAAAAALxJuNP9CrG3tze+trGxUUJCguLj45+63Weffaaff/5ZtWrV0sCBA7Vnz54kbeLj49WzZ0/9+eefGj9+fLrWDQAAAABvKkL3K+Rpd7RT4uLioi1btqh///4yGAzq3LmzcXi59M9z3e3bt9fFixc1b9485cuXL71KBgAAAIA3GqH7DXDz5k1lzpxZtWrV0pAhQzR16lQtXLhQ0j9Dynv06KFMmTJp9uzZyp07dwZXCwAAAACvD0L3G6Bp06aaPn26Hjx4oPj4eB0+fFjFihWTJK1evdo4pDxLliwZXCkAAAAAvF6eKXSvXbtWbdu2VYMGDSRJcXFxmjlzpgwGQ3rWhnQSHBysrVu3qlKlSqpcubJ27dqlMWPGSJKWLVumCxcuqGLFinJxcTH++eqrrzK4agAAAAB49aV59vIpU6Zo0aJFatKkiUJCQiT9Mzt2WFiY7ty5o+7du6d3jW8se3t7nThxQpK0ZcsWk3Wenp7GdU/j5OSkBQsWJLtuzpw5z1ckAAAAACBFab7TvWjRIs2YMUNBQUGysLCQJOXLl09TpkzRypUr071AAAAAAABeVWm+033nzh29/fbbSZYXKFBAN27cSJeikHrr169X7969U1xfoUIFhYaGvsCKAAAAAACPpDl0ly5dWqtWrVL9+vVNloeGhqpUqVLpVhhSx9fXV76+vhldBgAAAAAgGWkO3d26dVOnTp00f/58xcfHq2PHjjp58qRu3bqlKVOmmKNGAAAAAABeSWkO3V5eXlq/fr3WrFkjR0dH2djYyNvbW35+fsqVK5cZSgQAAAAA4NWU5tA9ffp0tW3bVm3atDFHPQAAAAAAvDbSPHv5nDlzmDANAAAAAIBUSPOd7s8++0zdunVTnTp1VKRIEVlZWZms9/b2TrfiAAAAAAB4laU5dI8YMUKStG/fviTrLCwsdOzYseevCgAAAACA10CaQ/fx48fNUQcAAAAAAK+dND/TDQAAAAAAUifNd7qdnJxkYWGR4nqGlwMAAAAA8I9n+sqwxyUmJurcuXNas2aNPvvss3QrDC+Gj4+P2rZtq2bNmmV0KQAAAADw2klz6K5atWqyy6tXr66+ffvqgw8+eO6insfGjRvl6OioYsWKZWgdAAAAAACk2zPdhQoVeikmWZswYYLOnTuX0WUAAAAAAJD20L1o0aIkf2bPnq0OHTrIwcHBHDUmKyoqSoGBgXJ3d1fNmjU1d+5c1a9fX6dOnVJQUJD69eun6OhoOTo6av78+apYsaLWrFnz1H779u2rIUOGaPjw4apYsaIqVapkMqT+1q1b6t27t7y9veXu7q527dopOjpakpLd3/Lly1WvXj0tWrRIVapUUcWKFTV//nz98ssv+uCDD1S+fHkNGjQo1cft6Oio2bNny9vbW9OmTZMkrVq1SnXq1JG7u7t8fHw0f/58Y/uJEyeqY8eOmj59uqpUqaIKFSpo6NChyfYdFxen5s2bq2/fvqmuBwAAAACQsjQPL//++++TLMuSJYuKFSumkSNHpktRqdG5c2dVrFhRkydP1tmzZ9WiRQtNnDhRgYGBmjJliqpVq2YMw3v37tWWLVuULVu2VPW9Zs0a9e3bVzt27NCqVas0YMAA+fv7q0CBAvrqq68UExOjVatWydraWl9++aW6d++upUuXGrd/fH8rVqzQhQsXdOXKFW3dulUzZszQ6NGjVatWLa1YsUJ//PGHAgIC9Mknn6hs2bKpqm/z5s0KCwtT3rx5FRUVpT59+mjmzJny8vLS7t27FRgYqPLly8vJyUmSdPDgQbm6umrr1q06cOCAWrdurfr168vV1dWk30GDBsna2lpDhgxJVR0AAAAAgCdLc+jevHmzLC2T3iBPSEjQtWvX0qWopzl69KhOnDihOXPmKGvWrHJ2dtakSZNUsGDBZNs3aNBA2bNnT3X/9vb2+uijjyRJderU0ZdffqmzZ8/K2tpamzZt0qJFi5QnTx5JUteuXeXn56eoqCjjrO7/3l9sbKzatm0ra2tr1axZU+PHj1fTpk2VLVs2VaxYUTly5NC5c+dSHbp9fX2VL18+Y627d+9Wzpw5JUleXl7Kmzev/vjjD2PotrKyUvv27WVpaSkvLy/lyZNHkZGRJqF75syZioiI0IIFC5Q5c+ZUnysAAAAAQMrSHLrd3d11+PDhJMvv3bunevXqad++felS2JOcP39e2bNnV65cuYzLKleunGL7IkWKpKl/e3t74+usWbNK+ic4X7x4UQaDQaVKlTKufzSk/sKFC8bt/r2/nDlzGvuxtraWJJMPCLJkyaIHDx6kur7H+7ewsNCCBQu0dOlSXb16VQaDQXFxcYqLizNp//gHJVmzZlVsbKzx/fbt27Vt2zbNnDlTOXLkSHUdAAAAAIAnS3Xo3rBhgzZs2KD4+Hh98cUXSdZfvHhRVlZW6VpcSiwtLZWYmJjq9mmtK7k7+ZJMguy/Pf7d5f/eX3L9Pem7zp/m8f6XLFmiadOmacqUKapQoYKsrKxUvXr1p+7/cb/99puqV6+ucePGydPT84X9PQIAAADA6y7VE6mVKVNG77zzjqR/7tb++4+jo6MmTZpktkIfV7RoUd29e1dXr141Ltu8ebP27t1r9v1K0unTp43LHr1+kZPIPS4iIkIeHh6qVKmSrKysdO3aNZPzkhpdunTR2LFjdePGDYWEhJipUgAAAAB486T6TnfRokXVpk0bWVhYKDAwMNk2W7duTbfCnsTZ2VllypRRcHCw+vfvrwsXLqh///4aPXq0smTJonPnzikmJibd95s3b155e3tr/PjxGjt2rCwsLBQcHCxPT08VLlzYOHHbi2RnZ6edO3fq1q1biomJ0ahRo1SkSBFduXIl1X1YWloqW7ZsGj58uNq0aaOaNWuqTJkyZqwaAAAAAN4MaX6mOzAwUH///bdOnTplMtz6ypUrGjp0qH777bd0LTAlISEh6t27typXrqy8efMqKChI1apVU9OmTTVq1Cjt3LlT/fv3T/f9jhw5Ul9//bV8fX2NE5MNHz483feTWs2aNdPevXtVvXp12dnZafDgwfr9998VHBys/Pnzp6mvihUrqlmzZurdu7eWL19ufP4cAAAAAPBsLAwGgyEtG2zatEk9e/bUgwcPZGFhoUebv/XWW2rQoIG+/PJLsxSKjHfv3j0dO3ZMpUuXZsI1pIuEhAQdOnRIbm5uzCWAdME1BXPgukJ645pCeuOayhiP8pGzs7NsbW1TbJfqZ7ofCQ4O1tdff60jR44oc+bMOnr0qBYvXqxKlSqpSZMmz1U0AAAAAACvkzQPL7948aIaNGgg6Z8ZuC0tLeXq6qquXbuqX79+Wrx4cXrXmG6GDBnyxPo6duyooKCgF1iRKQ8Pjyd+ddhPP/0kOzu7F1gRAAAAAOB5pDl058uXT5GRkSpVqpRy586t48ePy8nJSfb29jp16pQ5akw3AwYM0IABAzK6jBTt378/o0sAAAAAAKSjNIfuFi1aqGHDhtqxY4dq166tDh06qFatWjp+/LgcHR3NUSMAAAAAAK+kNIfu1q1bq2zZssqePbt69eqlrFmzKiIiQqVKlVKHDh3MUSMAAAAAAK+kNIdu6Z9njyUpU6ZM6t69e3rWAwAAAADAayPNs5cnJCRo2rRpqlOnjipUqCBJunv3rr755psnTgIGAAAAAMCbJs2he8SIEVq7dq3at29vDNnx8fGKjIzU8OHD071AAAAAAABeVWkO3WvXrtWUKVPk7+8vCwsLSVKuXLk0ZswYbd68Od0LBAAAAADgVZXm0B0fH69ChQolWZ41a1bdvXs3XYoCAAAAAOB1kObQ/c477yg0NNRk2f379zVmzBiVLVs23QoDAAAAAOBVl+bZy/v27avPPvtMc+bMUVxcnOrXr6+oqCjlzp1bU6dONUeNAAAAAAC8klIVuuvWras1a9ZIknr06KGff/5ZW7du1fnz52VjYyMHBwd5e3srU6Zn+gYyAAAAAABeS6lKyXfu3FH37t3l4OCg8+fPa8qUKTIYDJKkmJgYXb9+XQcPHpQkff755+ar9g3m4+Ojtm3bqlmzZhldCgAAAAAglVIVukeNGqU5c+bo0KFDSkxMNAbsf3s0mzleHbt379Z3332nU6dOKXv27KpRo4b69Omj7NmzZ3RpAAAAAPDKS1Xo9vT0lKenpyQpICBAP/zwg1mLwotx9epVtW/fXgMHDpS/v78uX76sdu3aacKECfryyy8zujwAAAAAeOWlefZyAnfqODo6avbs2fL29ta0adMkSatWrVKdOnXk7u4uHx8fzZ8/39h+4sSJ6tixo6ZPn64qVaqoQoUKGjp0aLJ9x8XFqXnz5urbt2+qapk2bZpq1qypcuXKqXbt2lq5cqUkKSEhQd98840aNWqkTJkyyd7eXlWrVtWpU6ee8+gBAAAAANIzzF6O1Nu8ebPCwsKUN29eRUVFqU+fPpo5c6a8vLy0e/duBQYGqnz58nJycpIkHTx4UK6urtq6dasOHDig1q1bq379+nJ1dTXpd9CgQbK2ttaQIUOeWsPBgwc1d+5cLV68WIULF9aOHTvUpUsXeXt7q3DhwvL395ckGQwG/fHHH9q0aZM6duyY/icDAAAAAN5AhG4z8vX1Vb58+SRJ9vb22r17t3LmzClJ8vLyUt68efXHH38YQ7eVlZXat28vS0tLeXl5KU+ePIqMjDQJ3TNnzlRERIQWLFigzJkzP7WGO3fuyNLSUjY2NrKwsJC3t7cOHDggS8v/DXLYt2+fWrduLQsLC3Xo0EGffPJJep4GAAAAAHhjEbrNqEiRIsbXFhYWWrBggZYuXaqrV6/KYDAoLi5OcXFxJu0fD8NZs2ZVbGys8f327du1bds2zZw5Uzly5EhVDV5eXipTpox8fHzk5eWlatWqyd/fX7a2tsY2FSpUUEREhE6ePKlevXopLi6OWegBAAAAIB2k+ZlupJ6VlZXx9ZIlSzRt2jQNHTpUv/32myIiIlSoUCGT9o8H7uT89ttvql69usaNG6eEhIRU1WBtba2QkBAtXLhQZcuW1bx58+Tv7687d+4k2beTk5Pat2+vH374wfiVcAAAAACAZ0fofkEiIiLk4eGhSpUqycrKSteuXdPVq1fT1EeXLl00duxY3bhxQyEhIanaJj4+XjExMXJyclKnTp0UFhYmCwsL7dy5U2FhYQoICDBpb2lpqUyZMvH1bwAAAACQDgjdL4idnZ1Onz6tW7du6cKFCxo6dKiKFCmiK1eupLoPS0tLZcuWTcOHD1dISIiOHj361G1CQ0PVtm1bXb58WZIUGRmpW7duycHBQe+++66OHDmiuXPnKi4uThcuXNCMGTNUs2bNZz5OAAAAAMD/8Ez3C9KsWTPt3btX1atXl52dnQYPHqzff/9dwcHByp8/f5r6qlixopo1a6bevXtr+fLlsra2TrHtp59+qosXL6pBgwaKjY1V4cKF1bNnTzk7O0uSZsyYoeHDh2v06NHKmTOnfHx81KtXr+c6VgAAAADAPywMPLyLVLp3756OHTum0qVLp3oiN+BJEhISdOjQIbm5uZnMgQA8K64pmAPXFdIb1xTSG9dUxniUj5ydnU0mqv43hpcDAAAAAGAmDC9/xXl4eOjBgwcprv/pp59kZ2f3AisCAAAAADxC6H7F7d+/P6NLAAAAAACkgOHlAAAAAACYCaEbAAAAAAAzIXQDAAAAAGAmhG4AAAAAAMyE0A0AAAAAgJkQugEAAAAAMBNCNwAAAAAAZkLoBgAAAADATF7Z0B0YGKjg4OCntuvbt6969Ohh/oJesO3bt8vR0TGjywAAAAAAPEGmjC4gtW7evKlNmzbpk08+kSSFhoZmcEUAAAAAADzZK3One/fu3VqyZElGlwEAAAAAQKplaOiOiIhQ8+bN5eHhocqVK2vQoEGKj4/Xnj175O7urtmzZ6t8+fKaPHmyPv/8cx05ckQuLi6KiopSQECAxowZY+wrNDRUNWvWVPny5dWmTRtFR0cnu89169bJ399fbm5uqlWrlhYtWpTqes+fP682bdrI09NTnp6e+vzzz3X79m3j+m3btqlGjRpyd3dXv379NH78eAUEBKTLvs+ePaumTZvK3d1dn3zyic6dO2eyPjw8XA0bNpS7u7uqVq2qCRMmSJIuXrwoJycnnThxwqT9e++9p0WLFun69evq1KmTPD09Vb58ebVu3VpRUVGprgsAAAAAkLIMDd09evRQpUqVtGfPHi1dulRbt27VwoULJUnx8fE6d+6cdu7cqaCgIHXs2FGurq6KiIhQ0aJFTfrZvHmzpk+frqlTp2r37t0qXLiwevbsmWR/ERER6t+/v3r16qUDBw5o5MiRGjFihA4ePJiqer/66isVKFBAv/76q9avX68zZ85oypQpkqSrV6+qS5cuat26tfbs2aN3331X8+bNS7d99+3bV3Z2dtqxY4dGjBhhEtjv3bunLl26qFmzZjp48KBmzJihWbNmacuWLSpSpIgqVKig1atXG9sfO3ZMly9f1ocffqjx48crZ86c2r59u8LDw+Xg4KCRI0emqiYAAAAAwJNlaOgOCwtThw4dZGVlZQyHv//+u6R/Qnfz5s1lY2MjCwuLJ/azbNky+fn5ycnJSdbW1urRo4datWqlxMREk3bLly9XjRo15O3tLSsrK3l4eMjX11crV65MVb3Tpk3T4MGDZW1trTx58qhq1arGenfv3i1bW1sFBATI2tpaH3/8sUqWLJku+7527Zp+++03tWvXTra2tipVqpQaNmxoXG9ra6vt27erUaNGsrCwkKOjoxwdHY21NWjQQGvXrpXBYJAkbdy4UdWrV1fOnDl1+/ZtZc6cWdbW1rK1tdXgwYM1adKkVJ0PAAAAAMCTZehEart379bkyZN19uxZPXz4UA8fPtSHH35oXF+kSJFU9RMVFSVPT0/j+7x588rX1zdJu/Pnz2vXrl1ycXExLjMYDPL29k7Vfn7//XeNHTtWJ06cUHx8vBISElS2bFlJ/wTjQoUKycrKyti+bNmyxmHdz7PvK1euSJLs7e2Ny4oXL27SZv369Zo9e7YuXLigxMRExcfHy8PDQ5JUu3ZtDRkyRPv371eFChW0adMmde7cWZL02WefqWPHjvr111/l7e0tX19feXl5pep8AAAAAACeLMNCd2RkpLp166Y+ffqocePGsrGxUa9evfTw4cP/FZcpdeVZWFgY7+I+iY2NjZo1a6YBAwakud5bt26pXbt2atasmaZPn67s2bMrODhYO3fulCQlJiYmqdfS8n8DCZ5n33FxcZKkhIQE47LH7+Lv2rVLgwcP1pgxY/T+++8rc+bMat68uXF99uzZVatWLa1evVr58+fX5cuXVbNmTUmSi4uLtmzZol9//VXbtm1T586d1bhxY/Xp0yfNdQIAAAAATGXY8PJjx47J2tpa//3vf2VjYyODwaBjx449U19FixbVmTNnjO9v3Lih0NBQxcfHm7RzcHBIMqHY5cuXTcJsSk6fPq27d++qTZs2yp49uyTp6NGjxvV58+bV5cuXTcJ/REREuuy7QIECkqRLly4Zl0VGRhpfHzlyRCVKlFCdOnWUOXNmPXjwwGS99M8Q802bNmnNmjX64IMPlCVLFkn/fBVb5syZVatWLQ0ZMkRTp041PlcPAAAAAHg+GRa67ezsFBsbq2PHjunWrVsaPXq0rK2tdfXq1WTvWmfJkkXXrl3TzZs3jXd+H2nUqJHWrl2rw4cPKy4uTpMnT9ZPP/2kzJkzm7T7+OOPdfDgQS1btkxxcXE6duyYPvnkE23YsOGp9RYpUkSWlpb67bffdO/ePc2ePVvXr1/X9evX9fDhQ1WoUEE3btzQwoULFRcXp2XLlpnMMP48+7a3t1epUqUUGhqq+/fv6+TJkybPgtvZ2eny5cu6dOmSrl+/rsGDB6tAgQLGYemSVLlyZVlZWWnWrFmqV6+ecXnTpk01ffp0PXjwQPHx8Tp8+LCKFSv21JoAAAAAAE+XYaHb3d1dLVq0UMuWLeXn5yc7Ozt9+eWXOnnypD7//PMk7d977z0ZDAbVqFHDOEHYI7Vq1VKPHj3UqVMnVapUSWfPntXYsWOT9FGqVCmNHTtWM2bMkIeHh7p06aI2bdqoTp06T623YMGC+vzzz/Xll1+qZs2aunXrlsaMGaO4uDg1b95cRYsW1bBhwzRhwgRVqVJFx48fl7+/v3ESuOfZtyRNmDBBp0+flpeXl/r166c2bdoY19WuXVvVqlVTnTp11KRJE9WoUUMdO3bU5s2bNXr0aEmSlZWV6tWrJ1tbW5Pn34ODg7V161ZVqlRJlStX1q5du0y+ig0AAAAA8OwsDKl5GBqpEhcXp8yZMxuDdp8+fZSYmGgMvhmtT58+Kly4sLp37/5M29+7d0/Hjh1T6dKllSNHjvQtDm+khIQEHTp0SG5ubiaTEALPimsK5sB1hfTGNYX0xjWVMR7lI2dnZ9na2qbYLkO/Mux1cu/ePXl5eWn+/PlKTEzUH3/8oZ9//lnVq1fP6NIkST///LO2bdumFi1aZHQpAAAAAPDGyNCvDHuZ1K9f32Qytn8LDQ1VhQoVUlxva2ur8ePHa8yYMRo9erTy5MmjwMBA+fn5mX3fT/Phhx8qLi5Oo0aNUv78+Z+5HwAAAABA2hC6/9+qVaueuw9vb+9Uf+d3eu/7SX766Sez9g8AAAAASB7DywEAAAAAMBNCNwAAAAAAZkLoBgAAAADATAjdAAAAAACYCaEbAAAAAAAzIXQDAAAAAGAmhG4AAAAAAMyE0A0AAAAAgJkQugEAAAAAMBNCNwAAAAAAZkLofszEiRPVuHHjjC7jiZYvX64qVapIkvbt2ycXFxfFxcU9cZvo6Gg5OjoqMjLyRZQIAAAAAPh/hO4XZNeuXYqIiEjXPitUqKCIiAhZW1una78AAAAAgPRB6H5BZs+erd9//z2jywAAAAAAvECvbeieNm2aatasqXLlyql27dpauXKl9uzZI0dHRz148MDYrkePHurbt6/JtiEhIfLy8lLlypU1btw4GQwGSVJCQoLGjBmjKlWqqEKFCurWrZtu3rwpSXrw4IG++uoreXt7q3z58mrevLlOnjwpSerQoYO2bdumoUOHqlWrVqmqPzw8XPXr15ebm5v8/f21a9euJG3+fTxRUVEKDAyUu7u7atasqblz5ybb94ULF1S5cmUtW7YsxXMFAAAAAHh+r2XoPnjwoObOnat58+bp0KFDGjBggAYPHqy//vrrqdueOnVK9+/f19atWzVhwgTNmjVLP/30kyTphx9+0KZNm7Ro0SJt27ZN9+/f15AhQyRJ06dP1+HDh7VmzRrt3r1bJUuWNIb5kJAQ2dnZ6auvvtKcOXOeWsOVK1fUpUsXdejQQfv27VOrVq3UqVMnY8BPSefOnVWqVCnt3LlTU6ZMUXBwsHbs2GHS5u7du+rQoYOaNGmiRo0aPde5AgAAAAA8WaaMLsAc7ty5I0tLS9nY2MjCwkLe3t46cOCA9u3b99RtLS0t1alTJ1lbW8vDw0NVq1bV9u3b5evrq+XLl6tZs2ayt7eXJA0YMMA4OVn79u3VunVrZc+eXZL04Ycfavny5Xr48KEyZUrbaV6/fr2KFi2qOnXqSJIaNmyoLFmyKDExMcVtjh49qhMnTmjOnDnKmjWrnJ2dNWnSJBUsWNDYxmAwqGfPnnJyclK3bt2eeK4sLV/Lz2MAAAAA4IV6LUO3l5eXypQpIx8fH3l5ealatWry9/dP1bYODg4mE5M5ODjoxIkTkv4Zvv0ocEtS0aJFVbRoUUnSjRs3NHToUO3du1d3796V9M9w9ISEhDSH7vPnz5vsR5L8/Pyeuk327NmVK1cu47LKlStL+mf2ckkKDg7Wzp07Te5+p3SubG1t01QzAAAAACCp1/J2prW1tUJCQrRw4UKVLVtW8+bNk7+/v+7cuZOkbUJCgsl7CwsLk/cGg8EYwi0sLFK829yjRw/FxMRo5cqV+v333zV9+vRnrt/S0vKJd7WfdZvLly/LwcFBkyZNMi5Ly7kCAAAAAKTNaxm64+PjFRMTIycnJ3Xq1ElhYWGysLDQqVOnJEn37983to2KijLZNjo6WvHx8cb358+fNw7RLlq0qM6cOWNcd+7cOc2bN0+SdOTIETVu3FiFChWSJP3xxx/PXL+9vb3JfiTpxx9/TFLr44oWLaq7d+/q6tWrxmWbN2/W3r17je+HDx+uUaNGad68ecah9imdq507dz5z/QAAAACAf7yWoTs0NFRt27bV5cuXJUmRkZG6deuWKleuLCsrK23YsEEPHz7UihUrdOnSJZNt4+PjNX36dMXFxenQoUPasWOH3n//fUlSo0aNtGDBAp0+fVp3797V6NGjtX//fkmSnZ2djhw5ovj4eG3fvt04hPvKlSuSpCxZsuj8+fOpuoNct25dXbp0SYsXL1ZcXJzWrl2r7777TtmyZUtxG2dnZ5UpU0bBwcG6e/euTp48qf79+ys2NtbYxtLSUs7OzurQoYP69OmjmJiYFM+Vg4NDak83AAAAACAFr2Xo/vTTT1W6dGk1aNBAbm5u6t69u3r27Kly5cqpZ8+eCg4OVqVKlXTs2DHjZGWPuLi4yGAwqGrVqurUqZPatm0rb29vSVJAQIAaNGigZs2aqWbNmrKystKAAQMkSQMHDtTGjRtVsWJFLV26VN99953KlSunhg0b6vr162rcuLHmz5+vli1bPrX+fPnyaebMmZo9e7YqVKigadOmafLkycqTJ88TtwsJCTF+HViHDh0UFBSkatWqJWnXvn175cmTR8OHD0/xXDk7O6f2dAMAAAAAUmBhePQl1MBT3Lt3T8eOHVPp0qWVI0eOjC4Hr4GEhAQdOnRIbm5usrKyyuhy8BrgmoI5cF0hvXFNIb1xTWWMR/nI2dn5iRNRv5Z3ugEAAAAAeBm8ll8Z9jI7cuSIWrRokeL6IkWKaMOGDS+wIgAAAACAuRC6XzBXV1dFRERkdBkAAAAAgBeA4eUAAAAAAJgJoRsAAAAAADMhdAMAAAAAYCaEbgAAAAAAzITQDQAAAACAmRC6AQAAAAAwE0I3AAAAAABmQugGAAAAAMBMCN0AAAAAAJgJoRsAAAAAADMhdL/ELly4IBcXF505cyajSwEAAAAAPANC90vMzs5OERERKlGixAvZ35UrV+Tu7q6JEye+kP0BAAAAwOuO0A2joUOHysrKKqPLAAAAAIDXBqH7JRYdHS1HR0dFRkbKx8dHS5YsUbt27eTu7q733ntP4eHhqern+vXr6tSpkzw9PVW+fHm1bt1aUVFRJm1++eUX/fnnn6pRo4YZjgQAAAAA3kyE7lfIzJkz1blzZ+3Zs0cVK1bUt99+m6rtxo8fr5w5c2r79u0KDw+Xg4ODRo4caVwfGxurb775RoMGDVKmTJnMVT4AAAAAvHEI3a+QmjVrytXVVdbW1qpdu7bOnj2rxMTEp253+/ZtZc6cWdbW1rK1tdXgwYM1adIk4/rJkyfLzc1NlSpVMmf5AAAAAPDGIXS/Quzt7Y2vbWxslJCQoPj4+Kdu99lnn+nnn39WrVq1NHDgQO3Zs8e47s8//9SSJUvUt29fs9QMAAAAAG8yQvcrxNLy2f66XFxctGXLFvXv318Gg0GdO3fWyJEjZTAYNHjwYHXp0kX58+dP52oBAAAAADzA+wa4efOmcubMqVq1aqlWrVqqV6+e2rdvr5YtW2rfvn06deqUJkyYIEm6d++eLC0ttWXLFq1YsSKDKwcAAACAVxuh+w3QtGlTNWzYUK1atZKlpaUOHz6sYsWKqVChQvrll19M2g4fPlyFChXSZ599lkHVAgAAAMDrg9D9BggODtbXX3+tqVOnKlOmTHJxcdGYMWNkZWWlQoUKmbTNmjWrsmfPznBzAAAAAEgHhO6XmL29vU6cOCFJ2rJli8k6T09P47qncXJy0oIFC1LVdsSIEWkrEgAAAACQIiZSAwAAAADATLjT/Ypbv369evfuneL6ChUqKDQ09AVWBAAAAAB4hND9ivP19ZWvr29GlwEAAAAASAbDywEAAAAAMBNCNwAAAAAAZkLoBgAAAADATAjdAAAAAACYCaEbAAAAAAAzIXQDAAAAAGAmhG4AAAAAAMyE0A0AAAAAgJkQugEAAAAAMBNCt5mFhYXJx8fnqe327NkjR0dHPXjw4Ln32bdvX/Xo0eO5+wEAAAAAPB9CtxksXbpUN27ckCQ1aNBAW7ZsyeCKAAAAAAAZgdCdzhISEjRixAj9/fffGV0KAAAAACCDvZah29HRUWvXrlXDhg3l6uqqdu3a6fLly2rTpo3c3d3VsGFDRUdHp6qvBw8e6KuvvpK3t7fKly+v5s2b6+TJk8b1Pj4+mjp1qmrVqqVBgwapYsWKunPnjvz9/TVp0iQtX75cVapUMbb/448/1KRJE7m5ual27dpat25dsvu9cOGCOnToIE9PT1WoUEG9e/dWTEzMM52PdevWyd/fX25ubqpVq5YWLVpkXNe3b18NGTJEw4cPV8WKFVWpUiVNnz79mfYDAAAAADD1WoZuSVq4cKFCQkK0atUq7dq1S23bttUXX3yhX3/9VQkJCZo1a1aq+pk+fboOHz6sNWvWaPfu3SpZsqT69u1r0mbt2rUKDQ3V4MGDtXLlSknSypUr1blzZ5N29+/fV/v27fXBBx9o7969GjhwoPr06aPIyEiTdgaDQUFBQSpcuLC2bdumn376SVeuXNHIkSPTfB4iIiLUv39/9erVSwcOHNDIkSM1YsQIHTx40NhmzZo1cnJy0o4dO9SrVy+NGzdOV69eTfO+AAAAAACmXtvQ7efnpwIFCqh48eIqWbKkXFxcVKZMGWXPnl0VK1bU2bNnU9VP+/bttWDBAuXKlUvW1tb68MMPdfz4cT18+NDYpmrVqipWrJgsLCye2Fd4eLji4+PVunVrWVtbq0qVKgoODpaNjY1Ju4iICJ06dUq9evVS1qxZlTdvXnXp0kWrVq2SwWBI03lYvny5atSoIW9vb1lZWcnDw0O+vr7GDwckyd7eXh999JEyZ86sOnXqKCEhIdXnBwAAAACQskwZXYC5FC5c2Pg6S5YsKliwoMn7uLi4VPVz48YNDR06VHv37tXdu3cl/fPcdkJCgjJl+uf02dnZpaqv8+fPq1ChQrKysjIuq1WrliSZDHePiopSQkKCPD09TbZPSEjQ33//rTx58qRqf4/2uWvXLrm4uBiXGQwGeXt7G9/b29sbX2fNmlWSFBsbm+p9AAAAAACS99qG7n/fdba0fLab+j169FCWLFm0cuVKFSpUSLt27VLr1q1N2jweop/E0tJSiYmJT22XJUsW2dra6rfffnuWkk3Y2NioWbNmGjBgwBPrAgAAAACkP9LWUxw5ckSNGzdWoUKFJP0zEdqzKlq0qC5cuGBylz0sLEzHjh0zaefg4KB79+4pKirKuCwmJuaZZkR3cHDQiRMnTJZdvnxZCQkJae4LAAAAAJA2hO6nsLOz05EjRxQfH6/t27drx44dkqQrV64k2/7R89lnz55NMtt4tWrVZGtrq5CQED148EB79+7VoEGDktwpL126tNzd3TVs2DDduHFDt2/f1qBBg9S7d+801//xxx/r4MGDWrZsmeLi4nTs2DF98skn2rBhQ5r7AgAAAACkDaH7KQYOHKiNGzeqYsWKWrp0qb777juVK1dODRs21PXr15O0z5cvn2rXrq1u3bopODjYZJ21tbVmzZqlX375RRUqVNCAAQP07bffqnTp0kn6GTt2rAwGg2rVqqX333/f+P3faVWqVCmNHTtWM2bMkIeHh7p06aI2bdqoTp06ae4LAAAAAJA2Foa0ToeNN9a9e/d07NgxlS5dWjly5MjocvAaSEhI0KFDh+Tm5pbquRGAJ+GagjlwXSG9cU0hvXFNZYxH+cjZ2Vm2trYptuNONwAAAAAAZvLazl6eGkOGDNHixYtTXN+xY0cFBQW9wIqebubMmUmGrT/O399fQ4cOfXEFAQAAAABS9EaH7gEDBjzxq7ReRm3atFGbNm0yugwAAAAAQCowvBwAAAAAADMhdAMAAAAAYCaEbgAAAAAAzITQDQAAAACAmRC6AQAAAAAwE0I3AAAAAABmQugGAAAAAMBMCN0AAAAAAJgJoRsAAAAAADMhdKejvn37qkePHpKkKVOmqGXLlk/dZvny5apSpcoz7c9gMKhr165yc3PTmjVrnqkPAAAAAID5ZMroAl5XQUFBCgoKMus+jh07pg0bNmjVqlVydHQ0674AAAAAAGnHne5XWExMjCSpePHiGVsIAAAAACBZb2Tojo6OlqOjoyIjI43LxowZo4CAgFRtHxoaqpo1a6p8+fJq06aNoqOjk7SZOHGiGjdubHwfHh6u+vXry83NTf7+/tq1a1eyfW/dulUVKlTQ8ePHn1jDjh07FBgYKEny8PBQWFiY+vbtq/79+ysgIEB169aVJN28eVM9e/aUt7e33N3d1bFjR125csXYz5o1a1StWjW5u7vr888/16hRo1J9HgAAAAAAT/ZGhu7nsXnzZk2fPl1Tp07V7t27VbhwYfXs2fOJ21y5ckVdunRRhw4dtG/fPrVq1UqdOnXSzZs3TdqdPHlSffr00bhx4+Tk5PTEPqtUqaKZM2dKkvbv368GDRpIkn7++WcFBgZq9erVkv55zjw2NlZr167Vr7/+KltbW/Xr10+SdOvWLX355Zdq06aN9uzZo/r162vp0qXPcFYAAAAAAMnhme40WrZsmfz8/IyhuEePHtq7d68SExNT3Gb9+vUqWrSo6tSpI0lq2LChsmTJYrLNjRs31LFjR/Xq1Uve3t7PXJ+dnZ1q1qwpSfrrr7+0detWrVu3Tjlz5pQk9ezZUzVq1NC1a9e0d+9eZc2aVS1btpSVlZVq1Kih8uXL6+7du8+8fwAAAADA/xC60ygqKkqenp7G93nz5pWvr+8Ttzl//rzs7e1Nlvn5+RlfP3z4UF27dlWBAgX0ySefPFd9dnZ2JrVKMt4Ff8TKykqXLl3S5cuXVahQIVlZWRnXFS9eXH/88cdz1QAAAAAA+Aeh+/8lJCSkqp2FhYUMBkOa+ra0tHzinfCbN28qf/782rZtm7Zs2SIfH5809f+4xwO0jY2NJGn79u3KnTt3krZ79+5NsuxJdQIAAAAA0uaNfKY7S5YskqTY2Fjjskd3hZ+maNH/a+/eo6qq8/+Pv4DkoqAOWqh4KycJJ26KCmapaAHeQCpD02qkMlHnK2WNTTo6SVZe0kzQFNP8ZqiQiUoSmZpKXlJDj2gapKWoeCE1VAQP5/dHP89XRlRQDkfl+ViLtc7e+7M/+7UPnwW82Z+9TxMdOHDAvJyfn69PPvlExcXF19yncePGpfaRpM8++8x8TFdXV02dOlUjR47UmDFjlJ+fX+5zuR53d3fZ2tpq37595nXFxcXmB6ndd999OnbsWKlC++eff66UYwMAAAAAqmnR7erqKhcXF6Wnp8toNGrjxo3KzMws175PPvmkUlNTtXPnThUVFSkuLk5paWmqUaPGNffp2bOnjh49qiVLlqioqEipqan64IMPVKtWLUl/XgmXpP79++vBBx/UuHHjbvUUJUkuLi7q3r27Jk+erGPHjqmwsFAffPCBBg0aJJPJpI4dO+rcuXNKTExUcXGxvv32WxkMhko5NgAAAACgmhbddnZ2Gjt2rL788kvzx209++yz5dq3a9euiomJ0dChQxUQEKCDBw9qypQp192nfv36mjt3rubPn6+2bdtq9uzZiouLk6ura6l2NjY2mjBhgjIyMpSSknLT53elMWPGqFmzZurRo4ceffRRZWdnKz4+XjY2NnJ1ddXEiRM1b948tW3bVitXrlSfPn0q5bgAAAAAAMnGVNEblHFXmzx5snbu3Kn//d//vWrb+fPntXfvXrVs2VIuLi5WSIe7jdFoVGZmpnx9fUs9jwC4WYwpWALjCpWNMYXKxpiyjsv1kaenp2rWrHnNdtXySjcAAAAAAFWBp5dfYe7cuZo2bdo1t4eFhSk2NrbaZQEAAAAA3ByK7itERUUpKirK2jEkWS/LyJEjq/yYAAAAAHC3Yno5AAAAAAAWQtENAAAAAICFUHQDAAAAAGAhFN0AAAAAAFgIRTcAAAAAABZC0Q0AAAAAgIVQdAMAAAAAYCEU3QAAAAAAWAhFNwAAAAAAFkLRfRvbv3+/goOD5evrK0lKSEiQv7+/xo0bp/j4eA0YMKBc/YwePVpvvPGGBZMCAAAAAMpyj7UD3G3mzZungQMH6p57bv2tXbJkiWrXrq3U1FRJ0syZMzVixAgNHDhQkhQdHV2ufmJjY82vjUajFixYoL///e+3nA8AAAAAcH1c6a5E+fn5ev/992U0Giulv3Pnzqlx48bmAr6goEDNmjW7pT737NmjhISEyogHAAAAALgBiu5rOHTokAYNGiQ/Pz916dJFCxYskCQZDAb1799f/v7+6tChg8aOHavi4mKdPHlSjz32mEwmk/z9/bV06dIbHuPixYsaPXq0OnbsqNatW6t///7av3+/JOmNN97QsmXLlJaWJg8PD3l5eUn68+r26NGj9dFHH6lv376SpC1btqhNmzZav369QkJC5Ovrq6ioKJ05c0aSNGrUKMXExGjXrl2KjIzUyZMn5eXlpbi4OD388MP6/fffzZkKCwvl5+enjRs3Vur7CQAAAADVEUX3NQwbNkwtWrTQ999/r/j4eE2bNk0ZGRmKiYlRQECAtmzZouTkZK1du1aLFi1S/fr1NXfuXEnStm3bFBERccNjzJkzRzt37tTKlSu1efNmPfDAAxo1apQkaeLEiQoLC1NISIj27dsng8EgSYqPjy81XfyyCxcuKDU1VYsXL1ZaWpr27dunJUuWlGrj7e2t8ePHq379+jIYDIqOjpabm5vS0tLMbTZu3KhatWopMDDwpt87AAAAAMCfKLrLsGfPHu3bt09Dhw6Vk5OTPD09NWPGDDVo0EDLli3TK6+8Ijs7OzVq1Eht27bV7t27b+o4gwcPVmJiourWrSt7e3uFhITop59+0qVLlyrcl9Fo1Isvvqg6deqoQYMGatOmjX755Zfr7mNjY6OwsDCtWLHCvC49PV3du3eXnZ1dhTMAAAAAAErjQWpl+O233+Ts7Ky6deua13Xo0EGStHr1asXFxengwYO6dOmSLl26pJCQkJs6Tn5+vmJjY7V161adO3dO0p/Fs9FovKkHsTVu3Nj82snJSYWFhTfcJzw8XDNnzlRubq7uu+8+rVu3znzFHgAAAABwa7jSXQZbW1uVlJRctT4nJ0f/8z//oz59+mjTpk0yGAzq2bPnTR8nJiZGBQUFSklJ0e7duzVnzpxbiS1b24p/O5s2bSofHx+lpqZq69atcnV1Nd8/DgAAAAC4NRTdZWjSpInOnTun48ePm9etXr1aq1atkr29vZ577jk5OjrKZDJp7969N32cXbt2qW/fvmrQoIEkKSsr65az34zw8HClpaVp1apV6tWrl1UyAAAAAMDdiKK7DJ6enmrVqpWmTZumc+fOaf/+/XrrrbdUo0YNFRYWau/evTpz5owmTZoke3t7HT9+XCaTSY6OjpKkAwcO6Pz58zc8jru7u3bt2qXi4mKtX79eGRkZkqS8vDyLnZujo6P++OMP5eXlmaefd+/eXdnZ2RTdAAAAAFDJKLqvYdasWcrNzVWHDh30yiuvKDo6WoMHD9azzz6rAQMGqEePHnJ3d9e//vUv7d+/XzExMfL09JSfn5+eeuopJSYm3vAY//73v5Wenq527dopOTlZH3zwgXx8fBQREaGTJ09a5LwCAgLUuHFjdevWTWvWrJEk1a5dW507d9Zf//pXNW3a1CLHBQAAAIDqyMZkMpmsHQLWN2DAAIWFhenpp5++Zpvz589r7969atmypVxcXKowHe5WRqNRmZmZ8vX15Yn5qBSMKVgC4wqVjTGFysaYso7L9ZGnp6dq1qx5zXY8vbyaM5lMSkxMVG5uLlPLAQAAAKCSUXRbyPjx47VkyZJrbh8yZIiio6OrMFHZfHx81KRJE3344Yfme9IBAAAAAJWDottCxowZozFjxlg7xg3t2rXL2hEAAAAA4K7Fg9QAAAAAALAQim4AAAAAACyEohsAAAAAAAuh6AYAAAAAwEIougEAAAAAsBCKbgAAAAAALISiGwAAAAAAC6HoBgAAAADAQii6AQAAAACwEIru29g777wjPz8/zZ49W3l5eYqIiJCPj4+OHj0qLy8vZWRk3LCP3NxceXl56cCBA1WQGAAAAABwpXusHaCqpKeny8PDQ82aNbvpPk6fPq1vvvlGTz/9dCUmu/axFixYoJkzZyooKEjz58/XqVOntGXLFjk6OspgMJSrH3d391JtN23aJGdnZ3l5eVkqOgAAAADg/6s2V7qnT5+uX3/99Zb62Lx5s5KSkiop0fWdO3dOksz/JCgoKJCbm5scHR1vqd/58+dr9+7dt5wPAAAAAHBj1aLo7t27t37++WdFR0frzTff1MaNGxURESE/Pz89+uijmj59urntyZMnNXToULVv316tW7fWCy+8oEOHDmnVqlV69dVXtWvXLnl5eenQoUMqKSnR9OnT1a1bN/n4+OjJJ5/U9u3by53rWjkOHDig4OBgSVJYWJg8PDwUHx9vPnZubq48PDy0fv16SdLAgQM1a9Ysvf7662rdurUeffRRpaSkSJIOHz4sDw8P5eTk6JVXXtG6desUGxur559/Xs8//7zee++9Upni4uIUGRl5S+83AAAAAOBP1aLoXr58uSQpPj5eY8aM0fDhw9WvXz/t2LFDCQkJmjdvntasWSNJ+vDDD1WnTh2tX79eGzduVNOmTfX+++8rNDRUQ4YMkbe3twwGg5o0aaJPP/1UqampSkhI0A8//KDw8HANGTJE58+fv2Gm8+fPXzPH/fffr7S0NElSSkqK9u3bV+rY7u7uV/W3cOFC9e7dW1u2bFHfvn319ttvq7i4uFSbWbNmyd3dXaNHj9ann36q8PBwpaamqqSkxNwmPT1dvXr1uun3GgAAAADwf6pF0X2lmjVrav369XryySdlY2MjDw8PeXh4mKdcnz17VjVq1JC9vb1q1qypcePGacaMGWX2lZycrBdeeEHNmzeXvb29Bg4cqNq1a2vdunW3nKOiLl8tr1GjhkJDQ1VQUKDjx49fd58nnnhCBQUF2rJliyTp0KFDysnJUWho6E1lAAAAAACUVm0epHalVatWaf78+crNzVVJSYmKi4vl7+8vSXrxxRc1ZMgQbdiwQR07dlRoaKgCAwPL7Oe3337TO++8owkTJpjXlZSU6OjRo7eco6IaN25sfn35vu/CwkI5ODhcc59atWqpW7duWr58uQIDA5Wenq5HHnlErq6uN5UBAAAAAFBatbvSvWnTJo0bN07Dhg3Ttm3bZDAY1Lp1a/N2Ly8vrVmzRm+99ZZMJpOGDRum999/v8y+HB0dNWXKFBkMBvNXVlaWoqKibjlHRdna3ty3Mjw8XOnp6SoqKtI333zD1HIAAAAAqETVrujetWuX7r//fnXv3l01atTQxYsXlZOTY95++vRp1ahRQ127dtX48eM1c+ZMLVq0qMy+mjRpon379pVad/jw4UrJUVUCAwNVq1YtJSUl6eeff1bXrl2rPAMAAAAA3K2qTdHt4OCgX3/9Vffdd5+OHTumo0eP6uTJkxo3bpzuu+8+5eXlSZIiIyM1Z84cXbx4UcXFxdq5c6f5Y7scHBx04sQJnT59WkVFRYqMjNTChQuVmZkpo9Gor776Sj179tSRI0dumMfd3f26OSz5Pvz222/6448/JP15hbxXr1764IMP1LVrVzk5OVn0+AAAAABQnVSbojsyMlITJ07UqlWr9Nhjj6l79+565pln1LlzZw0ZMkSrV6/WpEmTNG3aNK1du1YBAQHq0KGDNm3apMmTJ0uSunXrJpPJpM6dO2v37t166qmn1L9/fw0bNkxt2rRRQkKCZsyYoUaNGt0wT3Bw8HVzWErfvn31+eefa8CAAeZ14eHhKigoYGo5AAAAAFQyG5PJZLJ2CFjX5s2b9a9//UurV6++7r3h58+f1969e9WyZUu5uLhUYULcrYxGozIzM+Xr6ys7Oztrx8FdgDEFS2BcobIxplDZGFPWcbk+8vT0VM2aNa/Zrtpc6UbZjh8/rgkTJigqKuqmH8YGAAAAAChbtfzIMEs7efKkunTpct02BoOhitJc28cff6zZs2crPDxc/fr1s3YcAAAAALjrUHRbQP369W+LovpGBg8erMGDB1s7BgAAAADctZhPDAAAAACAhVB0AwAAAABgIRTdAAAAAABYCEU3AAAAAAAWQtENAAAAAICFUHQDAAAAAGAhFN0AAAAAAFgIRTcAAAAAABZC0Q0AAAAAgIVQdAMAAAAAYCEU3ZUgNzdXXl5eOnDggLWjmHl5eSkjI8PaMQAAAACgWrvH2gHuBu7u7jIYDNaOUcrtlgcAAAAAqiOudAMAAAAAYCEU3ZXg8OHD8vDwUE5OjoKCgpSYmKiBAwfKx8dHkZGROnr0qF577TX5+fkpODhYu3fvliQtXbpUjz/+uJKSkvToo4/K19dX//73v3Xp0qVyHXfdunXq1auX/Pz81LFjR02aNEklJSWSJA8PD61fv9489f3KLw8PD82YMUOSVFhYqLfffludO3eWr6+vBg4cqOzsbMu8UQAAAABQzVB0W8Dnn3+ut99+W99++60OHz6sZ599VhEREdq8ebOaNGliLnglKS8vTwaDQenp6friiy+0Zs0aLVy48IbHKC4uVkxMjN58803t2LFDn332mb7++mutWbOmVLvLU98vf82YMUPOzs7q2bOnJGny5Mnas2ePFi9erM2bN8vLy0vDhg2TyWSq3DcFAAAAAKohim4L6Ny5s+6//37Vr19f3t7eatKkiR555BE5ODioY8eOOnjwoLntxYsXNWLECDk5OalFixbq0aOH1q1bd8NjXLx4UYWFhapZs6ZsbGzUvHlzpaenq1u3btfcJy8vT6NGjdJ//vMfNW/eXCUlJVq6dKmio6Pl5uYmR0dHjRgxQkeOHNGuXbsq4Z0AAAAAgOqNB6lZQIMGDcyvHRwc5OzsXGq5qKjIvFynTh25urqalxs1aqSNGzfe8BjOzs4aOnSoBgwYIG9vbz3yyCOKiIhQw4YNy2xfUlKikSNHqmvXruar3KdOndK5c+cUHR0tGxubUm2PHj0qHx+f8p80AAAAAOAqFN0WYGtre93lKxmNxlLLJpOpVAF8PcOGDdPTTz+t1atXa/Xq1UpISNCnn34qb2/vq9rGx8fr9OnTmjNnjnmdo6OjJGnRokV6+OGHy3VMAAAAAED5Mb3cygoKCpSfn29ePnLkiNzc3Mq17+nTp+Xm5qZnn31W8+bNU0hIiFJSUq5qt3XrVn3yySeaNm2audCWJBcXF9WtW1f79u0r1f7w4cM3eTYAAAAAgCtRdFuZvb294uLiVFhYqOzsbKWmpiooKOiG+/34448KDQ3Vrl27ZDKZdOrUKR04cEBNmzYt1S4/P18jR47U6NGj1aJFi6v6iYyM1MyZM5WTk6Pi4mLNnz9fTz31lC5cuFBp5wgAAAAA1RXTy62sdu3aatmypR5//HH98ccf6t27tyIjI2+4n5+fn4YMGaIRI0bo5MmTqlu3rkJDQ/Xss8+Wavfdd98pLy9PY8eO1dixY83r27Ztq08++UTR0dE6e/as+vfvr+LiYnl6emrOnDlycnKq9HMFAAAAgOrGxsRnQ1nN0qVLNWXKFGVkZFg7SrmcP39ee/fuVcuWLeXi4mLtOLgLGI1GZWZmytfXV3Z2dtaOg7sAYwqWwLhCZWNMobIxpqzjcn3k6empmjVrXrMd08sBAAAAALAQppffpl555ZXrXgEfP368wsPDqy4QAAAAAKDCKLqtKCIiQhEREWVumzVrVhWnAQAAAABUNqaXAwAAAABgIRTdAAAAAABYCEU3AAAAAAAWQtENAAAAAICFUHQDAAAAAGAhFN0AAAAAAFgIRTcAAAAAABZC0Q0AAAAAgIVQdAMAAAAAYCEU3QAAAAAAWAhF9x0iKChIiYmJ1o4BAAAAAKgAim6YDR06VEFBQdaOAQAAAAB3DYpuSJLWrl2rLVu2WDsGAAAAANxVKLotxMPDQ/Pnz1fHjh01e/ZsSdLy5cvVvXt3+fn5KSgoSJ9//rm5/UcffaQhQ4Zozpw5euSRR9S2bVvFxsaW2XdRUZH69++vUaNGlSvL7Nmz1aVLF/n4+Cg4OFgpKSmltl+4cEHjx4/XoEGDbvJsAQAAAABlucfaAe5mq1ev1rJly1SvXj0dOnRI//znPzV37lwFBgZq8+bNGjRokFq3bq2HHnpIkrRjxw55e3tr7dq12r59u1544QX17t1b3t7epfodO3as7O3tNX78+Btm2LFjhxYsWKAlS5aoYcOGysjI0PDhw9WxY0fVq1dPkjRjxgy1bdtWbdq0UXJycuW/EQAAAABQTVF0W1BoaKjq168vSWrcuLE2b96sOnXqSJICAwNVr149ZWVlmYtuOzs7DR48WLa2tgoMDJSrq6tycnJKFd1z586VwWBQYmKiatSoccMMf/zxh2xtbeXo6CgbGxt17NhR27dvl63tn5Mc9u/fry+//FIrVqxQdnZ2Zb8FAAAAAFCtUXRbUKNGjcyvbWxslJiYqOTkZB0/flwmk0lFRUUqKioq1f5yMSxJTk5OKiwsNC+vX79e69at09y5c+Xi4lKuDIGBgWrVqpWCgoIUGBioxx57TGFhYapZs6ZMJpPGjRunYcOGqV69ehTdAAAAAFDJuKfbguzs7Myvk5KSNHv2bMXGxurHH3+UwWBQgwYNSrW/suAuy48//qhOnTpp6tSpMhqN5cpgb2+vWbNmadGiRXr44Ye1cOFChYWF6Y8//lBycrIuXbqkyMjIip8cAAAAAOCGKLqriMFgkL+/vwICAmRnZ6cTJ07o+PHjFepj+PDhmjJlivLz8zVr1qxy7VNcXKyCggI99NBDGjp0qJYtWyYbGxt9//33Wr58uX7++WcFBgaqffv2io6O1tGjR9W+fXtt3779Zk4TAAAAAHAFppdXEXd3d33//fc6c+aMCgoKNHHiRDVq1Eh5eXnl7sPW1la1atXSu+++q6ioKHXp0kWtWrW67j6ffPKJ1q1bp6lTp6pBgwbKycnRmTNn1LRpU3344Yelprf/+OOPeu+997R48WK5urre9LkCAAAAAP5E0V1F+vXrp61bt6pTp05yd3fXuHHjtHv3bk2bNk333ntvhfpq166d+vXrpzfeeENLly6Vvb39Ndv+/e9/15EjRxQeHq7CwkI1bNhQI0eOlKen51VtXV1dZWdnd9W0dwAAAADAzbExmUwma4fAneH8+fPau3evWrZsWe4HuQHXYzQalZmZKV9f31LPQABuFmMKlsC4QmVjTKGyMaas43J95OnpqZo1a16zHfd0AwAAAABgIUwvv8P5+/vr4sWL19yelpYmd3f3KkwEAAAAALiMovsOt23bNmtHAAAAAABcA9PLAQAAAACwEK50o9xKSkokSYWFhTygAZXCaDRK+vMhFIwpVAbGFCyBcYXKxphCZWNMWceFCxck/V+ddC08vRzldurUKR08eNDaMQAAAADgttG8eXPVq1fvmtspulFuly5d0pkzZ+Tg4CBbW+5MAAAAAFB9lZSU6OLFi6pTp47uuefak8gpugEAAAAAsBAuVwIAAAAAYCEU3QAAAAAAWAhFN0rJzc3Vyy+/rPbt26tLly6aNGnSNZ/Gt2DBAgUHB6t169bq16+fdu/eXcVpcSeoyJhKTExUcHCw/Pz8FBYWptWrV1dxWtwJKjKmLsvLy5Ofn58++uijKkqJO0lFxlROTo4GDhwoHx8fderUSfPnz6/asLhjlHdclZSUaPr06QoKCpKfn5969eqlr776ygqJcbvbsGGDOnTooJiYmOu2Kykp0dSpU9W1a1e1bdtWUVFROnToUBWlRFkoulHK8OHD5ebmptWrV2vevHlavXq1Pv3006varVmzRh999JEmTpyo77//Xl26dNErr7yi8+fPWyE1bmflHVNff/21pkyZogkTJmjr1q0aMGCARowYwS8JXKW8Y+pKsbGxfIQKrqm8Y6qwsFAvvviiOnXqpM2bN+ujjz5ScnKycnJyrJAat7vyjqvExEQlJSUpISFB27Zt06uvvqrXX39dP/30kxVS43Y1Z84cxcbGqlmzZjdsu3DhQq1YsUKzZ8/W2rVr1bx5cw0dOlQ8yst6KLphZjAY9NNPP2nkyJFycXFR8+bN9cILL2jx4sVXtV28eLEiIiLk4+MjR0dHvfjii5KktWvXVnVs3MYqMqYKCwv16quvqk2bNqpRo4aefvpp1apVS5mZmVUfHLetioypy7777jtlZ2erc+fOVRcUd4yKjKlVq1bJ2dlZL774opycnOTt7a2VK1eqRYsWVkiO21lFxlVWVpbatGmjBx54QHZ2durSpYvq1q2rffv2WSE5blcODg5KTk4uV9G9ePFivfDCC2rRooWcnZ0VExOjnJwc7dy5swqSoiwU3TDLysqSu7u76tSpY173t7/9TQcOHFBBQcFVbVu1amVetrW1laenpwwGQ5Xlxe2vImMqLCxM/fv3Ny+fPXtW586dk5ubW5Xlxe2vImNK+vOfOW+//bbGjh173Y/yQPVVkTG1fft2tWzZUm+++ab8/f0VEhKi5cuXV3Vk3AEqMq46d+6srVu3au/evSoqKtK3336rCxcuqF27dlUdG7ex5557Ti4uLjdsV1hYqOzs7FJ/pzs7O6tZs2b8nW5FFN0wO336tGrXrl1q3eVfFr///vtVba/8RXK57X+3Q/VWkTF1JZPJpNGjR8vHx4c/OlBKRcdUXFycfH19FRAQUCX5cOepyJg6duyYvv32W3Xo0EEbNmzQ4MGD9c9//lN79uypsry4M1RkXD3xxBN65plnFB4eLi8vL7322mt699131bBhwyrLi7vHmTNnZDKZ+Dv9NsO//VFKRe714L4QlEdFx0lxcbFGjRql7OxsLViwwEKpcCcr75jKzs5WUlKSVqxYYeFEuNOVd0yZTCb97W9/U69evSRJffr00aJFi5SWllbqqhIglX9cLVu2TMuWLVNSUpI8PDy0adMmvfbaa2rYsKG8vb0tnBJ3K/5Ov71wpRtmrq6uOn36dKl1p0+flo2NjVxdXUut/8tf/lJm2/9uh+qtImNK+nNK1ODBg3XkyBEtXLhQ9evXr6KkuFOUd0yZTCaNGzdOw4cP17333lvFKXEnqcjPqXvvvfeq6Z3u7u46ceKEpWPiDlORcfXZZ5/pmWeekbe3txwcHNS5c2cFBARw6wJuSt26dWVra1vm+KtXr551QoGiG//n4Ycf1tGjR5Wfn29eZzAY9Ne//lW1atW6qm1WVpZ52Wg0as+ePfLx8amyvLj9VWRMmUwmxcTE6J577tH8+fP1l7/8parj4g5Q3jF15MgR/fDDD5o+fbrat2+v9u3bKzU1VQkJCerTp481ouM2VZGfUy1atND+/ftLXUHKzc2Vu7t7leXFnaEi46qkpERGo7HUuqKioirJibuPg4ODHnzwwVJ/p589e1a//fYbMyesiKIbZq1atZKXl5emTJmigoIC5eTkaN68eerXr58kKSQkRNu2bZMk9evXT8uWLVNmZqYuXLigmTNnyt7enqcDo5SKjKkVK1YoOztbH374oRwcHKwZG7ex8o6pBg0a6LvvvlNKSor5KygoSJGRkZo9e7aVzwK3k4r8nOrdu7d+//13zZo1S4WFhVq5cqWysrLUu3dva54CbkMVGVdBQUFKTk7WTz/9pEuXLmnjxo3atGmTunbtas1TwB0kLy9PISEh5o9Z7devnxYsWKCcnBwVFBRo8uTJ8vT0lJeXl5WTVl/c041Spk+frjFjxuiRRx6Rs7OzIiMjzU+UPnDggPlzuB977DG9+uqrGjFihE6dOiUvLy/Nnj1bjo6O1oyP21B5x9QXX3yh3Nzcqx6cFhYWptjY2CrPjdtXecaUnZ2dGjRoUGo/JycnOTs7M90cVynvzyk3Nzd9/PHHeueddxQfH69GjRopLi5OTZs2tWZ83KbKO64GDx6sS5cuaejQocrPz5e7u7tiY2MVGBhozfi4zVwumC9duiRJWr16taQ/Z1AUFxfrwIED5hkSkZGROnHihAYOHKhz586pffv2mjFjhnWCQ5JkY+IuewAAAAAALILp5QAAAAAAWAhFNwAAAAAAFkLRDQAAAACAhVB0AwAAAABgIRTdAAAAAABYCEU3AAAAAAAWQtENAAAAAICFUHQDAAAAAGAhFN0AAOC63nnnHfn5+Wn27NnWjnKV4OBgJSUlWTvGVW7XXABQnW3YsEEdOnRQTExMhffdsWOHIiIi5O3trSeeeEIrVqwo9742JpPJVOEjAgCAO9bAgQPl4+OjkSNH3rDt6dOn1b59e82cOVNBQUFVkO76Dh06pKysLIWEhFg7CgDgDjJnzhwlJyfL1dVVDRo00NSpU8u97/Hjx9WzZ0/961//UmhoqDZv3qxJkybps88+U926dW+4P1e6AQDANZ07d06S1KxZMysn+VN6erq+/vpra8cAANxhHBwclJycfM3fZ1999ZXCwsLk6+urrl27avHixeZtS5YsUevWrRUeHi4HBwd16tRJK1euLFfBLVF0AwBQrR0+fFgeHh7KyMhQeHi4fH19FRkZqcOHD+vAgQMKDg6WJIWFhSk+Pl6StGjRIoWGhsrHx0chISH66quvzP0NHDhQkyZNUq9evfTyyy9Lkjw8PJSammqelvfyyy/r2LFjioqKkp+fnyIiInT48GFzH/Pnz1e3bt3k5+en0NBQpaenS5Lmzp2ryZMnKy0tTV5eXjIajQoKClJiYqIkqaSkRHFxcXr88cfl7e2tPn36aNOmTeZ+g4KClJSUpJdffll+fn7q1q2bNm7ceM33JikpSR06dJC/v78mTZqkt956S6NGjZIkmUwmTZ48WZ06dZKfn5/69OmjH374odSxLucaNWqUxo8fr3fffVft2rVTQECA5syZc/PfNABAhT333HNycXEpc5vBYNBbb72l119/Xdu3b9f777+v9957Tzt27JAkbd++XU2aNFF0dLTatGmjsLAwZWRklPvYFN0AAEALFizQxx9/rHXr1un8+fNKSEjQ/fffr7S0NElSSkqKoqOjtWbNGk2aNEnjx4/Xtm3b9I9//EOvv/669u3bZ+4rNTVV77zzjj7++GPzukWLFmnWrFlavny5Nm3apJdeekmvvfaaNmzYIKPRqHnz5kmSfvjhB02ZMkXx8fHasWOHXnrpJY0cOVL5+fmKiopSWFiYQkJCZDAYZGdnV+ocFi5cqKSkJM2YMUPbtm1Tr169FB0drVOnTpnbzJ07V8OGDdOWLVvUrl07TZgwocz3IysrS2PGjNHYsWOVkZEhJycnffPNN+btKSkpWrZsmRYvXqxt27apa9eu+sc//iGj0VhmfytXrtRDDz2kjIwMvf7665o6daqOHz9ewe8SAMASli5dqs6dO6tjx46ys7OTv7+/QkNDlZKSIkk6duyYli9frgEDBmjDhg0KCQnR0KFDlZeXV67+KboBAID69esnNzc31a1bVx07dlROTk6Z7ZKTk9WzZ0/5+/urRo0a6t69uzw9PUtN+fb29pa3t7dsbGzM63r06KH77rtPzZs31wMPPCAvLy+1atVKzs7OateunQ4ePChJatOmjTIyMtSyZUvZ2NioZ8+eunjxovbv33/Dc0hOTlb//v3l4eEhe3t7DRo0SE5OTlq3bp25TZcuXeTt7S17e3sFBwfr4MGDKikpuaqv9evXy8PDQ8HBwXJwcNCQIUPk5ORk3t6rVy+tWrVKDRo0kJ2dnXr06KH8/HwdOXKkzGyNGzdWnz59zO+Z0Wg0nzMAwLp+++03ff311/Ly8jJ/LV++3FxUm0wmderUSR06dFDNmjU1ePBgubi4lPr9cj33WDA7AAC4QzRu3Nj82snJSRcvXiyz3eHDhxUQEFBqXbNmzZSbm2tednd3v2q/hg0bml87ODjIzc2t1HJRUZEkyWg0Ki4uTmlpacrPzze3ubz9eg4fPqwWLVqUWte0adNS2a48T0dHRxmNRhUXF8vBwaHUfidOnCh1HnZ2dmrVqpV5+cKFC5owYYLWr1+vM2fO3DDnf7+/klRYWHjDcwIAWJ6jo6P69eunMWPGlLn93nvvVe3atc3Ltra2atSokU6cOFGu/rnSDQAASl2Vvp5rFZVX7v/f077L6t/Wtuw/QeLi4rRq1SrNnDlTO3fuVGZmZrlylTfbtY7730pKSnTPPaWvTVy573/+8x9lZWVp4cKFMhgMpe5rL0t5jwsAqHpNmzYtdZuU9OeU8su3DLVo0UJ79+41bzOZTDpy5EiZ/2QuC78BAABAuTVt2lS//PJLqXW//PKLmjRpUin9GwwGde3aVa1atZKtra2ysrJuOtulS5f066+/3lS2evXqlZoqbjQatWfPHvPyrl271Lt3bzVv3lw2NjYVygkAuL089dRT2rFjh7744gsVFRVp7969evrpp823TvXt21eZmZn68ssvdfHiRc2dO1cXL15Ut27dytU/RTcAACi3sLAwrVixQpmZmSouLtbSpUv1888/q0ePHpXSv7u7u3766SdduHBB2dnZSkhIkIuLi/m+OgcHBx09elRnz57VpUuXrsr2+eefKycnR0VFRZo1a5b5CecVFRAQoN27d2vdunUqKirSzJkzS00Hb9y4sQwGg4qKipSZmanU1FRJ4uFoAHCbunyvdkpKivlTMLy8vCT9eSV7ypQpSkhIkL+/v4YPH66oqCh1795dktSqVSt98MEHmjVrlvz9/bVy5Urz76fy4J5uAABQbj169FBubq7eeOMNnTx5Ug888IA++eQTNW/evFL6Hzx4sGJiYhQQEKAHH3xQ7777rtzc3BQbGytXV1f16tVLaWlp6tKli1asWFFq30GDBun333/XSy+9pLNnz8rT01MLFiwodR9eebVt21YjRozQyJEjVaNGDT3//PNq3769ear6a6+9pjfeeEPt2rWTj4+PJk6cKEmKjo7WZ599dutvBACgUhkMhutuDw0NVWho6DW3BwcHmz9Gs6JsTCaT6ab2BAAAuIsVFRXJ3t7evDxgwAD5+/trxIgR1gsFALjjML0cAADgvxw6dEh+fn5as2aNSkpKtHHjRv3444967LHHrB0NAHCH4Uo3AABAGVasWKH4+HgdPXpUbm5uioqKUt++fa0dCwBwh6HoBgAAAADAQpheDgAAAACAhVB0AwAAAABgIRTdAAAAAABYCEU3AAAAAAAWQtENAAAAAICFUHQDAAAAAGAhFN0AAAAAAFgIRTcAAAAAABZC0Q0AAAAAgIX8P4zakzbjOG5kAAAAAElFTkSuQmCC
)
    


             feature   importance
            s5_score 1072311.9034
           tfidf_sim  123980.9660
        n_retrievers   35544.2821
    recent_tfidf_sim   20091.2337
               in_s3    3077.7619
             rank_s2    1791.9725
       u_click_count    1033.4886
        m_log_clicks     612.8205
          m_log_impr     486.6427
               in_s2     406.9860
       ctr_norm_rank     231.7343
             rank_s3     176.9706
    article_age_days     160.2546
       subcat_clicks     128.9271
               in_s4      90.8844
       m_article_len      54.0000
        u_click_freq      33.2342
        cat_affinity      24.5358
      taste_affinity      18.8506
            imp_size      15.1181
             rank_s4       1.2116


> **🔍 Reading the meta-ranker feature importance:**
>
> Feature importance by **gain** measures how much each feature reduces the ranking loss *on average when it is used as a split*. High-gain features are the model's primary decision levers.
>
> **What a healthy importance distribution looks like for this pipeline:**
>
> | Expected rank | Feature | Why |
> |---------------|---------|-----|
> | 1–2 | `tfidf_sim` or `s5_score` | Content relevance and base LGB scores are the strongest signals |
> | 3–4 | `cat_affinity` / `taste_affinity` | Category preference is reliable for warm users |
> | 5–6 | `m_log_clicks` / `bayesian_ctr` | Popularity has broad coverage |
> | 7–9 | `n_retrievers` / retriever rank flags | Ensemble metadata (how many retrievers agreed) |
> | Low | `u_click_freq` / `active_days` | User engagement features are useful but secondary |
>
> If `s5_score` ranks #1 by a wide margin, it suggests the meta-ranker is largely *distilling* the base LGB rather than learning genuinely new patterns — consider adding features that the base LGB cannot see (e.g. session-level context, recency of the article relative to the session).

[Back to top](#top)

---

## <a id="sec-12"></a>12. Leaderboard & takeaways



```python
# Master leaderboard printout
for k_val in [5, 10]:

    print(f'{"="*70}')
    print(f'  LEADERBOARD  @  K = {k_val}')
    print('='*70)
    lb = leaderboard[leaderboard['K'] == k_val].copy()
    lb[metric_keys + ['composite']] *= 100
    print(lb[['strategy'] + metric_keys + ['composite']].to_string(index=False, float_format='%.2f'))
    print()

```

    ======================================================================
      LEADERBOARD  @  K = 5
    ======================================================================
                    strategy  precision  recall    f1  ndcg  hit_rate  composite
           S5: LightGBM Base      10.70   42.62 16.47 29.34     49.50      39.42
              S1: Popularity       9.08   36.58 14.07 25.40     42.30      33.85
      S6: Meta-LGB (2-Stage)       8.36   33.76 12.92 23.35     39.40      31.38
    S7: Ensemble (LGB + XGB)       8.36   33.76 12.92 23.35     39.40      31.37
                 S3: Item-CF       8.06   33.17 12.55 22.58     38.40      30.49
       S2: Category Affinity       7.80   32.36 12.19 21.97     37.30      29.64
          S4: Temporal Taste       7.80   32.36 12.19 21.97     37.30      29.64
    
    ======================================================================
      LEADERBOARD  @  K = 10
    ======================================================================
                    strategy  precision  recall    f1  ndcg  hit_rate  composite
           S5: LightGBM Base       7.82   58.87 13.38 35.06     66.50      50.78
              S1: Popularity       7.19   55.79 12.39 32.09     63.80      47.94
      S6: Meta-LGB (2-Stage)       6.71   52.47 11.58 29.83     59.70      44.77
    S7: Ensemble (LGB + XGB)       6.70   52.42 11.56 29.81     59.60      44.70
                 S3: Item-CF       6.39   51.14 11.10 28.78     58.10      43.44
       S2: Category Affinity       6.32   50.59 10.98 28.30     57.60      42.95
          S4: Temporal Taste       6.32   50.59 10.98 28.30     57.60      42.95
    



```python
# Lift metrics
for K in [5, 10]:

    base = all_results[('S1: Popularity', K)]['composite']
    best = all_results[('S6: Meta-LGB (2-Stage)', K)]['composite']
    lift = (best - base) / base * 100
    print(f'K={K}: S1 composite={base*100:.2f}%  →  S6={best*100:.2f}%  '
          f'(+ {lift:.1f}% relative lift)')
```

    K=5: S1 composite=33.85%  →  S6=31.38%  (+ -7.3% relative lift)
    K=10: S1 composite=47.94%  →  S6=44.77%  (+ -6.6% relative lift)


[Back to top..](#top)

---

### 🏆 Key takeaways

### What Was Built

```
MIND-small Dataset (160K users, 65K articles, 1M+ impressions)
         │
         ▼
Feature Engineering ──► user_stats · article_feat · category affinity · TF-IDF centroids
         │
         ├──► Stage 1 Retrieval ──► 200-candidate pool (4 complementary retrievers)
         │
         └──► Stage 2 Reranking ──► Base LGB (LambdaMART) + Meta-LGB + XGB Ensemble
                                            │
                                            ▼
                              S1→S7 Leaderboard (NDCG · HR · P · R · F1 @ K=5,10)
```

### Design Decisions Recap

| Decision | Alternative | Trade-off |
|----------|-------------|-----------|
| Two-stage generate & rerank | Single-stage end-to-end | Lower inference cost; established industry standard |
| LambdaMART objective | BPR / pointwise logistic | Directly optimises NDCG; needs query groups |
| Bayesian CTR smoothing | Raw CTR | Prevents low-impression articles from appearing falsely viral |
| Per-impression evaluation | Global ranking evaluation | Matches deployment; prevents popularity dominance |
| OOF split for meta-ranker | In-sample scoring | Prevents leakage; gives honest meta-feature estimates |
| 7-day decay half-life | Fixed window | Smoother than hard cutoffs; tunable to domain |

### Potential Extensions

1. **Neural text encoder** — Replace TF-IDF centroids with a fine-tuned BERT/DistilBERT news encoder (e.g. the NAML or NRMS architectures from the MIND paper) for richer semantic representations.
2. **Session context** — Add within-session features: position of the candidate in the impression list, time since last click, number of articles already clicked in this session.
3. **Graph-based CF** — Use LightGCN or PinSage over the user–article bipartite graph for higher-quality embeddings, especially for sparse users.
4. **Online evaluation** — A/B test against a production system; offline NDCG gains do not always translate 1:1 to online CTR improvements.
5. **Diversity regularisation** — Add a category-diversity penalty to the final top-K selection to avoid filter bubbles (e.g. maximum marginal relevance).
6. **Freshness feature** — Articles less than 1 hour old should receive a freshness bonus; MIND's fixed 6-week window masks this but it matters in production.

---
*Cite MIND: Fangzhao Wu et al. (2020), "MIND: A Large-scale Dataset for News Recommendation", ACL 2020. Dataset: https://msnews.github.io/*
---

[Back to top..](#top)
