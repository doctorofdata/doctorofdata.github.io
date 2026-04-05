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

In this post, we build a series of recommendation engines for the Microsoft MINDS dataset using popular heuristic strategies and a combination of machine learning algorithms.

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
from xgboost import XGBClassifier, XGBRanker
from joblib import Parallel, delayed
from google.colab import drive
import zipfile
import joblib

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

    # Calculate composite score which avoids double counting f1 or recall
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

    # Removed is_cold gate.
    # The meta-ranker trained on all SET_B impression rows, including thin-history
    # users. Falling back to s1_score for click_count < 2 quietly scored those
    # users below S5, which always runs the ML model regardless of history depth.

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

    # Build retriever lists without the seen-filter.
    # During meta-training the seen-cache was cleared before computing retriever
    # lists, so all articles were reachable. At inference we must match that:
    # pop this user's seen-set before calling the retrievers, then restore it.
    # Without this, in_s2/rank_s2/... are computed on a different article
    # distribution at inference than training — a systematic feature mismatch.
    _uid_seen = _seen_cache.pop(uid, None)
    cands_s2  = s2_category(uid, N_STAGE1)
    cands_s3  = s3_itemcf(uid,   N_STAGE1)
    cands_s4  = s4_temporal(uid, N_STAGE1)
    if _uid_seen is not None:
        _seen_cache[uid] = _uid_seen

    X_meta = _build_meta_features(uid, candidates, cands_s2, cands_s3, cands_s4, s2_vec, s4_vec, base_scores)
    scores = meta_lgb.predict(X_meta)

    return [candidates[i] for i in np.argsort(-scores)]

def s7_score(uid, candidates):

    # Bug fix 1: removed is_cold gate (same reasoning as s6_score).
    # Bug fix 2: disable seen-filter for retriever lists (same reasoning as s6_score).

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

    _uid_seen = _seen_cache.pop(uid, None)
    cands_s2  = s2_category(uid, N_STAGE1)
    cands_s3  = s3_itemcf(uid,   N_STAGE1)
    cands_s4  = s4_temporal(uid, N_STAGE1)
    if _uid_seen is not None:
        _seen_cache[uid] = _uid_seen

    X_meta     = _build_meta_features(uid, candidates, cands_s2, cands_s3, cands_s4, s2_vec, s4_vec, base_scores)
    lgb_scores = meta_lgb.predict(X_meta)
    xgb_scores = xgb_meta.predict(X_meta)
    scores     = _borda_blend(lgb_scores, xgb_scores)

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
    X_meta      = _build_meta_features(uid, candidates, cands_s2, cands_s3, cands_s4, s2_vec, s4_vec, base_scores)
    lgb_scores = meta_lgb.predict(X_meta)
    xgb_scores = xgb_meta.predict(X_meta)
    scores     = _borda_blend(lgb_scores, xgb_scores)

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

# Function to blend scores
def _borda_blend(lgb_scores, xgb_scores, w_lgb = 0.6, w_xgb = 0.4):

    """Rank-based (Borda) blend of two ranking score vectors.
    Both models are now LambdaRank / rank:ndcg — their raw scores are on
    different scales and should NOT be averaged directly. Converting to
    ranks first makes the blend scale-invariant."""

    n = len(lgb_scores)
    lgb_ranks = np.argsort(np.argsort(-lgb_scores)).astype('float32')
    xgb_ranks = np.argsort(np.argsort(-xgb_scores)).astype('float32')

    # Higher rank-score = better; invert so rank 0 (best) gets n-1 points
    return w_lgb * (n - lgb_ranks) + w_xgb * (n - xgb_ranks)
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






  <div id="df-75856ae9-1a5b-494a-bc93-a827d85e7eed" class="colab-df-container">
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
    <button class="colab-df-convert" onclick="convertToInteractive('df-75856ae9-1a5b-494a-bc93-a827d85e7eed')"
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
        document.querySelector('#df-75856ae9-1a5b-494a-bc93-a827d85e7eed button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-75856ae9-1a5b-494a-bc93-a827d85e7eed');
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





  <div id="df-0fa6b2b0-4e8d-4917-ac4f-2f1b59882910" class="colab-df-container">
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
    <button class="colab-df-convert" onclick="convertToInteractive('df-0fa6b2b0-4e8d-4917-ac4f-2f1b59882910')"
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
        document.querySelector('#df-0fa6b2b0-4e8d-4917-ac4f-2f1b59882910 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-0fa6b2b0-4e8d-4917-ac4f-2f1b59882910');
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






  <div id="df-f29ab227-9d55-48d2-8e9a-70da6a2ece11" class="colab-df-container">
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
      <td>{N496, N37204}</td>
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
    <button class="colab-df-convert" onclick="convertToInteractive('df-f29ab227-9d55-48d2-8e9a-70da6a2ece11')"
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
        document.querySelector('#df-f29ab227-9d55-48d2-8e9a-70da6a2ece11 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-f29ab227-9d55-48d2-8e9a-70da6a2ece11');
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






  <div id="df-49d07bcf-90f4-483f-a45a-20f7c0173725" class="colab-df-container">
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
    <button class="colab-df-convert" onclick="convertToInteractive('df-49d07bcf-90f4-483f-a45a-20f7c0173725')"
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
        document.querySelector('#df-49d07bcf-90f4-483f-a45a-20f7c0173725 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-49d07bcf-90f4-483f-a45a-20f7c0173725');
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

    ax.text(bar.get_width()*1.01, bar.get_y()+bar.get_height()/2, f'{row["ctr"]:.2%} CTR', va='center', fontsize=8)

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


    
![png](data:image/png;base64,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tr1v3z41adLEcp1g1apVGjRokNXoc4ZhaPbs2apdu7ZKliypChUqqFOnTlavEJwyZYrKlSunLVu2qEqVKurcufMznxMASA0oigMAXjubN29W+vTpVb58ecu80NBQffrpp7Kzs9Py5cu1ceNG+fr6avDgwZYCcywfHx8dPHhQV69efep9L1u2TOnSpdOyZcvUrVs3rV69Wp988omio6O1aNEide7cWatWrbJKaKWHiUr+/Pm1fPlyffPNN9qzZ0+cYdx///13pU2bVuvWrVPFihUVGRmpTz75RPv27dOQIUO0bt06NWrUSAMGDNDq1aslPSzWdu3aVbVr19aaNWu0YsUKlSxZUp9//rkuX74sSWrXrp1cXFwUGBiojRs3Wu7C37hxY7zH+P3336tv376qVauWVq9erfnz5ysqKkotW7bUlStXrNoOGTJEH3zwgdauXauPPvpI3333nTZt2iRJ2rlzp4YMGaJPP/1UGzZsUGBgoFxdXdW6dWs9ePAg0fM8YsQIVatWTT/88IMGDhyo9evXW5JwSfrqq680Z84cffLJJ1q/fr38/f21YsUKDR482Go7x48f17lz57Ry5Uo1atQo3n2dPXtWrVq1Ut68ebV8+XJNnTpVhw8ftgzDZxiG5XzOnz9fP/30k7p06aJp06Zp0aJFkqT+/furZs2aypEjh7Zv367WrVtLkjp27Kj169erS5cu2rBhg9q2baupU6daXRwaMmSIfv75Zw0aNEgrVqxQtmzZNHToUKsYT548qU8++UTOzs5auHChfvjhB5UtW1bdu3fXli1brNrOmTNH77//vn7++Wc1adJEjo6Olv4Sa9OmTXJwcFCDBg0S/R4AAAAApE7x5d2xJk6cqHLlylkKeT/++KPGjBljWW5jY6MaNWro999/T7To+ddff8nLy0tOTk7xLndyclKWLFmeKN7IyEi1adNG58+f17fffqvly5db8s/4cv/w8HC1b99eadOm1fTp02Vvbx/vdr///ntlypRJ7777rrZv3y4PDw/LstGjR6t+/fpavXq1WrdurUWLFmnNmjVW6/83P0ubNu1T5XaP8yS5ZKzhw4frs88+0+rVq/Xuu+9qypQp2rlzZ6Lbv3XrlrZs2aLAwEB98cUXkqQePXpo7969mj59ujZv3qzBgwdr5cqVlmHuW7duLT8/P0kPh7fv37+/pCfP4eMz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qY4dO2rChAmaNWuWVZvjx4/L399fEyZMUNGiRZNy6gE8IbPZbDXdo0cPderUSX379tXs2bP16aefWpbFxMTIMAzLOmazWTY2NpZpwzC0ePFirVy50rKsXr168vPzU+XKlbVjxw4VLlxYZ86c0RtvvKH3339fHTp0UNmyZa1icHJyUlRUVJzYHvVoPLExjB07VgEBAQoPD9eVK1fUtWtXVahQ4bHbw8vz3z4EpFT0c6QG9PPUje/9+fLy8tKBAwfk4+OjiIgInT9/XiaTSQ8ePJCTk5P27t0rLy8vdejQQZ6enpowYYLlBvavv/5aw4YN04ULF+Tv7685c+bI29tbu3btUuvWreXp6al3331XJ0+e1LZt2yyvTJIeFtvfeustbdu2Td9//71GjBihBg0aKHPmzOrTp4/s7Oy0YcMGmUwmDR48WH379tXcuXMt6//111/69ddflTZt2uQ4bQAAAHhFkS88G/JsJEVC/edJ+1OS3yke648//tCDBw/UqVMn2dvbK1++fPr44481e/Zs9e7dWytXrpS3t7dq1aolSfL29lb16tW1ceNGVatWTZs3b1bz5s2VPXt2SVLbtm31448/WrYfEBAgGxsb2dvbK1OmTKpatar2799vFYOvr69l/SpVqlje+71582a99dZblsLUF198YfWU+uP2LUlVq1ZV/vz5Ezz+Ro0aqUCBApIePlk+Z84cq+U3b95U+/bt1atXL1WpUuUJzyqApAoKCpL08EkPs9ls+Xfs5uamzZs3y8PDw9I2NDRUZ86csVonU6ZMlunjx4/L1dVVZ8+etaxTuHBhde/eXffv39fYsWPVr18/bdq0SQUKFNChQ4d069Yty/qxsmTJovXr18cZKv3EiRMqXLiwZfrKlSuKioqyWj80NFTvvfeeypQpo/DwcPXt21cmkynOPvBqiP09BKRk9HOkBvRzIOkqVqyoVatWSZL+/vtvFS5cWHZ2dvr7779VsWJF7d27V5UrV9aWLVu0ZMkSOTk5ycnJSZ06dVKbNm00dOhQ5cmTR7t27bKMsOTt7a3MmTPrn3/+SfDG8zx58ui9996TJNWrV09fffWVzp8/L+nhSG4bN260bK9nz56qXr26rl27Zlm/cePGcnFxeWHnBQAAAK8nrscmDXk2kuJZ+89zK4pfvHhRefPmlb29vWVe/vz59e+//yomJkYXL15UwYIFrdbJnz+/pbB95coV5cmTx7LsjTfesGp76NAhjR8/XseOHbM8YVmyZEmrNv9d38nJyTK8+dWrV/Xmm29alqVJk8aq7eP2LUm5c+dO9PgT2rckRUdHq3PnzsqWLZvV8HAAXjx3d3dJD/+dL126VPPnz5ednZ1++ukneXp6WpZLUvHixRUQEKAMGTLojTfe0K+//qq6deta2uzfv18VK1a0WifWkCFD5O/vrwoVKuj8+fMyDEOlSpWSvb19nPa9evVSmzZt1LRpUxUvXlyStHLlSn3//fdatGiR7Owe/mjevn27IiMjrdZ3cXFRgQIFLPMGDx6sKVOmaNmyZZb1kPzMZrOCg4NVqlQpmUym5A4HeCHo50gN6OepW1hYmI4fP57cYaQYXl5eGjp0qKKjo7Vnzx55enrKxsZG+/btU9myZS3FcbPZLC8vL6t1zWazbt26pUyZMmnJkiX6/vvvFRISIsMwFBkZqcjIyAT3+99c3dHRUZIUGRlpGV69cePGVu1NJpMuX76sTJkySZJy5cr1PA4fAAAAKUx814jxeOTZSIqE+s+T5u/PrYKSUBIa+yTk45Y/OpSwYRiWz3fu3NHnn38uX19fzZo1Sy4uLpo4caJ27NgR77biiy06OtpqXkzM/95rkdi+Yz3uH2dC+5Yevqcsa9as2rp1q3799Vf5+Pgkui0Az0/sv926devqyJEj8vPzk8lkUpYsWTRixAiZTCbLEIqlS5fW6NGjLU9fZ82aVSNHjrRs4+rVq8qZM2ecnwcHDhxQdHS0ZRSIunXrqkOHDlq5cqU+/vjjOO0LFy6sadOmadSoUQoPD5e9vb0KFCigOXPmyMHBwdLO1tZWNjY2Vuvb2NjI1tbWMq9mzZpat26dZs2apY4dOz7/E4gkMZlM/HGHFI9+jtSAfp468Z0/X7lz51aWLFl0+PBh7d27V23btpUkzZkzR4cPH1amTJlUuHBhOTs768CBA/FuY8WKFQoICND06dNVvnx5mUwmVatWLdH9JpSrxxbI//jjD2XMmDHO8osXL0qiHwAAACB+/J2YNOTZSIpH+8+T9qXnVhTPmzevLly4oMjISMvT4qdPn1aePHlka2urfPny6fTp01brnD59Wnnz5pUkZcuWTZcvX7YsO3nypFW7+/fvq02bNpZhyw4fPvzEsWXLlk1XrlyxTP/3rvDH7ft5yJQpkyZMmKBFixZp4MCBcnd3t9x1DuDlsLGxUc+ePeNd1r9/f8tnb29veXt7x9tuwIAB8c738PCwGoY9Xbp0CgwMTDSeUqVKafHixYm26dSpU5x58W134sSJiW4HAAAAeBV4eXlpz549Cg4Olru7u2JiYhQcHKw9e/bI29tb+fLlU1hYmC5cuGC5VhAaGqqoqChlzJhRwcHBKleunCpWrChJunbtmkJCQp4plty5c8vW1lbHjh2zbC8qKko3b960vFoNAAAAAJBy2D6vDb311luys7PTtGnTFBkZqdOnT2vBggWWocgaNmyoP//8U7/99puio6O1bds2bd261bK8atWqWr58ua5du6abN29q9uzZlm3nypVLtra2OnDggMLCwjR//nxdv35d169fj/MEeEKxbd++XQcPHlR4eLimTp1q9aR4Yvt+HmxtH57m5s2bq3DhwhoyZMhz3T4AAAAAAK+6ihUrasWKFXrjjTfk7OwsFxcX5cyZUz/88IO8vb1VpEgReXh4aMSIEbp586bu3r2rwYMHq3fv3pIeFrJPnz6tO3fu6NKlSxo+fLhy5cqlq1evSpIcHBx07do13b59O9Eh1aWHN7LWrVtX48aN05UrVxQeHq5vvvlGrVu3jnf0OAAAAADA6+25FcXTpk2rgIAAyx3ebdu2VaNGjdSuXTtJsiS248ePV/ny5TVmzBiNGzdOFSpUkPTwHbtvvvmm6tSpo/fff1/vvfee5f242bNnV/fu3dWvXz/VqFFDd+7c0bhx4xQZGanmzZs/NrZ3331XLVu2VLt27VStWrU47/hNbN/Pk42NjUaOHKk///xTa9asee7bBwAAAADgVeXl5aUzZ86obNmylnmenp46deqUZbSm8ePHyzAM1axZU2+//bbMZrNGjx4tSfL19VX+/PlVrVo1ff755/Lz85Ofn5/mzZunRYsWqVatWjIMQ9WrV9ehQ4ceG8/AgQOVP39+1atXT1WrVtXJkyc1ffr0RF+PBgAAgLgMWxudqpBBpypkkGHL31IAXk02BrdAvzLCwsJ05MgR/XNmmMIizid3OECK0O7j/ckdAlIhs9msoKAgubu7824cpFj0c6QG9PPULTY/K1asmJydnZM7HLxiyN8BAABSN647PxvybCRFQv3nSfP35/akOAAAAAAAAAAAAAAArxqK4gAAAAAAAAAAAACAFIuiOAAA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6M0335Svr6+qVq2qyZMnKzk5WZI0bdo0de3aVb1791a5cuU0cuRIhYWFad68eapbt65lG7GxserZs6el77tv3z7Lsl27dqlFixby9fVVtWrVNHXqVMuyNWvWqHHjxlqxYoWqVKkiPz8/hYWF6bvvvtNrr72mcuXKadiwYZb28fHxGjFihGrWrCkfHx+1a9dOv/32m9WxLliwQFWrVtXs2bNT9fF/+eUXvfXWW/Lx8VG9evW0efPm+4oTAAAAAJB12XVRvE+fPqpcubL27dunVatWafv27Vq+fPld12nQoIG6deumMmXKKDIyUsWKFdPSpUv15Zdfavr06Tpw4IAaN26s4OBgXb16VRcvXtSIESM0depURUREaNq0afriiy+sLj4AD4vRaFTu3LklSWfPntWOHTvUsGFDqza3h0zfvHmz2rVrp44dO1ouIJUuXVpbt26VJP3444/6559/dP36dSUnJyshIUGnTp1S0aJFNWHCBHXv3l1z5szRoEGDtGXLllSxHD16VN7e3qnm582b94GPKy4uTlu2bFG5cuUeeF0AAAAAKUJDQzVq1Cjt2bNHBQsW1OTJkyVJf/75pzp27KimTZtq3759mj17tlatWmX1jPCIiAj5+flp//79GjJkiCpWrKgOHTpYRp2SpFWrVqlTp07at2+fKlSooJCQEEkpn+d79OihNm3a6ODBgwoNDdX8+fP17bffWtaNjo7WpUuXtH37dr3zzjsaP368NmzYoP/973+aNWuWli9friNHjkiSJkyYoKNHj2rFihXau3evvL291b17d5nNZsv2tm3bprVr16pz585W5+DmzZvq2rWrXnvtNf34448aOnSoBgwYoFOnTt1XnAAAAEiDq6t0+XLKj6urraMBgDTZ9TPF165dKycnJxmNRhUpUkQVK1bUkSNHVKlSpQfazqpVq9S2bVt5eXlJkjp06KDQ0FDt2LFDZcuWVXJyslz//z+CV155RXv27JGDg11/HwEPmclksnp9/Phx9ezZUyEhISpcuLDVcpPJpJs3b8rT01Nt2rTR+vXr1b9/f61atUohISEaO3as/ve//6lixYp67rnn5OjoqPbt2+vdd9/VCy+8oFu3bik2NlZ58uRRQkKCQkJC1LlzZ9WpU8cqr11cXJSYmJgqttuvb/+bnJwss9ls1c5sNmv+/PnatGmTkpKSdOrUKXXp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27Ztq/bt28toNMpkMql9+/YKDw9X3bp1H+h8AAAAAMiaFi5cKB8fH02ZMkUODgxiBwAAkKU5OkqNGtk6CgC4K3qegJ3o2rWrunTpIoPBYJl348YNFShQQJJUqFAhXb16VefOndPu3bsVGBiY7rZmzZqlDh06WAriUkqhe9GiRVbbvx9Go1He3t46ceLEAx4RAAAAgKwqJiZG7u7uFMQBAAAAAI+F3fU+vby8tGnTJrVo0UJlypRRly5ddPHiRXXs2FG+vr5q0aKFzp8/LynlLtc333zTav0qVapozZo1kqSff/5Zb775pnx9fVWpUiV99NFHio+PlyQFBARo2bJlkiSTyaQJEyaoSpUqqlixonr16qUbN248voNGthEdHa0ePXrorbfe0vjx45WUlJSqTUxMjPr27avWrVtr6NChiomJkSTlyZMnVdv8+fPr0qVLMpvNOnPmjIoUKaKQkBANGDBAY8aM0cCBA/Xjjz+mWu/o0aPy9vZONT9v3rwPfEzXrl3T9u3bVa5cuQdeFwAAAEDWExQUpP3792vevHmqV6+evLy8dOvWLUkpffatW7eqTZs28vHxUePGjXX06FHLuuvXr1fDhg3l6+urgIAAhYWFWZZNmzZN3bp105w5cyz975CQEMvymzdvasiQIapUqZIqV66sIUOGWB4HFR8frxEjRqhmzZry8fFRu3bt9Ntvvz2mMwIAAJDFJSZKCxak/CQm2joaAEiTXQ6fvnz5cs2aNUtxcXFq3LixOnfurLFjx8rd3V2BgYGaP3++hgwZcs/tfPjhh+rUqZNatmypP//8U8HBwVqxYoXat29v1W7x4sUKDw/XihUrlC9fPvXp00cjR47UxIkTH9UhIpsxmUwqVqyYunfvrgYNGshsNqt79+5atmyZ2rZta2n31FNPqU+fPmrUqJFy5sypoUOHavr06erfv7+ljdlsVnJyskwmkwwGgxo1aqSgoCBVqVJFu3fvlqenp86cOaPixYurVatWCg4OVvny5a3icXFxUWJiokwm013jTk5OltlstmpnNpsVFhamb775Rrdu3dLFixfVu3dv+fn53XN7QFZxO5fJaWRn5DnsAXlu3/i9PzpLlixRu3btVLZsWVWrVk1vv/221fLQ0FB9+umnKly4sLp3767Jkydrzpw5ioqK0oABAzR37lz5+/tr79696tChg8qVK6cX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OHBAf//9d7oF8Vu3bqlz58768MMPVbduXUkpRfeuXbtq7dq1mjx5siRp3759mjhxohYvXmxZ9+jRo3r99dfVr18/y7zJkydr+vTp+vjjjzN/ogAAAABkeTExMSpbtqwcHBg0DgAAAABgX7JdT9jLy0ubNm1SixYtVKZMGXXp0kUXL15Ux44d5evrqxYtWuj8+fOW9tu2bVOTJk3k4+OjgIAALVq0SJIUFhamgIAAq20fPXpUpUuX1qVLl9SuXTtNmDDBsmzJkiVq0KCBypYtq0aNGmnbtm2WZTt27FDjxo3l6+urqlWravz48UpOTn7EZwL3o0KFCpo4caIkKSEhQZcvX1bhwoVTtevatau6dOkig8FgNX/q1Kny8fGRJJ0/f16FCxeWwWBQcnKycubMqUKFCunPP/9UYmKiJkyYoI8++ijdWNauXavSpUtbCuKSVKZMGX311Vd67rnnHvjYfH19dfLkyQdeDwAAAED2ExQUpP3792vevHmqV6+eTp48qbffflsVKlRQpUqVNGzYMN26dcvSPr2+siTFxcWpb9++qlChgurUqaNvv/3Wal/nzp1Tx44dValSJVWqVEl9+/bV33//LSml3+Tl5aUffvhBzZo1k4+Pj1q3bm3VT58xY4aqVq2qSpUqacaMGVbbvte+AwIC9Pnnn6t27doaNmyY9u3bJy8vL8uxeXl5aevWrWrTpo18fHzUuHFjHT16NM1ztn37dlWsWFG//vprBs44AAAAAOBJku2K4pK0fPlyzZo1S+vXr9eePXvUuXNnffDBB/r+++9lMpk0f/58SdKvv/6qXr16qWfPntq/f79GjRqliRMn6rvvvtNrr72mixcvWnV+w8PDVaFChVQFyq1bt2r69OmWYbh79eql3r17648//lBiYqL69OmjQYMG6eDBg1qyZIm+/vrrVB132Na4ceMUEBCgUqVKpfl8vdvDn6flt99+U8uWLfXRRx9ZCuwuLi66ceOGTp06pWLFimnu3Llq0qSJvv76a3300UdaunRpqu0cPXpUr7zySqr5efPmfeDjSUpK0rp161SuXLkHXhcAAABA9rNkyRJVrFhRHTp00IYNG9ShQweVLVtWu3bt0pdffqn9+/frs88+k3T3vrIkzZo1S7/++qs2bdqkVatW6auvvrLa18cff6yCBQvq+++/15YtW3TmzBnNnDnTqs2i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ysvLS6dOnZIk/f7772rdurV8fX31xhtv6OzZs1br/Prrr2rfvr0qVKigypUrKyQkRImJiZbl27ZtU5MmTeTj46OAgAAtWrTo0R9cNjRu3DgFBASoVKlSatiwodWy4sWL6+LFi5bf2c6dO3X58mVdvnzZUvyWJCcnJ+XMmVOSdOnSJZnNZp05c0ZFihRRSEiIBgwYoDFjxmjgwIH68ccfU8Vw9OhReXt7p5qfN2/eBz6ea9euafv27SpXrtwDrwsAAAAA9xIdHa333ntPlSpVUsWKFfXhhx8qJiZG+/btU/ny5bVz507Vr19fPj4+6tixo/766y9t2bJFffv21eHDh+Xt7a2oqCglJydr6tSpqlOnjsqWLauWLVvqp59+suwnICBAn3/+uWrXrq1hw4bJz89P//zzj5o2bWrpc69fv14NGzaUr6+vAgICFBYWZll/4MCB6tOnjyRpzZo1atKkidauXauAgAD5+vqqT58+lj72wIED9cknn2jo0KHy9fVV7dq1dfDgQc2ePVv+/v7y9/fXmjVr7nkOJD3weQAAAAAAZG12+0zx+vXra8OGDerevbtl3jfffKMePXpo165dlnkDBw6Um5ub5s2bpwsXLqhXr16WZTdv3lSnTp3Url07zZkzR5cuXVJwcLDmzp2r9957T7/++qt69eqlzz77TDVq1NCBAwf03nvvycPDQzVq1Hisx5tVmUwmSdIHH3ygHj16aNCgQQoNDdW7775raZMrVy59+umn+vjjj5UzZ05VrlxZTk5OSk5OttqGJJnNZhkMBjVq1EhBQUGqUqWKdu/eLU9PT505c0bFixdXq1atFBwcrPLly1vF4uLiosTERKvtpSU5OVlmsznVfsePH6/Zs2crPj5eFy9eVO/eveXn53fP7WU3t4/X3o4b9oU8hz0gz2EPyHP7lpV/72azWcHBwSpXrpwmT56suLg49e3bV2PHjtXrr7+umzdvatOmTVqxYoVu3rypVq1aaeXKlercubN+++03ff/991q5cqUkaf78+dq0aZNCQ0NVpEgRrVixQt26ddOOHTvk6uoqSdq0aZPmzZsnd3d3de7cWbVr19a6detUsmRJRUVFacCAAZo7d678/f21d+9edejQQeXKldOLL76YKvbo6GgdOXJEGzduVHR0tFq0aKHw8HDLl6M3b96sTz/9VB999JG6d++uvn37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
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





  <div id="df-3316d8b0-a784-4d6b-a58d-7774a1bc1502" class="colab-df-container">
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
    <button class="colab-df-convert" onclick="convertToInteractive('df-3316d8b0-a784-4d6b-a58d-7774a1bc1502')"
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
        document.querySelector('#df-3316d8b0-a784-4d6b-a58d-7774a1bc1502 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-3316d8b0-a784-4d6b-a58d-7774a1bc1502');
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


    
![png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAB8YAAAHqCAYAAAB2uSQnAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAA2uJJREFUeJzs3Xlc1NX+x/H3MLKIyOaCKCpoihvimonmgppKoWlqafW7XW9lenPJbqlXvWppZYbZZmq7ZjevZWnllrllLplpIpKWW4q7AqKgwDC/P0YGR8YFRGYGXs/Hw0fMOWe+8/nOOUx8v5855xjMZrNZAAAAAAAAAAAAAACUUG6ODgAAAAAAAAAAAAAAgNuJxDgAAAAAAAAAAAAAoEQjMQ4AAAAAAAAAAAAAKNFIjAMAAAAAAAAAAAAASjQS4wAAAAAAAAAAAACAEo3EOAAAAAAAAAAAAACgRCMxDgAAAAAAAAAAAAAo0UiMAwAAAAAAAAAAAABKNBLjAAAAAAAAAAAAAIASjcQ4AKDYjR49WuHh4df99+ijj97SayxatEjh4eHat29fEUVtsWXLFoWHh2vLli3XbRceHq7XXnutSF/7drpd71dhHTlyROHh4Vq0aFGRHO+ZZ55RdHS09XF0dLSeeeaZIjn2lUaPHq02bdrc9teRpEcffVT9+vW7LccGAAAAgFwbNmzQU089pdatW6tRo0bq0KGDhg8frl9++cXRodl46623FB4erkuXLt3ScVztWutm7j9cfU3s6q6+9i7q9gCAkquMowMAAJQ+Y8eO1bPPPmt9PGHCBCUkJOiLL76wlrm7u9/Sa8TExOjuu+9WYGDgLR2nsDZs2CBvb++bbv/mm2/q6NGjeuWVV25jVM5r8+bN+ve//63Vq1cXy+t98cUXBRpjX375pb7++mvNmzfvuu3Gjh2rrKysWw3Pro4dO+qVV15Rq1atJFlu+gAAAADA7TRjxgzNmTNHDz/8sIYMGaLAwEAdOnRIH3/8sR599FFNnDhRDz74oKPDvCVca928UaNGKSQkREOHDr3lY2VmZqpZs2Zavny5QkJCCvTc23ntDQAo2UiMAwCKXfny5VW+fHnrY09PTxmNRlWqVKnIXsPLy0teXl5FdryCKui5bN++XUFBQbcpGue3ffv2Yn29gn5h4mbju3JcF6UTJ07o6NGjNmX+/v635bUAAAAAQJLWrVund999V//5z3/08MMPW8tDQkIUFRWl4cOH67XXXlO3bt3k5+fnwEgLj2utgtm+fXuBk9jXEh8fX+jk9u269gYAlHwspQ4AcFq5y3uvW7dOnTp10gMPPCBJys7O1htvvKFOnTqpYcOGatOmjYYNG6YjR47ke27u0uCjR49Wz549tWXLFvXu3VuRkZHq0qWLvvrqK5vXvNkl0C9evKixY8eqRYsWatKkiUaOHKnz58/bPY7ZbNasWbPUtWtXNW7cWHfddZeefvppHT58WJJlue2NGzfqq6++slmmfd++fXrqqafUokULNWrUSDExMflmLIeHh2vOnDkaNGiQIiIiNG/ePLtLvWdmZqpFixaaNm3aTb33krR+/Xo98sgjuvPOO9WsWTM98cQTNkut577He/fu1RNPPKGmTZuqbdu2eumll5STk2Nt9+eff+qRRx5R48aNdffdd+u9997T7NmzFR4eLsnSNzNmzFBSUpLCw8Ntvp1vMpn02muv6a677lJERIT+8Y9/6MSJE9eNe9u2berVq5ciIiIUHR2tTz/9NF+bq5c4X7BggWJjY9WkSRO1bNlSAwcOVEJCgiTLMnoLFy7Uzz//bF3ePXdJ/WXLlik2NlatW7e2nou95dnmz5+vjh07qlGjRurdu7d27NhhrbP3nNyl5P/73/9qy5YtateunSTp//7v/6zL3129vF9mZqbi4uIUHR2tRo0aKSoqSqNHj9aZM2dsXutmfg8AAAAA4MMPP1RoaKgGDBiQr85gMOiFF17QDz/8YE2K29tKyt61+X333ad169YpJiZGERERuv/++5WYmKhNmzapZ8+eioyM1AMPPKDff//dehx7y5vnXpetX7/ebvw3undwo2sts9msjh072qx4l+s///mPWrVqZU3s3uj6+Vq++uorxcbGqnHjxurcubPeeOMNZWdnW+tv5r7A1W7mmtiezMxMvfLKK4qOjlZERITatGmjUaNGKTk5WZLl/sOhQ4f09ttvKzw83Po+LlmyxPp6zZs3V//+/fXzzz9bj5t7fbtw4UI99NBDatSokT755BPruOrUqdM1t9O72Wvvn3/+WY888ohatmypJk2aqFevXvruu++uea45OTkaNmyY7r77bpt7SQCAko/EOADA6c2ePVsvvfSSZs2aJUmaNWuW3nvvPT333HNatWqV3n33XSUlJWnYsGHXPc7Zs2f19ttva9y4cfr6669Vu3ZtjR8/XseOHbO22bBhgwYPHnzDmN544w01adJEX375pcaOHavvvvtO77//vt22X3zxhWbPnq3nnntOy5cv15w5c3Tu3DkNGjTIWh8YGKju3btrw4YNatq0qc6cOaOHH35YKSkpmjNnjr799lv17NlTU6ZM0dy5c22Ov3DhQjVv3lzLli3Tgw8+qODg4HyJzvXr1ystLU19+/a94blJlovKQYMGqXLlyvrss8/0ySefKDMzU4888ojOnj1r03bixInq27evlixZogcffFCffPKJli1bJslyYf3kk0/qxIkTev/99/Xee+9p27Zt+vLLL63PHzt2rDp16qQqVapow4YNGjhwoLVu7ty58vPz0+eff664uDht3br1ul9cSElJ0VNPPSVPT099/vnneuedd/Tzzz9r69at13zOpk2bNHHiRP3973/Xd999p3nz5snPz08DBw5URkaG3nrrLTVs2FBNmzbVhg0bFBMTY33urFmzNHz48Osmln/99Vdt2bJF7777rv773//KbDZr8ODBSk9Pv3YHXKFp06aKi4uTZFnS78otB640btw4ffbZZxo2bJiWLl2ql19+WVu2bNETTzwhs9lsbXczvwcAAAAASrfs7Gz9+uuvat++vQwGg902/v7+8vX1LfCxk5OTNW/ePMXFxenTTz/V2bNn9fzzz2vmzJmaPHmy5s2bp1OnTmnKlCm3dA43undwo2stg8Gge++9V2vWrLHZtzw7O1srV67UvffeK3d39wJdP1/pm2++0dixY/XAAw/om2++0ejRo/Xxxx9r+vTpklSg+wK5CnNNnGvmzJn67rvvNGXKFK1cuVJvvPGGdu/ereeee06SrFufDRw4UBs2bFBwcLC2bt2q5557Tu3bt9fSpUu1cOFChYaGatCgQfm+1P7BBx+oT58+Wrlypfr27at//etfkiz3NG60fP31rr3T0tI0aNAg1atXT//73/+0ZMkSde3aVc8++6zNl9Kv9PLLL2vz5s364IMPimwGPADANZAYBwA4vZiYGLVq1cq6PPmAAQO0ZMkSdevWTcHBwWrcuLH69OmjhISE6150njx5UuPHj1ezZs0UFhamf/zjH8rKytLu3butbSpVqqRy5crdMKZWrVqpb9++qlmzpvr27avatWtr586ddtsmJCQoODhYnTt3VtWqVdW4cWPNmDFDr776qnJychQYGCg3Nzd5eXmpUqVK8vDw0BdffKHU1FS9+eabatasmfXCskOHDvm+HV6+fHk9+eSTCgkJkYeHh/r27asVK1bowoUL1jbfffedWrZsqdDQ0BuemyTNmTNH1apV07Rp03THHXcoIiJCcXFxOn/+vP73v//ZtI2JidE999yj6tWra/DgwXJ3d7e+F1u3blVSUpL+/e9/684771S9evX05ptv2iSFy5cvb7Oc/pXvf1hYmJ544gmFhobqnnvuUatWra75PkvS999/r3PnzmnKlClq2LCh6tevr9dee+26y7Pt2rVLZcuWVY8ePVStWjXVq1dPU6ZM0Zw5c2Q0GuXv768yZcrI3d1dlSpVslmiPyoqSp07d1aVKlWuefz09HRNmzZN9erVU0REhMaNG6ezZ8/qp59+unYHXMHDw8N6s8nPz8/uMvAnTpzQkiVL9NRTT+n+++9XjRo11L59e40ePVoJCQnatm2bte3N/B4AAAAAKN2Sk5OVmZmpatWqFfmxT58+rbFjx6p+/frWVaz27t2rESNGKCIiQo0bN1aXLl2UmJh4S69zo3sHN3OtFRsbqwsXLujHH3+0lm3evFnJycnq2bOnpIJdP19pzpw56tChgx577DHVrFlTnTt31vPPPy+TySRJBbovkKsw18S5EhISFB4ertatWys4OFgtWrSwfrFAkipWrChJ8vb2VqVKlWQ0GtWwYUN9++23evrpp1W9enXVqlVLjz/+uNLT0/Xrr7/aHL9u3brq06ePqlatKm9vb/n4+EiybHV2o+Xrr3ftfeDAAaWnpys2NlZhYWGqUaOGnnrqKS1YsMDuPZAPP/xQCxcu1OzZs1W3bt0bvi8AgJKFxDgAwOk1atTI5rGnp6eWLFmi2NhY3XnnnWratKleeuklSbIu8WWPt7e3zUVP7kXvuXPnChxT06ZNbR4HBgbaJKKv1LFjRx08eFCPPfaYvvrqKx07dkyBgYFq1KiR3Nzs/684Pj5eNWrUUOXKlfO97l9//WWzbPvV70/fvn2VmZmp5cuXS7IkZtesWaM+ffrc9Pnt3LlTd911l4xGo7WsYsWKqlOnTr4EamRkpPXnMmXKyNfX1/qe/vXXX5KkiIgIaxsPDw+7y43bU5D3WZL27t2rsmXLqnbt2javd/V7dKU2bdooJydHDz74oP773//qwIED8vb2VmRkpDw8PK4b3/WOe2UbT09P6+PcJeT3799/w+ferF27dslsNqtFixY25bnv35V9VpS/BwAAAABKptxZ4leuPlVUvL29FRYWZn2cuxR7/fr1bcrS0tJu6XUKe+/gSuHh4apbt65WrFhhLVu6dKlCQ0Ot18IFuX7OdfHiRe3du9fmelqS+vfvrzFjxkgq2H2BXIW5Js7VqVMn/fjjj9ZVyM6cOaMqVapYr2Ht8fb21o4dO/TII48oKipKTZs2tW6Dl5KSYtP2ZmK4lus994477lDNmjU1dOhQvfvuu/rtt9+Uk5OjyMjIfAn3pUuX6rXXXtMbb7yR734DAKB0IDEOAHB65cuXt3n8r3/9y7oE19y5c/X111/fcBl1yXLBZk9hLvSvnDUsWW4aXOs47du319y5c+Xr66spU6aoQ4cO6tevn80s3qudP38+33lLsn6j+srk8NVL11WuXFnR0dFatGiRJGnNmjVyd3dXt27dbu7kLr/+119/raZNm9r8+/3333Xy5Embtle/r1e+F7kXwlfPwrf3TXx7CvI+S5b3pWzZsvnKr7cKQIMGDbRgwQLdcccdevPNN9WtWzfde++9+uGHH24Yn70+utrV/ZP7ft3sUuo3I/eGyNXx2BsvRfl7AAAAAKBkCggIUNmyZXXo0KEiP7a9a8iry6+1fHtBFPbewdViY2O1evVqZWZmKisrS6tWrVKPHj2s9QW5fs6V+8Xk612rFuS+QK6buSb+5ZdfbOJ8/PHHJUkPPfSQZs2apYyMDI0ZM0Zt27bV3//+d/3555/XjPHjjz/WuHHjVK9ePc2aNUtff/215syZY7ftzVw/X8v1nuvt7a3PP/9csbGx+vrrr9WvXz+1bdtWc+bMsbnOPXfunMaMGSOTyaQzZ84UOhYAgGsr4+gAAAAoiPPnz2vNmjV64okn9Le//c1anpOT48CobqxFixZq0aKFsrOztW3bNr399tt64okntHbtWrt7svn6+trd8zn3G/O5F8LX8uCDD+rxxx/X0aNHtXTpUsXGxuZLMl+Pr6+v2rZtq6FDh+aru9EsanttMzIybC7Or/7meFHx9vbWxYsX85XfaKZBeHi4pk6dKrPZrPj4eL333nsaOnSodSbArbj6ZkVuQjz3xoS9ZH9Bk+a5Y+jq88x9XJh9/wAAAACUXkajUS1bttTq1as1duxYlSmT/zZyamqqVqxYod69e1vrb/Xa5noKcuyivHdw3333afr06dqwYYPc3Nx07tw5m8R4Ya6fAwIC5ObmptTU1Gu+bmHuC9zMNXGjRo309ddfWx9fea+gY8eO6tixozIzM7Vx40bFxcXpySefvOYXx5csWaImTZpo4sSJ1rLrbXF3uwQGBuq5557Tc889p8OHD+uLL77Q66+/rsDAQOvqeWazWdOnT9fGjRv14osvqkmTJqpVq1axxwoAcCxmjAMAXEpWVpbMZrPNjGOTyaQlS5Y4MKrr+/HHH63fsC5TpoxatWqlMWPG6MKFCzp8+LC13ZUX+Y0bN9bhw4d14sQJm2Nt27ZNtWvXvuE+6G3atFH16tW1YMECrVu3Tn379i1QzE2aNNG+fftUs2ZNm3/Z2dnWvd5vRs2aNSVZloDLlZGRofXr1+drWxQzlmvVqqX09HT98ccf1rKLFy9q165d13zOtm3b9Ntvv0myJKkbN26syZMny2Qyae/evbcc386dO21uTCQkJEiS6tSpI8nyzfdz584pOzvb2iY3nqtdK4bcZfm3bt1qU567KsGVS9kDAAAAwM0YOHCgjh8/rpkzZ+arM5vNeuGFF/Tyyy/r1KlTkiyJ3KuTojt27CiSWAp67ILeO7je9V7VqlXVrFkzrVq1SsuXL1ezZs1UvXp1a31hrp/d3d0VFhaW7xrus88+05NPPimpcPcFbuaa2MvLyybOoKAg5eTkaOXKldZEvIeHhzp06KBhw4YpKSnJJoF/5XuVlZWlgIAAmxi++uqrfO2u51bvBRw8eFCrV6+2Pq5evbqeeeYZ1alTR7///ru13M/PT506ddLzzz+vkJAQjRw5UpmZmbf02gAA10NiHADgUgICAhQaGqpFixZpz549SkxM1ODBg9W8eXNJ0tatW+3us3WzTp06dd09rAtj0aJF+uc//6kNGzbo6NGj2rt3rz766CNVqFDBuu+Xr6+vdu/ercTERJ0+fVq9e/eWv7+/nnnmGe3cuVMHDhzQm2++qfXr11svkq/HYDCoX79+ev/991W3bl2bvdpuxuOPP649e/Zo4sSJ+v3333Xw4EHNmTNHsbGxWrdu3U0fp3Xr1vL391dcXJx27NihPXv26Nlnn8134ezr66tTp07pl19+sfmyQEHdc8898vb21gsvvKDExEQlJibq2Wefveby4ZJlqfkhQ4Zo5cqVSkpK0v79+zVr1ix5eXlZE8q+vr46ePCg4uPj7X5j/3q8vLw0duxY7d27Vzt37tSUKVMUFBSkqKgoSZabHVlZWZo1a5YOHz6sVatWWZfBz5W7595PP/2k3bt357txUKlSJfXq1Utz5szRt99+q8OHD+uHH37Qyy+/rFatWqlx48YFihkAAAAAWrduraFDh+qdd97RqFGj9OuvvyopKUlbtmzRk08+qe+//17Tpk1TcHCwJMu1za+//qpVq1bpr7/+0scff2z9YvCtaty4sY4cOaL//e9/Onz4sBYtWnTda9ObvXdwo2utXLGxsdqwYYPWrVunnj172tQV9vr5ySef1KZNmzRr1iwlJSVp9erVmjFjhnUWc2HuCxTmmliS3Nzc9P7772vEiBH65ZdfdOzYMSUkJOjzzz9X3bp15e/vLw8PD3l5eWnHjh36/fffde7cOTVp0kRbtmzRxo0bdejQIU2bNk05OTkyGo3auXPndWeP577369at0549e64b3/X89ddfevrpp/XRRx/p4MGDSkpK0qJFi3TgwAG1bNkyX3tPT09Nnz5dBw4c0Kuvvlro1wUAuCYS4wAAlzNt2jS5u7urb9++GjZsmLp06aJx48apWbNmmjx5spYvX17oY7dt21bvvvtuEUYrvfjii2rdurXGjh2re+65R4899pjOnz+vDz/80Lpk2aBBg3T8+HH1799fW7duVWBgoObNm6fy5cvr73//u2JjY7Vq1SpNnTpV999//029bkxMjLKzs/Xggw8WOOYWLVro/fff1549e/Tggw8qNjZWK1eu1Ouvv65OnTrd9HHKlSund999V0ajUY888oj++c9/ql27drr77rttlpTr37+/goKC9Nhjj2nu3LkFjjdXxYoV9c477yglJUV9+/bVkCFDdNddd6l9+/bXfM7w4cPVp08fTZ06Vd26dVP//v2VmJio9957z3qD5+9//7vMZrMGDBhQ4PHVtm1bRURE6IknntCAAQPk6empWbNmydPTU5Klnx599FF99tlnio2N1fz58/Xiiy/aHCMiIkKdOnXSRx99pMcff9zu8n8TJ07UQw89pNdee03dunXThAkT1LlzZ7uzOwAAAADgZjz99NP6+OOPlZqaqiFDhqhbt27697//rYoVK2rRokXq3Lmzte2wYcPUpk0bjRo1Sn369NG+ffv0zDPPFEkcjz76qGJjY/Xaa6+pZ8+eWrt2rcaNG3fd59zMvYObudaSpG7duuns2bM6f/68unXrZlNX2Ovn+++/X1OmTNGSJUvUrVs3TZkyRY888oieffZZSSrUfYHCXBPneuedd1S9enUNHz5cXbp00VNPPSV/f3/rPRKDwaAhQ4Zo+/btevjhh7Vv3z6NGDFCrVq10tNPP62HHnpI2dnZmjBhgh555BF99913mjZt2jVfr3379mrWrJleeeUVjRkz5obxXUu7du300ksvafHixbr//vt13333ae7cuRo3bpy6du1q9zl16tTR6NGjNW/evGsuEw8AKJkM5qJYtxQAADidjz76SHPmzNGaNWsKtL94UcudwX/l/mdDhgzRoUOH9N133zkqLAAAAAAAAABAKVLG0QEAAICidfz4cf388896/fXXNW7cOIcmxbOzs9WjRw8FBgZq/PjxCgwM1I8//qg1a9Zo1KhRDosLAAAAAAAAAFC6MGMcAIASpmHDhgoMDNQjjzyiQYMGOTocHTx4UNOmTdO2bduUkZGhkJAQ9e3bV48++qiMRqOjwwMAAAAAAAAAlAIkxgEAAAAAAAAAAAAAJZqbowMAAAAAAAAAAAAAAOB2IjEOAAAAAAAAAAAAACjRSIwDAAAAAAAAAAAAAEq0Mo4OwBlkZ2crNTVVnp6ecnPjuwIAAAAAgNsnJydHly5dkp+fn8qUKVmX5VxfAwAAAACKU0GusUvWFXghpaam6uDBg44OAwAAAABQioSGhqpChQqODqNIcX0NAAAAAHCEm7nGJjEuydPTU5LlDStbtqyDo7mGjAypTRtJkmn9ehl9fBwcEJCfyWTS3r17VbduXRmNRkeHA+TDGIWzY4zC2TFG4excZYxmZGTo4MGD1mvRksQlrq8vc5XxUprRR66BfnJ+9JFroJ9cA/3k/Ogj10A/uQZX6aeCXGOTGJesy7uVLVtW3t7eDo7mGsxmac8eSZLJy0tGZ40TpZrJZJIkeXt7O/WHJEovxiicHWMUzo4xCmfnamO0JC417hLX15e52ngpjegj10A/OT/6yDXQT66BfnJ+9JFroJ9cg6v1081cY5e8q3AAAAAAAAAAAAAAAK5AYhwAAAAAAAAAAAAAUKKRGAcAAAAAAAAAAAAAlGgOS4xv3bpVERERNv8aNWqk8PBwSdKmTZvUp08fNWvWTPfee6+WLFli8/y5c+eqa9euatasmfr3769du3ZZ6y5duqT//Oc/ateunVq1aqVhw4YpOTm5WM8PAAAAAAAAAAAAAOAcHJYYb9mypeLj423+Pf300+revbtOnjypIUOG6KGHHtKmTZs0duxYjR8/XvHx8ZKk1atX66233tKrr76qjRs3qmPHjnrqqaeUnp4uSXr99deVkJCgBQsWaMWKFTKbzRozZoyjThUAAAAAAAAAAAAA4EBOs5T60aNH9dFHH+n555/XN998o9DQUPXp00eenp6KiopSdHS0Fi5cKElasGCBevfurcjISHl5eenxxx+XJK1Zs0bZ2dn64osvNGTIEAUHB8vf318jRozQ2rVrdeLECUeeIgAAAAAAAAAAAADAAco4OoBcb7zxhh544AFVrVpVCQkJatCggU19gwYNtGzZMklSQkKCYmJirHVubm6qX7++4uPjVb9+faWlpalhw4bW+tq1a8vLy0sJCQkKCgq6Zgwmk0kmk6mIz6yImEwyWn80Sc4aJ0q13N8fp/09QqnHGIWzY4zC2TFG4excZYw6e3wAAAAAAJRETpEYP3LkiFauXKmVK1dKklJSUvIlsP39/a37hKekpMjPz8+m3s/PT8nJyUpJSZEk+fr62tT7+vrecJ/xvXv33spp3FZuGRlqevnnhIQE5ZQt69B4gOvJ3fYAcFaMUTg7xiicHWMUzo4xCgAAAAAAruYUifH58+frnnvuUaVKlW76OWaz+Zbq7albt668vb0L/LxiceGC9ceGDRvKeFXiH3AGJpNJ8fHxioiIkNFovPETgGLGGIWzY4zC2TFG4excZYymp6c79RezAQAAAAAoiZwiMb5ixQqNGjXK+jggIMA68ztXcnKyAgMDr1mfkpKiOnXqWNukpKSoXLly1vrU1FRVqFDhunEYjUbnvXlyRVxOHScgxiicH2MUzo4xCmfHGIWzc/Yx6syxAQAAAABQUrk5OoDExEQlJSWpTZs21rKIiAjt2rXLpt2uXbsUGRkpSWrUqJESEhKsdSaTSbt371ZkZKSqV68uPz8/m/q9e/cqMzNTjRo1us1nAwAAAACwZ9GiRTbXfVd77bXX9Oijj173GJ988okeffRR5eTkFHV4Ny0zM1P33Xefvv32W4fFAAAAAAAACs7hifHdu3fL399fPj4+1rLY2FglJSVp4cKFunTpktatW6d169apX79+kqT+/fvr66+/1o4dO5SRkaF3331XHh4e6tChg4xGo/r166dZs2bp2LFjSk5O1vTp09WlSxdVrFjRUacJAAAAAC5t//79evbZZxUVFaXIyEhFR0dr8uTJ+Vbzul327Nmjt956S6+88orc3CyXsocOHVLv3r2vm3DPNX/+fHXt2lVNmzZV165dNW/ePGvd8uXL1bZtW7Vt21bff/+9zfN27typbt266dKlS5IkDw8PvfLKK5o4caKOHTtWhGcIZxYeHq5GjRopIiLC+u/FF1+UJG3atEl9+vRRs2bNdO+992rJkiXW523dulWdOnVSq1atNH/+fJtjJiUlqUOHDjp79myxngsAAAAA3FCOSTq8VgEnlkuH11oelwAOX0r99OnT+fYWr1ChgmbPnq3Jkydr0qRJqlatmqZNm6Z69epJktq1a6eRI0dqxIgROnPmjCIiIjRnzhx5eXlJkoYNG6YLFy6oZ8+eys7OVseOHTVx4sTiPjUAAAAAKBESExP18MMPq3///lqyZIkCAgK0d+9evfTSS+rfv7+++uor6/XY7fL222/r3nvvVbVq1SRZkpHPP/+8mjZtqhMnTlz3uevWrdO0adP0ySefKCIiQvHx8frb3/6m6tWrq127dnrxxRf1/vvvy2Aw6IknnlDnzp1lMBiUnZ2t//znP5owYYI8PT2tx2vUqJFatGihDz74QOPGjbut5w3nsXz5coWEhNiUnTx5UkOGDNHYsWMVGxurbdu2afDgwQoLC1NERIReeeUVjRkzRo0bN1ZsbKxiY2Pl6+srSXrxxRc1dOhQ65ZwAAAAAOAU/lgkrR4u4/kjqiVJiZJ8QqToN6Q6vR0c3K1xeGJ80KBBGjRoUL7yli1bavHixdd83oABAzRgwAC7dR4eHpowYYImTJhQZHECAAAAQGn1wgsvqG3btnruueesZfXr19e7776rKVOm6OTJk6pRo4aOHz+uSZMm6ddff1V2drbatWunCRMmyN/fP98xV69eralTp+rkyZNq3779dVf4OnXqlFatWqUvvvjCWpaSkqKPP/5Yv/32m7Zt23bd+Hft2qU6depYt+eKjIxU3bp1tXv3bjVo0MB6PpKUnZ1t/QL33LlzVa9ePbVu3TrfMR966CE988wzev755+Xh4XHd10fJ9c033yg0NFR9+vSRJEVFRSk6OloLFy5URESE9uzZo7vvvluenp6qXr269u/fryZNmmjFihW6cOGCHnjgAQefAQAAAABc4Y9F0pI+ksy25eeTLOU9vnDp5LjDl1IHAAAAgFLNbJYuXCjef2bzjeO67MyZM/r111/1yCOP5Kvz8fHRyy+/rBo1akiShgwZovLly+uHH37QihUrdPLkSbtfWD537pyeeeYZPfLII9qyZYt69eqlr7/++poxbNmyReXLl7cmryWpe/fuql279k2dw913360///xTW7ZsUWZmprZv3659+/apbdu2MhgMNnuWm81mGQwGHT16VJ9++qm6du2qAQMG6MEHH9TatWut7Vq0aKFLly5p586dNxUDXF9cXJw6dOigFi1aaPz48bpw4YISEhKsX67I1aBBA+3atUuSbMZX7tg6f/68pk2bpoEDB+qxxx5T3759bb70AQAAAAAOkWOSVg9XvqS4lFe2ZoRLL6vu8BnjAAAAAFBqmc1S27bSxo3F+7pt2kg//igZDDdsevjwYUlSWFjYddslJiYqISFBs2fPlo+Pj3x8fPTkk0/qn//8pzIzM23abtiwQd7e3nr44Yfl5uam9u3bq0WLFrpw4YLdY//xxx+64447rHuLF1Tjxo01ZswYDRw4UNnZ2SpTpoxGjx6txo0by2w2y93dXb/99ptMJpO8vb1VsWJFPfXUUxo2bJji4uI0ceJEVatWTX379tWaNWvk7u4uHx8fValSRX/88YdatGhRqLiQJy0tTefPn7cp8/LyUkBAgLKzs3Xq1Kl8zwkODpZk2aItKyvLps7f319ly5bVhQsXdO7cOZs6Dw8PVahQoUDxNWnSRFFRUZo6daoOHz6sESNGaNKkSUpJSVFQUFC+105OTpYkNWzYUGvWrFFERISSkpJUu3ZtzZgxQ/fff78+//xz9erVS506dVJMTIw6duxY4LgAAAAAoEiYc6T496XzR67XSEo7LCX9KFXvUFyRFSkS4wAAAADgSDeRnHYkw+X4rpxVbc+RI0fk5+enSpUqWctq1KihrKysfHuAHz9+XMHBwTaJ7tDQUCUkJNg9dkpKivz8/Ap7Ctq8ebPi4uL0/vvvq1mzZoqPj9fw4cMVHByszp07a8KECRo6dKgMBoNeeOEFrVy5UhcvXlSnTp300ksvWRPflSpV0v79+xUeHi5JCggI0NmzZwsdF/Js27ZN69atsymLiIhQ7969de7cOc2ZMyffc3JXI1i8eLGOHLG9edOrVy81btxYCQkJWrZsmU1d7dq17a6AcD0LFiywef6//vUvDR48WM2bN7/u80aPHq3nn39eaWlpeu6553TgwAFt2bJFX375pdq2bau4uDj5+PiocePG+u233xQdHV2guAAAAACg0DLOSodWSgeWSQeXS+knb+5554/d3rhuIxLjKBRTTo6MhZytUZTHAAAAAFyawWCZuZ2eXryv6+190wn53GXS//jjj3wzY6909azwKxmueq3MzEyZTLZLr90o8X71MQriv//9r+655x7rXuEtWrTQvffeqy+++EKdO3dWp06d1KlTJ0nS+fPn1bt3b7333ns6f/68vL29rccpW7as0tLSbGIyF2BZelxb8+bNrV84yOXl5SVJ8vX11ZNPPnnN5/bs2dPujHHJMmO7evXqNnVFsSd8SEiITCaT3NzclJKSYlOXnJyswMBASZaZ5itXrpQkmUwm9e3bVxMmTJCHh4fS0tKs4+vqsQUAAAAARc6cI53cYUmEH1gqHdtsKSson+AiD624kBhHoRjd3PTwokVKtLOc3c2oX6mS5vfuXcRRAQAAAC7IYJDKlXN0FNcUEBCgO++8Ux999JHatm1rU5eRkaGHH35YY8eOVfXq1ZWamqrTp0+rYsWKkqT9+/fL09MzX0K9cuXKOnHihHXPZUnat2/fNWPw9/fXH3/8UehzyMnJyZeIv1Yif8aMGerdu7dq1qyptLQ0m2W4U1JS5OPjY3189uxZawIUt6Z8+fIqX7683boyZcpYl023J3e82VOuXDmVu8Xfr927d2vJkiUaPXq0tWzfvn3y8PBQ+/bt9dVXX9m037VrlyIjI/MdZ968eWrYsKF1BQIfHx+lpqYqICBAKSkptxwnAAAAAORzMUU69P3lWeHLpAvH7bdzLydVj7Ysk34pVfb3GTdI5UOkanffxoBvL6brotAST53S9uPHC/WvsAl1AAAAAMVv7Nix2rFjh0aOHKnjx48rJydHiYmJevzxx+Xl5aXGjRsrIiJCtWvXVlxcnNLT03XixAm9++67uvfee+Xu7m5zvKioKJ0/f16ff/65MjMztWrVKv3222/XfP06depo3759BZqd/be//U1Lly6VJEVHR2vFihX65ZdflJ2drZ07d2rZsmXq0qWLzXN27dqln3/+Wf/4xz8kWZK1QUFB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DIVxAAAAAABcyKuvvqrff/9dU6ZMUe3atfXSSy/pyJEjunTpkqpUqaJ///vfqlevntExARSEs/ul1V2kS8czHnv4SF1XSPc8ZGwuAAAAwAm5G/nkMTExev7559WsWTO1aNFCL730ki5evChJioqK0uOPP65GjRqpY8eOWrp0qcO269evV7du3dSwYUP17t1b27dvty+zWq2aM2eO2rdvryZNmujpp5/WiRMnCvTYAAAAAADIL/Xq1dOqVatUo0YNTZ48Wb6+vho+fLgGDRp020Xx3JyjAzDAXxulz1tkFsVLVpYG7KAoDgAAAGTD0ML4s88+q1KlSmnr1q1avXq1Dh06pBkzZujy5csaNmyYmjdvrh9//FFz5szRokWLtGnTJkkZJ+Tjx4/XuHHjtHv3bg0ePFgjRozQmTNnJEnLly/XunXrtHjxYn3//feqVq2ann/+edlsNiMPFwAAAACAPOPu7q6nnnpKn376qTZs2KCnn35aJ0+evO393e45OgAD/LYg40rx1EsZj8s3kh79SSrHbBEAAABAdgwrjF+8eFHBwcEaO3asSpQooQoVKqhXr1763//+p/DwcKWlpem5556Tt7e36tatq759+2rFihWSpJUrVyo0NFShoaHy9PRU9+7dVatWLa1du1aStGLFCg0ePFjVq1eXj4+PwsLCFB0drX379hl1uAAAAAAA5Inz589rzpw5euaZZzR06FB9+eWXeuutt/TII4/o6aef1pIlS2SxWHK0z9ycowMoQFaL9H2YtGW4ZPvndV6jl/TINsmnorHZAAAAACdnWGG8VKlSmj59usqWLWtvO336tO68805FRkYqMDBQJpPJviwoKEj79++XJEVGRiooKMhhf0FBQYqIiNDly5d1+PBhh+U+Pj6qWrWqIiIi8vmoAAAAAADIXyNGjFBKSooef/xxPfbYYzp79qxGjRqljh076ssvv9Tff/+tPn365GifuTlHB1BAUhOkNb2kX97JbGvyotR9leRRwrBYAAAAgKsoZnSAKyIiIvTZZ59pwYIF+vbbb1WqVCmH5X5+fjKbzbJarTKbzfL19XVY7uvrq8OHD+vChQuy2WxZLo+Pj79hBovFkuNv1RcYi0Um+48WKRc5r/4wA4WL0eP3yvMbnQPIDmMUzo4xCmfHGIWzc5Uxmtt8f/zxhz766CMVL15cktS0aVO1bt1a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sWNH29KlS42OhSKua9eutpdeesmWkJBgO336tK137962V155xZacnGy7//77bfPmzbMlJiba9u/fb2vatKlt48aNRkdGEXXgwAFb06ZNbbVq1bLZbDZbbGysrUGDBraVK1faLl++bNuxY4etXr16tt9//93gpChqPvvsM9uDDz5oi46Otl26dMn2+uuv215//XXGKJxCWlqarWXLlrbZs2fbUlJSbOfPn7c9+eSTtpEjRzJGYYjFixfbOnbsaOvfv7/thRdecFh24MABW3BwsC08PNx2+fJl25o1a2z169e3nT592maz2WyffPKJrW3btrbDhw/bLl26ZHvttdds3bp1s1mtViMOBQXoVs+lt2zZYmvcuLHtt99+syUnJ9sWLVpka9mypS0xMdGA1EVPdueW1/ryyy9tjz/+uAEJYbPZbLVq1bKdOHHipuvxenIuCxYssI0ePfq69nfffdc2fvz4gg9UBOXmPcy1RowYYRs6dKjt3LlztjNnztgeeeQR2+uvv14Qh1Go3aiP9u3bZwsJCbF99913trS0NFt4eLitbt26tj179mS5r8cff9z25ZdfFkTsIudG/ZTT9wi8lvLPjfrpWmlpabYuXbrYwsPDs1zetm1b2+7du/MjZpGXV7UdVz3XLrRXjBcWEREROnjwoMaNG6eSJUuqWrVqGjx4sFasWGF0NBRhFy9eVHBwsMaOHasSJUqoQoUK6tWrl/73v/8pPDxcaWlpeu655+Tt7a26deuqb9++jFkYwmq1avLkyRo8eLC9bd26dapWrZr69OkjT09PtWjRQu3atdPKlSuNC4oiaenSpQoLC9M999wjHx8fTZw4URMnTmSMwinExcUpLi5OPXr0UPHixVW6dGk98MADioqKYozCEJ6enlq1apWqVq163bKVK1cqNDRUoaGh8vT0VPfu3VWrVi37TAYrVqzQ4MGDVb16dfn4+CgsLEzR0dHat29fQR8GClBOzqVXrFih3r17q379+vLy8tKQIUMkSd9//31Bxy5ybnRuCdfE68l5nDp1SsuWLdOLL75odJQiLTfvYa529uxZbd68WWFhYfL391f58uU1fPhwffnll0pLSyuIQym0btRHZrNZw4YNU4cOHVSsWDGFhoaqVq1a/D9lgBv1U07wWspfOemnjz/+WHfddZdCQ0MLIBmuyMvajquea1MYd3KRkZEKCAiQr6+vva1u3bo6evSoEhISDEyGoqxUqVKaPn26ypYta287ffq07rzzTkVGRiowMFAmk8m+LCgoiKnTYIj//Oc/8vT0VLdu3extkZGRCgoKcliPMYqCFhsbq5MnT+rChQt66KGH1KxZM40aNUrnz59njMIplC9fXnXq1NGKFSuUmJioc+fOadOmTWrTpg1jFIYYNGiQSpYsmeWy7MZkRESELl++rMOHDzss9/HxUdWqVRUREZGvmWGsnJxLXzuG3N3dVadOHcZIAbjRuWVWTp8+rSeffFJNmjRR+/bttWbNmoKKCkmzZ89WmzZt1LhxY02aNEmJiYnXrcPryXnMnTtXDz/8sO66664sl//xxx/q37+//dY4N5rCG7fvdt/DXCsqKkomk8lhavy6desqKSlJR44cydvQRcyN+qh169Z6/vnn7Y/T09MVFxen8uXLZ7u/9evX66GHHlLDhg01ePBgHT9+PM8zF0U36ifp1t8j8FrKXzfrpysuXryohQsX6l//+tcN1/vkk0/UoUMHNWzYUKNGjdK5c+fyKmqRlVe1HVc+16Yw7uTMZrNKlSrl0HblxD4+Pt6ISMB1IiIi9Nlnn+m5557Lcsz6+fnJbDZnez8/ID+cPXtW8+bN0+TJkx3asxuj/JuKgnTmzBlJ0oYNG7Rs2TKtWbNGZ86c0cSJExmjcAru7u6aN2+etmzZonvvvVctWrRQenq6xo4dyxiF0zGbzQ7FTynjnCk+Pl4XLlyQzWbLdjkKr5ycS99oDKFgXX1ueS1/f39Vq1ZN//rXv7Rjxw6NGTNGr7zyinbt2mVA0qKnQYMGatGihTZt2qQVK1bot99+09SpU69bj9eTczh58qQ2bdqkJ598MsvlFSpUUOXKlTVjxgzt2LFDffv21bPPPktRqIDl5PViNpvl4+MjNzc3h3UlPiMuSLNmzZK3t7ceeuihLJdXr15dNWvW1P/93/9py5Yt8vf315AhQ5SamlrASYuWnLxH4LXkHD777DM1adJENWvWzHadOnXqqF69elqzZo3Wr18vs9ms0aNHF2DKouF2azuufK5NYdwF2Gw2oyMA2dq7d6+efvppjR07Vi1atMh2vavfbAAFYfr06erdu7dq1KhhdBTgOlf+bx8yZIjKly+vChUqaOTIkdq6davByYAMqampevbZZ/Xggw/qf//7n3744QeVLFlS48aNMzoakKWbnTNxTlU05aTfGSPGu9m5ZZs2bfTBBx8oKChIxYsXV5cuXfTAAw9o9erVBqQtelasWKG+ffuqePHiql69usaNG6dvvvkmy2IPryfjLV++XB07dlS5cuWyXN63b1+9++67qlq1qu644w4NHjxYderUyXIKb+Qv/q9yDTabTTNnztQ333yjBQsWyNPTM8v1pkyZovHjx8vPz0/+/v567bXXFBMTo7179xZw4qIlp+8ReC0Zy2KxaPny5Ro0aNAN15s/f76GDRumEiVKqGLFipo8ebL27NnDLAx5KC9qO674eqIw7uT8/f1lNpsd2sxms9zc3OTv729MKOAfW7du1dChQ/XKK6/Y/yPz9/fP8goMPz8/ubvzTw4Kxq5du/Trr786THd1RenSpa/7dzU+Pp5/U1GgrkxXdPW3MAMCAmSz2ZSWlsYYheF27dqlkydPasyYMSpZsqTKly+vUaNG6bvvvpO7uztjFE4lq//bzWaz/P397e9Bs1pepkyZgguJApeTc+kbjSEUjKzOLW9FQECA/v7773xMhuxUqlRJFovluilNeT05h40bN6pdu3Y52obXU8HLyevF399fCQkJslgsDutK4j1NPrNarXrppZe0detWff7557rnnntueVsfHx/5+voqNjY2HxMiK9n9m8ZryXh79uxRamqqGjdunKPtAgICJIn/q/JIbms7rnyuTZXKyQUHB+v06dM6f/68vS0iIkI1atRQiRIlDEyGou6XX37R+PHjNXfuXPXs2dPeHhwcrD/++EPp6en2toiICNWvX9+AlCiq1q5dq3Pnzqlt27Zq1qyZevfuLUlq1qyZatWqdd19Ufbv388YRYGqUKGCfHx8FBUVZW+LiYmRh4eHQkNDGaMwnMVikdVqdfjm75Urwlq0aMEYhVMJDg6+bkxeef/p6empmjVrKjIy0r7s4sWLOn78uOrVq1fQUVGAcnIuHRwc7DBGLBaLDhw4wL9rBSS7c8trff7551q/fr1DW3R0tCpXrpzPCXHgwAG99dZbDm3R0dEqXrz4dfeD5/VkvKioKMXExKhly5bZrvP+++9fN8Uwr6eCd6P3MNeqU6eObDabDh486LBuqVKldPfdd+d71qLszTff1KFDh/T555/f8DWSkJCgKVOmOBTBz58/r/Pnz/Paymc5eY/Aa8l4W7ZsUfPmzVWsWLFs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+PDhevnllyVJhw4d0qBBg9S4cWM1a9ZMkydPVkpKin0/Xbt21VtvvaUGDRroq6++UosWLWSxWOzPferUKdWuXfum07dZrVa9++676tChg+rXr6+HH35Ye/futS9v166dVq5cqaFDh6phw4bq0KGDtm/fbl8eHh6uNm3aqGHDhnr55Zc1d+5cDRw4UL///rv69++vs2fPKiQkRLt377Zv8/nnn6tVq1Zq0KCBZsyYkW221atXq0qVKqpfv769LTIyUo888ogaNGigTp06af369Tf9vZ8/f16jRo1So0aN1KpVK7399tv2AuzVLl68qE6dOtmnFs+uv27FmjVr1KlTJzVs2FD9+/dXVFSUfdlHH32kDh06qGHDhurcubM2bdpkXzZw4EC9/fbbeuGFF9SgQQOFhobqu+++sy8PDAzU6tWr1adPH9WrV089e/bUkSNH7MsPHjyoJ554Qo0bN1bz5s01bdo0paWl2ZfPnz9frVq1UrNmzTR//vwbHsOV39t9992nxo0b65lnntHp06clZT1mr9WgQQMtXrxYXl5e1y1bsWKFBg8erOrVq8vHx0dhYWGKjo7Wvn37dPbsWW3evFlhYWHy9/dX+fLlNXz4cH355Zf2Y3nkkUcUHh6u2NjYGx4DAAAAABQkPge4/c8Bnn32WYWHh2vatGl64okn7O1xcXF64oknVK9ePT300EP6888/JWXcUq1hw4b66KOPdO+99+rXX3+VJH322Wfq3Lmz6tevry5dumjz5s0O2bt166aGDRuqVatWmjlzpqxWq315YmKiRo0aZe+vn376yb7sRv1zrS+++ELt2rVTo0aNNHXqVIfnAADgdlEYBwC4jDNnzui1117Tu+++q99++03z5s3TokWLdODAAa1du1aS9P7772v69OlKTU3VU089pfr162v79u1auXKl9uzZo7lz59r39/fff8vT01N79uxRp06dlJycrB07dtiXb9q0ScHBwTedvu3jjz/Wf//7X33wwQfas2ePevbsqeeee05JSUn2dT788EONGDFCP/30k5o2bao333zTnmHkyJEaPHiwfvrpJzVq1EjLly+XJNWrV0+vv/66ypYtq4iICDVv3lySdOzYMV24cEFbt27V3LlztXTpUkVGRmaZbffu3fbtJCk5OVnDhg1Tx44d9fPPP+vVV1/V+PHj7R84ZGfixImSpG3b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Hi8vL40dO1ZPPfVUluuOGDFCffv21ebNm7V582Z98MEH+vjjj1WvXr0bPtet9M/V697quTcAADnBVOoAAJdhtVp18eJFVa1aVU8//bS++OIL1ahRQ999991161auXFknTpxwOPE6cuSIKlWqlO0Jp7u7u7p166YNGzbo22+/vaXp064817Xf7M7u6utrlSlTRmfOnHEoTkZERNzStrfCz8/PYersypUrKyYmxuH38vXXXysqKirbfQQEBMjd3d1hqrXdu3c73ENswoQJmj17tnbv3q2vv/5aUs7661qVK1d2eL7U1FR9+OGHio+PV0REhNq3b6+goCC5u7tneV+1EydO2H+2Wq06c+aMKlSoYG87fvy4/edTp05JkipUqKAqVaroxIkTSkxMtC+Pj49XQkKCJOnOO+/UmTNnHHJd/VxXO3v2rGJiYjRw4ED5+/tL0i3dX/1WXCngX33sFy9e1PHjx1WvXj3VqVNHNptNBw8etC+PiIhQqVKlHK54P3/+vD0bAAAAADgbPgcoeFWqVLnu2E6dOmXPazabVb58eT322GNatmyZHnzwQfvtx24kJ/1z7bm3JEVHR9/uIQEAYEdhHADgMtavX6++ffvqyJEjkqSYmBjFxsbap9z29PTUsWPHlJCQoNatW6tYsWKaP3++UlNTdeTIEX3yySfq2bPnDZ+jZ8+e+vHHH/X777/rgQceuKVc/fv31/Lly/Xbb7/JYrFo/fr16tq1q73geiNNmjTR+fPn9Z///Eepqan68ssvdezYMftyLy8vXbp0SbGxsbp8+fIt5blazZo1He4h3bp1a3l7e2vhwoVKSUnRzz//rMmTJztM4XYtPz8/tW/fXvPnz5fZbNapU6c0adIkxcbG2tdxd3dX+fLlNWHCBL3xxhs6c+bMTfvrxRdf1LJly7J8zt69e+unn37S999/r7S0NH300Uf65JNP5OPjo4CAAB08eFDJyck6fPiwPvjgA5UsWdIhz6+//qqdO3cqNTVVn332mRITE9WyZUv78jVr1ujYsWNKTEzUkiVLFBwcrHLlyqlVq1by9/fXjBkzlJCQoLi4OI0ePVqzZs2y//62b99unxr9vffey/Zb6/7+/vL29tZvv/2mlJQUrVu3TlFRUUpISLAX3q8eszk1YMAAffLJJ4qOjlZCQoJmzZqlOnXqKCQkRP7+/urUqZPeeecdnT9/XmfOnNH8+fPVp08fhysBoqOjHe5BDwAAAADOhM8Bcv45gKenp44fP65Lly7leFtJeuSRR7R+/XqFh4crPT1du3fvVteuXbVv3z79+uuv6ty5s37//XfZbDadO3dOR48edbgVWnZy0j+tW7fWgQMHFB4ertTUVC1fvtzhnB8AgNvFVOoAAJfRpUsXHTp0SE888YQuXryosmXLqm/fvurQoYOkjBPTf//739q5c6cWLFigxYsX66233tJ9990nPz8/9ezZU88+++wNn6N69eqqXr26qlatqhIlStxSrj59+uj06dMaMWKEEhISdM899+i9995zuEdXdipXrqw33nhDM2fO1Ntvv62ePXuqR48e9iuBmzdvrkqVKqlDhw6aMWPGLeW5WvPmzR2mBS9evLiWLVuml156SR988IEqVqyoN998U7Vq1dJPP/2U7X6mT5+uCRMmqG3btvLx8VHPnj31yCOPXLdez549tXnzZr3yyiv68MMPb9hfp0+fdriK+2p16tTRrFmz9Prrr+v8+fOqXbu2FixYIA8PDw0bNkxhYWFq3ry5atasqenTp6t8+fKaNm2a/ern7t27a8WKFRo+fLhKlSqluXPnOtzbu0+fPho7dqz+/PNP3X333fZ7yHl4eOj999/XtGnT1LJlS/n4+Kh9+/YaP368JKlz5876448/9Oyzz8pisWjgwIFq0KBBlsdQrFgxTZkyRTNnztTcuXPVpUsXzZs3T48//rg6duyoHTt2XDdmr7Znzx771HWpqamaNm2a3nzzTTVp0kRLly5V//79FRcXp4EDB9qn/Hvvvffs27/22muaPHmy2rdvLw8PD3Xt2lVhYWH25YcPH9b58+fVvHnzbPsdAAAAAIzE5wA5/xygX79+euedd7Rz585bupL7Wi1bttT48eP12muv6ezZs6pUqZKmTJliP/d97rnn9MILL+js2bPy8/NT586d9dhjj910vyVKlLjl/qlfv74mTpyoKVOm6OLFi+rWrZsefPBBbgUGAMg1Nxv/mwAAYGexWNSxY0dNmzZN9913X4E8Z2pqqjw8POz31Ro/frysVqtmzpyZ632fO3dO7dq106effmq/35ez2LVrl/bv369nnnkmT/c7cOBA1a9fX+PGjctyeWBgoJYsWaLWrVvn6fO6mjfeeEMnTpzQwoULjY4CAAAAAIYpbJ8DAACA7DGVOgAA/0hPT9fcuXPl7+9fYFfRJiUl6b777tP//d//yWq1KjIyUlu2bFFoaGie7L9MmTIaNGiQ3n///TzZX17asmWLmjRpYnSMIik2NlZff/21RowYYXQUAAAAADBMYfwcAAAAZI/COAAAkk6dOqWGDRtq9+7dmj17tv1b2/nN29tbc+fO1cqVK3Xvvfdq5MiReuqpp9SlS5c8e45Ro0bp3LlzWrVqVZ7tMy9MnDgx22nIkX9sNptefvllDRkyRMHBwUbHAQAAAABDFObPAQAAQNaYSh0AAAAAAAAAAAAAUKhxxTgAAAAAAAAAAAAAoFCjMA4AAAAAAAAAAAAAKNQojAMAAAAAAAAAAAAACjUK4wAAAAAAAAAAAACAQo3COAAAAAAAAAAAAACgUKMwDgAAAAAAAAAAAAAo1CiMAwAAAAAAAAAAAAAKNQrjAAAAAAAAAAAAAIBCjcI4AAAAAAAAAAAAAKBQ+3+5sZd8XUj+uAAAAABJRU5ErkJggg==
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





  <div id="df-71f2cb1e-9140-44b5-8b0e-58096e9da59b" class="colab-df-container">
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
    <button class="colab-df-convert" onclick="convertToInteractive('df-71f2cb1e-9140-44b5-8b0e-58096e9da59b')"
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
        document.querySelector('#df-71f2cb1e-9140-44b5-8b0e-58096e9da59b button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-71f2cb1e-9140-44b5-8b0e-58096e9da59b');
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





  <div id="df-06e7b191-c73b-4b29-b255-4481dd664f0b" class="colab-df-container">
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
    <button class="colab-df-convert" onclick="convertToInteractive('df-06e7b191-c73b-4b29-b255-4481dd664f0b')"
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
        document.querySelector('#df-06e7b191-c73b-4b29-b255-4481dd664f0b button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-06e7b191-c73b-4b29-b255-4481dd664f0b');
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





  <div id="df-d8b7ea0c-2a67-4a97-a45d-283c2389a5c3" class="colab-df-container">
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
    <button class="colab-df-convert" onclick="convertToInteractive('df-d8b7ea0c-2a67-4a97-a45d-283c2389a5c3')"
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
        document.querySelector('#df-d8b7ea0c-2a67-4a97-a45d-283c2389a5c3 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-d8b7ea0c-2a67-4a97-a45d-283c2389a5c3');
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
    CPU times: user 24min 28s, sys: 1min 22s, total: 25min 50s
    Wall time: 3min 23s



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






  <div id="df-8bc9aec9-be60-4e8e-8b94-d467b737ac93" class="colab-df-container">
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
    <button class="colab-df-convert" onclick="convertToInteractive('df-8bc9aec9-be60-4e8e-8b94-d467b737ac93')"
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
        document.querySelector('#df-8bc9aec9-be60-4e8e-8b94-d467b737ac93 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-8bc9aec9-be60-4e8e-8b94-d467b737ac93');
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
    CPU times: user 205 ms, sys: 1.98 ms, total: 207 ms
    Wall time: 207 ms



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

           0/7713  1s
        1000/7713  1s
        2000/7713  2s
        3000/7713  2s
        4000/7713  2s
        5000/7713  3s
        6000/7713  3s
        7000/7713  4s
    
    Item-sim lookup: 7,713 articles in 4s



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






  <div id="df-2e5ed32c-9bd4-48b9-a10d-9d7c2df96463" class="colab-df-container">
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
    <button class="colab-df-convert" onclick="convertToInteractive('df-2e5ed32c-9bd4-48b9-a10d-9d7c2df96463')"
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
        document.querySelector('#df-2e5ed32c-9bd4-48b9-a10d-9d7c2df96463 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-2e5ed32c-9bd4-48b9-a10d-9d7c2df96463');
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
# user_click_sets is needed by s3_itemcf at retrieval and evaluation time.
# The user-level SET_A/SET_B split is removed: the base LGB now trains on ALL
# training impressions (impression-level OOF via tr_mask/val_mask in Cell 53),
# giving the meta-ranker OOF signal from every user, not just 30%.
user_click_sets = train_clicks.groupby('userId')['newsId'].apply(set).to_dict()
print(f'Training users with click history: {len(user_click_sets):,}')
```

    Training users with click history: 50,000



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

# Build training pairs from actual MIND impression rows (ALL warm users).
# Training on all users (not just SET_A) gives the base model more signal
# and lets the meta-ranker train on OOF impressions from every user.

print('Parsing training impressions for all warm users...', end = ' ', flush = True)

# Init
imp_rows = []

# Iterate
for _, r in raw_train.iterrows():

    uid = r['userId']

    if uid not in train_users:

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
imp_train_df = imp_train_df.merge(_subcat_lkp, on = ['userId', '_subcat'], how = 'left')
imp_train_df['subcat_clicks'] = imp_train_df['subcat_clicks'].fillna(0).astype('float32')
imp_train_df.drop(columns=['_subcat'], inplace=True)
del _subcat_lkp

# Within-impression context features
imp_train_df['imp_size'] = (imp_train_df.groupby('impressionId')['newsId'].transform('count').astype('float32'))
imp_train_df['ctr_norm_rank'] = (imp_train_df.groupby('impressionId')['m_bayesian_ctr'].transform(lambda x: (x.rank(ascending=False, method='average') - 1).div(max(1, len(x) - 1))).astype('float32'))
imp_train_df[FEATURE_COLS] = imp_train_df[FEATURE_COLS].fillna(0).astype('float32')
print(f'imp_train_df shape: {imp_train_df.shape}')
```

    Parsing training impressions for all warm users... done  (5,843,444 rows | 156,965 impressions | pos=236,344 neg=5,607,100)
    Computing TF-IDF affinities... done.
    Computing recent TF-IDF affinities... done.
    imp_train_df shape: (5843444, 19)
    CPU times: user 2min 49s, sys: 4.58 s, total: 2min 54s
    Wall time: 2min 54s



```python
imp_train_df.head()
```





  <div id="df-6b938f47-35d9-4a4d-9bab-5b92711e8a13" class="colab-df-container">
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
      <td>1</td>
      <td>U13740</td>
      <td>N55689</td>
      <td>1</td>
      <td>3.0000</td>
      <td>0.6827</td>
      <td>8.3703</td>
      <td>9.8155</td>
      <td>0.2351</td>
      <td>224.0000</td>
      <td>1.6506</td>
      <td>sports</td>
      <td>0.5774</td>
      <td>0.5571</td>
      <td>0.5821</td>
      <td>0.5821</td>
      <td>1.0000</td>
      <td>2.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>U13740</td>
      <td>N35729</td>
      <td>0</td>
      <td>3.0000</td>
      <td>0.6827</td>
      <td>8.1158</td>
      <td>9.6434</td>
      <td>0.2164</td>
      <td>169.0000</td>
      <td>1.7506</td>
      <td>news</td>
      <td>0.5774</td>
      <td>0.6972</td>
      <td>0.0096</td>
      <td>0.0096</td>
      <td>0.0000</td>
      <td>2.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>U91836</td>
      <td>N20678</td>
      <td>0</td>
      <td>14.0000</td>
      <td>3.0372</td>
      <td>6.3613</td>
      <td>8.8622</td>
      <td>0.0816</td>
      <td>169.0000</td>
      <td>1.2982</td>
      <td>sports</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0162</td>
      <td>0.0162</td>
      <td>0.0000</td>
      <td>11.0000</td>
      <td>0.4000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>U91836</td>
      <td>N39317</td>
      <td>0</td>
      <td>14.0000</td>
      <td>3.0372</td>
      <td>6.2500</td>
      <td>8.7554</td>
      <td>0.0812</td>
      <td>199.0000</td>
      <td>1.3305</td>
      <td>news</td>
      <td>0.8321</td>
      <td>0.8076</td>
      <td>0.0327</td>
      <td>0.0327</td>
      <td>0.0000</td>
      <td>11.0000</td>
      <td>0.5000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>U91836</td>
      <td>N58114</td>
      <td>0</td>
      <td>14.0000</td>
      <td>3.0372</td>
      <td>5.1705</td>
      <td>9.0280</td>
      <td>0.0211</td>
      <td>72.0000</td>
      <td>1.3626</td>
      <td>autos</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0250</td>
      <td>0.0250</td>
      <td>0.0000</td>
      <td>11.0000</td>
      <td>0.9000</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-6b938f47-35d9-4a4d-9bab-5b92711e8a13')"
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
        document.querySelector('#df-6b938f47-35d9-4a4d-9bab-5b92711e8a13 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-6b938f47-35d9-4a4d-9bab-5b92711e8a13');
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
    0    3925239
    1     165641
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
val_imp_ids = set(rng_ltr.choice(all_imp_ids, size=int(len(all_imp_ids) * 0.15), replace = False))

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

    [100]	valid_0's ndcg@5: 0.965266	valid_0's ndcg@10: 0.967923
    [200]	valid_0's ndcg@5: 0.965549	valid_0's ndcg@10: 0.968252
    
    Base LGB trees: 169
    Features used : ['u_click_count', 'u_click_freq', 'm_log_clicks', 'm_log_impr', 'm_article_len', 'cat_affinity', 'taste_affinity', 'tfidf_sim', 'recent_tfidf_sim', 'article_age_days', 'ctr_norm_rank', 'imp_size', 'subcat_clicks']
    CPU times: user 8min 57s, sys: 1.73 s, total: 8min 59s
    Wall time: 1min 58s



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

![image.png](data:image/png;base64,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hd2Z/z2Ds8wuICALzO5+z+ezzJ17zz33nO+ZGPjN2d8JQSFAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIENhAILXBGScIECBAgAABAgQIbAeBW4fP3S8Tmu+RTlftGULRbtvhEZokQIBAgxUoLgnpTHHxnKry6rlt27Wa873urec22MHoOAECBAgQyBMQgM7DcEiAAAECBAgQIFB/AncOW9yuPJW6dFXFqi+UNGt+fro6XbR6TcXalatWpFavLitJAtHN6+9pWiJAgEDDFkgVFVW1b9euqm2bNpnWLVs2X7u2cvna6qrB7XZp2/Om83Z9pmGPTu8JECBAoCkLCEA35dk3dgIECBAgQIDAdhAYNSpTMn7NgptDqugn8+bNL58waVKbKdOmh8VLlm6Hp2mSAAECjVNgn732DAd02z8cevABlS1KS5cWp5rd9Ksr93igcY7WqAgQIECgMQsIQDfm2TU2AgQIECBAgMAOFrhl5JJvVVeW/++sWXObjX7++Tbz5i/YwT3wOAIECDQ+gcMOOSiccerJlc1bNJu5S9s21/34ws5jG98ojYgAAQIEGquAAHRjnVnjIkCAAAECBAjsYIEf9ZnWs6RZ8RWDh41sO33GzB38dI8jQIBA4xc4/pijw4XnnR3mz593853XHnVL4x+xERIgQIBAYxAQgG4Ms2gMBAgQIECAAIGdKNC3b6b566kZY1etXH1Q34FPdlyzZs1O7I1HEyBAoHEL7LfvPuHKyy+pWLFsRb87rz3iS417tEZHgAABAo1BoKgxDMIYCBAgQIAAAQIEdp7Aa5lpT30wd/4BD/bqLfi886bBkwkQaCICM2bODg8+8liLlq1aXvXtB969o4kM2zAJECBAoAELCEA34MnTdQIECBAgQIDAzha4sc/Uv5dXVB79xJNDdt3ZffF8AgQINBWBZctXhP6DhrTqvNuu37nhnjevbyrjNk4CBAgQaJgCAtANc970mgABAgQIECCw0wV+PXLRVS1KW17Tb+CTnXZ6Z3SAAAECTUwgbvL65LDhRV327PLn20et8CVgE5t/wyVAgEBDEhCAbkizpa8ECBAgQIAAgQISqFxT9tvnRo9rvnzFygLqla4QIECg6QhMmDQ5JD+ZmbNm39d0Rm2kBAgQINDQBASgG9qM6S8BAgQIECBAoAAEfjVk4TfWrC7r8Obbb7cogO7oAgECBJqswLjn/9GsS+fOV/xm2OLDmyyCgRMgQIBAQQsIQBf09OgcAQIECBAgQKBQBdI3Pjt2vNQbhTo9+kWAQJMRWLZ8efjHK6+FxUuX/F+TGbSBEiBAgECDEhCAblDTpbMECBAgQIAAgZ0v8Pun5h+TTld3mTx12s7vjB4QIECAQHhv4qTQuk3rM1EQIECAAIFCFBCALsRZ0ScCBAgQIECAQAELrKkOV06bMWN1AXdR1wgQINCkBOKGhJWVa0tue3rJ6U1q4AZLgAABAg1CQAC6QUyTThIgQIAAAQIECkcglcpc9s57k3YrnB7pCQECBAhMmDSp5dLlK75CggABAgQIFJqAAHShzYj+ECBAgAABAgQKXKA6k+m2aPGSAu+l7hEgQKBpCSxZujSkq9P7N61RGy0BAgQINAQBAeiGMEv6SIAAAQIECBAoIIGSopKOq1bLwFFAU6IrBAgQCKtWrQ4lqaI9URAgQIAAgUITEIAutBnRHwIECBAgQIBAAQv8cezqPdLpdGVFRUUB91LXCBAg0PQE4heDJSUluza9kRsxAQIECBS6gAB0oc+Q/hEgQIAAAQIECkigurq6ffXa6soC6pKuECBAgEAiEL8YLCkubgmDAAECBAgUmoAAdKHNiP4QIECAAAECBAgQIECAAAECBAgQIECgkQgIQDeSiTQMAgQIECBAgAABAgQIECBAgAABAgQIFJqAAHShzYj+ECBAgAABAgQIECBAgAABAgQIECBAoJEICEA3kok0DAIECBAgQIAAAQIECBAgQIAAAQIECBSagAB0oc2I/hAgQIAAAQIECBAgQIAAAQIECBAgQKCRCAhAN5KJNAwCBAgQIECAAAECBAgQIECAAAECBAgUmoAAdKHNiP4QIECAAAECBAgQIECAAAECBAgQIECgkQgIQDeSiTQMAgQIECBAgAABAgQIECBAgAABAgQIFJqAAHShzYj+ECBAgAABAgQIECBAgAABAgQIECBAoJEICEA3kok0DAIECBAgQIAAAQIECBAgQIAAAQIECBSagAB0oc2I/hAgQIAAAQIECBAgQIAAAQIECBAgQKCRCAhAN5KJNAwCBAgQIECAAAECBAgQIECAAAECBAgUmoAAdKHNiP4QIECAAAECBAgQIECAAAECBAgQIECgkQgIQDeSiTQMAgQIECBAgAABAgQIECBAgAABAgQIFJqAAHShzYj+ECBAgAABAgQIECBAgAABAgQIECBAoJEICEA3kok0DAIECBAgQIAAAQIECBAgQIAAAQIECBSagAB0oc2I/hAgQIAAAQIECBAgQIAAgY8hkMlkUh/jNrcQIECAAIHtKiAAvV15NU6AAAECBAgQIECAAAECBAgQIECAAIGmKyAA3XTn3sgJECBAgAABAgQIECBAgAABAgQIECCwXQUEoLcrr8YJECBAgAABAgQIECBAgAABAgQIECDQdAUEoJvu3Bs5AQIECBAgQIAAAQIECBAgQIAAAQIEtquAAPR25dU4AQIECBAgQIAAAQIECBAgQIAAAQIEmq6AAHTTnXsjJ0CAAAECBAgQIECAAAECBAgQIECAwHYVEIDerrwaJ0CAAAECBAgQIECAAAECBAgQIECAQNMVEIBuunNv5AQIECBAgAABAgQIECBAgAABAgQIENiuAgLQ25VX4wQIECBAgAABAgQIECBAgAABAgQIEGi6AgLQTXfujZwAAQIECBAgQIAAAQIECBAgQIAAAQLbVUAAervyapwAAQIECBAgQIAAAQIECBAgQIAAAQJNV0AAuunOvZETIECAAAECBAgQIECAAAECBAgQIEBguwoIQG9XXo0TIECAAAECBAgQIECAAAECBAgQIECg6QoIQDfduTdyAgQIECBAgAABAgQIECBAgAABAgQIbFcBAejtyqtxAgQIECBAgAABAgQIECBAgAABAgQINF2BkqY7dCMnQIAAAQIECBAgQIBA4xE4oOse4XOXdA8LlywPvYeOCStXlzWawZW2aB5aJj+xVFZVhdVryhvN2AyEAAECBAg0dgEB6MY+w8ZHgAABAgQIECCwUwX27rJrOP7wA8JhB3YNLZqVhMkz54Q33p0aJk7/IGQymZ3at8bw8JalLULb1i0/HEomLF62MlRXpxvD0LZqDM2Tz9Yt3/tKSKVS2fsO3m/P8MPf3r9VbRRq5TikO396fWjfplW2i+/PWxT++7Z7C7W7+kWAAAECBAjUERCArgPiLQECBAgQIECAAIH6EPjKleeGS848sSYgmGvziIP2DVeed0o2+PyvyTPDb/7eJ6TTAtE5n615veD048O1nz4vFBetyywY4/n/+9de4Z3Js7ammUZR9xNHH1Lrs7Z3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AAEEEHALEIB2e7CHAAIIIIAAAggggAACCCCAAAIIIIAAAgggECIBAtAhgmQYBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAbcAAWi3B3sIIIAAAggggAACCCCAAAIIIIAAAggggAACIRIgAB0iSIZBAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQcAsQgHZ7sIcAAggggAACCCCAAAIIIIAAAggggAACCCAQIgEC0CGCZBgEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABtwABaLcHewgggAACCCCAAAIIIIAAAggggAACCCCAAAIhEiAAHSJIhkEAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBwCxCAdnuwhwACCCCAAAIIIIAAAggggAACCCCAAAIIIBAiAQLQIYJkGAQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAG3AAFotwd7CCCAAAIIIIAAAggggAACCCCAAAIIIIAAAiESIAAdIkiGQQABBBBAAAEEEEAAAQQQQAABBBBAAAEEEHALEIB2e7CHAAIIIIAAAggggAACCCCAAAIIIIAAAgggECIBAtAhgmQYBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAbcAAWi3B3sIIIAAAggggAACCCCAAAIIIIAAAggggAACIRIgAB0iSIZBAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQcAsQgHZ7sIcAAggggAACCCCAAAIIIIAAAggggAACCCAQIgEC0CGCZBgEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABtwABaLcHewgggAACCCCAAAIIIIAAAggggAACCCCAAAIhEiAAHSJIhkEAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBwCxCAdnuwhwACCCCAAAIIIIAAAggggAACCCCAAAIIIBAiAQLQIYJkGAQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAG3AAFotwd7CCCAAAIIIIAAAggggAACCCCAAAIIIIAAAiESIAAdIkiGQQABBBBAAAEEEEAAAQQQQAABBBBAAAEEEHALEIB2e7CHAAIIIIAAAggggAACCCCAAAIIIIAAAgggECIBAtAhgmQYBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAbcAAWi3B3sIIIAAAggggAACCCCAAAIIIIAAAggggAACIRIgAB0iSIZBAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQcAsQgHZ7sIcAAggggAACCCCAAAIIIIAAAggggAACCCAQIgEC0CGCZBgEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABtwABaLcHewgggAACCCCAAAIIIIAAAggggAACCCCAAAIhEiAAHSJIhkEAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBwCxCAdnuwhwACCCCAAAIIIIAAAggggAACCCCAAAIIIBAiAQLQIYJkGAQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAG3AAFotwd7CCCAAAIIIIAAAggggAACCCCAAAIIIIAAAiESIAAdIkiGQQABBBBAAAEEEEAAAQQQQAABBBBAAAEEEHALEIB2e7CHAAIIIIAAAggggAACCCCAAAIIIIAAAgggECIBAtAhgmQYBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAbcAAWi3B3sIIIAAAggggAACCCCAAAIIIIAAAggggAACIRIgAB0iSIZBAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQcAsQgHZ7sIcAAggggAACCCCAAAIIIIAAAggggAACCCAQIgEC0CGCZBgEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABtwABaLcHewgggAACCCCAAAIIIIAAAggggAACCCCAAAIhEiAAHSJIhkEAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBwCxCAdnuwhwACCCCAAAIIIIAAAggggAACCCCAAAIIIBAiAQLQIYJkGAQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAG3AAFotwd7CCCAAAIIIIAAAggggAACCCCAAAIIIIAAAiESIAAdIkiGQQABBBBAAAEEEEAAAQQQQAABBBBAAAEEEHALEIB2e7CHAAIIIIAAAggggAACCCCAAAIIIIAAAgggECIBAtAhgmQYBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAbcAAWi3B3sIIIAAAggggAACCCCAAAIIIIAAAggggAACIRIgAB0iSIZBAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQcAsQgHZ7sIcAAggggAACCCCAAAIIIIAAAggggAACCCAQIgEC0CGCZBgEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABtwABaLcHewgggAACCCCAAAIIIIAAAggggAACCCCAAAIhEiAAHSJIhkEAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBwCxCAdnuwhwACCCCAAAIIIIAAAggggAACCCCAAAIIIBAiAQLQIYJkGAQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAG3AAFotwd7CCCAAAIIIIAAAggggAACCCCAAAIIIIAAAiESIAAdIkiGQQABBBBAAAEEEEAAAQQQQAABBBBAAAEEEHALEIB2e7CHAAIIIIAAAggggAACCCCAAAIIIIAAAgggECIBAtAhgmQYBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAbcAAWi3B3sIIIAAAggggAACCCCAAAIIIIAAAggggAACIRIgAB0iSIZBAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQcAsQgHZ7sIcAAggggAACCCCAAAIIIIAAAggggAACCCAQIgEC0CGCZBgEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABtwABaLcHewgggAACCCCAAAIIIIAAAggggAACCCCAAAIhEiAAHSJIhkEAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBwCxCAdnuwhwACCCCAAAIIIIAAAggggAACCCCAAAIIIBAiAQLQIYJkGAQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAG3AAFotwd7CCCAAAIIIIAAAggggAACCCCAAAIIIIAAAiESIAAdIkiGQQABBBBAAAEEEEAAAQQQQAABBBBAAAEEEHALEIB2e7CHAAIIIIAAAggggAACCCCAAAIIIIAAAgggECKBmBCNwzAIIIAAAggggAACCCCAQFgJvPTSS/Xz8w7N8XhkZ2phzMTxt966rbwT9Pl8nuef+c/bXk9UX5GEC2+88cYfy3st/RBAAAEEEEAAAQSOCZABfcyCLQQQQAABBBBAAAEEEIgggaLcA+M84ussPt+Zh2IL/1mRR3v26f8+4hPPxR6fr4148y6ryLX0RQABBBBAAAEEEDgmQAD6mAVbCCCAAAIIIIAAAgggEEECsUnJM83j5FmP5PVd9uyzj11Xnsd74YVHzzVZ0z+z+3p88pm9zScCCCCAAAIIIIBAxQQIQFfMi94IIIAAAggggAACCCBQQwSuuuqGtZ5o3932dD0+7z/fevLJ5vZ+sE8tvVFUKI/Y53y+qIduuOWOWfY+nwgggAACCCCAAAIVEyAAXTEveiOAAAIIIIAAAggggEANErj++rseF598pFP2+ST9eKU4tPSG+DxdtL8nyrPwxpvv+I1u0xBAAAEEEEAAAQQqJ0AAunJuXIUAAggggAACCCCAAAI1RCA+KekXZqrHLcURWHrDV+jxZ0/XkEdlmggggAACCCCAQNgJEIAOu1fChBBAAAEEEEAAAQQQQCCUAsFKccQVSRPnPbKysym94QRhGwEEEEAAAQQQCJFAdIjGYRgEEEAAAQQQQAABBBBAIGwFPvzw0wXnjxnVTzzSyUwywRflaeITX0d7wvmewlRTdGO07mvpjRtuunOCfY5PBBBAAAEEEEAAgcoLkAFdeTuuRAABBBBAAAEEEEAAgRok4CzF4fV5xzin7hPPbfY+pTdsCT4RQAABBBBAAIETFyAAfeKGjIAAAggggAACCCCAAAI1QCCwFEewKft8UQ/dcMsds4Kd4xgCCCCAAAIIIIBAxQU8Fb+EKxBAAAEEEEAAAQQQQACBmivw7FP/+dCU4hgb+ARaeuP6G+4cEHicfQQQQAABBBBAAIHKC5ABXXk7rkQAAQQQQAABBBBAAIEaKKClODweyQ+cOqU3AkXYRwABBBBAAAEETlyADOgTN2QEBBBAAAEEEEDg5Aj4fJ5rv331PG+Mp1lebmErb0FRG19RUeOTMxnuikDNEkgo8LZJLPC1sWedF+3ZnBMftdbe5xMBBIILeHw+rycueld8bNxaiYrd6o3xbp3U74pPg/fmKAIIIIAAAmaBZxAQQAABBBBAAAEEapbAxAWv/uTIoexrklISBx3ee7Dw0P6DcTkHs6KPZOVIQV6JpM6a9XDMFgEEEEAg7AXiEuIlISVJElNTCtIbpBcmp6d48nILpiWlxL360oCrXg/7B2CCCCCAAALVKkAAulq5uRkCCCCAAAIIIFB5gSvmvnqVtyjvrwf3Hqy/dcW6+B1rt0hW5qHKD8iVCCCAAAIIhECgTv10ado+Q1r2aJ+dnJqyIzYx4VeTBlzxbgiGZggEEEAAgQgQIAAdAS+RR0AAAQQQQACByBa4aeF77Q5kH3gvLyen4/JZ38VvWbE+sh+Yp0MAAQQQqLECrXt0kG5D++fExMb9WL9B0rgnuo3fWWMfhokjgAACCIREgAB0SBgZBAEEEEAAAQQQqBqB0R/+6/LU+nWfXTN/adKyWYuq5iaMigACCCCAQIgF+pw78GDrrh0K4xJjLnhpwMS5IR6e4RBAAAEEapAAAega9LKYKgIIIIAAAgjULoErv3n9uqgo73MLPvpKNi5ZU7senqdFAAEEEKjxAu36dZG+I88Qn8dzCSU5avzr5AEQQACBSgsQgK40HRcigAACCCCAAAJVJzDmo0fG1m3c+MNZr0+VXRu2Vd2NGBkBBBBAAIEqFGjeqbUMvOisgqK4+FNe6zP+hyq8FUMjgAACCISpQFSYzotpIYAAAggggAACtVbgupUfNEtOS3/lu0/nEnyutd8CHhwBBBCIDIFtqzbKklk/5BYeOvzRdXvm1ImMp+IpEEAAAQQqIhBTkc70RQABBBBAAAEEEKh6gcPb976wadnaOuu+W1H1N+MOCCCAAAIIVLHAyrnf1amTlhzXIjruv+ZW11Tx7RgeAQQQQCDMBMiADrMXwnQQQAABBBBAoHYLnD/1ifOj42LOXPbVQv49rXZ/FXh6BBBAIKIElsxaFB8dEzXx2oWv9I2oB+NhEEAAAQSOK8B/2ByXiA4IIIAAAggggED1CURHef6+fNbCuKLCwuq7KXdCAAEEEECgigVys3Jkzbc/HsrOyn+4im/F8AgggAACYSZAADrMXgjTQQABBBBAAIHaK3D1olf7xcbHtV797dLai8CTI4AAAghErMCSrxbWjUuIG3TTwvfaRexD8mAIIIAAAiUECECXIOEAAggggAACCCBwcgQyd+6/dduaDXEn5+7cFQEEEEAAgaoV8BZ5ZffmnZm53pwxVXsnRkcAAQQQCCcBAtDh9DaYCwIIIIAAAgjUaoG4xISRO1Zv8dRqBB4eAQQQQCCiBTYtWVM/70j+lRH9kDwcAggggIBLgAC0i4MdBBBAAAEEEEDg5AjcPv+l+gkpSc13rN1ycibAXRFAAAEEEKgGgR1rN0fHJcYN8Pl8xCOqwZtbIIAAAuEgwP/gh8NbYA4IIIAAAgggUOsFsouim+VlH8k3/0Fe6y0AQAABBBCIXIH8I3ni9XrzJ86f3Cxyn5InQwABBBBwChCAdmqwjQACCCCAAAIInCSBqLiYpkeyjxB9Pkn+3BYBBBBAoPoE8nPzcySqoGn13ZE7IYAAAgicTAEC0CdTn3sjgAACCCCAAAJHBbweX7PcQ1l4IIAAAgggEPECRw7n+OK8QgZ0xL9pHhABBBAoFiAAzTcBAQQQQAABBBAIA4GiwqKEgvyChDCYClNAAAEEEECgagV83nyJKuT/51WtMqMjgAACYSNAADpsXgUTQQABBBBAAAEEEEAAAQQQQAABBBBAAAEEIkuAAHRkvU+eBgEEEEAAAQQQQAABBBBAAAEEEEAAAQQQCBsBAtBh8yqYCAIIIIAAAggggAACCCCAAAIIIIAAAgggEFkCBKAj633yNAgggAACCCCAAAIIIIAAAggggAACCCCAQNgIEIAOm1fBRBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAgcgSIAAdWe+Tp0EAAQQQQAABBBBAAAEEEEAAAQQQQAABBMJGgAB02LwKJoIAAggggAACCCCAAAIIIIAAAggggAACCESWAAHoyHqfPA0CCCCAAAIIIIAAAggggAACCCCAAAIIIBA2AgSgw+ZVMBEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQCCyBAhAR9b75GkQQAABBBBAAAEEEEAAAQQQQAABBBBAAIGwESAAHTavgokggAACCCCAAAIIIIAAAggggAACCCCAAAKRJUAAOrLeJ0+DAAIIIIAAAggggAACCCCAAAIIIIAAAgiEjQAB6LB5FUwEAQQQQAABBBBAAAEEEEAAAQQQQAABBBCILAEC0JH1PnkaBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAgbAQIQIfNq2AiCCCAAAIIIIAAAggggAACCCCAAAIIIIBAZAkQgI6s98nTIIAAAggggAACCCCAAAIIIIAAAggggAACYSNAADpsXgUTQQABBBBAAAEEEEAAAQQQQAABBBBAAAEEIkuAAHRkvU+eBgEEEEAAAQQQQAABBBBAAAEEEEAAAQQQCBsBAtBh8yqYCAIIIIAAAggggAACCCCAAAIIIIAAAgggEFkCBKAj633yNAgggAACCCCAAAIIIIAAAggggAACCCCAQNgIEIAOm1fBRBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAgcgSiImsx+FpEEAAAQQQQAABBBBAAAEEarpAWmrq0UfwyeGsbPF6vTX9kZg/AggggAACtVaAAHStffU8OAIIIIAAAgggUDsEPB6PtGvbWrp37SytW7WU/Px8Wb12vSxbvlJ27NxVOxCq8CmbNmksQwcPlIYN6ktCfLxs2rJVFv+4zBivE5/PV4V3ZuhIFTj9lP7ykyvH+x/vb4/8VzZu2uLfZwMBBBBAAAEEapYAAeia9b6YLQIIIIAAAggggEA5BWJjYuRnd9wsbVq3FA1CO1uPbl3k4gvOk6KiIpn6+XT55NMvnKfZLodAYkKC/OruO6VRo4au3m3btJJhQ86wgs9z5s2X196c7DrPDgLHE0hITDheF84jgAACCCCAQA0SIABdg14WU0UAAQQQQAABBBAon4Bm4973izslOTmpzAuio6NlzKhzpGOHtvKfJ561AtLOCxrUrye//80vxQ5ffzTlc/nsyxnOLrVyOy0tVX77q19IclLpvhr0HzzwNGnUsIE8+vgzEZ0NPXb0SBk5/Ezru6A53w/++R+yd9/+Wvnd4KERQAABBBBAAIFAARYhDBRhHwEEEEAAAQQQQKBGC2hm7u/uv6dE8Dk3N1e2bd8pe/bulcLCQtczdmzfTu79xR2uY7qTkpIiMSZIrYFq/UlIiC/Rp7YdiIuLkwcfuNcVfC4sLJKt23bIilVr5NDhwy6STh3ayy03XO06Fmk7SSZj1/6OREdHSYzJvqchgAACCCCAAAIIFAvwb0Z8ExBAAAEEEEAAAQQiSuCScWOtoLH9ULl5efLkMy/K6jXr7EPW5/nnnSvnnjPMX56jZYvm0rZ1K1m/cZOrHztugbGjR0i8CULbbX/mAXnwoX+Y2toF9iE5/7yRMmrE2f59LXkSFxfr6uM/yQYCCCCAAAIIIIBARAuQAR3Rr5eHQwABBBBAAAEEapeAlsro2KGd66H//d+nSgSftcOHn3wq07+a4+p7yoC+rv0T2UlJSZbevbpbCyAG1qAuz7j16qZLj+5dpXmzpuXpXmqf1Dp1rAUY+/buaeo1N/AH3Eu9oIwT6tu3Vw9/D6/XK3/4899LBJY//OQzK9vc7ugxRUx0AchQtKioKOluAtrt2rQW3Q7WNAO5U8f21nNrZnJFWrxZSLF9uzYyoF8faZnRvMqzmRvUry8D+vYWXcyxupva6OKcnTt1KPM5QzVHLWnTr08vqV+vbrU+aijfaUNTUqZXj27WopvV+hDcDAEEEEAAgRosQAZ0DX55TB0BBBBAAAEEEEAgQMDUHdbAr90KzSKDm7ZstXdLfH7w8afSrUsnfyDzyJEjVp9f3HmLFYQMvODcc84ytX6HWYe3bd8hf/77v11dGpsF+bTchNagdgY+fT6fHDElQJYsXS4vTnrTdY1zJ8rMf+IVl1rBTy39YTe9ftXqtfLMi5NEM7xPG9DPOqWlL356z/12N9fnRWaRxSFnnCYafHO2oiKvLF+xUl589U3JySl+Xuf5MrfN/JIdvpkHDkpBgbuciX39mrXrTPC8SfGuiVw3MiaBWeh238BPDdj/958P+d/L94uXyKdfTJebrvuJFby0A/rq8qMxffbFV62yKk0aN5Kbr/+J6KfdtE92do68ZJ536fKV9uESn6eaXz5cMOZcqZueXuLclq3bZNIb78jmLdv853QO//zLH0xZlgRXUF+D7b/79d3+ftNmzJJ3P/jEv68bvXt2l8vHXyR1jKX9LHpc53o4K0umfDZNvpr9tR464RbM8utvFsiVEy6W9LS04vubwtX//d+zsnzlav/9KjvH35vyN/rPgbYDBw/JAw/+RX562w3SplUrKwvevoE+q87j1TffrVR98P59e8l1P7nCHs6q337fb//P9Z2u6Du1B7vCvJtBA0+1d+W2n90nt910rWg5Gc3k16Y1vn/7x7/6+7CBAAIIIIAAAqULEIAu3YYzCCCAAAIIIIAAAjVNwAS1DpmgV8LRAJgGcVtltCg1CF1QUGDKRzxc4iljY2NcgUFnBztgmJx8LNCt50eNOEt0MTr7fOA1SYmJcqoJHGtG898eeaxEHWrtf9ftN5XI4NbjOqZmqT704G9ktQlE2/eIiTkWpNZ+2hJMwPnX9/zUZDsXBwGLjx77U2sUa2b1A/f9wgqgFZkgfXmbBg1ffvUtqxa2ZkNv27Gz1Es1MGs3XZhvw6Yt9m65PjW72X5OXUzy7p/eWiKYruc1G/VXd98pz730mrUwon2NfRPd119K3HrjNSbI+pysNHWqA9s1EyeYd1N69nuGKc/yy5/fIf/3l3/K7j17/ZdHRwf/njjnkGaCvM52xy3XW7/0cB6zt/U6zVifcMmF0tQE0d9453371Al9Oi3TUlPlFmPh/AVH4OCVnaN+JzT73H5+/efonrtukzatS2a/a58zTj/Fyoz/098eCZxCmfv6SyMNPtv30e/lU8+97Ao+V/ad6o2dz6D7V11+qWgZGRoCCCCAAAIIVE4g+N9Zq9xYXIUAAggggAACCCCAwEkV0EDnmnUbXHPQwKFmM9rBKtfJUna279hlZfZqBrWzackJzfjVn4Mm0G231NQ6JYLPGhTTDGP9dLYWzZvJ9Vcfy9y0z1160flBg8/OOWjtZS0/UVb72R03lQg+6xhaC9vZ6qanBV140dkn2PZ3P/xoZa7ONdmrG0sJKmvAs0d3xzyNgWaMV7Zp5qmdye30sMfToP79v7zL/47VPDCwrnO64Zor7Uv8n2NGjSgRfNbrs3NyXO9OA7Ya2NdguDbtsz8zM2gGuC5yWfw9KbAymu2bacBVg6fOpt+pYJnoQwcPNJnwvZ1dQ7Ldrm3rksFnEznOy8+3xg/lHFPML2mcwWdnnXD7YTRL/pT+pQf/7X72p5ZHuf3m61zv+slnX3Rlt1f2ndr3CPwceNqAwEMlvl8lOnAAAQQQQAABBPwCZED7KdhAAAEEEEAAAQQQiASBOfO+FQ0Y2QFnzfi98rKLrazSTZu3WmUGtKTD9jKyd195/W3RH61bfN8v7vCzfD5tpmjZjsB22cUX+O+n53QOrx0tLaCBT617e83Ey/wlJdq3beMaQoOCZw0d5Dr27cLv5cMpn8k+81f9NVis5T+GDDrddR/XBWZHA5atWmb4D2sQ9t+PPy1rjwblNVB+3y/uFK0vrU0XXtRyA6+9Ndl/TSg2NIM3Oak4UKvjOUs7VHb8vLx8efjRJ2Trtu1WeZM7b73eKolgj6dZqxoUVnf116b1jbV0gv1d0Cx07acBYm11UlJk9MizrW37j3c/+FimzZhtjRUbGyvXXjVB+hyte61Z3ff+7Hb5/Z//YXW3s+f1/Z855AzrmE98VmmWnbt220Nanybh1/olhX1Q56pZ24u+X2wdSkxMkJFnD5ORZmFMu2lt6AWLfrB3Q/o579uF1i8S1q3fKHpvDYJX1RynfPalVUJFg/KJ5h1oNru/PIt5Kl0M9NuF3x33+TJaNJOfmb8lYL9PNdTM5yVLV/ivPdF36h8oYCMrK9v651H/t0OtNLubhgACCCCAAALlEyADunxO9EIAAQQQQAABBBCo08q73gAAQABJREFUIQIbN22WFye94cpe1alrTea2bVrJmFHnWKUa/vPwn616zRrUOtG21WT3LvxusfWjtYpfNfWCNTimTTNcFyz6XuY7AmyaRRsfH+e/bX8ToHa2FaZMxAuvvG4Fn/W41lp+/e33rIChs1/g9sjhZ/kP6f0ff/oFf/BZTxw6dFgeMnWr7WxXPRbqLNvTT+kvg03tabtpsFdrNJ9I02f56z//YwWfdRwNrD/6+DOu59DjH0/9wh981n2t+ax1hu2mgUtnBvKIs4f6g5na54vpX8mX02f5352WaHn6+Vf899U+ugid/kKgok2/Dlr+w/6eaE1qO/isYx05kmt+uTFV9u/P9A/dpnUr/3YoN/T7qaVU9BcTamtnYFfFHHWhz4+mfO7PFNc663//12PWPxf2M+kvRo7XGhn3e83fZrBrq+u89d0sXrLMdWlVvFP92w6/efAhmT33G9FAtP4zrb8QoSGAAAIIIIBA+QT4tW35nOiFAAIIIIAAAgggUIMENHtYg0SXjjtfSgtuaXar1g/WHy0P8ZeH/1Ppv1Y/1Swad7y2bZujBIUJhDZs0MAf2HSWKdDA2mOmVnGwpgvh9TdZsfZCaM4+Glxt1rSx/9AeU6t4hWNROfuElpb4YfFSf9kJLW2hiybu2bvP7lLpz66dO8pVZhFFZ3vz3Q8k1yzAeCJNF7MLzChWp10my7ilqfGtTfdnzp5b4jYrV6+xag3bJ+rXq2ttmoRk6dO7p33Y+r5MDlgs0D6pWe9a9kGbOmtJFw2qVrTpL0bKavorC13crt7ROSYkuBeQ1JIWdeqklDWEdW7vvn3+gG9gZw102xniged0/0Tn6BxT38nHQZzyTbkPraVtLxapdcvLarpY4s/uuNnKXtd+Ou4zL0ySH35c6rqsqt6p/s2HYOVDXDdnBwEEEEAAAQRKFSAAXSoNJxBAAAEEEEAAAQRqsoCdadq0SWM579xzpGvnDtZf/w/2TFpD+MEH7pU//uXhEw40acmNzh3bW4sGpqelSpIpRaGlH1q1LA6UBt5fA5qNHQsGaoBYg+fBmgbeNFDsLF9g99N76Vh200UIf3HnLfau6zMjo7l/vzhw3eSEA9BarsRZm1dv8NWceTLn6/n+e1V2Y8fOXUEvzc09Vte6yJjZmbzOznv27nfuHts2Vro4od30vZXmpWU7nE1/eXCiTbOo9ZcJDUzwP8VkxOt3REt8tG51rIRK4D0mXn6J9QuTwOOB+4u+/9FknU8KPGzta7Z+eVtl5ugcWwPNR0r55YNm9dsBaOf31nm9vX3TdVe5vttaukRLYZRoVfBO9Z85599eKHFPDiCAAAIIIIDAcQXc/yZ13O50QAABBBBAAAEEEECgZglo8NIOxmndVs14HnjqAOnQoZ1rMTbNjL3ogjHyhil1UZmmWcw3X3+1pJoM1eMF1ALHtxfY0+MHTGCurLZr956gAehGDRuWuKxD+7YljgU70KB+vWCHy32sedMmcvddt/prXOuFulhhZS0Db1x0tGZz4PET3Y8zWfDOVl6vtHKUjHCO69zWWtEXjh0luqBkVbWy6hPnmhIYx2uhmqPWfC6t2XW4SzvvPB74z1OGWcjTWb7E2bcq3ml2do7zFmwjgAACCCCAQAUFCEBXEIzuCCCAAAIIIIAAAjVXQANidmZ0nAkA6oJmzvIXHcsZsA0UOOesoTLu/NEVDjzb4+SbWsN2QDK1Ttn1cLVcRrCmiy1WtpWWpVqe8XQ+v7rnp65g/pJlK6wSCeW5vib20fdVmaYLIvbo1qUyl1bbNTVhjiOHD5OVa9ZaNbVDBVPZdxqq+zMOAggggAACkSxAADqS3y7PhgACCCCAAAIIIFCqgJYHePjRJ+TRf/zJX1u2tOBuqYOYE5qdOWbUCFfwWesOaxmELVu3WRnNhw5niWaVjr/o/BJDWWU19uyTFs2bWuecZSECO+u9GjYMHoDeZWrqOpvWwdaF7srTSiv5cbxr000Zid/c93O/n/bXhf+eMIsf1oSmv5Cw62nvzzwgv/3jX8s17cp4aZa5M/isJUO+XfCdLFuxSjRL/+ChQ6KZtj+74ybp1KF90Hk8//JrEhd7/Mzp/ILKLZAXijkGnXgIDq7bsFHatWldPJKpNKOB8vt/92fJys52jV6d79R1Y3YQQAABBBBAoFQBAtCl0nACAQQQQAABBBBAoKYJdO/aWW654Wr/tN81i8rN+GqOfz9wQ4O/hUVFrgBqYB/nfmApAD3XzNSYtoOYur9m3QZ59PFndNPV2poSHaW1DZs2+QPQWov42qsmyAuvlFywbuzoEVLagm2ZJoCqz2PPMaNFs1JrSZc2j4oc10D57399tz9zW6/VYOrjTz1fkWFOYl+fVZ/Yfne6WGVlAsslH8DjfwfOc4PPOM25K+9/NEW+nD7Ldcz8fsFVD9x10uzoQnhVuRheKOYYOOdQ7OvCiBqs/9Pvf+1foDHW1OX+5c9vl9//6e+OW1TVO3Xcgk0EEEAAAQQQqLBA5f+eXoVvxQUIIIAAAggggAACCFStgC7QFx0d7f8ZPeLsMm/YqmWGK6AbbBE75wBtzEJ7ga2zWdzQ2WbPnefctbY1qNy1c6cSx+0D3//gXlDtlP59i+sExxdnu8aaWsVadmBUGc/jMxm1ex0L7ukCb/6MUftGRz+7du4od9xyvfWjAfv4o/cJ6FbqbqJZLO/3999jLZpnd9Lg82P/e87eDftPE6uXpaZUiN1izPdm9Mjh9q7rs05Kitxx83WW153GrayFAnUZSPUNbM5r9BcF02bMDuwi6Wlp1k+JE9V0IFznqHXPzeuSv/3rMesXRjZHo4YN5JqJl9m75hcwVfNO/TdgAwEEEEAAAQQqJUAAulJsXIQAAggggAACCCAQjgK7TRmK3Nxc/9Q0S/ev//dbEzAsGTg+c/BAuddkUDrb4qXLnLsmoLvPtd+mTStp17a169jiH93XaJDYGdDVjOUHH7hXkpISXdc5d1asWiNaMsPZNOD877//SR5+6A9WmRBduK6spgG6r+Z87e+imdBazqFlRnP/Md3o27uH3GrKF3Tr0sn/U9Zica6LzY4Gw3//m19KSnKy/9T6DZvkZVPuQxdgLO0nxmSshlv77MsZVta4PS/NMB9qvhfO1tRkuD9gyox0M9n1atbV/Pi8qn2safkOZ9NyK7qopbOtXrPOv6vv5vLx41yZ0pqx/off3OvvczI2wn2Ohw4dlqeefclFc+qAfjKgX2//sVC9U/+AbCCAAAIIIIDACQuE378FnvAjMQACCCCAAAIIIIBAbRXQzNI33vnAlRWZZkor3PeLO6zMyWxTL9bjiZI6JjBtl6qwrfJMTejJH0yxd63P7JwcqyyDZjBr07/2f89dt0lhYZGsNzVp//XYU7Jvf6bk5eWZoHO81UcDlho4PnjwkOh1GgQPvJfVMeCPF155XfRaDUQ6W3Jykn9X77vKLL6mgdBgbfrM2Vataa3lq02Dvr++5y4z30I5YOajcwks4fHljFkVKj2hQW01dba2JjD/tz/9znmoxLbW8H3430+UOH4yD2jG/Lz5C2XgaQP805hwyYVy2cUXiJY00Xfq9NdOm01d701btvr764Y+m7Opv5aLKDCLFWpA9JNPv5Svv1lg1Qq3+w0eeJoMOv1U6/sT7L3Y/arzsybMUWuMz5g1V4aZIL/drr7yMtmwcbPs3bdfQvVO7bH5RAABBBBAAIETFyAD+sQNGQEBBBBAAAEEEEAgjATmL1gkTz//iuuv6uv0tMRCWmqqlaEbGBDWIPIDf/iLK3tar9GAtgblAltMTLRZDLCBdVj7vP72+65MWj2RlpYqdUxGsH0vzXI+XvuHKTHww49LS4yl12mA7cGH/iGHzYKGpTXNy9WFFTVT1Nk0EK1B0cDg86LvF8sHH3/q7FrmtpaXSDeGlWmaOR2O7dU33xFn5q/OUd9ZPZPBHBh83m++J//41+MlHmODyQDfYxaSDGz6zI0bNbIOZx44aAW7nX30Ps73UmTqkW/bvtPZpVq3a8IcFeStdz8wTjv8Nlp2R+tB278oCsU79Q/OBgIIIIAAAgicsAAZ0CdMyAAIIIAAAggggAAC4Sbw/eIl8sCDf5Gf3X6TtaibHQQOnKdmBv+4dIU8++KkoEFf7f/6W5OtwPTgQae7FtzTmst206B3Zmam3HTdT0oELTWL+s133jfHk6VLJ7tetM8sJpdvX+7/LDDzeeq5l60AqNZwbta0iZUhu8Vk3WpwUltCQnGmtf+igA3NvL7/Dw/JlRMulgF9ewddYFGzod94+z1ZvMRdPiRgqKC7umhjdTQN7NvvLb+gMOgtQzEXrymnoZnsw88aYtXZdpYWsW+al5cvn0+bKVM/nxb0e6KB/7+bXx6MO3+09Ovby/U9cS5s+PJrb8nuvXvl3HPOcvXR++zctVuee/FVuXjcWPu2Qe/lP1mBjfJY2sOdyBzVwf6e6njObXv8YJ86v4q2fz76pPzl/x7wO6bWqSOaCa1/k+BE32m+yVynIYAAAggggEDoBDSJgYYAAggggAACCCBwkgWumj/pls0r1j85//0ZJ3kmkXl7LW2hC8NltGhuMqMLZdXqdbJ85SrJzs6p8ANrVmycyWzVIG6wwJlmG3c2geb4uDirTIdmlR6v1aubLrogot20zEawBRH1X97/+LtfWVmz2ldrN//0nvvty4J+ahC7ebOmpnZzjOzatccqH6GBd1pwAX0X+j3RhQcPHDwo6zduCvougl9dfDQxMcG8/3jru5aVlR20q36PunXpbMbOkXXrN8oRR+3yoBechIM1YY7lYQnFOy3PfehTfoHh1124q2GT+ne9cMrEN8t/FT0RQAABBGqqABnQNfXNMW8EEEAAAQQQQACBcgvs2LlL9CcUTYPWwUOKxaNrcHfpshUVulXLjBYme/oq/zUatNYMbmf2rJ481yxwaNd31v39Juv6eG37jp2iP7TyCeiCgoGLCpbvymO9jhzJFf0pq+n36NuF35XV5aSfqwlzLA9SKN5pee5DHwQQQAABBBAILkAAOrgLRxFAAAEEEEAAAQQQqDaBxUuWSpZZINEu/1A3PU3+9bc/WmUZNm/ZJppR275d2xKL/73/0dRqmyM3QgABBBBAAAEEEECgMgIsQlgZNa5BAAEEEEAAAQQQQCCEAloCV+tEO0t6xJkSHpoZPWjgqdKvTy9X8Fn7vTX5Q2vBwhBOg6EQQAABBBBAAAEEEAi5ABnQISdlQAQQQAABBBBAAAEEKi6w8LvFpmb0Zrnz1utFFyAsrWlN4ceffl42btpSWheOI4AAAggggAACCCAQNgIEoMPmVTARBBBAAAEEEEAAgdouoDWdH3zoYUlKSpQOpuRGm9YtpUmjRrJ7z15ZsXqNrFm7XlhAsLZ/S3h+BBBAAAEEEECgZgkQgK5Z74vZIoAAAggggAACCNQCgZycI7J4yTLrpxY8Lo+IAAIIIIAAAgggEMEC1ICO4JfLoyGAAAIIIIAAAggggAACCCCAAAIIIIAAAidTgAD0ydTn3ggggAACCCCAAAIIIIAAAggggAACCCCAQAQLEICO4JfLoyGAAAIIIIAAAggggAACCCCAAAIIIIAAAidTgAD0ydTn3ggggAACCCCAAAIIIIAAAggggAACCCCAQAQLEICO4JfLoyGAAAIIIIAAAggggAACCCCAAAIIIIAAAidTgAD0ydTn3ggggAACCCCAAAIIIIAAAggggAACCCCAQAQLEICO4JfLoyGAAAIIIIAAAggggAACCCCAAAIIIIAAAidTgAD0ydTn3ggggAACCCCAAAIIIIAAAggggAACCCCAQAQLEICO4JfLoyGAAAIIIIAAAggggAACCCCAAAIIIIAAAidTgAD0ydTn3ggggAACCCCAAAIIIIAAAggggAACCCCAQAQLEICO4JfLoyGAAAIIIIAAAggggAACCCCAAAIIIIAAAidTgAD0ydTn3ggggAACCCCAAAIIIIAAAggggAACCCCAQAQLEICO4JfLoyGAAAIIIIAAAgggEEkC0THRkfQ4PAsCCCCAAAIIIFArBGJqxVPykAgggAACCCCAAAII1FKBmLhYadiyiTRo0UTyc/Nkz+Ydkrljr/h8vnKJRMfESJN2LSStUT2Jjo4y1++UvVt3SWF+Qbmud3XyiMTGxVmHCvLyXafK2kltkC7Dr71QokwAetGUObJh8aqyupf73NArRkvdpg1k1/ptMu+9aeW+rrIdPVEeadSqmaQ2qCvJ6SlyeN9B2b1ph/k8UPaQR91iE+LEcgt4deWx1PcYZd6ftsKCQvF5vWXf03E2OtZcG3X0WvPey/vdcQzBJgIIIIAAAgjUYgEC0LX45fPoCCCAAAIIIIAAApEroEHbISbAmpiSVOIhNYC4bdUmmf/+dPGWEYjsOrivdBnY2x+41IE6DywebsuK9TL/g5kVCmSmmcDriBsvtgb44YtvZM2CpcWDHefPtn26iAZBtXU8tUfIAtBpDetKbHycCQbXOc4MTuy0x+OR7mf2lw79u/mfwzmivo9d67fK15OnSZEJDgc2p1vgOd3X648czpH923fLoqlzJP9IXoluQ68YJfVbNLaOr124TL7/fF6JPsEOREVHy7i7fyKeowHoL1943/oFRrC+HEMAAQQQQAABBIIJUIIjmArHEEAAAQQQQAABBBCowQItOreRETdc5Ao+O7NWNSDaonNrGXXbeEmskxz0SfuNGiTdTADazprV651Zsxld2sp5t18WNKAadMCAg3ZAOeBw0N3NS9f6s243LlkdtE91H2zbp7Ncev8N1k964/ql3l4z0M+5fpx0Pr2XyyrwfTRplyFj7rjcyo4udbBSTuj7TEpNNu+0jYy+9TKpUz+9lJ7Fh1v36lTmeefJdn27+IPPzuNsI4AAAggggAAC5RUgA7q8UvRDAAEEEEAAAQQQQKAGCGgw8tQLhvmDhpqpvOyrRXJ4/0HRGsoNWzaVU8YOlfjkRBO0TJE+I06Xr9/90vVkzTu1Fg2watNyDZopvX3NZmtfs4bPuHSElTWsweuug/rIkhkLrHNV9cf+HXvk/X++bM0/Lye3qm5T6XFj42NLvfbMiedZ5Uu0g7eoSFbM/cHK/C7IK7DekZZH6XX2qaJB7LjEeDnnhnHy4b9eMaU2gpc42bFui6yev8R/P32HGV3bWu9V36+W6Rh48XD57Ol3/H0CN2JMNnnLbu1l87K1gadK7GvGOQ0BBBBAAAEEEDgRATKgT0SPaxFAAAEEEEAAAQQQCDOBxm2a+7OW1y1aId+8N90KPus0iwqLZKcp9fDxf1+36kHrsaYdWvqD1bqvrc/Io3U2zPbX73zhDz7ruYN7MuXTp97x14DuMKCb/356vqqa1pwOx+BzWc/bpncnqdukgdVFM55nvPKxLJ/zvT+4rBnluzduly+ee082LVlj9dNay10H9S112L1bdlnX6HX6s/HH1TL7jU/lg0deNqU3ioPzdeqnHfeddBnYq9R72CfqNWtoZVbb+3wigAACCCCAAAKVESADujJqXIMAAggggAACCCCAQJgKtDClMexW2mJ9Wvf5x+nfSpczepsCwmIFGbMPHLYu04CpXTdaF8jbtWGbPZz/UzN5v3ptijTIaGJdr5m33qLyL2rnH6icG5oZPGDMEOteS2ctkoO795e4smW3dqLlIpLS6liZxgd27ZOlMxdawXfNCNes360rN8gmU84jWNNSI216d5am7TMk3Sy4mJV52Dz71v9n7yzA6zjO7v+KJUu2bMvMzEwxQwyxA06cxIEG2iRNk6acfv2K/zRfuSlTsA00zInj2E4MMTMzsy3ZsiSLJYv+c+ZqVntRFyT5Sjrv81zv7uzM7Mxv96rN2feeUX7Th6Uor8BqgvPIDm/aMtkqGz5rvORn50p5abls/mSVEvodPs4DJ4+06uz4bL3yaE63jl130A4vA2Lj46TnyP46q9yXP7dre7xcOKlE7D5jBguy4Fu0by0ZarFIb4GFEJNaNFPzzPFWReABziABEiABEiABEiCBUAlQgA6VINuTAAmQAAmQAAmQAAmQQBgRgBBqokX7VpKVdskcOm1P7Dwk+LgG7DdMQDT1FhBTfQmq3toFU46FAjv07qqbQhB3FaAn3z1HkPltD4ir8ETevmSdtqiAKFuisqg9CtARIpPvvl7ZWChBvTJgL4LjvmOHqozvdy0ROqVjW2sspm6yEqzxQexasUkKsvMcFiWVC0BivMe2HzDVvW63KBG66+De2u86Wtl6eFpM0GtjdSI6psoKxJWRaVeUXyhxTeK1SA2BefOCleaU0xbM2/XopMsgUoMngwRIgARIgARIgASCIUABOhhqbEMCJEACJEACJEACJEACYUog7egZGVSZeTti9gS1cGCFtmmwL3rna+itu7bXp1E/JyNL2nTrIIOnjlYL2yULFtTLV2Jk6rGzuk9kGV/tGDpjrCU+6zErixDYhCA7GwvzwU4E4rOvMDYZJcVXJF1lfUOoRqYzspHh7zz9K3Pl03++pbtIP52qsqyTBF7YZvFBWGEU5ubr8yVFV/S2Y59ueot/LvnIRLYqqR34bBuvbXu5P/u4P92G9NFVIT7DssRTlClPb9iwtFeLHsI7euvC1eIp07r3mEEWN3hO41likAAJkAAJkAAJkEAwBChAB0ONbUiABEiABEiABEiABEggTAkg4zn9dJrO3oXwOuqGSUo8HK/8glO18Jh67LTkZXq3XYhLiNczQ/btiOsmaFsL+1STlPVEb3yU9/M6+EMfPmU/Xaf7EI77KKEUAVuQxc++q7OPzSCQAT3u1unm0OcWYvqyFz/S2ceoCEuO2Y/Ol0Rl6YGF/iC+Q9RFBjY+sOEYOWei7nP/2u2auf0CEKlN+LLCMHX83fYeNUCi1Nh0qPsLuxRkX8OvGYHFC1e9sVjve/oHIv3+1du1AA2/aczj6Lb9blVxHUTOpSzLQ9ytEgtIgARIgARIgARIwA8CFKD9gMQqJEACJEACJEACJEACJFCfCKx6/VOZdNdsKzM4MipK2vXspD/DZo4V2DDAfuPw5j1uNg8x8bF6qvBdhqcyAiJkmsp6RrTt0VFl/zrsJsbfNkPWvfO5yog+o8/V9T+Yk4n17y9zEp9RDs/nI1v2abHc1PO2XfHfTyzxGXXgab1v1TYZM3eqbtJaZVQHMk9YeOhQHtuXzqQ59m3/gm9nm1+37ZTePXPguNu9wYl4JTh782YuLiiU1W8uUYs1Frp253ScmZouBTn5OkO8z9jBbgI0rDdiK19EHFi3y6ktD0iABEiABEiABEggUAIUoAMlxvokQAIkQAIkQAIkQAIkEOYEkOW6+s3F2oe4zzWDpeugXspKwiEsY+jxiQl6AULYLCz9zwdOGdGRkQ67CmNbsXfVVjmwbmfVjJeL9J84XNt8oM6IORMse4qqSnWz17pLlV1IqrIe8RRYiBHZ2r6iWAnysKZwDbvFSNOU5gEJ0MjI1qFwYoFA12im+vNla1GhFoo87sGjG/1cKSq2ukMWM7KzEXFNEmTmQ/MELNa+85lVx9POoY27ZfiscTrDG5nkdq9w3F9EqWJyev9RadO1g6cuWEYCJEACJEACJEACfhGgAO0XJlYiARIgARIgARIgARIggfpHIP9yrmAhQXzilOjcXmUMd1FiNARFiMfRMdEy66u3yoK/vm55BhfmFVjZr7CacBKfKxEcWLtD2qlF/xw+y0kSFR2tRFZ3Abe2iTWvXPivSI3ZW8APGYK8EdQ91UM2sKcIdBFAex96EcCBPXVRs1YtdNa5/TwyrCEyO4W6J77Gibp7Vm6Vg+ttLwRUWYQSoVuqBSfHqYx0WHLAv3qYEpd3fr7BqXv7wfEdB2XYjGt02wGThsu6d5fq03g50apTW72vF6lUGdwMEiABEiABEiABEgiFQKV5WChdsC0JkAAJkAAJkAAJkAAJkEC4E0CW78ndR2S18gde9PTbyqahSA8Z4nFbJSabyM2o8oc2thvmnH17zub9DFuOqxEmE9hkAHsaAxYRrE7U9dQu1LLLF6sWaExu08KtO9hgvPe7F50+25ess+oVVd4fq8DHDoTsjHMXZZFaKNFkcncb1NtHC4dn9un9x3Wd9r26WFnU/cYPtdq5Ct3WCe6QAAmQAAmQAAmQQAAEKEAHAItVSYAESIAESIAESIAESKAhECjIzpONH62wpgLPXxPwezbhy0sYgrYJLNJ3NSLjfLq+LOxFvInQyNK+GpF9IdO6bM+RjgX9rAIvOxCCTWScvWh2/d6WKyH60rkLuj68vPFywVfsX7Ndn4ZAD5sSbLsP7avLLp254Ja17asvniMBEiABEiABEiABbwQoQHsjw3ISIAESIAESIAESIAESqIcEbv3fB+T2Hz8kMx64xefoK8o9eytcTrtktWvTzbv3r10szcvMttrU5U76qfPW5cxigVaB2omKjlI+yxPtRbWyHxHh/p9VsDIxXJq2TBa7yO9pEJFRkcoaxeFpDf9oX+K/p/amLNomOldUuFh8mEqV27ysHMlOdwjlvUcPkk79ultCvhGnXZrwkARIgARIgARIgAQCJuD+/5QC7oINSIAESIAESIAESIAESIAEwoVA+ulUncnaQnkCp1R6+XoaW/+Jw6xi+wJ0sNYw2c3wi05u3dKqZ3bgvdypXzd9iKzb9NNp5lSdbjHW0isl+pod+3QViNBYUA9ibvuenWX6AzdLk2aJtTIm+EqbaNXFc5b15k9WmSranxn3xFNEKS9uLB5osrjPHjzhqVq1Zcj2Nvc8V70UgM90dWE8vuOaxMuoGybp6rj/F06eq64pz5MACZAACZAACZCAXwR8/ybLry5YiQRIgARIgARIgARIgARIIFwIHNqwy8q2vfb+mwQLyR3feVCwKF5kVJQSKNvI0OnXCBbGQyDb9uTuw07D37JwtUy88zqJVIvbQcTdtmiNGM/njn27yajrJ+nF69Do2LYDQS1AiGzfwpw8p+vaD4oLi8SXBzXqlhRdkRX//USLt7CP6KoEc3zsgUxkLKyH8zUZuRmXre5gXwFbE2Rk56utCfgyXzyVqjObseDj9K/cLEc271UsT0rmuXRp1rqFtFNCea+R/SWhqUMoLym+Ils/XWO6cNtigcFcm01KhBLbE5s3lbbdOjp5eZ/ee8ytraeCswdOSOn1JVr8NgL4ka37PFVlGQmQAAmQAAmQAAkERYACdFDY2IgESIAESIAESIAESIAEwpMABM8D63dK//GODOfuw/oKPp4C2cNfvLrQLVM29dgZOXPguHTu30PbWHiyt0B/EK53Lt3gqetqy7DwoX3xQ9cGyDB+77f/cS12O4awvvadz+Sam6dJbHyc03n4GG/8cLnc8M27RCnQTudCPchKvaQX/EP2Mq47+sbJusuF/3hTCnPzre7XvLVExt4yTSDcQwTvc81g/bEq2HZKS0plzdufKUG/zFbqvNtKZbW3un2mc6HL0aGNu2X/Woe/s8spt0NwPq5eUvQZM0ifw/GRLRSg3UCxgARIgARIgARIIGgCFKCDRseGJEACJEACJEACJEACJBCeBPau3CoXT56XcfOuldiEeLdBQmSE7QbE0SuFxW7nUbDxwxWSeuS0FlYjVCa0PSBcH99xUHYt32Qvrna/TGVbBxPVWUkgU/rjP78qzdumqMzu5sqWo1QyU9OlSGU/6zDis5p3IFFu88mG1Yg9IBKvfP1TGTZjrGV7gfOumdbIMF///jLpNqSPDJ81zrLZsPeF+3Fq71HZvnidx2zy6rihfWFOvly+kCEndh2S8+q+uYaZS4XLPFDv4PpdlgCNtsbWxPSBOZgoK6naN2XckgAJkAAJkAAJkIAvAjWbBuDrSjxHAiRAAiRAAiRAAiTglcB9m1579PSB489s+ugLr3V4ggSCIRClFqVr2bG19kaGsIis4BybhUN1fUZERkjzNinSqnNbnZmbevSMU4Zvde1r8zx8i5ERjMhKy5ACD5YeycrmYtbDt+k62xav1cK5PqjBf7DYYYzKgi5TGcyw0PAVsLlorbyak5WPdpHyWs48nx7Q/fDVN8+RQH0hMOPBWy60bpfynZfG3Pt2fRkzx0kCJEACJBA8AWZAB8+OLUmABEiABEiABEiABEgg7AmUlZYqb+JU/QlmsBUqCxjZ0vaFCoPppzbawP5ifKUdRUlxiSx6+i2njG54P0+8c7Z16WAX97M68LKDbOgyk23tpY4pxksAWJzgwyABEiABEiABEiCBxkCAAnRjuMucIwmQAAmQAAmQAAmQAAk0QAJY+O/CyXN6Ab6YuBiZ+917VXZ2gc4ohviMLGNjiXHhxDkncboB4uCUSIAESIAESIAESCAsCTibuYXlEDkoEiABEiABEiABEiABEiABEvBMYP27S+XcoZP6JMTmJs0SpV2PTtoP2ojPR7fuk9VvLvbcAUtJgARIgARIgARIgARqlQAzoGsVLzsnARIgARIgARIgARIgARKoTQKlyncZi/y17NBa2nbvpLyuUwTZz3lZOXLp7AW5eOq85GXm1OYQ2DcJkAAJkAAJkAAJkIAPAhSgfcDhKRIgARIgARIgARIgARIggfpBAIv54cMgARIgARIgARIgARIILwK04Aiv+8HRkAAJkAAJkAAJkAAJkAAJkAAJkAAJkAAJkAAJkECDIUABusHcSk6EBEiABEiABEiABEiABEiABEiABEiABEiABEiABMKLAAXo8LofHA0JkAAJkAAJkAAJkAAJkAAJkAAJkAAJkAAJkAAJNBgCFKAbzK3kREiABEiABEiABEiABEiABEiABEiABEiABEiABEggvAhQgA6v+8HRkAAJkAAJkAAJkAAJkAAJkAAJkAAJkAAJkAAJkECDIUABusHcSk6EBEiABEiABEiABEiABEiABEiABEiABEiABEiABMKLAAXo8LofHA0JkAAJkAAJkAAJkAAJkAAJkAAJkAAJkAAJkAAJNBgCFKAbzK3kREiABEiABEiABEiABEiABEiABEiABEiABEiABEggvAhQgA6v+8HRkAAJkAAJkAAJkAAJkAAJkAAJkAAJkAAJkAAJkECDIRDdYGbCiZAACZAACZAACZAACZCAK4EIkWYpzaVN1w6S2KKpZJ5Pl/RTqVKUX+haUx9Hx0RLRJT/ORolRVc89lNXhdGxMRIRqSapouxKqZSXl/u8dGRUlETFROk6JcVq7BWeq8fExUqbbh00u9gm8ZrbpdOpUphX4LlBZWmkYoe2UdFRUnKlxLluRYWUFLuUOdfgEQlUS6B1l/YyYf5M/eyu+O8Cybl0udo29gpT7rleWrRrJReOn5MNHy63n+I+CZAACZAACZBALRGgAF1LYNktCZAACZAACZAACZDAVSQQESHDZoyVXqMGSITadw0Ix+veWyrpSlS1x4wHb5GmSrD2N7YtWiPHdx7yt3qN1xt1wyTp3L+H7vfiyfOy6o1FXq8BcfjGb90tcUpQhvC8/JWPtbBsbxCXmCAjZ0+QDn26euRWVlomhzfvkb0rt9qbWft9xw6RQVNGWceuO+Vl5ZJ/OUdSj56R3Ss2S4USpRkkEAiBJs0S9UsOtImKDvw/Z5NbtdDtE5s3DeSyrEsCJEACJEACJBACgcD/FzuEi7EpCZAACZAACZAACZAACdQ2AWQxT7nnBmnZobXTpSB2GjE6Jj5Wpt57g+z5Yosc3LDLqheoHBqlrnU1Y+vC1dKuR2clqMXojOXOA3rImf3HPQ5p+HXjHeKzOnt631E38bmZEuamf2WuIKvaHnZuyGzuP36YtO3WUVa/uVhlNAeWAQ4RHAI/Psiw/uLVhVLqmiltv3iY79/8+H0SGx+nWF6U5S8vCPPRcngkQAIkQAIkQAIkcHUIXN3/x3x15syrkgAJkAAJkAAJkAAJNGAC/cYPtcRn2EBApE07flYLnU2Sk6TPmMHSe/RATWDwtNFycvdhy5Jj/btLtVWHdzwR0n/CMGnVqa3OIg705//e+w3uTGlJqWz4YJlMvnuO7mD0jVO0tcCVomKnDlM6tpUew/rpsuKCItny6Wqn87DNmPHAzcqew/GfB7Ao2bl0g5w7dFKQtYzyLgN6yuBpo5SInaD5Tlf1lzz7rlM/9oPtS9ZJbka2o0glocP2ANnazdul6BcBzdum6GxpXKe+R3SMs2hf3+fD8ZMACZAACZAACZBATRKgAF2TNNkXCZAACZAACZAACZDAVSfQuX9PPYaK8gpZ+Pc3nDJsC7LztLCam3FZRiirCUTv0YNkz8otej83M1vw8RYQYifcrvxnVeRn58qFE+e8Va22HJmzyAguKy0NyRsZYzi196h0HdRLey+PnXetzk42A4Dv88Q7ZjkOVRb42nc+06KyOY/tmJumWOIzxOslz77jNKYyJXSf2HVIZVcfk5kP3SpJLZtJ05bJ0qpzO7l0Js3elbUPexO7QA+LkEMbd0ty6xYy86u3ahG6fc/O+n5YjerRjsmmr0dD5lBJgARIgARIgARI4KoQoAB9VbDzoiRAAiRAAiRAAiRAArVBAIIuxFHE5QsZTuKz/XrHdxyUHsP7aS9Y2Ff4GwMnjdCiMerv+Hy9v8081rv2y3OVFUWypB07I2ve/sxjHX8Lt366Rtr36qztINp27yid+neXswdO6OYjZo+X2IQ4vX94yz436w3YkcDzGQHRfuWrnzqJz/pE5T/IuF795iK5/ht36ZKh08cEbD2RnZ6lMqMvCyw/fPnwtmzfWvqOG6IF65i4OLlSWCRZaZfk4PpdkqPaew2Vbd2pb3fpMaKfJDVvpu5XlMpwL9AvCyCAXyl0zg6399Ohd1fp2LebyvBuJchqzrmUJZfOXlDC+R4l2pdZVcfMnep4dirtSnAf9cJ4qgYWtzu6bb9Vt9odNd7O/XpIzxH9HTyUZ3lBTp6cUN7ip/YccfPJTmiaqF6ejNfd7lq+WW+7D+kjbXt0lHjl4Y0XKOcOnvQ5Bn/n6TT2EMcZpb6b+M6169FJd5tx7qLsXLbRuh94GdF7zCBtz1KiXoKkn06TY9v3O73EcBpP5QGeX/iOt1YvQ5CdX5iXL2cPnpDDm/Y63TNPbT2VwVe6jxoHvhMJym8aL1+QyQ+rHm8vWzz1wzISIAESIAESIIEqAhSgq1hwjwRIgARIgARIgARIoJ4TgF0EPhCiIW4iS9XTQncoW/qfDwOaLbyRIUwh8rIcC+kF1EEtVoY4uu6dz2Xa/Tfpq4y5aaoWXJNbt5TuQ/vqsvzLubJLCX6uAfHSxOEteyU7PdMcetzmqyxyeGfDUgPCbsChhEyzeJy39rA5cV3MMD4pQZqp7Omug3vL5k9WaXHW7drqfk9T3t7IzLZHQtMmery4f5+/8IFblntkZKRcc/M0Ldzb28GypR2EUWXZsuqNxZJ90cGmi/LajlBtTGAfoi4Cc/NbgFbjnar8ylt3cR4vFtqDzUuvkQOUwP+x0zMMkdlcKyv1krKEGW69FMH1IVC36dpBuiuxVz/jtoUeA50n+tMR4jgvX8hU3uFDnZglqQz6TsqS5bPn3tPWLp2VxYsVav7JbVrqlwjLX/pYv0yyztl2WrRvpX3L7fcCzwmeTbBb9PQ7AYnQ+FXCnMfu0C9ybJeRpBbN9AueA+t3el2A016f+yRAAiRAAiRAAs4Eqv5fk3M5j0iABEiABEiABEiABEigXhLIvpihx42sX/gUQ0SsiRg8dZRDQKtA9nP4+RYjUxeZ3QgsFjjhtplWVi4ym9e8tcQjhva9u1jlF/20FEE26IYPlwfOQYnPXQf2sjKfT+87Zl3b7EBgNuIzXhTAxgP1IPqbgGUIFjF0jbFKRDbiM15EgAmyYYsLCnVVZENPf+AWnSlsbwsvcGSNI4qV/zXmh8/lSsEZmbUT5lfamKg6x7YfcGQnl5frNlhIEdnK+Jh7oE9U88/IORMs8RlWLLBTSTvm8CtHUwisyLb2FgMnj9TiMzLKYZGSevSMJVY3VwJuP5VBbo9A52nahjrOAROHi3obJMh6xr009wPP6ayHbxWIz+WKJWxasEAmPMgREMwt+xgzGNt25JyJ+juJxTDPHT6l7zf6QUCI99XW1o3exUur6V+52RKf8Rzg2ck8n67HhkpYgBNWNwwSIAESIAESIIHACDADOjBerE0CJEACJEACJEACJBDmBHYu2yRTVRYssp+RCXmDsouAX3PqkTN6McKLp87rn9UHMg0s0geLBETe5RxtmxFI+7qqu+Oz9So7tovEJzWR1l3bW5fdu2qrW9avOZmg6prIOHfB7Ia8HTBxhLbaQEcQfpHVm6Kyeo3tBoS93SscFhLmYhANIS4jIJp//sL7TnYbmBMyhhGT7potHz71siUOdlOZ3J1VZjICAuaif72l/LWrbDMGqRcIEBBhuYJFGz//9we6Lv4x7WDPsfAfb1p9ItMbntpYPBHjx8sM+IibFxAQ75E1izJkZQcSnfopm5DKhSGRnY4FHY14CuH1uq/dru1kugzsqexUjmuB1VP/2xavdRK9W6hFHmc8OE9XxTMLyxITgc4T7WpqnOve/VwL5OgTYu91D9+m5pcs+GVBubpPS1/8SFuemPOzH50viclNtZAMHoYNztsDgvamj7+wiuDTfsM375K4hHiBHQ1+AQBxvroYdf0ky77n5O4jsmVh1f3EMzv7kdv1czxaPZ/nj5xWNjVXquuS50mABEiABEiABCoJMAOajwIJkAAJkAAJkAAJkECDIgCf1lWvL3L66T2ErF6jBuiMyHn/82UtQLbr6fCi9WfyQ64dU5X9rETekEOJ41Vh368qDWYPIp2rnzT8hJHN6y1ilVCHQBZvSXGJWzVYIUDIdPuM7C/dhlbZd7g2hNg5QHlm49NP2S90UZmjRnxOV/dorbIMcY12ysfYxNZFq53EZ5Snn0qVA+t26ioQJVt2bGOqa+EdB8iaXvHKAifxGeV7V24VWFYgYOVhD9haICBAuwqdGz9aIR//5VVZ8NfXtNBsbxfKvhGDcb1lL33kdF2U2a03jEe36/WQUeyacZ2VlqG9slHXzMu0M8eBzLMmxpl2/KwlPmMsyE6HH7mJE7sPW+KzOX9i52FzWmeCWwe2HTzbmxestJWIfrm08r8LrTK8kPEnYLWCQKa9XXxGGV4QbPhwBXb1iy37yx1dyH9IgARIgARIgAR8EmAGtE88PEkCJEACJEACJEACJFAfCaSfTpUP/vCyzlztPXqQFrCQEY3AFpmR+EC8Qwapr4CVR/dhDh/lvKxsnUXtq77rOWTNYgyugYXrEBDCh84Y63K6Qvat3u51EUWXyk6HrpYjRnR0qmQ7MIIrMo49BSwH+o51tnKw1zuvrA88LewHIdieJQpvZFguILBg3Nzv3iM7l26QIzYhsnWXqqztU3vd7TnQ9sTOg8r3eBh2lddxe2thuBZq0UIEspEhGHqKU8reAbYWeAbgj238rrFwHxZFxAKWMx+aJ/vX7lCC6WktlIrC4ml+nvoPpAyLLCIwXoj/yAq2BxZ8LC4o0iKyqWs/j31k83sK2JYg+x+Z5/YIZp7m2qGMEwszugZsQ0x4sn4x9wZ1jGe4qW+2Z9RCm3jOXAOLVBbm5uvs6eYqI7y6AKe4Jo4XMbABcb0XaI+XHybwnOK5Z5AACZAACZAACfhHgAK0f5xYiwRIgARIgARIgARIoJ4RgKCKn+fjExEJO47Wymqhu3RRHsRGlO2hFmqDyAeLCm8xdPo1WrDE+e1BZD83a9XcWrzQ2zXM4ob280c27wtYgIZVCHyQ7QFxDQuyeVsYL19lfMKGIyY+1t7M2ocQWlHpq2sVKgHXCPpWmcuOts9QQqg9YI/QqW83GXXDZC3yDZs5TvIy1YKOx87oai07OERZCL5u16zsCIsgQnTE9Y3fM/ax0CAiOz2rsqb7JlNlDJto0629JUBv/XSNTLvvRp3l3rxtioy/bYauBiH71N6jcnTrPv2cmLahbvV4laUHAgvc3fbDB312CWHcU+S68DV1SoqKza7TNtB51tQ4jaez02BsB3arFFPs7YWIOY9tpvL49hYQ4WHpgk91kWLLpMffBHx8RatOVZn3vurxHAmQAAmQAAmQgIOA82t2UiEBEiABEiABEiABEiCBBkgAYlbm+YuyS/lDf/K3150E566De3mdMYRqLIqHQPYoFokLNCCmIsPV9WPvx/Ucjk1msr1edfvjbp0uEHkRJ9WCeCY7dOjMscrDOMljcwh1JjzV2b9mu7z3uxedPljszkSpB9sOc851W6bEbAi6S/9T5b/cU4njJuCnjCgtcbcCMXWwhWcwwmStYs4QSxFXCov01tM/xbZzcZW2G6gHK4vPnn9fzuw/7sh6rmwMyxAsoDf3u/cqv2ZHFrynfgMts4/Xn7YRym7EU3hI/vVUzSoLdJ41NU5rADW8Y7+frl2brHU8F54ymu314e8dSETHen5ZE0gfrEsCJEACJEACjYkAM6Ab093mXEmABEiABEiABEiABDQB+AjDWqJpSnMtzELg85RxC2sMI2zuWLI+KHpYbO/DP77i1nb2I/PV9ZP1goauvs1ulf0o6DKgp7YVQdU8JZZvUYvi5WZky2C1+B78kifdeZ18phb1c43si5lWEaw2dnxe/TxbdW6r20DkC0Yoh+gNkR0L0LVUlhgmLquxwBoFiyh6i2glNhuRHWwRRsBHf7DW8BYtVRa8iYwzztmzeMEAv2cE7Dw69umqM2GNyD1SLVKXoa5n52X6CnRrHy9sNOBZXlcRyDyv5jj94YFfNaSfTvNYtVnr5rocNjDwnPYVl2yZ1HhJdXjzHl/VeY4ESIAESIAESCBAAp5fpQfYCauTAAmQAAmQAAmQAAmQQDgQGKgWvLv9xw/pT6tODpHU27hMdjDOexKfYUthFmCDX+2Fk4FnP3u7dk2XQyQdfdMU3S3mZQTtg+t3ahEaJ7DwXr/xw9wuDYsSM/8eI/oJbDx8BewgTKay3cfXVxtP50xWb4kSok1gcUJEZESkJCrvbE9hbDdwDl7fJoz1hvHWNuX2bUqlcI4ycy37ebOflZqus+Sx8OChjbtNsXTq193ad9qpzL52KqvmwGSeN29TvUdxNV0FfdqfeYbDOL1NsFWXdh5P4aVRUguHx7o/zyisVoxI3dpLnx4vxEISIAESIAESIAG/CFCA9gsTK5EACZAACZAACZAACdQHAid3H9YZyxCg4C9sspddxw5fZiw6hzA/1XetM2xWVfvtQWY/u/ZZW8fjb59p2QzsWblF8pSvs4k1by+xrDgGTRmpF9oz57C9ovyCD25wiKzIlJ7+lbmChRc9BcTdWQ/dap2yLyBoFfqxM0C9KDALEqZV+j+jWbpZVE+5aUy6a7ZbT8h8HjN3qlWOBeNMZFRmsSILeri6d66R3Kal9BzRXxeXFF3RWdM4aKEWqZv11Vv1B1nxrgFbDhNuz1OlB0ZSi6amit/bjHOODGyw9vRiALYR8KW+/rE7ZUzlywW/O/dQMdh51vU4PQzdaxGy1Nt07eB2fuT1E63nK+OcI0verZJLARYYRbTv3UU8vbyCV/ecR+er+3GH9Ry5dMFDEiABEiABEiABLwRoweEFDItJgARIgARIgARIgATqHwEsUIeMR1hrtFDWDjd++0uyc+kGuaTsFgrz8rXojCxW+Pqa8LQ4XxOVfWuyXZEBCpuEcI3uypvYCGY5agG+Q5Vishkvsjv3rtqmrTggoE6+a44sevptc1pv96/dLr1HD9SWGGB3wzfvlr0rt2rPawhzyDpu37OzwK/ZCMdgcuZAlTjr1KE66KDEwaYtHVmoOAdhGCJep/7dLfEf5fCENpFx9qJkpV7S9w7juPb+m7QdAmwW2qnr9x8/1PJ9PnvwpNgXt0OmMhaPw3V6jRqoxhmtfLAPS0FOvnTu30PfcyMg716xyVxSZ4gjOxznkEWOzOxzh04KrBuwIOGE+TOtusd3HLT2sVOYV6DE+ngl/kdpkfiIWqywMLdAilR5dXFw/S7pPrSvHi9sUhKTEzWL7ItZkqIWuRtx3QSBBzXi5O4j1XVX7XnYsQQzz7oeZ7UTcakw6a7rZP/aHXqxUfxqAd+HbkP66FpY3PDghl0uLTwfwnoDLz3wHEy553rZv2aHnD14QpCh30GJ0sOUHY+xfoGXNoMESIAESIAESMB/AhSg/WfFmiRAAiRAAiRAAiRAAvWAwNp3P1eZozcJFhDEZ+wt13od9Ymdh2Tf6m1u54fPGm+V7fhsvbUfbjsQ3EbOnqCHZbfecB0nrDi6qcUUkcEMUXPw1NGCTGkTsB+AP/TUe27Q5+GzPEwtXOgtkGG94YPl3k7rclzDV2C8Gz9coRaHrMpQRdkXr34i8MfGS4AUZaMyzoOVSqayyNjwwTKn7iFGL3vpI5XJfJvOBocQiY9rwP/7uLrvJkrVwoiHN+0R+F9rEfrGyTJafTAWI1ijLqxKCnLyTDO9RXZ08pSWeh+LVeKDhSpXv7nYqZ6nA4x3xSsLZOZD87To3WN4fyWgOzK07fWxeOaBdTvsRUHtBzvPuh5nIJODLU7bbh1l0JRR+uPadvUbi/x6GYB2acfPytZPV8uoGybrFwqD1EsBfFwDz8/lCxmuxTwmARIgARIgARLwQYAWHD7g8BQJkAAJkAAJkAAJkED9I5CXmSOf/vMtncVqvI1dZ4HF1bYtWiNb1cc14KeMjEcEMoprK/sZC51BAD+x67DrEPw+vkaJ68ZLeefSjW4Cqb0juxVHP5VJDIHXHgUqe3zxM+8IsnyNH679PPbLy8pURvlGXc+TdYm3dqYfZKTmXMrSmb6fv/CBzjA158wWdZa++KH23IYIbA8cIyt15asL7cXWPrJ8IWB78v2FAIuXDXtXbbXqm53dKzbL+veWKjuWIlNkic/FqmzLwlWy6eMvrHNmB9nByLBFtnQwAd/qla99qheNdG3vGO92WfHfhZaFCurgHlQX3u5DsPOsjXH6M4/q5rlnxRbZq+6p6/cctjJ4QWJfXLC6vnAe38VtS9ZpWxrX+oW5+bpPT8+Pa10ekwAJkAAJkAAJOBNQ7moMEiABEiABEiABEiCBq03gvk2vPXr6wPFnNn3kLnJd7bHV9+sj47e1spCIV9nCly9mSsbZNCUYVi18V9/nV1vjB682XdtroTovM1vgpQsRri4Ddh9JysYDmey4Nl4ulJeX+zUELKYIG48YZcmRq2xEILD7E/DBTm7bUrCFHYi/18OLC7wMKC4ochNE/bluTHysNFPjhXVIdnqm7sefdsHWCXaedT1Ov+an/qu2qVp0EC9VctWz6u+99tV3QtNE/YsBvKyCAF+mXmAwao7AjAdvudC6Xcp3Xhpzr7MfUM1dgj2RAAmQAAmEEQFacITRzeBQSIAESIAESIAESIAEap4APJDxYQRGAD7GsJ24moFs6Gz10sCxPFxgI0FWMuwrAg0IzhCeAw0Iz6EEFkasS2/hYOdZ1+P0i6lKlIfwjE9NBV541PULl5oaO/shARIgARIggXAjQAuOcLsjHA8JkAAJkAAJkAAJkAAJkAAJkAAJkAAJkAAJkAAJNBACFKAbyI3kNEiABEiABEiABEiABEiABEiABEiABEiABEiABEgg3AhQgA63O8LxkAAJkAAJkAAJkAAJkAAJkAAJkAAJkAAJkAAJkEADIUABuoHcSE6DBEiABEiABEiABEiABEiABEiABEiABEiABEiABMKNAAXocLsjHA8JkAAJkAAJkAAJkAAJkAAJkAAJkAAJkAAJkAAJNBACFKAbyI3kNEiABEiABEiABEiABEiABEiABEiABEiABEiABEgg3AhQgA63O8LxkAAJkAAJkAAJkAAJkAAJkAAJkAAJkAAJkAAJkEADIUABuoHcSE6DBEiABEiABEiABEiABEiABEiABEiABOgewn0AAEAASURBVEiABEiABMKNAAXocLsjHA8JkAAJkAAJkAAJkAAJkAAJkAAJkAAJkAAJkAAJNBACFKAbyI3kNEiABEiABEiABEiABEiABEiABEiABEiABEiABEgg3AhQgA63O8LxkAAJkAAJkAAJkAAJkAAJkAAJkAAJkAAJkAAJkEADIUABuoHcSE6DBEiABEiABEiABEiABEiABEiABEiABEiABEiABMKNAAXocLsjHA8JkAAJkAAJkAAJkECtEYiKia61vtkxCYQjAT7z4XhXOCYSIAESIAESaFwE+P/AG9f95mxJgARIgARIgARIoNESGDN3qnQd1EvyL+fK4mfekYqKikbLoiFPfMCkEdJnzCC5UlQsi/71dr2farDziYiMlDlfny+JyU3l9L6jsunjlXXGIiYuVvB9S+nQWmLi49R1K2TBX1+TkuKSOhsDL0QCJEACJEACJBA+BJgBHT73giMhARIgARIgARIgARKoRQJdBvbUvSc2byqtOrerxSux66tJoFmr5gIBNKFp4tUcRo1dO9j5tO7cVovPGEjnAY5nv8YG5aMjZFxf97XbpEPvLhKXmCCRUZHqEyUSEeGjFU+RAAmQAAmQAAk0ZALMgG7Id5dzIwESIAESIAESIIEGQGDC/FlazCovK5P3f/9S0DO6eOq8tO3WUWVhXpGMsxeC7ocNSaA+ELiknnE86xDj00+l1tmQ23Rpb4n/+LXBgXU75fKFDCm9StnPcU0SZO5379Hz37d6u+xfu73OWPBCJEACJEACJEACDgIUoPkkkAAJkAAJkAAJkAAJhDUBkzgJS4FQYvUbi6VJcpIU5uTTfiMUkGxbLwiUl5XLx39+VRKaJUpBdl6djblVl6pfF2z8aIVknk+vs2tXd6GY+JjqqvA8CZAACZAACZBALRCgAF0LUNklCZAACZAACZAACZBAeBKoSyEuPAlwVI2JAHzO6/qZx0seBK4dTuJzY7rvnCsJkAAJkAAJhBsBCtDhdkc4HhIgARIgARIgARKoZwT6jhsirTq1lazUDDm4YZf0HNlfWWZ0FXjXYvGzXcs2WTOKio6S3mMGS8c+XfXP9MtKSyU3I1u12ymXzjjbYjj6bScpqm9EhEqFnjB/pt4vzC2Q7UvW6X14/Y6YPR7rnMmuFZslNj5Wug/tK226dRBcb+l/PpTigiLpPXqgLstJvyx7Vm7Rbe3/BDK2oTPGSlKLpiqbWo3jM8c47H1hH17Tw2aO1eM6smWfwALEHp36d5deIwboeqLscQtUZvaJnQfl5O4j9mp63985ujW0Fdjv04H1O6THsH7SsW83dZ9aqIUZc+Totv1yZv9x3SImNkYGTR0lrZWdQmxCnLJQyJTUo6fl2PYDth6ddwPhh5btenaWniP6SWlJqWxesFLadu8k3Qb31vf7SmGRnD98Stkl7HBkq6t730fdv079uqss9qaSn50rF0+ck8Ob92qbCeeROB81TWkug6aMlOZtU7Qfcf7lPDm6Za+cPXTSuaLLUYt2raT/hGGS3LqFRMXESFFegaQeOyOHNu6W0ivui+mNuWmKtrs4ueeIFl7xPWiv5hivfJDxrJ5T8zEBf2Swb9mhjUQrz+ScS1kCywz0jcxlX4G2nfr3UD7mbaWk6Iq21zilvmdZqZfcmo2cM1Ff/8yBE/q7aCrYnwVYUqR0bCP9Jw7X31nUycvM0dYZ6af9t+4YOBmMW6q+3L+vRXmFsm3xWnN5vY1PaiID1YKRuDZsMq4UFWlu+9dsV/fXe8Y2/h50G9JbcH/iE5tIXlaOZJy7IKf3HpPs9EzrGk1U5vfw68arvwFV/8nbdVBv9b1tpr+TeHbM/IZOv0aSWjaTtGNnPT7jMXExelFF/I2xtwv0exnsnEN5Xiwg3CEBEiABEiCBq0ig6n+Nr+IgeGkSIAESIAESIAESIIH6S6CzEsMgBqUoMa1Vl7baZ9nMJrl1S7OrhOE4mfPYHXprFaodCELte3XWgtfeVVutU6Zfq0DtQNhGQAA0AjQEPlMOMar3mEFarNYV1T9GgOrQp5u06dpei6quAnSgY0tQ4pm+phKkDm3arQTcXHM5a9tr1EBrXKhjj6n33qjGUWVVgHNNmiVpIb/XyIGy7KWP7NW1iOjPHJ0auRwYnrhPzVo3FxybSGjaRC/M2KZrBzm6dZ/MeHCeFmurzifqe9RGeWhv+GCZKba2gfJDQwiPZk6XTqeplwgTrP4gHkIwxsKRS1/8SK57+DZtn2Iq6PFCiBzaRz57/n2PgjDq4lkYpl4W2AOcwR5iOywiPEW/cUNl8LTRTqdwzRbtW0kf1eeip99RLzUKnc5jrLCJiYiMkFHXT1TCfbx13mQFR6rzY26e6sQelXAegjxekqxSVjHZF6uEVKsTteM2rmTRnHqpdmveWiIXlChvj66DeinxPFrfS7wMMmGeBQjgEFf7XDPYnNLbRCXyt+3eUfYpMRiCsD+B+Wtx11bZ3F9kQ9sFaNz7afff5PQ9jU9K0C9Dug3pI6teX+T2wsbBdZISn/vYriDqRVYTfT/7jh2ins3lcq7yxQLEaXN90yCuSbxVdlH5YhsBuovihL8jCUmJHgXoaPUCwvSVfibNaufv3x5cP5g5h/q8mHlzSwIkQAIkQAJXm0BoRnpXe/S8PgmQAAmQAAmQAAmQQNgQiFMCDhb5g9gEIfjC8bMqqzlNjw9CyvSvzLXEZ2Qknz14Qmc8VpQ7Mj6RbQrBzAQygU/tPaqzl03ZKZVdig8yrT0FhDRkSkOgxrWRVV1a4p6tam8bzNgOb9rj6EJlLvccOcDenbUPQQ5xpbDYKbsbWalGfC4rLdOiYZpiZbJqIXJec/M0qx/XnWDmaO8D9wkC5JWiYi3C2heo6zG8n7pPN2vBEtYN4I/zuKeITv266Sxke3/B8LO3xz7EZywyiQxjZD6DCyKpZbLc9O0vaYEWfCAunj9y2joPMXnknCrhWjeq/AfjMuJzYW6+nmtuxmWrSucBPbTgaxVU7iAz2YjPmHdW2iX9rBblOwTnaJUdPv2BuU4Cvb2P9r26aPEZmcxYfA+iMDKKEejXCP/4DuA5xscIzsgENln+9j6xj/mgPTilK8H+xM5D6tcDjvngmZ9012ydSezaztcxBFQjPuM7e3rfMZ1dbtogQxkvh/yJ4zsO6uelUGWKm3B8X48KzplIVGL7lHtusMRnvLzBy4C8zGxTRSbdPVtnJFsFaqen+rWAEZ8NA4zXXA8Mxt86w/GLAlUf9/zUnqPq3p20usG1zN8QV7HeqhTkjq/vZbBzDuV5CXIabEYCJEACJEACtUKAGdC1gpWdkgAJkAAJkAAJkEDjJICs0M///aG2K7ATGHXDJC0mogwWBVs+WWWdhlXF7Efma0FvtLIxgMBYUnxFZ+Ki0sQ7ZikRrIsWQTfb2lkduOwgi/rAup0upd4PgxlbZmq6FsaRUdlFCZm7l1fZjOBK+Dk/xD0E5msCFhIQeREQw5Y8+66UVwrwEBive+R2nUUK8frsgeNOtg2mD2wDnaO9Lfaz07Nk6b8/sIRl/MR/wvxZuhoyZl2zgzspUXbcbTP0+XY9OmlBVh+of4LhZ9qaLWw4wAKiIQKC3exH79DPBARfWG4g07lM1dPn1TMz59H5OuPYWD7oEx7+2fTxSif7ieQ2LWXmQ/O0ADps5jjN2Pgk456Nu3W67gUiJ64JYdbEqOsnSfdhfdX4mirhe6JsWbjanHLawm5l9ZtLxLxcMScheiPwUmLh39+w7v2eL7bI2HnXanEafSMj2ozJtMUWIvxixQl2ICbwnYF1CQTYtt07aBHZnPN3+8Wrnzi9JIF1yMQ7r9PNx98+Uz78w8vVWoPAPgRxzS3T1Heip56bp+/r1Ptu0tY4qIsMdGP7gmN8P8Af34UpX7pePv3nWyjWMUBZhCDwcmLJc+868bEyw9ULoY7qlw6HN+/RwvTmT1ZqUR4vThDnDp90sgTShTX4j7fvZbBzDvV5qcGpsSsSIAESIAESCIlAZEit2ZgESIAESIAESIAESIAEDAGVJPvFq586iWPmFOwFEBDz7OIzyiDEbvxwOXa1iNZa2WQEG8h4DkR8xnWCHRsyUBH42T5EdHv0HNHfOjyivGZNGEEJojNsNoz4jPPYX/7Sx1WisPLJ9hTBzNG1H9g1mKxmnDOiP/ZRDk9me8Av2XgTt+jQ2n4qaH72TiDAGvEZ5fAAvniyyk5i++J1lvisz6tnJrPS89jYW9j7M/uwnbBbT6Ac2cY7P99gqjhZxrRV4jqEXMT695Y5ic8o27pojSV8entO4WsOCwlX8RntzUsJeFzb7z3OQYz9+C+v6o8n8Rl1IHjbxWeU2ecCz+5AY9/qbU7iM9ojE/2IsmJBQAxu6XLP9Ykg/olR/uywV0EgK9ouPqMMv4o4ufswdrUlTWRU1X+uLn72HQefP79q3QNdUf0D/3LzPLft0dEU1+nW2/cylDmH+rzUKQBejARIgARIgAR8EKj6X3QflXiKBEiABEiABEiABEiABKojUKoXFKyyODD1I6OiVBaiww/34snzOqsVwpL9Az9WE8GIaKbt+SOnzK5f21DGdkQtZKdD6ZV2wRllnVUGKAIWCQU5VQuqtWzvEG8hMCKb1c4A+8gEhjUDwtTVB7Z/Ap2jranexTXsYq85j4XiEBA4XcVRU44tBEkTofAzfWBhNyxw6Br2zGM8N66Rc8n9WXOtc2zbAdcifXxil0PkxAEW8zNhf/bwTLreH8w349xFXR32H54i83y6p2JdlltpMwFrEWRhw+4D19ChOCAzGh9v4ck2Ar8WqLIsUQvsBRgQbz3FCZttRqvOzn7lnur7U2bnC6sVd76Ryrqn6sWDPbsdCy6CDQR+E3hZgEUyIWqbFwfImL8a4e17GcqcQ31ergYHXpMESIAESIAEPBGgBYcnKiwjARIgARIgARIgARIImIDxMHZtiMW3TMB+wlhQmDLXbatOVfVdz1V3XJhbZU1QXV2cD2Vs8ATOTs8ULLQIy4zdKzbrS2IhNixSiDhqE0AhkCVUZn+izm0/fFDX8fYPbDw8RaBzdO3D230y9YyYaY7NtgJKsUuEws/qSgn4Rvy2ylx2PAniKuXVpZb7IfybPQVETHCAWGkX+lM6VmV33/q/X/HU1CrD/YSdh/FvNid8zWXrp2tk2n03ausQLLI4vtLWRHsTK79tLABpXkCY/swWDCA2ewpHtnWUp1M+y8DBm+ANgR9ZxZinXQj22WE1J7EIqAlj8WGOPW0h3pqFAnE+RS082U8tNghBHJnFRnT21Lauy7x9L0OZcyjPS13Pn9cjARIgARIgAV8EqtIXfNXiORIgARIgARIgARIgARIIkoD5yb2/zaNjY/2tGnK9UMd2ZIvDpkDbcCjfXoTJhoZ4Z2w6UA5v5UAEswhbpjHah2OEyq825wT+3sR0XBeZ4AgImSbiEhyZ+ua4um28WjQwkED2NHylYT1hLE3QHhYu8Die+917pYfymK6rKC91LADq6XrgV1HuEPnNLxg81QukLL7yxYy/bWKbxFlVsUjptfffJB2UNQ2ynu3fJU92J1bDq7wTypzD7Xm5yih5eRIgARIggXpMgBnQ9fjmcegkQAIkQAIkQAIkUB8IXDp7wRrmrmWb9AJhVsFV3gl1bKf2HJWRsyfojNYeyvcZXsadVTY0ArYRdrsAZNyarFssUgef4PoeofKrzflHSIQWdpFd7CmMqIoFGU1kX8qSNmohQojTWHivNgK2CvB7RrRQliwdlaCKXwWY8YxUCx1mKBsP18zq2hiLziJWLzo8CbgQeY09SJZadLMmIvNcul5oEcn0i555W/u/+9MvxjloyihdFYtD7l25VdKUVUfe5RzLG/zmx++T2Pgqwdqffl3rREQ6/L9dy6Nig//P5mDnbMYQTs+LGRO3JEACJEACJBAoAWZAB0qM9UmABEiABEiABEiABAIiAAHQZHu27lIzXrIBDcBH5VDHBjEs7fhZfYWug3ppwdPYbxzevMftysa3uHmbFLdz9bEgVH61OmelJbbp2sHjJZq1am5l0GacdXg6o+Kl02m6frTKVocAW9sBYXfvqq2y4K+vyaGNu63LderX3dqv7Z0ULwsM2r2L7S8aQhkPXrzo8HFvPPVv57FvzXY5tGmPtr8pM1nscbF+ic8REZ7/89e8KDIvAVzHEMr3Ndg5u44Bx+HwvHgaF8tIgARIgARIoDoCnv8XuLpWPE8CJEACJEACJEACJEACARDIy8rWtdv37iKtlI+ra8ATec6j8+X6x+6wLCxMHWMDgJ/cwze3piOUsWEsh5UYhkhomiiDp47W+8igTTvmEKZ1QeU/Gecc2eAQN/uNH2Y/pfeRcQqP4Osfu1PG3DTF7Xw4FoTKrzbnNGzWOCuz2FwH1iYT77jOHMqlMw7RGQX2xTAn3D7LqmPfGa76xP2Z9dVb7cXV7rdol6LboB1eVrgGbDlM2O0lTFltbccpH+qoaGf/aHhjj75xsnXJ9Eph3ioIcgdZ3bD2QAydfo1HkR9+6uA75+t3aI9t1I2zWZ148qzuN34oqnkMe3a33ePbXvlyWoY+xHfYk2UG7FGCjWDnHK7PS7Ac2I4ESIAESKBxE6AA3bjvP2dPAiRAAiRAAiRAAnVCANYbCAhrU+65Xvor8bVpy2Qt9sB+AKJckjpObN5M4Htqj7ysHOsQoli7Hp10prFVGOJOKGPDpSFamsXhOg/ooUdzet8xj6M6uH6XtuHAycFTR8nIORPUgmptJUZlcLbr2UlmPzJfL7AGT+C8rFyPfYRbYaj8anM+yGSe+dA8bXERr6w1kEkLH2HwReQo+w0rQ1UdQ4y+fMEhRuK+TL33Rm2RgfsDuwwsnNdr1EDd3mTN6o78+Cc3I1uatW6hRdXR6uVCtyF99H1HU7xYmTB/ptXL8R0Hrf3a3gGXGQ/eohbS7CVxar/bkN4yUx1jzojUY2fUIpGBLe7pbcwQn02mN2w1Zn31Nu15jTHg+w9ReszcqZpvQtMmkqsWQkScPVAlzg+5doxe9BOe6rgnqN9vnHcB+kpRsSV6oz68pJEBb+aH/u1/c/B8tGjXStuPQATGCyEsNhlsBDvncH1eguXAdiRAAiRAAo2bQPBmVo2bG2dPAiRAAiRAAiRAAiQQAAHYVGz9dLWMumGyEnaiZJASX/FxjQPrdloCoDkHMbfPNYP1IYS6SXfN1iLuh398xVQJaRvK2MyFT+05ooVJfawSPA/b7BRMHWyL8gtlxSsLtCiKTNwew/vrj70O9jPPX5QD63a4FoflcU3wq42JIfPViHgj50xUYv9Ep8uUFJfIilc/cSqDWLjiv5/obHxkw8IyxpNtTEnRFVn33jKnttUdICse2fJ9xw7RL2LwMgUfXNOe8YznvSAnr7ruauQ8XpyAU7NWLeSam6e69YlxrH93qVt5KAXwSccvHvAyACIzPK9HunQIJuvUdcvV2BB4CZWVdkkLwxCOr7l5mlMLeKuDIURpT5GpXmqlqF9eoA68pPHZuXSjHNmyV1c/tv2A9B49UP+KAS8nIMjboyA7T5pULjJqL/d3P5g5h+Pz4u98WY8ESIAESIAEXAkwA9qVCI9JgARIgARIgARIgARqhcCJXYdl25J1goxE1yjMzZcNHyzXfriu5yA8bV6w0uuCZfBh9ifsP8V3rR/s2Ew/hzc7hCwc5+fkChYO8xZY9G7la59Knoc6EJ32rd6uRNCFVtYm+vF3jt6u6U95ebl/HD31FSo/T30a33BP5/wpg9HDiv8ukDO27FnTDl7ceBEAIdk14Cu87MWPnOw4TB0IoxCIFz/7TlBZwbtXbJb17y2VK4VFpktLfC5WZVsWrpJNH39hnfN3xwi15aXO9xDj9RVlqv7n//7Q41zxEmT5ywssEdhXP/Zz1hh8XHvjhysEWd6evpOZagFG8L9w4py9WzWWj/W9dG2DjPWlL35oLfjpeh6drH9/mYe2VWwgYC/9z0c6I96VGcax+s3F1ljsz2Ug38tg5lxbz4s1Ge6QAAmQAAmQQB0RUMs/MEiABEiABEiABEiABK42gXs3v/VA6pETT6979/P4qz2Wurg+skubpiTrTGYIsmYxsequHRMXo7IcY7RwGKgFQnV9m/PBjs20D2QLG4JmKc2VB2+0XlStuKBKmAykn3CqW5f8/J13pMo2T27bUqLVs5OdnqkEYPeXIJ76wn1JatlMYBGBFwb5KhO2psKMCdus1EsBC72hjAMZvrCZQEb+J397XXeFuTZXlhP4D0R8J42tTCjXqa4tMpITVTZ0osouLswt0C9uPAnIrv3AyiRG+VTjRUIg48T1YDMCX/nigkLXbh3HCgAWHYQfeyB/mzx35l4a7Jyv5vPiPovQS2Y8OC+reYe2D7426s6PQu+NPZAACZAACYQ7Ac+/UQr3UXN8JEACJEACJEACJNDQCESVpyY2TaxKyWto83OZDzKe8Qk0YJuAT21GsGMLZkzIwLX7zwbTR7i1qUt+/s4dGcIQeQMNvOTAInLe89kD7bGqfrBjquqhZvcw14yzjkUya7Zn770h2xjCvqdfA3hv5fDu9nXe2zlcr1o/a/VX2PiAe+snlPJg5xxuz0soDNA2IblJRUxJcWqo/bA9CZAACZBA/SBAC476cZ84ShIgARIgARIggQZOQP2fsvOxSQn8/2YN/D5zeiRAAiRAAiIxMbGJCVEx58mCBEiABEigcRDgf+Q0jvvMWZIACZAACZAACYQ5gcL86NSEpPiYMB8mh0cCJEACJEACIRGIVvYpUdFRcW3y2zEDOiSSbEwCJEAC9YcABej6c684UhIgARIgARIggQZM4N3Jd6QX5ReltuvZqQHPklMjARIIBwJYNPPQxt1yYN3OcBgOx9DICHTu262g9Erp1ienTSttZFPndEmABEig0RKgB3SjvfWcOAmQAAmQAAmQQLgRKL1StqhDn25fTjt2NjbcxsbxkAAJNBwCp/celdMNZzqcST0j0GlQj6yo2Ni369mwOVwSIAESIIEQCDADOgR4bEoCJEACJEACJEACNUkgpV3zFzv27VJWk32yLxIgARIgARIIJwJtOndIaRIZszCcxsSxkAAJkAAJ1C4BCtC1y5e9kwAJkAAJkAAJkIDfBP499K6NEWVypueI/n63YUUSIAESIAESqC8EhkwZda68rGzT86NuP1hfxsxxkgAJkAAJhE6AFhyhM2QPJKAJ5D7708llERE3RkhEx/Lyss4iFR0rIiKaEw8JkAAJkAAJBEKg7OjxmOL2LSqKp42MCKQd65IACZAACdQMgQqJkNzSUkkrLpFUfIqKZWl6lmzMzKmZCzTSXmLiYqXP2MEpUdGRdzZSBJw2CZAACTRaAhSgG+2t58RrgkDuv58cVFZW8p3ysvJ5OcXFkYeOHS9Py7jULCcnPyYvv0CKrlypicuwDxIgARIgARIgARIgARIggTokEB8bK0mJTaRJUpIMSGkp8wZ0k5iYWPlMCdEvnEyVQ3kFdTiahnGpgZNH5leUy8JXRt+7rmHMiLMgARIgARLwlwAFaH9JsR4J2AhUPPeT9tkVET9T4vNjG7bvzt196HDTtPRLthrcJQESIAESIAESIAESIAESaCgEVquJtG/dSnr06yOrJg2T/5xKlb8fOycXiplw4s897nfNkHM9RvSPLitr9rg/9VmHBEiABEigYRHgTzsb1v3kbOqAwOXnfjo/IkJe23XgcMGKDZubZ+fm1cFVeQkSIAESIAESIAESIAESIIFwIJDctKmMGjtaRvfpJd/YfUQWpmWEw7DCdgzte3bJnnjnrOSoqMjJL4+8e03YDpQDIwESIAESqDUCUbXWMzsmgQZI4PRfvvubmKioP72zaGns2q074otpsdEA7zKnRAIkQAIkQAIkQAIkQALeCeC/AU4cPyHnsrPlB+NGSqmqujkr13uDRnym84AeZRNunx5XIdEP/Xf03R83YhScOgmQAAk0agIUoBv17efkAyGQ+dzPnioqKn7klQ8+STx17nwgTVmXBEiABEiABEiABEiABEiggRG4nJEpR0+cktuGDZS2yi96TUZ2A5thaNMZcu2Ys8Omj4mNioi9/b9j7no7tN7YmgRIgARIoD4ToABdn+8ex15nBM7+5fHHK8rK/veFtz9olnmZ/8eyzsDzQiRAAiRAAiRAAiRAAiQQxgTyCwrlxImTcs/YkZJTXiG7c/LDeLR1M7R2PTrJ+NtmZLfv0TkrISF65osj715bN1fmVUiABEiABMKVAD2gw/XOcFxhQyDv2Seml0aULXv5/QXCzOewuS0cCAmQAAmQAAmQAAmQAAmEDYGOnTvJV+fdKLdv3idrG2EmdHRsjLTv1UW6De2d3bpLu7KYiKgnXhl777/C5gZxICRAAiRAAleVQPRVvTovTgL1gEBBWdGfP1uzvlyJz5H1YLgcIgmQAAmQAAmQAAmQAAmQQB0TOHfmrHy8co38fMxoue/MxTq+et1fLiY+VhKSEiWpZVJu87YpxSkd27UqKy3dHBkd++5Do9v9dVrENFhjM0iABEiABEhAE6AAzQeBBHwQUIsOfiMnt6Db9j0HKD774MRTJEACJEACJEACJEACJNDYCezcvU9Gjxld+tNrR59bldIioyHzKCsry4mOjzkWUR55XioiTiclRi58euA9aZjzaw154pwbCZAACZBAUAQoQAeFjY0aC4HY6OgnlnyxtlljmS/nSQIkQAIkQAIkQAIkQAIkEDyBjavX5M2LjYp8+MZvjwy+F7YkARIgARIggYZFgFmdDet+cjY1SODyc0/OKigoanLg2Ika7JVdkQAJkAAJkAAJkAAJkAAJNFQCew4fa15YfCXx8gtPzGyoc+S8SIAESIAESCBQAhSgAyXG+o2GQObl9IcPHDmW2GgmzImSAAmQAAmQAAmQAAmQAAmETODgiVPFpWWlt4TcETsgARIgARIggQZCgAJ0A7mRnEbNE2iSkHDtoZOnI2q+Z/ZIAiRAAiRAAiRAAiRAAiTQUAnsOXi4fUV5+byGOj/OiwRIgARIgAQCJUABOlBirN8oCFQ895P2sdGxzc6k6nU0GsWcOUkSIAESIAESIAESIAESIIHQCZw5nybRkRGt8l56ql3ovbEHEiABEiABEqj/BChA1/97yBnUAoHsyKgOeQX5pbXQNbskARIgARIgARIgARIgARJo4AQKi67kR1y53KGBT5PTIwESIAESIAG/CFCA9gsTKzU2AjEVFe3z8vMb27Q5XxIgARIgARIgARIgARIggRogUFBUeEVls7Svga7YBQmQAAmQAAnUewIUoOv9LeQEaoNASUREh+yCwtromn2SAAmQAAmQAAmQAAmQAAk0cAK5eXkSERHNDOgGfp85PRIgARIgAf8IRPtXjbVIoJERKJPIstLS+EY2a06XBEjAB4GmzZtbZ/NzcqS8vNw6romduPh4iVUfRGlJiRRepV9hREVFS5OmSXocFeUVkpeTrff5T2gEwuX+YhZDx4yTMVOvlYM7t8u6ZZ/V+LMcGikvrSMiBKsCt+vURVq2bq2fyzPHj9XI2Kt75pskNZWo6Cg9sCL1crrkSrE1yOraWhXrwY6vedaD4XOIJBB2BMrLKqS8ooQJX2F3ZzggEiABEiCBq0GAAvTVoM5rkgAJkEADJRAZGSldevaWPoOGSKfuPaSkuFhOHDkkh/fulvTU81JRUeH3zKOioyW5eQurfkF+nhQVXp1fJgwfN1Fu+8pXrbE8+9tfyNmTx63jmth5/FdPSWLTZrorsPrbkz+piW4D7uPuR78p/YYM0+3Kysrk5489FHAfbOBCQImn9vt7Ud3fv1+l+9t74GCZ/9AjeoBd1XcVz9yS9992GXB4Hbbt0FG+8t0fSFKzZJVNCBla9N+SF//ylJw4dCDkwTo986Wl8vNvVH3Xo2Ni5Cd/+od1jQ0rlsqnb79uHftqa1WqBzvVzbMeTIFDJAESIAESIAESIAESCGMCFKDD+OZwaCRAAiRQXwhAvHjwez+Uzj16WgKRGXtfJWbOvu1OlalYJqsWLZTln3xoTnnddu7eU7787e9LfJMmVp2ta1fJR6++ZB3X5U58QkKtXw6ZlOEQsbFx4TCMBjeGcLm/oydNdWLbf9gIdwG6MttYvy4K4KWRU8c1dNC+S1d59EdPSFSUIwO5hrp16iY2zvszj5dqvsJXW1/twu1cdfMMt/FyPCRAAiRAAiRAAiRAAvWLQHj81279YsbRkgAJkAAJ2Ai0bN1GHv3xE9Ik0WHbYDvltBsZGSXTbrxZuvXpJy//9Q9SVqaW5nENJXxdf/tdMm76LDch27Uqj0mABAInsGnVChkwfIRqGKGziPfv2ObcifoO/uzP/5KYSlH2pPoFw0sq0/hqxcQZs53EZ/yKIj0tVXKyMuVyxqWrNSxelwRIgARIgARIgARIgARIIAACFKADgMWqJEACJEACzgSQGfydJ3+j/FGr/ucEAlFxUZEWh2JiYyW5RUtBhrSJ7n36KsH6/8m/fvVzU6S3zVq0kIce/5GktGnrVM4DEiCBmiNw7MA+eev5p2WUyoQ+snePbFjxuVPnMLiA+GwyjhOaJDqdr9MDJYZ37NbduiT+tvzm8W9KYUG+VcYdEiABEiABEiABEiABEiCB8CdQpRiE/1g5QhIgARIggTAjMGf+l5zEZwjPrz39Nzdf1hk33ypT5txkZTW379xVuvToJaePH9UzGjJmrNz+wMOCLGkTEJuM36sp45YESCB0Anu3bRF86kM0s/nAX87MoPhcH24ax0gCJEACJEACJEACJEACLgR8G9u5VOYhCZAACZAACdgJ9Ojbz34oL/75927iMyos+/gDWb/cOdNy2NjxVtux02Y4ic9pZ8/If/70W+t8Te3A57SNWtAMvrfIzHYSuFW2JY4dZY6FzoK6ruoDvrDwsR46Zpx0V5YjCYkqi1SVBxMYT/OUFD1mjD0gr1bVFvWRRTp49DV6ETc9jiDH4s/4DUMHx8oWehxRaoHKXoJF8BKbNg2Mh2ofDNOaGovHfjzA8Leeh6ZVRWqu6AfWNngxg3uO4+qeH0/XRhl+UTBg2EinXxZ4qosB6OtUjcTaM/XtYzBl3tpYjSv7NfXt5b72df3KtqZeTlaWHqO+Jph4ClWO84Hy89RVKGUYAzzs+w4eqv8W6F+BeBuz64VqYA6aQavW+t73Ud+5gK7vOh77ceXY0D8+DBIgARIgARIgARIgARLwhwAzoP2hxDokQAIkQAIeCTRJUkJiZZSVlcm5UyfModt26UfvafHRCKiFBQVudZD1/MXCj2XFwo+kTfsObueDLYDYfPuDj0i33n2cRBOMebPyxF307pvyvV/+XlqktNKXuHj+nPzjFz8L+HIQ1affNE9aKOHHNQrycmXDF8vU/BaoU3p5N9cqTsewNbnxrntlxPhJlh0CKoDR0f375L2Xnpf83BynNtaBEoZm3XK7XDN1uhZu7UJRWWmp5FzOkrdeeFrOnfR+v6y+Atjp0be/PPC9/7VavPfi85J27ozc/cg3pVXbdlY5doqLCvW9Xrt0iVO560GwTGtqLBNnzZHrbr3DGtZfn/iRZFy8YB2bHYj8WCzPxILXX5Eta1aaw2q3uEd4MTL3S1/WAr39nuGe415/8ekC2bRyuVtfc+/5stgXF3zi6w/Kvd/4roABbHAQWZfS5U8//YHg1weP/fRJXYZ/Pn37ddmonsvJs2+Umbfcpsvt1+6gFgH8xTMv6vLS0hL5v29+TW657wEZOWGyLsM/S95/W9Z5uY+Y05ce/ZZVF/f7M1XfV1w//27tA4869rF07dXbGkt5ebkay8NqcdNy3VUo/HyNJdBzGMfoydP0MxMXH+/U/GLqOfnwlRflzIljTuXmoCbmkNS0mdxy/wPqb+0Qp78buEaR+pt77tRJef2Zv8mV4mJzWb+3EPW/+f9+aT1TaPjmc/8UNx9xv3tkRRIgARIgARIgARIggcZCgAJ0Y7nTnCcJkAAJ1AKB3OzLEhfvEBbhGQsRzpuoWVpSIn9/8ieeR6EENgi0L/75KS1Yeq4UXGlsbJwWTXQWsksXGPO4a2dqYTpaCb5G7GqS5HtBRZdu9OFdX/uGDBo52tMpXQaxHuL0wOGj5Lnf/UJKFA+voUSs7/zfb6WlByEbY+w9cJB8/9dPyT9/+YRkpl906iapWbI88qP/Z4npTifVAYRtCOQQS99+4RllxbDZtUrQx5GKp2GITjp27aaE/685lZnO4+ITZLZacLKLEhXfeOYfpthpGwrTmhqL/bnA4MwLFKeBojwi0mmeUdFVdjKudV2Pwey+bz0uyFT1FDiP+3rT3ffpFzOfvPmqUzXXMc67/0GdeetUqfIA47ffI2N7g/Hay+1tTXl0tMPLHYL1qIlTrCrDx03wKkBfM+Vaq18I6Xu2bLLaedtBtq65pmsdU4554AMBGmWh8HO9RtDHahyu4ry9rzbtO8rXfvgzj9+7mphDl5695UH1AsjuuW+/PjKye/YfID/43Z/V36BfyqULafbTPvebJierv6O/0C+0UBH3cv3yzyg++6TGkyRAAiRAAiRAAiRAAoYALTgMCW5JgARIgAQCJnDyyCGnNo8ocWX05KleRTqnyraDd1Wm7O9+8B0n8TlCCXo1EY/+5OcOCwxbZxBPkAlsMpGRFdo0ubmtRmC7s2+/0018xjWuFBdpocbeW7tOneX+b3/fXuS237pde5v4XKFFNvRnj9i4eHn4B+6C/q1ffshJfEa7kpIremFIex8QvO58+OtO2Yz2/mtif9z0WZaQiGvrbFWXecAeYtKs690uV9NMQxmL2+BquADZxE7is2KE7PzC/Hy35+eaqdcqa5exPkeArHkT4I5PqX7eTan7Nj01VT8neFbsgbYoM88QzqWdOyt5tux7CKtm0UJ7WzxjXXv1sYown/OnT1rH3nYuqF8gmGva69jHghdaOEbUND/7NQPZB4OqzHAHdzNG0w+Y3PHVRx12OKZQbUOdQ4ISlx98/IfO4rPig+dI/62rZIVLYmHJbz7xK4/3zDYka7dJYpJ8Wy02i785OlRfG1YslcXvvmXV4Q4JkAAJkAAJkAAJkAAJ+CLADGhfdHiOBEiABEjAJ4Gta1Zp4QSiCgLZlDff8xVlHXGfnFd2HEf27ZV9O7YKBCWlFnntC/YAtRFzVIat3coDYtBnH7wjW5U1QlFhocrQ7a5tLjr36GkJpYGOA4ukTZgx26nZzo3rZPF7bynbhFwt2iDL2tgboCJ8oeENe2jPLqd2rgewNFn0zhty6ugR5QPdSiZdd70go9QERPOxymZjY6UtQ3xCgs6ONuch4v39yZ9q+wWUIfMZ2bGwZkDgvg0eNUa2r1+rj2v6H/QP5isXLdA2D+ABD2hYi8Cb2AQWqVzz+SJzKLXBNNixWIOqrR3FaPrceVbvFRXl8s6/n5U9Wx2Z6fEJTfR9nzLnxso66p6NHiu7Nm+02njaAetlCz5QGapbtZDtLSvWtEUmPD7g9OS//m2Jk6lnTsvTv/65qebYqnu6W11/vHrBgEAmMqxStq1b4zhf+S8yco0FCIr2bvdv4UPYjGirETWWn//9OasPiNfP/Ob/nK6hBlwr/JwvEthRvvo1x6dvvaYXeoxRfvAj1QuB2bfdKcjKR4AXfhnw8l//4Oi4BuZw19e+KciEN4EFYd96/l/qb/AeXYTv3dwv3W/ZA6Eu/m6t/uxT08TjFt7r337y11q0dlRQ4rPKgMffJQYJkAAJkAAJkAAJkAAJ+EugZtLL/L0a65EACZAACTQoAmdPHpd3X3zOykQ0k0MmYOceveTam26Rb6lMuyf/8bzc89h3BH6ySvY01Wp922fQEKdrQDRZ+/liLT7jBATe55/6lZuNhVOjag4mz75Bi3amGvxd33vpBS0+owxZ0KsWfyKbV39hqujtlDk3OR27HsCn+dnf/kKLzzh3OeOSfPLGf2XnpvVOVfsPH2kdw6Zh79Yt2uYAVgcvPPUbS3xGJQj9L/3lKaf7NXCEd9sQq+MQdpYrEXT5gg8tHhDE/vWrn+uMbNMtbEFg12GitpgGMxYzplrbKjH32IF91j17/6V/W+IzrllUWCBLP35f338zBixw6Svw7PzxJ9+XLeqZgxCNzPNgPH99XWPV4oVOp0dOmOJ0jINx186wysyLCKugpnZqgV8oQ0O2Mb63u9X3D9yL1YsuLMCKX3nYw/684+VcKM8ABO3uffta3YP1y3/7gyU+4wS+d88/9WupqPTMRtkYlU3vK/C9/NbPf23L1q5QLwZWaN9wX+14jgRIgARIgARIgARIgARcCVSlSrie4TEJkAAJkAAJ+EEAmZAQNa6/40vKxiJZtXAXmJF92X/ocP25oH6+//Rvnqy0wPDjAiFUsS8GmJ2VqX827todxJr//Pn38oPf/sn1lF/HWGTNHi+qvjwFFqUbPnaClc3pJEB5aPDvP/zGSSg2VbCwH3ykTWZpm3YdzCnt6fr2v5+xjj3tqOlqURwezIjmLVM8VauRssuZGSr7+RO3viDKf/Tqi8oC5DHrXK8Bg/UCaSioDabBjsUaYC3u4IWFr8A3KuvSJZ0Fj3pxKtPdV+AlS8kVZysNX/WDOYcsX3gIm8UlXZ9nZFL37D/Q6hr8c7KyrOOa3KlpfqGMDcK8p1907Nm6SWUcz5JOlS8PkNlusvJxvVDmgO+L8fJGX/hlxZnj7gsdwrMffx969BuAaloc1zse/kG2NjyfzcKssCvavGqluPqPe2jKIhIgARIgARIgARIgARJwI0AB2g0JC0iABEiABAIlALsAfGB3Me3GW6TXgIHiEFjQk7Mg3bZjJ/neL34vf1MLEpZcKQ70Un7Xx0KCdtsBCN/eIluJY8hcRMZfoAF/VBPIMvUl/EEEN4IdrmUWUTPtzRa+rZk+bEmw8CA4IhKbNTXNnLYQtzp06Sb9lPDfrHlz7YONewKvWCM+OzWohYM0Zd/gLQ7t2e10qkvPXtZxbTANdizWoOpgB/esWYsWyhblGmnZuo2AA/x64xLipVO3Hn6NAC9Udm50zpL3q2GgldR1Nq9aoV88oSl+9dBvyDA5uHun7imlbTubbYPI9nWrA71CwPVrgl/AF3VqUKHEX8f8nYorD04fO2oJ0OpWSxf1K5FTx444VQ1mDuZvgeno1NHDZtdtu3fbFm0N4nbCpcBu9YNTOzdukAVvvOJSi4ckQAIkQAIkQAIkQAIk4B+BwP9L279+WYsESIAESKAREriYel7efuFpPXNH1vMI5RE9SXse28Xd5ikpMkct3LdAWUoEGoNuucupyd6PPC+EBXHHHmnnztgP3fYL8vOCWojQLnLD+sBXpCs+RoBGPWRoZ1y84NYk37bAm9tJVQBPbSM6IfMRWYpZyqIDAQFr/kOPysARoywfX33iKvxz+vhRr1dFFjQWxjO+tW07dLTq1gbTYMdiDaqWd8ZOmyGz5s1XnuFxIV8Jz3JdxCYlQM+Zf7fDgkY9d9coP3IjQBt/aIwDovi6ZZ/X6pBqkl+wA8WvC86fPuW1+ckjh2X8jOsqz0dIj/4DnAToYOeQ3ML+K4YKOXH4oNcxBHuibceq72ewfbAdCZAACZAACZAACZBA4yVAAbrx3nvOnARIgARqlUBpSYnKit6kP7CLePB7P1S+0FXetViIL5iIiouXHpOn66Yn1iz32sW5UyedzrVu197p2PUgwZbJ7HrO/2OlQPkIeMLaA2Kxp4Bg5yvsPq6oZ/rBIoTffvI3ehE/X+3r6pzrON2ua5tnhPKx9Ry+WfjN1IW927X8GotbqxopuO8b35W+Knu4vgV+NQCrB5O93rVXH8cU1HMNyx0TaWfPaNsXc1zT23Di5+u7W64WmLQHfgFhoibnEBFR1a/pP9Rt+85d9cuGxe++GWpXbE8CJEACJEACJEACJNAICVCAboQ3nVMmARIggbomAFuKF/7wa3lCLUZoMl5hMVCbAb9Tu61Gu46dvV4usWlTa1xeK3k5AaHd+DE3a97CSy1HMSxK7AErDU+R1LSZp2KrrF2nqrlA8DJ2HWOnzXQSn4uLCvWiYSePHNILLeZcvqyFwJ/95WmJV1YctR1devZWl1js8TJgZs90Rna4idpgGuxYzJjM1p7Jb8qwheVLMNFSZcHbxedyZb+yc9MGtWjcbrl4/rzgOUZG84OP/1B69O0fzCVqtc2azxfJPV//tr5GbFysspjoof2qmyY3t6674Yul1n5N74QTP7xPat+5i6R6sZ7p1rtSoNcQKuTo/n16L9Q5XM50/PrBwVZlVvftp14MeP/1gb/3YP/ObTJgWNUipxOmz1Jj3uu0uKG/fbEeCZAACZAACZAACZBA4yZAAbpx33/OngRIgASCJtBn0BC557HvWO2XvP+2bFju/Wf2yFSFIGwEaKthLe5g4bOUNm31FWB3MWT0WNm9ZaPbFe/75uNuZf4WFBUUWAJ0rMrOxvxgLeEpkm0L/sHn2TV717SByAkx25ulR4tWrUxVycvJtvYHjRxl7WPnn798wm1BNHj1VreInVMnIRzYhXLXbnoNGORUZPfCrQ2mwY7listifq3Voo/I6HWNDl27uRb5dTx68jSnep9/+K6sXbrEqUyluDtZtzifrN0jk13v7SoHdu0QvDBwvEyIUAvtXedkK4Pv/I71a701D7k8vPhFCP4uehOg7bZASLg3InGoc7hw9qwTR/t1nE6oAzyn8BPHfb2sbHuwYKGngGf+G8/8Q25/4GsybOx4RxXVBn/z//jjx9XfnRxPzVhGAiRAAiRAAiRAAiRAAh4J1Pxv9DxehoUkQAIkQAINjQCydyFmms+062/yOUWIHnHx8VadwoJ8a7+2do4fOuDU9fyHHtELvBlRLalZM7lXCSqdunV3qhfIwUGXRcfu//b3PTaffdudTv6+qWdOeaxnCh/+wU/MrtN27j1fVv1UcUxPTbXOt2hVlVWel53tJj6jIsQuM3+rYS3tQPS3ewGby0CsvO3LXzWHemuyQXFQG0yDHQuyx+0x0EXkN+fGuAjJpry6LTKGTTi8kj8zh9YWi0hWl11vVa7hnZatW/vuUSmpdhETLxawiKKJk2pBPF+2FKZesNtw44dFWO3Z32ZefQcPVVYl+EWAI4oKC6wXUKHO4eDuHaqvMtO19FHX8rRoZUJiojzyvz+TuV+6X266+z6Zd/+DVhvXHfO3872XX7B+YYE6eMH2yI+ecK3OYxIgARIgARIgARIgARLwSYACtE88PEkCJEACJOCNwKULaQKLBxNNkprKj576m/oJfpXPszmHxcke+dH/M4d6i8zJ2o4Fr7+is/zMdSC83vnw1+UXz7woP/3Lv+RHf/i79LN51Zp6gWxXL/nUSWCDTcINd95jZUXD5xXi5MRZc5y6Xfv5Yqdj1wMIprBdMN7VCU0SZfrcW3Vf9roQn0ykp54zu5KoxPVBI0dbx9gZNWmKHptTYS0fYJG6ceqn+8amBBmYX//xz50sQCCenT1xzBpJbTENZiznlZe4XUAdOHyUE1ftb67ukyfR0ZqQjx0j9KEKns+5X/qy0wsCeO9+7xe/99FDzZ6C4zYymk3ExSfICLWQqK+XFl98+rGpLnhOza8OFDhZvWShda42dsKNHwTax376pGWXgu//0DHj5EuPfstp+vYs6VDngF9SnDhc9aIE9+qB7/2vtkMxF22ufn3x1e//WOwWMhtWLDOnvW/VPXzud790+lUHFj298+HHvLfhGRIgARIgARIgARIgARJwIUALDhcgPCQBEiABEvCfwCdvvqp/om1aJCUny6NKaMbP7uFbCyEkUfkZu4pX8IRe8t7bplnA2+yzvrOHTYcQDv/1q5/LD373J6esYYwHQpmJrEvpeozNlbASaKDt1jUrdWaxaTvu2pmCDxjgOq7zP3vyuOzdtsVU97qFmP2d//utwK4DmeauAfuN9Tbbk33bt1ovAHDNu772DS0m5igfYWTQ1qX9iRkrxnHDHV+S65UQDX9juwBm6qz5bLGTyFtbTIMZC8YIK5TkFi31cA1XPMN4zmFngrJgY9u61TL9pnlW89GTp+oXBbBHwEsd+68GrEq1vAPf6bj4dtZVbr3/IZl334NSkJcnv/0fZyEVldKUXUNBXq4er9VI7Vy5UizHDuy3F9X4fjjyw8sIvDwyLy5cnw8Ixu+9+LzFoibm8Nbz6oWaegFovl94bh5VmcrGDsj1u49nd63y7/Yn8nNz5PV//VW+/J3/saoPHjVGDu/dLTs21J69inUx7pAACZAACZAACZAACdR7AsyArve3kBMgARIggatHYOfG9fLGs//QQpx9FBBBIMIkNUt2E+cgrP3hx993yp62t61uf9fbL8uK3/5Mf3a+9XJ11QVWH0//+klBJqtrQCDauWm9/P3/fup6KqDjj1Wm9bEDjgXF7A2bJCa5zT/j4gV5+a9/sFdz2790IdXJ/9mT+Iws1Rf/7JwZu+GLZYK29oDdBRY5MwJUrlqI0J65bq9b0/ubV61wEuGMOGa/zskjh2XpR+/Zi/R+TTMNZSz/+aN6CaAEO3sg8xkLORpx0W5DYa9X3X5OVpZsX7/GqRr6RAa8EZ/xAgKevHUS6jux6J033C6FMXldaFG12bXZ3Vv94O6dbv3UdEE48cN9sr9YAjPzfJh542/Oh/990en7XRNzKMzPl1f+/ie35xTfe/PdN2O4UlysFoX9jX6xZcqq2x5Riw+6/moDFh7mFxrVted5EiABEiABEiABEiCBxk2AAnTjvv+cPQmQAAmETGD/jm3yx5/8j6SnpVpio6dOkYmHDN0//fQHOlvSU53/396dgFdVXQscXyEjmRPmKYEwhDDPIMogVIqIgGitSFutr4o+W/v6+qptX2v9vldrqa0dXlVU0GrrU1GhzpZWEEVFQEDmeQokQJhDyJy8vU5yDic395Kb5AbuDf/9feGee4Z99v6dMK27srZ7n2ZPuptmnDa0abmQJ371kDz03Ttlnvlx8leeeVL+9NBP5ed3f9vKRNS+IyIine7tzEVnhx8bz5mg8jsmcOdeFNB9WUlxkXy29J/y+58/IEWF50uXuM+xtysrKuUPD/7E8vIci74/sHuXPGb6OZqbY19ivWpQWq/bbgJ/GgxzN71OMxY12K79283zPHt/IF410/vp3zxcowyK3a+O9WNThmT+b39l76r1GkjTxozlhMly13loZrDn89DyIR+++6Z8+M6bNcbvuXhhjYMebxY9v8AE4V8XDQx6trzDOfKk+d49a7JQ7VZZWWFvWq+lrpIZNQ748abU4/eZXqLfJy/N+7O12KLnfH11+eG7b3kcqqxl4nGCX2/dC3W6S4O4L26sn7uvRm2b32OaifzBW4trBYK1X83s/8sfH/WaNRyIOWgpj8d+dr/sNa/enltZWankHNhvFhH8oejvh/q2919faF1vX6flRW53ZUXb+3lFAAEEEEAAAQQQQMBToOE/M+rZE+8RaEYCp+b9/O5NO3c8+fr7ftRHbEbzZioIBEKgbYeO0qNvf+loateWl5fJnm1bZOfmTVY5ikD0X58+tN6w1ivVVlFeIVu/XOv1cs0wfujx+U62otZnffyXD3o915+diaZcQ8cuaVYGqwZ4Dx/MNpnJh/25tNY5mkHZKb2btOvUWXThx0Mmk1uD2f40zXzultnb3PuIHDIBJ/vH8f25tiHn6AJ07oDUoufnm+zeqh/RjzGlKrr2zLQC46dPHK/zAwvP+9fXtKnGos9D+9bsZA1M55qAnrdgn+f4/X2vWca9+g6wfr8c2L2zzg8r/O23oedpkFF/kkHnWGhKyvj6HtLv0Xt++gvnNhqsn3v/fzjvL9ZGMPm1btfeLDzYwypdkm1qnBfk5/vFEIg56AdqHcyfQVVlhSpFg9OIYan+AAAar0lEQVT+3t+vQXISAgjUKfCNaVOOdu/a+cHkOY88VefJnIAAAggggEAzF6AGdDN/wEwPAQQQuNgCmpXrmZl7scdg3++qa66VAcNH2m9l45pVJvv5Cee9bmhA8Y7//LETfNZ9GixqTDtz8oToVyCaBv40W7EhGYsaINWvYGia9d2YkgyBNG3MWPR57Ny8sclItc6yloUJlqYZyFoD+4LN/B66Zc69NU754M3FNd5frDfB5KcfOjXkg6dAzEGznfXPscb+WXaxnhv3QQABBBBAAAEEEGjeAgSgm/fzZXYIIIDAZS3wzit/E10sy67Dqtvde2dZi6blmUB5Umor876PaD1fu2lJhSWLFtpveUUAAR8CYydfZ9Wq7tVvgLNIo56qCxKuWbHcx1XsRgABBBBAAAEEEEAAgctNgAD05fbEmS8CCCBwGQnoj5yv/vhDGTH2amfWsfEJkpGZZX05O6s3tLSFLs5VV41mz+t4j8DlJqAf6lwz4ybnw53z86+Ut19+8fxbthBAAAEEEEAAAQQQQOCyF2ARwsv+WwAABBBAoHkLvPni86KL2RUX+V74T8sqHDl0UH7zwA9E6z/TGiagju7mXkDOvf9ibAfTWC7GfIPhHlof+i9//J1sWL0yGIbDGBBAAIFLLMByS5f4AXB7BBBAAIEgEiADOogeBkNBAAEEEGgagd1bN8v/fP8ea0HC9J69JC2jh7Wwmi7ot3PLRskxr7TGC+w1C529+OSfnI6aslaycxMfG8E0Fh9DDOndGuBfY366ICo6Rk6ZRSX379ohe3dsk9KSkpCeF4NHAAEEEEAAAQQQQACBwAsQgA68KT0igAACCASpwMnjx0S/1q8MnkXegpSqQcPSjOet69c26NpAXxRMYwn03IKlvzfMTxfQEEAAAQQQQAABBBBAAIG6BCjBUZcQxxFAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQaJEAAukFsXIQAAggggAACCCCAAAIIIIAAAggggAACCCBQlwAB6LqEOI4AAggggAACCCCAAAIIIIAAAggggAACCCDQIAEC0A1i4yIEEEAAAQQQQAABBBBAAAEEEEAAAQQQQACBugQIQNclxHEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQACBBgkQgG4QGxchgAACCCCAAAIIIIAAAggggAACCCCAAAII1CVAALouIY4jgAACCCCAAAIIIIAAAggggAACCCCAAAIINEggokFXcRECCCCAAAIIIOBDIL1TR5k1dbJUmuMLXl0sx06c9HGm993f+9YsiW3ZUtZs3CwffPq595PYiwACCCCAAAIIIIAAAgggEBICBKBD4jExSAQQQAABBEJHICkhTqKjo6wBR4aH13vgKUmJEhYWJonxcfW+lgsQQAABBBBAAAEEEEAAAQSCS4AAdHA9D0aDAAIIIIAAApdQYGi/PjJ1wlhrBE+99Joczjt20Ucz6/prpVe3dCkrL5eHH3/mot+/ITcMBreGjJtrEEAAAQQQQAABBBBAoOkFqAHd9MbcAQEEEEAAAQRCUCA6KvKSjrpFi9D8Z9qldrukD42bI4AAAggggAACCCCAQC2B0PyfTa1psAMBBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAg2AQowRFsT4TxBIVARHjF6djYlkVmMDFBMSAGgQACCARAIKtHhgzr30dSEqtqLGt5if2HcmX91u1SVFzs8w7u6yIiwiW/4JzsOXBQPl27XgqLfF/ns0NzoG/P7jJiYD9JTkiQUlNqQsey9LNVcuLU6QtdVu9jcWYxw2ED+kpax/bSJjVFzp4rlMNH82Tdlu2SnXvY6a9n1zTRMhKtUpKdfVPGj5GTp89IuRnf4n8uk7KyMudYuKltPXrIQOnSoarf8ooKyTmSZzxz5ItNW5zz7A2tZ639aVuy4jNpGR0tg/v2lm6dO0lkRIRouY9BfTKt/nSs2lqYOti3mMUctan5O8s+srb9+UX7HD1kkGRmpEt8XKycKyyW/Tk5svTTVdIyJlomj73S6mbZytVy5NjxGl3WZ271ddMb6bMf1r+vpCQlmHdhcjr/rKzbvNX6PqwxEN4ggAACISwQFR0VER4eEdi/1ELYg6EjgAACCFzeAgSgL+/nz+x9CISHt8hJjI31cZTdCCCAQOgJ3DZzmnTt3LHGwJMTE6R3924yftQwee61N2oFIvXkb9803QRvO9S4LiEuTjq2bSOjBg+QeS8ulOP1DBp//bqvWvd1d9oqOckKTL6//BP37kZtayD3GzOmWgFeuyMde4c2rU3wN0tWfblJ3lu+wjqkgeTMjK72adZr21apol/alqxYaQKl+dZ25/btZPb0KRJjgsjupnPon9nDCvIvePXvNQLWcebvFLv/E6dPy6hBA6yFFu3rI0zAWAOzHYyru9nXFJeU+h2Ajo6KkvtumyWxJvhuN513u9apMtTM+/2PPnHGsmH7zhrPvb5zq4+bjuX2G6dLeqea309JCfHWBwTDzQcSz7z8uj1kXhFAAIGQFkiKi62MKKvICelJMHgEEEAAAQQCJEAJjgBB0k3zEigtC8+Ni43j90fzeqzMBoHLVmDGpAlO8LnCZOpq5u/2PfukpLTUMtGA5ZxZN1mZsm6kGyd/xQk+29dt2bXHZNMWWqeZ1C6585YbJS72fKDTfb237dFDB9UIPuedOCmbduwyGb4F1umTx11ZIzDrrQ9/9mn9ZA26ayawtm2798oHn34uW81raXUms2Zg2wHefQdzZMO2HTUWHdybfcja96XZb2eIh5ms5NnTr3OCz5ohreNXUzXS1t4EuG+eMsna9vbLFYMHWnPUoPKBnFzzdVhKzbNYv3WHdT/bV6/VMenXJ1+s89ZVrX067zu+NsMJPmv2ts5Nn3dRcYlodvN1V1ctsuh5cUPm5q+b3mvqhHFO8LmsrFz2ZB+UXfuzne9D/VBj5lcneg6L9wgggEBICsSYTwFLosNyQ3LwDBoBBBBAAIEAC5ABHWBQumseAgnJbXIqKkujNKhSYH5cm4YAAgiEqsCA3j1lYO9e1vC1XMYfnvubE/DTnRNHj5Srhg22AqJjhg+R9z6syggelJUp/Xr1sK7TPwf1ujITzLTbhNEjZMywIaLB62+aLON5//eqfcjna6rJEL7mylHW8crKSnnSZE9rANpug/v0lmlfGW+/bdRrlsnsthfxe/fDj2X1hs1Of6lJSXLvN79uHR8xoJ8VnNVgqH5pGY6pE6oCtMtXrbFKlDgXmg21jImOsnZ9sWmrvL10uXNY7/fju++wgt6eWb7OSdUbWm7k49Vra+xe9eVG6/2s66+VXt3SpcIYLV6ytMY5db3RZ2hnbWtw/M9/fdkJjOu1108cJ0NMFrS31pC5+evWx5R/Gdqv6r6nzuTL/77wkjMuddPnoc+lf2ZP2bJzt2wzAXMaAgggEKoC8eb/EFEREbGJ4Uk5oToHxo0AAggggEAgBQhAB1KTvpqNQNjN957d8+h9m3p1Te+3bsu2ZjMvJoIAApefQO/uGdakNeD77Gt/rxF81gMaCO3WpZNoOY5Obds6QHZmsAZB5y9cVCP4bF1nagn3SOtilYxoU12mwrnYx4aeb7e3l35UI/is+/XP2+7pXaxSFPZ5DX21g7B6vWddaS2BMfep50w2cAuTDX0+qO7PvTTbefvefdapnvWvNQNas5U1iB0VGSkJpu5z/tmqzG5335rx7Bl8dh9vzHb3tM7W5TqWBQsXO0Feu8+3Plju1K2299mvgZib3Zfna9/qDzN0XE+bMht2triep9vPvLxI7r/rduuDkMyMbgSgPQF5jwACISXQq1tG8bnCkjXJc35U+y+BkJoJg0UAAQQQQCAwAgSgA+NIL81QoLS0fFFW94yeJiBSs8hnM5wrU0IAgeYr0Km6prAu9HbMlW1sz1gD0/NfWWS/dV61HIK2M+Y6zVj11rR+sNYs1sXyNOB79PgJb6c5+7pUL66nO3ThQ29NF/DTWsieLdxkyY4fNVy07IevtnPfASuLWY/vMeUzxo4Yap1667QpsnLdBlm9cbMzF6v8SFUFEl/ded2vWeDuTHD7JC31EdsyxixYWFWGQ/fHmOzwfKkde9ByGE3V2rZqZXV97OQpKaguleJ5r80mw3j8yGGeu615NXZutTqt3tGpXdX3k34flpRqKZCaVa60LIqON97UyrbP9dUX+xFAAIFgFxiY2fOUqcDxYrCPk/EhgAACCCBwsQQIQF8sae4TcgJtWnVZ0C4l/2fxcbFytuBcyI2fASOAAAJa2kCzcLUdOXbcbxCtBWxfd6Gg8sHcI06fmkV9oXP1RDuorZnD7gxYpxOzcejIUfdbZ1sX8dNSIRdqkZERTgBaaytrfeEeJqNaHbT2tH5poFOPfbb2S9l94OCFuvN5TIPhGtzWBRxbJSfXCqb6vLD6gF3vuq7z6ns8KirSlESJtC47nHfM5+X7D/n+ifDGzs3bTfX7KTE+3jqUkpQoP7v3Lm+nOftSTKkWGgIIIBCqAgnm/w5pndq3iYosfztU58C4EUAAAQQQCLQAAehAi9JfsxFInfPjAzsf+fd/jRsxZMI7y1bwe6XZPFkmgsDlI6BZphr801ZcUuL3xDWQa193rqjI53XuY/4sRBhdXTvZXvzQW8elpWXedotmal/oOr1Is2vtpue/+MY7ojWeRw7qL1p/WptmKnc3pUD0S2skP/XSa/Wy0UDqd26+wQnQ2/fTV72n7ebef7G21c4eg2Zj+2oJcVUfSngeb6q5ub+fPO/p7b1+YEBDAAEEQlXgmqtGHS04V7Q0+e7fZYfqHBg3AggggAACgRYgqBZoUfprVgI9UtPubJOasn/1hi11ZvY1q4kzGQQQaBYCGpAsKi6xFs1r17qqNIM/EyspKTVB2VIrm9ZdS9nzWjujWfdnm7rGdbW84yclrnNL0Z8s8dVapyR7PaRZw488ucDrsQvtXLVhk+hXTHS09OqaJv3NIoJaJ1kDxZqNq+U5njO1sf1tk8Zc4QSfNZNYazkfMaVHdKFGDf7qQntTJ4zzt7uAnqf315rTiQnx0r51a599d+3c0euxppqbfj/phwdaF3vvwUPywqK3vN6fnQgggECoC+jfmf0ze7WtkLIHQn0ujB8BBBBAAIFACpBiEkhN+mp2AmEmC9rUP503aczo+q1S1ewkmBACCISqQN6JqrrMmgHsK7M0rWMHGdSntxWYteeZV13PWUtM+GpprprO+w/l+jrN2Z99uCpIrWUeNPjrrfkKjno7tz77ioqLRWtWa1b0b+c/79Rx7tT+/MKLnv21CKv9zyR7kT8NqD5vAqlaxkPLNGnwV1t6J+/BXc++m+q9XcJEg/xD+mbVuk3r1BQZlJVZa7/uCNTcvLnlVdcfv1Bg3Oug2IkAAgiEkMB1468qKiwueyx1ztwDITRshooAAggggECTC9T+n1WT35IbIBBaAmn/9cd7UhMTN02dOLY4tEbOaBFAAAEROzCspSdumDShFkn7Nq3l9hunyfSvjJcJo0c6x+1gsdYUvnbcVc5+e0Mzqof172u91eBuXeUx9MQDh85nSc+efp3dlfOqC9CZD/yc943ZmDZxvNwz+2b51g3X16rRfK6wSOwAu+c97ECy7k/r1MHzsEQYR2262KD7XN2nZUi0LnRjm92vLu7Yvo3/met635XrNzq3nzphrFU3O9HUAY82CyIO6pNpyofM9PlBRGPmZo9Zb+7NLTu36tm3jImWMcOGOGO0N/RDiW/fNF3uu+1WmeHl+9Q+j1cEEEAgWAVmTpqQl5qcsrbDfXN/GKxjZFwIIIAAAghcKgHfS8lfqhFxXwSCUOAXX7v6pZSEhDkmvy3OvehWEA6VISGAAAI1BHKPHjOB4j5W4FR/NFhLXGgA1lQslsF9e5vA89VWXWS9aPE/lsqpM/nW9YfNooVD+/WViPBw0SzhJBPELDSBZm1D+/WRGddMMH1W/TPiveWfSK5r0TsNTmd1z7DO/WLTFjl7rmoh11P5+dKvZ3fR+sSxMTFWHWYNXmuZkAG9e8rMyROlpSmVYTddOHHb7r3223q9ZpgyG726pVuZ1rpAoi7Kp+PQQPz4kcOkb68eVn97TQazZkbbTesVDzbZ4Np0HgWFhdb4dIzaMrp0luTEBKsffT1+6rQViNb5fmPGVKvMhHWi+WX1hs3V1iJad1ndtG01c7rQgo0d27WRLh3aW+d2bt/O1LbOlwqTYW2PwTrg4xetg11g5qlz1zIjOt4rBg+0AtG9M7pZz1OzkeNatrR62LJrj9jZyQ2dm3ZUl9uRvOPW92G4+X7S55FgPmwoKim26m937dxJZk+fYoLtrUUD1PrML7RQoo+psxsBBBC4ZAJjRgw5PTAr61zLluFXzn37E5JWLtmT4MYIIIAAAsEqULUyUbCOjnEhEEQC2X/+ycjw0pI3t+7bm/z2v5ZHBdHQGAoCCCBwQQENlN77zVus4KOvEzfv3C2vvffPGodbmWD1PbfeXCuD2H3SR6YG8rLPVrl3WcHkGyZNtPY9bRb5cwenNfj8/dtn1wjUui/WQLEGfjV4umHbDlm8ZKn7sN/bmk39vdtm1biPZum6FwnUGtnPmvrPek+7aRbw/Xfd7gTl7f2PPftXq75y9/QuMtvUjXb3Y5+jr8dOnrKC/Lr9xN9ecYK7Hdq2kbtuuVF3y6J/fCAbXUFva6frF/e59m6tyf3ref7XwB47fIiMM4F2d9kVnf+6Ldvki01b5c6vz7S6ftU88y3m2Wtr6Nz02rrc9Bz9AGTOrJtqjEn3u9uhw0dlwauLa2WXu89hGwEEEAgmgZlfnXCiV3paYVRc0vUp33lwXTCNjbEggAACCCAQLAJkQAfLk2AcQS/w+3dXHPrvmWOfiYmImZbVs3tC7tG8SM2MoyGAAALBLqCZs7v2HRCt2axlItytsKhYln76uSxZ8Zl7t7VdWFQku7OzzXUdrKxl9wkavP1o9Rey/PM17t3Wtq8MaD1YWlYmO81YupmsVw1Gu5sGZRe+u8TK1tUAr9Yz3rF3v/sUv7e1JIhmIOtYUkwAXvuzg8bl5eWiJSHmL1zkZHzbHVdUVMie7INiFqCVJLOYn90+N6UtiktK5OTpM+b4IcnMSDdZv5H2YdHrlq1cLXtMRnVmRldr/8p1G52scc34Hj6gqmRJXRnQWlNa79PejF0zgrVpyY8Va/yPa+zPyZVP166X3fsPWmVYVn25Sd7/6BPRjOeUpAQny1s/eDhWXZ+5oXPT8dXlpufo35l7jV26KW3S0ni4m34/fbxmrbzxr2UEn90wbCOAQNAK6Idqs6ZOPtGuVeudMXFRV6Te9ct9QTtYBoYAAggggMAlFiAD+hI/AG4fmgL75n7v6eTkxDv1P/Qfm+w/+8fLQ3M2jBoBBC4ngZjoKNHMZi2todmmZSYY60+zr4uOjLLKTmhZiMY2LQPRplWKlJjsXi35oUHMpmoJZlE+DRYcNcHW/LMFft1GjWJMAFiDoxp89mxaH1sD3FrSRDOfm6Jp7WYtb6EfIpSZ4H1dTRcZ1DIrpsKK7DqQ7fWasSOGytWjhltd/W7BC9Yiip79NmZudbnpvWJMqRUdp5ZEOWIWvDzHB7qej4D3CCAQpAK6yOsY81MmIwb0M3+fFD7e5YePfTdIh8qwEEAAAQQQCBoBAtBB8ygYSKgJVD7167Q9Jw89kZoSP2X77v1hmqWnX2RFh9qTZLwIIIBA8xEY0jdLrp84zpqQfsCgWd7u1rFdW/m3r82wymDo31e/feZ592G2EUAAAQS8COgHplpbP7NbemVm93TJzy9cHJcQ/YPUOXMPeDmdXQgggAACCCDgIUAA2gOEtwjUV+DEUw+k5eeX3FpRWTErJTF+QHFJcenpggI5c/psZJGXjLn69s/5CCCAAAII1EegT48Mp86y1n3WciSaxR1lMrajXGVDtDb38SbK3K7PeDkXAQQQCEaBGPMTKEmJ8SWJ8XFh0VHRkafOFKyPigx/ITkx5rXYOx7ODsYxMyYEEEAAAQSCVYAAdLA+GcYVkgKVjz4aV9y+rENlwdmOZREtOpaXlyWF5EQYNAIIIIBAyAqES1jbCmnx3UqpbOttEqYadoU5Nt9Ubl7r7Tj7EEAAAQTELMAbcVrKKnIi4+Jzog9H5Ib96Ef+1W8CDwEEEEAAAQRqCRCArkXCDgQQQAABBBBAIPQFzjz74HRTVvuKsPKKwaa6doyEVW4IM2sZJqVWLA27+ZG80J8hM0AAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBooMD/AwCFStMrAY8qAAAAAElFTkSuQmCC)

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



### 🎯 Retriever Fusion Strategy

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

# Meta-ranker training data = val_mask impressions from Cell 53.
#
# Why this is better than the previous SET_B approach:
#   Before: meta-ranker trained on SET_B users only (30%), with s5_score from
#           a model trained on SET_A (70%). At eval time, ~70% of users are
#           SET_A users whose impression patterns the meta-ranker never learned.
#
#   Now: every warm user contributes impressions to both base-model training
#        (via tr_mask, 85%) and meta-ranker training (via val_mask, 15%).
#        The val_mask impressions are not gradient-fitted by S5 (only used for
#        early-stopping monitoring), so lgb_model.predict() on them is OOF.
#        All base features (tfidf_sim, cat_affinity, etc.) are already computed
#        in imp_train_df — no need to recompute anything.

meta_imp_df = imp_train_df[val_mask].copy().reset_index(drop=True)
n_pos_meta  = int(meta_imp_df['label'].sum())
print(f'Meta-ranker data : {len(meta_imp_df):,} rows | '
      f'{meta_imp_df["impressionId"].nunique():,} impressions | '
      f'{meta_imp_df["userId"].nunique():,} users')
print(f'Labels           : pos={n_pos_meta:,}  neg={len(meta_imp_df)-n_pos_meta:,}')
print(f'Avg group size   : {len(meta_imp_df)/meta_imp_df["impressionId"].nunique():.1f} candidates/impression')

# ── Retriever membership features ─────────────────────────────────────────────
# Seen-filter is cleared so all articles are reachable, matching the inference
# path in s6_score/s7_score which also pops the seen-cache before retrieval.
relevant_m = meta_imp_df['userId'].unique()
print(f'Building retriever lists for {len(relevant_m):,} users...', end=' ', flush=True)

s2_rows_m, s3_rows_m, s4_rows_m = [], [], []
_orig_seen_m = dict(_seen_cache)
_seen_cache.clear()

for uid in relevant_m:

    for rank, nid in enumerate(s2_category(uid, N_STAGE1)):

        s2_rows_m.append((uid, str(nid), rank))

    for rank, nid in enumerate(s3_itemcf(uid, N_STAGE1)):

        s3_rows_m.append((uid, str(nid), rank))

    for rank, nid in enumerate(s4_temporal(uid, N_STAGE1)):

        s4_rows_m.append((uid, str(nid), rank))

_seen_cache.update(_orig_seen_m)
del _orig_seen_m
print('done.')

s2_df_m = pd.DataFrame(s2_rows_m, columns=['userId','newsId','rank_s2']); del s2_rows_m
s3_df_m = pd.DataFrame(s3_rows_m, columns=['userId','newsId','rank_s3']); del s3_rows_m
s4_df_m = pd.DataFrame(s4_rows_m, columns=['userId','newsId','rank_s4']); del s4_rows_m

meta_imp_df = meta_imp_df.merge(s2_df_m, on=['userId','newsId'], how='left')
meta_imp_df = meta_imp_df.merge(s3_df_m, on=['userId','newsId'], how='left')
meta_imp_df = meta_imp_df.merge(s4_df_m, on=['userId','newsId'], how='left')
del s2_df_m, s3_df_m, s4_df_m; gc.collect()

meta_imp_df['in_s2'] = meta_imp_df['rank_s2'].notna().astype('int8')
meta_imp_df['in_s3'] = meta_imp_df['rank_s3'].notna().astype('int8')
meta_imp_df['in_s4'] = meta_imp_df['rank_s4'].notna().astype('int8')
meta_imp_df[['rank_s2','rank_s3','rank_s4']] = (meta_imp_df[['rank_s2','rank_s3','rank_s4']].fillna(N_STAGE1))
meta_imp_df['n_retrievers'] = (meta_imp_df[['in_s2','in_s3','in_s4']].sum(axis=1).astype('int8'))

print(f'meta_imp_df ready: {meta_imp_df.shape}')
```

    Meta-ranker data : 874,126 rows | 23,544 impressions | 17,442 users
    Labels           : pos=35,576  neg=838,550
    Avg group size   : 37.1 candidates/impression
    Building retriever lists for 17,442 users... done.
    meta_imp_df ready: (884828, 26)
    CPU times: user 1h 51min 49s, sys: 3.12 s, total: 1h 51min 52s
    Wall time: 14min 48s



```python
%%time

# OOF s5_score: lgb_model was trained on tr_mask (85% of impressions) so predicting on val_mask (meta_imp_df) is genuinely out-of-fold.
meta_imp_df['s5_score'] = lgb_model.predict(meta_imp_df[FEATURE_COLS].values.astype('float32'))

# Sort by impressionId for contiguous LightGBM groups
meta_imp_df = meta_imp_df.sort_values('impressionId').reset_index(drop=True)

# Impression-level 85/15 split for meta-ranker's own early-stopping
all_meta_imp_ids = meta_imp_df['impressionId'].unique()
rng_meta_split   = np.random.default_rng(100)
val_meta_ids     = set(rng_meta_split.choice(all_meta_imp_ids, size = int(len(all_meta_imp_ids) * 0.15), replace = False))

tr_m  = ~meta_imp_df['impressionId'].isin(val_meta_ids)
val_m =  meta_imp_df['impressionId'].isin(val_meta_ids)

xm_tr  = meta_imp_df.loc[tr_m,  STAGE2_FEATURE_COLS].values.astype('float32')
ym_tr  = meta_imp_df.loc[tr_m,  'label'].values.astype('int')
gm_tr  = meta_imp_df.loc[tr_m].groupby('impressionId', sort=True).size().values

xm_val = meta_imp_df.loc[val_m, STAGE2_FEATURE_COLS].values.astype('float32')
ym_val = meta_imp_df.loc[val_m, 'label'].values.astype('int')
gm_val = meta_imp_df.loc[val_m].groupby('impressionId', sort=True).size().values

print(f'Meta train: {tr_m.sum():,} rows | {gm_tr.shape[0]:,} impressions | avg {gm_tr.mean():.1f}')
print(f'Meta val  : {val_m.sum():,} rows | {gm_val.shape[0]:,} impressions')

meta_lgb_params = {
    'objective'        : 'lambdarank',
    'metric'           : 'ndcg',
    'ndcg_eval_at'     : [5, 10],
    'label_gain'       : [0, 1],
    'learning_rate'    : 0.03,
    'num_leaves'       : 63,
    'feature_fraction' : 0.8,
    'bagging_fraction' : 0.8,
    'bagging_freq'     : 5,
    'min_child_samples': 5,
    'verbose'          : -1,
    'n_jobs'           : -1,
}

meta_lgb = lgb.train(meta_lgb_params,
                     lgb.Dataset(xm_tr, label=ym_tr, group=gm_tr),
                     num_boost_round = 800,
                     valid_sets      = [lgb.Dataset(xm_val, label=ym_val, group=gm_val)],
                     callbacks       = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],)

tr_sorted_m  = meta_imp_df[tr_m].sort_values('impressionId')
val_sorted_m = meta_imp_df[val_m].sort_values('impressionId')

xgb_meta = XGBRanker(n_estimators          = 1000,
                    learning_rate         = 0.05,
                    max_depth             = 6,
                    subsample             = 0.8,
                    colsample_bytree      = 0.8,
                    objective             = 'rank:ndcg',
                    eval_metric           = 'ndcg@10',
                    early_stopping_rounds = 30,
                    verbosity             = 0,)

xgb_meta.fit(tr_sorted_m[STAGE2_FEATURE_COLS].values.astype('float32'),
             tr_sorted_m['label'].values.astype('int'),
             group      = tr_sorted_m.groupby('impressionId', sort=True).size().values,
             eval_set   = [(val_sorted_m[STAGE2_FEATURE_COLS].values.astype('float32'), val_sorted_m['label'].values.astype('int'))],
             eval_group = [val_sorted_m.groupby('impressionId', sort=True).size().values],
             verbose    = False,)

print(f'Meta-LGB best iteration : {meta_lgb.best_iteration}')
print(f'Meta-XGB best iteration : {xgb_meta.best_iteration}')
print(f'STAGE2_FEATURE_COLS ({len(STAGE2_FEATURE_COLS)}): {STAGE2_FEATURE_COLS}')
```

    Meta train: 751,188 rows | 20,013 impressions | avg 37.5
    Meta val  : 133,640 rows | 3,531 impressions
    [100]	valid_0's ndcg@5: 0.963325	valid_0's ndcg@10: 0.965637
    [200]	valid_0's ndcg@5: 0.964396	valid_0's ndcg@10: 0.966579
    Meta-LGB best iteration : 174
    Meta-XGB best iteration : 10
    STAGE2_FEATURE_COLS (21): ['u_click_count', 'u_click_freq', 'm_log_clicks', 'm_log_impr', 'm_article_len', 'cat_affinity', 'taste_affinity', 'tfidf_sim', 'recent_tfidf_sim', 'article_age_days', 'ctr_norm_rank', 'imp_size', 'subcat_clicks', 'in_s2', 'in_s3', 'in_s4', 'rank_s2', 'rank_s3', 'rank_s4', 'n_retrievers', 's5_score']
    CPU times: user 2min 3s, sys: 198 ms, total: 2min 3s
    Wall time: 17.3 s



```python
del xm_tr, xm_val, ym_tr, ym_val, gm_tr, gm_val, tr_sorted_m, val_sorted_m; gc.collect()
```

    Computing TF-IDF affinities... done.
    Computing recent TF-IDF affinities... done.
    Building retriever membership flags for SET_B users... done.
    meta_imp_df: (1745119, 26)  impressions=46,921  avg_cands=37.2
    CPU times: user 1h 29min 10s, sys: 4.28 s, total: 1h 29min 15s
    Wall time: 12min 25s



```python
meta_imp_df.head()
```





  <div id="df-3bfeb3a2-9f4e-447c-8c4d-823cfef50428" class="colab-df-container">
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
      <td>4</td>
      <td>U34670</td>
      <td>N35729</td>
      <td>0</td>
      <td>4.0000</td>
      <td>3.2912</td>
      <td>8.1158</td>
      <td>9.6434</td>
      <td>0.2164</td>
      <td>169.0000</td>
      <td>1.7506</td>
      <td>news</td>
      <td>0.4082</td>
      <td>0.4294</td>
      <td>0.0060</td>
      <td>0.0060</td>
      <td>0.0000</td>
      <td>4.0000</td>
      <td>0.3333</td>
      <td>200.0000</td>
      <td>2.0000</td>
      <td>200.0000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>U34670</td>
      <td>N35729</td>
      <td>0</td>
      <td>4.0000</td>
      <td>3.2912</td>
      <td>8.1158</td>
      <td>9.6434</td>
      <td>0.2164</td>
      <td>169.0000</td>
      <td>1.7506</td>
      <td>news</td>
      <td>0.4082</td>
      <td>0.4294</td>
      <td>0.0060</td>
      <td>0.0060</td>
      <td>0.0000</td>
      <td>4.0000</td>
      <td>0.3333</td>
      <td>200.0000</td>
      <td>87.0000</td>
      <td>200.0000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>U34670</td>
      <td>N33632</td>
      <td>0</td>
      <td>4.0000</td>
      <td>3.2912</td>
      <td>5.1705</td>
      <td>7.5776</td>
      <td>0.0884</td>
      <td>89.0000</td>
      <td>1.6507</td>
      <td>sports</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0033</td>
      <td>0.0033</td>
      <td>0.0000</td>
      <td>4.0000</td>
      <td>0.6667</td>
      <td>200.0000</td>
      <td>200.0000</td>
      <td>200.0000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>U34670</td>
      <td>N49685</td>
      <td>1</td>
      <td>4.0000</td>
      <td>3.2912</td>
      <td>7.7385</td>
      <td>8.8860</td>
      <td>0.3154</td>
      <td>187.0000</td>
      <td>1.7186</td>
      <td>music</td>
      <td>0.4082</td>
      <td>0.3807</td>
      <td>0.5246</td>
      <td>0.5246</td>
      <td>1.0000</td>
      <td>4.0000</td>
      <td>0.0000</td>
      <td>200.0000</td>
      <td>29.0000</td>
      <td>200.0000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>U34670</td>
      <td>N49685</td>
      <td>1</td>
      <td>4.0000</td>
      <td>3.2912</td>
      <td>7.7385</td>
      <td>8.8860</td>
      <td>0.3154</td>
      <td>187.0000</td>
      <td>1.7186</td>
      <td>music</td>
      <td>0.4082</td>
      <td>0.3807</td>
      <td>0.5246</td>
      <td>0.5246</td>
      <td>1.0000</td>
      <td>4.0000</td>
      <td>0.0000</td>
      <td>200.0000</td>
      <td>82.0000</td>
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
    <button class="colab-df-convert" onclick="convertToInteractive('df-3bfeb3a2-9f4e-447c-8c4d-823cfef50428')"
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
        document.querySelector('#df-3bfeb3a2-9f4e-447c-8c4d-823cfef50428 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-3bfeb3a2-9f4e-447c-8c4d-823cfef50428');
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
meta_imp_df.head()
```





  <div id="df-a6f03841-fcea-441a-8ece-7050c9108c78" class="colab-df-container">
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
      <th>s5_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>U1004</td>
      <td>N45368</td>
      <td>0</td>
      <td>2</td>
      <td>2.0000</td>
      <td>1.0986</td>
      <td>4.2341</td>
      <td>0.0341</td>
      <td>142.0000</td>
      <td>1.4650</td>
      <td>sports</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>159.0000</td>
      <td>0.6076</td>
      <td>200.0000</td>
      <td>9.0000</td>
      <td>200.0000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>-6.5286</td>
    </tr>
    <tr>
      <th>1</th>
      <td>U1004</td>
      <td>N55689</td>
      <td>0</td>
      <td>2</td>
      <td>2.0000</td>
      <td>8.3703</td>
      <td>9.8155</td>
      <td>0.2351</td>
      <td>224.0000</td>
      <td>1.6506</td>
      <td>sports</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0013</td>
      <td>0.0013</td>
      <td>0.0000</td>
      <td>159.0000</td>
      <td>0.0316</td>
      <td>200.0000</td>
      <td>64.0000</td>
      <td>200.0000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>-6.5286</td>
    </tr>
    <tr>
      <th>2</th>
      <td>U1004</td>
      <td>N25791</td>
      <td>0</td>
      <td>2</td>
      <td>2.0000</td>
      <td>5.0938</td>
      <td>6.4489</td>
      <td>0.2409</td>
      <td>175.0000</td>
      <td>1.3492</td>
      <td>news</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0098</td>
      <td>0.0098</td>
      <td>0.0000</td>
      <td>159.0000</td>
      <td>0.0253</td>
      <td>200.0000</td>
      <td>63.0000</td>
      <td>200.0000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>-6.5286</td>
    </tr>
    <tr>
      <th>3</th>
      <td>U1004</td>
      <td>N53585</td>
      <td>0</td>
      <td>2</td>
      <td>2.0000</td>
      <td>7.9502</td>
      <td>9.2012</td>
      <td>0.2849</td>
      <td>132.0000</td>
      <td>1.5704</td>
      <td>tv</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0044</td>
      <td>0.0044</td>
      <td>0.0000</td>
      <td>159.0000</td>
      <td>0.0190</td>
      <td>200.0000</td>
      <td>62.0000</td>
      <td>200.0000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>-6.5286</td>
    </tr>
    <tr>
      <th>4</th>
      <td>U1004</td>
      <td>N60750</td>
      <td>0</td>
      <td>2</td>
      <td>2.0000</td>
      <td>4.8363</td>
      <td>5.8693</td>
      <td>0.3152</td>
      <td>303.0000</td>
      <td>0.2710</td>
      <td>sports</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0032</td>
      <td>0.0032</td>
      <td>0.0000</td>
      <td>159.0000</td>
      <td>0.0127</td>
      <td>200.0000</td>
      <td>61.0000</td>
      <td>200.0000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>-6.5286</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-a6f03841-fcea-441a-8ece-7050c9108c78')"
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
        document.querySelector('#df-a6f03841-fcea-441a-8ece-7050c9108c78 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-a6f03841-fcea-441a-8ece-7050c9108c78');
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
# cleanup
del meta_imp_df; gc.collect()
```




    90



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
        res = evaluate_strategy(fn, eval_warm, K = K, n = EVAL_N)
        print(f'{time.time()-t0:.0f}s  composite={res["composite"]:.4f}')
        all_results[(name, K)] = res
```

      S1: Popularity  @K=5... 0s  composite=0.3385
      S1: Popularity  @K=10... 0s  composite=0.4794
      S2: Category Affinity  @K=5... 0s  composite=0.2964
      S2: Category Affinity  @K=10... 0s  composite=0.4295
      S3: Item-CF  @K=5... 0s  composite=0.3056
      S3: Item-CF  @K=10... 0s  composite=0.4359
      S4: Temporal Taste  @K=5... 0s  composite=0.2964
      S4: Temporal Taste  @K=10... 0s  composite=0.4295
      S5: LightGBM Base  @K=5... 27s  composite=0.3988
      S5: LightGBM Base  @K=10... 28s  composite=0.5157
      S6: Meta-LGB (2-Stage)  @K=5... 109s  composite=0.3850
      S6: Meta-LGB (2-Stage)  @K=10... 111s  composite=0.5004
      S7: Ensemble (LGB + XGB)  @K=5... 158s  composite=0.3488
      S7: Ensemble (LGB + XGB)  @K=10... 157s  composite=0.4777
    CPU times: user 1h 15min 10s, sys: 2.45 s, total: 1h 15min 12s
    Wall time: 9min 52s



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
           S5: LightGBM Base     0.1090  0.4274 0.1669 0.3016    0.4960     0.3988
      S6: Meta-LGB (2-Stage)     0.1042  0.4094 0.1596 0.2890    0.4810     0.3850
    S7: Ensemble (LGB + XGB)     0.0926  0.3670 0.1424 0.2626    0.4350     0.3488
              S1: Popularity     0.0908  0.3658 0.1407 0.2540    0.4230     0.3385
                 S3: Item-CF     0.0808  0.3320 0.1257 0.2262    0.3850     0.3056
       S2: Category Affinity     0.0780  0.3236 0.1219 0.2197    0.3730     0.2964
          S4: Temporal Taste     0.0780  0.3236 0.1219 0.2197    0.3730     0.2964
    
    =================================================================
      LEADERBOARD  @  K = 10
    =================================================================
                    strategy  precision  recall     f1   ndcg  hit_rate  composite
           S5: LightGBM Base     0.0786  0.5942 0.1345 0.3604    0.6710     0.5157
      S6: Meta-LGB (2-Stage)     0.0758  0.5760 0.1298 0.3468    0.6540     0.5004
              S1: Popularity     0.0719  0.5579 0.1239 0.3209    0.6380     0.4794
    S7: Ensemble (LGB + XGB)     0.0718  0.5563 0.1236 0.3275    0.6280     0.4777
                 S3: Item-CF     0.0642  0.5130 0.1114 0.2887    0.5830     0.4359
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
ax.set_xlabel('Composite score (%) — mean of NDCG@10, HR@10')
ax.set_title('News Recommendation Benchmark  |  Composite @ K=5')

for bar, val in zip(bars, lb10['composite']):

    ax.text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2, f'{val*100:.2f}%', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('benchmark_composite.png', dpi=150, bbox_inches='tight')
plt.show()
```


    
![png](data:image/png;base64,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Zmn9+vWKiIjQb7/9pt69e6tdu3aKjY1NdV8+Pj66ceOGJk2apHPnzungwYP67LPPzELqpJccvvzyS128eFE7duzQ2LFjVadOHUVEROjcuXNKTExU1qxZFRkZqSNHjhhrMz+uWbNmyp49u/r166djx47p3Llzmjlzpvbs2aOuXbv+4/siSQ8ePFBkZKTx8+eff2rSpEnau3ev+vbtKylt79+zvj8vU7ly5ZQhQwYFBgYqIiJC+/fvV69evdSwYUNFR0fr+PHjZi84PI27u7u6d++uGTNm6LfffnupdUuPXsj49NNP9f3336tbt27av3+/Ll26pF9//VWDBw/WV199pZEjRxqh94cffqiff/5Zs2bNUnh4uEJCQtSnTx9lz55dzZs3T5OalixZol9++UXnzp3T5MmTdfXqVf3vf/+TJNWtW1fFixc3/p65cOGCli5dqjVr1qhTp07KmDGjLly4oI8++kiLFi1SeHi4Ll26pLVr1+rcuXNPnB48a9asCg8PV2hoqK5cuaI6deqoVKlSGjhwoPbt26dLly5p9+7dateunUaNGvXE2p2cnFShQgWtXbtWkjR58mQ9ePBAderUUb9+/dSqVSv16dNHGTNmVM2aNXXx4kUtXLjwiUH70+TJk0czZszQ2bNnNWTIEJlMJmON8af9SFKBAgW0Z88effTRR8bfE+vXr9eMGTP09ttvq1y5cs9dDwAAAPA6Yip1AAAAAHjNlShRQm3bttWSJUvMtnfr1k0ODg765ptvNHnyZNna2qpy5cr65ptvjFG40qPRp3v37jWbMrhixYqKiopStmzZVLZsWUmPQpxFixZpxowZ8vPz0/3795UvXz55eXmpT58+L1R7gwYNVKtWLX3xxRd65513lCdPHjVp0kTW1tZasGCBvvzyS2XIkEEuLi766quvjACmePHiWrRokfz9/eXr6ytHR0d5eHhoyJAhxpTJs2fP1uTJkzVixAhFR0crb968ateunT766KMXqjUtjB49Wnny5FFAQICuXr0qBwcH1ahRQ8uWLUu2VvXT1KlTR8OGDdOiRYu0fPlyFS9eXMOGDdOYMWOMNhUrVtSAAQO0dOlSrVixQi4uLpoyZYpu3rypjz76SK1bt9aOHTvUpk0b/fLLL+rYsaPatGmjDh06mJ0rR44cWrp0qSZPnqxOnTrpwYMHevPNN/X555+brQ//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ndW6dWtt2LBBTZs2lSSVLFlSsbGxioiIUPHixZ9+0wAAAAAAAAAAAJCuEhISnrivTZs2at26tXbv3q22bdtq+PDhWrZsmXbu3KncuXNr586d6tixo7Zu3ZosKxw/frzRv5ubm6pVq6Y9e/YoU6ZMio2NNTvv/fv3lTlzZlWtWlXbt29X48aN1bRpUwUFBcnT01P29vayt7dX5syZ5erqqq1btz617he5/rTq77/mee7baxOMFyhQQCtWrNCZM2e0b98+HT58WCNHjtSMGTO0bNky5c2bVyaTST4+Psbc/oMGDdLq1asVHBysGjVqpOo8Tk5Oql69urZs2aK2bdtq06ZNRqiaJGlN86QANomNjY2uXLmiHDlymG23srLS8uXLtWbNGl2/fl0mk0lxcXEvPGrZyspKCQkJMplMxtQP69ev16hRoyQ9mrqhUqVKZmuIb9iwQcWLF5fJZNLly5c1ceJEde/eXQsXLkzWf3R0tCQZQf7jTCaT/P39tWnTJn399dfJ3ph5XLZs2TRv3jxFREQYv7PJkydr6tSpWrRoUYovBZw/f16TJk3SsWPHdP/+fSUmJip79uySpMjISDk6OprV5eLiYgTjFy5cUGhoqFxcXMzqLVasmPE5aW2Hl7n+OwAAAAAAAAAAANJGSEhIsm3nz59XTEyMypYtK0nKnj27HBwcNHPmTJUtW1aXLl3SpUuXlDNnTt25c0dbt27Vm2++aRx///59/fXXXypYsKCxLSoqStevX1e+fPn0559/mp335MmTqlatmn777Td5e3vL29tbV69e1cyZMzV27FiFhobq7t27CgkJ0aVLlxQTE5Ni3f9EaGhomvaH5F6bYDxJiRIlVKJECbVv316RkZFq2bKllixZosGDByt37tzKmjWr0dbBwUFOTk66cePGc53Dx8dHixYtUuvWrbVt2zatXr3abM3uTJkySZL27NljBK2Pu3jxotnn1atXa/78+ZozZ44qV64sGxsbszXLGzRooMuXL0uSevTooZ49ez61vmLFiikuLk4REREqXLiwpEchfVJQHxAQkGz68iRWVlYqWLCghg0bJk9PT4WFhT1x5PTf11tITEzUsGHDdOzYMS1fvtxsfYWnXUOhQoXUqlUrtWrVSjExMWrfvr3mzp2rGTNmJOu/W7duqlSpkrZt26YcOXJo9erVxrrwiYmJypDB/Cv5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nJ9++kkuLi7y8/NLs27r1q1yd3eXk5OTPD09FRISkgkVAq+eh/WrdevWqUKFCiafXfb29jp27FgmVQq8Oi5evKh+/frJ2dlZLi4uGjFihG7duiVJOnnypN59911Vq1ZNTZs21ZIlSzK5WuDV8LB+FRERofLly6f5vFq8eHFmlwy89H7//Xd1795d1apVk4uLiwYPHqyrV69Kkvbt26d27drprbfeUosWLbRx48ZMrhZ4NTysXx04cCDdz6vvvvsus0sGXhkTJkxQ+fLljdd8Vr0Yj34COWAGBgwYoMqVK2vHjh26fv26+vTpo4IFC+q9997L7NKAV9q2bdtUvHjxzC4DeGUtXLhQQUFBKlWqVJp1J0+e1IcffqjZs2erZs2a2r59u/r3769t27bptddey4RqgVfDo/qVJNWoUUPLly9/wVUBr76+ffuqSpUqCg4O1u3bt9WvXz999tlnGj16tPr06aMOHTpowYIFOnfunLy9vVW8eHE1bdo0s8sGXmoP61c+Pj6SpLCwsEyuEHi1JCQkyNvbW126dNHChQsVExOjQYMGaezYsRozZox8fX310Ucfyd3dXYcPH5aPj49Kly4te3v7zC4deGk9ql9169ZNxYoVU3BwcGaXCbySTp48qQ0bNhivIyMj+ax6QRgxDrMWFham33//XUOHDlXu3LllZ2enHj166Jtvvsns0gAA/3HW1tYPDfDWrFkjV1dXubq6ytraWq1atdKbb77JnaLAYzyqXwF4Nrdu3VKVKlX0wQcfKFeuXHrttdfUpk0bHTp0SLt27VJiYqJ8fHyUM2dOVa5cWe3bt+f3LeAxHtWvADybuLg4+fn5qU+fPrKyspKtra2aNGmi06dPa9OmTbKzs1O7du1kbW0tFxcXNWzYUGvWrMnssoGX2qP6FYBnl5KSojFjxqhHjx7GMj6rXhyCcZi148ePq1ixYsqbN6+xrHLlyjp37pxiYmIysTLg1RcYGKj69eurevXqGj16tO7cuZPZJQGvlG7duil37tzprjt+/LgqVapksqxSpUqMHAIe41H9SpIuXbqk9957TzVq1FCjRo1M7s4GkL48efJo4sSJKliwoLHs0qVLKly4sI4fP67y5cvL0tLSWFepUiX99ttvmVEq8Mp4VL+6b/jw4apTp45q1qypwMBAJSYmZkapwCsjb968at++vbJmvTdB6tmz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SUnRjh07tH//fmXLlk3x8fGaMGGCunfv/kRtVqlSRZcuXTK+f9K971e5cuVMvr9PKnv27BozZozmzZtnMu38P2v/8ccflZKSokOHDmnr1q2ytrZWYmKiJk2apI4dOz7RsSpWrKjU1FSTvhoWFqY8efI89nEOGaFgwYKKjIw0agkNDVVcXJwuXryoadOmKXfu3MqWLZvWr1+vAgUKPNGz5a2trfXGG2+YPHP61q1bOn/+vDGi/mnY2tpq4MCBCggIMEaCP0pCQoLmz5+vli1bysLCQi4uLsqePbu++uqrNNueOnVKzZs3161bt+Ti4qJcuXJp+fLl6bb74I0/r8J1exq+vr6Kj483uRkpf/78un37thISElSxYkUdOXJEUVFRioqK0vjx45UvXz5ly5ZN+/bt04ULF4wbpB6nSpUqJueYnJysEydOGKPPAQAAgIxGMA4AAAAAj/DRRx/p6NGjGjJkiC5fvqyUlBSdPHlSvXr1Uvbs2VW1alXZ29urbNmyCgwMVGxsrK5cuaJ58+apRYsWT/SM2vT88ssvCg0NVUJCglasWKE7d+6odu3aKlCggOrUqaMZM2YoOjpaN2/e1PTp0+Xs7KzXX39d48eP14cffqioqCilpqbq+PHjSklJSTO17f1RkH/++adiYmLUokULRUdHa+7cubp7964uXLggb29vLVu2LE1tly9fVsOGDRUSEqKUlBTdvn1bp06dMsJtT09PLVq0SFeuXNGNGzc0btw4nT59WlmyZFHLli21YMECXb58WbGxsZo6dapKlCghJyendK/D09Ql3Xs2+OnTp9Ndt3z5clWpUsUIbezt7XX06FFduXJF27ZtM6nhjz/+UPbs2TM0CH5eXl5eWrlypY4ePark5GRt3bpVLVu21N9//62iRYsqS5Ys+uWXXxQbG6svv/xS165d07Vr154oQPyn1q1ba9OmTTp69KgSExO1bt06nT59Wi1atMiw83n77be1e/duSdInn3yi4OBgubm56dixY5o0aZIsLS3l7u6uevXqqWHDhk/UZqVKlWRvb6/AwEDFxMTozJkzWrp0qckI1nfeeUeHDh164jrr1q2rJk2ayN/f3whEa9SoodDQUCUmJmro0KE6f/68mjRpoo0bN+rTTz9VpUqV5ObmpgIFCujdd999ouPY2trKzc1N06dPV1RUlC5fvqw5c+aoXbt2xtTc3bt319atW5+49qdRsmRJJScn6/Tp02rTpo0qV66sli1bauzYsfL19VXbtm3Vu3dv/fjjjwoICHjidjt16qSvvvpKZ86cUUxMjKZOnaqKFSsaMy8EBgZq0qRJT9xe586dZWlpqbVr1z50m9TUVIWHh8vHx0eWlpYaMGCAJMnGxkajRo3SrFmzNH/+fMXExCghIUG7d+/W+++/rwYNGihPnjzKmTOnxowZoxkzZmju3LnGM78jIiI0bdo0bd261aj/VbluT+r+zSALFizQqVOnJElWVlZG6H//0RkdOnTQwIED1aVLF3Xu3FmffvqpFi1apOnTpz/VOa5fv15Hjx5VXFyc5s2bJysrK9WvXz/DzwsAAACQmEodAAAAAB6pQoUK+vbbbzVr1iy1adNGsbGxeu2119SyZUu9//77RvA9d+5cjRs3TvXr11eOHDnUuHFjDR069JmP26pVK33zzTfy9fVVnjx5NGPGDGP62s8++0yffPKJmjVrpixZsqhWrVqaOHGiJOmDDz7QmDFj5ObmpqSkJJUqVUqBgYFpRj4XLFhQbm5uGjRokLy8vOTv76+5c+dq8uTJ+uKLL2Rra6vWrVvL29s7TW2vvfaaxo8fr/Hjx+vvv/+WjY2N6tWrZ0xX/sEHHyggIEDNmzeXlZWVGjdurP79+0u6NwXxuHHj1L59eyUkJMjJyUlLly41mT79Qfnz53/iuiSpVq1amj59ujGi/b7Lly9r1apVCgoKMpYVKVJEffv2lbu7u1577TWTQOfAgQOqVq2arKysHvOdenHatWunS5cuqX///oqJiVGZMmU0e/Zs46aH+899l+6Fh1OnTlW3bt3UuXNnff755091rBYtWujixYsaPny4rl27pjJlymjJkiUmz29/XrVq1VJSUpLWrFmj9u3ba/Xq1Sbr16xZo6ioqDTvXX9/f23YsMGYReD+jQ7jxo2Th4eHZs6cqdGjR6t27dqysbGRl5eXOnfubOx/7tw53b1796lqHTlypJo3b66VK1eqa9eusrOzU5UqVTRnzhwNHjxYixcvNtl+8eLFio6OVp48eUymgL//nOr7td9/trOPj498fX316aefasyYMWrUqJGyZcumli1bmjwT+sKFCyZT/j+t+6Hq/ZslduzYIeneyHRJeu+99xQQEKBFixZp3LhxJvv6+flpwIABio2NVZ48eYzlBw8eNPpjQkKC8Wz5GjVqaMmSJfLy8tLVq1fVtWtX3blzR87Ozpo9e7ax/9WrV5WYmPjE52BpaamPP/5YXbp0SbPu/vVMTU1VoUKF1KRJEwUGBpo8D7tt27YqVKiQFi5cqPnz58vCwkJ2dnby8/OTh4eHsV3z5s1VuHBhzZ8/X19++aXu3r2r/Pnzq3r16lq+fLnJjTSvwnV7GvdvBvnoo4+0evVqWVpaytvbW1OmTFH16tU1ZMgQDRkyxNi+UqVK8vLyUnR0tMnP3YsXL+qdd96RJCUmJurw4cNatmyZ8WiJevXqaciQIRo8eLCuX78ue3t7LViwwLhxCwAAAMhoFqnpPTANAAAAAJBpunbtKgcHh+cK1v/LPD091bx5c/Xq1euZ9k9ISFCDBg00ZswYNW3aNIOrw4POnj0rb29vNWvWTB06dFDJkiUVFxenU6dOaevWrdq2bZs2b95s8kzj5zVjxgw1aNDguaekvnbtmrp3766qVauqe/fuKlu2rJKSknT27Fl9//33Wrt2rb788kuVK1cugyqXgoKClDNnTjVv3jzD2nxQSkqKMTvGoEGD5OjoqKxZs+rSpUvas2ePli9frl69eql9+/YZdswLFy5o8eLFGjt2bIa1+aL9V67bZ599pj179sjPz081a9ZUjhw5FBkZqdDQUK1cuVL16tXT4MGDX1g9AAAAwNMiGAcAAACAlwzB+PPZs2eP/P39tXXrVuN5209j2bJl2rBhg4KCgpQlC08g+7dFRUVp4cKF+vHHH3XlyhVZW1urdOnSatCggdq1a5fuc96fx7vvvqslS5ZkyGwAsbGxWrp0qbZv366IiAhlzZpVJUqUUJ06ddSxY8c0jzB4Xv3799fo0aNVpEiRDG33Qampqcb7/9SpU0pKSlLhwoX19ttvq127dhn+jOulS5eqcOHCGTpNf2b4r1y3H3/8UStWrNDx48cVHx+vggULysnJSR4eHnJxcXmhtQAAAABPi2AcAAAAAF4yBOPPb8KECYqOjtbkyZOfar8///xTnTt31qpVqzJ02nAAAAAAAJC5CMYBAAAAAAAAAAAAAGaNOeEAAAAAAAAAAAAAAGaNYBwAAAAAAAAAAAAAYNYIxgEAAAAAAAAAAAAAZo1gHAAAAAAAAAAAAABg1gjGAQAAAAAAAAAAAABmjWAcAAAAAAAAAAAAAGDWCMYBAAAAAAAAAAAAAGaNYBwAAAAAAAAAAAAAYNYIxgEAAAAAAAAAAAAAZu3/AClxYZrwj7W5AAAAAElFTkSuQmCC
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


    
![png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAB8YAAAJOCAYAAADF3G1CAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAt4hJREFUeJzs/XeYVeXZP25/BhAEERB7R1SIilJENATFWKMolthj1AdbLEH9Yu8aa+zB3h4SNQaxazQaYnlQsYsiYkNQsBKKSIeZef/gnf1zRBDGgT1szvM4OHSvvfa9rrXWnmvKZ99rlVVWVlYGAAAAAAAAAEpUvWIXAAAAAAAAAACLkmAcAAAAAAAAgJImGAcAAAAAAACgpAnGAQAAAAAAAChpgnEAAAAAAAAASppgHAAAAAAAAICSJhgHAAAAAAAAoKQJxgEAAAAAAAAoaYJxAAAAAAAAAEqaYBwAAGApd/rpp6dt27bV/rVr1y6/+c1vcsMNN2TmzJnFLnGRa9u2ba688soav/6VV15J27Zt83//93+1WNXC+/3vf5/99ttvoV+333775fe///0iqKj2jRw5Mpdddll69uyZjh07pl27dtlmm23Su3fvPP/88ws8zrzO+d///ve0bds29957b43qGz16dPbZZ5+0bds2I0aMmOv5mTNn5vLLL88222yTdu3aZZdddskDDzxQo20BAACw4BoUuwAAAACKr2XLlnn00UcLjydNmpTBgwfnyiuvzIgRI3L11VcXsbqaO/jgg7P33ntn7733nu96L7zwQpo0abKYqqKmbr311txwww3Zcccd06dPn6y33npp0KBBvvrqqwwaNCinnHJKtt5661x22WVZZpllFnr8J554In/605/Sp0+fHHjggQv9+n/96185++yzs8oqq8xznfPOOy/PPvtsLrnkkqy//vp57rnncvbZZ6dx48bZddddF3qbAAAALBjBOAAAAKlXr15WXnnlwuOVV14566+/fsaPH58bbrghp556alZbbbUiVrjwZs+enXffffcnQ/Ek1faduuncc8/NK6+8kocffjjrrbdetefWWGONdOrUKfvvv38OPfTQXH311TnttNMWavwXX3wxp556anr16pWjjjqqRjVedtllOfvss1NRUZEzzjhjruc///zzPPTQQ7nggguy3XbbJUkOPfTQvP3227nuuusE4wAAAIuQS6kDAAAwT7/4xS+SJF988UVh2SOPPJJ99903nTp1SpcuXXLSSSfl66+/Ljzft2/fdO7cOQMHDky3bt3Su3fveY7/4IMPpm3btnn77bfz+9//Pu3bt8+2226bRx99NF988UV69eqVjh07Zvvtt88TTzxR7bVvv/12Dj/88HTt2jUdOnTI7373u7z55ptJkjFjxmSTTTbJtGnTcsYZZ6Rt27ZJ5lw2fo899si9996bLl265PLLL08y92W1v/nmm/Tp0yddunTJ5ptvnsMOOyxDhw79yeP13Xff5f/9v/+XTp06pVOnTjnllFMyderUwvNt27bNrbfemqOPPjqbbrppPvjggyRzLg/+xz/+Mdtss00222yz7L333nnmmWeqjV21TpcuXdKuXbvsuOOOuemmm1JRUTHPembMmJEDDzwwu+22W7799tskycCBA/Ob3/ymcLn8Hx7XZM7lvq+66qpst912adeuXbp27ZrTTz8948aNS5L06dMnv/3tb6u9pk+fPmnbtm0++uijwrKXX365cEnxqmP/yiuvZO+990779u2z44475qGHHvrJ4/rPf/4zzzzzTO69996st956efrpp7PHHntk0003zW677ZZXXnklhxxySO6///5cccUVueuuu/Lf//73J8et8s477+T444/PXnvtlVNOOWWBX/dDf/3rX7PnnnvO8/kXX3wxlZWV2Xbbbast32abbTJq1KiMHj26xtsGAABg/gTjAAAAzNOoUaOSJKuvvnqSOaH4qaeemg4dOuTBBx/MjTfemE8++SSHHXZYtXuRl5eX56677spNN92U888//ye3c/nll+eoo44qzAY+99xzc+aZZ+bggw/Ogw8+mHXWWSdnn312pkyZkmROSHzooYemvLw8t912W/r375/VVlstvXr1yogRI7L66qvnnnvuSZKceeaZeeGFFwrbmjBhQgYOHJi77rorRx999Fy1zJw5M4cffng+++yz3HLLLbnvvvvSvHnz9OrVq9oHAH7Mtddem86dO+fBBx/Mueeem3/961/585//XG2dAQMGZPPNN8+TTz6Z9dZbLxMmTMjBBx+c0aNH5+qrr85DDz2Uzp0757jjjsvLL7+cJKmsrMxRRx2VL7/8Mv369ctTTz2VE044ITfccENhP3+ooqIiJ598cr7++uvccccdad68eT755JOccMIJWX/99fPggw/m8ssvT//+/fPZZ59Ve+3ZZ5+dv//97+ndu3eeeOKJXHrppXnllVdy5JFHprKyMt26dcvw4cML5yOZc5/11VdfPa+++mq1ZWuuuWbWX3/9JMn48eNz/fXX5+yzz87DDz+c9ddfP+ecc06+/PLLeR7TysrKXHPNNTn77LPTsmXLPPHEE+ndu3d23nnnPPHEEzn11FNz2WWX5d13303Hjh2z2WabpXXr1hk8ePB8z1WVESNG5Kijjkr37t1zwQUXzPX8zTffnI4dO87z37nnnltYd911153vtkaOHJmGDRtm1VVXrbZ8nXXWSZJ88sknC1QzAAAAC8+l1AEAAJjLrFmz8sorr+TOO+/MTjvtVAjGb7755myxxRY566yzkiStWrXKZZddlj333DNPPfVUdt999yTJ1KlTc9hhh2XTTTddoO3tscce2XrrrZMkBxxwQF566aVsueWWhctNVy377LPPstFGG6Vfv36pV69e+vbtm+WXXz5Jcskll2S77bZLv3798qc//SkrrLBCkmT55Zevdqn0r7/+OrfffnvatGnzo7U8++yz+fDDD/Pwww9no402SpJccMEFufDCCzN69Oi5Qs3v69q1aw466KDCsXn99dfz+OOP57zzzktZWVmhnu9fqrtfv34ZN25c7r333kJAeuaZZ+bVV1/Nrbfemq222ipJcuedd6Zx48ZZaaWVkiRrrrlm/va3v2XQoEH5/e9/P1ctl1xySd544438/e9/L9T8yCOPpKysLJdddlnhuF199dWFY191fB599NH06dOnMPt5nXXWyemnn57evXvnjTfeyK9+9auUl5fnzTffzNZbb50RI0bku+++S69evfLqq6/md7/7XZI5M8a7detWGPubb77JHXfcUTj2hx9+eJ599tm89957hffYD7366quZNm1adtxxx8yYMSOXX355/ud//ifHHntskmTttdfO6NGjc9FFF6V9+/ZJkvXXX3++YXuVL7/8MocffngmTJiQfffdN/XqzT1/4IADDsguu+wyzzGaNm36k9upMnny5Cy33HLzHOO7775b4LEAAABYOIJxAAAAMm7cuHTs2LHweMaMGWnQoEH22GOPnH766UnmhHqffPJJevbsWe21G220UVq0aJH33nuvEIwnSbt27RZ4+5tssknh/5s3b14Y94fLqoLDd955J+3bty+Eu0nSqFGjdOrUKcOGDZvvtho1ajTPULxq7GWWWaba9lu0aJGrr776J/dj8803r/a4bdu2GTBgQMaOHZtVVlklydzH5Z133sk666xTCMWrbLXVVoXLjJeVlWXSpEm5+uqr8/bbb2fixImprKzM9OnTf/TDB7fffnvuv//+3HPPPWnVqlVh+UcffZR11lmn2nFbccUVq2373XffTWVlZTp37lxtzKr3x3vvvZfOnTunTZs2ef3117P11lvn5ZdfTvv27dO1a9fcd999SeZ8OGLo0KHp1atXYYwmTZpUO/YtW7ZMkkyaNOnHDmeS5LXXXsuWW26Z+vXr5z//+U+++uqrHHbYYdXWqVevXjbccMNCwDxt2rQfDbl/6PHHH89ee+2VsWPHpk+fPnnggQey5pprVlunRYsWadGixU+OBQAAQN0mGAcAACAtWrRI//79C48bNGiQlVdeOQ0bNiwsmzx5cpLkhhtuyK233lrt9dOmTcs333xTbVmzZs0K///666/nyCOPLDzefPPNc/vttxceN27cuPD/VTOrf2xZZWVloZYPPvigWpifzLkMelXYOi/fD4V/zHffffejs3oXRFWAX6VqH6ZNm1ZY9v3jkszZl9GjR8+1L7NmzcqsWbMyc+bMjBs3LgcffHDWXXfdnHvuuVl77bXToEGDnHzyyXPV8PHHH2fYsGGpqKiYK3CeMmVKmjRpMtdrvr+/Vef5h8epKnSuunx6t27d8tprryWZMzO8S5cu2WyzzTJp0qSMGDEiX3zxRSorK/PLX/6yMMaPbTv5/87rj/nmm28KM95HjRqV5s2bzzVrf9CgQenQoUPh8ciRI7PPPvvMc8wqe+yxRy677LJMmDAhv/3tb3Psscfm3nvvnWedP9fyyy9f7fLzVao+8PHD9wYAAAC1RzAOAABA6tev/5P3R64KSg877LDsu+++cz0/vzCxXbt2efjhhwuPl1122ZoV+v/XrFmzrLbaarnooovmem5BZgrPT8uWLTN58uRUVlYWAvkF9cPQc+rUqUky36C9WbNmWXvttXPbbbf96PMNGjTIwIEDM3Xq1Fx99dVp3bp14blJkybNFcbXr18/d999d2655ZaccsopefTRRwsfFmjcuHHGjRs31za+++67wvmrCmd/eFnvH4a3v/rVr3L33Xdn2rRpefXVV/P73/8+jRo1yqabbppXX301n3/+eTp06LBQlxr/MfXr1y/8f+PGjecK0d95550899xzhSsbvPPOO/n6668Ll6Cfn6pZ/CussEL69u2bAw88MKeffnquu+66wrm/+eabc8stt8xzjN133z0XXnjhAu1L69atM3PmzHz55ZfVLh0/atSoJMkGG2ywQOMAAACw8H7eXwsAAABYaiy33HJp06ZNRo4cmXXXXbfav5kzZ2bFFVec52uXXXbZauvP7z7dC6JDhw4ZOXJkVl999WrjVlZWFsLOKvObjfxj2rRpk9mzZ+eNN94oLJs2bVoOPvjg/Otf/5rva1955ZVqj997772ssMIKhfuCz2tfvvzyyzRt2rTavtSvXz8rrrhi6tWrl1mzZiVJtdnwb775ZkaNGjXX/q233nrp2LFjLrnkkiTJaaedVlhn/fXXz6hRo/Ltt98W1v/6668zevTowuN27dqlXr16hdngVaqOR9Wl27fYYouUlZWlf//+mTZtWmHGdufOnfPqq68WLrP+c62zzjr5+OOPkyS//OUvM3ny5PzjH//I5MmT8+yzz+aaa65J+/bt8+2332bMmDE57bTT8oc//GGhA/lNNtkk559/fp566qnceOONheUHHHBAHn744Xn+O+GEExZ4G1tvvXXq1auXZ555ptrygQMHpm3btlljjTUWqmYAAAAWnGAcAACABXb00UfnP//5T/r27ZsRI0bk448/zuWXX5699tor77333mKr45BDDsmUKVPSp0+fDB06NKNHj859992XPffcs3BJ+KqZ1K+++mref//9TJ8+fYHG3mGHHdK6deuce+65GTp0aD755JOce+65ef/999O+ffv5vvall17KgAED8tlnn+WBBx4o3MN6fvbee+80b948vXv3zhtvvJExY8bkiSeeyL777pu+ffsmSSF0vuWWWzJmzJgMHDgwF154YX79619n9OjRGTlyZCoqKqqN27Jly1xxxRUZNGhQ+vXrl2TO7Oby8vKcf/75+fjjj/P222/n5JNPrvahhpVXXjl77bVXbr311jz++OMZPXp0/vOf/+TSSy/Nlltumc022yzJnHu1d+7cOf369Uv79u0Ll93v3LlzXnvttQwbNizdunVboGM+P927d8/LL7+cL774IhtssEH+9Kc/5bbbbkv37t3z8MMP589//nP233///O1vf8sBBxyQPffcM0cffXSNtrX33nvnwAMPTN++fTNw4MAkc24z8MMPgnz/X9WxmzlzZsaOHZuxY8cWZtdPmDAhY8eOzfjx45Mkq666ag466KD85S9/yTPPPJPPP/88t912W5599tmcdNJJP/tYAQAAMG8upQ4AAMAC22233VKvXr3cdtttueWWW9KgQYNsuummuf3229OuXbvFVse6666bu+66K9dcc00OOeSQzJo1K61atcppp52WAw88MEmy0kor5aCDDsoDDzyQ5557rtql3OenYcOG6devXy699NL06tUrFRUV2WSTTdKvX79ql7/+MaeddloeffTRXHLJJalXr1722GOPnHjiifN9TYsWLfL3v/89V155Zf7whz9k6tSpWX311XPooYcW7sveqVOn9OnTJ3fddVf+8Y9/ZNNNN81VV12VCRMm5Pjjj88BBxxQCHK/75e//GWOOOKIXHXVVdliiy3Srl27XHHFFbnuuuuy5557Zs0118wJJ5yQBx54IDNnziy87vzzz0/Lli1z5ZVXZuzYsVlhhRWy4447pk+fPtXG79atW1588cVq9/Pu1KlTxo8fn+bNm2eTTTb5qcP9k1q3bp1dd901p5xySm699dbss88+c90/fM8998yee+6Zb7/9dq5Lyy+sM888M8OHD88pp5yS/v37p02bNgv0urfeeiuHHHJItWW/+93vkiRrrrlmYZb4GWeckaZNm+b888/P+PHjs9566+Waa67Jr3/9659VNwAAAPNXVrmw15QDAAAAWIymTp2aY445Jl999VWOOeaYbLPNNmnZsmXKy8vzzTff5K233sqTTz6ZoUOH5umnny7MXgcAAIAqgnEAAACgzisvL8+AAQNy33335b333kuDBg0ye/bsNGjQIO3atcvOO+9cuCw9AAAA/JBgHAAAAFiiTJ8+PRMmTMgyyyyTFi1apEEDd4oDAABg/gTjAAAAAAAAAJS0esUuAAAAAAAAAAAWJcE4AAAAAAAAACVNMA4AAAAAAABASWtQ7AJqw+zZs/Ptt9+mUaNGqVdP1g8AAAAAAABQ6ioqKjJjxow0b948DRrMP/ouiWD822+/zahRo4pdBgAAAAAAAACLWatWrbLiiivOd52SCMYbNWqUZM4ON27cuMjVQO0rLy/Phx9+mDZt2qR+/frFLgcoIv0AqKIfAFX0A6CKfgBU0Q+AKvoBpW7atGkZNWpUIS+en5IIxqsun964ceM0adKkyNVA7SsvL0+SNGnSxDcuWMrpB0AV/QCooh8AVfQDoIp+AFTRD1haLMjttt2QGwAAAAAAAICSJhgHAAAAAAAAoKQJxgEAAAAAAAAoaYJxAAAAAAAAAEqaYBwAAAAAAACAkiYYBwAAAAAAAKCkCcYBAAAAAAAAKGmCcQAAAAAAAABKmmAcAAAAAAAAgJImGAcAAAAAAACgpAnGyc4775wBAwbU2nrUXdttt13uvffeJMnvf//7XHnllUWuqO55+eWXs80222TXXXctdikl7aSTTsrpp59e7DIAAAAAAIClRINiF1BMMy69bbFtq9EZRy62bS2sp556qlbXq2v+ff9ei3V7O+7z0GLd3pJm74HXLNbtPbjDSQu1/l//+td06NAh1157bZLk4YcfzgUXXJDf/e53OfnkkxdBhQtm/yffXKzb679Lp8W6vSXBu9dMXazba3dSk4Vav23btrntttuyzTbbzPXcww8/nL/85S955plnaqs8AAAAAABYohR1xnjbtm3Trl27bLrppoV/f/rTn5IkgwcPzj777JNOnTqlR48eefTRR4tZKrCUmDx5ctZZZ53Uq1cvF1xwQe6+++6sscYaxS4LfpY999yzWih+//33Z/z48Ytse08//XQ+/fTTRTY+AAAAAAAsrKJfSv1f//pXhg4dWvh3zjnn5Jtvvsmxxx6bAw44IIMHD85ZZ52Vc845J0OHDi12uYvNmDFj0rZt2zz11FPp0aNHNttssxx88MEZO3ZsXnnllXTs2DH9+vVLp06d8tZbbyVJ7r777uyyyy5p3759evTokYEDBxbGmzZtWs4555xsueWW2WqrrXLOOedk5syZSapfXvvtt9/Ofvvtl44dO2bLLbfMWWedlenTp8+1XkVFRW644YbsuOOO2WyzzbLXXntl8ODBhe1tt912GTBgQI466qh07NgxO+ywQ1544YXFcuyWNG3btk2/fv3SrVu33HrrrUnmfDBk//33T8eOHbP11lvnpptuqvaaO++8M7/+9a/TqVOnHH744RkzZkySpLKyMldeeWW6d++ejh07Zq+99sprr7222PdpSXXwwQfntddey5133pmdd945q6++ev7+97+nZcuWxS6tzmvbtm2efvrpHHjggenQoUN23333vPfee4Xn77vvvmy33XbZfPPNc8EFF6SioqLa6+f1ni4vL8+FF16Yjh07Ztttt80///nP7LTTTnnwwQcX6/6VkvLy8lx22WWZMGHCItvGX/7yF8E4AAAAAAB1StGD8R/z2GOPpVWrVtlnn33SqFGjdO3atRC0Lm3uvvvu3HnnnRk0aFDKyspy/vnnJ0lmzZqVTz/9NC+99FI6dOiQp59+Otdff32uuOKKvPHGGznhhBNy4okn5osvvkiSXH311fn444/z5JNP5oknnsiwYcNyww03zLW9U089Nfvuu2/eeOONPPbYY/nggw/Sv3//uda75557MmDAgFx//fV5/fXXs/vuu+fYY4/NuHHjCuvccccdOf744/PKK6+kS5cuueSSSxbNQSoBAwcOzMMPP5wjjzwyX331VY499tgceOCBef3113P77benf//+efHFFwvr3nbbbbnpppvy8ssvZ/XVVy9c4vuRRx7Jww8/nP79++f111/P9ttvn969e6e8vLyYu7fEuPvuu7PFFlukV69eeeqpp3LUUUelYcOGxS5riXH77bfn4osvzuDBg7PKKqvkmmvmXDb/k08+ybnnnpszzzwzgwcPziabbJLnn3++8Lr5vafvuuuuPPnkk7nvvvvy6KOP5sknn8w333xTlP1bEowdOzaHHnpoNttss+y666758MMPkyQPPvhgfvWrXyVJunTpku+++y577LFHrr/++p8cs2/fvjn66KNz4oknplOnOZfYHz9+fHr37p1f/vKX6dy5c4488sh8+eWXSZKePXvmo48+yrHHHpszzjgjSfL+++/n0EMPTefOnbPVVlvloosuyqxZsxbFIQAAAAAAgB9V9GD8qquuyrbbbpvOnTvnnHPOyZQpUzJs2LBsvPHG1dbbeOON8+677xapyuI56KCDsuqqq6Z58+Y57LDD8n//93+pqKjIrFmzctBBB2XZZZdNWVlZ7r///uyzzz5p165dGjRokJ122imbb755Hn/88VRWVubhhx9Or1690rJly7Rs2TKXXHJJIST5vkmTJqVJkyapV69eVlllldx333059NBD51rv/vvvz0EHHZS2bdumYcOG6dWrVxo3bpznnnuusM6vf/3rbLbZZmnYsGF23nnnjBo1aq5Zosyxyy67ZKWVVkpZWVkef/zxbLjhhtlzzz1Tv379tG3bNvvvv39hxv0DDzyQHj165Be/+EUaNmyYk046KYceemgqKiqy++6758knn8xqq62W+vXrp0ePHhk/fnzhAxKwKO2xxx5p3bp1GjdunO222y4jRoxIMif43njjjbPDDjukYcOG2WeffbL22msXXje/9/Tzzz+f3XbbLRtuuGGaNWuWPn36ZNq0acXaxTqvf//+Of/88/PSSy9lpZVWytVXXz3XOo888kjhv8cff/wCjTtkyJB06dKlcAWKK664IlOmTMl//vOfwoccqj78VHXrkxtvvDGXXnpppk2bliOOOCJdu3bNSy+9lAEDBuSVV17JHXfc8bP3FwAAAAAAFlSDYm68Q4cO6dq1ay6//PKMHj06J554Yi644IJMnDgxq666arV1W7Ro8ZOXfS0vL1+ombGVNaq6ZhZ2xm5VgLzuuusWXrvaaqtl5syZhfvCrrrqqoXnPvvss7z44ov561//Wm2M1q1bZ9y4cZk0aVJWX331wvobbrhhoa7KyspUVFSkvLw8J554Ys4888zccccd6dq1ayHoSlJtvTFjxmS99dartl9rr712Ro8eXRhzzTXXLDzfsGHDlJeXZ/r06WnUqNFCH7+fo3KxnumFP9fJnHNb9bpPP/00Q4cOzaabblp4vrKysrDOZ599li5duhTWb9GiRXbaaadUVlZm8uTJueSSSzJo0KBMmjSp8Prp06fPda4rKytTWVlZhNnkdft8/NhxKd6x+n5di3d7NdnX7/eYRo0aFd53X375ZbV+kMzpbVXHdH7v6W+++SbbbLNN4bl11lknTZs2LbyPF6s6cg6qlv/Y8z179sw666yTJNl2223Tv3//lJeXp6KionC8q76/LOgxrKioSP369bPffvsVtnvuuedm9uzZhX6+3Xbb5ZZbbqk2XtX4zzzzTCoqKnLEEUckSdZYY4306tUrt956a4488sgFPRzAj5hfPwCWLvoBUEU/AKroB0AV/YBStzDv7aIG49+/RPf666+fk08+Occcc0w233zzGo1XdcnYBbXB9Bk12k5NDBsyZKHWHzt2bJJk+PDhhdmRo0aNSpLCfVuHDRtWuMxzRUVF9t9///To0WPubQ8blmTOpWxnzJh7n2fOnJkxY8ZkyJAh2WCDDXLdddfljTfeyBtvvJF+/frl+OOPzxZbbFFtvRkzZmTkyJEZ8r39mjJlSr7++usMGTIkM2fOzOeff154/uOPP04y5x7mi/vS1D+2z4vSkIU818mcc1r1uu+++y7t27fPKaecMtd6Q4cOzYwZMzJ69Ogf3c6NN96Yzz77LGeeeWZWW221fPPNNznppJMyfPjwTJo0qdo5nDx5cuF8LU51/Xz82HEp1rH6vhkzZi7W7dVkX0eNGpXmzZsnmfNhndmzZ2fIkCH58ssvM2nSpGpjTpgwITNmzCj0k3m9p6dNm5avvvqq2nMVFRX57LPPFv/5mLHeYt3ckCHD5/v80KFD51pWdUyTOd9HJk+enCFDhlQ7H9///vL9D9DMy1dffZVmzZrl7bffLiwbM2ZM7rrrrowYMSKzZs1KRUVFmjZtWu2cfPLJJ1l++eXz6quvZvz48Wnfvn3hucrKyiyzzDJF/ZqCUvJj/QBYOukHQBX9AKiiHwBV9AMocjD+Q2uttVbKy8tTr169TJw4sdpzEyZMSMuWLef7+jZt2qRJkyYLvL1ZT79ZkzJrpEOHDgu1/ueff54kady4ceG148aNy7LLLlt43L59+8JsvbZt22by5MnVtvPFF19k9dVXT1lZWZo1a5aGDRsWnn/vvffy8ccfp2fPnmnYsGHWWmutdOjQIRMnTkyLFi2y9dZbJ0luuOGGvPXWWznyyCOrrVc1I7FqvNmzZ2fcuHHp0qVLOnToUG3dZE74/sOaF5dxIxfv9hb2XCdJ69atC68bNmxY/vd//zft27dPWVlZkuTrr7/OZ599lk6dOmXDDTfMzJkzC+uPHz8+jzzySA4++OCMGTMm++67b3bZZZckyb/+9a8kyUYbbZTWrVtXOy9NmzbNqquuWqN6f45Gzz7/0yvVooXdvx87LsU6Vt/X6Ou3f3qlWtShQ/ufXukHvv8+HjlyZBo0aJAOHTpko402ynPPPVft+I0bN67wXpzfe3qNNdZIvXr1Cs99+umnmTp1atZZZ53Ffj6GP794P9Sx0Tz2r7y8vHBVifr161d7bv311y8cl/fff7/Q979/Pqq+v1T1hZ/ywgsvZPnlly+MW1FRkZNPPjmbb755brrpprRs2TIPPPBArr322mrnpOr9MHTo0GywwQaFS7gDtWd+/QBYuugHQBX9AKiiHwBV9ANK3dSpUxd48nTRgvH33nsvjz76aE4//fTCshEjRqRhw4bp3r17HnrooWrrv/vuu9Vmm/2Y+vXrL9QX9eyFK/lnWdhmU6/enNu/9+/fP126dEmDBg1y1113pXv37oWxvr+/BxxwQP7whz9k1113Tbdu3fL666/n2GOPzZ133pkOHTpk7733zp133pktt9wyDRs2zMUXX5zOnTunfv36KSsrS7169TJ27Njssssu6du3b7p27ZopU6bk448/zrrrrlttvfr162fPPffMvffem+233z5rr712brvttpSXl2f77befa93v78/CnqPaUJayxbq9muzf94/V7rvvnmuvvTa33HJLDj/88IwdOzbHHHNMOnfunC222CL77LNPTj311Oy7777ZaKONcvPNN2fo0KE54ogjstZaa2XYsGEpLy/Pe++9lyeffDJJ8t///jcbbrhhtfNSVlaWsrKyInwjrNvn48eOS/GO1ffrWrzb+7nv43r16hWO2bbbbpubbropgwYNSteuXTNgwIB8/fXXhefn957eaqut0r9//+y3335ZaaWV8pe//CVNmjSptq3Fpo6dgx/rp/M6Bz/8/x+uOz/ff20y50MNX3zxRa677rqsvPLKSebMPv/h10jV+Ouuu27GjBmT6dOnZ7nllksy58NuyyyzTJo2bbqARwOYn2L8fAXUTfoBUEU/AKroB0AV/YBStTDv66IF4yuuuGL69++fli1b5rDDDsvnn3+e6667Lvvvv3/22GOPXH/99RkwYEB69uyZl19+Oc8//3y1S68vLXr27JlDDz00n332WTp06JDzzjuvcFny7/vVr36V0047LRdeeGH++9//Zq211sr5559fmL3Xp0+fXHTRRdl1113TsGHD7LDDDjn++OOrjbHaaqvl4osvzsUXX5wvvvgiTZs2zTbbbJPevXvPtb1evXplwoQJOfLIIzNp0qRstNFG+dvf/pZmzZotkuOwNFlhhRVy44035s9//nNuvvnmtGzZMrvvvnu6d++eJNl+++1z0kkn5bjjjsvUqVPTsWPHXHXVVUnmnOdTTz01Xbp0Sfv27fPnP/85SXLsscfm7rvvLto+LayZEypqbax3r5m6UOtPGV2esTNn5ZkLP07vf+yVJJldPiuvv/ZG+t3516y59hp56qmnaq2+pUH79u1z9tln5/zzz8+kSZOy++675ze/+U0q//83Tp/fe/qII47IiBEjsscee2T11VfPGWeckVdffbUQ7rLwll122SRzLn2/6qqrLnQ43bJlyzRp0iRDhgxJ27Zt8/TTT2f48OGZPHlypkyZkuWWWy6NGjXKp59+msmTJ6dbt25p2bJl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POkxdZt+UILq7ildes7BEko/KEF195QuPedgCaUflKA6ekqXmuO/hFoaesH3x1oq3o9195QuHcd/CaYflKC6e0qXjuO/BNMPSlDdPaVLx/EvkoU5tkULxqt8/vnn2XnnnTN79uzst99+6d27d4488sisuuqq1dZr0aJFJkyYMN+xPvzww0VZKnXUjBkzil3CPA0ZMqRWx6vN2wnMmDGz1saqbTNmTC92CfM0ZMjwYpfAfOgHJWjGesWuYJ70g7pNPyhB+gE1pB+UoDraD/SCum1p6gXJUtIP6mgvSPSDuk4/KEH6ATWkH5Qg/YCfUPRgfM0118zQoUPz6aef5txzz82pp55a47HatGmTJk2a1GJ1LAnGjWxU7BLmqUOHDrUyTnl5eYYOHZpNN9009evXr5UxG31dd2eMN2q0bLFLmKeNaumcsmjoB6Vn+PN19xcU/aBu0w9Kj35ATekHpaeu9gO9oG5bGnpBsnT1g7raCxL9oK7TD0qPfkBN6QelRz9YOk2dOnWBJ08XPRhPkrKysrRq1SonnXRSDjjggHTv3j0TJ06sts6ECRPSsmXL+Y5Tv379kv+iZm5lKSt2CfNU2+/H2nyPl9Xdw5Y6fEr1mDpOPyhBdfeULh3HfwmmH5SguntKl47jvwTTD0pQHT2lS8WxX4ItTb2gasySf0/W3VNa+sd+CacflKC6e0pL/9gv4fSDElR3T2npH/siWphjW28R1jFfgwcPzs4775yKior/r5h6c8rZbLPN8u6771Zb/91330379u0Xa40AAAAAAAAALPmKFoy3a9cukydPzhVXXJFp06Zl/Pjx6du3bzp37pwDDzwwn3/+eQYMGJAZM2bk+eefz/PPP5/99tuvWOUCAAAAAAAAsIQqWjC+/PLL584778y7776brbbaKj169Mjyyy+fq6++OiuuuGJuueWW3H333dl8881zySWX5IorrsgvfvGLYpULAAAAAAAAwBKqqPcYb9u2be66664ffW6LLbbII488spgrAgAAAAAAAKDU1CgYnz17dkaPHp3x48ensrIyLVu2zDrrrJMGDYqaswMAAAAAAADAXBYqyR44cGDuvffevPnmm5k2bVq15xo3bpxOnTrlwAMPzA477FCrRQIAAAAAAABATS1QMP7ZZ5/lpJNOytdff52ePXvmkEMOyYYbbpgVVlghZWVlGT9+fD766KO8+uqrOf/883PTTTfl2muvzdprr72o6wcAAAAAAACA+VqgYPzAAw/MUUcdlQMPPDANGzac6/k11lgja6yxRrp3754TTjgh9957bw488MC88MILtV4wAFC79n/yzWKXME/n5BfFLgEAAAAAgBKwQMH4Pffck1atWi3QgA0bNsyhhx6abbfd9meUBQAAAAAAAAC1Y4GC8R+G4qNHj85ll12W119/PVOmTMlyyy2Xjh075vTTTy+su+6669Z2rQAAAAAAAACw0OrV5EXnnXdeevTokf/85z9555138sQTT2TrrbfOiSeeWMvlAQAAAAAAAMDPs8DB+GmnnZZvv/02STJ58uRss802adq0aerVq5cVV1wxO+64Y7788stFVigAAAAAAAAA1MQCXUo9Sdq0aZO99torvXv3zr777ptddtklHTt2TJMmTTJx4sS88847Oe644xZlrQAAAAAAAACw0BY4GD/88MPzm9/8Jn/6058yffr0XHPNNZk0aVK+++67NG3aNBdeeGFWWWWVRVkrAAAAAAAAACy0BQ7Gk2TNNdfMzTffnKeeeipnnHFG9t577xx55JFp0GChhgEAAAAAAACAxWaB7zH+fTvvvHMefvjhjBs3Lr/97W/z1ltv1XZdAAAAAAAAAFArFniq98iRI3P99ddn+PDhKSsry2abbZbjjz8+e+yxR84///y0a9cup5xySpo2bboo6wUAAAAAAACAhbLAM8ZPOOGEtG/fPn379s11112XVq1a5Y9//GM23XTTDBgwIOutt15++9vfLspaAQAAAAAAAGChLfCM8S+//DJ77713YUb4SiutlDvvvDNJUq9evRx22GHZZZddFk2VAAAAAAAAAFBDCxyM77ffftlzzz3Tvn37VFRUZMiQIfn9739fbZ1VV1211gsEAAAAAAAAgJ9jgYPxU045JXvttVc++OCDlJWVpXfv3llvvfUWZW0AAAAAAAAA8LMt0D3GTz/99EyZMiUbbLBBevTokV133XW+ofiUKVNy+umn11qRAAAAAAAAAFBTCxSMN2vWLLvuumvuuOOOTJw4cZ7rffvtt7nzzjvTo0ePtGjRopZKBAAAAAAAAICaW6BLqZ955pnp1q1brrvuulx11VVp27Zt2rRpk+bNm6esrCwTJ07MRx99lA8++CAbbbRR/vSnP2Xrrbde1LUDAAAAAAAAwE9a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AAAAAMASpM7eYzxJvvrqq5xzzjl59dVX07Rp0xxwwAE5/vjjU1ZWVuzSAAAAAKCoJk2alGbNmhW7DAAAWCLU6WAcAJZWH374Ydq0aVPsMoAiGzVqVJo2bZqVVloplZWVPiAKpLy8PPXr1y92GUARDRs2LAMGDMjw4cNz6KGHZtdddy12SUCR+fkAlm7l5eUpKytLvXp19iLRUGcIxqGO8AMsUOWFF17IxRdfnIMPPji/+93vil0OsJhNnz49yy67bD766KPcfvvtmTZtWk455ZSsvfbawnEgSfQCWIpUfb1PmjQpf/vb3zJo0KCUl5dnu+22y5577pk11lij2CUCRfBjPwuUl5ensrIyDRo0KFJVwOIyr98HJkyYkGbNmskZYD58fATqiKpvVl9++WWRKwEWt/Ly8lRUVBQed+vWLddee20efvjh3HjjjZk6dWoRqwMWh0mTJuWvf/1rDj744Jx55pkZOHBgNtxww1x++eVZZZVVcsopp2T06NGCMFhKTJs2bZ7f/59//vnstddeSeb8QQwoXd//o/dDDz2Ue+65J4cddljuv//+HHvssUJxWMp89913hZ8Pfvh7wUsvvZRevXpl8uTJxSgNWIzGjh2b//u//0uSwt8TBw4cmF69emWfffbJ6aefnrfeequYJUKdJhiHxayioqJaAFa17K677sree++do446Kn369MnXX39dpAqBxa1+/fqpV69exo8fn/HjxydJ2rZtm+OPPz5DhgzJ3//+9yJXCNS2qjDryy+/zKGHHpqDDjoo7733Xo4//vi0a9cul156ad58880kydlnn53VVlstV1xxRSZOnFjEqoFFafz48YXfAV599dV88MEHSZLRo0fn888/L6xXVlaW5ZdfPjNmzPBhGShRzz33XE444YScd955eeCBBzJ+/PhsueWW2WKLLTJjxoxq606ZMiWffPJJkSoFFrWvv/46n332WZI5V5d76aWXksz5PeKNN94orNeiRYu8//77adGiRTHKBBahysrKannCd999l0ceeSRPPPFExo0bl+nTp+f+++/PzjvvnP/85z9p1apVbrzxxgwbNqyIVUPdJRiHxaxevXqpV69eysvLC8s+/PDDPPfcczn55JPz2GOPpX79+unbt28+/fTTIlYK1LaKiopqM7uq+sDgwYOz//7756CDDsqFF16Yp59+OsmcmeM77LBDnnjiCR+WgRIyZcqUQpg1ZcqUvPfeeznllFNy+eWXZ6uttspee+2VTTbZpNovsX/84x8zduzY/Pvf/y5W2cAiNGrUqNx8882FD78MGjQoV111VS644IKccMIJufbaawvh+KhRo7LOOutk9uzZRawYWBSGDh2a/fffP3fccUe22mqr/OIXv8hjjz2Wt99+O61atcrqq6+eIUOGJEmeeeaZ9O7dOz179swbb7yRWbNmFbd4oNaNHz8+9913X8aMGZNkThh+8cUX5/e//33uueeenHbaaRk3blySObdj6tSpk78lQgmpCsR/eO/wf/7zn3niiSfSt2/fVFRU5Kmnnsqnn36a/fffP8mcyTbDhg3LG2+84QpT8CME47CI/PCTXFVefPHF9O7dO7169coTTzyRJHn00Uez3HLLpWvXrkmSddddN4MGDSrMEgGWbFU/hNarV68Qhn344YepX79+Jk+enDvuuCMHHHBA/vWvf2XbbbfN7bffnrfffjv169fPVlttleWXXz6PPPJIMXcB+Jk+/fTTXH311dl7771z9tln5/rrr8+4ceOy/vrrZ4cddsijjz5aWLeysjITJkzItttuW1i2/vrrZ+ONN84777yTr776qgh7ANS2739QtlWrVjnzzDPTuHHjvP7663nrrbcycuTIbL/99rnlllsyefLk/PnPf06SrLbaahkyZEiWW265H/19A1gyTZ8+PXfddVd23HHH3HXXXTnwwANz0EEH5eabb84WW2yRZZddNr/4xS/yxhtvpHv37unfv3+6d++eJ554Ivvuu2+WWWaZYu8CUAu+//NBy5Yt88c//jHNmjXL2LFj88QTT2Ty5Mn57W9/m5NPPjnrrrtuLrvssvz3v//NrFmzMm3atKyyyipFrB6oTVWB+Ndff53bbrst119/fT755JNstdVW2WmnnbLNNttk1VVXzSqrrJIZM2bkoosuyh577JH//d//zRlnnJGDDjrIFabgRwjGoZbN65NcSTJgwIBcffXV2XnnnXPkkUdmypQpmTBhQtZaa618+OGHOeuss7Lrrrtm6NChueqqq7LTTjsVaS+An+v7n8is+iH03XffzQ033JB77rknvXv3zn//+98MHTo006ZNy+67754kWWGFFTJixIg8+eSTSeb88ftXv/pVnn/++cW/E0CtuPnmm3P88cdn9uzZueiii9K1a9dMnTo1jRs3TpJsv/32ee211/L+++/n6quvzh577JEpU6bk1ltvzcMPP1wYp0uXLvnmm2/MAoElVGVlZbWfD+rXr58kefPNN/PUU0/lH//4R84+++xMmzYtF110UTbffPNMmTIlK6+8ck466aR88sknue2227LOOutk5ZVXzqRJk+b6fQNYck2cODEvvPBC4YNxM2fOTJIsu+yyadq0aZKkXbt2WXvttdOzZ8/ccsst+e1vf5tGjRr5kAws4b7/NVz188F///vfJHP+jvD//t//y2uvvZZ+/fqle/fuhavInHrqqSkvL8/ll1+eNm3a5OOPP86yyy67+HcA+Fl+7NaryZzfH/r165dDDjkkI0eOTMOGDfPoo49miy22yNZbb51PPvkkn376adZaa62stNJKGTVqVO65557cc8892W233XL77bdX+7ANMEeDYhcAS7pZs2ZV+2R2WVlZysrK8vXXX+fJJ59MkyZN0qNHjzRs2DDDhw/P5ptvnh49eiSZcz+Q5ZdfPhtuuGHKy8uz/PLLF2aRf/rpp/nb3/6WQw45pCj7BSy8qkul169ff65PZN5888257777csABB+Ttt9/O119/nWHDhqVRo0aZNm1azjvvvAwdOjTNmzfP2Wefnd122y1J0rBhw7Ru3TqDBg3KmDFjstZaaxVj14AFUFlZmbKysrzyyivZZJNN0rRp0wwZMiQDBw7MHXfcUZi9sfHGG1d7Xfv27dOyZcsceuih6dWrV/73f/83G2ywQf7617/mnnvuybrrrpuOHTtmiy22yE033eSP37CEqfqa/WGI/dJLL+Xee+/Ncsstlx49emTDDTfMSy+9lJEjR2brrbdO48aN88Ybb2TrrbdOmzZtcuyxx+aee+7JSy+9lHbt2rksIixhqj5EXxV6/dCbb76Ztm3bFn6PaNiw4VzrrLvuumnTpk1Gjx6dKVOmZLnllktlZaUPycASZvTo0Vl++eVzxBFH5Nprry38nv/f//43//73v3Pddddlww03zL777puePXvmV7/6VQYNGpRdd901bdq0yfPPP5/jjjsubdu2zcknn5x99tknN998czbaaKOMHj0666yzTpH3EPgp77zzTu65556UlZXlvPPOK3xw/s0338yqq66aNddcM7Nnz87jjz+eK664Iptttlm116+33npZZpll8sorr2S//fbLDjvskCeffDKNGzfOhAkT8te//jWTJ0/2MwL8CF8VUEPPPPNMjjrqqBxxxBHp169fvv322yTVP8n1zjvv5JFHHskFF1yQevXqZf3118/gwYNz3HHHpVu3btlrr71y7bXXpqysLDvttFNhBtinn37qD9+whPj+vfzq1auX+vXrZ8aMGXn22WczcuTIJHPuITxs2LAcccQROeqoo/LHP/4xO+ywQx577LFsscUWmTx5cqZPn57rrrsud911V/baa6/84x//yGeffVYYd/nll/cpT6jDqq4W89Zbb6V3796Fe/8+++yzad26dVq2bFn4Gq6oqMhjjz2WP/zhD+nXr19WXnnl/PKXv8wmm2ySo48+OhtssEGS5JBDDsmIESMKfyCvGqPqHoNCMaibZs2alTfffDMTJkxIMuf7eL169TJ27Ng8+uijef/995PMCbgGDx6cBg0aZOutt07r1q2z5ppr5r333suUKVPSuXPnfPXVV4XbK+2yyy7Zf//9M3jw4Lz++utp3ry5PgB13A+vIvVjoXjVzwyrr756xo0blylTpvzoWE8++WSmTJmSTp06ZfLkyXnrrbfm2gZQd02dOjWTJ09O3759s8suu2T69Om57bbbsvrqqydJ+vXrl169eqWioiJ///vfs/vuu+fCCy9MkmyxxRb54IMPMn78+Gy11VaZPXt2XnzxxSTJGmuskXPOOSfPPvtsvvzyy6y66qpF20dg/saMGZPrr78+O+20U84444y0bt06J510Uho3bpzbb789PXv2zNlnn50TTzwx//rXv7LMMsukcePGOeuss9K3b9/8z//8T0499dQ899xz6dy5c9Zcc80MGTIkSXLUUUcV/qbwu9/9Lp9//nkOPvhgl1KHHyEYh4UwZcqU9O3bN927d8+AAQPym9/8Jn/84x/Tv3//3H///UmSCRMm5G9/+1uuu+66XH311Tn11FPzxRdf5IUXXsjvfve7XHvttfnNb36Tq666KpdddlkGDRqUt99+O3/4wx/SokWLHHTQQTnxxBOz4oorZs899yzuDgPzNXjw4MIfq5Nk2rRp+fOf/5wePXrkhhtuyJlnnpmBAwcW/hjesmXLzJ49O2uvvXa23nrrjBo1KhMnTkzPnj2TJN9++20qKipy8cUXZ9SoUVljjTWSzPnD+fDhwwu/MAN1w4QJE3L33Xcn+f9mgv773//O3nvvnRYtWiRJPvrooyy77LJp0KBB6tevn5EjR+aggw7K448/nhYtWuSBBx5Ikmy33Xb54IMPMm3atCRzLp967733pnv37ll33XUL22zfvn2mT5+eJH7BhTqm6kOtzz//fM4777xCAD516tSceuqp+Z//+Z888sgjOfLII/Pqq69mzTXXTOfOndOoUaNMmjQpSdKpU6dMmDAhQ4cOzVZbbZVp06Zl2LBhGTduXJJkt912y7HHHpt69erl22+/1Qegjqq6dcL3v0bHjBmTSy65JJdcckm++eabJHPuJdygwZyLOXbs2DENGjTIq6++msmTJyf5/y6n/umnn+a+++7LZ599lvXXXz9lZWV57bXXksx9NQqgbpk5c2YhCBszZkwmT56cffbZJ7Nnz86nn36a9u3bJ0latWqVTz/9NBtuuGFat26dXXfdNY0aNcrAgQOz1VZbpWnTpvnPf/6TjTfeOGussUaefvrpwofwdthhh5x44omFnxeAumXWrFk57bTTcswxx2TKlCmZOXNmDj300Bx99NFZddVV88477+SZZ57JpZdemieeeCI77LBDrr766iTJpZdemsMPPzxfffVVOnTokDXWWCPHHHNMvvzyy2y55ZZ57733st1222XgwIG56KKLcu655+af//xnrrjiirRq1aq4Ow51lEupw0IoLy/PkCFDsv322+fcc88tLN9qq60K9/6ZNWtWzjvvvGy44YZ58cUX849//COffPJJnnnmmXTv3j3rr79+1ltvvcIvr2ussUbatm2b5ZdfPpdeeqlLJUMdNXr06Dz22GNZe+21s8kmm6R169b5+uuvc8UVV2S77bbLLrvsku+++y7PPfdc/vnPf6ZRo0a57LLL8vjjj2f77bfPiiuumPfeey/dunVL06ZNs8oqq2TixIl59tlnc8wxx+Suu+7KX/7yl3z++efZcsstc/DBBxf+SDZjxoxssskmmTnz/9fefYZVdW0LH/9vuiJVpYmKoICIYAGRKE1RBGuwd01iSzwaG8aSkxg1atQUo8ZoYom9RBB7r9hAVESFABaKCiiIonR4P/DsFbGce+59zxFMxu9Lwt57tTxZa805xxxjFr62pKIQonIkJSWxefNmLl++TO/evWndujX37t2jadOmym9at27Nxo0blb8bNGjAL7/8Qo0aNSguLqZbt25cunSJli1bUrduXaZPn46J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
)
    



```python
# Ft imp for metaranker
imp_df = pd.DataFrame({'feature'   : STAGE2_FEATURE_COLS,
                       'importance': meta_lgb.feature_importance(importance_type='gain'),}).sort_values('importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data = imp_df, x = 'importance', y = 'feature', palette = 'viridis', ax = ax)
ax.set_title('Meta-ranker feature importance (gain) — S6 LightGBM')
ax.set_xlabel('Information gain')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print(imp_df.to_string(index = False))
```


    
![png](data:image/png;base64,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VKpUiYEDB9KhQwfTsbp161KtWjW8vb1p3rw5NWrUoGvXrmzdupVx48YBYGNjQ8OGDXFwcDB7/n3SpEls376dihUrUrlyZfbs2WP2KjYRERERERF5elbG1DwMLakSFxdH1qxZTYl2//79SUxMNCW+Ga1///4UKlSIXr16PdX5d+/e5dixY0xYFsrF67f4ZVban08Xy0lISODw4cO4u7ubLegnGU99k7mpfzI39U/mpv7JvNQ3mZv6J3NLqX8e5kOurq44ODikur4MfWXYy+Tu3btUqlSJhQsXkpiYyB9//MHPP/9M9erVMzo0AH7++Wd27NhB69atMzoUERERERGRV0aGvjIsM2nUqJHZYmz/FRwcTLly5VI87uDgwHfffcf48eMZN24cefLkoX379vj4+Fi87SepV68ecXFxjB07ljfeeOOp6xEREREREZG0UdL9/61evfqZ6/Dy8kr1O7/Tu+3H2bhxo0XrFxERERERkeQp6ZY0+35Qe3LmzJnRYYiIiIiIiGR6eqZbRERERERExEKUdIuIiIiIiIhYiJJuEREREREREQtR0i0iIiIiIiJiIUq6Jc06jA+mcvcRGR2GiIiIiIhIpqekW0RERERERMRClHSLiIiIiIiIWIiSbhERERERERELUdItIiIiIiIiYiFKul9xtWrVYtGiRRkdhoiIiIiIyEvppUu6N2/ezPnz5zM6DBEREREREZGXL+mePHmykm4RERERERHJFF7YpDsiIoL27dvj4eFBzZo1mT9/Po0aNeLUqVMEBAQwcOBAIiMjMRgMLFy4kPLly7N27don1jtgwABGjBjB6NGjKV++PBUrVmTmzJmm4zdv3qRfv354eXnh4eFBp06diIyMBEi2vZUrV9KwYUOWLFlClSpVKF++PAsXLuSXX37h/fffp2zZsgwbNizV120wGJg7dy5eXl7MmDEDgNWrV+Pt7Y2Hhwe1atVi4cKFpvJTpkyha9euzJw5kypVqlCuXDlGjhyZbN1xcXG0atWKAQMGpDoeERERERERSdkLm3R3796dYsWKsXv3bgIDA5k0aRL9+/cHIDAwkNGjR5vK7t+/n23btuHj45OquteuXUuJEiXYtWsXffv2ZeLEiVy9ehWAwYMHc+3aNVavXs2vv/6Kvb09vXr1Mjv/v+1dvHiRqKgotm/fTrt27Rg3bhxr1qzhp59+IigoiMWLF3P06NFUX/vWrVsJCQmhY8eORERE0L9/fwYPHsyhQ4cYNWoUI0aM4Pjx46byhw4d4sGDB2zfvp3Jkyfzww8/EBYWlqTeYcOGYWtry4gRI1Idi4iIiIiIiKTshUy6//zzT06cOEG3bt3Ili0brq6uTJ06lYIFCyZbvnHjxuTIkQMrK6tU1e/k5MQHH3xA1qxZ8fb2JiEhgXPnznHjxg22bNlCr169yJMnDzly5ODTTz8lPDyciIiIFNuLjY2lY8eO2NraUrNmTe7evUuLFi3Inj075cuXJ2fOnGmaEl+/fn3y5cuHlZUVTk5O7N27l8qVK2NlZUWlSpXImzcvf/zxh6m8jY0NnTt3xtbWlkqVKpEnTx5Onz5tVufs2bMJDw9nypQpZM2aNdWxiIiIiIiISMqyZHQAT+PChQvkyJGD3Llzm/ZVrlw5xfKFCxdOU/1OTk6mr7Nlywb8mzhfunQJo9FIsWLFTMddXFyAf0ezH5733/Zy5cplqsfW1haAAgUKmI7b2dlx//79VMf3aP1WVlYsWrSI5cuXc/XqVYxGI3FxccTFxZmVt7b+v89XsmXLRmxsrGl7586d7Nixg9mzZ5MzZ85UxyEiIiIiIiKP90KOdFtbW5OYmJjq8jY2NmmuPzmPJrL/9ego+n/bS66+1I66J+fR+pctW8aMGTMYOXIkv//+O+Hh4UlG/FO6nod+//13qlevzsSJE0lISHjquERERERERMTcC5l0Ozs7c+fOHdNz1vDvc8779++3eLsAZ86cMe17+PXDEe/nLTw8HE9PTypWrIiNjQ3Xrl0zuy+p0aNHDyZMmEB0dDRBQUEWilREREREROTV80Im3a6urpQsWZJJkyZx584dTp48yaBBg4iNjcXOzo7z588TExOT7u3mzZsXLy8vvvvuO27cuMHNmzeZNGkSFSpUoFChQuneXmo4Ojpy5swZbt68ycWLFxk5ciSFCxcmKioq1XVYW1uTPXt2Ro8eTVBQEH/++acFIxYREREREXl1vJBJN0BQUBAXL16kcuXKdOnShYCAAKpVq0aLFi0YO3Ysffv2tUi7Y8aMwcHBgfr16+Pt7U2OHDn47rvvLNJWarRs2ZIiRYpQvXp1OnXqRJs2bWjTpg1z5sxhwYIFaaqrfPnytGzZkn79+j12Kr2IiIiIiIikjpXRaDRmdBDyYrh79y7Hjh1jzOpQIv++xe6pQzI6JHlEQkIChw8fxt3dPc3rGIhlqW8yN/VP5qb+ydzUP5mX+iZzU/9kbin1z8N8yNXVFQcHh1TX98KOdIuIiIiIiIhkdi/kK8Oe1ogRI1i6dGmKx7t27UpAQMBzjMicp6fnY18dtnHjRhwdHZ9jRCIiIiIiIvIsXqmke8iQIQwZknmnRB84cCCjQxAREREREZF09Eol3ZI+ZvdpT86cOTM6DBERERERkUxPz3SLiIiIiIiIWIiSbhERERERERELUdItIiIiIiIiYiFKukVEREREREQsREm3pNnH3wVT4fMRGR2GiIiIiIhIpqekW0RERERERMRClHSLiIiIiIiIWIiSbhERERERERELUdKdDi5evIibmxtnz57N6FBM3Nzc2LVrV0aHISIiIiIi8krLktEBvAwcHR0JDw/P6DDMZLZ4REREREREXkUa6RYRERERERGxECXd6SAyMhKDwcDp06epVasWixYtwt/fnzJlytCiRQsuX77M559/joeHB3Xr1uXo0aMArFy5kvfee49ly5ZRtWpV3N3dGTp0KA8ePEhVuzt27KBhw4Z4eHjg5eXFuHHjSExMBMBgMLBz507T1PdH/xkMBqZOnQpAbGwsX331FTVq1MDd3R1/f3/++usvy9woERERERGRV4ySbgtYuHAhX331FT///DORkZG0bt0aPz8/9u7di7OzsynhBYiKiiI8PJzNmzezYsUKtm3bxoIFC57YRnx8PL1792bgwIEcOnSIH3/8kU2bNrFt2zazcg+nvj/8N3XqVHLkyEGDBg0AGD9+PH/++SdLlixh7969uLm50b17d4xGY/reFBERERERkVeQkm4LqFGjBm+++Sb58uWjdOnSODs7U6VKFezs7PDy8uLcuXOmsvfv36dXr15ky5aNYsWK4ePjw44dO57Yxv3794mNjcXBwQErKyuKFi3K5s2bqVOnTornREVFMWDAAL788kuKFi1KYmIiK1euJCAggAIFCmBvb0+vXr24dOkSYWFh6XAnREREREREXm1aSM0CChYsaPrazs6OHDlymG3HxcWZtnPlykWePHlM24ULFyY0NPSJbeTIkYNu3brRpk0bSpcuTZUqVfDz86NQoULJlk9MTKRPnz7Url3bNMr9999/c+fOHQICArCysjIre/nyZcqUKZP6ixYREREREZEklHRbgLW19WO3H5WQkGC2bTQazRLgx+nevTtNmzZl69atbN26lVmzZjFv3jxKly6dpGxgYCA3btxg5syZpn329vYALF68mFKlSqWqTREREREREUk9TS/PYDExMURHR5u2L126RIECBVJ17o0bNyhQoACtW7dmzpw51KtXj1WrViUpt3//foKDg5k0aZIp0QbImTMnuXPn5sSJE2blIyMjn/JqRERERERE5FFKujOYra0t06ZNIzY2lr/++ot169ZRq1atJ573+++/U79+fcLCwjAajfz999+cPXsWFxcXs3LR0dH06dOHwYMHU6xYsST1tGjRgunTp3P69Gni4+OZO3cuH374Iffu3Uu3axQREREREXlVaXp5BnvttdcoXrw47733Hrdv36ZRo0a0aNHiied5eHjQtWtXevXqxfXr18mdOzf169endevWZuV++eUXoqKiGDZsGMOGDTPtL1euHMHBwQQEBHDr1i1atWpFfHw8rq6uzJw5k2zZsqX7tYqIiIiIiLxqlHSnAycnJ9MU7f++smvixIlm2y1btqRly5Zm+5o3b07z5s3T3O5HH33ERx99lOyxR6eMf/DBBynWYWdnlyQhFxERERERkfSh6eUiIiIiIiIiFqKR7kyqS5cu7Nq1K8XjI0aMoHHjxs8vIBEREREREUkzJd0ZyM/PDz8/v2SPBQUFPedoREREREREJL0p6ZY0m9OzPTlz5szoMERERERERDI9PdMtIiIiIiIiYiFKukVEREREREQsREm3iIiIiIiIiIUo6RYRERERERGxECXdkmYfBc3m3UFfZXQYIiIiIiIimZ6SbhERERERERELUdItIiIiIiIiYiFKukVEREREREQsREm3iIiIiIiIiIUo6bawkJAQatWq9cRy+/btw2AwcP/+/Wduc8CAAfTu3fuZ6xEREREREZFno6TbApYvX050dDQAjRs3Ztu2bRkckYiIiIiIiGQEJd3pLCEhgW+++YZ//vkno0MRERERERGRDPZSJt0Gg4F169bh5+dH6dKl6dSpE1euXKFDhw54eHjg5+dHZGRkquq6f/8+gwcPxsvLi7Jly9KqVStOnjxpOl6rVi2mT59O7dq1GTZsGOXLl+f27dv4+voydepUVq5cSZUqVUzl//jjD5o3b467uzt169Zl/fr1ybZ78eJFunTpQoUKFShXrhz9+vUjJibmqe7H+vXr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3n33XXbu3Em9evVwd3enQ4cO3Lx5E4ABAwbQu3dvwsLCaNGiBdevX8fNzY1p06ZRqlQp/vnnH1NMsbGxeHh4EBoamq73U0RERERE5FWkpDsF3bt3p1ixYuzevZvAwEAmTZrErl276N27NxUrVmTfvn0sX76c7du3s3jxYvLly8fs2bMBOHDgAH5+fk9sY+bMmRw5coS1a9eyd+9e3nrrLQYMGADA2LFj8fX1pV69epw4cYLw8HAAAgMDzaaLP3Tv3j3WrVvHkiVL2LhxIydOnGDp0qVmZUqXLs2IESPIly8f4eHhBAQEUKBAATZu3GgqExoaSvbs2alUqdJT3zsRERERERH5l5LuZPz555+cOHGCbt26kS1bNlxdXZk6dSoFCxYkJCSELl26YGNjQ+HChSlXrhxHjx59qnY6d+7MokWLyJ07N7a2ttSrV4/jx4/z4MGDNNeVkJDAJ598Qq5cuShYsCDvvvsuZ86ceew5VlZW+Pr6smbNGtO+zZs34+3tjY2NTZpjEBEREREREXNaSC0ZFy5cIEeOHOTOndu0r3LlygBs3bqVadOmce7cOR48eMCDBw+oV6/eU7UTHR3NyJEj2b9/P3fu3AH+TZ4TEhKeaiE2Jycn09fZsmUjNjb2iec0btyY6dOnc/HiRfLnz8+OHTtMI/YiIiIiIiLybDTSnQxra2sSExOT7D99+jQ9e/bkgw8+YM+ePYSHh9OgQYOnbqd3797ExMSwatUqjh49ysyZM58lbKyt096dLi4ulClThnXr1rF//37y5Mljen5cREREREREno2S7mQ4Oztz584drl69atq3detWNmzYgK2tLR999BH29vYYjUaOHTv21O2EhYXRrFkzChYsCMAff/zxzLE/jcaNG7Nx40Y2bNhAw4YNMyQGERERERGRl5GS7mS4urpSsmRJJk2axJ07dzh58iSDBg0ia9asxMbGcuzYMW7evMm4ceOwtbXl6tWrGI1G7O3tATh79ix37959YjuOjo6EhYURHx/Pzp072bVrFwBRUVEWuzZ7e3tu375NVFSUafq5t7c3f/31l5JuERERERGRdKakOwVBQUFcvHiRypUr06VLFwICAujcuTOtW7emTZs2+Pj44OjoyBdffMHJkyfp3bs3rq6ueHh48OGHH7Jo0aIntjF06FA2b95M+fLlWb58Od9++y1lypTBz8+P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PDxcKSkpkqTi4mJNmzZN//M//yMfHx8988wz2rJli2ne8PBwpaamqm/fvvL391e7du20cePGW342qampat26tQICAjRu3DgNGzZMQ4cOlSQZjUaNHz9eoaGh8vf31zPPPKMdO3aUeK/rdQ0dOlQjR47UmDFjFBgYqFatWmnWrFl33jQAQJm99NJLcnFxKXVbdna2hg0bpjfffFOZmZn64IMP9P7772vXrl2SpMzMTD388MOKi4tTixYtFB0drU2bNt1xLYRuAACs2Lx58zRz5kytW7dOV65c0ezZs/XII49o5cqVkqQlS5YoLi5OGRkZGjdunEaOHKmdO3fqtdde05tvvqkDBw6Y5vryyy81atQozZw50/TawoULlZiYqKVLl2rLli3q06eP3njjDW3YsEFFRUX6+OOPJUk7duzQhAkTNH36dO3atUt9+vTRoEGDdPbsWfXq1UvR0dFq3769srOzZTAYShzDggULlJqaqqlTp2rnzp3q2LGj4uLidObMGdOY5ORk9evXT9u2bVNgYKBGjx5d6ueRk5Ojt99+W8OHD9emTZvk7Oys//znP6btS5YsUXp6uhYtWqSdO3cqIiJCr732moqKikqdb/ny5WratKk2bdqkN998UxMnTtSpU6fK2SUAwN20ePFitW3bViEhITIYDAoICFBUVJSWLFkiSTpx4oSWLl2qbt26acOGDWrfvr3i4+N18uTJO3o/QjcAAFYsJiZGdevWVc2aNRUSEqLc3NxSx6Wlpenpp59WQECA7O3t9dRTT8nT07PEkm8fHx/5+PjIxsbG9FqHDh300EMPqWHDhnr00Ufl7e0tLy8vVa9eXYGBgfrpp58kSS1atNCmTZv02GOPycbGRk8//bTy8/P1/fff3/YY0tLS1LVrVzVp0kQODg7q2bOnnJ2dtW7dOtOYsLAw+fj4yMHBQZGRkfrpp59UXFx801zr169XkyZNFBkZKUdHR7366qtydnY2be/YsaNWrFghV1dXGQwGdejQQWfPntWxY8dKrc3d3V3PPPOM6TMrKioyHTMAwDJ++eUXff311/L29jb9W7p0qSlUG41GhYaGqnXr1qpWrZpiY2Pl4uJS4v+V8rC7i7UDAIAqxt3d3fTY2dlZ+fn5pY47cuSIWrVqVeK1Bg0a6OjRo6bnbm5uN+1Xr14902NHR0fVrVu3xPOCggJJUlFRkaZNm6aVK1fq7NmzpjHXt/+ZI0eOqFGjRiVe8/DwKFHbjcfp5OSkoqIiFRYWytHRscR+p0+fLnEcBoNBXl5epudXr17V6NGjtX79ep0/f/62df7x85WkvLy82x4TAMB8nJycFBMTo7fffrvU7XXq1NH9999vem5ra6v69evr9OnTd/R+nOkGAMCK3XhW+s/cKlTeuP8fl32XNr+tbelfPaZNm6YVK1ZoxowZ2rNnj7KysspUV1lru9X7/lFxcbHs7Eqek7hx33fffVc5OTlasGCBsrOzS1zXXpqyvi8AoOJ4eHiUuDxK+n1J+fVLhRo1aqR9+/aZthmNRh07dqzUXy6XBf8TAACA2/Lw8NCPP/5Y4rUff/xRDz/88F2ZPzs7WxEREfLy8pKtra1ycnLuuLZr167p559/vqPaateuXWKpeFFRkfbu3Wt6/u2336pTp05q2LChbGxsylUnAKByeO6557Rr1y59/vnnKigo0L59+/T888+bLpl64YUXlJWVpS+++EL5+flKTk5Wfn6+2rVrd0fvR+gGAAC3FR0drWXLlikrK0uFhYVavHixfvjhB3Xo0OGuzO/m5qb9+/fr6tWrOnjwoGbPni0XFxfT9XWOjo46fvy4Lly4oGvXrt1U26effqrc3FwVFBQoMTHRdIfz8mrVqpW+++47rVu3TgUFBZoxY0aJ5eDu7u7Kzs5WQUGBsrKy9OWXX0oSN0cDgErm+rXaS5YsMf31C29vb0m/n8meMGGCZs+erYCAAPXv31+9evXSU089JUny8vLShx9+qMTERAUEBGj58uWm/5fuBNd0AwCA2+rQoYOOHj2qwYMH69dff9Wjjz6qOXPmqGHDhndl/tjYWA0cOFCtWrVS48aNNWbMGNWtW1cJCQmqVauWOnbsqJUrVyosLEzLli0rsW/Pnj3122+/qU+fPrpw4YI8PT01b968EtfjlVXLli01YMAADRo0SPb29vrHP/6hoKAg01L1N954Q4MHD1ZgYKB8fX01duxYSVJcXJzmz5//1z8IAMBdkZ2d/afbo6KiFBUVdcvtkZGRpj+f+VfZGI1G412ZCQAA4B5QUFAgBwcH0/Nu3bopICBAAwYMsFxRAIAqi+XlAAAA/+/w4cPy9/dXRkaGiouLtXHjRu3evVtt2rSxdGkAgCqKM90AAAA3WLZsmaZPn67jx4+rbt266tWrl1544QVLlwUAqKII3QAAAAAAmAnLywEAAAAAMBNCNwAAAAAAZkLoBgAAAADATAjdAAAAAACYCaEbAAAAAAAzIXQDAAAAAGAmhG4AAAAAAMyE0A0AAAAAgJkQugEAAAAAMJP/A6VFBcKXnlI0AAAAAElFTkSuQmCC
)
    


             feature   importance
            s5_score 1534605.5917
           tfidf_sim  413536.1745
    recent_tfidf_sim   64859.9335
       u_click_count    8288.6118
        u_click_freq    7291.2530
          m_log_impr    6724.4048
    article_age_days    6541.8435
       ctr_norm_rank    6532.4631
            imp_size    5827.9240
       m_article_len    5747.0365
        m_log_clicks    5397.1749
             rank_s3    4494.6235
      taste_affinity    4470.5311
        cat_affinity    4251.8797
       subcat_clicks    3474.2606
               in_s3     234.4659
        n_retrievers      99.0991
             rank_s2      46.7221
             rank_s4       5.1219
               in_s4       0.0000
               in_s2       0.0000


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
           S5: LightGBM Base      10.90   42.74 16.69 30.16     49.60      39.88
      S6: Meta-LGB (2-Stage)      10.42   40.94 15.96 28.90     48.10      38.50
    S7: Ensemble (LGB + XGB)       9.26   36.70 14.24 26.26     43.50      34.88
              S1: Popularity       9.08   36.58 14.07 25.40     42.30      33.85
                 S3: Item-CF       8.08   33.20 12.57 22.62     38.50      30.56
       S2: Category Affinity       7.80   32.36 12.19 21.97     37.30      29.64
          S4: Temporal Taste       7.80   32.36 12.19 21.97     37.30      29.64
    
    ======================================================================
      LEADERBOARD  @  K = 10
    ======================================================================
                    strategy  precision  recall    f1  ndcg  hit_rate  composite
           S5: LightGBM Base       7.86   59.42 13.45 36.04     67.10      51.57
      S6: Meta-LGB (2-Stage)       7.58   57.60 12.98 34.68     65.40      50.04
              S1: Popularity       7.19   55.79 12.39 32.09     63.80      47.94
    S7: Ensemble (LGB + XGB)       7.18   55.63 12.36 32.75     62.80      47.77
                 S3: Item-CF       6.42   51.30 11.14 28.87     58.30      43.59
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

    K=5: S1 composite=33.85%  →  S6=38.50%  (+ 13.7% relative lift)
    K=10: S1 composite=47.94%  →  S6=50.04%  (+ 4.4% relative lift)



```python
# Save the artifacts for loading at a later time
OUT = '/content/drive/MyDrive/mind_artifacts'
os.makedirs(OUT, exist_ok = True)

# Feature stores
user_stats.to_parquet(f'{OUT}/user_stats.parquet')
article_feat.to_parquet(f'{OUT}/article_feat.parquet')
user_cat_affinity.to_parquet(f'{OUT}/user_cat_affinity.parquet')
user_taste_norm.to_parquet(f'{OUT}/user_taste_norm.parquet')
article_cat.to_parquet(f'{OUT}/article_cat.parquet')
eval_warm.to_parquet(f'{OUT}/eval_warm.parquet')
interactions_train.to_parquet(f'{OUT}/interactions_train.parquet')
train_clicks.to_parquet(f'{OUT}/train_clicks.parquet')
news.to_parquet(f'{OUT}/news.parquet')

# Trained models
lgb_model.save_model(f'{OUT}/lgb_base.txt')
joblib.dump(meta_lgb, f'{OUT}/meta_lgb.pkl')
joblib.dump(xgb_meta, f'{OUT}/xgb_meta.pkl')
joblib.dump(tfidf,    f'{OUT}/tfidf.pkl')

# Lookup dicts
joblib.dump(item_sim_lookup,          f'{OUT}/item_sim_lookup.pkl')
joblib.dump(user_tfidf_centroids,     f'{OUT}/user_tfidf_centroids.pkl')
joblib.dump(user_recent_tfidf_centroids, f'{OUT}/user_recent_tfidf_centroids.pkl')
joblib.dump(pop_stats,                f'{OUT}/pop_stats.pkl')
joblib.dump(tfidf_mat,                f'{OUT}/tfidf_mat.pkl')
joblib.dump(tfidf_idx,                f'{OUT}/tfidf_idx.pkl')

# Baseline results for comparison
joblib.dump(all_results, f'{OUT}/baseline_results.pkl')
print('Artifacts saved.')
```

    Artifacts saved.



```python
raw_train.to_parquet(f'/content/drive/MyDrive/mind_artifacts/raw_train.parquet')
imp_train_df.to_parquet(f'/content/drive/MyDrive/mind_artifacts/imp_train_df.parquet')
```

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
