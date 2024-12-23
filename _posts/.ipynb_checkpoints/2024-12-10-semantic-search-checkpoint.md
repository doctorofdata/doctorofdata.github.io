---
layout: post
title:  "Using Semantic Re-Ranking to Improve Results w/ Semantic Search"
excerpt_separator: <!--more-->
---

This demonstration involves using a technique called semantic re-ranking as a means for improving the search relevance of results retuned using the Wayfair WANDS dataset as a benchmark for accuracy.

<!--more-->

<a class = "anchor" id = "top"></a>

# Semantic Search

In this exercise, we will use various techniques to assess search relevance by testing "Mean Absolute Percent Error" (MAPE) of our algorithms and their ability to retrieve relevant products based on various search queries. 

The techniques we will use include:

-[TF-IDF](#tfidf)

-[BM25](#bm)

-[Sentence Embeddings](#embeddings)

-[Semantic ReRanking](#reranking)

-[Evaluation](#eval)

---


```python
# Install the required packages into the environment
!pip install rank-bm25 sentence-transformers torch transformers
```

    Collecting rank-bm25
      Downloading rank_bm25-0.2.2-py3-none-any.whl.metadata (3.2 kB)
    Collecting sentence-transformers
      Downloading sentence_transformers-2.7.0-py3-none-any.whl.metadata (11 kB)
    Requirement already satisfied: torch in /opt/conda/lib/python3.10/site-packages (2.1.2+cpu)
    Requirement already satisfied: transformers in /opt/conda/lib/python3.10/site-packages (4.39.3)
    Requirement already satisfied: numpy in /opt/conda/lib/python3.10/site-packages (from rank-bm25) (1.26.4)
    Requirement already satisfied: tqdm in /opt/conda/lib/python3.10/site-packages (from sentence-transformers) (4.66.1)
    Requirement already satisfied: scikit-learn in /opt/conda/lib/python3.10/site-packages (from sentence-transformers) (1.2.2)
    Requirement already satisfied: scipy in /opt/conda/lib/python3.10/site-packages (from sentence-transformers) (1.11.4)
    Requirement already satisfied: huggingface-hub>=0.15.1 in /opt/conda/lib/python3.10/site-packages (from sentence-transformers) (0.22.2)
    Requirement already satisfied: Pillow in /opt/conda/lib/python3.10/site-packages (from sentence-transformers) (9.5.0)
    Requirement already satisfied: filelock in /opt/conda/lib/python3.10/site-packages (from torch) (3.13.1)
    Requirement already satisfied: typing-extensions in /opt/conda/lib/python3.10/site-packages (from torch) (4.9.0)
    Requirement already satisfied: sympy in /opt/conda/lib/python3.10/site-packages (from torch) (1.12)
    Requirement already satisfied: networkx in /opt/conda/lib/python3.10/site-packages (from torch) (3.2.1)
    Requirement already satisfied: jinja2 in /opt/conda/lib/python3.10/site-packages (from torch) (3.1.2)
    Requirement already satisfied: fsspec in /opt/conda/lib/python3.10/site-packages (from torch) (2024.2.0)
    Requirement already satisfied: packaging>=20.0 in /opt/conda/lib/python3.10/site-packages (from transformers) (21.3)
    Requirement already satisfied: pyyaml>=5.1 in /opt/conda/lib/python3.10/site-packages (from transformers) (6.0.1)
    Requirement already satisfied: regex!=2019.12.17 in /opt/conda/lib/python3.10/site-packages (from transformers) (2023.12.25)
    Requirement already satisfied: requests in /opt/conda/lib/python3.10/site-packages (from transformers) (2.31.0)
    Requirement already satisfied: tokenizers<0.19,>=0.14 in /opt/conda/lib/python3.10/site-packages (from transformers) (0.15.2)
    Requirement already satisfied: safetensors>=0.4.1 in /opt/conda/lib/python3.10/site-packages (from transformers) (0.4.3)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /opt/conda/lib/python3.10/site-packages (from packaging>=20.0->transformers) (3.1.1)
    Requirement already satisfied: MarkupSafe>=2.0 in /opt/conda/lib/python3.10/site-packages (from jinja2->torch) (2.1.3)
    Requirement already satisfied: charset-normalizer<4,>=2 in /opt/conda/lib/python3.10/site-packages (from requests->transformers) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /opt/conda/lib/python3.10/site-packages (from requests->transformers) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/conda/lib/python3.10/site-packages (from requests->transformers) (1.26.18)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.10/site-packages (from requests->transformers) (2024.2.2)
    Requirement already satisfied: joblib>=1.1.1 in /opt/conda/lib/python3.10/site-packages (from scikit-learn->sentence-transformers) (1.4.0)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /opt/conda/lib/python3.10/site-packages (from scikit-learn->sentence-transformers) (3.2.0)
    Requirement already satisfied: mpmath>=0.19 in /opt/conda/lib/python3.10/site-packages (from sympy->torch) (1.3.0)
    Downloading rank_bm25-0.2.2-py3-none-any.whl (8.6 kB)
    Downloading sentence_transformers-2.7.0-py3-none-any.whl (171 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m171.5/171.5 kB[0m [31m1.4 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: rank-bm25, sentence-transformers
    Successfully installed rank-bm25-0.2.2 sentence-transformers-2.7.0



```python
# Import libraries
import numpy as np 
import pandas as pd 
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
```


```python
# Check data files
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

    /kaggle/input/wayfair-wands-dataset/query.csv
    /kaggle/input/wayfair-wands-dataset/product.csv
    /kaggle/input/wayfair-wands-dataset/label.csv



```python
# Compile functions for later implementation
def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:

    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# Function to lookup query matches
def get_exact_matches_for_(query):

    labels = label_df[label_df['query_id'] == query]

    return labels[labels['label'] == 'Exact']['product_id'].values

# Function to get tf-idf results
def get_tfidf_products(x):

    return cosine_similarity(vec.transform([x]), matrix).flatten().argsort()[-10:][::-1]

#define functions for evaluating retrieval performance
def map_at_k(true_ids, predicted_ids, k = 10):
    
    """
    Calculate the Mean Average Precision at K (MAP@K).

    Parameters:
    true_ids (list): List of relevant product IDs.
    predicted_ids (list): List of predicted product IDs.
    k (int): Number of top elements to consider.
             NOTE: IF you wish to change top k, please provide a justification for choosing the new value

    Returns:
    float: MAP@K score.
    """
    #if either list is empty, return 0
    if not len(true_ids) or not len(predicted_ids):
        return 0.0

    score = 0.0
    num_hits = 0.0
    
    # Calculate score
    for i, p_id in enumerate(predicted_ids[:k]):

        if p_id in true_ids and p_id not in predicted_ids[:i]:

            num_hits += 1.0
            score += num_hits / (i + 1.0)
    
    # Return MAPE
    return score / min(len(true_ids), k)

# Function to execute bm search
def execute_bm_search(q):

    return np.argsort(bm25.get_scores(q.split(' ')))[-10:]

# Function to perform reranking
def execute_reranking(data, query):

    # Translate query
    query_embeddings = biencoder.encode(query, convert_to_tensor = True) #.cuda()

    # Get cosine similarity
    hits = util.semantic_search(query_embeddings, corpus_embeddings, top_k = topk)
    hits = hits[0]

    # Perform reranking
    cross_inp = [[query, data[hit['corpus_id']]] for hit in hits]
    cross_scores = crossencoderembeddingmodel.predict(cross_inp)

    # Sort results by the cross-encoder scores
    for idx in range(len(cross_scores)):

        hits[idx]['score'] = cross_scores[idx]

    sortie = sorted(hits, key = lambda x: x['score'], reverse = True)[0:10]

    return [item['corpus_id'] for item in sortie]
```


```python
# Load the components
query_df = pd.read_csv("/kaggle/input/wayfair-wands-dataset/query.csv", sep = '\t')
product_df = pd.read_csv("/kaggle/input/wayfair-wands-dataset/product.csv", sep = '\t')
label_df = pd.read_csv("/kaggle/input/wayfair-wands-dataset/label.csv", sep = '\t')
```


```python
# Inspect initial dataframes
product_df.head()
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
      <th>product_id</th>
      <th>product_name</th>
      <th>product_class</th>
      <th>category hierarchy</th>
      <th>product_description</th>
      <th>product_features</th>
      <th>rating_count</th>
      <th>average_rating</th>
      <th>review_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>solid wood platform bed</td>
      <td>Beds</td>
      <td>Furniture / Bedroom Furniture / Beds &amp; Headboa...</td>
      <td>good , deep sleep can be quite difficult to ha...</td>
      <td>overallwidth-sidetoside:64.7|dsprimaryproducts...</td>
      <td>15.0</td>
      <td>4.5</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>all-clad 7 qt . slow cooker</td>
      <td>Slow Cookers</td>
      <td>Kitchen &amp; Tabletop / Small Kitchen Appliances ...</td>
      <td>create delicious slow-cooked meals , from tend...</td>
      <td>capacityquarts:7|producttype : slow cooker|pro...</td>
      <td>100.0</td>
      <td>2.0</td>
      <td>98.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>all-clad electrics 6.5 qt . slow cooker</td>
      <td>Slow Cookers</td>
      <td>Kitchen &amp; Tabletop / Small Kitchen Appliances ...</td>
      <td>prepare home-cooked meals on any schedule with...</td>
      <td>features : keep warm setting|capacityquarts:6....</td>
      <td>208.0</td>
      <td>3.0</td>
      <td>181.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>all-clad all professional tools pizza cutter</td>
      <td>Slicers, Peelers And Graters</td>
      <td>Browse By Brand / All-Clad</td>
      <td>this original stainless tool was designed to c...</td>
      <td>overallwidth-sidetoside:3.5|warrantylength : l...</td>
      <td>69.0</td>
      <td>4.5</td>
      <td>42.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>baldwin prestige alcott passage knob with roun...</td>
      <td>Door Knobs</td>
      <td>Home Improvement / Doors &amp; Door Hardware / Doo...</td>
      <td>the hardware has a rich heritage of delivering...</td>
      <td>compatibledoorthickness:1.375 '' |countryofori...</td>
      <td>70.0</td>
      <td>5.0</td>
      <td>42.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
product_df.shape
```




    (42994, 9)




```python
query_df.head()
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
      <th>query_id</th>
      <th>query</th>
      <th>query_class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>salon chair</td>
      <td>Massage Chairs</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>smart coffee table</td>
      <td>Coffee &amp; Cocktail Tables</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>dinosaur</td>
      <td>Kids Wall Décor</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>turquoise pillows</td>
      <td>Accent Pillows</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>chair and a half recliner</td>
      <td>Recliners</td>
    </tr>
  </tbody>
</table>
</div>




```python
query_df.shape[0]
```




    480




```python
label_df.head()
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
      <th>id</th>
      <th>query_id</th>
      <th>product_id</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>25434</td>
      <td>Exact</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>12088</td>
      <td>Irrelevant</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>42931</td>
      <td>Exact</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>2636</td>
      <td>Exact</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0</td>
      <td>42923</td>
      <td>Exact</td>
    </tr>
  </tbody>
</table>
</div>




```python
product_df.isnull().sum()
```




    product_id                0
    product_name              0
    product_class          2852
    category hierarchy     1556
    product_description    6008
    product_features          0
    rating_count           9452
    average_rating         9452
    review_count           9452
    dtype: int64




```python
# Combine fields and process text
product_df['text'] = product_df['product_name'] + ' ' + product_df['product_description'].fillna('')
product_df['text'] = product_df['text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
product_df['text'] = product_df['text'].apply(lambda x: re.sub('\d+', '', x).lower().strip().replace('  ', ' '))
```

[Back to top..](#top)

---

# TF-IDF <a class="anchor" id="tfidf"></a>


```python
# Calculate TF-IDF
vec = TfidfVectorizer()
tfidf = vec.fit(product_df['text'])
matrix = tfidf.transform(product_df['text'])
```


```python
# Use the product ledger to calculate exact matches for each suggestion
query_df['matches'] = query_df['query_id'].apply(get_exact_matches_for_)
```


```python
# Use TF-IDF to calculate cosine similarity and return similar entries
query_df['suggestions'] = query_df['query'].apply(get_tfidf_products)
```


```python
# Calclate Mean Average Precision MAPE
query_df['score'] = query_df.apply(lambda x: map_at_k(x['matches'], x['suggestions'], k = 10), axis = 1)
```

[Back to top..](#top)

---

# BM25 <a class="anchor" id="bm"></a>


```python
# Init model
bm25 = BM25Okapi(product_df['text'].apply(lambda x: x.split(' ')))
```


```python
# Add BM result to data
query_df['bm_suggestions'] = query_df['query'].apply(execute_bm_search)
```


```python
# Score BM25
query_df['bm_score'] = query_df.apply(lambda x: map_at_k(x['matches'], x['bm_suggestions'], k = 10), axis = 1)
```

[Back to top..](#top)

---

# Embeddings <a class="anchor" id="embeddings"></a>


```python
# Prepare an embedding model
queries = []
passages = []

for idx, row in query_df.iterrows():

    input = f"query: {row['query']}"
    queries.append(input)

for idx, row in product_df.iterrows():

    input = f"passage: {row['text']}"
    passages.append(input)

input_texts = queries + passages
```


```python
# Init model
embedder = SentenceTransformer("all-MiniLM-L6-v2")
```


    modules.json:   0%|          | 0.00/349 [00:00<?, ?B/s]



    config_sentence_transformers.json:   0%|          | 0.00/116 [00:00<?, ?B/s]



    README.md:   0%|          | 0.00/10.7k [00:00<?, ?B/s]



    sentence_bert_config.json:   0%|          | 0.00/53.0 [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/612 [00:00<?, ?B/s]



    model.safetensors:   0%|          | 0.00/90.9M [00:00<?, ?B/s]



    tokenizer_config.json:   0%|          | 0.00/350 [00:00<?, ?B/s]



    vocab.txt:   0%|          | 0.00/232k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/466k [00:00<?, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/112 [00:00<?, ?B/s]



    1_Pooling/config.json:   0%|          | 0.00/190 [00:00<?, ?B/s]



```python
# Apply embeddings to data
%time corpus_embeddings = embedder.encode(passages, convert_to_tensor = True)
```


    Batches:   0%|          | 0/1344 [00:00<?, ?it/s]


    CPU times: user 31min 22s, sys: 3min 1s, total: 34min 24s
    Wall time: 17min 8s



```python
%time query_embeddings = embedder.encode(queries, convert_to_tensor = True)
```


    Batches:   0%|          | 0/15 [00:00<?, ?it/s]


    CPU times: user 3.81 s, sys: 21.2 ms, total: 3.83 s
    Wall time: 1.92 s



```python
# Embed the corpus
#corpus_embeddings = corpus_embeddings.to("cuda")
corpus_embeddings = util.normalize_embeddings(corpus_embeddings)
```


```python
# Embed the queries
#query_embeddings = query_embeddings.to("cuda")
query_embeddings = util.normalize_embeddings(query_embeddings)
```


```python
# Comnpute cosine simi9larities
hits = util.semantic_search(query_embeddings, corpus_embeddings, score_function = util.dot_score, top_k = 10)
```


```python
# ECompile semantic similarity back into dataset
sims = [[y['corpus_id'] for y in x] for x in hits]

# Add the scores to the original data
query_df['semantic_suggestions'] = sims

query_df['semantic_score'] = query_df.apply(lambda x: map_at_k(x['matches'], x['semantic_suggestions'], k = 10), axis = 1)
```

[Back to top..](#top)

---

# Re-Ranking <a class="anchor" id="reranking"></a>


```python
# Init models for reranking
biencoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
crossencoderembeddingmodel = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Set sequence length
biencoder.max_seq_length = 512

# Num docs
topk = 100
```


    modules.json:   0%|          | 0.00/349 [00:00<?, ?B/s]



    config_sentence_transformers.json:   0%|          | 0.00/116 [00:00<?, ?B/s]



    README.md:   0%|          | 0.00/11.6k [00:00<?, ?B/s]



    sentence_bert_config.json:   0%|          | 0.00/53.0 [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/612 [00:00<?, ?B/s]



    model.safetensors:   0%|          | 0.00/90.9M [00:00<?, ?B/s]



    tokenizer_config.json:   0%|          | 0.00/383 [00:00<?, ?B/s]



    vocab.txt:   0%|          | 0.00/232k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/466k [00:00<?, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/112 [00:00<?, ?B/s]



    1_Pooling/config.json:   0%|          | 0.00/190 [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/794 [00:00<?, ?B/s]



    pytorch_model.bin:   0%|          | 0.00/90.9M [00:00<?, ?B/s]


    /opt/conda/lib/python3.10/site-packages/torch/_utils.py:831: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
      return self.fget.__get__(instance, owner)()



    tokenizer_config.json:   0%|          | 0.00/316 [00:00<?, ?B/s]



    vocab.txt:   0%|          | 0.00/232k [00:00<?, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/112 [00:00<?, ?B/s]



```python
# Perform initial encoding
corpus_embeddings = biencoder.encode(product_df['text'], convert_to_tensor = True, show_progress_bar = True)
query_embeddings = biencoder.encode(query_df['query'].iloc[0], convert_to_tensor = True, show_progress_bar = True) #.cuda()
```


    Batches:   0%|          | 0/1344 [00:00<?, ?it/s]



    Batches:   0%|          | 0/1 [00:00<?, ?it/s]



```python
# Extract 100 relevant passages for each query
hits = util.semantic_search(query_embeddings, corpus_embeddings, top_k = topk)
hits = hits[0]
```


```python
# Inspection Re-Rank the products
cross_inp = [[query_df['query'].iloc[0], product_df['text'][hit['corpus_id']]] for hit in hits]
cross_scores = crossencoderembeddingmodel.predict(cross_inp)

# Sort results by the cross-encoder scores
for idx in range(len(cross_scores)):

    hits[idx]['score'] = cross_scores[idx]
    
# Examine outputs
print(f"Query Selection- {query_df['query'].iloc[0]}")

print("Top-3 Re-Ranked Hits")
hits = sorted(hits, key = lambda x: x['score'], reverse = True)

for hit in hits[0:3]:

    print("\t{:.3f}\t{}".format(hit['score'], product_df['product_name'][hit['corpus_id']].replace("\n", " ")))
```


    Batches:   0%|          | 0/4 [00:00<?, ?it/s]


    Query Selection- salon chair
    Top-3 Re-Ranked Hits
    	8.191	barberpub salon massage chair
    	8.156	hair salon chair
    	7.957	reclining faux leather massage chair



```python
%%capture

# Perform semantic reranking on the data
query_df['reranked'] = query_df['query'].apply(lambda x: execute_reranking(product_df['text'], x))
```


```python
query_df['reranking_score'] = query_df.apply(lambda x: map_at_k(x['matches'], x['reranked'], k = 10), axis = 1)
```

[Back to top..](#top)

---

# Evaluation <a class="anchor" id="eval"></a>


```python
print('Semantic Scoring Results- MAPE scores')
print(f"TF-IDF on the Queries was {round(query_df.loc[:, 'score'].mean(), 3)}")
print(f"BM25 on the Queries was {round(np.mean(query_df['bm_score']), 3)}")
print(f"Cosine similarity of the embeddings on the Queries was {round(query_df.loc[:, 'semantic_score'].mean(), 3)}")
print(f"Semantic ReRanking on the Queries was {round(query_df.loc[:, 'reranking_score'].mean(), 3)}")
```

    Semantic Scoring Results- MAPE scores
    TF-IDF on the Queries was 0.272
    BM25 on the Queries was 0.261
    Cosine similarity of the embeddings on the Queries was 0.323
    Semantic ReRanking on the Queries was 0.437


[Back to top..](#top)
