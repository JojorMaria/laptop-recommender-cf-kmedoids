# Laptop Recommendation System using Collaborative Filtering and K-Medoids

A laptop recommendation system that combines **User-Based Collaborative Filtering** and **K-Medoids clustering** to address the **user cold-start problem**, where new users have no rating history. Built as an undergraduate thesis at Sanata Dharma University, Yogyakarta.

## Overview

Choosing a laptop is difficult because of the many options in brand, processor, RAM, storage, and price. This system provides recommendations for two types of users:

- **Returning users** (with rating history): recommendations are generated with User-Based Collaborative Filtering using Cosine Similarity and weighted-sum rating prediction.
- **New users** (cold-start): users enter their preferences (price, RAM, storage, brand, CPU model). The system finds the closest laptop cluster using Euclidean distance, then recommends the most popular laptops in that cluster.

## Dataset

- **Source:** [Amazon Laptop Reviews Enriched](https://huggingface.co/datasets) on Hugging Face `[tempel link dataset yang tepat]`
- **Size:** 32,759 reviews, 31 columns
- **Features used for clustering:** price, RAM, hard drive, brand, CPU model

## Methodology

1. **Preprocessing:** feature selection, missing value handling, brand standardization, CPU model simplification, outlier range filtering, label encoding, and Min-Max normalization.
2. **Clustering:** K-Medoids (PAM, Euclidean distance) tested for k = 2 to 15, evaluated with Silhouette Coefficient.
3. **Recommendation:** User-Based CF (Cosine Similarity) for returning users; cluster-based popularity recommendation for new users.
4. **Evaluation:** Leave-One-Out Cross Validation with MAE and RMSE for k = 5 to 60 neighbors.
5. **System testing:** Alpha testing (functionality) and Beta testing (10 respondents).

## Results

| Metric | Result |
|---|---|
| Best number of clusters | k = 8 |
| Silhouette Coefficient | 0.7049 |
| Best neighbors (CF) | k = 5 |
| MAE | 1.3358 |
| RMSE | 1.6466 |
| Beta testing satisfaction | 85.67% |

Note: cold-start recommendations were not evaluated with MAE/RMSE, since these metrics apply only to users with rating history.

## Tech Stack

Python, pandas, NumPy, scikit-learn, scikit-learn-extra, SciPy, Matplotlib, Streamlit, Google Colaboratory

## Project Structure

```
laptop-recommender-cf-kmedoids/
├── app.py
├── notebooks/
│   ├── 01_kmedoids_clustering.ipynb
│   └── 02_collaborative_filtering_evaluation.ipynb
├── data/
└── README.md
```


## Screenshots

`[tambahkan screenshot halaman rekomendasi pengguna lama, input preferensi pengguna baru, dan halaman admin]`

## Author

**Jojor Maria Sitompul**
Informatics, Sanata Dharma University
GitHub: [jojormaria](https://github.com/jojormaria)
