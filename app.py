# ======================
# Import Library yang digunakan Untuk Rekomendasi 
# ======================
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import re
from datetime import datetime

# ======================
# Baca Dataset
# ======================
df_laptop     = pd.read_csv('data/whole_cluster_laptops.csv')
df_user       = pd.read_csv('data/amazon_laptop_reviews.csv')
df_laptopCore = pd.read_csv('data/laptop_features_coreData.csv')
df_userId     = pd.read_csv('data/users_id.csv')   # kolom: user_id, date_created

# ======================
# Preprocessing sedikit untuk Data User
# ======================
df_user_selected = df_user[['rating', 'parent_asin', 'user_id']]
df_user_selected = df_user_selected[
    df_user_selected['parent_asin'].isin(df_laptop['parent_asin'])
].reset_index(drop=True)

# ==================================================================
# Program Untuk Collaborative Filtering
# ==================================================================
df_reviews  = df_user_selected.dropna(subset=['user_id', 'parent_asin', 'rating'])
user_codes  = df_reviews['user_id'].astype('category').cat.codes
item_codes  = df_reviews['parent_asin'].astype('category').cat.codes
ratings_col = df_reviews['rating']

user_item_matrix = coo_matrix(
    (ratings_col, (user_codes, item_codes))
).tocsr()

user_mapping  = dict(enumerate(df_reviews['user_id'].astype('category').cat.categories))
item_mapping  = dict(enumerate(df_reviews['parent_asin'].astype('category').cat.categories))
item_to_index = {v: k for k, v in item_mapping.items()}
user_to_index = {v: k for k, v in user_mapping.items()}


def get_similar_users(user_index, uim, top_k=5):
    user_vector   = uim[user_index]
    similarities  = cosine_similarity(user_vector, uim).flatten()
    similar_users = similarities.argsort()[::-1][1:top_k+1]
    return similar_users, similarities[similar_users]

def recommend_items(user_index, user_item_matrix, similar_users, scores, top_n=15):
    user_vector = user_item_matrix[user_index].toarray().flatten()
    similar_matrix = user_item_matrix[similar_users].toarray()

    # Hitung weighted sum per item
    weighted_sum = np.dot(scores, similar_matrix)

    # Buat mask: 1 jika user similar merating item, 0 jika tidak
    rated_mask = (similar_matrix > 0).astype(float)
    # Sum similarity hanya untuk user yang merating
    sum_similarity_per_item = np.dot(scores, rated_mask)

    # Hindari pembagian dengan nol
    predicted_ratings = np.where(sum_similarity_per_item > 0,weighted_sum / sum_similarity_per_item,0)

    # Hapus item yang sudah dirating user target
    predicted_ratings[user_vector > 0] = 0

    candidate_items = np.where(predicted_ratings > 0)[0]
    if len(candidate_items) == 0:
        return [], []

    candidate_scores = predicted_ratings[candidate_items]
    top_indices = candidate_scores.argsort()[::-1][:top_n]
    return candidate_items[top_indices], candidate_scores[top_indices]


# ==================================================================
# Program Untuk Cold Start
# ==================================================================
user_item_df = df_user_selected.pivot_table(
    index='user_id', columns='parent_asin', values='rating'
).fillna(0)
user_matrix = user_item_df.values
user_ids    = user_item_df.index.tolist()
item_asins  = user_item_df.columns.tolist()


def coldStartProblem(userid, user_preferences, n=15):
    """
    n : int
        Jumlah (Top-N) laptop yang ingin direkomendasikan ke user.
        Default = 15.
    """
    feature_cols    = ['price', 'Ram', 'Hard Drive', 'Brand', 'CPU Model']
    user_pref_array = np.array(user_preferences)
    laptop_features = df_laptop[feature_cols]
    distances = [euclidean(user_pref_array, row.values)
                 for _, row in laptop_features.iterrows()]
    best_idx          = np.argmin(distances)
    predicted_cluster = df_laptop.iloc[best_idx]['Cluster']
    hasil             = recommend_new_user(predicted_cluster, n=n)
    return hasil, predicted_cluster


def recommend_new_user(cluster_id, n=15):
    cluster_items = df_laptop[df_laptop['Cluster'] == cluster_id]['parent_asin']
    scores = {}
    for item in cluster_items:
        if item not in item_asins:
            continue
        item_idx      = item_asins.index(item)
        r             = user_matrix[:, item_idx]
        valid_ratings = r[r > 0]
        if len(valid_ratings) > 0:
            scores[item] = valid_ratings.mean()
    recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return recs[:n]

def get_cluster_description(cluster_id):
    cluster_laptops = df_laptopCore[
        df_laptopCore['parent_asin'].isin(
            df_laptop[df_laptop['Cluster'] == cluster_id]['parent_asin']
        )
    ]

    if cluster_laptops.empty:
        return None

    top_brands = (
    cluster_laptops['brand']
    .dropna()
    .value_counts()
    .head(3)
    .index.tolist()
    )

    total_laptops = len(cluster_laptops)

    return {
        'top_brands': top_brands,
        'total_laptops': total_laptops
    }

def render_cluster_info(cluster_id):
    """Menampilkan kartu karakteristik cluster."""
    info = get_cluster_description(cluster_id)
    if info is None:
        return

    brands_str = ", ".join(info['top_brands']) if info['top_brands'] else "-"

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #eef4f1 0%, #f9fafb 100%);
        border: 1.5px solid rgba(101,150,123,.35);
        border-radius: 16px;
        padding: 24px 28px;
        margin: 8px 0 24px;
    ">
        <div style="
            font-family:'Syne',sans-serif;
            font-size:15px; font-weight:700;
            color:#1a2b22; margin-bottom:16px;
            display:flex; align-items:center; gap:8px;
        ">
            📋 Karakteristik Cluster {cluster_id}
            <span style="
                font-size:11px; font-weight:600;
                background:#2d6a4f; color:#fff;
                padding:3px 10px; border-radius:20px;
                font-family:'DM Sans',sans-serif;
            ">{info['total_laptops']} Laptop</span>
        </div>
        <div style="
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
        ">
            <div style="background:#fff;border-radius:10px;padding:14px 16px;border:1px solid #e2e8e4;">
                <div style="font-size:10px;text-transform:uppercase;letter-spacing:.8px;
                            color:#8fa89e;font-weight:600;margin-bottom:6px;">🏷️ Brand Dominan</div>
                <div style="font-size:13px;font-weight:700;color:#2d6a4f;line-height:1.5;">
                    {brands_str}
                </div>
            </div>
        </div>
        <div style="margin-top:12px;font-size:12px;color:#8fa89e;font-style:italic;">
            ℹ️ Laptop yang Anda dapatkan diambil dari kelompok ini berdasarkan kedekatan preferensi Anda.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==================================================================
# Cold Start Fallback Untuk User Lama Tanpa CF Result
# ==================================================================
def get_cold_start_from_user_history(user_id):
    user_ratings = df_user_selected[df_user_selected['user_id'] == user_id]
    best_rated   = user_ratings[user_ratings['rating'] >= 4].sort_values('rating', ascending=False)
    if len(best_rated) == 0:
        return None, None
    best_asin  = best_rated.iloc[0]['parent_asin']
    laptop_row = df_laptop[df_laptop['parent_asin'] == best_asin]
    if laptop_row.empty:
        return None, None
    row              = laptop_row.iloc[0]
    feature_cols     = ['price', 'Ram', 'Hard Drive', 'Brand', 'CPU Model']
    user_preferences = row[feature_cols].values.tolist()
    return user_preferences, best_asin


def predict_rating_for_user_item(
    user_index,
    item_index,
    uim_loo,
    similar_users,
    scores,
    global_mean_rating
):

    # Validasi item
    if item_index >= uim_loo.shape[1]:
        return global_mean_rating

    # Rating item dari user-user mirip
    similar_users_ratings_for_item = (
        uim_loo[
            similar_users,
            item_index
        ]
        .toarray()
        .flatten()
    )

    # Weighted sum
    weighted_sum_ratings = np.dot(
        scores,
        similar_users_ratings_for_item
    )

    # Mask user yang benar-benar memberi rating
    rated_mask = (
        similar_users_ratings_for_item > 0
    ).astype(float)

    # Jumlah similarity hanya user yang merating
    sum_similarity_per_item = np.dot(
        scores,
        rated_mask
    )

    # Hindari pembagian nol
    if sum_similarity_per_item == 0:
        return global_mean_rating

    # Prediksi rating
    predicted_rating = (
        weighted_sum_ratings /
        sum_similarity_per_item
    )

    return predicted_rating


# =========================================================
# FUNGSI EVALUASI LEAVE ONE OUT
# =========================================================
def evaluate_model_loo(
    uim,
    user_to_index,
    item_to_index,
    get_similar_users_func,
    predict_rating_func,
    k_similar=10
):

    # Global mean rating
    global_mean_rating = np.mean(uim.data)

    # Menyimpan rating aktual dan prediksi
    actual_ratings = []
    predicted_ratings = []

    # Jumlah user
    num_users = uim.shape[0]

    # Counter evaluasi
    total_evaluations = 0

    # Loop semua user
    for u_idx in range(num_users):

        # Ambil item yang pernah dirating
        rated_items_indices = uim[u_idx].nonzero()[1]

        # Skip jika user tidak punya rating
        if len(rated_items_indices) == 0:
            continue

        # Leave-One-Out
        for i_idx in rated_items_indices:

            # Rating asli
            actual_rating = float(uim[u_idx, i_idx])

            # Copy matrix
            uim_loo = uim.copy()

            # Hapus sementara rating
            uim_loo[u_idx, i_idx] = 0

            # Cari similar users
            similar_users, scores = get_similar_users_func(
                u_idx,
                uim_loo,
                top_k=k_similar
            )

            # Prediksi rating
            predicted_r = predict_rating_func(
                u_idx,
                i_idx,
                uim_loo,
                similar_users,
                scores,
                global_mean_rating
            )

            # Simpan hasil evaluasi
            actual_ratings.append(actual_rating)
            predicted_ratings.append(predicted_r)

            total_evaluations += 1

    # Jika tidak ada evaluasi
    if total_evaluations == 0:
        return 0.0, 0.0, 0

    # Hitung MAE
    mae = mean_absolute_error(
        actual_ratings,
        predicted_ratings
    )

    # Hitung RMSE
    rmse = np.sqrt(
        mean_squared_error(
            actual_ratings,
            predicted_ratings
        )
    )

    return mae, rmse, total_evaluations



# ========================================================================================
# Brand Mapping & MinMax Scaler
# ========================================================================================
brand_mapping = dict(zip(df_laptopCore['brand'],     df_laptop['Brand']))
cpu_mapping   = dict(zip(df_laptopCore['CPU_Model'], df_laptop['CPU Model']))
brand_options = sorted(df_laptopCore['brand'].dropna().unique().tolist())
cpu_options   = sorted(df_laptopCore['CPU_Model'].dropna().unique().tolist())

scaler_price = MinMaxScaler().fit(df_laptopCore[['price']])
scaler_ram   = MinMaxScaler().fit(df_laptopCore[['RAM']])
scaler_hd    = MinMaxScaler().fit(df_laptopCore[['Hard_Drive']])

# Pilihan Top-N yang bisa dipilih user di seluruh halaman
TOP_N_OPTIONS = [5, 10, 15, 20, 25, 30]
TOP_N_DEFAULT_INDEX = TOP_N_OPTIONS.index(15)  # default = 15


# ========================================================================================
# Helper: Registrasi & Validasi User
# ========================================================================================
def is_user_registered(user_id: str) -> bool:
    return user_id in df_userId['user_id'].values


def register_new_user(new_user_id: str) -> str:
    global df_userId
    tanggal  = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    new_row  = pd.DataFrame([{'user_id': new_user_id, 'date_created': tanggal}])
    df_userId = pd.concat([df_userId, new_row], ignore_index=True)
    df_userId.to_csv('users_id.csv', index=False)
    return tanggal





# ========================================================================================
# DIBAWAH INI PROGRAM MAIN
# ========================================================================================
# ========================================================================================
# Streamlit — konfigurasi
# ========================================================================================
import streamlit as st

st.set_page_config(
    page_title="LaptopAI — Rekomendasi Cerdas",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Session State ────────────────────────────────────────────────
_defaults = {
    "page": "home",
    "username": None,
    "cs_submitted": False,
    "cs_result": None,
    "user_cluster": None,
    "cs_preferences": None,
    "cs_top_n": 15,
    "show_cold_start_form": False,
    "old_user_cs_result": None,
    "old_user_cs_cluster": None,
    "old_user_cs_preferences": None,
    "old_user_cs_top_n": 15,
    "pending_register_id": None,
    "registered_user_id": None,
    "registered_date": None,
    "login_info_msg": None,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def go_to(page):
    st.session_state.page = page
    st.rerun()


def reset_rekomendasi_state():
    st.session_state.cs_submitted         = False
    st.session_state.cs_result            = None
    st.session_state.user_cluster         = None
    st.session_state.cs_preferences       = None
    st.session_state.cs_top_n             = 15
    st.session_state.show_cold_start_form = False
    st.session_state.old_user_cs_result   = None
    st.session_state.old_user_cs_cluster  = None
    st.session_state.old_user_cs_preferences = None
    st.session_state.old_user_cs_top_n       = 15


ADMIN_ID = "ADMIN12345"

# ========================================================================================
# GLOBAL CSS — Modern Professional Theme
# ========================================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Syne:wght@600;700;800&display=swap');

/* ── Reset & Base ─────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* ── Hide Streamlit chrome ────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Root Variables ───────────────────────────────────────────── */
:root {
    --primary:       #65967b;
    --primary-light: #8ab5a0;
    --primary-dark:  #4a7560;
    --primary-bg:    #eef4f1;
    --accent:        #2d6a4f;
    --amber:         #f59e0b;
    --amber-bg:      #fef3c7;
    --rose:          #e05c5c;
    --rose-bg:       #fde8e8;
    --sky:           #3b82f6;
    --sky-bg:        #dbeafe;
    --violet:        #7c3aed;
    --violet-bg:     #ede9fe;
    --bg:            #f4f6f4;
    --surface:       #ffffff;
    --surface2:      #f9fafb;
    --border:        #e2e8e4;
    --border-strong: #c8d4cd;
    --text:          #1a2b22;
    --text-2:        #4b6358;
    --text-3:        #8fa89e;
    --shadow-sm:     0 1px 3px rgba(0,0,0,.07), 0 1px 2px rgba(0,0,0,.05);
    --shadow:        0 4px 16px rgba(0,0,0,.08), 0 2px 6px rgba(0,0,0,.05);
    --shadow-lg:     0 12px 32px rgba(0,0,0,.1), 0 4px 12px rgba(0,0,0,.06);
    --radius:        14px;
    --radius-sm:     8px;
    --radius-lg:     20px;
}

/* ── App Background ───────────────────────────────────────────── */
.stApp { background-color: var(--bg) !important; }
.block-container {
    padding: 0 2rem 3rem !important;
    max-width: 100% !important;
}

/* ── LOGIN PAGE top spacing ───────────────────────────────────── */
.login-page-wrapper {
    padding-bottom: 40px;
}

/* ── SIDEBAR ──────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--text) !important;
    border-right: none !important;
    box-shadow: 4px 0 24px rgba(0,0,0,.15) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
}
[data-testid="stSidebarContent"] {
    background: var(--text) !important;
    padding: 0 !important;
}

/* sidebar text overrides */
[data-testid="stSidebar"] * { color: #e8ede9 !important; }
[data-testid="stSidebar"] .stMarkdown p { color: #b0c4b8 !important; }

/* sidebar buttons */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: #c5d8cc !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 10px 16px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    text-align: left !important;
    width: 100% !important;
    transition: all .2s ease !important;
    margin: 2px 0 !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(101,150,123,.25) !important;
    color: #ffffff !important;
    transform: translateX(3px) !important;
}

/* Active nav button — injected via class on wrapper div */
.nav-active > div > button {
    background: var(--primary) !important;
    color: #ffffff !important;
    box-shadow: 0 3px 12px rgba(101,150,123,.4) !important;
}

/* ── Main Inputs ──────────────────────────────────────────────── */
.stTextInput label p,
.stNumberInput label p,
.stSelectbox label p { color: var(--text-2) !important; font-weight: 600 !important; font-size: 13px !important; }

div[data-baseweb="input"],
div[data-baseweb="base-input"] {
    background: var(--surface) !important;
    border-radius: var(--radius-sm) !important;
    border: 1.5px solid var(--border) !important;
}
div[data-baseweb="input"]:focus-within {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(101,150,123,.15) !important;
}
div[data-baseweb="input"] input {
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    font-size: 14px !important;
}

/* Selectbox */
div[data-baseweb="select"] > div {
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text) !important;
}

/* ── Buttons (Main content area) ─────────────────────────────── */
.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    background: var(--primary) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: .3px !important;
    transition: all .25s ease !important;
    box-shadow: var(--shadow-sm) !important;
}
.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    background: var(--primary-dark) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(101,150,123,.35) !important;
}
.stButton > button:active { transform: scale(.98) !important; }

/* Secondary ghost button */
.btn-ghost > div > button {
    background: transparent !important;
    color: var(--primary) !important;
    border: 1.5px solid var(--primary) !important;
    box-shadow: none !important;
}
.btn-ghost > div > button:hover {
    background: var(--primary-bg) !important;
    transform: translateY(-1px) !important;
    box-shadow: none !important;
}

/* ── Divider ──────────────────────────────────────────────────── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Metric widgets ───────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border-radius: var(--radius) !important;
    padding: 20px !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow-sm) !important;
}
[data-testid="stMetricLabel"] { font-size: 12px !important; color: var(--text-3) !important; text-transform: uppercase; letter-spacing: .5px; }
[data-testid="stMetricValue"] { font-size: 28px !important; font-weight: 700 !important; color: var(--text) !important; }

/* ── Alerts ───────────────────────────────────────────────────── */
.stAlert { border-radius: var(--radius-sm) !important; border: none !important; }

/* ── Spinner ──────────────────────────────────────────────────── */
.stSpinner > div { border-top-color: var(--primary) !important; }

/* ── Form ─────────────────────────────────────────────────────── */
[data-testid="stForm"] {
    background: var(--surface) !important;
    border-radius: var(--radius) !important;
    padding: 28px !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ── Custom Component Classes ─────────────────────────────────── */

/* Page Header */
.page-hero {
    background: linear-gradient(135deg, var(--text) 0%, #2d5042 60%, var(--primary-dark) 100%);
    border-radius: var(--radius-lg);
    padding: 48px 52px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.page-hero::before {
    content: '';
    position: absolute; top: -40px; right: -40px;
    width: 280px; height: 280px;
    border-radius: 50%;
    background: rgba(101,150,123,.2);
}
.page-hero::after {
    content: '';
    position: absolute; bottom: -60px; right: 120px;
    width: 160px; height: 160px;
    border-radius: 50%;
    background: rgba(101,150,123,.12);
}
.hero-eyebrow {
    font-size: 12px; font-weight: 600; letter-spacing: 2px;
    text-transform: uppercase; color: var(--primary-light);
    margin-bottom: 12px; display: block;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 40px; font-weight: 800;
    color: #ffffff; line-height: 1.15;
    margin-bottom: 16px;
}
.hero-sub {
    font-size: 15px; color: #a8c4b5;
    line-height: 1.7; max-width: 520px;
}
.hero-badges {
    display: flex; gap: 10px; margin-top: 24px; flex-wrap: wrap;
}
.hero-badge {
    background: rgba(255,255,255,.1);
    border: 1px solid rgba(255,255,255,.15);
    color: #cde0d6; font-size: 12px; font-weight: 500;
    padding: 5px 14px; border-radius: 20px; letter-spacing: .3px;
}

/* Stats strip */
.stats-strip {
    display: grid; grid-template-columns: repeat(4,1fr); gap: 16px;
    margin-bottom: 32px;
}
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px 24px;
    box-shadow: var(--shadow-sm);
    transition: box-shadow .2s, transform .2s;
    position: relative; overflow: hidden;
}
.stat-card:hover { box-shadow: var(--shadow); transform: translateY(-2px); }
.stat-card::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
}
.stat-card.green::before  { background: var(--primary); }
.stat-card.amber::before  { background: var(--amber); }
.stat-card.rose::before   { background: var(--rose); }
.stat-card.violet::before { background: var(--violet); }
.stat-icon  { font-size: 22px; margin-bottom: 10px; }
.stat-value { font-family: 'Syne',sans-serif; font-size: 28px; font-weight: 700; color: var(--text); }
.stat-label { font-size: 12px; color: var(--text-3); text-transform: uppercase; letter-spacing: .5px; margin-top: 4px; }

/* Section heading */
.section-head {
    display: flex; align-items: center; gap: 10px;
    margin: 32px 0 18px;
}
.section-head-line {
    flex: 1; height: 1px; background: var(--border);
}
.section-head-text {
    font-family: 'Syne',sans-serif;
    font-size: 13px; font-weight: 700; color: var(--text-3);
    text-transform: uppercase; letter-spacing: 1.5px;
    white-space: nowrap;
}

/* Laptop Catalog Card */
.laptop-card {
    background: var(--surface);
    border-radius: var(--radius);
    border: 1px solid var(--border);
    padding: 0;
    margin-bottom: 16px;
    box-shadow: var(--shadow-sm);
    transition: box-shadow .25s, transform .25s, border-color .25s;
    overflow: hidden;
    display: flex;
}
.laptop-card:hover {
    box-shadow: var(--shadow-lg);
    transform: translateY(-3px);
    border-color: var(--primary-light);
}
.card-rank-bar {
    width: 6px;
    background: var(--primary);
    flex-shrink: 0;
}
.card-rank-bar.rank-1 { background: linear-gradient(180deg, #f59e0b, #f59e0b88); }
.card-rank-bar.rank-2 { background: linear-gradient(180deg, #94a3b8, #94a3b888); }
.card-rank-bar.rank-3 { background: linear-gradient(180deg, #b45309, #b4530988); }
.card-body { padding: 22px 24px; flex: 1; }
.card-top { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 14px; }
.rank-pill {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 11px; font-weight: 700; letter-spacing: .5px; text-transform: uppercase;
    padding: 4px 12px; border-radius: 20px;
    background: var(--primary-bg); color: var(--accent);
}
.rank-pill.gold   { background: var(--amber-bg);  color: #92400e; }
.rank-pill.silver { background: #f1f5f9; color: #475569; }
.rank-pill.bronze { background: #fef3e2; color: #92400e; }
.score-chip {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--primary-bg); color: var(--accent);
    font-size: 13px; font-weight: 700;
    padding: 6px 14px; border-radius: 20px;
    border: 1px solid rgba(101,150,123,.3);
}
.card-title { font-family: 'Syne',sans-serif; font-size: 17px; font-weight: 700; color: var(--text); margin-bottom: 2px; }
.card-asin  { font-size: 12px; color: var(--text-3); margin-bottom: 16px; font-family: monospace; }
.specs-row  { display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 18px; }
.spec-item  { display: flex; flex-direction: column; gap: 3px; }
.spec-key   { font-size: 10px; text-transform: uppercase; letter-spacing: .8px; color: var(--text-3); font-weight: 600; }
.spec-val   { font-size: 14px; font-weight: 600; color: var(--text-2); }
.card-footer-inner {
    display: flex; align-items: center; justify-content: space-between;
    padding-top: 14px; border-top: 1px solid var(--border);
}
.price-display { font-family: 'Syne',sans-serif; font-size: 22px; font-weight: 800; color: var(--text); }

/* Cluster badge */
.cluster-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: linear-gradient(135deg, var(--primary-bg), #d4ece1);
    border: 1px solid rgba(101,150,123,.4);
    color: var(--accent);
    font-size: 13px; font-weight: 600;
    padding: 8px 18px; border-radius: 30px;
    margin-bottom: 20px;
}

/* Cold-start panel */
.cold-panel {
    background: var(--surface);
    border: 1.5px dashed var(--border-strong);
    border-radius: var(--radius);
    padding: 40px 32px;
    text-align: center;
    margin: 12px 0 24px;
}
.cold-panel-icon  { font-size: 40px; margin-bottom: 12px; }
.cold-panel-title { font-family:'Syne',sans-serif; font-size: 18px; font-weight: 700; color: var(--text); margin-bottom: 8px; }
.cold-panel-desc  { font-size: 14px; color: var(--text-2); line-height: 1.6; }

/* Form card */
.form-card {
    background: var(--surface);
    border-radius: var(--radius);
    border: 1px solid var(--border);
    padding: 32px;
    box-shadow: var(--shadow-sm);
    margin-bottom: 24px;
}
.form-title { font-family:'Syne',sans-serif; font-size: 18px; font-weight: 700; color: var(--text); margin-bottom: 4px; }
.form-sub   { font-size: 13px; color: var(--text-3); margin-bottom: 20px; }

/* Admin metrics */
.admin-kpi {
    background: var(--surface);
    border-radius: var(--radius);
    padding: 28px;
    border: 1px solid var(--border);
    box-shadow: var(--shadow-sm);
    text-align: center;
    position: relative; overflow: hidden;
    transition: box-shadow .2s, transform .2s;
}
.admin-kpi:hover { box-shadow: var(--shadow); transform: translateY(-2px); }
.kpi-icon  { font-size: 32px; margin-bottom: 10px; }
.kpi-value { font-family
:'Syne',sans-serif; font-size: 36px; font-weight: 800; color: var(--text); }
.kpi-label { font-size: 12px; color: var(--text-3); text-transform: uppercase; letter-spacing: .5px; margin-top: 6px; }
.kpi-accent {
    position: absolute; bottom: 0; left: 0; right: 0; height: 4px;
}
.kpi-green  .kpi-accent { background: var(--primary); }
.kpi-amber  .kpi-accent { background: var(--amber); }
.kpi-sky    .kpi-accent { background: var(--sky); }

/* Auth card */
.auth-card {
    background: var(--surface);
    border-radius: var(--radius-lg);
    padding: 40px;
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
    max-width: 460px;
    margin: 0 auto;
}
.auth-icon  { font-size: 36px; margin-bottom: 12px; }
.auth-title { font-family:'Syne',sans-serif; font-size: 26px; font-weight: 800; color: var(--text); margin-bottom: 6px; }
.auth-sub   { font-size: 14px; color: var(--text-2); line-height: 1.6; margin-bottom: 24px; }
.auth-divider { text-align:center; color:var(--text-3); font-size:12px; margin: 16px 0; }

/* Info box */
.info-box {
    background: var(--primary-bg);
    border: 1px solid rgba(101,150,123,.3);
    border-radius: var(--radius-sm);
    padding: 14px 18px;
    display: flex; gap: 10px; align-items: flex-start;
    margin: 12px 0;
}
.info-box-icon { font-size: 16px; margin-top: 1px; }
.info-box-text { font-size: 13px; color: var(--accent); line-height: 1.5; }

/* ── SIDEBAR inner layout ─────────────────────────────────────── */
.sb-logo {
    padding: 28px 24px 20px;
    border-bottom: 1px solid rgba(255,255,255,.08);
    margin-bottom: 8px;
}
.sb-logo-title {
    font-family: 'Syne',sans-serif;
    font-size: 20px; font-weight: 800; color: #ffffff;
}
.sb-logo-sub { font-size: 11px; color: #7aab91; margin-top: 2px; }

.sb-section-label {
    padding: 16px 24px 6px;
    font-size: 10px; font-weight: 700;
    letter-spacing: 1.5px; text-transform: uppercase;
    color: #5a8070 !important;
}
.sb-user-card {
    margin: 12px 16px;
    background: rgba(255,255,255,.06);
    border: 1px solid rgba(255,255,255,.1);
    border-radius: var(--radius-sm);
    padding: 12px 14px;
    display: flex; align-items: center; gap: 10px;
}
.sb-avatar {
    width: 32px; height: 32px; border-radius: 50%;
    background: var(--primary); color: #fff;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 13px; flex-shrink: 0;
}
.sb-user-name  { font-size: 13px; font-weight: 600; color: #e8ede9 !important; }
.sb-user-role  { font-size: 11px; color: #7aab91 !important; }
.sb-nav-area   { padding: 8px 16px; }

/* Tooltip-style version tag */
.sb-version {
    padding: 20px 24px 8px;
    font-size: 11px; color: #3d6050 !important;
    border-top: 1px solid rgba(255,255,255,.06);
    margin-top: 8px;
}

/* ── Login card ───────────────────────────────────────────────── */
.login-card {
    background: #ffffff;
    border-radius: 20px;
    padding: 48px 44px;
    border: 1px solid #e2e8e4;
    box-shadow: 0 16px 48px rgba(0,0,0,.10), 0 4px 16px rgba(0,0,0,.06);
    margin-top: 0;
}
.login-card-icon  { font-size: 32px; margin-bottom: 14px; }
.login-card-title {
    font-family: 'Syne', sans-serif;
    font-size: 26px; font-weight: 800;
    color: #1a2b22; margin-bottom: 8px;
}
.login-card-sub {
    font-size: 14px; color: #8fa89e;
    margin-bottom: 32px; line-height: 1.6;
}

</style>
""", unsafe_allow_html=True)


# ========================================================================================
# SIDEBAR — Navigation
# ========================================================================================
current_page = st.session_state.page
username     = st.session_state.username

with st.sidebar:
    # Logo
    st.markdown("""
    <div class="sb-logo">
        <div class="sb-logo-title">💻 Laptop</div>
        <div class="sb-logo-sub">Sistem Rekomendasi</div>
    </div>
    """, unsafe_allow_html=True)

    # User card
    if username:
        initials = "AD" if username == ADMIN_ID else username[:2].upper()
        role     = "Administrator" if username == ADMIN_ID else "Pengguna Aktif"
        st.markdown(f"""
        <div class="sb-user-card">
            <div class="sb-avatar">{initials}</div>
            <div>
                <div class="sb-user-name">{username}</div>
                <div class="sb-user-role">{role}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="sb-user-card">
            <div class="sb-avatar" style="background:#3d5a4d;">👤</div>
            <div>
                <div class="sb-user-name">Tamu (Guest)</div>
                <div class="sb-user-role">Belum login</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section-label">Navigasi</div>', unsafe_allow_html=True)

    # Helper to render nav button with active state
    def nav_btn(label, target_page, icon):
        is_active = (current_page == target_page)
        css_class = "nav-active" if is_active else ""
        with st.container():
            st.markdown(f'<div class="sb-nav-area {css_class}">', unsafe_allow_html=True)
            if st.button(f"{icon}  {label}", key=f"nav_{target_page}", use_container_width=True):
                go_to(target_page)
            st.markdown('</div>', unsafe_allow_html=True)

    nav_btn("Beranda", "home", "🏠")

    if username:
        if username == ADMIN_ID:
            nav_btn("Dashboard Admin", "admin", "📊")
        else:
            nav_btn("Hasil Rekomendasi", "hasil", "💡")

        st.markdown('<div class="sb-section-label">Akun</div>', unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="sb-nav-area">', unsafe_allow_html=True)
            if st.button("🚪  Log Out", key="nav_logout", use_container_width=True):
                st.session_state.username = None
                reset_rekomendasi_state()
                go_to("home")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="sb-section-label">Akun</div>', unsafe_allow_html=True)
        nav_btn("Login", "login", "🔐")
        nav_btn("Daftar", "daftar", "📝")

    # Version footer
    st.markdown("""
    <div class="sb-version">
        v1.1 · Collaborative Filtering + K-Medoids · Top-N Selectable
    </div>
    """, unsafe_allow_html=True)


# ========================================================================================
# HELPER: Render Laptop Card
# ========================================================================================
def render_laptop_card(row, rank, score_col='prediksi_CF', score_label='Skor CF'):
    rank_bar_cls = {0: 'rank-1', 1: 'rank-2', 2: 'rank-3'}.get(rank, '')
    rank_pill_cls = {0: 'gold', 1: 'silver', 2: 'bronze'}.get(rank, '')
    rank_labels   = {0: '🥇 Terbaik', 1: '🥈 #2', 2: '🥉 #3'}
    rank_text     = rank_labels.get(rank, f'#{rank+1}')
    score_val     = row.get(score_col, 0) or 0
    brand         = row.get('brand', '-')
    laptop_id     = row.get('Laptop_id', row.get('parent_asin', '-'))
    ram           = row.get('RAM', '-')
    hd            = row.get('Hard_Drive', '-')
    cpu           = row.get('CPU_Model', '-')
    price         = row.get('price', 0) or 0

    st.markdown(f"""
    <div class="laptop-card">
        <div class="card-rank-bar {rank_bar_cls}"></div>
        <div class="card-body">
            <div class="card-top">
                <div>
                    <span class="rank-pill {rank_pill_cls}">{rank_text}</span>
                    <div class="card-title">{brand}</div>
                    <div class="card-asin">{laptop_id}</div>
                </div>
                <span class="score-chip">⭐ {score_val:.2f}</span>
            </div>
            <div class="specs-row">
                <div class="spec-item">
                    <span class="spec-key">🧠 RAM</span>
                    <span class="spec-val">{ram} GB</span>
                </div>
                <div class="spec-item">
                    <span class="spec-key">💾 Storage</span>
                    <span class="spec-val">{hd} GB</span>
                </div>
                <div class="spec-item">
                    <span class="spec-key">⚡ CPU</span>
                    <span class="spec-val">{cpu}</span>
                </div>
            </div>
            <div class="card-footer-inner">
                <span class="price-display">${price:,.0f}</span>
                <span style="font-size:12px;color:var(--text-3);">{score_label}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ========================================================================================
# HELPER: Top-N Selector (dipakai berulang di beberapa halaman)
# ========================================================================================
def render_top_n_selector(key, label="📊 Jumlah Rekomendasi (Top-N)"):
    """
    Menampilkan selectbox untuk memilih jumlah (Top-N) laptop yang
    ingin ditampilkan sebagai rekomendasi. Default = 15.
    """
    return st.selectbox(
        label,
        options=TOP_N_OPTIONS,
        index=TOP_N_DEFAULT_INDEX,
        key=key,
        help="Pilih berapa banyak laptop yang ingin ditampilkan sebagai rekomendasi (default 15)."
    )


# ========================================================================================
# HELPER: Cold Start Form (K-Medoids) — dengan Top-N selectable
# ========================================================================================
def render_cold_start_form(form_key="cold_start_form"):
    with st.form(form_key):
        st.markdown('<div class="form-title">🎯 Preferensi Laptop Anda</div>', unsafe_allow_html=True)
        st.markdown('<div class="form-sub">Isi detail di bawah untuk mendapatkan rekomendasi yang paling sesuai.</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            price_input = st.number_input("💰 Budget Harga (USD)", min_value=100, max_value=5000, value=800, step=50)
            ram_input   = st.selectbox("🧠 RAM (GB)", options=[4, 8, 16, 32, 64], index=1)
        with c2:
            hd_input    = st.selectbox("💾 Storage (GB)", options=[128, 256, 512, 1000, 2000], index=2)
            brand_input = st.selectbox("🏷️ Brand", options=brand_options)
        cpu_input = st.selectbox("⚡ CPU Model", options=cpu_options)

        st.markdown("<br>", unsafe_allow_html=True)
        top_n_input = st.selectbox(
            "📊 Jumlah Rekomendasi (Top-N)",
            options=TOP_N_OPTIONS,
            index=TOP_N_DEFAULT_INDEX,
            help="Pilih berapa banyak laptop yang ingin ditampilkan sebagai rekomendasi (default 15)."
        )

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍 Temukan Laptop Terbaik", use_container_width=True, type="primary")
    return submitted, price_input, ram_input, hd_input, brand_input, cpu_input, top_n_input


def process_cold_start_form(price_input, ram_input, hd_input, brand_input, cpu_input, top_n_input=15):
    """
    Mengembalikan (recs, cluster, user_preferences).
    user_preferences (sudah di-scaling) disimpan agar bisa dipakai ulang
    ketika user hanya ingin mengubah Top-N tanpa mengisi ulang form.
    """
    brand_pref     = brand_mapping.get(brand_input)
    cpu_model_pref = cpu_mapping.get(cpu_input)
    if brand_pref is None:
        st.error("Brand tidak ditemukan di data.")
        return None, None, None
    if cpu_model_pref is None:
        st.error("CPU Model tidak ditemukan di data.")
        return None, None, None
    price_pref       = scaler_price.transform([[price_input]])[0][0]
    ram_pref         = scaler_ram.transform([[ram_input]])[0][0]
    hd_pref          = scaler_hd.transform([[hd_input]])[0][0]
    user_preferences = [price_pref, ram_pref, hd_pref, brand_pref, cpu_model_pref]
    with st.spinner("🔍 Sedang mencari laptop terbaik untuk Anda..."):
        recs, cluster = coldStartProblem(st.session_state.username, user_preferences, n=top_n_input)
    return recs, cluster, user_preferences


def render_inline_topn_adjuster(result_key, cluster_key, prefs_key, topn_key, widget_key):
    """
    Menampilkan kontrol untuk mengubah jumlah (Top-N) rekomendasi
    LANGSUNG di halaman hasil, tanpa perlu mengisi ulang form preferensi.

    result_key  : session_state key tempat DataFrame hasil disimpan (mis. 'cs_result')
    cluster_key : session_state key tempat cluster id disimpan (mis. 'user_cluster')
    prefs_key   : session_state key tempat preferensi (sudah discaling) disimpan
    topn_key    : session_state key tempat nilai Top-N saat ini disimpan
    widget_key  : key unik untuk widget selectbox & tombol
    """
    if st.session_state.get(prefs_key) is None:
        return  # tidak ada preferensi tersimpan, tidak bisa dihitung ulang

    current_n = st.session_state.get(topn_key, 15)
    current_index = TOP_N_OPTIONS.index(current_n) if current_n in TOP_N_OPTIONS else TOP_N_DEFAULT_INDEX

    st.markdown("<div style='margin: 4px 0 16px;'></div>", unsafe_allow_html=True)
    col_n, col_btn = st.columns([3, 1])
    with col_n:
        new_n = st.selectbox(
            "📊 Ubah Jumlah Rekomendasi (Top-N) Saat Ini",
            options=TOP_N_OPTIONS,
            index=current_index,
            key=f"{widget_key}_select",
            help="Ganti jumlah laptop yang ditampilkan tanpa mengubah preferensi lain."
        )
    with col_btn:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        update_clicked = st.button("🔄 Perbarui", key=f"{widget_key}_btn", use_container_width=True)

    if update_clicked:
        with st.spinner(f"🔍 Memperbarui rekomendasi menjadi Top-{new_n}..."):
            recs, cluster = coldStartProblem(
                st.session_state.username,
                st.session_state[prefs_key],
                n=new_n
            )
        st.session_state[result_key]  = pd.DataFrame(recs, columns=['Laptop_id', 'prediksi_rating']) if recs else None
        st.session_state[cluster_key] = cluster
        st.session_state[topn_key]    = new_n
        st.rerun()


def section_divider(text):
    st.markdown(f"""
    <div class="section-head">
        <div class="section-head-text">{text}</div>
        <div class="section-head-line"></div>
    </div>
    """, unsafe_allow_html=True)


# ========================================================================================
# HALAMAN HOME
# ========================================================================================
if st.session_state.page == "home":
    st.markdown("""
    <div class="page-hero">
        <span class="hero-eyebrow">✦ Sistem Rekomendasi Berbasis CF</span>
        <div class="hero-title">Temukan Laptop<br>Sempurna Untuk Anda</div>
        <div class="hero-sub">
            Didukung oleh <strong style="color:#8ab5a0;">Collaborative Filtering</strong> dan 
            <strong style="color:#8ab5a0;">K-Medoids Clustering</strong> untuk rekomendasi yang tepat sasaran,
            bahkan untuk pengguna baru.
        </div>
        <div class="hero-badges">
            <span class="hero-badge">🤖 AI-Powered</span>
            <span class="hero-badge">📊 Cold Start Handling</span>
            <span class="hero-badge">🎯 Personalized</span>
            <span class="hero-badge">⚡ Real-time</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    section_divider("Cara Kerja Sistem")
    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc in [
        (c1, "🔍", "Input Preferensi", "Masukkan kebutuhan laptop Anda: budget, RAM, storage, dan brand favorit."),
        (c2, "🤖", "Analisis Cerdas", "Sistem mencocokkan profil Anda dengan pengguna serupa menggunakan Collaborative Filtering."),
        (c3, "💡", "Rekomendasi Tepat", "Dapatkan daftar laptop terbaik yang sesuai dengan kebutuhan dan anggaran Anda."),
    ]:
        with col:
            st.markdown(f"""
            <div class="stat-card green" style="text-align:center;padding:28px 20px;">
                <div style="font-size:32px;margin-bottom:12px;">{icon}</div>
                <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:700;color:var(--text);margin-bottom:8px;">{title}</div>
                <div style="font-size:13px;color:var(--text-2);line-height:1.6;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    _, col_btn, _ = st.columns([1, 1.5, 1])
    with col_btn:
        if st.button("🚀 Mulai Dapatkan Rekomendasi", use_container_width=True, type="primary"):
            go_to("hasil" if st.session_state.username else "login")


# ========================================================================================
# HALAMAN LOGIN
# ========================================================================================
elif st.session_state.page == "login":

    # ── Top spacing wrapper ──────────────────────────────────────
    st.markdown('<div class="login-page-wrapper">', unsafe_allow_html=True)

    _, col_form, _ = st.columns([1, 2, 1])
    with col_form:

        # Card utama login
        st.markdown("""
        <div class="login-card">
            <div class="login-card-icon">👋</div>
            <div class="login-card-title">Selamat Datang!</div>
            <div class="login-card-sub">
                Masukkan User ID Anda untuk melanjutkan dan mendapatkan rekomendasi laptop terbaik.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Spacer antara card dekoratif dan form input
        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

        with st.form("login_form"):
            username_input = st.text_input(
                "🪪 User ID",
                placeholder="Contoh: BUDI, ADMIN12345 ...",
                help="Masukkan ID unik Anda yang sudah terdaftar."
            )
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("🚀  Masuk ke Akun Saya", use_container_width=True, type="primary")

            if submitted:
                if not username_input:
                    st.error("⚠️ User ID tidak boleh kosong.")
                elif username_input == ADMIN_ID:
                    st.session_state.username = username_input
                    st.balloons()
                    go_to("admin")
                elif not is_user_registered(username_input):
                    st.session_state.login_info_msg = (
                        f"User ID **{username_input}** belum terdaftar. Silakan buat akun terlebih dahulu."
                    )
                    go_to("daftar")
                else:
                    st.session_state.username = username_input
                    reset_rekomendasi_state()
                    st.balloons()
                    go_to("hasil")

        st.markdown("""
        <div style="display:flex;align-items:center;gap:12px;margin:20px 0 16px;">
            <div style="flex:1;height:1px;background:#e2e8e4;"></div>
            <div style="font-size:12px;color:#8fa89e;white-space:nowrap;">Belum punya akun?</div>
            <div style="flex:1;height:1px;background:#e2e8e4;"></div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("📝  Daftar Akun Baru", use_container_width=True):
            go_to("daftar")

        st.markdown("""
        <div style="margin-top:20px;text-align:center;font-size:12px;color:#b0c4b8;">
            🔒 Data Anda aman · Tidak diperlukan kata sandi
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # tutup login-page-wrapper


# ========================================================================================
# HALAMAN DAFTAR
# ========================================================================================
elif st.session_state.page == "daftar":
    _, col_form, _ = st.columns([1, 1.6, 1])
    with col_form:
        st.markdown("""
        <div class="auth-card">
            <div class="auth-icon">📝</div>
            <div class="auth-title">Buat Akun Baru</div>
            <div class="auth-sub">Masukkan ID unik Anda. Hanya huruf kapital (A–Z), tanpa angka dan spasi.</div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.login_info_msg:
            st.warning(st.session_state.login_info_msg)
            st.session_state.login_info_msg = None

        st.markdown('<div class="info-box"><span class="info-box-icon">💡</span><span class="info-box-text">ID hanya huruf kapital A–Z, maksimal 20 karakter, tanpa angka dan spasi. Contoh: <strong>JOKO</strong>, <strong>BUDI</strong></span></div>', unsafe_allow_html=True)

        user_input = st.text_input(
            "🪪 Masukkan ID Baru",
            max_chars=20,
            placeholder="Contoh: BUDI"
        ).upper().strip()

        format_valid = True
        format_error = ""
        if user_input:
            if " " in user_input:
                format_valid = False; format_error = "ID tidak boleh mengandung spasi."
            elif re.search(r'[0-9]', user_input):
                format_valid = False; format_error = "ID tidak boleh mengandung angka."
            elif not re.fullmatch(r'[A-Z]+', user_input):
                format_valid = False; format_error = "ID hanya boleh huruf A–Z."

        if user_input and not format_valid:
            st.error(f"⚠️ {format_error}")

        if st.button("Buat Akun →", use_container_width=True, type="primary"):
            if not user_input:
                st.warning("⚠️ ID wajib diisi.")
            elif not format_valid:
                st.warning("⚠️ Periksa kembali format ID Anda.")
            elif is_user_registered(user_input):
                st.error(f"❌ ID **{user_input}** sudah digunakan. Silakan gunakan ID lain.")
            else:
                st.session_state.pending_register_id = user_input

        if st.session_state.pending_register_id:
            @st.dialog("Konfirmasi Registrasi")
            def confirm_dialog():
                st.markdown(f"Apakah Anda yakin ingin menggunakan ID **{st.session_state.pending_register_id}**?")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Ya, Gunakan", use_container_width=True):
                        new_id  = st.session_state.pending_register_id
                        tanggal = register_new_user(new_id)
                        st.session_state.registered_user_id = new_id
                        st.session_state.registered_date    = tanggal
                        st.session_state.pending_register_id = None
                        st.session_state.username = new_id
                        reset_rekomendasi_state()
                        st.rerun()
                with col2:
                    if st.button("❌ Batal", use_container_width=True):
                        st.session_state.pending_register_id = None
                        st.rerun()
            confirm_dialog()

        if (st.session_state.registered_user_id
                and st.session_state.username == st.session_state.registered_user_id):
            st.success("✅ Akun berhasil dibuat! Mengarahkan ke halaman rekomendasi...")
            st.session_state.registered_user_id = None
            st.session_state.registered_date    = None
            go_to("hasil")

        st.markdown('<div class="auth-divider">— Sudah punya akun? —</div>', unsafe_allow_html=True)
        if st.button("🔐 Login", use_container_width=True):
            go_to("login")


# ========================================================================================
# HALAMAN ADMIN
# ========================================================================================
elif st.session_state.page == "admin":
    if st.session_state.username != ADMIN_ID:
        st.error("🚫 Akses ditolak. Halaman ini hanya untuk administrator.")
        st.stop()

    st.markdown("""
    <div class="page-hero">
        <span class="hero-eyebrow">✦ Administrator Panel</span>
        <div class="hero-title">Dashboard<br>Evaluasi Sistem</div>
        <div class="hero-sub">Pantau performa model rekomendasi menggunakan metode Leave-One-Out Cross Validation.</div>
    </div>
    """, unsafe_allow_html=True)

    # Dataset Stats
    section_divider("Statistik Dataset")
    total_users   = df_user_selected['user_id'].nunique()
    total_laptops = df_user_selected['parent_asin'].nunique()
    total_reviews = len(df_user_selected)
    avg_rating    = df_user_selected['rating'].mean()

    c1, c2, c3, c4 = st.columns(4)
    for col, icon, val, label, cls in [
        (c1, "👥", f"{total_users:,}",   "Total Pengguna",  "kpi-green"),
        (c2, "💻", f"{total_laptops:,}", "Model Laptop",    "kpi-amber"),
        (c3, "⭐", f"{total_reviews:,}", "Total Review",    "kpi-sky"),
        (c4, "📈", f"{avg_rating:.2f}", "Rata-rata Rating", "kpi-green"),
    ]:
        with col:
            st.markdown(f"""
            <div class="admin-kpi {cls}">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-accent"></div>
            </div>
            """, unsafe_allow_html=True)

    # Evaluasi Model
    section_divider("Evaluasi Model — MAE & RMSE")

    st.markdown("""
    <div class="info-box">
        <span class="info-box-icon">ℹ️</span>
        <span class="info-box-text">
            <strong>Leave-One-Out Cross Validation:</strong> Setiap rating user dihilangkan satu per satu, lalu diprediksi oleh sistem.
            Hasil prediksi dibandingkan dengan rating asli untuk menghitung MAE dan RMSE.
            Nilai lebih kecil = sistem lebih akurat.
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_slider, col_btn = st.columns([3, 1])
    with col_slider:
        k_similar_eval = st.slider("🎛️ Top-K Similar Users", min_value=3, max_value=20, value=15, step=1,
                                   help="Jumlah user serupa yang digunakan dalam prediksi rating.")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run_eval = st.button("▶️ Jalankan Evaluasi", type="primary", use_container_width=True)

    if run_eval:
        with st.spinner(f"⏳ Mengevaluasi model dengan LOO (K={k_similar_eval})... Mohon tunggu."):
            mae_val, rmse_val, n_eval = evaluate_model_loo(
                uim=user_item_matrix,
                user_to_index=user_to_index,
                item_to_index=item_to_index,
                get_similar_users_func=get_similar_users,
                predict_rating_func=predict_rating_for_user_item,
                k_similar=k_similar_eval
            )

        st.success(f"✅ Evaluasi selesai! {n_eval} pasangan data valid dievaluasi.")
        st.markdown("<br>", unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="admin-kpi kpi-green">
                <div class="kpi-icon">📉</div>
                <div class="kpi-value">{mae_val:.4f}</div>
                <div class="kpi-label">MAE — Mean Absolute Error</div>
                <div class="kpi-accent"></div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="admin-kpi kpi-amber">
                <div class="kpi-icon">📊</div>
                <div class="kpi-value">{rmse_val:.4f}</div>
                <div class="kpi-label">RMSE — Root Mean Squared Error</div>
                <div class="kpi-accent"></div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="admin-kpi kpi-sky">
                <div class="kpi-icon">🔢</div>
                <div class="kpi-value">{n_eval}</div>
                <div class="kpi-label">Pasangan Data Valid</div>
                <div class="kpi-accent"></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="form-card">
            <div class="form-title">📖 Interpretasi Hasil</div>
            <div class="form-sub">Panduan membaca metrik evaluasi model rekomendasi.</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:12px;">
                <div style="background:var(--primary-bg);border-radius:10px;padding:16px;">
                    <div style="font-weight:700;color:var(--accent);margin-bottom:6px;">📉 MAE</div>
                    <div style="font-size:13px;color:var(--text-2);line-height:1.6;">Rata-rata selisih absolut antara rating prediksi dan aktual (skala 1–5). Semakin kecil, semakin akurat.</div>
                </div>
                <div style="background:var(--amber-bg);border-radius:10px;padding:16px;">
                    <div style="font-weight:700;color:#92400e;margin-bottom:6px;">📊 RMSE</div>
                    <div style="font-size:13px;color:var(--text-2);line-height:1.6;">Memberikan penalti lebih besar pada error yang besar. Berguna untuk mengidentifikasi prediksi yang sangat meleset.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ========================================================================================
# HALAMAN HASIL
# ========================================================================================
elif st.session_state.page == "hasil":

    if st.session_state.username in user_to_index:
        # ── USER LAMA ────────────────────────────────────────────
        user_index = user_to_index[st.session_state.username]

        # Header
        st.markdown(f"""
        <div class="page-hero" style="padding:36px 44px;">
            <span class="hero-eyebrow">✦ Collaborative Filtering</span>
            <div class="hero-title" style="font-size:30px;">Rekomendasi Untuk Anda</div>
            <div class="hero-sub" style="font-size:14px;">Berdasarkan preferensi pengguna dengan profil serupa · Diurutkan dari skor tertinggi</div>
        </div>
        """, unsafe_allow_html=True)

        # Selector Top-N untuk Collaborative Filtering (default 15)
        top_n_cf = render_top_n_selector("top_n_cf", label="📊 Jumlah Rekomendasi Collaborative Filtering (Top-N)")

        # Jumlah similar users disesuaikan agar tetap cukup kandidat
        # walaupun user memilih Top-N yang besar (mis. 30)
        k_similar_cf = max(10, top_n_cf)

        sim_users, sim_scores = get_similar_users(user_index, user_item_matrix, top_k=k_similar_cf)
        rec_items, rec_scores = recommend_items(
            user_index, user_item_matrix, sim_users, sim_scores, top_n=top_n_cf
        )

        if len(rec_items) > 0:
            cf_df = pd.DataFrame({
                'Laptop_id': [item_mapping[idx] for idx in rec_items],
                'prediksi_CF': rec_scores
            })
            hasil_cf = cf_df.merge(df_laptopCore, left_on="Laptop_id", right_on="parent_asin", how="left")

            section_divider(f"Top {len(hasil_cf)} Rekomendasi Collaborative Filtering")
            for i, (_, row) in enumerate(hasil_cf.iterrows()):
                render_laptop_card(row.to_dict(), i, score_col='prediksi_CF', score_label='Skor CF')

            if len(hasil_cf) < top_n_cf:
                st.markdown(f'<div class="info-box"><span class="info-box-icon">ℹ️</span><span class="info-box-text">Sistem hanya menemukan {len(hasil_cf)} laptop yang relevan (dari {top_n_cf} yang diminta) berdasarkan Collaborative Filtering.</span></div>', unsafe_allow_html=True)

            # Popularitas section
            section_divider("Rekomendasi Berdasarkan Popularitas")
            st.markdown('<p style="font-size:14px;color:var(--text-2);margin-bottom:16px;">Ingin rekomendasi tambahan berdasarkan laptop paling populer di kategori yang sesuai?</p>', unsafe_allow_html=True)

            if st.session_state.old_user_cs_result is None:
                if not st.session_state.show_cold_start_form:
                    top_n_pop = render_top_n_selector("top_n_pop", label="📊 Jumlah Rekomendasi Popularitas (Top-N)")
                    if st.button("🌟 Tampilkan Rekomendasi Popularitas"):
                        user_prefs, best_asin = get_cold_start_from_user_history(st.session_state.username)
                        if user_prefs is not None:
                            with st.spinner("Menganalisis riwayat rating..."):
                                recs, cluster = coldStartProblem(st.session_state.username, user_prefs, n=top_n_pop)
                            st.session_state.old_user_cs_result      = pd.DataFrame(recs, columns=['Laptop_id', 'prediksi_rating']) if recs else None
                            st.session_state.old_user_cs_cluster     = cluster
                            st.session_state.old_user_cs_preferences = user_prefs
                            st.session_state.old_user_cs_top_n       = top_n_pop
                            st.rerun()
                        else:
                            st.session_state.show_cold_start_form = True
                            st.rerun()

            if st.session_state.show_cold_start_form:
                sub, p, r, h, b, c, n = render_cold_start_form("cs_form_olduser")
                if sub:
                    recs, cluster, prefs = process_cold_start_form(p, r, h, b, c, n)
                    if recs is not None:
                        st.session_state.old_user_cs_result      = pd.DataFrame(recs, columns=['Laptop_id', 'prediksi_rating'])
                        st.session_state.old_user_cs_cluster     = cluster
                        st.session_state.old_user_cs_preferences = prefs
                        st.session_state.old_user_cs_top_n       = n
                        st.session_state.show_cold_start_form = False
                        st.rerun()

            if st.session_state.old_user_cs_result is not None:
                st.markdown(f'<div class="cluster-badge">🎯 Cluster {st.session_state.old_user_cs_cluster}</div>', unsafe_allow_html=True)
                render_cluster_info(st.session_state.old_user_cs_cluster)

                # Kontrol untuk mengubah Top-N langsung dari halaman ini
                render_inline_topn_adjuster(
                    result_key="old_user_cs_result",
                    cluster_key="old_user_cs_cluster",
                    prefs_key="old_user_cs_preferences",
                    topn_key="old_user_cs_top_n",
                    widget_key="adjust_pop_topn"
                )

                hasil_pop = st.session_state.old_user_cs_result.merge(
                    df_laptopCore, left_on="Laptop_id", right_on="parent_asin", how="left"
                )
                for i, (_, row) in enumerate(hasil_pop.iterrows()):
                    render_laptop_card(row.to_dict(), i, score_col='prediksi_rating', score_label='Popularitas')

                if st.button("🔄 Reset Popularitas"):
                    st.session_state.old_user_cs_result      = None
                    st.session_state.old_user_cs_cluster     = None
                    st.session_state.old_user_cs_preferences = None
                    st.session_state.old_user_cs_top_n       = 15
                    st.session_state.show_cold_start_form = False
                    st.rerun()

        else:
            # Tidak ada CF result
            st.markdown("""
            <div class="cold-panel">
                <div class="cold-panel-icon">⚠️</div>
                <div class="cold-panel-title">Tidak Ada Rekomendasi CF</div>
                <div class="cold-panel-desc">Sistem belum menemukan item baru yang bisa direkomendasikan via Collaborative Filtering. Coba rekomendasi alternatif di bawah.</div>
            </div>
            """, unsafe_allow_html=True)

            section_divider("Rekomendasi Alternatif")
            user_prefs, best_asin = get_cold_start_from_user_history(st.session_state.username)

            if user_prefs is not None:
                st.markdown(f'<div class="info-box"><span class="info-box-icon">📋</span><span class="info-box-text">Kami menemukan laptop favoritmu (<strong>{best_asin}</strong>). Akan dijadikan referensi rekomendasi.</span></div>', unsafe_allow_html=True)
                if not st.session_state.cs_submitted:
                    top_n_hist = render_top_n_selector("top_n_hist", label="📊 Jumlah Rekomendasi (Top-N)")
                    if st.button("🚀 Dapatkan Rekomendasi Berdasarkan Riwayat", type="primary"):
                        with st.spinner("Menganalisis preferensi..."):
                            recs, cluster = coldStartProblem(st.session_state.username, user_prefs, n=top_n_hist)
                        if recs:
                            st.session_state.cs_result      = pd.DataFrame(recs, columns=['Laptop_id', 'prediksi_rating'])
                            st.session_state.user_cluster   = cluster
                            st.session_state.cs_preferences = user_prefs
                            st.session_state.cs_top_n       = top_n_hist
                            st.session_state.cs_submitted   = True
                            st.rerun()
                        else:
                            st.warning("Tidak ada rekomendasi ditemukan.")
            else:
                st.markdown("""
                <div class="cold-panel">
                    <div class="cold-panel-icon">📝</div>
                    <div class="cold-panel-title">Isi Preferensi Manual</div>
                    <div class="cold-panel-desc">Riwayat rating belum cukup. Silakan isi preferensi di bawah ini.</div>
                </div>
                """, unsafe_allow_html=True)
                if not st.session_state.cs_submitted:
                    sub, p, r, h, b, c, n = render_cold_start_form("cs_fallback_form")
                    if sub:
                        recs, cluster, prefs = process_cold_start_form(p, r, h, b, c, n)
                        if recs is not None:
                            st.session_state.cs_result      = pd.DataFrame(recs, columns=['Laptop_id', 'prediksi_rating'])
                            st.session_state.user_cluster   = cluster
                            st.session_state.cs_preferences = prefs
                            st.session_state.cs_top_n       = n
                            st.session_state.cs_submitted   = True
                            st.rerun()

            if st.session_state.cs_submitted and st.session_state.cs_result is not None:
                st.markdown(f'<div class="cluster-badge">🎯 Anda masuk ke Cluster {st.session_state.user_cluster}</div>', unsafe_allow_html=True)
                render_cluster_info(st.session_state.user_cluster)

                # Kontrol untuk mengubah Top-N langsung dari halaman ini
                render_inline_topn_adjuster(
                    result_key="cs_result",
                    cluster_key="user_cluster",
                    prefs_key="cs_preferences",
                    topn_key="cs_top_n",
                    widget_key="adjust_fallback_topn"
                )

                hasil_cs = st.session_state.cs_result.merge(
                    df_laptopCore, left_on="Laptop_id", right_on="parent_asin", how="left"
                )
                for i, (_, row) in enumerate(hasil_cs.iterrows()):
                    render_laptop_card(row.to_dict(), i, score_col='prediksi_rating', score_label='Popularitas')
                if st.button("🔄 Ubah Preferensi"):
                    st.session_state.cs_submitted   = False
                    st.session_state.cs_result      = None
                    st.session_state.cs_preferences = None
                    st.rerun()

    else:
        # ── USER BARU / COLD START ────────────────────────────────
        st.markdown("""
        <div class="page-hero" style="padding:36px 44px;">
            <span class="hero-eyebrow">✦ Cold Start — Pengguna Baru</span>
            <div class="hero-title" style="font-size:30px;">Temukan Laptop Ideal<br>Anda</div>
            <div class="hero-sub" style="font-size:14px;">Isi preferensi di bawah agar sistem dapat merekomendasikan laptop yang paling sesuai dengan kebutuhan Anda.</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="cold-panel">
            <div class="cold-panel-icon">👤</div>
            <div class="cold-panel-title">Pengguna Baru Terdeteksi</div>
            <div class="cold-panel-desc">Akun ini belum memiliki riwayat rating. Kami akan menggunakan preferensi Anda untuk menemukan laptop terbaik menggunakan K-Medoids Clustering.</div>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.cs_submitted:
            sub, p, r, h, b, c, n = render_cold_start_form("cold_start_new_user")
            if sub:
                recs, cluster, prefs = process_cold_start_form(p, r, h, b, c, n)
                if recs is not None:
                    st.session_state.cs_result      = pd.DataFrame(recs, columns=['Laptop_id', 'prediksi_rating'])
                    st.session_state.user_cluster   = cluster
                    st.session_state.cs_preferences = prefs
                    st.session_state.cs_top_n       = n
                    st.session_state.cs_submitted   = True
                    st.rerun()

        if st.session_state.cs_submitted and st.session_state.cs_result is not None:
            st.markdown(f'<div class="cluster-badge">🎯 Anda masuk ke Cluster {st.session_state.user_cluster}</div>', unsafe_allow_html=True)
            render_cluster_info(st.session_state.user_cluster)

            # Kontrol untuk mengubah Top-N (K) langsung di halaman ini,
            # tepat di bawah karakteristik cluster — tanpa perlu isi ulang form.
            render_inline_topn_adjuster(
                result_key="cs_result",
                cluster_key="user_cluster",
                prefs_key="cs_preferences",
                topn_key="cs_top_n",
                widget_key="adjust_newuser_topn"
            )

            section_divider(f"Hasil Rekomendasi Berdasarkan Preferensi ({len(st.session_state.cs_result)} Laptop)")

            hasil_cs = st.session_state.cs_result.merge(
                df_laptopCore, left_on="Laptop_id", right_on="parent_asin", how="left"
            )
            for i, (_, row) in enumerate(hasil_cs.iterrows()):
                render_laptop_card(row.to_dict(), i, score_col='prediksi_rating', score_label='Popularitas')

            if st.button("🔄 Ubah Preferensi"):
                st.session_state.cs_submitted   = False
                st.session_state.cs_result      = None
                st.session_state.cs_preferences = None
                st.rerun()
