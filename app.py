import streamlit as st
from utils import load_all_excels, compute_phrase_embeddings, semantic_search, keyword_search, get_model
import numpy as np
import time, psutil, os

st.set_page_config(page_title="Проверка фраз ФЛ", layout="centered")
st.title("🤖 Проверка фраз")

@st.cache_data(show_spinner=False)
def get_data():
    df = load_all_excels()
    return compute_phrase_embeddings(df)

start_time = time.time()
df = get_data()
load_time = time.time() - start_time

# ---- Логгирование ----
DEBUG = True
if DEBUG:
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / 1024 / 1024
    cpu_usage = psutil.cpu_percent(interval=0.1)

    st.sidebar.write(f"**Модель:** {type(get_model().model).__name__}")
    st.sidebar.write(f"**Используются префиксы:** {get_model().add_prefix}")
    st.sidebar.write(f"**Фраз загружено:** {len(df)}")
    st.sidebar.write(f"**Время пересчёта:** {load_time:.2f} сек")
    st.sidebar.write(f"**RAM:** {mem_usage:.2f} MB")
    st.sidebar.write(f"**CPU:** {cpu_usage:.1f}%")

# ---- UI ----
all_topics = sorted({topic for topics in df['topics'] for topic in topics})
selected_topics = st.multiselect("Фильтр по тематикам:", all_topics)
filter_search_by_topics = st.checkbox("Искать только в выбранных тематиках", value=False)

if selected_topics:
    st.markdown("### 📂 Фразы по выбранным тематикам:")
    filtered_df = df[df['topics'].apply(lambda topics: any(t in selected_topics for t in topics))]
    for row in filtered_df.itertuples():
        st.markdown(f"**{row.phrase_full}**  \nТемы: {', '.join(row.topics)}")

query = st.text_input("Введите ваш запрос:")

if query:
    search_df = df
    if filter_search_by_topics and selected_topics:
        mask = df['topics'].apply(lambda topics: any(t in selected_topics for t in topics))
        search_df = df[mask].reset_index(drop=True)
        full_embs = df.attrs.get('phrase_embs', None)
        full_norms = df.attrs.get('phrase_embs_norms', None)
        if full_embs is not None and full_norms is not None:
            indices = list(np.where(mask.values)[0])
            if indices:
                search_df.attrs['phrase_embs'] = full_embs[indices]
                search_df.attrs['phrase_embs_norms'] = full_norms[indices]
            else:
                emb_dim = full_embs.shape[1] if full_embs.size else 0
                search_df.attrs['phrase_embs'] = np.zeros((0, emb_dim), dtype='float32')
                search_df.attrs['phrase_embs_norms'] = np.zeros((0,), dtype='float32')

    results = semantic_search(query, search_df)
    if results:
        st.markdown("### 🔍 Умный поиск:")
        for score, phrase_full, topics, comment in results:
            st.markdown(f"**{phrase_full}** ({score:.2f})  \nТемы: {', '.join(topics)}")
    else:
        st.warning("Совпадений не найдено в умном поиске.")

    exact_results = keyword_search(query, search_df)
    if exact_results:
        st.markdown("### 🧷 Точный поиск:")
        for phrase, topics, comment in exact_results:
            st.markdown(f"**{phrase}**  \nТемы: {', '.join(topics)}")
    else:
        st.info("Ничего не найдено в точном поиске.")
