import streamlit as st
from utils import load_all_excels, semantic_search, keyword_search, compute_phrase_embeddings, MODEL_CONFIG, get_model
import numpy as np
import psutil

st.set_page_config(page_title="Проверка фраз ФЛ", layout="centered")
st.title("🤖 Проверка фраз")

@st.cache_data(show_spinner=False)
def get_data():
    df = load_all_excels()
    df = compute_phrase_embeddings(df)
    return df

df = get_data()

# 🔧 Логгирование (отладка сбоку)
with st.sidebar:
    st.markdown("### ⚙️ Отладка")
    st.write("Модель:", MODEL_CONFIG["name"])
    st.write("Префиксы:", MODEL_CONFIG["add_prefix"])
    st.write("Фраз загружено:", df.attrs.get("emb_count", 0))
    st.write("Время пересчёта эмбеддингов:", df.attrs.get("embedding_time", "-"), "сек.")
    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=0.1)
    st.write(f"CPU: {cpu}%")
    st.write(f"RAM: {mem.percent}%")

# 🔘 Все уникальные тематики
all_topics = sorted({topic for topics in df['topics'] for topic in topics})
selected_topics = st.multiselect("Фильтр по тематикам (независимо от поиска):", all_topics)
filter_search_by_topics = st.checkbox("Искать только в выбранных тематиках", value=False)

# 📂 Фразы по выбранным тематикам
if selected_topics:
    st.markdown("### 📂 Фразы по выбранным тематикам:")
    filtered_df = df[df['topics'].apply(lambda topics: any(t in selected_topics for t in topics))]
    for row in filtered_df.itertuples():
        with st.container():
            st.markdown(
                f"""<div style="border: 1px solid #e0e0e0; border-radius: 12px; padding: 16px; margin-bottom: 12px; background-color: #f9f9f9; box-shadow: 0 2px 6px rgba(0,0,0,0.05);">
                    <div style="font-size: 18px; font-weight: 600; color: #333;">📝 {row.phrase_full}</div>
                    <div style="margin-top: 4px; font-size: 14px; color: #666;">🔖 Тематики: <strong>{', '.join(row.topics)}</strong></div>
                </div>""",
                unsafe_allow_html=True
            )
            if row.comment and str(row.comment).strip().lower() != "nan":
                with st.expander("💬 Комментарий", expanded=False):
                    st.markdown(row.comment)

# 📥 Поисковый запрос
query = st.text_input("Введите ваш запрос:")

if query:
    try:
        search_df = df
        # Фильтр по тематикам
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
        else:
            search_df = df

        if search_df.empty:
            st.warning("Нет данных для поиска по выбранным тематикам.")
        else:
            # Умный поиск
            results = semantic_search(query, search_df)
            if results:
                st.markdown("### 🔍 Результаты умного поиска:")
                for score, phrase_full, topics, comment in results:
                    with st.container():
                        st.markdown(
                            f"""<div style="border: 1px solid #e0e0e0; border-radius: 12px; padding: 16px; margin-bottom: 12px; background-color: #f9f9f9; box-shadow: 0 2px 6px rgba(0,0,0,0.05);">
                                <div style="font-size: 18px; font-weight: 600; color: #333;">🧠 {phrase_full}</div>
                                <div style="margin-top: 4px; font-size: 14px; color: #666;">🔖 Тематики: <strong>{', '.join(topics)}</strong></div>
                                <div style="margin-top: 2px; font-size: 13px; color: #999;">🎯 Релевантность: {score:.2f}</div>
                            </div>""",
                            unsafe_allow_html=True
                        )
                        if comment and str(comment).strip().lower() != "nan":
                            with st.expander("💬 Комментарий", expanded=False):
                                st.markdown(comment)
            else:
                st.warning("Совпадений не найдено в умном поиске.")

            # Точный поиск
            exact_results = keyword_search(query, search_df)
            if exact_results:
                st.markdown("### 🧷 Точный поиск:")
                for phrase, topics, comment in exact_results:
                    with st.container():
                        st.markdown(
                            f"""<div style="border: 1px solid #e0e0e0; border-radius: 12px; padding: 16px; margin-bottom: 12px; background-color: #f9f9f9; box-shadow: 0 2px 6px rgba(0,0,0,0.05);">
                                <div style="font-size: 18px; font-weight: 600; color: #333;">📌 {phrase}</div>
                                <div style="margin-top: 4px; font-size: 14px; color: #666;">🔖 Тематики: <strong>{', '.join(topics)}</strong></div>
                            </div>""",
                            unsafe_allow_html=True
                        )
                        if comment and str(comment).strip().lower() != "nan":
                            with st.expander("💬 Комментарий", expanded=False):
                                st.markdown(comment)
            else:
                st.info("Ничего не найдено в точном поиске.")

    except Exception as e:
        st.error(f"Ошибка при обработке запроса: {e}")
