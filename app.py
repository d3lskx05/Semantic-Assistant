# app_no_faiss.py
import streamlit as st
from utils import load_all_excels, semantic_search, keyword_search, get_model
import torch
import numpy as np
import time
import psutil
import os

st.set_page_config(page_title="Проверка фраз ФЛ", layout="centered")
st.title("🤖 Проверка фраз")

@st.cache_data
def get_data():
    df = load_all_excels()
    # заранее считаем эмбеддинги для всего df
    model = get_model()
    phrase_embs_tensor = model.encode(df["phrase_proc"].tolist(), convert_to_tensor=True)
    df.attrs["phrase_embs"] = phrase_embs_tensor
    return df

df = get_data()

# 🔘 Все уникальные тематики
all_topics = sorted({topic for topics in df['topics'] for topic in topics})
selected_topics = st.multiselect("Фильтр по тематикам (независимо от поиска):", all_topics)
filter_search_by_topics = st.checkbox("Искать только в выбранных тематиках", value=False)

def get_resource_usage():
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 ** 2
    cpu_percent = psutil.cpu_percent(interval=None)
    return mem_mb, cpu_percent

# 📥 Поисковый запрос
query = st.text_input("Введите ваш запрос:")
if query:
    try:
        # 🔹 замер до поиска
        start_time = time.time()
        mem_before, cpu_before = get_resource_usage()

        search_df = df
        if filter_search_by_topics and selected_topics:
            mask = df['topics'].apply(lambda topics: any(t in selected_topics for t in topics))
            search_df = df[mask].copy()

        if search_df.empty:
            st.warning("Нет данных для поиска по выбранным тематикам.")
        else:
            results = semantic_search(query, search_df)
            exact_results = keyword_search(query, search_df)

        # 🔹 замер после поиска
        end_time = time.time()
        mem_after, cpu_after = get_resource_usage()

        # 📊 метрики
        st.markdown(f"""
        ### 📊 Метрики выполнения:
        - ⏱️ Время отклика: **{end_time - start_time:.3f} сек**
        - 💾 RAM до: **{mem_before:.1f} MB**
        - 💾 RAM после: **{mem_after:.1f} MB**
        - 🧮 CPU: **{cpu_after:.1f}%**
        """)

        # 🔍 вывод результатов
        if results:
            st.markdown("### 🔍 Результаты умного поиска:")
            for score, phrase_full, topics, comment in results:
                with st.container():
                    st.write(f"🧠 {phrase_full} | Тематики: {', '.join(topics)} | 🎯 {score:.2f}")
                    if comment and str(comment).strip().lower() != "nan":
                        st.expander("💬 Комментарий").markdown(comment)
        else:
            st.warning("Совпадений не найдено в умном поиске.")

        if exact_results:
            st.markdown("### 🧷 Точный поиск:")
            for phrase, topics, comment in exact_results:
                with st.container():
                    st.write(f"📌 {phrase} | Тематики: {', '.join(topics)}")
                    if comment and str(comment).strip().lower() != "nan":
                        st.expander("💬 Комментарий").markdown(comment)
        else:
            st.info("Ничего не найдено в точном поиске.")
    except Exception as e:
        st.error(f"Ошибка при обработке запроса: {e}")
