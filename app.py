import streamlit as st
from utils import load_all_excels, semantic_search, keyword_search, get_model
import torch  # для работы с тензорами

st.set_page_config(page_title="Проверка фраз ФЛ", layout="centered")
st.title("🤖 Проверка фраз")

@st.cache_data
def get_data():
    df = load_all_excels()
    model = get_model()
    df.attrs['phrase_embs'] = model.encode(df['phrase_proc'].tolist(), convert_to_tensor=True)
    return df

df = get_data()

# 🔘 Все уникальные тематики
all_topics = sorted({topic for topics in df['topics'] for topic in topics})

# --- Вкладки ---
tab1, tab2, tab3 = st.tabs(["🔍 Поиск", "🚫 Не используем", "✅/❌ Да и Нет"])

# ============= TAB 1: ПОИСК =============
with tab1:
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
            if filter_search_by_topics and selected_topics:
                mask = df['topics'].apply(lambda topics: any(t in selected_topics for t in topics))
                search_df = df[mask]

                # Согласуем эмбеддинги с фильтрованным DF
                full_embs = df.attrs.get('phrase_embs', None)
                if full_embs is not None:
                    indices = search_df.index.tolist()
                    if isinstance(full_embs, torch.Tensor):
                        if indices:
                            search_df.attrs['phrase_embs'] = full_embs[indices]
                        else:
                            search_df.attrs['phrase_embs'] = full_embs.new_empty((0, full_embs.size(1)))
                    else:
                        import numpy as np
                        arr = np.asarray(full_embs)
                        search_df.attrs['phrase_embs'] = arr[indices]

            if search_df.empty:
                st.warning("Нет данных для поиска по выбранным тематикам.")
            else:
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


# ============= TAB 2: НЕ ИСПОЛЬЗУЕМ =============
with tab2:
    st.markdown("### 🚫 Тематики, которые **не используем**")
    unused_topics = [
        "Пример тематики 1",
        "Пример тематики 2",
        "Пример тематики 3"
    ]
    for topic in unused_topics:
        st.markdown(f"- {topic}")


# ============= TAB 3: ДА/НЕТ =============
def render_phrases_grid(phrases, cols=3, color="#e0f7fa"):
    rows = [phrases[i:i+cols] for i in range(0, len(phrases), cols)]
    for row in rows:
        cols_streamlit = st.columns(cols)
        for col, phrase in zip(cols_streamlit, row):
            col.markdown(
                f"""<div style="background-color:{color};
                                padding:6px 10px;
                                border-radius:12px;
                                display:inline-block;
                                margin:4px;
                                font-size:14px;">{phrase}</div>""",
                unsafe_allow_html=True
            )

with tab3:
    st.markdown("### ✅ Интерпретации 'ДА'")
    yes_phrases = [
        "Подсказать", "Помню", "Хорошо", "Да", "Ага", "Угу",
        "Да по этому вопросу", "Остались", "Можно", "Жги", "Валяй", "Готов",
        "Ну-ну", "Быстрее", "Проверь", "Проверяй", "Все равно хочу",
        "Подскажите", "Расскажи", "Скажи", "Проверил", "Давал",
        "Я могу", "У меня вопрос есть", "Сказал", "Проконсультируйте", "Пробовала"
    ]
    render_phrases_grid(yes_phrases, cols=3, color="#d1f5d3")

    st.markdown("---")

    st.markdown("### ❌ Интерпретации 'НЕТ'")
    no_phrases = [
        "Не надо", "Не хочу", "Не готов", "Не помню", "Не пробовала"
    ]
    render_phrases_grid(no_phrases, cols=3, color="#f9d6d5")
