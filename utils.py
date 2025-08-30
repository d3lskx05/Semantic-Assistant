import pandas as pd
import requests
import re
from io import BytesIO
from sentence_transformers import SentenceTransformer
import pymorphy2
import functools
import os
import numpy as np
import time
import psutil

# ---------- загрузка модели ----------
@functools.lru_cache(maxsize=1)
def get_model():
    model_path = "fine_tuned_model"
    model_zip = "fine_tuned_model.zip"
    gdrive_file_id = os.getenv("GDRIVE_MODEL_ID", "1RR15OMLj9vfSrVa1HN-dRU-4LbkdbRRf")

    model = None
    if os.path.exists(model_path):
        print("✅ Используем локальную модель:", model_path)
        model = SentenceTransformer(model_path)
    else:
        try:
            print("📥 Пытаемся загрузить модель с Google Drive...")
            import gdown, zipfile
            gdown.download(f"https://drive.google.com/uc?id={gdrive_file_id}", model_zip, quiet=False)
            with zipfile.ZipFile(model_zip, 'r') as zf:
                zf.extractall(model_path)
            print("✅ Модель успешно загружена!")
            model = SentenceTransformer(model_path)
        except Exception as e:
            print(f"⚠️ Ошибка загрузки с GDrive: {e}")
            print("➡️ Используем fallback: intfloat/multilingual-e5-small")
            model = SentenceTransformer("intfloat/multilingual-e5-small")

    # ✅ Определяем необходимость префиксов
    model_name = getattr(model, "_model_card", None) or getattr(model, "model_card", None) or str(model)
    if "e5" in model_name.lower():
        model.add_prefix = True
    else:
        model.add_prefix = False

    return model

@functools.lru_cache(maxsize=1)
def get_morph():
    return pymorphy2.MorphAnalyzer()

# ---------- служебные функции ----------
def preprocess(text):
    return re.sub(r"\s+", " ", str(text).lower().strip())

def lemmatize(word):
    return get_morph().parse(word)[0].normal_form

@functools.lru_cache(maxsize=10000)
def lemmatize_cached(word):
    return lemmatize(word)

# Пример синонимов и форм глаголов
SYNONYM_GROUPS = [
    ["оплачивала", "оплатила", "платил", "платила"],
]

SYNONYM_DICT = {}
for group in SYNONYM_GROUPS:
    lemmas = {lemmatize(w.lower()) for w in group}
    for lemma in lemmas:
        SYNONYM_DICT[lemma] = lemmas

GITHUB_CSV_URLS = [
    "https://raw.githubusercontent.com/skatzrskx55q/data-assistant-vfiziki/main/data6.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data21.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data31.xlsx"
]

def split_by_slash(phrase: str):
    phrase = phrase.strip()
    parts  = []
    for segment in phrase.split("|"):
        segment = segment.strip()
        if "/" in segment:
            tokens = [p.strip() for p in segment.split("/") if p.strip()]
            if len(tokens) == 2:
                m = re.match(r"^(.*?\b)?(\w+)\s*/\s*(\w+)(\b.*?)?$", segment)
                if m:
                    prefix = (m.group(1) or "").strip()
                    first  = m.group(2).strip()
                    second = m.group(3).strip()
                    suffix = (m.group(4) or "").strip()
                    parts.append(" ".join(filter(None, [prefix, first,  suffix])))
                    parts.append(" ".join(filter(None, [prefix, second, suffix])))
                    continue
            parts.extend(tokens)
        else:
            parts.append(segment)
    return [p for p in parts if p]

def load_excel(url):
    resp = requests.get(url)
    if resp.status_code != 200:
        raise ValueError(f"Ошибка загрузки {url}")

    df = pd.read_excel(BytesIO(resp.content))
    topic_cols = [c for c in df.columns if c.lower().startswith("topics")]
    if not topic_cols:
        raise KeyError("Не найдены колонки topics")

    df["topics"] = df[topic_cols].astype(str).agg(lambda x: [v for v in x if v and v != "nan"], axis=1)
    df["phrase_full"] = df["phrase"]
    df["phrase_list"] = df["phrase"].apply(split_by_slash)
    df = df.explode("phrase_list", ignore_index=True)
    df["phrase"] = df["phrase_list"]
    df["phrase_proc"] = df["phrase"].apply(preprocess)
    df["phrase_lemmas"] = df["phrase_proc"].apply(lambda t: {lemmatize_cached(w) for w in re.findall(r"\w+", t)})

    if "comment" not in df.columns:
        df["comment"] = ""

    return df[["phrase", "phrase_proc", "phrase_full", "phrase_lemmas", "topics", "comment"]]

def load_all_excels():
    dfs = []
    for url in GITHUB_CSV_URLS:
        try:
            dfs.append(load_excel(url))
        except Exception as e:
            print(f"⚠️ Ошибка с {url}: {e}")
    if not dfs:
        raise ValueError("Не удалось загрузить ни одного файла")
    return pd.concat(dfs, ignore_index=True)

# ---------- Эмбеддинги ----------
def compute_phrase_embeddings(df, batch_size: int = 128):
    """
    Пересчитывает эмбеддинги фраз с учётом необходимости префиксов (model.add_prefix)
    """
    model = get_model()
    start_time = time.time()

    # ✅ Добавляем префикс passage: только если нужно
    if getattr(model, "add_prefix", False):
        phrases = [f"passage: {p}" for p in df['phrase_proc'].tolist()]
    else:
        phrases = df['phrase_proc'].tolist()

    # Batch encoding -> numpy.float32
    embeddings_list = []
    for i in range(0, len(phrases), batch_size):
        batch = phrases[i:i+batch_size]
        batch_embs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embeddings_list.append(batch_embs.astype('float32'))

    embeddings = np.vstack(embeddings_list) if embeddings_list else np.zeros((0, model.get_sentence_embedding_dimension()), dtype='float32')

    # Предвычисляем L2-нормы
    norms = np.linalg.norm(embeddings, axis=1)
    norms[norms == 0] = 1e-10

    df.attrs['phrase_embs'] = embeddings
    df.attrs['phrase_embs_norms'] = norms

    elapsed = time.time() - start_time

    # ✅ Возвращаем инфо для логирования
    return {
        "model": str(model),
        "add_prefix": getattr(model, "add_prefix", False),
        "num_phrases": len(df),
        "time_sec": round(elapsed, 2),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_mb": round(psutil.virtual_memory().used / (1024 * 1024), 1)
    }

# ---------- удаление дублей ----------
def deduplicate_results(results):
    best = {}
    for item in results:
        key = item[1]
        score = item[0]
        if key not in best or score > best[key][0]:
            best[key] = item
    return list(best.values())

# ---------- поиск ----------
def semantic_search(query, df, top_k=5, threshold=0.4):
    model = get_model()
    query_proc = preprocess(query)

    # --- 1. С префиксом (только если нужно) ---
    if getattr(model, "add_prefix", False):
        query_emb_pref = model.encode(f"query: {query_proc}", convert_to_numpy=True, show_progress_bar=False).astype("float32")
    else:
        query_emb_pref = model.encode(query_proc, convert_to_numpy=True, show_progress_bar=False).astype("float32")

    # --- 2. Без префикса ---
    query_emb_raw = model.encode(query_proc, convert_to_numpy=True, show_progress_bar=False).astype("float32")

    phrase_embs = df.attrs.get("phrase_embs", None)
    phrase_norms = df.attrs.get("phrase_embs_norms", None)
    if phrase_embs is None or phrase_embs.size == 0:
        return []

    q_norm_pref = np.linalg.norm(query_emb_pref) or 1e-10
    q_norm_raw  = np.linalg.norm(query_emb_raw) or 1e-10

    sims_pref = (phrase_embs @ query_emb_pref) / (phrase_norms * q_norm_pref)
    sims_raw  = (phrase_embs @ query_emb_raw) / (phrase_norms * q_norm_raw)

    sims = (sims_pref + sims_raw) / 2
    sims = np.nan_to_num(sims, neginf=0.0, posinf=0.0)

    top_indices = np.argsort(sims)[::-1][:top_k * 3]
    results = [
        (float(sims[idx]), df.iloc[idx]["phrase_full"], df.iloc[idx]["topics"], df.iloc[idx]["comment"])
        for idx in top_indices if float(sims[idx]) >= threshold
    ]

    return deduplicate_results(results[:top_k])

def keyword_search(query, df):
    query_proc = preprocess(query)
    query_words = re.findall(r"\w+", query_proc)
    query_lemmas = [lemmatize_cached(w) for w in query_words]

    matched = []
    for row in df.itertuples():
        lemma_match = all(
            any(ql in SYNONYM_DICT.get(pl, {pl}) for pl in row.phrase_lemmas)
            for ql in query_lemmas
        )
        partial_match = all(q in row.phrase_proc for q in query_words)
        if lemma_match or partial_match:
            matched.append((row.phrase_full, row.topics, row.comment))

    return deduplicate_results(matched)
