# utils2.py
import os
import re
import time
import functools
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import requests
import zipfile

import onnxruntime as ort
from transformers import AutoTokenizer
import pymorphy2

# ---------- глобальные настройки модели ----------
MODEL_CONFIG = {
    "name": "",           # будет задано при загрузке
    "add_prefix": True    # True = использовать query:/passage:, False = чистый текст
}

# ---------- внутренние константы ----------
DEFAULT_GDRIVE_ID = "1lkrvCPIE1wvffIuCSHGtbEz3Epjx5R36"
MODEL_DIR = Path("onnx-user-bge-m3")      # сюда распакуем zip
ZIP_PATH  = Path("onnx-user-bge-m3.zip")  # локальное имя архива

# ---------- загрузка ONNX-модели + токенайзера ----------
@functools.lru_cache(maxsize=1)
def _get_session_and_tokenizer() -> Tuple[ort.InferenceSession, AutoTokenizer, Path]:
    """
    Скачивает ZIP из GDrive (если нужно), распаковывает, находит первый .onnx,
    создаёт InferenceSession и загружает токенайзер.

    Возвращает: (session, tokenizer, model_dir)
    """
    gdrive_file_id = os.getenv("GDRIVE_MODEL_ID", DEFAULT_GDRIVE_ID)

    # 1) скачиваем архив, если его нет
    if not ZIP_PATH.exists() and not MODEL_DIR.exists():
        print("📥 Скачиваем архив модели из Google Drive...")
        import gdown
        gdown.download(f"https://drive.google.com/uc?id={gdrive_file_id}",
                       str(ZIP_PATH), quiet=False, fuzzy=True)

    # 2) распаковка (если модель ещё не распакована)
    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        if ZIP_PATH.exists():
            print("📦 Распаковка архива...")
            with zipfile.ZipFile(ZIP_PATH, "r") as zf:
                zf.extractall(MODEL_DIR)
            # по желанию можно удалить архив
            try:
                ZIP_PATH.unlink()
            except Exception:
                pass

    # 3) ищем первый .onnx
    onnx_files = list(MODEL_DIR.rglob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError("В распакованной папке не найден ни один .onnx файл.")
    model_path = onnx_files[0]
    print(f"✅ Найден ONNX-файл: {model_path}")

    # 4) создаём ONNXRuntime с CPU EP
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(model_path),
                                   sess_options=sess_options,
                                   providers=["CPUExecutionProvider"])

    # 5) токенайзер: сначала пытаемся из распакованной папки, затем из Hub
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        print("✅ Токенайзер загружен из распакованной папки.")
    except Exception:
        print("ℹ️ Токенайзер в архиве не найден, тянем с Hugging Face Hub (deepvk/USER-BGE-M3)...")
        tokenizer = AutoTokenizer.from_pretrained("deepvk/USER-BGE-M3")

    MODEL_CONFIG["name"] = str(MODEL_DIR)
    return session, tokenizer, MODEL_DIR


# ---------- публичная «обёртка» модели ----------
@functools.lru_cache(maxsize=1)
def get_model() -> Tuple[ort.InferenceSession, AutoTokenizer]:
    """
    Совместимый с прежним API загрузчик модели.
    Возвращает (onnx_session, tokenizer).
    """
    session, tokenizer, _ = _get_session_and_tokenizer()
    return session, tokenizer


def _pool_outputs(ort_outputs: List[np.ndarray]) -> np.ndarray:
    """
    Универсальный pooling для разных экспортов:
    - если есть тензор формы (B, S, H) — берём mean по оси=1
    - если (B, H) — уже pooled
    - иначе берём первый тензор, пытаемся привести к (B, H)
    """
    if not ort_outputs:
        raise RuntimeError("Пустой вывод ONNX модели.")

    # Найти «лучший» тензор
    arr = None
    for out in ort_outputs:
        if out.ndim == 3:  # (B, S, H)
            arr = out.mean(axis=1)
            break
    if arr is None:
        # искать (B, H)
        for out in ort_outputs:
            if out.ndim == 2:
                arr = out
                break
    if arr is None:
        # fallback — берём первый и пытаемся редуцировать ось seq
        first = ort_outputs[0]
        if first.ndim == 3:
            arr = first.mean(axis=1)
        elif first.ndim == 2:
            arr = first
        else:
            raise RuntimeError(f"Неизвестная форма выхода ONNX: {first.shape}")

    return arr


def encode_texts(texts: List[str],
                 normalize: bool = True,
                 max_length: int = 512) -> np.ndarray:
    """
    Кодирует список текстов в эмбеддинги через ONNX-модель.
    """
    session, tokenizer = get_model()

    # Токенизация в numpy (важно: onnxruntime ожидает np.int64)
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="np"
    )
    ort_inputs = {k: v for k, v in inputs.items()}

    # Прогон
    ort_outputs = session.run(None, ort_inputs)
    emb = _pool_outputs(ort_outputs)

    emb = emb.astype("float32", copy=False)
    if normalize:
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        emb = emb / norms
    return emb


# ---------- морфология ----------
@functools.lru_cache(maxsize=1)
def get_morph():
    return pymorphy2.MorphAnalyzer()


# ---------- служебные функции ----------
def preprocess(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text).lower().strip())


def lemmatize(word: str) -> str:
    return get_morph().parse(word)[0].normal_form


@functools.lru_cache(maxsize=10000)
def lemmatize_cached(word: str) -> str:
    return lemmatize(word)


# ---------- синонимы ----------
SYNONYM_GROUPS = [
    ["оплачивала", "оплатила", "платил", "платила"],
]

SYNONYM_DICT: Dict[str, set] = {}
for group in SYNONYM_GROUPS:
    lemmas = {lemmatize(w.lower()) for w in group}
    for lemma in lemmas:
        SYNONYM_DICT[lemma] = lemmas


# ---------- загрузка Excel ----------
GITHUB_CSV_URLS = [
    "https://raw.githubusercontent.com/skatzrskx55q/data-assistant-vfiziki/main/data6.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data21.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data31.xlsx"
]


def split_by_slash(phrase: str) -> List[str]:
    phrase = phrase.strip()
    parts: List[str] = []
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


def load_excel(url: str) -> pd.DataFrame:
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


def load_all_excels() -> pd.DataFrame:
    dfs = []
    for url in GITHUB_CSV_URLS:
        try:
            dfs.append(load_excel(url))
        except Exception as e:
            print(f"⚠️ Ошибка с {url}: {e}")
    if not dfs:
        raise ValueError("Не удалось загрузить ни одного файла")
    return pd.concat(dfs, ignore_index=True)


# ---------- пересчёт эмбеддингов ----------
def compute_phrase_embeddings(df: pd.DataFrame, batch_size: int = 128) -> pd.DataFrame:
    """
    Высчитывает эмбеддинги для фраз и кладёт их в df.attrs["phrase_embs"].
    """
    start = time.time()

    if MODEL_CONFIG["add_prefix"]:
        phrases = [f"passage: {p}" for p in df['phrase_proc'].tolist()]
    else:
        phrases = df['phrase_proc'].tolist()

    embeddings_list = []
    for i in range(0, len(phrases), batch_size):
        batch = phrases[i:i + batch_size]
        batch_embs = encode_texts(batch, normalize=True, max_length=512)
        # float32 на всякий случай
        embeddings_list.append(batch_embs.astype("float32"))

    if embeddings_list:
        embeddings = np.vstack(embeddings_list)
    else:
        # не знаем размерности заранее — определим по одиночному прогону
        dummy = encode_texts([" "], normalize=True, max_length=8)
        emb_dim = int(dummy.shape[1])
        embeddings = np.zeros((0, emb_dim), dtype="float32")

    norms = np.linalg.norm(embeddings, axis=1)
    norms[norms == 0] = 1e-10

    df.attrs["phrase_embs"] = embeddings
    df.attrs["phrase_embs_norms"] = norms
    df.attrs["emb_dim"] = embeddings.shape[1] if embeddings.size else 0
    df.attrs["embedding_time"] = time.time() - start
    return df


# ---------- удаление дублей ----------
def _score_of(item: Tuple) -> float:
    return item[0] if len(item) == 4 else 1.0


def _phrase_full_of(item: Tuple) -> str:
    return item[1] if len(item) == 4 else item[0]


def deduplicate_results(results: List[Tuple]) -> List[Tuple]:
    best = {}
    for item in results:
        key = _phrase_full_of(item)
        score = _score_of(item)
        if key not in best or score > _score_of(best[key]):
            best[key] = item
    return list(best.values())


# ---------- поиск ----------
def semantic_search(query: str, df: pd.DataFrame, top_k: int = 5, threshold: float = 0.4) -> List[Tuple]:
    query_proc = preprocess(query)
    if MODEL_CONFIG["add_prefix"]:
        query_text = f"query: {query_proc}"
    else:
        query_text = query_proc

    # (1, D) эмбеддинг запроса
    query_emb = encode_texts([query_text], normalize=True, max_length=512).astype("float32")[0]

    phrase_embs = df.attrs.get("phrase_embs", None)
    phrase_norms = df.attrs.get("phrase_embs_norms", None)
    if phrase_embs is None or phrase_embs.size == 0:
        return []

    q_norm = np.linalg.norm(query_emb) or 1e-10
    sims = (phrase_embs @ query_emb) / (phrase_norms * q_norm)
    sims = np.nan_to_num(sims, neginf=0.0, posinf=0.0)

    top_indices = np.argsort(sims)[::-1][:top_k * 3]
    results = [
        (float(sims[idx]), df.iloc[idx]["phrase_full"], df.iloc[idx]["topics"], df.iloc[idx]["comment"])
        for idx in top_indices if float(sims[idx]) >= threshold
    ]
    return deduplicate_results(results[:top_k])


def keyword_search(query: str, df: pd.DataFrame) -> List[Tuple[str, List[str], str]]:
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
