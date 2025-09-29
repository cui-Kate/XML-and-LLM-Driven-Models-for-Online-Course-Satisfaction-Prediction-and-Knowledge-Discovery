#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

os.environ["OMP_NUM_THREADS"] = "1"
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import os
import re
import sys
import json
import math
import random
import traceback
from pathlib import Path
from collections import Counter

import chardet
import logging
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import gensim
from gensim import corpora
from gensim.models import CoherenceModel
from gensim.models.phrases import Phrases, Phraser



# --- 可配置部分 ---
INPUT_FILE = r"./encode_result.txt"
OUTPUT_DIR = r"./lda_output"
EXTRACT_FIELD = "text"    
LANG = "en"                
SEED = 42
MIN_TOPICS = 2
MAX_TOPICS = 12
STEP = 1
TOP_WORDS = 12
NO_BELOW = 2
NO_ABOVE = 0.8
KEEP_N = 100000
COHERENCE_METRIC = "c_v"
DO_PYLDAVIS = True

# --- logger 配置 ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
logger = logging.getLogger("LDA_PIPELINE")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(os.path.join(OUTPUT_DIR, "lda_pipeline.log"), encoding="utf-8")
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

random.seed(SEED)

# ------------------------ 读取与解析 ------------------------
def detect_encoding(path):
    raw = Path(path).read_bytes()
    res = chardet.detect(raw)
    return res.get("encoding", None), res

def read_raw_text(path):
    p = Path(path)
    if not p.exists():
        logger.error(f"文件不存在: {path} (cwd={Path.cwd()})")
        raise FileNotFoundError(path)
    enc, info = detect_encoding(path)
    logger.info(f"detect_encoding -> {enc}; details: {info}")
    try:
        text = p.read_text(encoding=enc or 'utf-8', errors='replace')
        logger.info(f"读取成功：文件长度 {len(text)} 字符")
        return text
    except Exception as e:
        logger.warning(f"首次读取失败: {e}; 尝试二次读取")
        b = p.read_bytes()
        text = b.decode('utf-8', errors='replace')
        logger.info("二次读取成功（utf-8 replace）")
        return text

def split_by_index_labels(raw):
    parts = re.split(r'\n(?=\d{3}\s*$)', raw, flags=re.MULTILINE)
    if len(parts) > 1:
        return parts
    parts2 = re.split(r'(?=\n?^\d{3}\s*$)', raw, flags=re.MULTILINE)
    if len(parts2) > 1:
        return parts2
    parts3 = re.split(r'(?=\n?\d{3}\s*\n\s*\[)', raw, flags=re.MULTILINE)
    return parts3

def extract_json_arrays_from_part(part):
    arrays = []
    start = part.find('[')
    end = part.rfind(']')
    if start != -1 and end != -1 and end > start:
        candidate = part[start:end+1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, list):
                arrays.append(obj)
        except Exception:
            # 尝试简单修复常见问题
            cand2 = candidate.replace("'", '"')
            cand2 = re.sub(r',\s*]', ']', cand2)
            try:
                obj = json.loads(cand2)
                if isinstance(obj, list):
                    arrays.append(obj)
            except Exception:
                pass
    return arrays

def extract_texts_from_arrays(arrays, field):
    texts = []
    for arr in arrays:
        for item in arr:
            if not isinstance(item, dict):
                continue
            # 支持嵌套字段 "amc.context"
            if field in item and isinstance(item[field], str) and item[field].strip():
                texts.append(item[field].strip())
            elif '.' in field:
                top, sub = field.split('.',1)
                if top in item and isinstance(item[top], dict):
                    v = item[top].get(sub)
                    if isinstance(v, str) and v.strip():
                        texts.append(v.strip())
            else:
                # 尝试若干常见字段作为备选
                for cand in ["text","rationale","quote"]:
                    v = item.get(cand)
                    if isinstance(v, str) and v.strip():
                        texts.append(v.strip())
    return texts

def brute_force_json_extract(raw):
    arrays = []
    for m in re.finditer(r'\[[\s\S]{20,}\]', raw):
        candidate = m.group(0)
        try:
            obj = json.loads(candidate)
            if isinstance(obj, list):
                arrays.append(obj)
        except Exception:
            try:
                obj = json.loads(candidate.replace("'", '"'))
                if isinstance(obj, list):
                    arrays.append(obj)
            except Exception:
                continue
    return arrays

def normalize_quotes_and_whitespace(s):
    # 将智能引号等替换为 ASCII，去除多余空白
    replace_map = {
        '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
        '\u2014': '-', '\u2013': '-', '\u2026': '...',
        '\xa0': ' '
    }
    for k,v in replace_map.items():
        s = s.replace(k,v)
    # 统一多个空格与换行
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def read_and_extract_text_units(path, extract_field=EXTRACT_FIELD):
    raw = read_raw_text(path)
    parts = split_by_index_labels(raw)
    logger.info(f"split_by_index_labels 得到 {len(parts)} 个部分")
    extracted = []
    arrays_count = 0
    for i, part in enumerate(parts):
        arrays = extract_json_arrays_from_part(part)
        if arrays:
            arrays_count += len(arrays)
            ts = extract_texts_from_arrays(arrays, extract_field)
            for t in ts:
                extracted.append(normalize_quotes_and_whitespace(t))
    logger.info(f"初次分段解析得到 JSON arrays: {arrays_count}；提取 texts count = {len(extracted)}")
    if len(extracted) == 0:
        logger.info("首次解析未提取到文本，尝试暴力解析整个文件中的 JSON arrays ...")
        arrays = brute_force_json_extract(raw)
        logger.info(f"暴力解析找到 arrays: {len(arrays)}")
        for arr in arrays:
            ts = extract_texts_from_arrays([arr], extract_field)
            for t in ts:
                extracted.append(normalize_quotes_and_whitespace(t))
        logger.info(f"暴力解析后 texts count = {len(extracted)}")
    if len(extracted) == 0:
        # 打印样本以便排查（不打印过长）
        logger.error("未能提取到任何文本单元。打印文件头/尾片段供检查（长度受限）。")
        logger.info("FILE HEAD (1600 chars):\n" + raw[:1600])
        logger.info("FILE TAIL (1600 chars):\n" + raw[-1600:])
        raise RuntimeError("未能提取文本：请检查文件格式或修改 EXTRACT_FIELD。")
    return extracted

# ------------------------ 文本预处理（英文） ------------------------
def preprocess_texts_en(texts, nlp):
    stop_words = nlp.Defaults.stop_words
    docs = []
    for doc in nlp.pipe(texts, disable=['ner','parser']):
        toks = []
        for token in doc:
            if token.is_stop or token.is_punct or token.is_space or token.is_digit:
                continue
            lemma = token.lemma_.lower().strip()
            if len(lemma) < 3:
                continue
            if lemma in stop_words:
                continue
            toks.append(lemma)
        docs.append(toks)
    # n-grams
    phrases = Phrases(docs, min_count=2, threshold=10)
    bigram = Phraser(phrases)
    docs_bi = [bigram[doc] for doc in docs]
    trigram = Phraser(Phrases(docs_bi, min_count=2, threshold=10))
    docs_tri = [trigram[doc] for doc in docs_bi]
    return docs_tri

# ------------------------ 构造字典与训练 LDA ------------------------
def compute_dictionary_corpus(tokenized_texts):
    dictionary = corpora.Dictionary(tokenized_texts)
    dictionary.filter_extremes(no_below=NO_BELOW, no_above=NO_ABOVE, keep_n=KEEP_N)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    return dictionary, corpus

def train_lda_models(dictionary, corpus, tokenized_texts):
    results = []
    models = {}
    for k in range(MIN_TOPICS, MAX_TOPICS+1, STEP):
        try:
            logger.info(f"Training LDA: k={k}")
            lda = gensim.models.LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=k,
                random_state=SEED,
                chunksize=100,
                passes=20,
                iterations=400,
                eval_every=None
            )
            coherencemodel = CoherenceModel(model=lda, texts=tokenized_texts, dictionary=dictionary, coherence=COHERENCE_METRIC)
            coherence = coherencemodel.get_coherence()
     
            perplexity = (2 ** (-lda.log_perplexity(corpus))) if len(corpus) > 0 else float('inf')


            logger.info(f" k={k} | coherence({COHERENCE_METRIC})={coherence:.4f} | perplexity(approx)={perplexity:.2f}")
            results.append((k, coherence, perplexity))
            models[k] = lda
        except Exception as e:
            logger.error(f"训练 k={k} 时出错: {e}")
            logger.error(traceback.format_exc())
    df = pd.DataFrame(results, columns=["k","coherence","perplexity"]).sort_values("k").reset_index(drop=True)
    return df, models

# ------------------------ 保存与可视化 ------------------------
def save_metrics_plot(df, outdir):
    if df.empty:
        logger.warning("metrics dataframe is empty, skip plotting.")
        return
    plt.figure(figsize=(8,4))
    plt.plot(df['k'], df['coherence'], marker='o')
    plt.title(f'Coherence ({COHERENCE_METRIC}) vs num_topics')
    plt.xlabel('num_topics (k)')
    plt.ylabel(f'Coherence ({COHERENCE_METRIC})')
    plt.grid(True)
    plt.savefig(os.path.join(outdir, 'coherence_vs_k.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(df['k'], df['perplexity'], marker='o')
    plt.title('Perplexity (approx) vs num_topics')
    plt.xlabel('num_topics (k)')
    plt.ylabel('Perplexity')
    plt.grid(True)
    plt.savefig(os.path.join(outdir, 'perplexity_vs_k.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_topic_word_tables(lda_model, dictionary, k, outdir):
    rows = []
    for tid in range(k):
        terms = lda_model.show_topic(tid, topn=TOP_WORDS)
        row = {"topic_id": tid, "words": "; ".join([f"{w}:{prob:.4f}" for w,prob in terms])}
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(outdir, f'top_words_k{str(k).zfill(2)}.csv'), index=False, encoding='utf-8-sig')

def save_doc_topic_distributions(lda_model, corpus, texts, outdir, k):
    docs = []
    for i, bow in enumerate(corpus):
        dist = lda_model.get_document_topics(bow, minimum_probability=0.0)
        probs = [prob for _tid, prob in sorted(dist, key=lambda x: x[0])]
        row = {"doc_id": i, "text": texts[i]}
        row.update({f"topic_{t}": probs[t] for t in range(len(probs))})
        docs.append(row)
    pd.DataFrame(docs).to_csv(os.path.join(outdir, f'doc_topic_dist_k{str(k).zfill(2)}.csv'), index=False, encoding='utf-8-sig')

# ------------------------ 主流程 ------------------------
def main():
    try:
        logger.info("开始 LDA 管道（增强诊断版）")
        texts = read_and_extract_text_units(INPUT_FILE, EXTRACT_FIELD)
        logger.info(f"成功提取文本单位数量: {len(texts)}")
        # 打印几个样例
        for i, t in enumerate(texts[:10]):
            logger.info(f"sample[{i+1}] (len={len(t)}) : {t[:200]}")

        # 预处理
        if LANG == "en":
            try:
                import spacy
                nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                logger.error("spaCy 模型未安装或加载失败，请运行: python -m spacy download en_core_web_sm")
                raise
            logger.info("开始文本预处理（spaCy lemmatization + n-grams）...")
            tokenized_texts = preprocess_texts_en(texts, nlp)
        else:
            # 若为中文：在这里替换成 jieba 分词与停用词逻辑
            raise NotImplementedError("当前脚本默认英文处理；若为中文，请修改预处理函数为 jieba。")

        # 构建字典与语料
        dictionary, corpus = compute_dictionary_corpus(tokenized_texts)
        logger.info(f"词典大小: {len(dictionary)} ; 文档数: {len(corpus)}")

        # 训练与评估不同 k
        df_metrics, models = train_lda_models(dictionary, corpus, tokenized_texts)
        df_metrics.to_csv(os.path.join(OUTPUT_DIR, "lda_metrics.csv"), index=False, encoding='utf-8-sig')
        save_metrics_plot(df_metrics, OUTPUT_DIR)

        if df_metrics.empty:
            logger.error("未得到任何有效的 LDA 模型（df_metrics 为空）。请查看日志以获取错误细节。")
            return

        # 选最优 k（coherence 最大）
        best_row = df_metrics.loc[df_metrics['coherence'].idxmax()]
        best_k = int(best_row['k'])
        logger.info(f"最佳 k (by coherence={COHERENCE_METRIC}) = {best_k} (coherence={best_row['coherence']:.4f})")

        best_model = models[best_k]
        # 保存输出
        save_topic_word_tables(best_model, dictionary, best_k, OUTPUT_DIR)
        save_doc_topic_distributions(best_model, corpus, texts, OUTPUT_DIR, best_k)
        best_model.save(os.path.join(OUTPUT_DIR, f"lda_model_k{best_k}.model"))
        dictionary.save(os.path.join(OUTPUT_DIR, "dictionary.dict"))
        pd.DataFrame({"text": texts, "tokens": [" ".join(t) for t in tokenized_texts]}).to_csv(os.path.join(OUTPUT_DIR, "tokenized_texts.csv"), index=False, encoding='utf-8-sig')

        # pyLDAvis
        if DO_PYLDAVIS:
            try:
                import pyLDAvis
                import pyLDAvis.gensim_models as gensimvis
                vis = gensimvis.prepare(best_model, corpus, dictionary)
                pyLDAvis.save_html(vis, os.path.join(OUTPUT_DIR, f"pyldavis_k{best_k}.html"))
                logger.info("pyLDAvis HTML 已保存。")
            except Exception as e:
                logger.warning("pyLDAvis 生成失败（可能未安装或环境问题）。错误信息:")
                logger.warning(traceback.format_exc())

        logger.info(f"全部完成。结果保存在目录: {OUTPUT_DIR}")

    except Exception as e:
        logger.error("主流程发生异常，打印 traceback：")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
