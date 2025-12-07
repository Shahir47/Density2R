import os
import re
import nltk
import time
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sys import stdout
import pyterrier as pt

if not pt.java.started():
    pt.init()

from pyterrier_pisa import PisaIndex
from sklearn.decomposition import PCA
from pyterrier_t5 import MonoT5ReRanker
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class MonoBertReranker(pt.Transformer):
    def __init__(self, model_name="castorini/monobert-large-msmarco-finetune-only", batch_size=128, max_len=512, device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.batch_size, self.max_len = batch_size, max_len

    def transform(self, df):
        out_scores = []
        with torch.no_grad():
            for start in range(0, len(df), self.batch_size):
                batch = df.iloc[start:start+self.batch_size]
                enc = self.tokenizer(
                    batch["query"].tolist(),
                    batch["text"].tolist(),
                    truncation=True,
                    max_length=self.max_len,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)
                logits = self.model(**enc).logits
                scores = logits[:, 1].detach().cpu().tolist()
                out_scores.extend(scores)

        df = df.copy()
        df["score"] = out_scores
        # sort by qid, descending score
        return df.sort_values(["qid", "score"], ascending=[True, False])
