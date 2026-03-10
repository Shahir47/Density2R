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


class MonoT5_2Stage(pt.Transformer):
    def __init__(self):
        self.BATCH_SIZE = 512
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"*** self.DEVICE: {self.DEVICE} ***\n")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        all_query_start_time = time.time()

        combined_df = None
        unique_queries = list(df['query'].unique())

        for query in tqdm(unique_queries, desc="Progress..."):
            temp_df = df[df['query'] == query]

            scorer = MonoT5ReRanker(verbose=False, batch_size=128)

            query = temp_df['query'].iloc[0]
            topk_reranked = scorer.transform(temp_df)
            topk_reranked = topk_reranked.sort_values(by='score', ascending=False)

            final_rearranged = topk_reranked.head(1000)

            if combined_df is None:
                combined_df = final_rearranged
            else:
                combined_df = pd.concat([combined_df, final_rearranged], axis = 0)

        all_query_end_time = time.time()
        print(f"Total Run Time for all queries: {all_query_end_time - all_query_start_time:.4f} seconds => {(all_query_end_time - all_query_start_time)/60:.4f} minutes")
        print(f"Avg Run Time for all queries: {(all_query_end_time - all_query_start_time)/len(unique_queries):.4f} seconds => {(all_query_end_time - all_query_start_time)/len(unique_queries)/60:.4f} minutes")

        return combined_df
