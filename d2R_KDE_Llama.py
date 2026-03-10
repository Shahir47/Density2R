import os
import re
import nltk
import time
import torch
import random
import requests
import numpy as np
import transformers
import pandas as pd
import transformers
from tqdm import tqdm
from sys import stdout
import pyterrier as pt
from pathlib import Path

if not pt.java.started():
    pt.init()

from pyterrier.measures import *
from pyterrier_pisa import PisaIndex
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer


class DensityRerank(pt.Transformer):
    def __init__(self):
        self.BATCH_SIZE = 512
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"*** self.DEVICE: {self.DEVICE} ***\n")
        self.embedder = SentenceTransformer('./embedding_model/all-mpnet-base-v2', device = self.DEVICE)

        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"dtype": torch.bfloat16},
            device_map="auto",
            )
        print("** LLama loaded **")

    def generate_response_llama(self, query, sample):
        prompt = (
            f"Your task is to answer the given question. Directly answer the question. Also, try to answer the question based on the provided samples which are relevant to query. Convey your answer in paragraphs."
            f"Samples: \"{sample}\""
            f"Question: \"{query}\""
        )

        messages = [
            {"role": "system", "content": "You are a deterministic assistant. You are required to generate the same output that is most likely for the same input."},
            {"role": "user", "content": prompt},
        ]

        outputs = self.pipeline(
            messages,
            max_new_tokens=128,
            do_sample = False
        )
        return (outputs[0]["generated_text"][-1]['content'], 0, 0)

    def select_bandwidth_median(self, embeddings: torch.Tensor, sample_size: int = 1000) -> float:
        n = embeddings.size(0)
        if n <= 1:
            return 1.0
        idx = torch.randperm(n)[: min(n, sample_size)]
        sample = embeddings[idx]
        dists = torch.cdist(sample, sample, p=2.0)
        iu = torch.triu_indices(dists.size(0), dists.size(1), offset=1)
        m = dists[iu[0], iu[1]].median().item()
        return m if m > 0 else 1.0

    # KDE
    def compute_kde_density_epanechnikov(self, batch_emb, support_emb, bandwidth):
        dist2 = torch.cdist(batch_emb, support_emb, p=2.0).pow(2)
        u2 = dist2 / (bandwidth ** 2)
        K = torch.clamp(1.0 - u2, min=0.0)
        return K.mean(dim=1)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        all_query_start_time = time.time()

        combined_df = None
        unique_queries = list(df['query'].unique())
        all_prompt_tokens = 0
        all_response_tokens = 0
        positive_count = 0
        ctr = 0

        for query in tqdm(unique_queries, desc="Progress..."):
            ctr += 1
            temp_df = df[df['query'] == query]

            query = temp_df['query'].iloc[0]
            temp_df_text = list(temp_df['text'])
            # sample from BM25
            sample = temp_df_text[:min(10, len(temp_df_text))]

            start_time = time.time()
            ans_to_query, prompt_token_count, response_token_count = self.generate_response_llama(query, sample)
            end_time = time.time()

            all_prompt_tokens += prompt_token_count
            all_response_tokens += response_token_count

            ans_to_query = ans_to_query.replace('\n', '').replace('\r', '').strip()
            ans_to_query = sent_tokenize(ans_to_query)

            seed_emb = list()
            for i in range(0, len(ans_to_query), self.BATCH_SIZE):
                batch_embedding = self.embedder.encode(ans_to_query[i : i+self.BATCH_SIZE], convert_to_tensor=True)
                seed_emb.append(batch_embedding.cpu())
            seed_emb = torch.cat(seed_emb, dim = 0)
            
            rest_emb = list()
            for i in range(0, len(temp_df_text), self.BATCH_SIZE):
                batch_embedding = self.embedder.encode(temp_df_text[i : i+self.BATCH_SIZE], convert_to_tensor=True)
                rest_emb.append(batch_embedding.cpu())
            rest_emb = torch.cat(rest_emb, dim = 0)

            query_emb = self.embedder.encode(query, convert_to_tensor=True).cpu().unsqueeze(0)                 
            seed_emb = torch.cat([query_emb, seed_emb]) # seed_emb + query_emb

            h = self.select_bandwidth_median(seed_emb)
            density = self.compute_kde_density_epanechnikov(rest_emb, seed_emb, h)

            positive_count += int((density > 0).sum().item())
            
            sorted_density, sorted_idx = density.sort(descending=True)

            zero_idx = (density==0).nonzero(as_tuple=True)[0].tolist()
            if(len(zero_idx) > 0):
                first_zero_idx = (sorted_density==0).nonzero(as_tuple=True)[0][0].item()
                sorted_idx_positive = sorted_idx[:first_zero_idx]
                rearranged_rest_nonzero = temp_df.iloc[sorted_idx_positive]
                rearranged_rest_zero = temp_df.iloc[zero_idx]
                rearranged_rest = pd.concat([rearranged_rest_nonzero, rearranged_rest_zero], axis = 0)

            else:
                rearranged_rest = temp_df.iloc[sorted_idx]

            final_rearranged = rearranged_rest

            final_rearranged['score'] = list(range(len(final_rearranged), 0, -1))
            final_rearranged['rank'] = list(range(0, len(final_rearranged)))
            
            if combined_df is None:
                combined_df = final_rearranged
            else:
                combined_df = pd.concat([combined_df, final_rearranged], axis = 0)

            torch.cuda.empty_cache()

        print(f"Avg Prompt Tokens: {all_prompt_tokens/len(unique_queries)}")
        print(f"Avg Response Tokens: {all_response_tokens/len(unique_queries)}")
        print(f"Avg Relevant Documents: {positive_count/len(unique_queries)}")

        all_query_end_time = time.time()
        print(f"Total Run Time for all queries: {all_query_end_time - all_query_start_time:.4f} seconds => {(all_query_end_time - all_query_start_time)/60:.4f} minutes")
        print(f"Avg Run Time for all queries: {(all_query_end_time - all_query_start_time)/len(unique_queries):.4f} seconds => {(all_query_end_time - all_query_start_time)/len(unique_queries)/60:.4f} minutes")

        return combined_df
