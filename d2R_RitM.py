import os
import re
import nltk
import time
import torch
# import ollama
import random
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from sys import stdout
import pyterrier as pt

if not pt.java.started():
    pt.init()

from openai import OpenAI
from pyterrier.measures import *
from pyterrier_pisa import PisaIndex
# from pyterrier_t5 import MonoT5ReRanker
from Two_stage_MonoBERT import MonoBertReranker
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

client = OpenAI()

def generate_response_openai(query, sample):
    prompt = (
        f"Your task is to answer the given question. Directly answer the question. Also, try to answer the question based on the provided samples which are relevant to query. Convey your answer in paragraphs."
        f"Samples: \"{sample}\""
        f"Question: \"{query}\""
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role":"system","content":"You are an deterministic assistant. You are required to generate the same output that is most likely for the same input"},
                {"role":"user","content":prompt}
            ],
            temperature=0.00,
            max_tokens=128
        )
    except:
        time.sleep(1)
        print("!!! Rate Limit Faced. Trying after 1 second !!!")
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role":"system","content":"You are an deterministic assistant. You are required to generate the same output that is most likely for the same input"},
                {"role":"user","content":prompt}
            ],
            temperature=0.00,
            max_tokens=128
        )

    prompt_token_count = response.usage.prompt_tokens
    response_token_count = response.usage.completion_tokens
    return (response.choices[0].message.content, prompt_token_count, response_token_count)


class DensityRerank(pt.Transformer):
    def __init__(self):
        self.BATCH_SIZE = 512
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"*** self.DEVICE: {self.DEVICE} ***\n")
        self.embedder = SentenceTransformer('./embedding_model/all-mpnet-base-v2', device = self.DEVICE)
        # self.embedder = SentenceTransformer('./embedding_model/all-MiniLM-L6-v2', device = self.DEVICE)

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
            # print(f"query-{ctr}: {query}")
            temp_df = df[df['query'] == query]

            query = temp_df['query'].iloc[0]
            temp_df_text = list(temp_df['text'])
            # sample from BM25
            sample = temp_df_text[:min(10, len(temp_df_text))]

            start_time = time.time()
            ans_to_query, prompt_token_count, response_token_count = generate_response_openai(query, sample)
            end_time = time.time()
            # print(f"Run Time for generating answer: {end_time - start_time:.4f} seconds => {(end_time - start_time)/60:.4f} minutes")

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

            # selective re-ranking
            zero_idx = (density==0).nonzero(as_tuple=True)[0].tolist()
            # scorer = MonoT5ReRanker(verbose=False, batch_size=128)
            scorer = MonoBertReranker()
            flag = True

            if(len(zero_idx) > 0):
                first_zero_idx = (sorted_density==0).nonzero(as_tuple=True)[0][0].item()
                sorted_idx_positive = sorted_idx[:first_zero_idx]
                rearranged_rest_nonzero = temp_df.iloc[sorted_idx_positive]
                
                if(len(rearranged_rest_nonzero) <= 1000):
                    # re-rank the documents with positive density score
                    # rearranged_rest_nonzero = scorer.transform(rearranged_rest_nonzero).sort_values(by='score', ascending=False)
                    rearranged_rest_nonzero = scorer.transform(rearranged_rest_nonzero) # for MonoBERT and scores are already sorted
                    flag = False

                rearranged_rest_zero = temp_df.iloc[zero_idx]
                rearranged_rest = pd.concat([rearranged_rest_nonzero, rearranged_rest_zero], axis = 0)

            else:
                rearranged_rest = temp_df.iloc[sorted_idx]

            final_rearranged = rearranged_rest.head(1000)

            if(flag == True):
                # final_rearranged = scorer.transform(final_rearranged).sort_values(by='score', ascending=False)
                final_rearranged = scorer.transform(final_rearranged)

            final_rearranged = final_rearranged.copy()
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
