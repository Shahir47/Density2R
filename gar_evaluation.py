import time
import pandas as pd
import pyterrier as pt
if not pt.java.started():
    pt.init()

from GAR.gar import GAR
from pyterrier.measures import *
from pyterrier_pisa import PisaIndex
from GAR.corpus_graph import NpTopKCorpusGraph
from Two_stage_MonoBERT import MonoBertReranker

DATASET_NAME = "irds:msmarco-passage"
TEST_DATASET_NAME = "irds:msmarco-passage/trec-dl-2020/judged"
TOP_K = 1000

def main():
    dataset = pt.get_dataset(DATASET_NAME)
    retriever = PisaIndex.from_dataset('msmarco_passage').bm25(num_results=TOP_K)
    # scorer = pt.text.get_text(dataset, 'text') >> MonoT5ReRanker(verbose=False, batch_size=128)
    scorer = pt.text.get_text(dataset, 'text') >> MonoBertReranker()
    graph = NpTopKCorpusGraph('./Model/corpusgraph_bm25_k16').to_limit_k(8)
    

    test_dataset = pt.get_dataset(TEST_DATASET_NAME)

    start_time = time.time()
    eval_result = pt.Experiment(
        [retriever >> GAR(scorer, graph)],
        test_dataset.get_topics(variant='text'),
        test_dataset.get_qrels(),
        [nDCG@10, nDCG@1000, R(rel=2)@1000],
        names=['bm25 >> GAR_MonoBert'],
        verbose=True
    )
    end_time = time.time()
    
    print(eval_result.to_string())
    print(f"\nRun Time: {end_time - start_time:.4f} seconds => {(end_time - start_time)/60:.4f} minutes")

if __name__ == '__main__':
    main()
    print("Done")