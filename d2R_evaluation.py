import time
import pandas as pd
import pyterrier as pt
if not pt.java.started():
    pt.init()

from pyterrier.measures import *
from d2R_KDE import DensityRerank
# from d2R_RitM import DensityRerank
from pyterrier_pisa import PisaIndex
from pyterrier_t5 import MonoT5ReRanker

# dataset:
# ---------------------
# irds:msmarco-passage
# irds:beir/trec-covid
# irds:beir/dbpedia-entity
# irds:beir/fiqa
# irds:beir/scifact
# irds:beir/webis-touche2020/v2
# irds:beir/nfcorpus

# test_dataset:
# ---------------------
# irds:msmarco-passage/trec-dl-2019/judged
# irds:msmarco-passage/dev/small
# irds:beir/dbpedia-entity/test
# irds:beir/trec-covid
# irds:beir/scifact/test
# irds:beir/fiqa/test
# irds:beir/nfcorpus/test
# irds:beir/webis-touche2020/v2
    
DATASET_NAME = "irds:beir/nfcorpus"
TEST_DATASET_NAME = "irds:beir/nfcorpus/test"
PISA_INDEX_LOCATION = "./Index/pisa_nfcorpus"
TOP_K = 100

def main():
    dataset = pt.get_dataset(DATASET_NAME)

    retriever = PisaIndex.from_dataset('msmarco_passage').bm25(num_results=TOP_K)
    # retriever = PisaIndex(PISA_INDEX_LOCATION).bm25(num_results=TOP_K)
    scorer = pt.text.get_text(dataset, 'text') >> MonoT5ReRanker(verbose=False, batch_size=128)
    d2R = pt.text.get_text(dataset, 'text') >> DensityRerank()
      

    # # individual evaluation:
    # test_dataset = pt.get_dataset(TEST_DATASET_NAME)
    # query_id = test_dataset.get_topics()['qid'].to_list()
    # bm25_eval = 0
    # monoT5_eval = 0
    # d2R_eval = 0
    # ctr = 0
    # ctr_failed_query = 0
    # for qid in query_id:
    #     ctr += 1
    #     # try:
    #     print(f"query-{ctr}:")
    #     print(f"-----------")

    #     indv_topics = test_dataset.get_topics()[test_dataset.get_topics()['qid'] == qid]
    #     indv_qrels = test_dataset.get_qrels()[test_dataset.get_qrels()['qid'] == qid]
    #     # indv_topics = indv_topics.rename(columns={'text': 'query'})

    #     indv_eval_result = pt.Experiment(
    #         [retriever, retriever >> scorer, retriever >> d2R],
    #         indv_topics,
    #         indv_qrels,
    #         [nDCG@1, nDCG@5, nDCG@10, nDCG@15, nDCG@100, nDCG@1000, MAP(rel=1), P(rel=1)@10, P(rel=1)@50, P(rel=1)@100, P(rel=1)@1000, RR(rel=1)@10, RR(rel=1)@1000, R(rel=1)@100, R(rel=1)@1000, Success(rel=1)@5, Success(rel=1)@10],
    #         names=['bm25', 'bm25 >> MonoT5', 'bm25 >> Density2R'],
    #         verbose=True
    #     )
    #     bm25_eval += indv_eval_result.iloc[0][1:]
    #     monoT5_eval += indv_eval_result.iloc[1][1:]
    #     d2R_eval += indv_eval_result.iloc[2][1:]

    #     print(indv_eval_result.to_string())

    #     if(ctr%500 == 0):
    #         print(f"\n")
    #         print(f">>>>>>>>>>>>>>> Combined result from 0-{ctr}:")
    #         denominator = ctr - ctr_failed_query
    #         print(f"total failed queries: {ctr_failed_query}")
    #         temp_combined = pd.concat([bm25_eval/denominator, monoT5_eval/denominator, d2R_eval/denominator], axis=1)
    #         temp_combined = temp_combined.T
    #         temp_combined.insert(0, 'name', ['bm25', 'bm25 >> MonoT5', 'bm25 >> Density2R'])
    #         print(temp_combined.to_string())

    #     # except:
    #     #     print(f"query: {qid} failed")
    #     #     ctr_failed_query += 1
    #     # print("------------------------------------------------------------------------------------------------------------------------------")

    # denominator = ctr - ctr_failed_query
    # print(f"total failed queries: {ctr_failed_query}")
    # combined_eval = pd.concat([bm25_eval/denominator, monoT5_eval/denominator, d2R_eval/denominator], axis=1)
    # final_df = combined_eval.T
    # final_df.insert(0, 'name', ['bm25', 'bm25 >> MonoT5', 'bm25 >> Density2R'])
    # print(f">>>>>>>>>>>>>>> Final result:------------------------>")
    # print(final_df.to_string())
    
    test_dataset = pt.get_dataset(TEST_DATASET_NAME)

    start_time = time.time()
    eval_result = pt.Experiment(
        [retriever, retriever >> scorer, retriever >> d2R],
        test_dataset.get_topics(variant='text'),
        test_dataset.get_qrels(),
        [nDCG@1, nDCG@10, R(rel=2)@1000],
        names=['bm25', 'bm25 >> MonoT5', 'bm25 >> Density2R'],
        verbose=True
    )
    end_time = time.time()
    
    print(eval_result.to_string())
    print(f"\nRun Time: {end_time - start_time:.4f} seconds => {(end_time - start_time)/60:.4f} minutes")
    print("----------------------------------------------------------------\n")

if __name__ == '__main__':
    print(f"Dataset: {DATASET_NAME}")
    print(f"Test Dataset: {TEST_DATASET_NAME}")
    print(f"TOP_K={TOP_K}")

    for i in range(1):
        print(f"Run {i+1}:")
        print(f"------")
        main()
    print("Done")