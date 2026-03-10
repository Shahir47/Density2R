import pyterrier as pt
if not pt.java.started():
    pt.init()
from pyterrier_pisa import PisaIndex

def build_index(dataset_name, save_as, save_path):

    save_path = save_path + "/" + save_as
    dataset = pt.get_dataset(dataset_name)
    pisa = PisaIndex(save_path)
    pisa.index(dataset.get_corpus_iter())

    print(f"Indexing of {dataset_name} complete.")


def main():
    dataset_name = [
        'irds:msmarco-passage',
        'irds:beir/trec-covid',
        'irds:beir/dbpedia-entity',
        'irds:beir/fiqa',
        'irds:beir/scifact',
        'irds:beir/webis-touche2020/v2',
        'irds:beir/nfcorpus'
    ]

    save_as = [ 
        'msmarco-passage',
        'trec-covid',
        'dbpedia-entity',
        'fiqa',
        'scifact',
        'touche',
        'nfcorpus']

    save_path = "./PisaIndex"

    for i in range(len(dataset_name)):
        build_index(dataset_name[i], save_as[i], save_path)

if __name__ == "__main__":
    main()