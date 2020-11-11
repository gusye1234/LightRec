def test_mind():
    from tqdm import tqdm
    from .iterator import MindIterator
    from ..model import training
    param = training.params(
        file="./data/utils/nrms.yaml",
        wordDict_file = "./data/utils/word_dict_all.pkl",
        vertDict_file = "./data/utils/vert_dict.pkl",
        subvertDict_file = "./data/utils/subvert_dict.pkl",
        userDict_file = "./data/utils/uid2index.pkl"
    )
    print(param)
    news = "./data/valid/news.tsv"
    user = "./data/valid/behaviors.tsv"
    iterator = MindIterator(param)
    iterator.open(news, user)
    my_bag = [
        'user index', 'impression clicked', 'impression title', 'history title'
    ]
    for bag in tqdm(iterator.batch(data_bag=my_bag, test=True)):
        print({name: value.shape for name, value in bag.items()})
    print("MindIterator pass")


if __name__ == "__main__":
    test_mind()