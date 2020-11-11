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
    # news = "./data/train/news.tsv"
    # user = "./data/train/behaviors.tsv"
    # iterator = MindIterator(param)
    # iterator.open(news, user)

    news = "./data/valid/news.tsv"
    user = "./data/valid/behaviors.tsv"
    test_iterator = MindIterator(param)
    test_iterator.open(news, user)
    my_bag = [
        'user index', 'impression clicked', 'impression title', 'history title'
    ]
    count = 0
    # for bag in tqdm(iterator.batch(data_bag=my_bag)):
    #     count += 1
    # print(count)
    print(test_iterator.size)
    count = 0
    for bag in tqdm(test_iterator.batch(data_bag=my_bag, test=True, size=150)):
        # print(bag['user index'].shape)
        count += 1
    print(count)
    print("MindIterator pass")


if __name__ == "__main__":
    test_mind()