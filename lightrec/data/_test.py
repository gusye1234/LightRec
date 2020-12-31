def test_mind():
    from tqdm import tqdm
    from .iterator import MindIterator
    from ..model import training
    import numpy as np
    param = training.params(
        file="./data/utils/nrms.yaml",
        wordDict_file = "./data/utils/word_dict_all.pkl",
        vertDict_file = "./data/utils/vert_dict.pkl",
        subvertDict_file = "./data/utils/subvert_dict.pkl",
        userDict_file = "./data/utils/uid2index.pkl"
    )
    print(param)
    news = "./data/train/news.tsv"
    user = "./data/train/behaviors.tsv"
    iterator = MindIterator(param)
    iterator.open(news, user)
    print(iterator.size, iterator.user_num, iterator.news_num)

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
    print(test_iterator.size, test_iterator.user_num, test_iterator.news_num)
    count = 0
    click = {}
    for bag in tqdm(test_iterator.batch(data_bag=my_bag, test=True, size=250)):
        # print(bag['user index'].shape)
        label = bag['impression clicked'].squeeze()
        for i, u in enumerate(bag['user index']):
            if click.get(u, None):
                click[u].append(i)
            else:
                click[u] = [i]
        count += 1
    group_click = []
    for name in list(click):
        group_click.append(sum(click[name]))
    group_click = np.asarray(group_click)
    print(np.sum(group_click == 0))
    print("MindIterator pass")


if __name__ == "__main__":
    test_mind()