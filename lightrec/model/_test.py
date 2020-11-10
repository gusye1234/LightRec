def test_param():
    from .training import params
    paramM = params(
        wordfile="1",
        vecfile="2",
    )
    print(paramM.wordfile, paramM.vecfile)
    yaml_file = "./data/utils/nrms.yaml"
    paramM.open(yaml_file)
    print(paramM)


def test_nrms():
    from .zoo import NRMS
    from ..data.iterator import MindIterator
    from .training import params
    param = params(for_model="nrms",
                   file="./data/utils/nrms.yaml",
                   wordDict_file="./data/utils/word_dict_all.pkl",
                   vertDict_file="./data/utils/vert_dict.pkl",
                   subvertDict_file="./data/utils/subvert_dict.pkl",
                   userDict_file="./data/utils/uid2index.pkl",
                   wordEmb_file="./data/utils/embedding_all.npy")
    print(param)
    news = "./data/valid/news.tsv"
    user = "./data/valid/behaviors.tsv"
    iterator = MindIterator(param)
    model = NRMS(param)
    nrms_bag = model.offer_data_bag()
    iterator.open(news, user)
    for bag in iterator.batch(data_bag=nrms_bag):
        pred = model(bag)
        print(pred.shape)
        exit()


if __name__ == "__main__":
    test_param()
    test_nrms()
