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
    from .training import params, cal_metric
    from .training import timer
    from torch import optim
    from tqdm import tqdm
    import torch
    param = params(for_model="nrms",
                   file="./data/utils/nrms.yaml",
                   wordDict_file="./data/utils/word_dict_all.pkl",
                   vertDict_file="./data/utils/vert_dict.pkl",
                   subvertDict_file="./data/utils/subvert_dict.pkl",
                   userDict_file="./data/utils/uid2index.pkl",
                   wordEmb_file="./data/utils/embedding_all.npy")
    def evaluate(model, test_iterator):
        label_bag = model.offer_label_bag()
        nrms_bag = model.offer_data_bag()
        with torch.no_grad():
            preds = []
            labels = []
            for bag in test_iterator.batch(data_bag=nrms_bag, test=True):
                pred = model(bag, scale=True).cpu().numpy()
                truth = bag[label_bag]
                preds.append(pred.squeeze())
                labels.append(truth.squeeze())
        return cal_metric(labels, preds, metrics=param.metrics)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    print(param)
    print(device)
    news = "./data/train/news.tsv"
    user = "./data/train/behaviors.tsv"
    iterator = MindIterator(param)
    model = NRMS(param).to(device)

    news = "./data/valid/news.tsv"
    user = "./data/valid/behaviors.tsv"
    test_iterator = MindIterator(param)
    test_iterator.open(news, user)

    label_bag = model.offer_label_bag()
    nrms_bag = model.offer_data_bag()
    iterator.open(news, user)

    opt = optim.Adam(model.parameters(),
                     lr = param.learning_rate)
    print(evaluate(model, test_iterator))
    for epoch in range(param.epochs):
        with timer(name="epoch"):
            count, loss_epoch = 1, 0.
            for bag in tqdm(iterator.batch(data_bag=nrms_bag)):
                pred = model(bag)
                truth = bag[label_bag]
                # print(pred.shape)
                # print(truth.shape, pred.shape)
                loss = model.loss(pred, truth)
                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_epoch += loss.item()
                count += 1
                # print(f"    {loss_epoch/count}")
        loss_epoch /= count
        report = evaluate(model, test_iterator)
        print(f"[{epoch}/{param.epoch}]: {loss_epoch} - {report}")


if __name__ == "__main__":
    # test_param()
    test_nrms()
