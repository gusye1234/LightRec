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
    from ..data.tools import set_seed
    from .training import params, cal_metric
    from .training import timer
    from torch import optim
    from tqdm import tqdm
    import torch
    import numpy as np
    set_seed(2020)
    param = params(for_model="nrms",
                   file="./data/utils/nrms.yaml",
                   wordDict_file="./data/utils/word_dict_all.pkl",
                   vertDict_file="./data/utils/vert_dict.pkl",
                   subvertDict_file="./data/utils/subvert_dict.pkl",
                   userDict_file="./data/utils/uid2index.pkl",
                   wordEmb_file="./data/utils/embedding_all.npy")
    model_save = "nrms.pth.tar"
    def evaluate(model, test_iterator):
        model.eval()
        critical_size = 150
        label_bag = model.offer_label_bag()
        nrms_bag = model.offer_data_bag()
        nrms_bag.append('user index')
        group = {}
        with torch.no_grad():
            preds = {}
            labels = {}
            for bag in tqdm(
                    test_iterator.batch(data_bag=nrms_bag, test=True,
                                        size=250)):
                truth = bag[label_bag].squeeze()
                pred = model(bag, scale=True,
                             by_user=True).cpu().numpy().squeeze()
                for i, tag in enumerate(bag['user index']):
                    if preds.get(tag, None):
                        preds[tag].append(pred[i])
                    else:
                        preds[tag] = [pred[i]]

                    if labels.get(tag, None):
                        labels[tag].append(truth[i])
                    else:
                        labels[tag] = [truth[i]]
                        assert truth[i] == 1
                del bag
            group_pred = []
            group_label = []
            names = list(preds)
            for name in names:
                group_pred.append(np.asarray(preds[name]))
                group_label.append(np.asarray(labels[name]))
        return cal_metric(group_label, group_pred, metrics=param.metrics)

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device("cpu")
    print(param)
    print("device:", device)
    model = NRMS(param).to(device)

    news = "./data/train/news.tsv"
    user = "./data/train/behaviors.tsv"
    iterator = MindIterator(param)
    iterator.open(news, user)

    news = "./data/valid/news.tsv"
    user = "./data/valid/behaviors.tsv"
    test_iterator = MindIterator(param)
    test_iterator.open(news, user)
    print("Done loading")

    label_bag = model.offer_label_bag()
    nrms_bag = model.offer_data_bag()

    opt = optim.Adam(model.parameters(), lr=param.learning_rate)
    print(evaluate(model, test_iterator))
    for epoch in range(param.epochs):
        model = model.train()
        with timer(name="epoch"):
            count, loss_epoch = 1, 0.
            bar = tqdm(iterator.batch(data_bag=nrms_bag))
            start_loss = None
            for bag in bar:
                pred = model(bag, by_user=True)
                truth = bag[label_bag]
                # print(pred.shape)
                # print(truth.shape, pred.shape)
                loss = model.loss(pred, truth)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if start_loss is None:
                    start_loss = loss.item()
                loss_epoch += loss.item()
                bar.set_description(
                    f"loss: {loss_epoch/count:.3f}/{start_loss:.3f}")
                count += 1
                del bag
                # print(f"    {loss_epoch/count}")
        print()
        loss_epoch /= count
        report = evaluate(model, test_iterator)
        print(f"[{epoch+1}/{param.epochs}]: {loss_epoch:.3f} - {report}")


if __name__ == "__main__":
    # test_param()
    test_nrms()
