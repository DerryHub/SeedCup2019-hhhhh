from net.Classification import Classify
import torch
from torch import nn
from torch import optim
from tqdm import tqdm
from config import Config
from torch.utils.data import DataLoader
from dataset import MyDataset

conf = Config()

loader_train = DataLoader(
    MyDataset(mode='train'),
    batch_size=conf.batch_size,
    shuffle=True,
    num_workers=8,
    drop_last=False)
loader_evaluate = DataLoader(
    MyDataset(mode='evaluate'),
    batch_size=conf.batch_size,
    shuffle=True,
    num_workers=8,
    drop_last=False)

net = Classify(conf)
if conf.CUDA:
    net = net.cuda()

opt = optim.Adam(net.parameters(), lr=conf.lr)

lr_scheduler = optim.lr_scheduler.MultiStepLR(
    opt, milestones=[20, 50, 100], gamma=0.1)

cost = nn.CrossEntropyLoss()

max_acc = {}
max_acc['lgst_company'] = 0
max_acc['warehouse_id'] = 0
max_acc['shipped_prov_id'] = 0
max_acc['shipped_city_id'] = 0

for i in range(conf.EPOCHs):
    net.train()
    loader_t = tqdm(loader_train)
    loader_t.set_description_str(' Training[{}/{}]'.format(i + 1, conf.EPOCHs))
    total_loss = []
    total = 0
    acc = {}
    acc['lgst_company'] = 0
    acc['warehouse_id'] = 0
    acc['shipped_prov_id'] = 0
    acc['shipped_city_id'] = 0
    for data_dic in loader_t:
        del data_dic['payed_time']
        del data_dic['signed_time']
        if conf.CUDA:
            for k in data_dic.keys():
                data_dic[k] = data_dic[k].cuda()
        output_dic = net(data_dic)

        total += data_dic['lgst_company'].size(0)

        loss = sum(
            [cost(output_dic[k], data_dic[k]) for k in output_dic.keys()])

        opt.zero_grad()
        loss.backward()
        opt.step()

        for k in output_dic.keys():
            acc[k] += (torch.argmax(output_dic[k],
                                    dim=1) == data_dic[k]).sum().float()

        total_loss.append(loss)

    lr_scheduler.step()

    print('Train Loss is {}'.format(sum(total_loss) / len(total_loss)))
    print(
        'Accuracy:\n\tlgst company: {}\n\twarehouse id: {}\n\tshipped prov id: {}\n\tshipped city id: {}'
        .format(acc['lgst_company'] / total, acc['warehouse_id'] / total,
                acc['shipped_prov_id'] / total,
                acc['shipped_city_id'] / total))

    net.eval()
    loader_t = tqdm(loader_evaluate)
    loader_t.set_description_str(' Evaluating[{}/{}]'.format(
        i + 1, conf.EPOCHs))
    total_loss = []
    total = 0
    acc = {}
    acc['lgst_company'] = 0
    acc['warehouse_id'] = 0
    acc['shipped_prov_id'] = 0
    acc['shipped_city_id'] = 0
    for data_dic in loader_t:
        del data_dic['payed_time']
        del data_dic['signed_time']
        if conf.CUDA:
            for k in data_dic.keys():
                data_dic[k] = data_dic[k].cuda()
        with torch.no_grad():
            output_dic = net(data_dic)

        total += data_dic['lgst_company'].size(0)

        loss = sum(
            [cost(output_dic[k], data_dic[k]) for k in output_dic.keys()])

        total_loss.append(loss)

        for k in output_dic.keys():
            acc[k] += (torch.argmax(output_dic[k],
                                    dim=1) == data_dic[k]).sum().float()

    print('Evaluation Loss is {}'.format(sum(total_loss) / len(total_loss)))
    print(
        'Accuracy:\n\tlgst company: {}\n\twarehouse id: {}\n\tshipped prov id: {}\n\tshipped city id: {}'
        .format(acc['lgst_company'] / total, acc['warehouse_id'] / total,
                acc['shipped_prov_id'] / total,
                acc['shipped_city_id'] / total))

    for k in acc.keys():
        if acc['lgst_company'] / total > max_acc['lgst_company']:
            print('Saving...')
            for k in acc.keys():
                max_acc[k] = acc[k] / total
            torch.save(net.state_dict(), 'model/Classification.pkl')
            break
    print('lr is {}'.format(lr_scheduler.get_lr()))
    print()
