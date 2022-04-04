import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import MyDataset
from config import Config
from net.Regression import Regress
from torch import optim
from utils import MyCost
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
conf = Config()

loader_train = DataLoader(
    MyDataset(mode='train'),
    batch_size=conf.batch_size,
    shuffle=True,
    num_workers=8)
loader_evaluate = DataLoader(
    MyDataset(mode='evaluate'),
    batch_size=conf.batch_size,
    shuffle=True,
    num_workers=8)

net = Regress(conf)
if conf.CUDA:
    net = net.cuda()

opt = optim.Adam(net.parameters(), lr=conf.lr)

lr_scheduler = optim.lr_scheduler.MultiStepLR(
    opt, milestones=[20, 50, 100], gamma=0.1)

mse_cost = nn.MSELoss(reduction='sum')

max_loss = np.inf

for i in range(conf.EPOCHs):
    net.train()
    loader_t = tqdm(loader_train)
    loader_t.set_description_str(' Training[{}/{}]'.format(i + 1, conf.EPOCHs))
    loss_s_d = []
    loss_s_h = []
    loss_g_d = []
    loss_g_h = []
    loss_d_d = []
    loss_d_h = []
    total_loss_reg = []
    total = 0
    for data_dic in loader_t:
        del data_dic['payed_time']
        del data_dic['signed_time']
        if conf.CUDA:
            for k in data_dic.keys():
                data_dic[k] = data_dic[k].cuda()
        output_dic = net(data_dic)

        total += data_dic['shipped_time_day'].size(0)

        loss_reg_s_d = mse_cost(output_dic['shipped_time_day'].view(-1),
                                data_dic['shipped_time_day'].view(-1).float())
        loss_reg_s_h = mse_cost(output_dic['shipped_time_hour'].view(-1),
                                data_dic['shipped_time_hour'].float().view(-1))

        loss_reg_g_d = mse_cost(output_dic['got_time_day'].view(-1),
                                data_dic['got_time_day'].view(-1).float())
        loss_reg_g_h = mse_cost(output_dic['got_time_hour'].view(-1),
                                data_dic['got_time_hour'].float().view(-1))

        loss_reg_d_d = mse_cost(output_dic['dlved_time_day'].view(-1),
                                data_dic['dlved_time_day'].view(-1).float())
        loss_reg_d_h = mse_cost(output_dic['dlved_time_hour'].view(-1),
                                data_dic['dlved_time_hour'].float().view(-1))

        loss_reg = loss_reg_s_d + loss_reg_s_h + loss_reg_g_d + loss_reg_g_h + loss_reg_d_d + loss_reg_d_h

        loss_s_d.append(loss_reg_s_d)
        loss_s_h.append(loss_reg_s_h)
        loss_g_d.append(loss_reg_g_d)
        loss_g_h.append(loss_reg_g_h)
        loss_d_d.append(loss_reg_d_d)
        loss_d_h.append(loss_reg_d_h)

        total_loss_reg.append(loss_reg)

        opt.zero_grad()
        loss_reg.backward()
        opt.step()

    lr_scheduler.step()

    print(
        'Loss:\n\tloss reg: {}\n\tloss shipped time day: {}\n\tloss shipped time hour: {}\n\tloss got time day: {}\n\tloss got time hour: {}\n\tloss dlved time day: {}\n\tloss dlved time hour: {}'
        .format(
            sum(total_loss_reg) / total,
            sum(loss_s_d) / total,
            sum(loss_s_h) / total,
            sum(loss_g_d) / total,
            sum(loss_g_h) / total,
            sum(loss_d_d) / total,
            sum(loss_d_h) / total,
        ))

    net.eval()
    loader_t = tqdm(loader_evaluate)
    loader_t.set_description_str(' Evaluating[{}/{}]'.format(
        i + 1, conf.EPOCHs))
    loss_s_d = []
    loss_s_h = []
    loss_g_d = []
    loss_g_h = []
    loss_d_d = []
    loss_d_h = []
    total_loss_reg = []
    total = 0
    for data_dic in loader_t:
        del data_dic['payed_time']
        del data_dic['signed_time']
        if conf.CUDA:
            for k in data_dic.keys():
                data_dic[k] = data_dic[k].cuda()
        with torch.no_grad():
            output_dic = net(data_dic)

        total += data_dic['shipped_time_day'].size(0)

        loss_reg_s_d = mse_cost(
            output_dic['shipped_time_day'].view(-1).long().float(),
            data_dic['shipped_time_day'].view(-1).long().float())
        loss_reg_s_h = mse_cost(
            output_dic['shipped_time_hour'].view(-1).long().float(),
            data_dic['shipped_time_hour'].view(-1).long().float())

        loss_reg_g_d = mse_cost(
            output_dic['got_time_day'].view(-1).long().float(),
            data_dic['got_time_day'].view(-1).long().float())
        loss_reg_g_h = mse_cost(
            output_dic['got_time_hour'].view(-1).long().float(),
            data_dic['got_time_hour'].view(-1).long().float())

        loss_reg_d_d = mse_cost(
            output_dic['dlved_time_day'].view(-1).long().float(),
            data_dic['dlved_time_day'].view(-1).long().float())
        loss_reg_d_h = mse_cost(
            output_dic['dlved_time_hour'].view(-1).long().float(),
            data_dic['dlved_time_hour'].view(-1).long().float())

        loss_reg = loss_reg_s_d + loss_reg_s_h + loss_reg_g_d + loss_reg_g_h + loss_reg_d_d + loss_reg_d_h

        loss_s_d.append(loss_reg_s_d)
        loss_s_h.append(loss_reg_s_h)
        loss_g_d.append(loss_reg_g_d)
        loss_g_h.append(loss_reg_g_h)
        loss_d_d.append(loss_reg_d_d)
        loss_d_h.append(loss_reg_d_h)

        total_loss_reg.append(loss_reg)

    print('lr is {}'.format(lr_scheduler.get_lr()))

    print(
        'Loss:\n\tloss reg: {}\n\tloss shipped time day: {}\n\tloss shipped time hour: {}\n\tloss got time day: {}\n\tloss got time hour: {}\n\tloss dlved time day: {}\n\tloss dlved time hour: {}'
        .format(
            sum(total_loss_reg) / total,
            sum(loss_s_d) / total,
            sum(loss_s_h) / total,
            sum(loss_g_d) / total,
            sum(loss_g_h) / total,
            sum(loss_d_d) / total,
            sum(loss_d_h) / total,
        ))

    if sum(total_loss_reg) / total < max_loss:
        print('Saving...')
        max_loss = sum(total_loss_reg) / total
        torch.save(net.state_dict(), 'model/Regression.pkl')

    print()
