import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import MyDataset
from config import Config
from net.FullConnected import NeuralNet
from net.Classification import Classify
from net.Regression import Regress
from torch import optim
from utils import MyCost
import numpy as np

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

conf = Config()

loader_train = DataLoader(MyDataset(mode='train'),
                          batch_size=conf.batch_size,
                          shuffle=True,
                          num_workers=8,
                          drop_last=False)
loader_evaluate = DataLoader(MyDataset(mode='evaluate'),
                             batch_size=conf.batch_size,
                             shuffle=False,
                             num_workers=8,
                             drop_last=True)

net = NeuralNet(conf)
#classifier = Classify(conf)
# regresser = Regress(conf)
if conf.CUDA:
    net = net.cuda()
    #classifier = classifier.cuda()
    # regresser = regresser.cuda()

#classifier.load_state_dict(torch.load('model/Classification.pkl'))
# regresser.load_state_dict(torch.load('model/Regression.pkl'))
#classifier.eval()
# regresser.eval()

if conf.resume:
    print('Loading model......')
    net.load_state_dict(torch.load('model/NeuralNet.pkl'))

opt = optim.Adam(net.parameters(), lr=conf.lr) #,weight_decay=1e-9

lr_scheduler = optim.lr_scheduler.MultiStepLR(opt,
                                              milestones=[20, 40, 60, 80, 100, 200, 400, 600, 800],
                                              gamma=0.5)

reg_cost = MyCost(conf)
mse_cost = nn.MSELoss()
mse_cost_sum = nn.MSELoss(reduction='sum')

max_loss = np.inf

for i in range(conf.EPOCHs):
    net.train()
    loader_t = tqdm(loader_train)
    loader_t.set_description_str(' Training[{}/{}]'.format(i + 1, conf.EPOCHs))
    loss_d = []
    loss_h = []
    total_loss_reg = []
    total = 0
    acc_t = 0
    for data_dic in loader_t:
        del data_dic['payed_time']
        del data_dic['signed_time']
        if conf.CUDA:
            for k in data_dic.keys():
                data_dic[k] = data_dic[k].cuda()
        output_dic = net(data_dic)

        total += data_dic['delta_days'].size(0)

        loss_reg_1 = reg_cost(output_dic['delta_days'].view(-1),
                              data_dic['delta_days'].view(-1).float())
        loss_reg_2 = mse_cost_sum(output_dic['hour'].view(-1),
                                  data_dic['hour'].float().view(-1))

        #loss_reg_s_d = mse_cost(output_dic['shipped_time_day'].view(-1),
        #                        data_dic['shipped_time_day'].view(-1).float())
        #loss_reg_s_h = mse_cost(output_dic['shipped_time_hour'].view(-1),
        #                        data_dic['shipped_time_hour'].float().view(-1))

        #loss_reg_g_d = mse_cost(output_dic['got_time_day'].view(-1),
        #                        data_dic['got_time_day'].view(-1).float())
        #loss_reg_g_h = mse_cost(output_dic['got_time_hour'].view(-1),
        #                        data_dic['got_time_hour'].float().view(-1))

        #loss_reg_d_d = mse_cost(output_dic['dlved_time_day'].view(-1),
        #                        data_dic['dlved_time_day'].view(-1).float())
        #loss_reg_d_h = mse_cost(output_dic['dlved_time_hour'].view(-1),
        #                        data_dic['dlved_time_hour'].float().view(-1))

        #loss_reg_ = loss_reg_s_d + loss_reg_s_h + loss_reg_g_d + loss_reg_g_h + loss_reg_d_d + loss_reg_d_h

        loss_reg = conf.day_weight * loss_reg_1 + loss_reg_2 #+ loss_reg_

        total_loss_reg.append(loss_reg)

        acc_t += (data_dic['delta_days'].long().view(-1) >=
                  (output_dic['delta_days'].view(-1)-conf.day_reduce).long()).sum().float()

        opt.zero_grad()
        loss_reg.backward()
        opt.step()

        loss_d.append(loss_reg_1)
        loss_h.append(loss_reg_2)

    lr_scheduler.step()

    date_acc_t = acc_t / total

    print('Loss:\n\tloss reg: {}\n\tloss day: {}\n\tloss hour: {}'.format(
        sum(total_loss_reg) / total,
        sum(loss_d) / total,
        sum(loss_h) / total))
    print('Date Accuracy is {}'.format(date_acc_t))

    '''
    Below are evaluation
    '''

    net.eval()
    loader_t = tqdm(loader_evaluate)
    loader_t.set_description_str(' Evaluating[{}/{}]'.format(
        i + 1, conf.EPOCHs))
    loss_d_h_pn = {}
    loss_d = []
    loss_h = []
    loss_h_n = []
    loss_h_p = []
    loss_day_n = []
    loss_all = []
    acc_e = 0
    total = 0
    total_hour_n = 0
    total_hour_p = 0

    total_delta_days_n=0
    for data_dic in loader_t:
        del data_dic['payed_time']
        del data_dic['signed_time']
        if conf.CUDA:
            for k in data_dic.keys():
                data_dic[k] = data_dic[k].cuda()
        with torch.no_grad():
            '''
            cls_dic = classifier(data_dic)
            for k in cls_dic.keys():
                data_dic[k] = torch.argmax(cls_dic[k], dim=1)
            # reg_dic = regresser(data_dic)
            # for k in reg_dic.keys():
            #     data_dic[k] = (reg_dic[k].view(-1) + 0.5).long()
            '''
            output_dic = net(data_dic)

        total += data_dic['delta_days'].size(0)


        '''
        count the negetive bias
        '''

        hour_pre_n = output_dic['hour'][output_dic['hour'].view(-1).long()<data_dic['hour']]
        hour_real_n = data_dic['hour'][output_dic['hour'].view(-1).long()<data_dic['hour']]
        
        total_hour_n += hour_pre_n.size(0)

        loss_reg_2_n =  mse_cost((hour_pre_n.view(-1)).long().float(),
                              hour_real_n.long().view(-1).float())  
        

        delta_days_pre_n = output_dic['delta_days'][output_dic['delta_days'].view(-1).long()<data_dic['delta_days']]
        delta_days_real_n = data_dic['delta_days'][output_dic['delta_days'].view(-1).long()<data_dic['delta_days']]
        total_delta_days_n += delta_days_pre_n.size(0)

        loss_reg_1_n =  mse_cost((delta_days_pre_n.view(-1)).long().float(),
                              delta_days_real_n.long().view(-1).float())  

        '''
        count the positive bias
        '''
        hour_pre_p = output_dic['hour'][output_dic['hour'].view(-1).long()>data_dic['hour']]
        hour_real_p = data_dic['hour'][output_dic['hour'].view(-1).long()>data_dic['hour']]
        total_hour_p += hour_pre_p.size(0)

        loss_reg_2_p =  mse_cost((hour_pre_p.view(-1)).long().float(),
                              hour_real_p.long().view(-1).float())  


        


        loss_reg_1 = mse_cost(
            (output_dic['delta_days'].view(-1)).long().float(),
            data_dic['delta_days'].long().view(-1).float())


        loss_reg_2 = mse_cost((output_dic['hour'].view(-1)+0.5).long().float(),
                              data_dic['hour'].long().view(-1).float())

        loss_reg_3 = mse_cost(
            ((output_dic['delta_days'].view(-1)-conf.day_reduce).long() * 24 +
             (output_dic['hour'].view(-1) + 0.5).long()).float(),
            ((data_dic['delta_days']).long().view(-1) * 24 +
             data_dic['hour'].long().view(-1)).float())

        acc_e += (data_dic['delta_days'].long().view(-1) >=
                  (output_dic['delta_days'].view(-1)-conf.day_reduce).long()).sum().float()

        loss_d.append(loss_reg_1)
        loss_day_n.append(loss_reg_1_n)
        loss_h.append(loss_reg_2)
        loss_h_n.append(loss_reg_2_n)
        loss_h_p.append(loss_reg_2_p)
        loss_all.append(loss_reg_3)

    date_acc_e = acc_e / total

    print('lr is {}'.format(lr_scheduler.get_lr()))

    print('Loss:\n\tloss day: {}\n\tloss day negetive: {} ratio: {}\n\tloss hour: {}\n\tloss hour negetive: {}   count: {} ratio: {}\n\tloss hour possitive: {}   count: {} ratio: {}\n\tloss all: {}'.format(
        sum(loss_d) / len(loss_d),
        sum(loss_day_n) / len(loss_day_n),
        total_delta_days_n / total,
        sum(loss_h) / len(loss_h), 
        
        sum(loss_h_n) / len(loss_h_n),
        total_hour_n,total_hour_n/total,
        sum(loss_h_p) / len(loss_h_p),
        total_hour_p,total_hour_p/total,
        torch.sqrt(sum(loss_all) / len(loss_all))))

    print('Date Accuracy is {}'.format(date_acc_e))

    if date_acc_e > conf.threshold and date_acc_t > conf.threshold and torch.sqrt(
            sum(loss_all) / len(loss_all)) < max_loss + 1:
        print('Saving model...')
        max_loss = torch.sqrt(sum(loss_all) / len(loss_all))
        torch.save(net.state_dict(), 'model/NeuralNet.pkl')
    print('Min Loss is {}'.format(max_loss))
    print()
