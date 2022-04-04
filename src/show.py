#!/usr/bin/env python
# coding: utf-8

# In[16]:


with open('98.txt', 'r') as f:
    l2 = f.readlines()


# In[17]:


import torch
from net.FullConnected import NeuralNet
from torch.utils.data import DataLoader
from dataset import MyDataset
from tqdm import tqdm
from config import Config
from datetime import datetime
from dateutil.relativedelta import relativedelta

if __name__ == "__main__":
    conf = Config()

    loader = DataLoader(
        MyDataset(mode='test_new'), batch_size=1024, num_workers=8)
    net = NeuralNet(conf)

    net.load_state_dict(torch.load('model/NeuralNet.pkl'))

    if conf.CUDA:
        net = net.cuda()

    net.eval()

    s = ''

    pre = []
    real = []

    for data_dic in tqdm(loader):
        pay_times = data_dic['payed_time']
        del data_dic['payed_time']
        if conf.CUDA:
            for k in data_dic.keys():
                try:
                    data_dic[k] = data_dic[k].cuda()
                except:
                    continue

        with torch.no_grad():
            output_dic = net(data_dic)

        bs = output_dic['delta_days'].size(0)

        for i in range(bs):
            pay_date = datetime.strptime(pay_times[i], '%Y-%m-%d %H:%M:%S')
            day = (output_dic['delta_days'].view(-1)).long()[i].cpu().numpy()
            day = int(day)
            sig_date = pay_date + relativedelta(days=day)
            sig_date = sig_date.replace(
                hour=(output_dic['hour'].view(-1) + 0.5).long()[i])
            pre.append(sig_date.strftime('%Y-%m-%d %H'))
    


# In[23]:


a = 0
b = 0
c = 0
for i in range(300000):
    p = pre[i].split()[0]
    p98 = l2[i].split()[0]
    if p > p98:
        a += 1
    elif p == p98:
        b += 1
    else:
        c += 1


# In[24]:


print(a,b,c)


# In[ ]:




