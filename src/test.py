import torch
from net.FullConnected import NeuralNet
from torch.utils.data import DataLoader
from dataset import MyDataset
from tqdm import tqdm
from config import Config
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
conf = Config()

loader = DataLoader(MyDataset(mode='test_new'), batch_size=1024, num_workers=8)

net = NeuralNet(conf)

net.load_state_dict(torch.load('model/NeuralNet.pkl'))

if conf.CUDA:
    net = net.cuda()

net.eval()

s = []

c = 0

detal_day = []
detal_day_orig = []
hour_reco = []

for data_dic in tqdm(loader):
    pay_times = data_dic['payed_time']
    del data_dic['payed_time']
    if conf.CUDA:
        for k in data_dic.keys():
            data_dic[k] = data_dic[k].cuda()

    with torch.no_grad():
        output_dic = net(data_dic)

    bs = output_dic['delta_days'].size(0)
    day_list = (output_dic['delta_days'].view(-1) -  conf.day_reduce).long()
    tmp_day = []
    for i in range(bs):
        day = day_list[i].cpu().numpy()
        detal_day_orig.append(int(day))
        #day = day+1 if(day == 9) else day 
        tmp_day.append(day)
        
        #detal_day.append(day)

    

    hour_list = (output_dic['hour'].view(-1)+0.5).long()
    for i in range(bs):
        pay_date = datetime.strptime(pay_times[i], '%Y-%m-%d %H:%M:%S')
        #day = (output_dic['delta_days'].view(-1) -  conf.day_reduce).long()[i].cpu().numpy()
        #day = int(day)
        day = tmp_day[i]
        day = int(day)
        if day > 7:
            c += 1
        sig_date = pay_date + relativedelta(days=day)
        hour = hour_list[i]
        hour_reco.append(int(hour.cpu().numpy()))
        sig_date = sig_date.replace(hour=(hour if hour>22 else hour+1))
        s.append(sig_date.strftime('%Y-%m-%d %H'))

detal_dayor_df = pd.DataFrame({'delta_days':detal_day_orig})
detal_dayor_df.to_csv('test_detal_days_or.csv',index=False)
#detal_day_df = pd.DataFrame({'delta_days':detal_day})
#detal_day_df.to_csv('test_detal_days.csv',index=False)
hour_df = pd.DataFrame(
    {'hour':hour_reco}
)
hour_df.to_csv('hour_org.csv',index=False)
print(c)
with open('submission.txt', 'w') as f:
    for val in s:
        f.write(val+'\n')
