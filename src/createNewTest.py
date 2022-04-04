from tqdm import tqdm
from net.Classification import Classify
from net.Regression import Regress
from config import Config
from torch.utils.data import DataLoader
from dataset import MyDataset
import pandas as pd
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
conf = Config()

df = pd.read_csv('data/test.csv')
dic = df.to_dict(orient='list')

dic['lgst_company'] = []
dic['warehouse_id'] = []
dic['shipped_prov_id'] = []
dic['shipped_city_id'] = []

dic['shipped_time_day'] = []
dic['shipped_time_hour'] = []
dic['got_time_day'] = []
dic['got_time_hour'] = []
dic['dlved_time_day'] = []
dic['dlved_time_hour'] = []

loader = DataLoader(MyDataset(mode='test'), batch_size=512, num_workers=8)

#classifier = Classify(conf)
#regresser = Regress(conf)
#classifier.load_state_dict(torch.load('model/Classification.pkl'))
#regresser.load_state_dict(torch.load('model/Regression.pkl'))

if conf.CUDA:
    #classifier = classifier.cuda()
    #regresser = regresser.cuda()

#classifier.eval()
#regresser.eval()

for data_dic in tqdm(loader):
    del data_dic['payed_time']
    if conf.CUDA:
        for k in data_dic.keys():
            data_dic[k] = data_dic[k].cuda()

    with torch.no_grad():
        #cls_dic = classifier(data_dic)
        #for k in cls_dic.keys():
        #    data_dic[k] = torch.argmax(cls_dic[k], dim=1)
        #reg_dic = regresser(data_dic)

    for k in cls_dic.keys():
        dic[k] += list(torch.argmax(cls_dic[k], dim=1).cpu().numpy())
    for k in reg_dic.keys():
        dic[k] += list((reg_dic[k].view(-1) + 0.5).long().cpu().numpy())

df = pd.DataFrame(dic)
df.to_csv('data/test_new.csv', index=False)
