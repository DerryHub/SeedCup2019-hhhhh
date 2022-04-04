import torch
from net.FullConnected import NeuralNet
from net.Classification import Classify
from torch.utils.data import DataLoader
from dataset import MyDataset
from tqdm import tqdm
from config import Config
from datetime import datetime
from dateutil.relativedelta import relativedelta


def calculateAllMetrics(real_signed_time_array, pred_signed_time_array):
    if len(real_signed_time_array) != len(pred_signed_time_array):
        print(
            "[Error!] in calculateAllMetrics: len(real_signed_time_array) != len(pred_signed_time_array)"
        )
        return -1

    score_accumulate = 0
    onTime_count = 0
    correct_count = 0
    total_count = len(real_signed_time_array)
    
    total_delta_days_n=0
    
    total_delta_days_n_count=0
    for i in range(total_count):
        real_signed_time = datetime.strptime(real_signed_time_array[i],
                                             "%Y-%m-%d %H:%M:%S")
        real_signed_time = real_signed_time.replace(minute=0)
        real_signed_time = real_signed_time.replace(second=0)
        pred_signed_time = datetime.strptime(pred_signed_time_array[i],
                                             "%Y-%m-%d %H")
        gap = real_signed_time - pred_signed_time
        time_interval = int(
            gap.total_seconds() / 3600)

        delta_day=int(pred_signed_time.day-real_signed_time.day)
        #negetive positive bias
        if (delta_day<0):
            total_delta_days_n+=delta_day**2
            total_delta_days_n_count+=1
            


        # rankScore
        score_accumulate += time_interval**2

        # onTimePercent
        if pred_signed_time.year < 2019:
            onTime_count += 1
        elif pred_signed_time.year == 2019:
            if pred_signed_time.month < real_signed_time.month:
                onTime_count += 1
            elif pred_signed_time.month == real_signed_time.month:
                if pred_signed_time.day <= real_signed_time.day:
                    onTime_count += 1

        # accuracy
        if real_signed_time.year == pred_signed_time.year and real_signed_time.month == pred_signed_time.month and real_signed_time.day == pred_signed_time.day:
            correct_count += 1

    accuracy = float(correct_count / total_count)
    onTimePercent = float(onTime_count / total_count)
    rankScore = float((score_accumulate / total_count)**0.5)
    negetive_delta_day = float(total_delta_days_n / total_delta_days_n_count)
    negetive_delta_ratio = float(total_delta_days_n_count/total_count)


    print('evaluetion \n\t rankScore: {} onTimePercent: {} \n\t accuracy: {} \n\t negetive_delta_day:{} total_delta_days_n_count： {} negetivate_delta_ratio {} '.format(rankScore, onTimePercent, accuracy,negetive_delta_day,total_delta_days_n_count,negetive_delta_ratio))
    return (rankScore, onTimePercent, accuracy)


if __name__ == "__main__":
    conf = Config()

    loader = DataLoader(
        MyDataset(mode='evaluate'), batch_size=1024, num_workers=8)

    net = NeuralNet(conf)
    classifier = Classify(conf)
    if conf.CUDA:
        net = net.cuda()
        classifier = classifier.cuda()

   # classifier.load_state_dict(torch.load('model/Classification.pkl'))
    #classifier.eval()
    net.load_state_dict(torch.load('model/NeuralNet.pkl'))

    if conf.CUDA:
        net = net.cuda()

    net.eval()

    s = ''

    pre = []
    real = []
    

    for data_dic in tqdm(loader):
        pay_times = data_dic['payed_time']
        signed_time = data_dic['signed_time']
        del data_dic['payed_time']
        del data_dic['signed_time']
        if conf.CUDA:
            for k in data_dic.keys():
                data_dic[k] = data_dic[k].cuda()

        with torch.no_grad():
            #cls_dic = classifier(data_dic)
            #for k in cls_dic.keys():
               # data_dic[k] = torch.argmax(cls_dic[k], dim=1)
            output_dic = net(data_dic)

        bs = output_dic['delta_days'].size(0)




        detal_day = []
        day_list =  (output_dic['delta_days'].view(-1) -      conf.day_reduce).long()
        for i in range(bs):
            day = day_list[i].cpu().numpy()
            #print(int(day))
            #day = day-2 if (day >10) else day 
            #day = day+0.5 if (day ==3) else day 
            detal_day.append(day)

        hour_list = (output_dic['hour'].view(-1)+0.5).long()
        for i in range(bs):
            pay_date = datetime.strptime(pay_times[i], '%Y-%m-%d %H:%M:%S')
            #day = (output_dic['delta_days'].view(-1)-conf.day_reduce).long()[i].cpu().numpy()
            day = detal_day[i]
            day = int(day)
            sig_date = pay_date + relativedelta(days=day)
            hour = hour_list[i]
            sig_date = sig_date.replace(
                hour=hour)
            pre.append(sig_date.strftime('%Y-%m-%d %H'))
            real.append(signed_time[i])


    print(calculateAllMetrics(real, pre))
