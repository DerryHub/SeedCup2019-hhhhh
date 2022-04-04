import torch
from torch import nn


class Classify(nn.Module):
    def __init__(self, config):
        super(Classify, self).__init__()
        self.em_dic = {}
        self.plat_form_em = nn.Embedding(config.plat_form_range,
                                                config.embadding_dim)
        self.biz_type_em = nn.Embedding(config.biz_type_range,
                                               config.embadding_dim)
        self.cate1_id_em = nn.Embedding(config.cate1_id_range,
                                               config.embadding_dim)
        self.cate2_id_em = nn.Embedding(config.cate2_id_range,
                                               config.embadding_dim)
        self.cate3_id_em = nn.Embedding(config.cate3_id_range,
                                               config.embadding_dim)
        self.seller_uid_em = nn.Embedding(config.seller_uid_range,
                                                 config.embadding_dim)
        self.company_name_em = nn.Embedding(config.company_name_range,
                                                   config.embadding_dim)
        self.rvcr_prov_name_em = nn.Embedding(
            config.rvcr_prov_name_range, config.embadding_dim)
        self.rvcr_city_name_em = nn.Embedding(
            config.rvcr_city_name_range, config.embadding_dim)
        self.payed_hour_em = nn.Embedding(config.payed_hour_range, config.embadding_dim)

        self.em_dic['plat_form'] = self.plat_form_em
        self.em_dic['biz_type'] = self.biz_type_em
        self.em_dic['cate1_id'] = self.cate1_id_em
        self.em_dic['cate2_id'] = self.cate2_id_em
        self.em_dic['cate3_id'] = self.cate3_id_em
        self.em_dic['seller_uid'] = self.seller_uid_em
        self.em_dic['company_name'] = self.company_name_em
        self.em_dic['rvcr_prov_name'] = self.rvcr_prov_name_em
        self.em_dic['rvcr_city_name'] = self.rvcr_city_name_em
        self.payed_hour_em = nn.Embedding(config.payed_hour_range, config.embadding_dim)
        self.em_keys = self.em_dic.keys()

        self.pre_dic = {}

        self.share_layer = nn.Sequential(
            nn.Linear(len(self.em_dic) * config.embadding_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(), nn.Linear(
                4096, 2048), 
                nn.BatchNorm1d(2048),
                nn.ReLU()
        )
        
        self.lgst_company_pre = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),nn.Dropout(),
            nn.ReLU(), nn.Linear(
                1024, 1024), 
                nn.BatchNorm1d(1024),nn.Dropout(),
                nn.ReLU(),
            nn.Linear(1024, config.lgst_company_range))

        self.warehouse_id_pre = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),nn.Dropout(),
            nn.ReLU(), nn.Linear(
                1024, 1024), 
                nn.BatchNorm1d(1024),nn.Dropout(),
                nn.ReLU(),
            nn.Linear(1024, config.warehouse_id_range))

        self.shipped_prov_id_pre = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024), nn.Dropout(),
            nn.ReLU(), nn.Linear(
                1024, 1024), 
                nn.BatchNorm1d(1024),nn.Dropout(),
                nn.ReLU(),
            nn.Linear(1024, config.shipped_prov_id_range))

        self.shipped_city_id_pre = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024), nn.Dropout(),
            nn.ReLU(), nn.Linear(
                1024, 1024), 
                nn.BatchNorm1d(1024),nn.Dropout(),
                nn.ReLU(),
            nn.Linear(1024, config.shipped_city_id_range))


        self.pre_dic['lgst_company'] = self.lgst_company_pre
        self.pre_dic['warehouse_id'] = self.warehouse_id_pre
        self.pre_dic['shipped_prov_id'] = self.shipped_prov_id_pre
        self.pre_dic['shipped_city_id'] = self.shipped_city_id_pre
        self.pre_keys = self.pre_dic.keys()


    def forward(self, dic):
        em_list = [self.em_dic[k](dic[k]) for k in self.em_keys]
        em_vec = torch.cat(em_list, dim=1)
        em_vec = self.share_layer(em_vec)

        d = {}
        for k in self.pre_keys:
            d[k] = self.pre_dic[k](em_vec)
        return d

