import torch
from torch import nn

class block(nn.Module):
    def __init__(self, in_dim, out_dim, scale=0.2):
        super(block, self).__init__()
        if in_dim != out_dim:
            self.layer_1 = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),nn.Dropout(),
                nn.ReLU(inplace=True)
            )
        else:
            self.layer_1 = None
        
        self.layer_2 = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )


        self.scale = scale

    def forward(self, x):
        if self.layer_1 != None:
            x = self.layer_1(x)
        output = self.layer_2(x)
        output = self.scale*output + x
        return output



class NeuralNet(nn.Module):
    def __init__(self, config):
        super(NeuralNet, self).__init__()
        self.em_dic = {}
        self.plat_form_em = nn.Embedding(config.plat_form_range,
                                                config.embadding_dim)
        self.biz_type_em = nn.Embedding(config.biz_type_range,
                                               config.embadding_dim)
        self.product_id_em = nn.Embedding(config.product_id_range, config.embadding_dim)
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

        self.pre_days_em = nn.Embedding(config.pre_days_range, config.embadding_dim)

        # self.lgst_company_em = nn.Embedding(config.lgst_company_range, config.embadding_dim)
        self.warehouse_id_em = nn.Embedding(config.warehouse_id_range, config.embadding_dim)
        self.shipped_prov_id_em = nn.Embedding(config.shipped_prov_id_range, config.embadding_dim)
        self.shipped_city_id_em = nn.Embedding(config.shipped_city_id_range, config.embadding_dim)
        self.payed_hour_em = nn.Embedding(config.payed_hour_range, config.embadding_dim)

        # self.shipped_time_day_em = nn.Embedding(config.shipped_time_day_range, config.embadding_dim)
        # # self.shipped_time_hour_em = nn.Embedding(config.shipped_time_hour_range, config.embadding_dim)
        # self.got_time_day_em = nn.Embedding(config.got_time_day_range, config.embadding_dim)
        # # self.got_time_hour_em = nn.Embedding(config.got_time_hour_range, config.embadding_dim)
        # self.dlved_time_day_em = nn.Embedding(config.dlved_time_day_range, config.embadding_dim)
        # # self.dlved_time_hour_em = nn.Embedding(config.dlved_time_hour_range, config.embadding_dim)

        self.em_dic['plat_form'] = self.plat_form_em
        self.em_dic['biz_type'] = self.biz_type_em
        self.em_dic['product_id'] = self.product_id_em
        self.em_dic['cate1_id'] = self.cate1_id_em
        self.em_dic['cate2_id'] = self.cate2_id_em
        self.em_dic['cate3_id'] = self.cate3_id_em
        self.em_dic['seller_uid'] = self.seller_uid_em
        self.em_dic['company_name'] = self.company_name_em
        self.em_dic['rvcr_prov_name'] = self.rvcr_prov_name_em
        self.em_dic['rvcr_city_name'] = self.rvcr_city_name_em
        # self.em_dic['lgst_company'] = self.lgst_company_em
        # self.em_dic['warehouse_id'] = self.warehouse_id_em
        # self.em_dic['shipped_prov_id'] = self.shipped_prov_id_em
        # self.em_dic['shipped_city_id'] = self.shipped_city_id_em
        self.em_dic['payed_hour'] = self.payed_hour_em
        self.em_dic['pre_days'] = self.pre_days_em
        # self.em_dic['shipped_time_day'] = self.shipped_time_day_em
        # # self.em_dic['shipped_time_hour'] = self.shipped_time_hour_em
        # self.em_dic['got_time_day'] = self.got_time_day_em
        # # self.em_dic['got_time_hour'] = self.got_time_hour_em
        # self.em_dic['dlved_time_day'] = self.dlved_time_day_em
        # # self.em_dic['dlved_time_hour'] = self.dlved_time_hour_em
        self.em_keys = self.em_dic.keys()

        self.pre_dic = {}

        self.share_layer = nn.Sequential(
            nn.Linear(len(self.em_dic) * config.embadding_dim, 4096),
            nn.BatchNorm1d(4096),nn.Dropout(config.dropout_rate),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),nn.Dropout(config.dropout_rate),
            nn.ReLU(inplace=True)
        )

        # self.share_layers = []
        # self.share_layers.append(block(len(self.em_dic) * config.embadding_dim, config.blocks[0]))
        # self.share_layers += [block(config.blocks[i-1], config.blocks[i]) for i in range(1, len(config.blocks))]

        # self.share_layer = nn.Sequential(*self.share_layers)

        self.delta_days_pres = [block(config.fc_cells, config.fc_cells) for i in range(config.num_blocks)]
        self.delta_days_pre = nn.Sequential(
            nn.Linear(config.blocks[-1], config.fc_cells),
            nn.BatchNorm1d(config.fc_cells),nn.Dropout(config.dropout_rate),
            nn.ReLU(inplace=True),
            nn.Linear(
                config.fc_cells, config.fc_cells), 
            nn.BatchNorm1d(config.fc_cells),
            nn.Dropout(config.dropout_rate),
            nn.ReLU(inplace=True),
            nn.Linear(
                config.fc_cells, config.fc_cells), 
            nn.BatchNorm1d(config.fc_cells),
            nn.Dropout(config.dropout_rate),
            nn.ReLU(inplace=True),
            nn.Linear(
                config.fc_cells, config.fc_cells), 
            nn.BatchNorm1d(config.fc_cells),
            nn.Dropout(config.dropout_rate),
             nn.Linear(
                config.fc_cells, config.fc_cells), 
            nn.BatchNorm1d(config.fc_cells),
            nn.Dropout(config.dropout_rate),
            nn.ReLU(inplace=True),
            *self.delta_days_pres,
            nn.Linear(config.fc_cells, 1))
            
          
            

        self.hour_pres = [block(config.fc_cells, config.fc_cells) for i in range(config.num_blocks)]
        self.hour_pre = nn.Sequential(
            nn.Linear(config.blocks[-1], config.fc_cells),
            nn.BatchNorm1d(config.fc_cells), nn.Dropout(config.dropout_rate),
            nn.ReLU(inplace=True),
            nn.Linear(
                config.fc_cells, config.fc_cells), 
            nn.BatchNorm1d(config.fc_cells),
            nn.Dropout(config.dropout_rate),
            nn.ReLU(inplace=True),
            nn.Linear(
                config.fc_cells, config.fc_cells), 
            nn.BatchNorm1d(config.fc_cells),
            nn.Dropout(config.dropout_rate),
            nn.ReLU(inplace=True),
            nn.Linear(
                config.fc_cells, config.fc_cells), 
            nn.BatchNorm1d(config.fc_cells),
            nn.Dropout(config.dropout_rate),
            
            nn.ReLU(inplace=True),
            nn.Linear(config.fc_cells, config.fc_cells),
            nn.BatchNorm1d(config.fc_cells), 
            nn.Dropout(config.dropout_rate), 
            
            nn.ReLU(inplace=True), 
            *self.delta_days_pres,
            nn.Linear(config.fc_cells, 1))

        #self.shipped_time_day_pre = nn.Sequential(
        #    nn.Linear(config.blocks[-1], config.fc_cells),
        #    nn.BatchNorm1d(config.fc_cells),nn.Dropout(),
        #    nn.ReLU(inplace=True), nn.Linear(
        #        config.fc_cells, config.fc_cells), 
        #        nn.BatchNorm1d(config.fc_cells),nn.Dropout(),
        #        nn.ReLU(inplace=True),
        #    nn.Linear(
        #        config.fc_cells, config.fc_cells), 
        #        nn.BatchNorm1d(config.fc_cells),nn.Dropout(),
        #        nn.ReLU(inplace=True),
        #    nn.Linear(config.fc_cells, 1))

        #self.shipped_time_hour_pre = nn.Sequential(
        #    nn.Linear(config.blocks[-1], config.fc_cells),
        #    nn.BatchNorm1d(config.fc_cells), nn.Dropout(),
        #    nn.ReLU(inplace=True), nn.Linear(
        #        config.fc_cells, config.fc_cells), 
        #        nn.BatchNorm1d(config.fc_cells),nn.Dropout(),
        #        nn.ReLU(inplace=True),
        #    nn.Linear(
        #        config.fc_cells, config.fc_cells), 
        #        nn.BatchNorm1d(config.fc_cells),nn.Dropout(),
        #        nn.ReLU(inplace=True),
        #    nn.Linear(config.fc_cells, 1))

        #self.got_time_day_pre = nn.Sequential(
        #    nn.Linear(config.blocks[-1], config.fc_cells),
        #    nn.BatchNorm1d(config.fc_cells),nn.Dropout(),
        #    nn.ReLU(inplace=True), nn.Linear(
        #        config.fc_cells, config.fc_cells), 
        #        nn.BatchNorm1d(config.fc_cells),nn.Dropout(),
        #        nn.ReLU(inplace=True),
        #    nn.Linear(
        #        config.fc_cells, config.fc_cells), 
        #        nn.BatchNorm1d(config.fc_cells),nn.Dropout(),
        #        nn.ReLU(inplace=True),
        #    nn.Linear(config.fc_cells, 1))

        #self.got_time_hour_pre = nn.Sequential(
        #    nn.Linear(config.blocks[-1], config.fc_cells),
        #    nn.BatchNorm1d(config.fc_cells), nn.Dropout(),
        #    nn.ReLU(inplace=True), nn.Linear(
        #        config.fc_cells, config.fc_cells), 
        #        nn.BatchNorm1d(config.fc_cells),nn.Dropout(),
        #        nn.ReLU(inplace=True),
        #    nn.Linear(
        #        config.fc_cells, config.fc_cells), 
        #        nn.BatchNorm1d(config.fc_cells),nn.Dropout(),
        #        nn.ReLU(inplace=True),
        #    nn.Linear(config.fc_cells, 1))

        #self.dlved_time_day_pre = nn.Sequential(
        #    nn.Linear(config.blocks[-1], config.fc_cells),
        #    nn.BatchNorm1d(config.fc_cells),nn.Dropout(),
        #    nn.ReLU(inplace=True), 
        #    nn.Linear(
        #        config.fc_cells, config.fc_cells), 
        #        nn.BatchNorm1d(config.fc_cells),nn.Dropout(),
        #        nn.ReLU(inplace=True),
        #    nn.Linear(
        #        config.fc_cells, config.fc_cells), 
        #        nn.BatchNorm1d(config.fc_cells),nn.Dropout(),
        #        nn.ReLU(inplace=True),
        #    nn.Linear(config.fc_cells, 1))

        #self.dlved_time_hour_pre = nn.Sequential(
        #    nn.Linear(config.blocks[-1], config.fc_cells),
        #    nn.BatchNorm1d(config.fc_cells), nn.Dropout(),
        #    nn.ReLU(inplace=True), nn.Linear(
        #        config.fc_cells, config.fc_cells), 
        #        nn.BatchNorm1d(config.fc_cells),nn.Dropout(),
        #        nn.ReLU(inplace=True),
        #    nn.Linear(
        #        config.fc_cells, config.fc_cells), 
        #        nn.BatchNorm1d(config.fc_cells),nn.Dropout(),
        #        nn.ReLU(inplace=True),
        #    nn.Linear(config.fc_cells, 1))

        self.scale_dic = {'delta_days':1, 'hour':1, 'shipped_time_day':1, 'shipped_time_hour':1, 'got_time_day':1,'got_time_hour':1, 'dlved_time_day':1,'dlved_time_hour':1 }

        self.pre_dic['delta_days'] = self.delta_days_pre
        self.pre_dic['hour'] = self.hour_pre
        #self.pre_dic['shipped_time_day'] = self.shipped_time_day_pre
        #self.pre_dic['shipped_time_hour'] = self.shipped_time_hour_pre
        #self.pre_dic['got_time_day'] = self.got_time_day_pre
        #self.pre_dic['got_time_hour'] = self.got_time_hour_pre
        #self.pre_dic['dlved_time_day'] = self.dlved_time_day_pre
        #self.pre_dic['dlved_time_hour'] = self.dlved_time_hour_pre
        self.pre_keys = self.pre_dic.keys()


    def forward(self, dic):
        em_list = [self.em_dic[k](dic[k]) for k in self.em_keys]
        #print(em_list)
        em_vec = torch.cat(em_list, dim=1)
        em_vec = self.share_layer(em_vec)

        d = {}
        for k in self.pre_keys:
            d[k] = self.pre_dic[k](em_vec)*self.scale_dic[k]
        d['delta_days'] = torch.clamp(d['delta_days'], -2, 15)+2
        d['delta_days'] = torch.clamp(d['delta_days'], 1, 20)
        d['hour'] = torch.clamp(d['hour'], 3, 23) 
        #d['shipped_time_day'] = torch.clamp(d['shipped_time_day'], -8, 8) +8
        #d['shipped_time_hour'] = torch.clamp(d['shipped_time_hour'], -12,12)+12
        #d['got_time_day'] = torch.clamp(d['got_time_day'], -8, 8) +8
        #d['got_time_hour'] = torch.clamp(d['got_time_hour'], -12,12)+12
        #d['dlved_time_day'] = torch.clamp(d['dlved_time_day'], -8, 8) +8
        #d['dlved_time_hour'] = torch.clamp(d['dlved_time_hour'], -12,12)+12
        return d

