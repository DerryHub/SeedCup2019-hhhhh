class Config():
    def __init__(self):
        self.plat_form_range = 5
        self.biz_type_range = 6
        self.product_id_range = 141434
        self.cate1_id_range = 27
        self.cate2_id_range = 272
        self.cate3_id_range = 1609
        self.seller_uid_range = 1000
        self.company_name_range = 951
        self.lgst_company_range = 17
        self.warehouse_id_range = 12
        self.shipped_prov_id_range = 28
        self.shipped_city_id_range = 118
        self.rvcr_prov_name_range = 33
        self.rvcr_city_name_range = 435
        self.payed_hour_range = 24
        self.pre_days_range = 15

        self.shipped_time_day_range = 30
        self.shipped_time_hour_range = 24
        self.got_time_day_range = 30
        self.got_time_hour_range = 24
        self.dlved_time_day_range = 30
        self.dlved_time_hour_range = 24

        self.embadding_dim = 64
        self.dropout_rate = 0.7

        self.CUDA = True
        self.batch_size = 4096
        self.EPOCHs = 1000
        self.lr = 1e-3
        self.resume = False
        self.scale = 30
        self.day_weight = 5
        self.day_reduce = 0

        self.blocks = [4096]
        self.num_blocks = 0
        self.fc_cells = 1024

        self.threshold = 0.996
