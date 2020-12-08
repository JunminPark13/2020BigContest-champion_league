#!/usr/bin/env python3
"""
    Subject: Preprocessing
    
    Status: Development
    Version: 0.1
    
    Python Version: 3.7.4
"""


""" Setup
"""
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

import dict_grp as dict_grp
# import source.input_colName_dict as colName_dicts

class ProcessSale():
    def __init__(
        self,
        data_dir, #data
        source_dir, #source
        data_train_dir, #01_제공데이터
        data_test_dir, #02_평가데이터

        headline, #2020 빅콘테스트 데이터분석분야-챔피언리그
        train_target, #2019년 실적데이터
        rate_target, #시청률 데이터
        test_target  #2020년 6월 판매실적예측데이터(평가데이터)
    ):
        self.data_dir = data_dir
        self.source_dir = source_dir
        self.data_train_dir = data_train_dir
        self.data_test_dir = data_test_dir
        self.headline = headline
        self.train_target = train_target
        self.rate_target = rate_target
        self.test_target = test_target

        dfs_train_path = self.headline + '_' + self.train_target + '.csv'
        dfs_test_path = self.headline + '_' + self.test_target + '.csv'
        dfs_rate_path = self.headline + '_' + self.rate_target + '.csv'

        self.dfs_train_path = os.path.join(self.data_train_dir, dfs_train_path)
        self.dfs_test_path = os.path.join(self.data_test_dir, dfs_test_path)
        self.dfs_rate_path = os.path.join(self.data_train_dir, dfs_rate_path)

        self.dfs_w_train_path = os.path.join(self.source_dir, 'w_train.csv')
        self.dfs_w_test_path = os.path.join(self.source_dir, 'w_test.csv')

        self.dfs_o_train_path = os.path.join(self.source_dir, 'o_train.csv')
        self.dfs_o_test_path = os.path.join(self.source_dir, 'o_test.csv')

        self.w_train = pd.read_csv(self.dfs_w_train_path, encoding='euc-kr')
        self.w_test = pd.read_csv(self.dfs_w_test_path, encoding='euc-kr')

        self.o_train = pd.read_csv(self.dfs_o_train_path, encoding='euc-kr')
        self.o_test = pd.read_csv(self.dfs_o_test_path, encoding='euc-kr')

        if os.path.exists(self.dfs_train_path):
            print("Train : file already exist!")
            df_train_ = pd.read_csv(self.dfs_train_path)
            df_rate_ = pd.read_csv(self.dfs_rate_path)

        if os.path.exists(self.dfs_test_path):
            print("Test : file already exist!")
            df_test_ = pd.read_csv(self.dfs_test_path)
        
        df_train_.columns = df_test_.columns

        self.df_train = df_train_
        self.df_test = df_test_
        self.df_rate = df_rate_

        self.col_name = ['방송일시', '연', '월', '일', '시', '분', '요일', '공휴일', '월순서', 'order_grp',
                    '노출(분)', '마더코드', '상품코드', '상품명', '상품군', 
                    '기온(°C)', '강수량(mm)', '풍속(m/s)', '풍향(16방위)', '습도(%)', '증기압(hPa)', 
                    '현지기압(hPa)', '해면기압(hPa)', '적설(cm)',
                    '판매단가', '취급액', '판매수량']

        self.col_name_en = ['datetime', 'year', 'month', 'day', 'hour', 'minute', 'weekday', 'holiday', 'month_order', 'order_grp',
                            'exposure(min)', 'mother_cd', 'product_cd', 'product_name', 'product_grp',
                            'temp', 'rainfall', 'wind_speed', 'wind_direction', 'humidity', 'pressure',
                            'spot_pressure', 'sea_level_pressure', 'snowfall',
                            'unit_price', 'sell_price', 'sales_cnt']

        # print("train data : {}".format(len(self.df_train)))
        # print("test data : {}".format(len(self.df_test)))


    #preprocessing function
    def del_comma(self, dfs):
        self.dfs = dfs
        self.dfs['판매단가'] = self.dfs['판매단가'].str.replace(',','')
        self.dfs['취급액'] = self.dfs['취급액'].str.replace(',','')
        self.dfs['판매단가'] = self.dfs['판매단가'].map(lambda x: float(x) if x!=' - ' else np.nan)
        self.dfs['취급액'] = self.dfs['취급액'].map(lambda x: float(x) if x!=np.nan else x)

        return self.dfs

    def make_count(self, dfs):
        self.dfs = dfs
        self.dfs['판매수량'] = self.dfs['취급액'] / self.dfs['판매단가']

        return self.dfs

    def divide_time(self, dfs):
        self.dfs = dfs
        self.dfs["방송일시"] = pd.to_datetime(self.dfs["방송일시"])

        self.dfs["연"] = self.dfs["방송일시"].dt.year
        self.dfs["월"] = self.dfs["방송일시"].dt.month
        self.dfs["일"] = self.dfs["방송일시"].dt.day
        self.dfs["시"] = self.dfs["방송일시"].dt.hour
        self.dfs["분"] = self.dfs["방송일시"].dt.minute
        self.dfs["요일"] = self.dfs["방송일시"].dt.day_name()

        return self.dfs

    def holiday_dummy(self, dfs):
        # Saturday & Sunday == 1
        # 2019 & 2020 holiday == 1
        # else == 0
        self.dfs = dfs

        self.dfs['공휴일'] = 0
        self.dfs.loc[self.dfs['요일']=='Saturday', '공휴일'] = 1
        self.dfs.loc[self.dfs['요일']=='Sunday', '공휴일'] = 1
        
        if (self.dfs['연'] == 2019).all() == True : year = '2019'
        else : year = '2020'

        for day_list in dict_grp.dict_holiday[year]:
            self.dfs.loc[(self.dfs['월']==day_list[0])&(self.dfs['일']==day_list[1]), '공휴일'] = 1

        return self.dfs

    #for train data, delete 판매량 50,000
    # def delete_oman(self, dfs):
    #     self.dfs = dfs
    #     self.dfs.loc[self.dfs['취급액']==50000, '취급액'] = 0
    #     return self.dfs

    def delete_muhyung(self, dfs):
        self.dfs = dfs
        self.muhyung_index = self.dfs[self.dfs['상품군']=='무형'].index
        self.dfs = self.dfs.drop(self.muhyung_index).reset_index().copy()
        return self.dfs

    def month_order(self, dfs):
        self.dfs = dfs
        self.dfs['월순서'] = '0'
        self.dfs.loc[self.dfs['일']<=7, '월순서'] = '초'
        self.dfs.loc[(self.dfs['일']>7)&(self.dfs['일']<=21), '월순서'] = '중'
        self.dfs.loc[self.dfs['일']>21, '월순서'] = '말'

        return self.dfs

    #return dataset
    def train_preprocess(self):
        self.df_train = self.del_comma(self.df_train)
        self.df_train = self.divide_time(self.df_train)
        self.df_train = self.holiday_dummy(self.df_train)
        # self.df_train = self.delete_oman(self.df_train)
        self.df_train = self.delete_muhyung(self.df_train)
        self.df_train = self.make_count(self.df_train)
        self.df_train = self.month_order(self.df_train)

        self.df_train = pd.merge(self.df_train, self.w_train, how='left', 
                                 left_on=['연', '월', '일', '시'],
                                 right_on=['year', 'month', 'day', 'hour']).copy()

        self.df_train = pd.concat([self.df_train, self.o_train], axis=1).copy()

        self.df_train = self.df_train.loc[:,self.col_name]
        self.df_train.columns = self.col_name_en

        self.df_train = self.df_train[~self.df_train.sales_cnt.isnull()].copy()
        self.df_train = self.df_train.reset_index(drop=True)
        self.df_train['exposure(min)']=self.df_train['exposure(min)'].fillna(method='ffill')
        
        return self.df_train

    def test_preprocess_2(self):
        self.df_test['판매단가'] = self.df_test['판매단가'].str.replace(',','')
        self.df_test['판매단가'] = self.df_test['판매단가'].map(lambda x: float(x) if x!=' - ' else np.nan)
        self.df_test = self.divide_time(self.df_test)
        self.df_test = self.holiday_dummy(self.df_test)
        self.df_test = self.delete_muhyung(self.df_test)
        self.df_test = self.make_count(self.df_test)
        self.df_test = self.month_order(self.df_test)

        self.df_test = pd.merge(self.df_test, self.w_test, how='left', 
                                left_on=['연', '월', '일', '시'],
                                right_on=['year', 'month', 'day', 'hour'])

        self.df_test = pd.concat([self.df_test, self.o_test], axis=1).copy()

        self.df_test = self.df_test.loc[:,self.col_name]
        self.df_test.columns = self.col_name_en
        self.df_test = self.df_test.reset_index(drop=True)
        self.df_test['exposure(min)']=self.df_test['exposure(min)'].fillna(method='ffill')

        return self.df_test

    def test_preprocess(self):
        self.df_test['판매단가'] = self.df_test['판매단가'].str.replace(',','')
        self.df_test['판매단가'] = self.df_test['판매단가'].map(lambda x: float(x) if x!=' - ' else np.nan)
        self.df_test = self.divide_time(self.df_test)
        self.df_test = self.holiday_dummy(self.df_test)
        # self.df_test = self.delete_muhyung(self.df_test)
        self.df_test = self.make_count(self.df_test)
        self.df_test = self.month_order(self.df_test)

        self.df_test = pd.merge(self.df_test, self.w_test, how='left', 
                                left_on=['연', '월', '일', '시'],
                                right_on=['year', 'month', 'day', 'hour'])

        self.df_test = pd.concat([self.df_test, self.o_test], axis=1).copy()

        self.df_test = self.df_test.loc[:,self.col_name]
        self.df_test.columns = self.col_name_en
        self.df_test = self.df_test.reset_index(drop=True)
        self.df_test['exposure(min)']=self.df_test['exposure(min)'].fillna(method='ffill')

        return self.df_test

    def rate_preprocess(self):
        return self.df_rate