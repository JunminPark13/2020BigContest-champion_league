"""
    Subject: Spliting
    
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

from sklearn.model_selection import train_test_split

class Spliting():
    def __init__(
        self
    ):

        self.variables = ['month',  'hour', 'minute', 'weekday',
                            'holiday', 'month_order', 'order_grp', 'exposure(min)', 'mother_cd',
                            # 'product_cd',
                            'product_grp', 
                            'temp','humidity', 
                            # 'rainfall''snowfall', 'wind_speed',
                            'unit_price','unit_price_group', 'cpi', 'csi', 'log_sales_cnt',
                            'group', 'product_name']

        # self.drop_variables = ['rainfall','humidity', 'snowfall', 'wind_speed']
        # self.dummy_list = ['month',  'weekday', 'month_order', 'order_grp','product_grp','unit_price_group']


    def return_g(self, 
                 df_train):

        self.df_train = df_train
        self.df_train = self.df_train[self.variables]
        # self.df_train = pd.get_dummies(self.df_train, columns = self.dummy_list)

        self.g1 = self.df_train[self.df_train['group'] == 'group1']
        self.g2 = self.df_train[self.df_train['group'] == 'group2']
        self.g3 = self.df_train[self.df_train['group'] == 'group3']
        self.g4 = self.df_train[self.df_train['group'] == 'group4']

        #g1
        self.g1['installment'] = self.g1['product_name'].str.contains('무이자|무')
        self.g1['samsung'] = self.g1['product_name'].str.contains('삼성')
        self.g1['lg'] = self.g1['product_name'].str.contains('lg')

        self.g1['washer'] = self.g1['product_name'].str.contains('세탁기')
        self.g1['refrigerator'] = self.g1['product_name'].str.contains('냉장고')
        self.g1['laptop'] = self.g1['product_name'].str.contains('노트북')
        self.g1['air_fresher'] = self.g1['product_name'].str.contains('공기청정기|공청기')
        self.g1['dryer'] = self.g1['product_name'].str.contains('건조기')
        self.g1['cleaner'] = self.g1['product_name'].str.contains('청소기|로보킹')
        self.g1['air_conditioner'] = self.g1['product_name'].str.contains('에어컨')

        self.g1['bed'] = self.g1['product_name'].str.contains('침대|베드')
        self.g1['sofa'] = self.g1['product_name'].str.contains('소파')
        self.g1['drawer'] = self.g1['product_name'].str.contains('붙박이장|서랍장')

        # self.g1 = self.g1.drop(columns=self.drop_variables)

        self.g1 = self.g1.drop(columns=['group', 'product_name'])
        self.g2 = self.g2.drop(columns=['group', 'product_name'])
        self.g3 = self.g3.drop(columns=['group', 'product_name'])
        self.g4 = self.g4.drop(columns=['group', 'product_name'])


        return self.g1, self.g2, self.g3, self.g4

    def return_g_test(self, 
                    df_train):

            self.df_train = df_train
            self.df_train = self.df_train[self.variables]
            # self.df_train = pd.get_dummies(self.df_train, columns = self.dummy_list)

            self.g1 = self.df_train[self.df_train['group'] == 'group1']
            self.g2 = self.df_train[self.df_train['group'] == 'group2']
            self.g3 = self.df_train[self.df_train['group'] == 'group3']
            self.g4 = self.df_train[self.df_train['group'] == 'group4']
            self.g5 = self.df_train[self.df_train['group'] == 'group5']

            #g1
            self.g1['installment'] = self.g1['product_name'].str.contains('무이자|무')
            self.g1['samsung'] = self.g1['product_name'].str.contains('삼성')
            self.g1['lg'] = self.g1['product_name'].str.contains('lg')

            self.g1['washer'] = self.g1['product_name'].str.contains('세탁기')
            self.g1['refrigerator'] = self.g1['product_name'].str.contains('냉장고')
            self.g1['laptop'] = self.g1['product_name'].str.contains('노트북')
            self.g1['air_fresher'] = self.g1['product_name'].str.contains('공기청정기|공청기')
            self.g1['dryer'] = self.g1['product_name'].str.contains('건조기')
            self.g1['cleaner'] = self.g1['product_name'].str.contains('청소기|로보킹')
            self.g1['air_conditioner'] = self.g1['product_name'].str.contains('에어컨')

            self.g1['bed'] = self.g1['product_name'].str.contains('침대|베드')
            self.g1['sofa'] = self.g1['product_name'].str.contains('소파')
            self.g1['drawer'] = self.g1['product_name'].str.contains('붙박이장|서랍장')

            # self.g1 = self.g1.drop(columns=self.drop_variables)

            self.g1 = self.g1.drop(columns=['group', 'product_name'])
            self.g2 = self.g2.drop(columns=['group', 'product_name'])
            self.g3 = self.g3.drop(columns=['group', 'product_name'])
            self.g4 = self.g4.drop(columns=['group', 'product_name'])
            self.g5 = self.g5.drop(columns=['group', 'product_name'])


            return self.g1, self.g2, self.g3, self.g4, self.g5