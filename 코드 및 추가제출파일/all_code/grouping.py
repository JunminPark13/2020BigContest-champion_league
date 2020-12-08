"""
    Subject: Grouping
    
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

class Grouping():
    def __init__(
        self,
        df_train,
        df_test
    ):
        self.df_train = df_train
        self.df_test = df_test

        self.group_1 = ['가구', '가전']
        self.group_2 = ['농수축', '이미용']
        self.group_3 = ['생활용품', '주방', '잡화']
        self.group_4 = ['건강기능', '속옷', '의류', '침구']

        self.group_5 = ['무형']

        self.train_csi = {'1' : 97.5, '2' : 99.5, '3' : 99.8,
                          '4': 101.6, '5' : 97.9, '6' : 97.5, '7' : 95.9 ,
                          '8' : 92.5, '9' : 96.9, '10' :98.6, '11':101.0, 
                          '12' : 100.5}
        self.train_cpi = {'1' : 100.8, '2' : 100.5, '3' : 100.4,
                          '4': 100.6, '5' : 100.7, '6' : 100.7, '7' : 100.6 ,
                          '8' : 100.0 , '9' : 99.6, '10' :100, '11':100.2,
                          '12' : 100.7}
        self.test_cpi = {'6': 100, '7': 100.3}
        self.test_csi = {'6': 81.8, '7': 84.2}
            

    def make_train(self):
        self.df_train = self.make_group(self.df_train)
        self.make_g1(self.df_train) 
        self.df_train = self.make_up_ind(self.df_train)
        self.df_train = self.make_cpi_csi(self.df_train, self.train_cpi, self.train_csi)
        self.df_train['log_sales_cnt'] = np.log(self.df_train['sales_cnt'])

        return self.df_train

    def make_test(self):
        self.df_test = self.make_group(self.df_test)
        self.make_g1(self.df_train)
        self.df_test = self.make_up_ind(self.df_test)
        self.df_test = self.make_cpi_csi(self.df_test, self.test_cpi, self.test_csi)
        self.df_test['log_sales_cnt'] = np.log(self.df_test['sales_cnt'])

        return self.df_test
          

    def make_group(self, dfs):
        self.dfs = dfs
        self.dfs.loc[self.dfs.product_grp.isin(self.group_1), 'group'] = 'group1'
        self.dfs.loc[self.dfs.product_grp.isin(self.group_2), 'group'] = 'group2'
        self.dfs.loc[self.dfs.product_grp.isin(self.group_3), 'group'] = 'group3'
        self.dfs.loc[self.dfs.product_grp.isin(self.group_4), 'group'] = 'group4'

        return self.dfs

    def make_group_test(self, dfs):
        self.dfs = dfs
        self.dfs.loc[self.dfs.product_grp.isin(self.group_1), 'group'] = 'group1'
        self.dfs.loc[self.dfs.product_grp.isin(self.group_2), 'group'] = 'group2'
        self.dfs.loc[self.dfs.product_grp.isin(self.group_3), 'group'] = 'group3'
        self.dfs.loc[self.dfs.product_grp.isin(self.group_4), 'group'] = 'group4'
        self.dfs.loc[self.dfs.product_grp.isin(self.group_5), 'group'] = 'group5'

        return self.dfs

    def make_g1(self, dfs):
        self.dfs = dfs
        self.g1 = self.dfs[self.dfs.product_grp.isin(self.group_1)]
        self.g2 = self.dfs[self.dfs.product_grp.isin(self.group_2)]
        self.g3 = self.dfs[self.dfs.product_grp.isin(self.group_3)]
        self.g4 = self.dfs[self.dfs.product_grp.isin(self.group_4)]

    def make_g1_test(self, dfs):
        self.dfs = dfs
        self.g1 = self.dfs[self.dfs.product_grp.isin(self.group_1)]
        self.g2 = self.dfs[self.dfs.product_grp.isin(self.group_2)]
        self.g3 = self.dfs[self.dfs.product_grp.isin(self.group_3)]
        self.g4 = self.dfs[self.dfs.product_grp.isin(self.group_4)]
        self.g5 = self.dfs[self.dfs.product_grp.isin(self.group_5)]


    def make_up_ind(self, dfs):
        self.dfs = dfs
        self.dfs.loc[(self.dfs.group=='group1')&(self.dfs.unit_price<np.percentile(self.g1.unit_price, 33)), 'unit_price_group'] = 'cheap'
        self.dfs.loc[(self.dfs.group=='group1')&(self.dfs.unit_price>=np.percentile(self.g1.unit_price, 33))\
                                      &(self.dfs.unit_price<np.percentile(self.g1.unit_price, 66)), 'unit_price_group'] = 'medium'
        self.dfs.loc[(self.dfs.group=='group1')&(self.dfs.unit_price>=np.percentile(self.g1.unit_price, 66)), 'unit_price_group'] = 'expensive'

        self.dfs.loc[(self.dfs.group=='group2')&(self.dfs.unit_price<np.percentile(self.g2.unit_price, 33)), 'unit_price_group'] = 'cheap'
        self.dfs.loc[(self.dfs.group=='group2')&(self.dfs.unit_price>=np.percentile(self.g2.unit_price, 33))\
                                      &(self.dfs.unit_price<np.percentile(self.g2.unit_price, 66)), 'unit_price_group'] = 'medium'
        self.dfs.loc[(self.dfs.group=='group2')&(self.dfs.unit_price>=np.percentile(self.g2.unit_price, 66)), 'unit_price_group'] = 'expensive'

        self.dfs.loc[(self.dfs.group=='group3')&(self.dfs.unit_price<np.percentile(self.g3.unit_price, 33)), 'unit_price_group'] = 'cheap'
        self.dfs.loc[(self.dfs.group=='group3')&(self.dfs.unit_price>=np.percentile(self.g3.unit_price, 33))\
                                      &(self.dfs.unit_price<np.percentile(self.g3.unit_price, 66)), 'unit_price_group'] = 'medium'
        self.dfs.loc[(self.dfs.group=='group3')&(self.dfs.unit_price>=np.percentile(self.g3.unit_price, 66)), 'unit_price_group'] = 'expensive'

        self.dfs.loc[(self.dfs.group=='group4')&(self.dfs.unit_price<np.percentile(self.g4.unit_price, 33)), 'unit_price_group'] = 'cheap'
        self.dfs.loc[(self.dfs.group=='group4')&(self.dfs.unit_price>=np.percentile(self.g4.unit_price, 33))\
                                      &(self.dfs.unit_price<np.percentile(self.g4.unit_price, 66)), 'unit_price_group'] = 'medium'
        self.dfs.loc[(self.dfs.group=='group4')&(self.dfs.unit_price>=np.percentile(self.g4.unit_price, 66)), 'unit_price_group'] = 'expensive'

        return self.dfs

    def make_up_ind_test(self, dfs):
        self.dfs = dfs
        self.dfs.loc[(self.dfs.group=='group1')&(self.dfs.unit_price<np.percentile(self.g1.unit_price, 33)), 'unit_price_group'] = 'cheap'
        self.dfs.loc[(self.dfs.group=='group1')&(self.dfs.unit_price>=np.percentile(self.g1.unit_price, 33))\
                                      &(self.dfs.unit_price<np.percentile(self.g1.unit_price, 66)), 'unit_price_group'] = 'medium'
        self.dfs.loc[(self.dfs.group=='group1')&(self.dfs.unit_price>=np.percentile(self.g1.unit_price, 66)), 'unit_price_group'] = 'expensive'

        self.dfs.loc[(self.dfs.group=='group2')&(self.dfs.unit_price<np.percentile(self.g2.unit_price, 33)), 'unit_price_group'] = 'cheap'
        self.dfs.loc[(self.dfs.group=='group2')&(self.dfs.unit_price>=np.percentile(self.g2.unit_price, 33))\
                                      &(self.dfs.unit_price<np.percentile(self.g2.unit_price, 66)), 'unit_price_group'] = 'medium'
        self.dfs.loc[(self.dfs.group=='group2')&(self.dfs.unit_price>=np.percentile(self.g2.unit_price, 66)), 'unit_price_group'] = 'expensive'

        self.dfs.loc[(self.dfs.group=='group3')&(self.dfs.unit_price<np.percentile(self.g3.unit_price, 33)), 'unit_price_group'] = 'cheap'
        self.dfs.loc[(self.dfs.group=='group3')&(self.dfs.unit_price>=np.percentile(self.g3.unit_price, 33))\
                                      &(self.dfs.unit_price<np.percentile(self.g3.unit_price, 66)), 'unit_price_group'] = 'medium'
        self.dfs.loc[(self.dfs.group=='group3')&(self.dfs.unit_price>=np.percentile(self.g3.unit_price, 66)), 'unit_price_group'] = 'expensive'

        self.dfs.loc[(self.dfs.group=='group4')&(self.dfs.unit_price<np.percentile(self.g4.unit_price, 33)), 'unit_price_group'] = 'cheap'
        self.dfs.loc[(self.dfs.group=='group4')&(self.dfs.unit_price>=np.percentile(self.g4.unit_price, 33))\
                                      &(self.dfs.unit_price<np.percentile(self.g4.unit_price, 66)), 'unit_price_group'] = 'medium'
        self.dfs.loc[(self.dfs.group=='group4')&(self.dfs.unit_price>=np.percentile(self.g4.unit_price, 66)), 'unit_price_group'] = 'expensive'

        self.dfs.loc[self.dfs.group=='group5', 'unit_price_group'] = '-'


        return self.dfs

    def make_cpi_csi(self, dfs, cpi, csi):
        self.dfs = dfs

        self.dfs['cpi'] = self.dfs['month']
        self.dfs['csi'] = self.dfs['month']
        self.dfs['cpi'] = self.dfs['cpi'].astype(str)
        self.dfs['csi'] = self.dfs['csi'].astype(str)

        self.dfs['cpi'] = self.dfs['cpi'].map(cpi)
        self.dfs['csi'] = self.dfs['csi'].map(csi)

        return self.dfs


