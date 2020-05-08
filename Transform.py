# -*- coding: utf-8 -*-
"""
Created on Thu May  7 07:23:48 2020

@author: khuze
"""

import numpy as np

class Transform:
    def createTrainTest(self,df,split):
        msk = np.random.rand(len(df)) < 0.8
        X_train = df[msk]
        X_test = df[~msk]
        Y_train = df[msk]
        Y_test = df[~msk]
        return {'X_train' : X_train, 'X_test':X_test,'Y_train' : Y_train, 'Y_test':Y_test}
        