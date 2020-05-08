# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:59:47 2020

@author: khuze
"""
import pandas as pd

class GetData:
    def read_csv(self, filepath):
        df = pd.read_csv(filepath,error_bad_lines=False)
        return df 