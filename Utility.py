# -*- coding: utf-8 -*-
"""
Created on Tue May  5 12:17:21 2020

@author: khuze
"""
import yaml

class Util:
    def read_config():
        return yaml.load(open('config.yaml','r'))