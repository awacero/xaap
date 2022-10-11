# -*- coding: utf-8 -*-

import os,sys
from pathlib import Path
from pyqtgraph.parametertree import Parameter, ParameterTree


from get_mseed_data import get_mseed_utils as gmutils
from get_mseed_data import get_mseed

from obspy.signal import filter  
from obspy import Trace, Stream
from obspy.signal.trigger import coincidence_trigger
from obspy import read, UTCDateTime
from obspy.signal.trigger import classic_sta_lta
from obspy.signal.trigger import classic_sta_lta, classic_sta_lta_py, recursive_sta_lta_py, plot_trigger, trigger_onset
from datetime import date, datetime

import numpy as np
import pandas as pd
from aaa_features.features import FeatureVector 
from sklearn.preprocessing import StandardScaler
import pickle 
from sklearn.ensemble import RandomForestClassifier

import logging, logging.config
        
# Librerias para CLI implementation
from models.xaap_filter import XaapFilter
from models.xaap_sta_lta import StaLta

xaap_dir = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])))
xaap_config_dir = Path("%s/%s" %(xaap_dir,"config"))
print("###")
#print(xaap_config_dir)
print(sys.argv)
logging.config.fileConfig(xaap_config_dir / "logging.ini" ,disable_existing_loggers=False)
logger = logging.getLogger('stdout')
logger.setLevel(logging.INFO)


def main():
   
    is_error=False
    
    if len(sys.argv)==1:
        is_error=True
    
    else:   
        try:
            run_param=gmutils.read_parameters(sys.argv[1])
        
        except Exception as e:
            logger.error("Error reading configuration sets in file: %s" %(str(e)))
            raise Exception("Error reading configuration file: %s" %(str(e)))
        """Load parameters from txt config file"""    
        try:
            filter_file = run_param['FILTER_FILE']['file_path']
            
        except Exception as e:
            logger.error("Error getting parameters: %s" %str(e))
            raise Exception("Error getting parameters: %s" %str(e))

        """Create filter list"""
        try:
            filter_dict = gmutils.read_config_file(filter_file)
            print("Filter dict:\n",filter_dict)
            filter_list = create_filter_list(filter_dict)
            print("Filter List:\n",filter_list)
        except Exception as e:
            logger.error("Failed to read filters file %s" %str(e))
            raise Exception("Failed to read filters file %s" %str(e))


    if is_error:
        print(f'Usage: python {sys.argv[0]} CONFIGURATION_FILE.txt ')  



def create_filter_list(filters_dict):
        """Create a rsam_filter object from a json file"""
        
        filter_list=[]
        for filter_key,filter in filters_dict.items():       
            xaap_filter = XaapFilter(**filter)
            filter_list.append(xaap_filter)
            
        print(filter_list)
        return filter_list

def create_trigger_list(trigger_dict):
    """"""

    trigger_list = []
    for trigger_key,trigger in trigger_dict.items():
        xaap_trigger = StaLta(**trigger)
        trigger_list.append(xaap_trigger)
    
    print(trigger_list)
    return trigger_list


if __name__ == '__main__':
    main()