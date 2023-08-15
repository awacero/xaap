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




'''
import logging, logging.config
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config/logging.ini')
logging.config.fileConfig(log_file_path)
xaap_dir = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])))
xaap_config_dir = Path("%s/%s" %(xaap_dir,"config"))
logger = logging.getLogger('stdout')
logger.setLevel(logging.INFO)
'''


from xaap.configuration.xaap_configuration import (
    configure_logging, configure_parameters_from_config_file)
from xaap.process import pre_process, request_data, detect_trigger, classify_detection



def main():
   
    is_error=False
    
    if len(sys.argv)==1:
        is_error=True
    
    else:   
        try:
            logger.info(f"Check if configuration file {sys.argv[1]} exists")
            file_path = gmutils.check_file(sys.argv[1])
            
        except Exception as e:
            logger.error(f"Error reading configuration  file: {e}" )
            raise Exception(f"Error reading configuration file: {e}" )
        """Load parameters from plain text config file"""    
        try:
            logger.info("Create xaap configuration object") 
            
            xaap_config = configure_parameters_from_config_file(file_path)
            print(xaap_config)

        except Exception as e:
            logger.error("Error getting parameters: %s" %str(e))
            raise Exception("Error getting parameters: %s" %str(e))


        try:
            logger.info("Start processing. Request data")
            volcan_stream = request_data.request_stream(xaap_config)
            print(volcan_stream)
        
        except Exception as e:
            logger.info(f"Error in processing. Error in request data was:{e}")

        
        try:
            logger.info("Continue processing. Run detection with STA/LTA")
            triggers = detect_trigger.get_triggers(xaap_config,volcan_stream)
            for detection in triggers:
                print(detection)
        
        except Exception as e:
            logger.info(f"Error in processing. Error in detection was:{e}")





        try:
            picks,detections = detect_trigger.coincidence_pick_trigger_deep_learning(xaap_config,volcan_stream)

            if len(picks) >0:
                picks = picks
                
            if len(detections) > 0:
                triggers = detections
                
                logger.info(f"Coincidence triggers found: {len(triggers)}")

        except Exception as e:
            logger.info(f"Error in processing. Error in detection  DL was:{e}")


        try:
            logger.info(f"Classify triggers")
            classify_detection.classify_detection_SVM(xaap_config,volcan_stream, detections)
        
        except Exception as e:
            logger.info(f"Error in classify triggers: {e}")




    if is_error:
        print(f'Usage: python {sys.argv[0]} CONFIGURATION_FILE.txt ')  











if __name__ == '__main__':
    logger = configure_logging()
    logger.info("Logging configurated")
    main()
