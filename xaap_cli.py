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
from obspy.signal.trigger import classic_sta_lta, classic_sta_lta_py, recursive_sta_lta_py, plot_trigger, trigger_onset
from datetime import date, datetime

import json, csv 
import numpy as np
import pickle 

import pandas as pd
from aaa_features.features import FeatureVector 
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from validate_detection import seisbench_compare_results

import argparse
from xaap.configuration.xaap_configuration import (configure_logging, configure_parameters_from_config_file)
from xaap.process import process_deep_learning, request_data, classify_detection, pre_process,detect_trigger




##TODO
#LIMPIA Y ORDENA EL CODIGO
# QUE UNA FUNCION SOLO HAGA UNA COSA
## es necesario USAR un metodo que permita recuperar el canal. 
"""PROGRAMA DE FORMA SIMPLE, """




def main(args):

    """Load parameters from plain text config file"""
   
    configuration_file = args.xaap_cli_config
    try:
        logger.info(f"Check if configuration file {configuration_file} exists")
        file_path = gmutils.check_file(configuration_file)
    except Exception as e:
        logger.error(f"Error reading configuration  file: {e}" )
        raise Exception(f"Error reading configuration file: {e}" )

    try:
        logger.info("Create xaap configuration object") 
        xaap_config = configure_parameters_from_config_file(file_path)
    except Exception as e:
        logger.error("Error getting parameters: %s" %str(e))
        raise Exception("Error getting parameters: %s" %str(e))

    try:
        logger.info("Start processing. Request data")
        volcan_stream = request_data.request_stream(xaap_config)
        print(volcan_stream)
    except Exception as e:
        logger.info(f"Error in processing. Error in request data was:{e}")


    if args.detection_method=="sta_lta":   
        try:
            logger.info("Continue processing. Run detection with STA/LTA")
            triggers = detect_trigger.get_triggers(xaap_config,volcan_stream)
            sta_lta_filename =f"{xaap_config.output_detection_folder}/sta_lta_detections{UTCDateTime.now().strftime('%Y.%m.%d.%H.%M.%S')}.csv"
            if len(triggers) > 0:
                sta_lta_pd = pd.DataFrame(triggers)
                sta_lta_pd.to_csv(sta_lta_filename,index=False)
        except Exception as e:
            logger.info(f"Error in processing. Error in detection was:{e}")
        
    
    if args.detection_method=="deep":  
        
        """
        Try to process each stream individually with the DL model
        Process a single stream to recover the channel
        The function should return a detection window with a P/S phase if available. 
        """
        try:
            logger.info("Try to create DL model")
            deep_learning_model = process_deep_learning.create_model(xaap_config)
            logger.info(f"SB model: {deep_learning_model.name} created")
        except Exception as e:
            logger.info(f"Error in creating DL model was:{e}")
            raise Exception(f"Failed to create deep learning model: {e}")

        try:
            logger.info("Continue processing. Run detection with deep learning")
            detections = []
            for station_stream in volcan_stream:
                
                detections.extend(process_deep_learning.get_detections(xaap_config,Stream(station_stream),deep_learning_model))

            print("XAAP DETECTIONS")
            print(detections)

            for d in detections:
                try:
                    print(d, d.pick_detection.start_time, d.pick_detection.end_time)
                except:
                    print(d)

            try:
                coincidence_detections = process_deep_learning.coincidence_detection_deep_learning(xaap_config,detections)

                logger.info(f"@@@@@@@@@@@@FIN DE COINCIDENCE PICKS WAS:")
                for c_p in coincidence_detections:
                    print(c_p)

            except Exception as e:
                print(f"Error in coincidence detection was: {e}")
            
            ##picks,detections = detect_trigger.coincidence_pick_trigger_deep_learning(xaap_config,volcan_stream)
            #picks =  detect_picks.get_picks_deep_learning(xaap_config,volcan_stream)
            #print("Print results of get_picks_deep_learning()")
            #for p in picks:
            #    print(p.trace_id,p.start_time,p.end_time)
               

            sys.exit(0)



        except Exception as e:
            logger.info(f"Error in processing. Error in detection  DL was:{e}")


    if args.classification == True :

        try:
            logger.info(f"Classify triggers")
            classify_detection.classify_detection_SVM(xaap_config,volcan_stream, detections)
        
        except Exception as e:
            logger.info(f"Error in classify triggers: {e}")


    if args.comparation == True :

        logger.info(f"run in comparation mode")        
        try:
            logger.info(f"Compare results")
            results = seisbench_compare_results.calculate_metrics(xaap_config.out_temp)
            print(xaap_config.deep_learning_model_name, xaap_config.deep_learning_model_version)
            print(results)
            results_file_name = f"./{xaap_config.deep_learning_model_name}_{xaap_config.deep_learning_model_version}.{UTCDateTime.now().strftime('%Y.%m.%d.%H.%M.%S')}.json"
            with open(results_file_name,'w') as file:
                json.dump(results,file)

        except Exception as e:
            logger.info(f"Error in classify triggers: {e}")

        






if __name__ == '__main__':

    logger = configure_logging()
    logger.info("Logging configurated")

    '''Call the program with arguments or use default values'''
    parser = argparse.ArgumentParser(description='XAAP_CLI will use default configuration found in ./config/xaap_cli_config.cfg')
    parser.add_argument("--xaap_cli_config",type=str, default="./config/xaap_cli_config.cfg", help='Text config file for XAAP_CLI')
    parser.add_argument('--comparation', type=bool, default=False ,help='run comparation mode')   
    parser.add_argument('--classification', type=bool, default=False ,help='run clasification mode')   

    parser.add_argument("--detection_method",  type=str, default="sta_lta" ,help='run detection method')

    args = parser.parse_args()
    # Ahora args.time y args.config contienen los valores proporcionados por el usuario
    logger.info(f'xaap_cli_config set to: {args.xaap_cli_config}')
    logger.info(f'comparation set to: {args.comparation}')
    logger.info(f'detection_method set to: {args.detection_method}')

    main(args)
