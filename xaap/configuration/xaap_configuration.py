import logging.config

from  .xaapConfig import xaapConfig

from pathlib import Path

import logging

import json
import os
import configparser

logger = logging.getLogger(__name__)

xaap_config_dir = os.path.join(os.path.dirname(__file__),'..','..','config')
xaap_data_dir = os.path.join(os.path.dirname(__file__),'..','..','data')

def configure_logging():

    print("Start of logging configuration")
    logging.config.fileConfig(Path(xaap_config_dir,'logging.ini'), disable_existing_loggers=True)
    logger = logging.getLogger("xaap")
    
    logger.info(f"Logger configured was: {logging.getLogger().handlers}")
    return logger



def configure_parameters_from_gui(json_xaap_config):
    
    logger.info("start configuration of xaap")       
    json_config = json.loads(json_xaap_config)
    mseed_client_id = json_config['children']['parameters']['children']['mseed']['children']['client_id']['value']
    mseed_server_config_file = Path(xaap_config_dir, json_config['children']['parameters']['children']['mseed']\
                                                                    ['children']['server_config_file']['value'])

    volcan_volcanoes_configuration_file = Path(xaap_config_dir,json_config['children']['parameters']['children']['volcan_configuration']\
                                                        ['children']['volcanoes_config_file']['value'])
    volcan_station_file = Path(xaap_config_dir,json_config['children']['parameters']['children']['volcan_configuration']['children']\
                                            ['stations_config_file']['value'])
    volcan_volcan_name = json_config['children']['parameters']['children']['volcan_configuration']['children']\
                                            ['volcan_name']['value']


    datetime_start = json_config['children']['parameters']['children']['dates']['children']\
                                            ['start']['value']


    datetime_end = json_config['children']['parameters']['children']['dates']['children']\
                                            ['end']['value']


    filter_freq_a = json_config['children']['parameters']['children']['filter']['children']\
                                            ['freq_A']['value']

    filter_freq_b = json_config['children']['parameters']['children']['filter']['children']\
                                            ['freq_B']['value']

    filter_type = json_config['children']['parameters']['children']['filter']['children']\
                                            ['filter_type']['value']


    sta = json_config['children']['parameters']['children']['sta_lta']['children']['sta']['value']
    lta = json_config['children']['parameters']['children']['sta_lta']['children']['lta']['value']
    trigon = json_config['children']['parameters']['children']['sta_lta']['children']['trigon']['value']
    trigoff = json_config['children']['parameters']['children']['sta_lta']['children']['trigoff']['value']
    coincidence = json_config['children']['parameters']['children']['sta_lta']['children']['coincidence']['value']
    endtime_buffer = json_config['children']['parameters']['children']['sta_lta']['children']['endtime_buffer']['value']

    deep_learning_model_name = json_config['children']['parameters']['children']['deep_learning_picker']['children']['model_name']['value']
    deep_learning_model_version = json_config['children']['parameters']['children']['deep_learning_picker']['children']['model_version']['value']
    deep_learning_coincidence_picks = json_config['children']['parameters']['children']['deep_learning_picker']['children']['coincidence_picks']['value']


    output_detection_folder = json_config['children']['parameters']['children']['output_data']['children']["output_detection_folder"]['value']
    output_classification_folder = json_config['children']['parameters']['children']['output_data']['children']["output_classification_folder"]['value']

    config = configparser.ConfigParser()

    config.add_section("mseed")
    config.set("mseed","client_id",mseed_client_id)
    config.set("mseed","server_config_file",f"{mseed_server_config_file}")

    config.add_section("volcan_configuration") 
    config.set("volcan_configuration","volcanoes_config_file",f"{volcan_volcanoes_configuration_file}")
    config.set("volcan_configuration","stations_config_file",f"{volcan_station_file}")
    config.set("volcan_configuration","volcan_name",volcan_volcan_name)

    
    config.add_section("dates")
    config.set("dates","start",datetime_start)
    config.set("dates","end",datetime_end)


    config.add_section("filter")
    config.set("filter","freq_a",f"{filter_freq_a}")
    config.set("filter","freq_b",f"{filter_freq_b}")
    config.set("filter","type",filter_type)

    config.add_section("sta_lta")
    config.set("sta_lta","sta",f"{sta}")
    config.set("sta_lta","lta",f"{lta}")
    config.set("sta_lta","trigon",f"{trigon}")
    config.set("sta_lta","trigoff",f"{trigoff}")
    config.set("sta_lta","coincidence",f"{coincidence}")
    config.set("sta_lta","endtime_buffer",f"{endtime_buffer}")

    config.add_section("deep_learning")
    config.set("deep_learning","model_name",deep_learning_model_name)
    config.set("deep_learning","model_version",deep_learning_model_version)
    config.set("deep_learning","coincidence_picks",f"{deep_learning_coincidence_picks}")

    config.add_section("output_data")
    config.set("output_data","output_detection_folder",f"{output_detection_folder}")
    config.set("output_data","output_classification_folder",f"{output_classification_folder}")

    config.add_section("xaap_directories")
    config.set("xaap_directories","configuration_folder",f"{xaap_config_dir}")
    config.set("xaap_directories","data_folder",f"{xaap_data_dir}")


    xaap_configuration = xaapConfig(config)

    return xaap_configuration




def configure_parameters_from_config_file(config_file):
    
    logger.info("start configuration of xaap from config file")       
    
    config = configparser.ConfigParser()
    config.read(config_file)   

    mseed_server_config_file = Path(xaap_config_dir, config['mseed']['server_config_file'])
    volcan_volcanoes_configuration_file = Path(xaap_config_dir,config['volcan_configuration']['volcanoes_config_file'])
    volcan_station_file = Path(xaap_config_dir,config['volcan_configuration']['stations_config_file'])


    config.set("mseed","server_config_file",f"{mseed_server_config_file}")
    config.set("volcan_configuration","volcanoes_config_file",f"{volcan_volcanoes_configuration_file}")
    config.set("volcan_configuration","stations_config_file",f"{volcan_station_file}")

    config.add_section("xaap_directories")
    config.set("xaap_directories","configuration_folder",f"{xaap_config_dir}")
    config.set("xaap_directories","data_folder",f"{xaap_data_dir}")

    '''
    for section_name in config.sections():
        print(f"[{section_name}]")
        for key, value in config[section_name].items():
            print(f"{key} = {value}")
        print()  
    '''

    xaap_configuration = xaapConfig(config)

    return xaap_configuration