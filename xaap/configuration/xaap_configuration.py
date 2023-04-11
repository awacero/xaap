import logging.config

from .xaapConfig import xaapConfig
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


    trigger_path = Path(xaap_config_dir)
    
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


    xaap_configuration = xaapConfig(config)

    return xaap_configuration