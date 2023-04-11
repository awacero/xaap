import logging.config

from .xaapConfig import xaapConfig
from pathlib import Path

import logging

import json
import os
import configparser

logger = logging.getLogger(__name__)

xaap_config_dir = os.path.join(os.path.dirname(__file__),'..','..','config')

def configure_logging():

    print("Start of logging configuration")
    logging.config.fileConfig(Path(xaap_config_dir,'logging.ini'), disable_existing_loggers=True)
    logger = logging.getLogger("xaap")
    
    logger.info(f"Logger configured was: {logging.getLogger().handlers}")
    return logger



def configure_parameters_from_gui(json_xaap_config):
    
    logger.info("start configuration of xaap")       
    json_config = json.loads(json_xaap_config)
    mseed_client_id = json_config['children']['Parameters']['children']['MSEED']['children']['client_id']['value']
    mseed_server_config_file = Path(xaap_config_dir, json_config['children']['Parameters']['children']['MSEED']\
                                                                    ['children']['server_config_file']['value'])

    volcan_volcanoes_configuration_file = Path(xaap_config_dir,json_config['children']['Parameters']['children']['Volcan configuration']\
                                                        ['children']['volcanoes_config_file']['value'])
    volcan_station_file = Path(xaap_config_dir,json_config['children']['Parameters']['children']['Volcan configuration']['children']\
                                            ['stations_config_file']['value'])
    volcan_volcan_name = json_config['children']['Parameters']['children']['Volcan configuration']['children']\
                                            ['volcan_name']['value']


    datetime_start = json_config['children']['Parameters']['children']['Dates']['children']\
                                            ['start']['value']


    datetime_end = json_config['children']['Parameters']['children']['Dates']['children']\
                                            ['end']['value']


    filter_freq_a = json_config['children']['Parameters']['children']['Filter']['children']\
                                            ['Freq_A']['value']

    filter_freq_b = json_config['children']['Parameters']['children']['Filter']['children']\
                                            ['Freq_B']['value']

    filter_type = json_config['children']['Parameters']['children']['Filter']['children']\
                                            ['Filter type']['value']

    config = configparser.ConfigParser()
    config.add_section("MSEED")
    config.set("MSEED","client_id",mseed_client_id)
    config.set("MSEED","server_config_file",f"{mseed_server_config_file}")

    config.add_section("Volcan configuration") 
    config.set("Volcan configuration","volcanoes_config_file",f"{volcan_volcanoes_configuration_file}")
    config.set("Volcan configuration","stations_config_file",f"{volcan_station_file}")
    config.set("Volcan configuration","volcan_name",volcan_volcan_name)

    
    config.add_section("Dates")
    config.set("Dates","start",datetime_start)
    config.set("Dates","end",datetime_end)


    config.add_section("filter")
    config.set("filter","freq_a",f"{filter_freq_a}")
    config.set("filter","freq_b",f"{filter_freq_b}")
    config.set("filter","type",filter_type)


    xaap_configuration = xaapConfig(config)

    return xaap_configuration