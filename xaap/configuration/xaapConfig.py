
from obspy import UTCDateTime

import logging
from get_mseed_data import get_mseed_utils as gmutils
logger = logging.getLogger(__name__)

class xaapConfig():


    def __init__(self,xaap_parameter):

        logger.info("start of create xaap_config object")
        
        self.mseed_client_id = xaap_parameter['mseed']['client_id']
        self.mseed_server_config_file = xaap_parameter['mseed']['server_config_file']

        try:
            self.mseed_server_param = gmutils.read_config_file(self.mseed_server_config_file)
        except Exception as e:
            raise Exception(f"Error reading mseed server config file : {e}")

        self.volcan_volcanoes_configuration_file = xaap_parameter['volcan_configuration']['volcanoes_config_file']
        self.volcan_station_file = xaap_parameter['volcan_configuration']['stations_config_file']
        self.volcan_volcan_name = xaap_parameter['volcan_configuration']['volcan_name']
        
        
        try:
            self.volcanoes_stations = gmutils.read_config_file(self.volcan_volcanoes_configuration_file)
        except Exception as e:
            raise Exception(f"Error reading volcano config file : {e}")

        try:
            self.stations_information = gmutils.read_config_file(self.volcan_station_file )

        except Exception as e:
            raise Exception(f"Error reading station config file : {e}")

        self.datetime_start = UTCDateTime(xaap_parameter["dates"]["start"])
        self.datetime_end = UTCDateTime(xaap_parameter["dates"]["end"])

        self.filter_freq_a = xaap_parameter["filter"]["freq_a"]
        self.filter_freq_b = xaap_parameter["filter"]["freq_b"]
        self.filter_type = xaap_parameter["filter"]["type"]

        self.sta_lta_sta = xaap_parameter["sta_lta"]["sta"]
        self.sta_lta_lta = xaap_parameter["sta_lta"]["lta"]
        self.sta_lta_trigon = xaap_parameter["sta_lta"]["trigon"]
        self.sta_lta_trigoff = xaap_parameter["sta_lta"]["trigoff"]
        self.sta_lta_coincidence = xaap_parameter["sta_lta"]["coincidence"]
        self.sta_lta_endtime_buffer = xaap_parameter["sta_lta"]["endtime_buffer"]

        self.output_detection_folder = xaap_parameter["output_data"]["output_detection_folder"]
        self.output_classification_folder = xaap_parameter["output_data"]["output_classification_folder"]

        logger.info("xaapConfig object created")
