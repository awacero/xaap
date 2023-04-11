from obspy import UTCDateTime

from get_mseed_data import get_mseed_utils as gmutils
from get_mseed_data import get_mseed

from obspy import Trace, Stream
import json

import os
from pathlib import Path
import logging
logger = logging.getLogger(__name__)



def request_stream(xaap_config):
    logger.info("Start of request_stream")

    mseed_client_id = xaap_config.mseed_client_id
    try: 
        mseed_client = get_mseed.choose_service(xaap_config.mseed_server_param[xaap_config.mseed_client_id])
    except Exception as e:
        raise Exception("Error connecting to MSEED server : %s" %str(e))

        
    volcanoes_stations = xaap_config.volcanoes_stations
    stations_information = xaap_config.stations_information
    volcan_name = xaap_config.volcan_volcan_name
    start_time = xaap_config.datetime_start
    end_time = xaap_config.datetime_end
    volcan_stations = volcanoes_stations[volcan_name][volcan_name]

    volcan_stations_list =  []

    for temp_station in volcan_stations:
        volcan_stations_list.append(stations_information[temp_station])

    st = Stream()

    for st_ in volcan_stations_list:
        for loc in st_['loc']:
            if not loc:
                loc = ''
            for cha in st_['cha']:
                stream_id = "%s.%s.%s.%s.%s.%s" %(st_['net'],st_['cod'],st_['loc'][0],cha,start_time,end_time)
                mseed_stream=get_mseed.get_stream(mseed_client_id,mseed_client,st_['net'],st_['cod'],loc,cha,start_time=start_time,end_time=end_time)
                logger.debug(stream_id)
                if mseed_stream:
                    mseed_stream.merge(method=1, fill_value="interpolate",interpolation_samples=0)
                    st+=mseed_stream
                else:
                    logger.info("no stream: %s" %stream_id)

    volcan_stream = st.copy()
    logger.info(f"End of data request. I got {volcan_stream}")

    return volcan_stream




