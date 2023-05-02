from obspy.signal.trigger import coincidence_trigger
import pandas as pd

from obspy import UTCDateTime
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

def get_triggers( xaap_config, volcan_stream ):

    sta = float(xaap_config.sta_lta_sta)
    lta = float(xaap_config.sta_lta_lta)
    trigon = float(xaap_config.sta_lta_trigon)
    trigoff = float(xaap_config.sta_lta_trigoff)
    sta_lta_coincidence = float(xaap_config.sta_lta_coincidence)
    

    triggers = coincidence_trigger("recstalta", trigon, trigoff, volcan_stream, 
                                    sta_lta_coincidence, sta=sta, lta=lta)
    #for i,trg in enumerate(triggers):
    #    logger.info("%s:%s.%s %s" %(i,trg['time'],trg['trace_ids'][0],trg['duration']))
    triggers_pd = pd.DataFrame(triggers)

    detected_triggers_file = Path(xaap_config.output_detection_folder) / UTCDateTime.now().strftime("trigger_xaap_%Y.%m.%d.%H.%M.%S.csv")
    triggers_pd.to_csv(detected_triggers_file)
    ##print(triggers_pd)
    logger.info("Total picks detected: %s " %(len(triggers_pd.stations)))

    return triggers

def create_trigger_traces(xaap_config,volcan_stream,triggers):

    """
    Create traces of detected picks
    sta_lta_endtime_buffer multiplied by trigger['duration'] gives a better trigger duration time
    """
    sta_lta_endtime_buffer = float(xaap_config.sta_lta_endtime_buffer)
    #triggers_on =[]
    triggers_traces = []

    for i,trigger in enumerate(triggers):
        trigger_start_timestamp = trigger['time'].timestamp
        trigger_start = trigger['time']
        trigger_duration = trigger['duration'] * sta_lta_endtime_buffer
        #triggers_on.append(trigger_start_timestamp)
        
        trigger_stream_temp = volcan_stream.select(id=trigger['trace_ids'][0])
        trigger_trace = trigger_stream_temp[0].slice(trigger_start, trigger_start + trigger_duration)
        #MINIMUM TRIGGER LENGTH
        if trigger_trace.count() > 1:
            triggers_traces.append(trigger_trace)


    return triggers_traces 
