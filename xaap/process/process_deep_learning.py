import sys
from pathlib import Path

# Agrega el directorio 'process' al sys.path
current_dir = Path(__file__).parent.resolve()
sys.path.append(str(current_dir))

from obspy import UTCDateTime
from obspy import Stream
from  obspy.core.event import Pick
from obspy.signal.trigger import (classic_sta_lta, classic_sta_lta_py,
                                  coincidence_trigger, plot_trigger,
                                  recursive_sta_lta_py, trigger_onset)


import seisbench
import pandas as pd
import numpy as np

import seisbench.models as sbm
from  seisbench.util.annotations import Pick as sb_pick
from  seisbench.util.annotations import Detection as sb_detection
import logging
logger = logging.getLogger(__name__)


from detection_xaap import DetectionXaap

def create_model(xaap_config):

    """
    """
    model_name = xaap_config.deep_learning_model_name
    model_version = xaap_config.deep_learning_model_version

    logger.info(f"Try to create model {model_name} with version:{model_version}")

    try:
        model = getattr(sbm,model_name).from_pretrained(model_version)
        return model
    except Exception as e:
        logger.error(f"Error while creating deep learning model from seisbench: {model_name} and {model_version}: {e}")


def coincidence_detection_deep_learning( xaap_config, detections_list,
                        #thr_coincidence_sum, 
                        trace_ids=None,
                        max_trigger_length=1e6, delete_long_trigger=False,
                        trigger_off_extension=0, details=False,
                        event_templates={}, similarity_threshold=0.7,
                        **options):
    

    thr_coincidence_sum = xaap_config.deep_learning_coincidence_picks

    triggers = []
    picks =[]
    triggers_raw = []
    picks_raw = []
    picks_raw_channel = []
    tmp_triggers = detections_list
    tmp_picks = []

    if trace_ids is None:
        trace_ids = [detection.trace_id for detection in detections_list]

    # we always work with a dictionary with trace ids and their weights later
    if isinstance(trace_ids, list) or isinstance(trace_ids, tuple):
        trace_ids = dict.fromkeys(trace_ids, 1)
    # set up similarity thresholds as a dictionary if necessary
    if not isinstance(similarity_threshold, dict):
        similarity_threshold = dict.fromkeys(
            #[tr.stats.station for tr in stream], similarity_threshold)
            [detection.trace_id.split(".")[1] for detection in detections_list],similarity_threshold)

            #tr_id.split(".")[1]

    for detection_xaap in detections_list:
        '''
        if detection_xaap.peak_time is None:
            on = detection_xaap.start_time
        else:
            on = detection_xaap.peak_time
        '''
        on = detection_xaap.start_time
        off = detection_xaap.end_time


        picks.append((on.timestamp,off.timestamp,detection_xaap.trace_id))

    
    """Start of coincidence pick modified from coincidence trigger"""

    picks.sort()


    try:

        for i,(on,off,tr_id) in enumerate(picks):
            simil = None
            picks[i] = (on, off,tr_id,simil)

        coincidence_picks = []
        last_off_time = 0.0
        while picks != []:

            
            on, off, tr_id, simil = picks.pop(0)
            sta =  tr_id.split(".")[1]

            

            coincidence_pick = {}
            coincidence_pick['time']=UTCDateTime(on)
            coincidence_pick['stations'] = [sta]
            coincidence_pick["trace_ids"] = [tr_id]
            coincidence_pick["coincidence_sum"] = float(trace_ids[tr_id])
            coincidence_pick["similarity"] = {}
            
            if simil is not None:
                coincidence_pick['similarity'][sta] = simil


            for(tmp_on,tmp_off,tmp_tr_id,tmp_simil) in picks:
                tmp_sta = tmp_tr_id.split(".")[1]
                # skip retriggering of already present station in current
                # coincidence trigger

                if tmp_tr_id in coincidence_pick['trace_ids']:
                    continue
                if tmp_on > off + trigger_off_extension:
                    break     

                coincidence_pick['stations'].append(tmp_sta)
                coincidence_pick['trace_ids'].append(tmp_tr_id)
                coincidence_pick['coincidence_sum'] += trace_ids[tmp_tr_id]
                off = max(off, tmp_off)
            # skip if both coincidence sum and similarity thresholds are not met




            if coincidence_pick['coincidence_sum'] < thr_coincidence_sum:
                if not coincidence_pick['similarity']:
                    continue
                elif not any([val > similarity_threshold[_s]
                            for _s, val in coincidence_pick['similarity'].items()]):
                    continue
            # skip coincidence trigger if it is just a subset of the previous
            # (determined by a shared off-time, this is a bit sloppy)
            if off <= last_off_time:
                continue
            coincidence_pick['duration'] = off - on
            coincidence_picks.append(coincidence_pick)
            last_off_time = off

        return coincidence_picks

    except Exception as e:

        logger.info(f"Error in coincidence detections was CTM: {e}")
        return []














def get_detections(xaap_config, single_stream,model):

    """
    Retrieves seismic detections using a specified deep learning model from SeisBench.
    
    If successful, it returns the detections otherwise, it returns an empty list.
    NOTES:
    Add channel name to the detection;
    If the model returns just phases, add a pad to the start and end (cusmizable by the operator)
    If the model returns phases and detections, convert both to detections

    Parameters:
    - xaap_config (object): Configuration object containing attributes for the deep learning model's name and version.
        - xaap_config.deep_learning_model_name (str): Name of the deep learning model to be used.
        - xaap_config.deep_learning_model_version (str): Version of the deep learning model to be used.
    - single_stream (object): Seismic stream data to be processed.

    Returns:
    - list: List of detections if successful, otherwise an empty list.

    Raises:
    - Exception: If there's an error while creating or using the deep learning model.
    """

    try:
        logger.info("Start of model.classify()")
        classify_results = model.classify(single_stream)

        detections = []
        padding_start = 10
        padding_end = 20
        if hasattr(classify_results, 'picks') and hasattr(classify_results, 'detections'):
            logger.info("Convert picks to detection_xaap")
            for pick in classify_results.picks:
                detection_xaap = DetectionXaap(pick,single_stream,padding_start, padding_end)
                detections.append(detection_xaap)

            for detection in classify_results.detections:
                detections.append(DetectionXaap(detection,single_stream,padding_start, padding_end))

            
        elif hasattr(classify_results, 'picks'):
            logger.info("Convert picks to detection_xaap")
            for pick in classify_results.picks:
                detection_xaap = DetectionXaap(pick,single_stream,padding_start, padding_end)
                detections.append(detection_xaap)

        elif hasattr(classify_results, 'detections'):
            for detection in classify_results.detections:
                detection_xaap = DetectionXaap(detection,single_stream,padding_start, padding_end)
                detections.append(detection_xaap)
        else:
            logger.info(f"Result was: {classify_results}")


        return detections

    except Exception as e:
        logger.error(f"Error using deep learning model from seisbench: {model.name}, error was: {e}")
        return []


