import pandas as pd
import numpy as np
from pathlib import Path

from obspy import UTCDateTime
from obspy import Stream
from  obspy.core.event import Pick
from obspy.signal.trigger import (classic_sta_lta, classic_sta_lta_py,
                                  coincidence_trigger, plot_trigger,
                                  recursive_sta_lta_py, trigger_onset)


import seisbench
import seisbench.models as sbm
from  seisbench.util.annotations import Pick as sb_pick
from  seisbench.util.annotations import Detection as sb_detection
import logging
logger = logging.getLogger(__name__)



def get_triggers( xaap_config, volcan_stream ):

    """
    Detects triggers in a seismic data stream using the Short Time Average to Long Time Average (STA/LTA) trigger algorithm applying Coincidence trigger algorithm

    Parameters:
        xaap_config (object): An object containing the configuration settings for the XAAP system, including the parameters for the STA/LTA trigger algorithm.
        volcan_stream (obspy stream): An obspy stream object containing seismic data from a volcano.

    Returns:
        list: A list of trigger (event) dictionaries, each containing the trigger time, trace ID, and duration. Returns an empty list if no triggers are detected.

    The function takes as input the configuration settings for the XAAP system and a seismic data stream from a volcano. 
    It then extracts the STA/LTA trigger parameters from the configuration object, sets up the coincidence trigger object using these parameters,
    and applies it to the seismic data stream to identify potential triggers.

    If triggers are detected, they are saved to a CSV file with a timestamped filename, and the function returns the list of trigger dictionaries. 
    If no triggers are detected, the function logs a message indicating that no triggers were detected and returns an empty list.
    """

    sta = float(xaap_config.sta_lta_sta)
    lta = float(xaap_config.sta_lta_lta)
    trigon = float(xaap_config.sta_lta_trigon)
    trigoff = float(xaap_config.sta_lta_trigoff)
    sta_lta_coincidence = float(xaap_config.sta_lta_coincidence)
    

    triggers = coincidence_trigger("recstalta", trigon, trigoff, volcan_stream, 
                                    sta_lta_coincidence, sta=sta, lta=lta)

    if len(triggers) != 0:

        triggers_pd = pd.DataFrame(triggers)
        detected_triggers_file = Path(xaap_config.output_detection_folder) / UTCDateTime.now().strftime("trigger_xaap_%Y.%m.%d.%H.%M.%S.csv")
        triggers_pd.to_csv(detected_triggers_file)
        logger.info("Total picks detected: %s " %(len(triggers_pd.stations)))
        return triggers

    else:
        logger.info("NO TRIGGERS DETECTED")
        triggers = []
        return triggers


def get_triggers_deep_learning(xaap_config, volcan_stream):

    """
    Retrieves seismic triggers using a specified deep learning model from SeisBench.
    
    This function attempts to load a deep learning model based on the provided configuration 
    and then uses this model to annotate and classify the given seismic stream. 
    If successful, it returns the picks and detections; otherwise, it logs the error and returns an empty list.

    Parameters:
    - xaap_config (object): Configuration object containing attributes for the deep learning model's name and version.
        - xaap_config.deep_learning_model_name (str): Name of the deep learning model to be used.
        - xaap_config.deep_learning_model_version (str): Version of the deep learning model to be used.
    - volcan_stream (object): Seismic stream data to be processed.

    Returns:
    - list: List of picks and detections if successful, otherwise an empty list.

    Raises:
    - Exception: If there's an error while creating or using the deep learning model.
    """

    model_name = xaap_config.deep_learning_model_name
    model_version = xaap_config.deep_learning_model_version

    logger.info(f"Try to create model {model_name} with version:{model_version}")

    try:
        model = getattr(sbm,model_name).from_pretrained(model_version)
    except Exception as e:
        logger.error(f"Error while creating deep learning model from seisbench: {model_name} and {model_version}: {e}")
        return []

    try:
        '''
        annotations = model.annotate(volcan_stream)
        if len(annotations)!=0:
            classify_results = model.classify(volcan_stream)
            return classify_results
        '''
        classify_results = model.classify(volcan_stream)
        if len(classify_results) !=0:
            return classify_results
        else:
            return []


    except Exception as e:
        logger.error(f"Error using deep learning model from seisbench: {model_name} and {model_version}. Error: {e}")
        return[]


def coincidence_trigger_deep_learning(xaap_config, stream,
                        #thr_coincidence_sum, 
                        trace_ids=None,
                        max_trigger_length=1e6, delete_long_trigger=False,
                        trigger_off_extension=0, details=False,
                        event_templates={}, similarity_threshold=0.7,
                        **options):

    model_name = xaap_config.deep_learning_model_name
    model_version = xaap_config.deep_learning_model_version
    thr_coincidence_sum = xaap_config.deep_learning_coincidence_picks
    
    # if no trace ids are specified use all traces ids found in stream
    if trace_ids is None:
        trace_ids = [tr.id for tr in stream]
    # we always work with a dictionary with trace ids and their weights later
    if isinstance(trace_ids, list) or isinstance(trace_ids, tuple):
        trace_ids = dict.fromkeys(trace_ids, 1)
    # set up similarity thresholds as a dictionary if necessary
    if not isinstance(similarity_threshold, dict):
        similarity_threshold = dict.fromkeys(
            [tr.stats.station for tr in stream], similarity_threshold)

    # the single station triggering
    triggers = []
    picks =[]
    tmp_triggers = []
    # prepare kwargs for trigger_onset
    kwargs = {'max_len_delete': delete_long_trigger}
    for tr in stream:
        tr = tr.copy()
        if tr.id not in trace_ids:
            msg = "At least one trace's ID was not found in the " + \
                  "trace ID list and was disregarded (%s)" % tr.id
            continue

        kwargs['max_len'] = int(max_trigger_length * tr.stats.sampling_rate + 0.5)


        ####tmp_triggers = trigger_onset(tr.data, thr_on, thr_off, **kwargs)
        #picks,tmp_triggers = get_triggers_deep_learning(xaap_config,Stream(tr))

        classify_results = get_triggers_deep_learning(xaap_config,Stream(tr))


        if classify_results is not None:
            
            if len(classify_results) != 0:

                if isinstance(classify_results,tuple) and len(classify_results) == 2:
                    logger.info(f"Model {model_name} for picks and detections" )
                    picks.extend(classify_results[0])
                    tmp_triggers = classify_results[1]
                else:

                    if isinstance(classify_results[0],sb_pick):
                        logger.info(f"Model {model_name} just for picks")
                        picks.extend(classify_results)
                        tmp_triggers=[]
                    
                    if isinstance(classify_results[0],sb_detection):
                        logger.info(f"Model {model_name} just for detections")
                        picks = []
                        tmp_triggers = classify_results
            else:
                logger.info(f"length of classify_results was {len(classify_results)}")


        else:
            logger.info("No pick or detection made")

            pass


        for trigger in tmp_triggers:
            try:
                cft_peak = trigger.peak_value
                cft_std = tr.slice(trigger.start_time,trigger.end_time).std()
                
            except ValueError:
                idx = tr.times().searchsorted(trigger.start_time)
                cft_peak = tr.data[idx]
                cft_std = 0


            on = trigger.start_time
            off = trigger.end_time
            triggers.append((on.timestamp,off.timestamp,tr.id,cft_peak,cft_std))
   
    triggers.sort()

    for i, (on, off, tr_id, cft_peak, cft_std) in enumerate(triggers):
        sta = tr_id.split(".")[1]
        templates = event_templates.get(sta)

        simil = None
        triggers[i] = (on, off, tr_id, cft_peak, cft_std, simil)

    # the coincidence triggering and coincidence sum computation
    coincidence_triggers = []
    last_off_time = 0.0
    while triggers != []:
        # remove first trigger from list and look for overlaps
        on, off, tr_id, cft_peak, cft_std, simil = triggers.pop(0)
        sta = tr_id.split(".")[1]
        event = {}
        event['time'] = UTCDateTime(on)
        event['stations'] = [tr_id.split(".")[1]]
        event['trace_ids'] = [tr_id]
        event['coincidence_sum'] = float(trace_ids[tr_id])
        event['similarity'] = {}
        if details:
            event['cft_peaks'] = [cft_peak]
            event['cft_stds'] = [cft_std]
        # evaluate maximum similarity for station if event templates were
        # provided
        if simil is not None:
            event['similarity'][sta] = simil
        # compile the list of stations that overlap with the current trigger
        for (tmp_on, tmp_off, tmp_tr_id, tmp_cft_peak, tmp_cft_std,
                tmp_simil) in triggers:
            tmp_sta = tmp_tr_id.split(".")[1]
            # skip retriggering of already present station in current
            # coincidence trigger
            if tmp_tr_id in event['trace_ids']:
                continue
            # check for overlapping trigger,
            # break if there is a gap in between the two triggers
            if tmp_on > off + trigger_off_extension:
                break
            event['stations'].append(tmp_sta)
            event['trace_ids'].append(tmp_tr_id)
            event['coincidence_sum'] += trace_ids[tmp_tr_id]
            if details:
                event['cft_peaks'].append(tmp_cft_peak)
                event['cft_stds'].append(tmp_cft_std)
            # allow sets of triggers that overlap only on subsets of all
            # stations (e.g. A overlaps with B and B overlaps w/ C => ABC)
            off = max(off, tmp_off)
            # evaluate maximum similarity for station if event templates were
            # provided
            if tmp_simil is not None:
                event['similarity'][tmp_sta] = tmp_simil
        # skip if both coincidence sum and similarity thresholds are not met
        if event['coincidence_sum'] < thr_coincidence_sum:
            if not event['similarity']:
                continue
            elif not any([val > similarity_threshold[_s]
                          for _s, val in event['similarity'].items()]):
                continue
        # skip coincidence trigger if it is just a subset of the previous
        # (determined by a shared off-time, this is a bit sloppy)
        if off <= last_off_time:
            continue
        event['duration'] = off - on
        if details:
            weights = np.array([trace_ids[i] for i in event['trace_ids']])
            weighted_values = np.array(event['cft_peaks']) * weights
            event['cft_peak_wmean'] = weighted_values.sum() / weights.sum()
            weighted_values = np.array(event['cft_stds']) * weights
            event['cft_std_wmean'] = \
                (np.array(event['cft_stds']) * weights).sum() / weights.sum()
        coincidence_triggers.append(event)
        last_off_time = off



    if len(picks) > 0:

        logger.info(f"Model {model_name} store picks and detections" )
        picks_df = pd.DataFrame(vars(p) for p in picks)
        detected_picks_file = Path(xaap_config.output_detection_folder) / UTCDateTime.now().strftime(f"picks_xaap_{model_name}_{model_version}_%Y.%m.%d.%H.%M.%S.csv")
        picks_df.to_csv(detected_picks_file)           

    if len(tmp_triggers) > 0:
        triggers_df = pd.DataFrame(vars(d) for d in tmp_triggers)
        detected_triggers_file = Path(xaap_config.output_detection_folder) / UTCDateTime.now().strftime(f"triggers_raw_xaap_{model_name}_{model_version}_%Y.%m.%d.%H.%M.%S.csv")
        triggers_df.to_csv(detected_triggers_file)

    if len(coincidence_triggers)>0:

        coincidence_triggers_df = pd.DataFrame(coincidence_triggers)
        detected_triggers_file = Path(xaap_config.output_detection_folder) / UTCDateTime.now().strftime(f"triggers_coincidence_xaap_{model_name}_{model_version}_%Y.%m.%d.%H.%M.%S.csv")
        coincidence_triggers_df.to_csv(detected_triggers_file)

    return picks,coincidence_triggers








def coincidence_pick_trigger_deep_learning(xaap_config, stream,
                        #thr_coincidence_sum, 
                        trace_ids=None,
                        max_trigger_length=1e6, delete_long_trigger=False,
                        trigger_off_extension=0, details=False,
                        event_templates={}, similarity_threshold=0.7,
                        **options):

    model_name = xaap_config.deep_learning_model_name
    model_version = xaap_config.deep_learning_model_version
    thr_coincidence_sum = xaap_config.deep_learning_coincidence_picks
    
    # if no trace ids are specified use all traces ids found in stream
    if trace_ids is None:
        trace_ids = [tr.id for tr in stream]
        logger.info(f"###VALORES EN EL TRACE_IDS {trace_ids}")
    # we always work with a dictionary with trace ids and their weights later
    if isinstance(trace_ids, list) or isinstance(trace_ids, tuple):
        trace_ids = dict.fromkeys(trace_ids, 1)
    # set up similarity thresholds as a dictionary if necessary
    if not isinstance(similarity_threshold, dict):
        similarity_threshold = dict.fromkeys(
            [tr.stats.station for tr in stream], similarity_threshold)

    # the single station triggering
    triggers = []
    picks =[]
    triggers_raw = []
    picks_raw = []
    tmp_triggers = []
    tmp_picks = []
    # prepare kwargs for trigger_onset
    kwargs = {'max_len_delete': delete_long_trigger}

    ##Procesar todas las trazas del stream

    for tr in stream:

        logger.info(f" ### Trace to feed to DL: {tr}")



        tr = tr.copy()
        if tr.id not in trace_ids:
            msg = "At least one trace's ID was not found in the " + \
                  "trace ID list and was disregarded (%s)" % tr.id
            continue

        kwargs['max_len'] = int(max_trigger_length * tr.stats.sampling_rate + 0.5)


        ####tmp_triggers = trigger_onset(tr.data, thr_on, thr_off, **kwargs)
        #picks,tmp_triggers = get_triggers_deep_learning(xaap_config,Stream(tr))

        classify_results = get_triggers_deep_learning(xaap_config,Stream(tr))


        if classify_results is not None:
            
            if len(classify_results) != 0:

                if isinstance(classify_results,tuple) and len(classify_results) == 2:
                    logger.info(f"Model {model_name} for picks and detections" )
                    tmp_picks= classify_results[0]
                    tmp_triggers = classify_results[1]
                    
                    logger.info(f"PICKS {[(pick.start_time, pick.trace_id)for pick in tmp_picks]}")
                    logger.info(f"TRIGGERS {[(trigger.start_time, trigger.trace_id) for trigger in tmp_triggers]}")

                else:

                    if isinstance(classify_results[0],sb_pick):
                        logger.info(f"Model {model_name} just for picks")
                        #picks.extend(classify_results)
                        tmp_picks= classify_results

                        ##EL REINICIO SE REALIZA ARRIBA? O NO?
                        ###tmp_triggers=[]
                    
                    if isinstance(classify_results[0],sb_detection):
                        logger.info(f"Model {model_name} just for detections")
                        tmp_triggers = classify_results
            else:
                logger.info(f"Not expected length of classify_results,it was {len(classify_results)}")


        else:
            logger.info(f"No pick or detection made. Result was {classify_results}")

            #pass


        for trigger in tmp_triggers:
            try:
                cft_peak = trigger.peak_value
                cft_std = tr.slice(trigger.start_time,trigger.end_time).std()
                
            except ValueError:
                idx = tr.times().searchsorted(trigger.start_time)
                cft_peak = tr.data[idx]
                cft_std = 0


            on = trigger.start_time
            off = trigger.end_time
            ###Agregar el canal en el TRACE.ID 
            triggers_raw.append(trigger)
            triggers.append((on.timestamp,off.timestamp,tr.id,cft_peak,cft_std))
   

        '''Crea una copia de la lista de objetos PICK en picks_raw'''
        for pick in tmp_picks:

            '''
            print("#####################################")
            print(f"####{type(pick)}")
            print(f"####{pick}")
            '''

            #on = pick.start_time
            if pick.peak_time is None:
                on = pick.start_time
            else:
                on = pick.peak_time
            
            off = pick.end_time
             
            picks_raw.append(pick)
            """Append to a list with picks tuple from sb.picks objects
               The channel information is recovered in this part
            """
            picks.append((on.timestamp,off.timestamp,tr.id,pick.phase))

    
    """Start of coincidence pick modified from coincidence trigger"""

    picks.sort()

    for i,(on,off,tr_id,phase) in enumerate(picks):
        simil = None
        picks[i] = (on, off,tr_id,simil,phase)

    coincidence_picks = []
    last_off_time = 0.0
    while picks != []:
        on, off, tr_id, simil,phase = picks.pop(0)
        sta =  tr_id.split(".")[1]
        coincidence_pick = {}
        coincidence_pick['time']=UTCDateTime(on)
        coincidence_pick['stations'] = [sta]
        coincidence_pick["trace_ids"] = [tr_id]
        coincidence_pick["coincidence_sum"] = float(trace_ids[tr_id])
        coincidence_pick["similarity"] = {}
        coincidence_pick["phase"] = [phase]
        if simil is not None:
            coincidence_pick['similarity'][sta] = simil

        for(tmp_on,tmp_off,tmp_tr_id,tmp_simil,tmp_phase) in picks:
            tmp_sta = tmp_tr_id.split(".")[1]
            # skip retriggering of already present station in current
            # coincidence trigger
            if tmp_phase != phase:
                continue
            if tmp_tr_id in coincidence_pick['trace_ids']:
                continue
            if tmp_on > off + trigger_off_extension:
                break     

            coincidence_pick['stations'].append(tmp_sta)
            coincidence_pick['trace_ids'].append(tmp_tr_id)
            coincidence_pick['phase'].append(tmp_phase)
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

    logger.info(f"@@@@@@@@@@@@FIN DE COINCIDENCE PICKS WAS:")
    for c_p in coincidence_picks:
        print(c_p)




    triggers.sort()

    for i, (on, off, tr_id, cft_peak, cft_std) in enumerate(triggers):
        sta = tr_id.split(".")[1]
        templates = event_templates.get(sta)

        simil = None
        triggers[i] = (on, off, tr_id, cft_peak, cft_std, simil)

    # the coincidence triggering and coincidence sum computation
    coincidence_triggers = []
    last_off_time = 0.0
    while triggers != []:
        # remove first trigger from list and look for overlaps
        on, off, tr_id, cft_peak, cft_std, simil = triggers.pop(0)
        sta = tr_id.split(".")[1]
        event = {}
        event['time'] = UTCDateTime(on)
        event['stations'] = [tr_id.split(".")[1]]
        event['trace_ids'] = [tr_id]
        event['coincidence_sum'] = float(trace_ids[tr_id])
        event['similarity'] = {}
        if details:
            event['cft_peaks'] = [cft_peak]
            event['cft_stds'] = [cft_std]
        # evaluate maximum similarity for station if event templates were
        # provided
        if simil is not None:
            event['similarity'][sta] = simil
        # compile the list of stations that overlap with the current trigger
        for (tmp_on, tmp_off, tmp_tr_id, tmp_cft_peak, tmp_cft_std,
                tmp_simil) in triggers:
            tmp_sta = tmp_tr_id.split(".")[1]
            # skip retriggering of already present station in current
            # coincidence trigger
            ###REVISAR AQUI QUE NO ESTA QUITANDO LOS CANALES 
            if tmp_tr_id in event['trace_ids']:
                #logger.info(f"trace already used {tmp_tr_id}")
                continue
            # check for overlapping trigger,
            # break if there is a gap in between the two triggers
            if tmp_on > off + trigger_off_extension:
                break
            event['stations'].append(tmp_sta)
            event['trace_ids'].append(tmp_tr_id)
            event['coincidence_sum'] += trace_ids[tmp_tr_id]
            if details:
                event['cft_peaks'].append(tmp_cft_peak)
                event['cft_stds'].append(tmp_cft_std)
            # allow sets of triggers that overlap only on subsets of all
            # stations (e.g. A overlaps with B and B overlaps w/ C => ABC)
            off = max(off, tmp_off)
            # evaluate maximum similarity for station if event templates were
            # provided
            if tmp_simil is not None:
                event['similarity'][tmp_sta] = tmp_simil
        # skip if both coincidence sum and similarity thresholds are not met
        if event['coincidence_sum'] < thr_coincidence_sum:
            if not event['similarity']:
                continue
            elif not any([val > similarity_threshold[_s]
                          for _s, val in event['similarity'].items()]):
                continue
        # skip coincidence trigger if it is just a subset of the previous
        # (determined by a shared off-time, this is a bit sloppy)
        if off <= last_off_time:
            continue
        event['duration'] = off - on
        if details:
            weights = np.array([trace_ids[i] for i in event['trace_ids']])
            weighted_values = np.array(event['cft_peaks']) * weights
            event['cft_peak_wmean'] = weighted_values.sum() / weights.sum()
            weighted_values = np.array(event['cft_stds']) * weights
            event['cft_std_wmean'] = \
                (np.array(event['cft_stds']) * weights).sum() / weights.sum()
        coincidence_triggers.append(event)
        last_off_time = off





    if len(picks_raw) > 0:

        logger.info(f"Model {model_name} store picks and detections" )
        picks_df = pd.DataFrame(vars(p) for p in picks_raw)
        detected_picks_file = Path(xaap_config.output_detection_folder) / UTCDateTime.now().strftime(f"picks_raw_xaap_{model_name}_{model_version}_%Y.%m.%d.%H.%M.%S.csv")
        picks_df.to_csv(detected_picks_file)           

    if len(triggers_raw) > 0:
        triggers_df = pd.DataFrame(vars(d) for d in triggers_raw)
        detected_triggers_file = Path(xaap_config.output_detection_folder) / UTCDateTime.now().strftime(f"triggers_raw_xaap_{model_name}_{model_version}_%Y.%m.%d.%H.%M.%S.csv")
        triggers_df.to_csv(detected_triggers_file)

    if len(coincidence_triggers)>0:

        coincidence_triggers_df = pd.DataFrame(coincidence_triggers)
        coincidence_triggers_file = Path(xaap_config.output_detection_folder) / UTCDateTime.now().strftime(f"triggers_coincidence_xaap_{model_name}_{model_version}_%Y.%m.%d.%H.%M.%S.csv")
        coincidence_triggers_df.to_csv(coincidence_triggers_file)
        ###SOLO PARA CRED
        ##coincidence_triggers_df.to_csv(xaap_config.out_temp)
    else:
        print("NO COINCIDENCE TRIGGERS")

    if len(coincidence_picks)>0:

        coincidence_picks_df = pd.DataFrame(coincidence_picks)
        coincidence_picks_file = Path(xaap_config.output_detection_folder) / UTCDateTime.now().strftime(f"picks_coincidence_xaap_{model_name}_{model_version}_%Y.%m.%d.%H.%M.%S.csv")
        coincidence_picks_df.to_csv(coincidence_picks_file)
        ###PARA TODOS EXCEPTO CRED
        #print("$$$$$$$$$$$$$ESCRIBIR LOS PICKS COINCIDENTES")
        #coincidence_picks_df.to_csv(xaap_config.out_temp)
    else:
        print("NO COINCIDENCE PICKS")

    return coincidence_picks,coincidence_triggers






def get_triggers_ml__(xaap_config, volcan_stream):

    #original coto ok
    #ethz chiles ok
    #model = sbm.EQTransformer.from_pretrained("original")
    model = sbm.PhaseNet.from_pretrained("")
    annotations = model.annotate(volcan_stream)

    if len(annotations)!=0:

        picks, detections = model.classify(volcan_stream)
        '''
        print("Picks:")

        for pick in picks:
            print(type(pick))
            print(pick)

        print("\nDetections:")    
        for detection in detections:
            print(detection)
        '''
        #picks_pd = pd.DataFrame(picks,columns=["pick_waveform_id","pick_time","pick_pahse_hint"])
        picks_split = [pick.__str__().split() for pick in picks]
        picks_df = pd.DataFrame(picks_split, columns=["pick_waveform_id","pick_time","pick_phase_hint"])
        
        detected_picks_file = Path(xaap_config.output_detection_folder) / UTCDateTime.now().strftime("picks_xaap_ml_%Y.%m.%d.%H.%M.%S.csv")
        picks_df.to_csv(detected_picks_file)


        triggers_pd = pd.DataFrame(detections)
        detected_triggers_file = Path(xaap_config.output_detection_folder) / UTCDateTime.now().strftime("triggers_xaap_ml_%Y.%m.%d.%H.%M.%S.csv")
        triggers_pd.to_csv(detected_triggers_file)
        return picks



def create_obspy_pick(pick_df):

    pick_time = UTCDateTime("2023-01-01T12:34:56")
    pick_phase_hint = "P"  # P, S, or any other phase label
    pick_waveform_id = "NW.STA01..HHZ"  # Network.Station.Location.Channel

    # Create the Pick object
    my_pick = Pick(time=pick_time, phase_hint=pick_phase_hint, waveform_id=pick_waveform_id)

    print(my_pick)


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
