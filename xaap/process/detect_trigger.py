from obspy.signal.trigger import coincidence_trigger
import pandas as pd

from obspy import UTCDateTime
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

def detect_triggers(self):

    sta = self.params['Parameters','STA_LTA','sta']
    lta = self.params['Parameters','STA_LTA','lta']
    trigon = self.params['Parameters','STA_LTA','trigon']
    trigoff = self.params['Parameters','STA_LTA','trigoff']
    coincidence = self.params['Parameters','STA_LTA','coincidence']
    endtime_extra = self.params['Parameters','STA_LTA','endtime_extra']
    
    
    
    self.triggers_traces = []

    self.triggers = coincidence_trigger("recstalta", trigon, trigoff, self.volcan_stream, coincidence, sta=sta, lta=lta)
    #for i,trg in enumerate(self.triggers):
    #    logger.info("%s:%s.%s %s" %(i,trg['time'],trg['trace_ids'][0],trg['duration']))
    triggers_pd = pd.DataFrame(self.triggers)

    detected_triggers_file = Path(xaap_dir,"data/detections") / UTCDateTime.now().strftime("trigger_xaap_%Y.%m.%d.%H.%M.%S.csv")
    triggers_pd.to_csv(detected_triggers_file)
    print(triggers_pd)
    logger.info("Total picks detected: %s " %(len(triggers_pd.stations)))
    triggers_on =[]
    trigger_dot_list =[]
    
    for i,trigger in enumerate(self.triggers):
        trigger_start_timestamp = trigger['time'].timestamp
        trigger_start = trigger['time']
        trigger_duration = trigger['duration'] * endtime_extra
        triggers_on.append(trigger_start_timestamp)
        trigger_dot_list.append(0)
        
        trigger_stream_temp=self.volcan_stream.select(id=trigger['trace_ids'][0])
        trigger_trace = trigger_stream_temp[0].slice(trigger_start, trigger_start + trigger_duration)
        #MINIMUM TRIGGER LENGTH
        if trigger_trace.count() > 1:
            self.triggers_traces.append(trigger_trace)


    for trigger in self.triggers:
        for trace_id in trigger['trace_ids']:
            for plot_item in self.plot_items_list:
                if plot_item.getAxis("left").labelText == trace_id:
                    trigger_trace_temp=self.volcan_stream.select(id=trace_id)
                    trigger_window = trigger_trace_temp.slice(trigger['time'],trigger['time']+trigger_duration)
                    #plot_item.plot([trigger['time']],[0],pen=None,symbol='x')
                    plot_item.plot(trigger_window[0].times(type='timestamp'),trigger_window[0].data,pen='r')

