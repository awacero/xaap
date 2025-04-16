# -*- coding: utf-8 -*-

import os, sys
#import symbol
from pathlib import Path

from get_mseed_data import get_mseed_utils as gmutils
from get_mseed_data import get_mseed
import numpy as np
from obspy import UTCDateTime
import pandas as pd
import logging, logging.config

#from mpl_axes_aligner import align

xaap_dir = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])))
xaap_config_dir = Path("%s/%s" %(xaap_dir,"config"))
logging.config.fileConfig(xaap_config_dir / "logging.ini" ,disable_existing_loggers=False)
logger = logging.getLogger('stdout')
logger.setLevel(logging.INFO)



def connect_to_mseed_server(self):

    try:
        self.mseed_client_id = self.params['mseed','client_id']
        mseed_server_config_file = xaap_config_dir / self.params['mseed','server_config_file']
        mseed_server_param = gmutils.read_config_file(mseed_server_config_file)
        
        self.mseed_client = get_mseed.choose_service(mseed_server_param[self.mseed_client_id])

    except Exception as e:
        raise Exception("Error connecting to MSEED server : %s" %str(e))

def load_csv_file(self):

    try:
        classification_file_path = Path(self.classifications_path ,\
                                        self.params['classification','classification_file']+
                                        '.csv')
        predicted_data  = pd.read_csv(classification_file_path,sep=',')
        rows_length,column_length = predicted_data.shape
        column_names = ['trigger_code','prediction']
        predicted_data.columns = column_names
        predicted_data['operator'] = ''
        predicted_data[['Network', 'Station', '','Component','year',\
                        'month','day','h','m','s','coda']] = \
            predicted_data['trigger_code'].str.split('.', expand=True)
        predicted_data["Hora"] = predicted_data['h'] +":"\
                                    + predicted_data["m"] +":"\
                                    + predicted_data["s"]
        predicted_data["Fecha"] = predicted_data['day']\
                                    + "/" + predicted_data['month']\
                                    + "/" + predicted_data['year']

        return predicted_data

    except Exception as e:
        logger.error("Error reading classification file : %s" %str(e))
        raise Exception("Error reading classification file : %s" %str(e))

def get_trigger(self,trigger_code):

    net,station,location,channel,Y,m,d,H,M,S,window = trigger_code.split(".")
    if not location:
        location = ''
    start_time = UTCDateTime("%s-%s-%sT%s:%s:%s"%(Y,m,d,H,M,S))
    window = int(window)
    end_time = start_time + window

    # poner try, llamar a preprocesar, usar informacion de filtros en
    # parametros, agregar pads?
    self.trigger_stream =\
        get_mseed.get_stream(self.mseed_client_id,self.mseed_client,net,\
                                station,location,channel,start_time=start_time,\
                                window_size=window)
    self.trigger_times = self.trigger_stream[0].times(type='timestamp')

    # Normalización a 0 de la gráfica superior
    self.trigger_stream[0].data =\
        np.array([x - np.mean(self.trigger_stream[0].data)\
        for x in self.trigger_stream[0].data])

    pad = 300
        
    self.paded_stream =\
        get_mseed.get_stream(self.mseed_client_id,self.mseed_client,\
                                net,station,location,channel,\
                                start_time=start_time - pad ,\
                                end_time=end_time +  pad)

    logger.info("Get paded stream")
    # Normalización a 0 de la gráfica inferior
    self.paded_stream[0].data =\
        np.array([x - np.mean(self.paded_stream[0].data)\
        for x in self.paded_stream[0].data])

    self.paded_times = self.paded_stream[0].times(type='timestamp')

    # Data about selected row
    # Max and min amplitude in trigger_stream
    max_trigger = float(max(self.trigger_stream[0].data))
    min_trigger = float(min(self.trigger_stream[0].data))
    amp_max = abs(max_trigger)+abs(min_trigger)
    # Localizar el máximo y mínimo
    max_loc = np.where(self.trigger_stream[0].data==max_trigger)
    min_loc = np.where(self.trigger_stream[0].data==min_trigger)
    # RMS
    #print("RMS:",np.std(self.trigger_stream[0].data))
    rms = np.sqrt(np.mean(self.trigger_stream[0].data.size**2))

    # T
    #T = [x for x in self.trigger_stream[0].data if (x >max_trigger and x<0)]
    #print(T)

    return self.trigger_times, self.paded_stream,\
            self.trigger_stream, self.paded_times,\
            max_trigger, min_trigger, amp_max, max_loc, min_loc, rms
