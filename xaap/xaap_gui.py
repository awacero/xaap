# -*- coding: utf-8 -*-

import os,sys
from pathlib import Path
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.parametertree import Parameter, ParameterTree
import pyqtgraph.configfile

from get_mseed_data import get_mseed_utils as gmutils
from get_mseed_data import get_mseed

from obspy.signal import filter  
from obspy import Trace, Stream
import numpy as np

from obspy import read, UTCDateTime
from obspy.signal.trigger import classic_sta_lta
from obspy.signal.trigger import classic_sta_lta, classic_sta_lta_py, recursive_sta_lta_py, plot_trigger, trigger_onset
from datetime import date, datetime

import pandas as pd
from aaa_features.features import FeatureVector 
from sklearn.preprocessing import StandardScaler
import pickle 
from sklearn.ensemble import RandomForestClassifier

import logging, logging.config

xaap_dir = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])))
xaap_config_dir = Path("%s/%s" %(xaap_dir,"config"))
logging.config.fileConfig(xaap_config_dir / "logging.ini" ,disable_existing_loggers=False)
logger = logging.getLogger('stdout')
logger.setLevel(logging.INFO)

class xaapGUI(QtGui.QWidget):  

    def __init__(self):

        logger.info("Start of all this @#$. Working directory %s" %xaap_dir)

        QtGui.QWidget.__init__(self)

        self.setupGUI()       
        self.xaap_dir = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])))       
        self.params = self.create_parameters()
        self.parameters_dir = Path(self.xaap_dir, 'config/xaap_parameters')
        if os.path.exists(self.parameters_dir):
            presets = [os.path.splitext(p)[0] for p in os.listdir(self.parameters_dir)]
            self.params.param('Load Preset..').setLimits(['']+presets)
        self.tree.setParameters(self.params, showTop=True)
        logger.info("Configure parameter signals")
        '''loadPreset is called with parameters: param and preset '''
        self.params.param('Load Preset..').sigValueChanged.connect(self.loadPreset)
        self.params.param('Save').sigActivated.connect(self.save)
        self.params.param('Request Data').sigActivated.connect(self.request_stream)
        self.params.param('Pre-process').sigActivated.connect(self.pre_process_stream)
        self.params.param('Plot stream').sigActivated.connect(self.plot_stream)
        self.window_region.sigRegionChanged.connect(self.set_p1_using_p2)
        self.p1.sigRangeChanged.connect(self.set_p2_using_p1)
        self.params.param('Detect triggers').sigActivated.connect(self.detect_triggers)
        self.params.param('Classify triggers').sigActivated.connect(self.classify_triggers)

        '''Start of automatic process '''
        self.request_stream()

        if self.stream == None:
            logger.info("No data available")
        elif len(self.stream[0].data)==0:
            logger.info("No data available")
        else:

            logger.info("ok request_stream() in __init__. Continue with Pre-process")
            self.pre_process_stream()
            self.plot_stream()
            self.detect_triggers()
            #self.classify_triggers()
            #self.window_region.sigRegionChanged.connect(self.set_p1_using_p2)
            





    def create_parameters(self):
        
        xaap_parameters = Parameter.create(
            
            name='xaap configuration',type='group',children=[
            {'name':'Load Preset..','type':'list','limits':[]},
            {'name':'Save','type':'action'},
            {'name':'Load','type':'action'},
            {'name':'Parameters','type':'group','children':[

                {'name':'MSEED','type':'group','children':[
                    {'name':'client_id','type':'list','values':['ARCLINK','SEEDLINK','ARCHIVE','FDSN']},
                    {'name':'server_config_file','type':'str','value':'%s' %('server_configuration.json')}
                                                                 ]},
                {'name':'Station','type':'group','children':[
                    {'name':'network','type':'str','value':'EC' },
                    {'name':'station','type':'str','value':'BULB' },
                    {'name':'location','type':'str','value':'' },
                    {'name':'channel','type':'str','value':'HHZ' }
                                                            ]},
                {'name':'Dates','type':'group','children':[
                    {'name':'start','type':'str','value':'%s' %(UTCDateTime.now()-86400).strftime("%Y-%m-%d %H:%M:%S") },
                    {'name':'end','type':'str','value':'%s' %UTCDateTime.now().strftime("%Y-%m-%d %H:%M:%S") }
                                                            ]},
                                                          
                {'name':'Filter','type':'group','children':[
                    {'name':'Filter type','type':'list','values':['highpass','bandpass','lowpass']},
                    {'name':'Freq_A','type':'float','value':0.5,'step':0.1,'limits': [0.1, None ]},
                    {'name':'Freq_B','type':'float','value':1.0,'step':0.1,'limits': [0.1, None ]}    ]},

                {'name':'STA_LTA','type':'group', 'children': [
                    {'name':'sta','type':'float','value':5,'step':0.5,'limits': [0.1, None ]},                    
                    {'name':'lta','type':'float','value':10,'step':0.5,'limits': [1, None ]},
                    {'name':'trigon','type':'float','value':1.5,'step':0.5,'limits': [0.5, None ]},
                    {'name':'trigoff','type':'float','value':0.5,'step':0.5,'limits': [0.5, None ]}
                                                                                                  ]},
                {'name':'GUI','type':'group','children':[
                    {'name':'zoom_region_size','type':'float','value':0.10,'step':0.05,'limits':[0.01,1] }
                                                            ]},
                                                          ]},
            {'name':'Request Data','type':'action'},                                                           
            {'name':'Pre-process','type':'action'},
            {'name':'Plot stream','type':'action'},
            {'name':'Detect triggers','type':'action'},
            {'name':'Classify triggers','type':'action'}                
                                                                         ]                                                                         
                                                                         )
        logger.info("End of set parameters") 
        return xaap_parameters


    def request_stream(self):

        try:
            mseed_client_id = self.params['Parameters','MSEED','client_id']
            #mseed_server_config_file = "%s/config/%s" %(self.xaap_dir,self.params['Parameters','MSEED','server_config_file'])
            mseed_server_config_file = xaap_config_dir / self.params['Parameters','MSEED','server_config_file']
            mseed_server_param = gmutils.read_config_file(mseed_server_config_file)
            
            self.mseed_client = get_mseed.choose_service(mseed_server_param[mseed_client_id])

        except Exception as e:
            raise Exception("Error connecting to MSEED server : %s" %str(e))
        
        try:
            mseed_client_id = self.params['Parameters','MSEED','client_id']
            network = self.params['Parameters','Station','network']
            station = self.params['Parameters','Station','station']
            location = self.params['Parameters','Station','location']
            channel = self.params['Parameters','Station','channel']
            start_time = UTCDateTime(self.params['Parameters','Dates', 'start'])
            end_time = UTCDateTime(self.params['Parameters','Dates', 'end'])
        except Exception as e:
            raise Exception("Error request_stream was: %s" %str(e))

        try:
            temp_mseed = get_mseed.get_stream(mseed_client_id,self.mseed_client,network,station,'',channel,start_time=start_time,end_time=end_time)
            logger.info("request_stream() result was: %s" %temp_mseed)
            self.stream = temp_mseed

        except Exception as e:
            logger.error("Error in request_stream was: %s" %str(e))


    def pre_process_stream(self):

        self.processed_stream = None
        logger.info("self.processed_stream cleaned :%s" %self.processed_stream)
        try:
            self.processed_stream = self.stream.copy()
            self.processed_stream.merge(method=1, fill_value="interpolate",interpolation_samples=0)
            logger.info("Stream merged: %s" %self.processed_stream)

            self.times = self.processed_stream[0].times(type="timestamp")
            sampling_rate = self.processed_stream[0].stats.sampling_rate

            f_a = self.params['Parameters','Filter','Freq_A']
            f_b = self.params['Parameters','Filter','Freq_B']

            filter_type = self.params['Parameters','Filter','Filter type']
            if filter_type == 'highpass':
                logger.info("highpass selected")
                temp_data = filter.highpass(self.processed_stream[0].data,f_a,sampling_rate,4)
            elif filter_type == 'bandpass':
                logger.info("bandpass selected")
                temp_data = filter.bandpass(self.processed_stream[0].data,f_a,f_b,sampling_rate,4)
            elif filter_type == 'lowpass':
                logger.info("lowpass selected")
                temp_data = filter.lowpass(self.processed_stream[0].data,f_a,sampling_rate,4)

            self.processed_stream[0].data = temp_data

        except Exception as e:
            logger.error("Error at pre_process_stream() was : %s" %str(e))


    def plot_stream(self):

        logger.info("start plot_stream(). Clean plots")
        self.p1.clearPlots()
        self.p2.clearPlots()

        try:

            logger.info("Plot in p1: %s" %self.processed_stream)
            self.p1.plot(self.times,self.processed_stream[0].data,pen='g')
            logger.info("Plot in p2")
            self.p2d = self.p2.plot(self.times,self.processed_stream[0].data,pen="w")
            
            min_region=0
            max_region=int(len(self.times)*self.params['Parameters','GUI','zoom_region_size'])
            self.window_region.setRegion([self.times[min_region], self.times[max_region]])
            self.p1.setXRange(self.times[min_region], self.times[max_region])

            logger.info("End plot_stream()")

        except Exception as e:

            logger.error("Error in plot_stream() was: %s" %str(e))


    def detect_triggers(self):

        sta = self.params['Parameters','STA_LTA','sta']
        lta = self.params['Parameters','STA_LTA','lta']
        trigon = self.params['Parameters','STA_LTA','trigon']
        trigoff = self.params['Parameters','STA_LTA','trigoff']

        triggers_on = []
        scan_window_size = 600
        start_pad = 30
        end_pad = 30
        sampling_rate = self.processed_stream[0].stats.sampling_rate
        start_time = self.processed_stream[0].stats.starttime
        end_time = self.processed_stream[0].stats.endtime

        n_windows = int(np.ceil(end_time - start_time)/scan_window_size) 
        self.triggers_traces = []
        triggers_on = []
        trigger_dot_list=[]
        for i in range(n_windows):
            window_start = start_time + scan_window_size*i - start_pad
            window_end = start_time + scan_window_size*(i + 1) + start_pad
            
            temp_trace = self.processed_stream[0].slice(window_start,window_end)

            cft = classic_sta_lta(temp_trace.data, int(sta * sampling_rate), int(lta * sampling_rate))

            on_off=trigger_onset(cft, trigon, trigoff)

            if len(on_off) != 0:
                for time in on_off:
                    #print(time)
                    time_on = int(time[0]/sampling_rate)
                    time_off = int(time[1]/sampling_rate)

                    triggers_on.append((window_start + time_on).timestamp)
                    trigger_dot_list.append(0)

                    trigger_trace = temp_trace.slice(window_start + time_on, window_start + time_off)
                    self.triggers_traces.append(trigger_trace)
        
        #print(triggers_on,trigger_dot_list)
        self.p1.plot(triggers_on,trigger_dot_list,pen=None, symbol='x')
        self.p2.plot(triggers_on,trigger_dot_list,pen=None, symbol='x')

    def classify_triggers(self):

        feature_config = {"features_file":"%s/config/features/features_00.json" %self.xaap_dir,
                    "domains":"time spectral cepstral"}
        features = FeatureVector(feature_config, verbatim=2)
        tungu_clf= pickle.load(open(os.path.join('%s/data/models' %self.xaap_dir,'tungurahua_rf_20211007144655.pkl'),'rb'))
        print(tungu_clf)
        categories = [' BRAM ', ' CRD ', ' EXP ', ' HB ', ' LH ', ' LP ', ' TRARM ', ' TREMI ', ' TRESP ', ' VT ']

        input_data = []

        logger.info("start feature calculation")
        for trace in self.triggers_traces:
            print(trace)
            file_code = "%s.%s.%s.%s.%s" %(trace.stats.network,trace.stats.station,trace.stats.location,trace.stats.channel,trace.stats.starttime.strftime("%Y.%m.%d.%H%M%S"))
            features.compute(trace.data,trace.stats.sampling_rate)
            row = np.append(file_code, features.featuresValues)
            input_data.append(row)

            column_names = ['data_code']
            data = pd.DataFrame(input_data)
            rows_length,column_length = data.shape

            for i in range(column_length - 1):
                column_names.append("f_%s" %i)

            data.columns = column_names
            data.columns = data.columns.str.replace(' ', '')

            x_no_scaled = data.iloc[:,1:].to_numpy()

            scaler = StandardScaler()
            x = scaler.fit_transform(x_no_scaled)
            data_scaled = pd.DataFrame(x,columns=data.columns[1:])

            print(data_scaled.shape)

            y_pred=tungu_clf.predict(data_scaled)

            print(type(y_pred))
            print(y_pred.shape)

            for i in range(rows_length):
                prediction = "%s,%s" %(data.iloc[i,0],categories[int(y_pred[i])])
                logger.info(prediction)
    def setupGUI(self):

        self.layout = QtGui.QVBoxLayout()
        self.layout.setContentsMargins(0,0,0,0)
        self.setLayout(self.layout)
        self.splitter = QtGui.QSplitter()
        self.splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.layout.addWidget(self.splitter)

        self.tree = ParameterTree(showHeader=False)
        self.splitter.addWidget(self.tree)

        self.splitter2 = QtGui.QSplitter()
        self.splitter2.setOrientation(QtCore.Qt.Orientation.Vertical)
        self.splitter.addWidget(self.splitter2)

        self.plot_window = pg.GraphicsLayoutWidget()
        self.plot_window.setWindowTitle("XAAP")
        self.splitter2.addWidget(self.plot_window)

        self.datetime_axis_1 = pg.graphicsItems.DateAxisItem.DateAxisItem(orientation = 'bottom')
        self.datetime_axis_2 = pg.graphicsItems.DateAxisItem.DateAxisItem(orientation = 'bottom')

        self.p1 = self.plot_window.addPlot(row=1, col=0,axisItems={'bottom': self.datetime_axis_1})
        self.p2 = self.plot_window.addPlot(row=2, col=0,axisItems={'bottom': self.datetime_axis_2})

        self.window_region = pg.LinearRegionItem()
        self.p2.addItem(self.window_region, ignoreBounds=True)        

    def save(self):
        filename = pg.QtGui.QFileDialog.getSaveFileName(self, "Save State..", "xaap_configuracion.cfg", "Config Files (*.cfg)")
        if isinstance(filename, tuple):
            filename = filename[0]  # Qt4/5 API difference
        if filename == '':
            return
        state = self.params.saveState()
        pg.configfile.writeConfigFile(state, str(filename)) 

    def loadPreset(self,param,preset):

        if preset == '':
            return
        fn = os.path.join(self.parameters_dir, preset+".cfg")
        state = pg.configfile.readConfigFile(fn)
        self.loadState(state)

    def loadState(self, state):
        if 'Load Preset..' in state['children']:
            del state['children']['Load Preset..']['limits']
            del state['children']['Load Preset..']['value']
        self.params.param('Parameters').clearChildren()
        self.params.restoreState(state, removeChildren=False)

        ##llamar a ploteo 
        #self.pre_process_stream()




    def set_p1_using_p2(self):
        
        minX,maxX = self.window_region.getRegion()
        self.p1.setXRange(minX,maxX,padding=0)


    def set_p2_using_p1(self,window,viewRange):

        rgn = viewRange[0]
        self.window_region.setRegion(rgn)





















if __name__ == '__main__':
    app = pg.mkQApp()
    app.setStyleSheet("""
    QWidget {font-size: 15px}
    QMenu {font-size: 10px}
    QMenu QWidget {font-size: 10px}
""")

    win = xaapGUI()
    win.setWindowTitle("xaap")
    win.show()
    win.resize(1100,700)

    pg.exec()