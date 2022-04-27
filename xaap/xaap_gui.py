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
from obspy.signal.trigger import coincidence_trigger
from obspy import read, UTCDateTime
from obspy.signal.trigger import classic_sta_lta
from obspy.signal.trigger import classic_sta_lta, classic_sta_lta_py, recursive_sta_lta_py, plot_trigger, trigger_onset
from datetime import date, datetime

import numpy as np
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

        logger.info("Start of all the process. Working directory %s" %xaap_dir)

        QtGui.QWidget.__init__(self)

        self.setupGUI()       
        #self.xaap_dir = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])))       
        self.params = self.create_parameters()
        self.parameters_dir = Path(xaap_dir, 'config/xaap_parameters')
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

        if self.volcan_stream == None:
            logger.info("No data available")
        elif len(self.volcan_stream[0].data)==0:
            logger.info("No data available")
        else:

            logger.info("ok request_stream() in __init__. Continue with Pre-process")
            self.pre_process_stream()
            self.plot_stream()
            self.detect_triggers()
            #self.classify_triggers()
            #self.window_region.sigRegionChanged.connect(self.set_p1_using_p2)
            





    def create_parameters(self):

        start_datetime = (UTCDateTime.now() - 3600).strftime("%Y-%m-%d %H:%M:%S")
        end_datetime = (UTCDateTime.now()).strftime("%Y-%m-%d %H:%M:%S")
        xaap_parameters = Parameter.create(
            
            name='xaap configuration',type='group',children=[
            {'name':'Load Preset..','type':'list','limits':[]},
            {'name':'Save','type':'action'},
            {'name':'Load','type':'action'},
            {'name':'Parameters','type':'group','children':[

                {'name':'MSEED','type':'group','children':[
                    {'name':'client_id','type':'list','values':['FDSN','SEEDLINK','ARCHIVE','ARCLINK']},
                    {'name':'server_config_file','type':'str','value':'%s' %('server_configuration.json')}
                                                                 ]},
                {'name':'Volcan configuration', 'type':'group','children':[
                    {'name':'volcanoes_config_file','type':'str','value':'%s' %('volcanoes.json')},
                    {'name':'stations_config_file','type':'str','value':'%s' %('stations.json')},
                    {'name':'volcan_name','type':'str','value':'TUNGURAHUA' }
                                                                            ]},

                {'name':'Dates','type':'group','children':[
                    {'name':'start','type':'str','value':'%s' %start_datetime },
                    {'name':'end','type':'str','value':'%s' %end_datetime }
                    #{'name':'start','type':'str','value':'%s' %(UTCDateTime("2021-11-02 11:20:00")) },
                    #{'name':'end','type':'str','value':'%s' %(UTCDateTime("2021-11-02 11:30:00")) }
                                                            ]},
                                                          
                {'name':'Filter','type':'group','children':[
                    {'name':'Filter type','type':'list','values':['highpass','bandpass','lowpass']},
                    {'name':'Freq_A','type':'float','value':0.5,'step':0.1,'limits': [0.1, None ]},
                    {'name':'Freq_B','type':'float','value':1.0,'step':0.1,'limits': [0.1, None ]}    ]},

                {'name':'STA_LTA','type':'group', 'children': [
                    {'name':'sta','type':'float','value':0.5,'step':0.5,'limits': [0.1, None ]},                    
                    {'name':'lta','type':'float','value':10,'step':0.5,'limits': [1, None ]},
                    {'name':'trigon','type':'float','value':3.5,'step':0.1,'limits': [0.1, None ]},
                    {'name':'trigoff','type':'float','value':1.0,'step':0.1,'limits': [0.1, None ]},
                    {'name':'coincidence','type':'float','value':3.0,'step':0.5,'limits': [1, None ]},
                    {'name':'endtime_extra','type':'float','value':2.5,'step':0.5,'limits': [1, None ]}
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
            mseed_server_config_file = xaap_config_dir / self.params['Parameters','MSEED','server_config_file']
            mseed_server_param = gmutils.read_config_file(mseed_server_config_file)
            self.mseed_client = get_mseed.choose_service(mseed_server_param[mseed_client_id])

        except Exception as e:
            raise Exception("Error connecting to MSEED server : %s" %str(e))
        
        try:
            volcanoes_config_file = xaap_config_dir / self.params['Parameters', 'Volcan configuration','volcanoes_config_file']
            volcanoes_stations = gmutils.read_config_file(volcanoes_config_file)

        except Exception as e:
            raise Exception("Error reading volcano config file : %s" %str(e))

        try:
            stations_config_file = xaap_config_dir / self.params['Parameters', 'Volcan configuration','stations_config_file']
            stations_param = gmutils.read_config_file(stations_config_file)

        except Exception as e:
            raise Exception("Error reading volcano config file : %s" %str(e))

        

        try:
            mseed_client_id = self.params['Parameters','MSEED','client_id']
            start_time = UTCDateTime(self.params['Parameters','Dates', 'start'])
            end_time = UTCDateTime(self.params['Parameters','Dates', 'end'])
            volcan_name = self.params['Parameters', 'Volcan configuration','volcan_name']
            volcan_stations = volcanoes_stations[volcan_name][volcan_name]
            self.volcan_stations_list = []
            
            for temp_station in volcan_stations:
                self.volcan_stations_list.append(stations_param[temp_station])

            
            st = Stream()

            for st_ in self.volcan_stations_list:
                for loc in st_['loc']:
                    if not loc:
                        loc = ''
                    for cha in st_['cha']:
                        stream_id = "%s.%s.%s.%s.%s.%s" %(st_['net'],st_['cod'],st_['loc'][0],cha,start_time,end_time)
                        mseed_stream=get_mseed.get_stream(mseed_client_id,self.mseed_client,st_['net'],st_['cod'],loc,cha,start_time=start_time,end_time=end_time)
                        print("####")
                        print(stream_id)
                        if mseed_stream:
                            mseed_stream.merge(method=1, fill_value="interpolate",interpolation_samples=0)
                            st+=mseed_stream
                        else:
                            logger.info("no stream: %s" %stream_id)
            ##COLOCAR EN PREPROCESS Y HABILITAR MULTIPLES FILTROS
            st.filter('highpass', freq=0.5)  # optional prefiltering
            #st.filter('bandpass', freqmin=10, freqmax=20)  # optional prefiltering
            self.volcan_stream = st.copy()

            
        except Exception as e:
            raise Exception("Error reading parameters was: %s" %str(e))



    def pre_process_stream(self):

        self.processed_stream = None
        logger.info("self.processed_stream cleaned :%s" %self.processed_stream)
        try:
            #self.processed_stream = self.stream.copy()
            self.processed_stream = self.volcan_stream.copy()
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

            #self.processed_stream[0].data = temp_data
            self.volcan_stream[0].data = temp_data

        except Exception as e:
            logger.error("Error at pre_process_stream() was : %s" %str(e))


    def plot_stream(self):

        logger.info("start plot_stream(). Clean plots")
        self.p1.clearPlots()
        self.p2.clearPlots()


        try:
            self.plot_window.clear()
            self.plot_items_list=[]
            n_plots = len(self.volcan_stream)
            for i in range(n_plots):
                datetime_axis_i = pg.graphicsItems.DateAxisItem.DateAxisItem(orientation = 'bottom',utcOffset=5)
                plot_i = self.plot_window.addPlot(row=(3+i),col=0,axisItems={'bottom': datetime_axis_i})
                self.plot_items_list.append(plot_i)

            for i,tr_t in enumerate(self.volcan_stream):
                self.plot_items_list[i].clearPlots()
                self.plot_items_list[i].plot(tr_t.times(type="timestamp"),tr_t.data,pen='g')
                self.plot_items_list[i].getAxis("left").setWidth(100)
                self.plot_items_list[i].setLabel("left",text="%s.%s.%s.%s" %(tr_t.stats['network'],tr_t.stats['station'],tr_t.stats['location'],tr_t.stats['channel']))

            vbox_0=self.plot_items_list[0]
            for plot_item in  self.plot_items_list:
                plot_item.setXLink(vbox_0)
                
        except Exception as e:
            logger.error("Error in multiple plots was: %s" %str(e))


    def detect_triggers(self):

        sta = self.params['Parameters','STA_LTA','sta']
        lta = self.params['Parameters','STA_LTA','lta']
        trigon = self.params['Parameters','STA_LTA','trigon']
        trigoff = self.params['Parameters','STA_LTA','trigoff']
        coincidence = self.params['Parameters','STA_LTA','coincidence']
        endtime_extra = self.params['Parameters','STA_LTA','endtime_extra']
        self.triggers_traces = []

        self.triggers = coincidence_trigger("recstalta", trigon, trigoff, self.volcan_stream, coincidence, sta=sta, lta=lta)
        for i,trg in enumerate(self.triggers):
            logger.info("%s:%s.%s %s" %(i,trg['time'],trg['trace_ids'][0],trg['duration']))
        
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



    def classify_triggers(self):

        ##leer esto desde la configuracion
        ## leer solo las mejores caracteristicas?
        feature_config = {"features_file":"%s/config/features/features_00.json" %xaap_dir,
                    "domains":"time spectral cepstral"}
        tungu_clf= pickle.load(open(os.path.join('%s/data/models' %xaap_dir,'tungurahua_rf_20211007144655.pkl'),'rb'))
        classified_triggers_file = Path(xaap_dir,"data/classifications") / UTCDateTime.now().strftime("out_xaap_%Y.%m.%d.%H.%M.%S.txt")
        classification_file = open(classified_triggers_file,'a+')
        #Como guardar categorias en  el modelo?
        categories = [' BRAM ', ' CRD ', ' EXP ', ' HB ', ' LH ', ' LP ', ' TRARM ', ' TREMI ', ' TRESP ', ' VT ']

        features = FeatureVector(feature_config, verbatim=2)
        input_data = []

        trigger_dir = Path(xaap_dir,"trigger")

        logger.info("start feature calculation")
        for trace in self.triggers_traces:
            print("!####")
            print(trace)
            ##Modificar file code para que incluya la ventana de end_time
            trace_window = int(trace.stats.endtime - trace.stats.starttime)
            file_code = "%s.%s.%s.%s.%s.%s" %(trace.stats.network,trace.stats.station,trace.stats.location,trace.stats.channel,trace.stats.starttime.strftime("%Y.%m.%d.%H.%M.%S"),trace_window)
            features.compute(trace.data,trace.stats.sampling_rate)
            row = np.append(file_code, features.featuresValues)
            input_data.append(row)

            '''
            try:
                trace.write("%s/%s.mseed" %(trigger_dir,file_code),format="MSEED")

            except Exception as e:
                print("error in write")
            
            '''

        '''Create pandas data frame from features vectors'''
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
            prediction = "%s,%s\n" %(data.iloc[i,0],categories[int(y_pred[i])])
            logger.info(prediction)
            classification_file.write(prediction)
    
    
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

        self.datetime_axis_1 = pg.graphicsItems.DateAxisItem.DateAxisItem(orientation = 'bottom',utcOffset=5)
        self.datetime_axis_2 = pg.graphicsItems.DateAxisItem.DateAxisItem(orientation = 'bottom',utcOffset=5)

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


##crear nueva ventana que cargue el archivo de texto en tablas y plotee los triggers sobre la señal
##con etiquetas y con ventana 


















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