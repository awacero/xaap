# -*- coding: utf-8 -*-

import os,sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.parametertree import Parameter, ParameterTree
import pyqtgraph.configfile
from obspy.signal import filter  
import numpy as np

from obspy import read, UTCDateTime
from obspy.signal.trigger import classic_sta_lta
from obspy.signal.trigger import classic_sta_lta, classic_sta_lta_py, recursive_sta_lta_py, plot_trigger, trigger_onset
from datetime import date, datetime

import logging, logging.config

xaap_dir = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])))

logging.config.fileConfig('%s/config/logging.ini' %xaap_dir ,disable_existing_loggers=False)
logger = logging.getLogger('stdout')
logger.setLevel(logging.INFO)

class xaapGUI(QtGui.QWidget):  

    def __init__(self):

        logger.info("Start of all this @#$. Working directory %s" %xaap_dir)

        QtGui.QWidget.__init__(self)

        self.setupGUI()
        self.region = pg.LinearRegionItem()


        self.xaap_dir = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])))
        '''
        Load parameters
        ##Create the configurations in object?
        ##cargar aqui la estacion y fecha a leer por defecto, 
        ##elegir entre el ultimo dia y una lista de estaciones
        '''
        
        self.params = Parameter.create(
            
            name='xaap configuration',type='group',children=[
            {'name':'Load Preset..','type':'list','limits':[]},
            {'name':'Save','type':'action'},
            {'name':'Load','type':'action'},
            {'name':'Parameters','type':'group','children':[
                {'name':'GUI','type':'group','children':[
                    {'name':'zoom_region_size','type':'float','value':0.10,'step':0.05,'limits':[0.01,1] }
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
                                                          ]},
            {'name':'Reprocess','type':'action'}              
                                                                         ]
                                                                         
                                                                         )


        def create_parameters(self):
            pass


        logger.info("End of set parameters") 

        self.tree.setParameters(self.params, showTop=True)
        logger.info("Configure parameter signals")
        self.params.param('Load Preset..').sigValueChanged.connect(self.loadPreset)
        self.params.param('Save').sigActivated.connect(self.save)
        self.params.param('Reprocess').sigActivated.connect(self.pre_process_stream)

        logger.info('''Preprocess a default stream to get initial values,otherwise some functions won't work''')
        self.pre_process_stream()

       
        
        ## read list of preset configs
        parameters_dir = os.path.join(self.xaap_dir, 'config/xaap_parameters')
        if os.path.exists(parameters_dir):
            presets = [os.path.splitext(p)[0] for p in os.listdir(parameters_dir)]
            self.params.param('Load Preset..').setLimits(['']+presets)

        ''' Configure region  '''

        logger.info("Create region")
        self.p2.addItem(self.region, ignoreBounds=True)        
        #self.region.setClipItem(self.p2d)

        self.region.sigRegionChanged.connect(self.update_region_size)





        ##Hacer que cambie la region al cambiar el valor maximo
        min_region=0
        max_region=int(len(self.times)*0.1)
        self.region.setRegion([self.times[min_region], self.times[max_region]])
        
        t_min,t_max= self.region.getRegion()
        t_max = datetime.utcfromtimestamp(t_max)
        t_min = datetime.utcfromtimestamp(t_min)
        logger.info("__init__ region %s,%s" %(t_min,t_max))
 
        #self.p1.sigRangeChanged.connect(self.update_region)



    def save(self):
        filename = pg.QtGui.QFileDialog.getSaveFileName(self, "Save State..", "xaap_configuracion.cfg", "Config Files (*.cfg)")
        if isinstance(filename, tuple):
            filename = filename[0]  # Qt4/5 API difference
        if filename == '':
            return
        state = self.params.saveState()
        pg.configfile.writeConfigFile(state, str(filename)) 


    def loadPreset(self, param, preset):
        if preset == '':
            return
        path = os.path.abspath(os.path.dirname(__file__))
        fn = os.path.join(path, 'config', preset+".cfg")
        state = pg.configfile.readConfigFile(fn)
        self.loadState(state)

    def loadState(self, state):
        if 'Load Preset..' in state['children']:
            del state['children']['Load Preset..']['limits']
            del state['children']['Load Preset..']['value']
        self.params.param('Parameters').clearChildren()
        self.params.restoreState(state, removeChildren=False)

        ##llamar a ploteo 
        self.pre_process_stream()


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


    def read_stream(self):
        
        '''Return a obspy stream object'''
        ##Llamar a un servidor MSEED
        ##retornar un stream valido, si no es posible salir y pedir cambios de parámetros 

        data_file="/home/wacero/proyectos_codigo/xaap/xaap/data/EC.RETU..SHZ.D.2012.187"
        self.stream = read(data_file)     
        logger.info("End of read_stream: %s" %self.stream)

    def pre_process_stream(self):

        logger.info("Start pre_process_stream(): region is %s, %s" %self.region.getRegion())

        self.read_stream()
        self.stream.merge(method=1, fill_value="interpolate",interpolation_samples=0)
        logger.info("Stream merged: %s" %self.stream)

        logger.info("Set self.times")
        self.times = self.stream[0].times(type="timestamp")

        sampling_rate = self.stream[0].stats.sampling_rate

        f_a = self.params['Parameters','Filter','Freq_A']
        f_b = self.params['Parameters','Filter','Freq_B']

        filter_type = self.params['Parameters','Filter','Filter type']
        if filter_type == 'highpass':
            logger.info("highpass selected")
            temp_data = filter.highpass(self.stream[0].data,f_a,sampling_rate,4)
        elif filter_type == 'bandpass':
            logger.info("bandpass selected")
            temp_data = filter.bandpass(self.stream[0].data,f_a,f_b,sampling_rate,4)
        elif filter_type == 'lowpass':
            logger.info("lowpass selected")
            temp_data = filter.lowpass(self.stream[0].data,f_a,sampling_rate,4)

        
        self.stream[0].data = temp_data
        
        self.plot_stream()
        self.detect_event()


    def plot_stream(self):

        #self.p1.clearPlots()
        #self.p2.clearPlots()

        
        ##Hacer que cambie la region al cambiar el valor maximo
        min_region=0
        max_region=int(len(self.times)*self.params['Parameters','GUI','zoom_region_size'])
        self.region.setRegion([self.times[min_region], self.times[max_region]])
        logger.info("region int values %s" %max_region)
        logger.info("Plot in p1")
        self.p1.plot(self.times,self.stream[0].data,pen='g')
        logger.info("Plot in p2")
        self.p2d = self.p2.plot(self.times,self.stream[0].data,pen="w")
        logger.info("Set clip item")
        #self.region.setClipItem(self.p2d)
        

        logger.info("End plot_stream()")


    def update_region_size(self):
        
        minX,maxX = self.region.getRegion()
        
        t_min,t_max= self.region.getRegion()
        t_max = datetime.utcfromtimestamp(t_max)
        t_min = datetime.utcfromtimestamp(t_min)
        logger.info("update_region_size() %s,%s" %(t_min,t_max))

        self.p1.setXRange(minX,maxX,padding=0)

    def update_region(self,window, viewRange):
        rgn = viewRange[0]
        self.region.setRegion(rgn)

        t_min,t_max= self.region.getRegion()
        t_max = datetime.utcfromtimestamp(t_max)
        t_min = datetime.utcfromtimestamp(t_min)
        logger.info("update_region() %s,%s" %(t_min,t_max))

    def detect_event(self):

        sta = self.params['Parameters','STA_LTA','sta']
        lta = self.params['Parameters','STA_LTA','lta']
        trigon = self.params['Parameters','STA_LTA','trigon']
        trigoff = self.params['Parameters','STA_LTA','trigoff']



        triggers_on = []
        scan_window_size = 600
        start_pad = 30
        end_pad = 30
        sampling_rate = self.stream[0].stats.sampling_rate
        start_time = self.stream[0].stats.starttime
        end_time = self.stream[0].stats.endtime

        n_windows = int(np.ceil(end_time - start_time)/scan_window_size) 

        triggers_on = []
        trigger_dot_list=[]
        for i in range(n_windows):
            window_start = start_time + scan_window_size*i - start_pad
            window_end = start_time + scan_window_size*(i + 1) + start_pad
            
            temp_trace = self.stream[0].slice(window_start,window_end)

            #print(temp_trace)

            cft = classic_sta_lta(temp_trace.data, int(sta * sampling_rate), int(lta * sampling_rate))

            on_off=trigger_onset(cft, trigon, trigoff)

            if len(on_off) != 0:
                for time in on_off:
                    #print(time)
                    time_on = int(time[0]/sampling_rate)
                    time_off = int(time[1]/sampling_rate)

                    triggers_on.append((window_start + time_on).timestamp)
                    trigger_dot_list.append(0)
        
        #print(triggers_on,trigger_dot_list)
        self.p1.plot(triggers_on,trigger_dot_list,pen=None, symbol='x')
        self.p2.plot(triggers_on,trigger_dot_list,pen=None, symbol='x')




















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