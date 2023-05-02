# -*- coding: utf-8 -*-

import json
import logging
import os
import pickle
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pyqtgraph.configfile
from aaa_features.features import FeatureVector
from obspy import Stream, Trace, UTCDateTime, read
from obspy.signal import filter
from obspy.signal.trigger import (classic_sta_lta, classic_sta_lta_py,
                                  coincidence_trigger, plot_trigger,
                                  recursive_sta_lta_py, trigger_onset)
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.Qt import QtCore, QtGui
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xaap.configuration import xaap_configuration

from xaap.configuration.xaap_configuration import (
    configure_logging, configure_parameters_from_gui)
from xaap.process import pre_process, request_data, detect_trigger

xaap_dir = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])))
#xaap_config_dir = Path("%s/%s" %(xaap_dir,"config"))
xaap_config_dir = Path(xaap_dir,"config")





pd.set_option("display.max_rows", None, "display.max_columns", None)
class xaapGUI(QtGui.QWidget):  
    """
    A class for creating and configuring the xaap GUI.
    """

    def __init__(self):
        """
        Initializes the xaapGUI class.
        """

        logger.info("Start of all the process. Working directory %s" %xaap_dir)

        QtGui.QWidget.__init__(self)

        self.setupGUI()       
        #self.xaap_dir = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])))       
        self.params = self.create_parameters()


        self.parameters_dir = Path(xaap_dir, 'config','xaap_parameters')
        if os.path.exists(self.parameters_dir):
            presets = [os.path.splitext(p)[0] for p in os.listdir(self.parameters_dir)]
            self.params.param('load_preset..').setLimits(['']+presets)
        self.tree.setParameters(self.params, showTop=True)
        logger.info("Configure parameter signals")

        '''loadPreset is called with parameters: param and preset '''
        self.params.param('load_preset..').sigValueChanged.connect(self.loadPreset)
        self.params.param('save').sigActivated.connect(self.save)
        

        """Create a configuration object using the gui parameters, then send it to the processing part"""
        self.xaap_config = configure_parameters_from_gui(self.json_xaap_state)

        """Get the stream for the configured volcano"""        
        self.params.param("request_data").sigActivated.connect(lambda: self.gui_request_stream())


        """Preprocess: sort, merge and filter the stream for the configured volcano"""       
        self.params.param("pre_process").sigActivated.connect(lambda: self.gui_pre_process_stream())

        self.params.param('plot_stream').sigActivated.connect(self.plot_stream)
        self.window_region.sigRegionChanged.connect(self.set_p1_using_p2)
        self.p1.sigRangeChanged.connect(self.set_p2_using_p1)


        ###self.params.param('detect_triggers').sigActivated.connect(self.detect_triggers)

        self.params.param('detect_triggers').sigActivated.connect(lambda: self.gui_detect_triggers( ))
        self.params.param('classify_triggers').sigActivated.connect(self.classify_triggers)



    def gui_request_stream(self):
        self.volcan_stream = request_data.request_stream(self.xaap_config)
        # do other processing as necessary
        print(self.volcan_stream)

    def gui_pre_process_stream(self):
        
        self.volcan_stream = pre_process.pre_process_stream(self.xaap_config,self.volcan_stream)

        print(self.volcan_stream)


    def gui_detect_triggers(self):

        self.triggers = detect_trigger.get_triggers(self.xaap_config,self.volcan_stream)

        self.plot_triggers()


    
    def plot_triggers(self):
        print(xaap_configuration)
        sta_lta_endtime_buffer = float(self.xaap_config.sta_lta_endtime_buffer)

        for trigger in self.triggers:
            for trace_id in trigger['trace_ids']:
                for plot_item in self.plot_items_list:
                    if plot_item.getAxis("left").labelText == trace_id:
                        trigger_trace_temp=self.volcan_stream.select(id=trace_id)
                        trigger_window = trigger_trace_temp.slice(trigger['time'],trigger['time']+trigger['duration']*sta_lta_endtime_buffer)
                        #plot_item.plot([trigger['time']],[0],pen=None,symbol='x')
                        plot_item.plot(trigger_window[0].times(type='timestamp'),trigger_window[0].data,pen='r')

            


    def create_parameters(self):


        xaap_parameters = Parameter.create(name='xaap_configuration',type='group',children=[])
        with open('./config/xaap_gui.json', 'r') as f:
            json_data = f.read()
        xaap_parameters.restoreState(json.loads(json_data))

        TEST_DATE = True 

        if TEST_DATE:
            start_datetime = UTCDateTime("2022-08-30 16:00:00")
            end_datetime = UTCDateTime("2022-08-30 20:00:00")
        else:
            start_datetime = (UTCDateTime.now() - 3600).strftime("%Y-%m-%d %H:%M:%S")
            end_datetime = (UTCDateTime.now()).strftime("%Y-%m-%d %H:%M:%S")

        xaap_parameters.child('parameters').child('dates').addChild({'name':'start','type':'str','value':f"{start_datetime}"})
        xaap_parameters.child('parameters').child('dates').addChild({'name':'end','type':'str','value':f"{end_datetime}"})

        xaap_state = xaap_parameters.saveState()

        self.json_xaap_state = json.dumps(xaap_state, indent=2)
        return xaap_parameters


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
                view_box_i = plot_i.getViewBox()
                view_box_i.setMouseMode(pg.ViewBox.RectMode)
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



    def classify_triggers(self):

        ##leer esto desde la configuracion
        ## leer solo las mejores caracteristicas?
        ####TODO: CAMBIAR A CONFIGRACION 
        feature_config = {"features_file":"%s/config/features/features_00.json" %xaap_dir,
                    #"domains":"time spectral cepstral"}
                    "domains":"spectral cepstral"}
        #tungu_clf= pickle.load(open(os.path.join('%s/data/models' %xaap_dir,'tungurahua_rf_20211007144655.pkl'),'rb'))
        #chiles_rf_20230410092541
        #volcano_classifier_model = pickle.load(open(os.path.join('%s/data/models' %xaap_dir,'chiles_rf_20220902115108.pkl'),'rb'))
        volcano_classifier_model = pickle.load(open(os.path.join('%s/data/models' %xaap_dir,'chiles_rf_20230410092541.pkl'),'rb'))

        classified_triggers_file = Path(xaap_dir,"data/classifications") / UTCDateTime.now().strftime("out_xaap_%Y.%m.%d.%H.%M.%S.txt")
        classification_file = open(classified_triggers_file,'a+')
        #Como guardar categorias en  el modelo?
        #categories = [' BRAM ', ' CRD ', ' EXP ', ' HB ', ' LH ', ' LP ', ' TRARM ', ' TREMI ', ' TRESP ', ' VT ']
        categories = ['LP', 'VT']
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

        y_pred=volcano_classifier_model.predict(data_scaled)

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

    print("call logging configuration")
    logger = configure_logging()
    logger.info("Logging configurated")



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
