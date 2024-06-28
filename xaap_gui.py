# -*- coding: utf-8 -*-

"""
LAST CHANGE
2024.05.07. 
Add save  json and then  load it via terminal and delete old unused code 
TODO
Add the code to decide demean, merge and filter 
"""

import sys,os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from aaa_features.features import FeatureVector
from obspy import UTCDateTime

from obspy.signal import filter
from obspy import Stream, Trace
import pickle
from datetime import date, datetime
from sklearn.ensemble import RandomForestClassifier
from xaap.configuration import xaap_configuration
from seisbench.util.annotations import Pick as sb_pick
from seisbench.util.annotations import Detection as sb_detection
from sklearn.preprocessing import StandardScaler

from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.parametertree import Parameter, ParameterTree


from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QMenuBar, QMenu, QAction, QWidget
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QVBoxLayout

from PyQt5.QtWidgets import QSplitter
from PyQt5.QtCore import Qt
from pyqtgraph import  GraphicsLayoutWidget
from pyqtgraph.graphicsItems.DateAxisItem import DateAxisItem
from pyqtgraph import LinearRegionItem

from PyQt5.QtWidgets import QFileDialog

from xaap.configuration.xaap_configuration import (configure_logging, configure_parameters_from_gui)
from xaap.process import pre_process, request_data, detect_trigger, process_deep_learning




xaap_dir = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])))
xaap_config_dir = Path(xaap_dir,"config")

pd.set_option("display.max_rows", None, "display.max_columns", None)



class xaapGUI(QWidget):  
    """
    A class for creating and configuring the xaap GUI.
    """

    def save_parameters(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save State..", "xaap_configuracion.json", "JSON Files (*.json)")
        if not filename:  # Check if the Cancel button was clicked
            return
        if not filename.endswith('.json'):  # Ensure the file has the correct extension
            filename += '.json'
        state = self.params.saveState()
       
        if "dates" in state["children"]["parameters"]["children"] :
            logger.info("Removed dates from parameter")
            del state["children"]["parameters"]["children"] ["dates"]["children"] ["start"]
            del state["children"]["parameters"]["children"] ["dates"]["children"] ["end"]
        
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)




    def __init__(self, detect_datetime='default',xaap_gui_json='xaap_gui.json',*args):
        """
        Initializes the xaapGUI class.
        """
        self.detect_datetime = detect_datetime
        self.xaap_gui_json = xaap_gui_json

        logger.info("Start of all the process. Working directory %s" %xaap_dir)

        QWidget.__init__(self)

        self.setupGUI()       
        self.params = self.create_parameters()

        self.tree.setParameters(self.params, showTop=True)
        logger.info("Configure parameter signals")

        """GUI buttons linked with functions"""

        self.params.param("update_parameters").sigActivated.connect(lambda: self.gui_update_parameters())
        """Get the stream for the configured volcano"""        
        self.params.param("request_data").sigActivated.connect(lambda: self.gui_request_stream())
        """Preprocess: sort, merge and filter the stream for the configured volcano"""       
        self.params.param("pre_process").sigActivated.connect(lambda: self.gui_pre_process_stream())
        self.params.param('plot_stream').sigActivated.connect(self.plot_stream)
        self.window_region.sigRegionChanged.connect(self.set_p1_using_p2)
        self.p1.sigRangeChanged.connect(self.set_p2_using_p1)
        self.params.param('detection_sta_lta').sigActivated.connect(lambda: self.gui_detection_sta_lta( ))
        self.params.param('detection_deep_learning').sigActivated.connect(lambda: self.gui_detection_deep_learning( ))
        self.params.param('classify_triggers').sigActivated.connect(self.classify_triggers)


    def create_parameters(self):
        """
        This function creates a ParameterTree object to store the configuration settings for the XAAP application. The structure
        and default values for the parameters are read from a JSON file (xaap_gui.json). Additionally, the function sets the start
        and end datetime parameters based on the current time or a test date.

        Returns:
        --------
        xaap_parameters: Parameter
            A Parameter object representing the XAAP configuration settings, including the start and end datetime parameters.
        """

        # Read the xaap_gui.json file and create a Parameter object from its contents
        xaap_parameters = Parameter.create(name='xaap_configuration', type='group', children=[])
        
        with open(f'{xaap_config_dir}/{self.xaap_gui_json}', 'r') as f:
            json_data = f.read()
        xaap_parameters.restoreState(json.loads(json_data))
 
        # Set the start and end datetime parameters for testing or based on the current time


        if self.detect_datetime == "test":
            start_datetime = UTCDateTime("2023-08-17 13:00:00")
            end_datetime = UTCDateTime("2023-08-17  20:00:00")
            #start_datetime = UTCDateTime("2023-04-01 02:00:00")
            #end_datetime = UTCDateTime("2023-04-01 03:00:00")
            
        elif self.detect_datetime =="default":
        
            start_datetime = (UTCDateTime.now() - 3600*3).strftime("%Y-%m-%d %H:%M:%S")
            end_datetime = (UTCDateTime.now()).strftime("%Y-%m-%d %H:%M:%S")
        
        else:
            start_datetime = UTCDateTime(self.detect_datetime)
            end_datetime = start_datetime + 86400

        # Add the start and end datetime parameters to the xaap_parameters Parameter object
        xaap_parameters.child('parameters').child('dates').addChild({'name': 'start', 'type': 'str', 'value': f"{start_datetime}"})
        xaap_parameters.child('parameters').child('dates').addChild({'name': 'end', 'type': 'str', 'value': f"{end_datetime}"})

        # Save the state of the xaap_parameters object as a JSON string
        xaap_state = xaap_parameters.saveState()
        
        #Not used since this value is overwritten in gui_update_parameters. Delete if does not seem to affect
        ##self.json_xaap_state = json.dumps(xaap_state, indent=2)

        return xaap_parameters


    """FUNCTIONS LINKED TO THE GUI BUTTONS"""
    def gui_update_parameters(self):
        xaap_state = self.params.saveState()
        self.json_xaap_state = json.dumps(xaap_state, indent=2)
        """Create a configuration object using the gui parameters, then send it to the processing part"""
        logger.info("Call update parameters")
        self.xaap_config = configure_parameters_from_gui(self.json_xaap_state)

    def gui_request_stream(self):
        self.volcan_stream = request_data.request_stream(self.xaap_config)
        # do other processing as necessary
        print(self.volcan_stream)

    def gui_pre_process_stream(self):        
        self.volcan_stream = pre_process.pre_process_stream(self.xaap_config,self.volcan_stream)
        print(self.volcan_stream)

    def gui_detection_sta_lta(self):
        self.triggers = detect_trigger.get_triggers(self.xaap_config,self.volcan_stream)
        self.plot_triggers()



    def gui_detection_deep_learning(self):

        """
        Try to process each stream individually with the DL model
        Process a single stream to recover the channel
        The function should return a detection window with a P/S phase if available. 
        """
        try:
            logger.info("Try to create DL model")
            deep_learning_model = process_deep_learning.create_model(self.xaap_config)
            logger.info(f"SB model: {deep_learning_model.name} created")
        except Exception as e:
            logger.info(f"Error in creating DL model was:{e}")
            raise Exception(f"Failed to create deep learning model: {e}")
    
        try:
            logger.info("Continue processing. Run detection with deep learning")
            detections = []
            for station_stream in self.volcan_stream:
                
                detections.extend(process_deep_learning.get_detections(self.xaap_config,Stream(station_stream),deep_learning_model))
            
            if len(detections) > 0:
                
                logger.info(f"$$$$ End of  DL detection was: Individual detections {len(detections)}")
                for individual_detection in detections:
                    print(individual_detection)
            else:
                logger.info(f"No detections made")

        except Exception as e:
            logger.info(f"Error in processing. Error in detection  DL was:{e}")

        if len(detections) > 0:

            self.detections = detections

            try:
                coincidence_detections = process_deep_learning.coincidence_detection_deep_learning(self.xaap_config,detections)

                self.triggers = coincidence_detections
                self.plot_triggers()
                
                
                for coincidence_detection in coincidence_detections:
                    print(coincidence_detection)
                
                logger.info(f"$$$$ End of  DL detection. INDIVIDUAL DETECTIONS: {len(detections)}")
                logger.info(f"COINCIDENCE DETECTIONS: {len(self.triggers)}")


                for detection in self.detections:
                    print(detection)
                    if hasattr(detection.pick_detection,'phase'):
                        print(detection.pick_detection.phase)
                self.plot_picks()

            except Exception as e:
                print(f"Error in coincidence detection was: {e}")


    
    def plot_triggers(self):

        sta_lta_endtime_buffer = float(self.xaap_config.sta_lta_endtime_buffer)

        for trigger in self.triggers:
            for trace_id in trigger['trace_ids']:
                for plot_item in self.plot_items_list:
                    if plot_item.getAxis("left").labelText == trace_id:
                        trigger_trace_temp = self.volcan_stream.select(id=trace_id)
                        trigger_window = trigger_trace_temp.slice(trigger['time'],trigger['time']+trigger['duration']*sta_lta_endtime_buffer)
                        #plot_item.plot([trigger['time']],[0],pen=None,symbol='x')
                        plot_item.plot(trigger_window[0].times(type='timestamp'),trigger_window[0].data,pen='r')


    def plot_picks(self):


        for detection in self.detections:
            if hasattr(detection.pick_detection,'phase'):
                #print(detection.pick_detection.phase)


                pick = detection.pick_detection

                pick_time = UTCDateTime(pick.start_time)

                for  plot_item in self.plot_items_list:
                    if plot_item.getAxis("left").labelText.startswith(pick.trace_id):


                        if pick.phase == 'P':
                            label = pg.TextItem(text=pick.phase, color='r', anchor=(0.5, 0))
                            y_range = plot_item.viewRange()[1]
                            label_y_pos = y_range[1] - (y_range[1] - y_range[0]) * 0.1
                            label.setPos(pick_time.timestamp, label_y_pos)
                            vertical_line = pg.InfiniteLine(pos=pick_time.timestamp, angle=90, pen='cyan')

                        if pick.phase == 'S':
                            label = pg.TextItem(text=pick.phase, color='r', anchor=(0.5, 0))
                            y_range = plot_item.viewRange()[1]
                            label_y_pos = y_range[1] - (y_range[1] - y_range[0]) * 0.1
                            label.setPos(pick_time.timestamp, label_y_pos)
                            vertical_line = pg.InfiniteLine(pos=pick_time.timestamp, angle=90, pen='blue')

                        plot_item.addItem(vertical_line)
                        plot_item.addItem(label)




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

        if self.triggers:

            self.triggers_traces = detect_trigger.create_trigger_traces(self.xaap_config,self.volcan_stream,self.triggers)


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

            classified_triggers_file = Path(xaap_dir,"data/classifications") / UTCDateTime.now().strftime("out_xaap_%Y.%m.%d.%H.%M.%S.csv")
            classification_file = open(classified_triggers_file,'a+')
            #Como guardar categorias en  el modelo?
            #categories = [' BRAM ', ' CRD ', ' EXP ', ' HB ', ' LH ', ' LP ', ' TRARM ', ' TREMI ', ' TRESP ', ' VT ']
            categories = ['LP', 'VT']
            features = FeatureVector(feature_config, verbatim=2)
            input_data = []


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
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0,0,0,0)
        self.setLayout(self.layout)


        # Create a QWidget container for the menu bar
        self.menu_container = QWidget()
        self.menu_container.setFixedHeight(25)
        self.layout.addWidget(self.menu_container)

        # Create menu bar and set it as the layout for the menu container
        self.menu_bar = QMenuBar(self.menu_container)
        self.menu_container_layout = QVBoxLayout(self.menu_container)
        self.menu_container_layout.setContentsMargins(0, 0, 0, 0)
        self.menu_container_layout.addWidget(self.menu_bar)



        # Create menu items and actions
        self.menu = QMenu("Menu")
        self.save_action = QAction("Save Parameter Configuration", self)
        self.toggle_action = QAction("Hide/Show Parameter Tree", self)

        # Connect actions to functions
        self.save_action.triggered.connect(self.save_parameters)
        self.toggle_action.triggered.connect(self.toggle_parameter_tree)

        # Add actions to the menu
        self.menu.addAction(self.save_action)
        self.menu.addAction(self.toggle_action)

        # Add menu to the menu bar
        self.menu_bar.addMenu(self.menu)

        self.splitter = QSplitter()
        self.splitter.setOrientation(Qt.Orientation.Horizontal)
        self.layout.addWidget(self.splitter)

        self.tree = ParameterTree(showHeader=False)
        self.splitter.addWidget(self.tree)

        self.splitter2 = QSplitter()
        self.splitter2.setOrientation(Qt.Orientation.Vertical)
        self.splitter.addWidget(self.splitter2)

        self.plot_window = GraphicsLayoutWidget()
        self.plot_window.setWindowTitle("XAAP")
        self.splitter2.addWidget(self.plot_window)

        self.datetime_axis_1 = DateAxisItem(orientation='bottom', utcOffset=5)
        self.datetime_axis_2 = DateAxisItem(orientation='bottom', utcOffset=5)

        self.p1 = self.plot_window.addPlot(row=1, col=0, axisItems={'bottom': self.datetime_axis_1})
        self.p2 = self.plot_window.addPlot(row=2, col=0, axisItems={'bottom': self.datetime_axis_2})

        self.window_region = LinearRegionItem()
        self.p2.addItem(self.window_region, ignoreBounds=True)




    # Add the toggle_parameter_tree function to hide/show the Parameter Tree when the button is clicked
    def toggle_parameter_tree(self):
        if self.tree.isVisible():
            self.tree.hide()
        else:
            self.tree.show()


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

    '''Call the program with arguments or use default values'''
    parser = argparse.ArgumentParser(description='XAAP_GUI will use default configuration found in ./config/xaap_gui.json')
    parser.add_argument('--detect_datetime', type=str, default="default" ,help='datetime to detect ')
    parser.add_argument('--xaap_gui_json', type=str, default="xaap_gui.json", help='JSON config file for XAAP_GUI')

    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)

    # Ahora args.time y args.config contienen los valores proporcionados por el usuario
    logger.info(f'detect_datetime {args.detect_datetime}')
    logger.info(f'xaap_gui_json: {args.xaap_gui_json}')


    app = pg.mkQApp()
    app.setStyleSheet("""
    QWidget {font-size: 15px}
    QMenu {font-size: 10px}
    QMenu QWidget {font-size: 10px}
                        """)

    win = xaapGUI(detect_datetime=args.detect_datetime, xaap_gui_json=args.xaap_gui_json )
    win.setWindowTitle("xaap")
    win.show()
    win.resize(1600,900)

    pg.exec()
