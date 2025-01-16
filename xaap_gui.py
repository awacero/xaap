# -*- coding: utf-8 -*-

"""
LAST CHANGE
2024.05.07. 
Add save json and then  load it via terminal and delete old unused code 
2025.01.14
Improve code, add docs, create UML diagrams
TODO
Add the code to decide demean, merge and filter 
"""

import sys,os
import json
import argparse
from pathlib import Path

import numpy as np ##import the whole library just for append?? 
import pandas as pd
import pyqtgraph as pg
from aaa_features.features import FeatureVector
from obspy import UTCDateTime
from obspy import Stream

import pickle
from sklearn.preprocessing import StandardScaler
from pyqtgraph.parametertree import Parameter, ParameterTree


from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QMenuBar, QMenu, QAction, QWidget
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtGui import QFont 
from PyQt5.QtWidgets import QLabel 
from PyQt5.QtWidgets import QVBoxLayout, QTextEdit, QWidget
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
        """
        Save the application state to a JSON file after optionally modifying the state.

        - Prompts the user to choose a file using QFileDialog.
        - Ensures the file has a .json extension.
        - Removes specific date parameters if they exist in the state.

        Returns:
            None
        """
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save State..", "xaap_configuracion.json", "JSON Files (*.json)")
        if not filename:  # Check if the Cancel button was clicked
            return
        if not filename.endswith('.json'):  # Ensure the file has the correct extension
            filename += '.json'

        try:
            state = self.params.saveState()
        
            # Safely modify the state dictionary
            dates_node = state.get("children", 
                                   {}).get("parameters", {}).get("children", {}).get("dates", {}).get("children", {})
            if dates_node:
                logger.info("Removing 'start' and 'end' dates from parameters...")
                dates_node.pop("start", None)  # Safely remove if it exists
                dates_node.pop("end", None)

            # Write the modified state to a file
            with open(filename, 'w') as f:
                json.dump(state, f, indent=2)
                logger.info(f"State successfully saved to {filename}")


        except Exception as e:
            logger.error(f"Failed to save parameters: {e}", exc_info=True)





    def __init__(self, args,detect_datetime='default',xaap_gui_json='xaap_gui.json'):
        """
        Initializes the xaapGUI class and sets up the graphical user interface (GUI).
        
        Args:
            detect_datetime (str): Default datetime for detection. Defaults to 'default'.
            xaap_gui_json (str): Path to the JSON file for GUI configuration. Defaults to 'xaap_gui.json'.
            *args: Additional arguments for customization.
        """
        super().__init__()  # Use `super()` for proper initialization
        ##QWidget.__init__(self)

        # Initialize instance variables
        self.detect_datetime = detect_datetime
        self.xaap_gui_json = xaap_gui_json

        if args.start_datetime:
            self.start_datetime = args.start_datetime
        if args.end_datetime:
            self.end_datetime = args.end_datetime

        logger.info(f"Start of all the process. Working directory {xaap_dir} " )

        # Configure parameter tree
        self.setupGUI()       
        self.params = self.create_parameters()

        self.tree.setParameters(self.params, showTop=True)
        logger.info("Parameter signals configured.")


        # Link GUI buttons to functions
        self._configure_gui_signals()

        # Link region changes and other signals
        self._configure_region_signals()
        
        """GUI buttons change colors"""
        ##self.params.param("update_parameters").clicked.connect(lambda: self.change_button_color(self.params.param("update_parameters")))


    def _configure_gui_signals(self):

        """GUI buttons linked with functions"""

        self.params.param("update_parameters").sigActivated.connect(lambda: self.gui_update_parameters())
        self.params.param("update_parameters").sigActivated.connect(lambda: self.log_change("UPDATE_PARAMETERS: Parameters updated. Continue"))
       
        """Get the stream for the configured volcano"""        
        self.params.param("request_data").sigActivated.connect(lambda: self.gui_request_stream())
        self.params.param("request_data").sigActivated.connect(lambda: self.log_change("REQUEST_DATA: Waveforms requested. Continue"))

        """Preprocess: sort, merge and filter the stream for the configured volcano"""       
        self.params.param("pre_process").sigActivated.connect(lambda: self.gui_pre_process_stream())
        self.params.param("pre_process").sigActivated.connect(lambda: self.log_change("PRE_PROCESS: Waveforms Preprocessed. Continue"))

        self.params.param('plot_stream').sigActivated.connect(self.plot_stream)
        self.params.param("plot_stream").sigActivated.connect(lambda: self.log_change("PLOT_STREAM: Waveforms Ploted. Continue"))

        self.params.param('detection_sta_lta').sigActivated.connect(lambda: self.gui_detection_sta_lta( ))
        self.params.param("detection_sta_lta").sigActivated.connect(lambda: self.log_change("DETECTION_STA_LTA:  STA/LTA finished. Continue"))
 
        self.params.param('detection_deep_learning').sigActivated.connect(lambda: self.gui_detection_deep_learning( ))
        self.params.param("detection_deep_learning").sigActivated.connect(lambda: self.log_change("DETECTION_DEEP_LEARNING:  Deep learning model finished. Continue"))

        self.params.param('classify_triggers').sigActivated.connect(self.classify_triggers)


    def _configure_region_signals(self):
        """Links region change signals to their handlers."""
        self.window_region.sigRegionChanged.connect(self.set_p1_using_p2)
        self.p1.sigRangeChanged.connect(self.set_p2_using_p1)


    def log_change(self,message):
        """
        Logs a formatted message to the GUI log.

        Args:
            message (str): The message to log. Uppercase words will be bolded.
        """
        formatted_message = self.format_bold_uppercase(message)
        self.log.append(f"{formatted_message}")

    def format_bold_uppercase(self, text):
        """
        Formats uppercase words in the input text to bold lowercase.

        Args:
            text (str): The input text to format.

        Returns:
            str: The formatted text with uppercase words bolded and lowercased.
        """
        # Reemplazar texto en mayúsculas con su versión en negritas
        words = text.split()
        formatted_words = [
            f"<b>{word.lower()}</b>" if word.isupper() else word
            for word in words
        ]
        return ' '.join(formatted_words)

    ##Creado Jorge Perez 
    def modify_parameter_style(self,param):
        """
        Modifies the style of a parameter's associated widget.

        - Sets the font to bold.
        - Changes text color to red.
        - Adds a tooltip indicating recent modification.

        Args:
            param: The parameter object. It should provide a `get_widget` method to access its widget.
        """
        try:
            widget = param.get_widget()  # Ensure this method exists
            if isinstance(widget, QLabel):
                font = QFont()
                font.setBold(True)
                widget.setFont(font)
                widget.setStyleSheet("QLabel { color : red; }")
                widget.setToolTip("Este parámetro fue modificado recientemente")
                logger.info("Parameter style modified successfully.")
            else:
                logger.warning(f"Widget is not a QLabel: {type(widget).__name__}")
        except AttributeError as e:
            logger.error(f"Failed to modify parameter style: {e}", exc_info=True)



    def create_parameters(self):
        """
        Creates a ParameterTree object to store the configuration settings for the XAAP application.

        The structure and default values for the parameters are loaded from a JSON file (`xaap_gui.json`).
        Additionally, the function sets the `start` and `end` datetime parameters based on the
        current time, a test date, or a user-specified datetime.

        Returns:
        --------
        Parameter:
            A Parameter object representing the XAAP configuration settings, including `start` and `end` datetime parameters.
        """

        # Initialize an empty Parameter object
        xaap_parameters = Parameter.create(name='xaap_configuration', type='group', children=[])
        # Read the JSON configuration file
        try:
            json_path = os.path.join(xaap_config_dir, self.xaap_gui_json)
            with open(json_path, 'r') as f:
                json_data = f.read()
            xaap_parameters.restoreState(json.loads(json_data))
            logger.info("Successfully loaded configuration from %s", json_path)
        except FileNotFoundError:
            logger.error("Configuration file not found: %s", json_path)
            raise
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in configuration file: %s", json_path)
            raise ValueError(f"Invalid JSON format: {e}")

        if self.start_datetime:
            # User-specified datetime
            try:
                start_datetime = UTCDateTime(self.start_datetime)

                # Check if end_datetime is provided; default to start_datetime + 1 day if not
                if hasattr(self, 'end_datetime') and self.end_datetime:
                    end_datetime = UTCDateTime(self.end_datetime)
                else:
                    end_datetime = start_datetime + 86400  # Add 1 day

            except Exception as e:
                logger.error("Invalid datetime format for detect_datetime or end_datetime: start=%s, end=%s",
                            self.detect_datetime, getattr(self, 'end_datetime', 'N/A'))
                raise ValueError(f"Invalid datetime format: {e}")

        elif self.detect_datetime == "test":
            start_datetime = UTCDateTime("2023-08-17 13:00:00")
            end_datetime = UTCDateTime("2023-08-17  20:00:00")
            
        elif self.detect_datetime =="default":
            # Default to the last 3 hours
            start_datetime = (UTCDateTime.now() - 3600*3).strftime("%Y-%m-%d %H:%M:%S")
            end_datetime = (UTCDateTime.now()).strftime("%Y-%m-%d %H:%M:%S")
        

        # Add datetime parameters to the `xaap_parameters` object
        try:
            dates_node = xaap_parameters.child('parameters').child('dates')
            dates_node.addChild({'name': 'start', 'type': 'str', 'value': str(start_datetime)})
            dates_node.addChild({'name': 'end', 'type': 'str', 'value': str(end_datetime)})
            logger.info("Added datetime parameters: start=%s, end=%s", start_datetime, end_datetime)
        except AttributeError as e:
            logger.error("Failed to add datetime parameters: %s", e)
            raise

        return xaap_parameters


    """FUNCTIONS LINKED TO THE GUI BUTTONS"""
    def gui_update_parameters(self):
        """
        Updates the application parameters based on the current GUI state.

        - Saves the current parameter state to a JSON string.
        - Creates a configuration object using the saved parameters.
        - Sends the configuration object to the processing component.

        Raises:
            ValueError: If saving the state or configuring parameters fails.
        """


        try:
            # Save the current state of the parameters
            xaap_state = self.params.saveState()
            self.json_xaap_state = json.dumps(xaap_state, indent=2)

            logger.info("Parameters saved successfully from the GUI state.")

            # Create a configuration object using the saved state
            self.xaap_config = configure_parameters_from_gui(self.json_xaap_state)
            logger.info("Configuration updated successfully.")
        except Exception as e:
            logger.error("Failed to update parameters: %s", e, exc_info=True)
            raise ValueError(f"Error in gui_update_parameters: {e}")
        



    def gui_request_stream(self):
        """
        Requests and retrieves the data stream for the configured volcano.

        Retrieves the data stream using the current XAAP configuration and stores it in `self.volcan_stream`.

        Raises:
            ValueError: If the data stream request fails.
        """

        try:
            logger.info("Requesting data stream...")
            self.volcan_stream = request_data.request_stream(self.xaap_config)
            logger.info("Data stream successfully retrieved.")
            logger.debug(f"Stream details: {self.volcan_stream}")
        except Exception as e:
            logger.error("Failed to request data stream: %s", e, exc_info=True)
            raise ValueError(f"Error in gui_request_stream: {e}")


    def gui_pre_process_stream(self):
        """
        Preprocesses the retrieved data stream.

        Processes the current data stream (`self.volcan_stream`) using the XAAP configuration
        to sort, merge, and filter the stream.

        Raises:
            ValueError: If the preprocessing fails.
        """
        try:
            if not self.volcan_stream:
                raise ValueError("No data stream available for preprocessing.")

            logger.info("Preprocessing the data stream...")
            self.volcan_stream = pre_process.pre_process_stream(self.xaap_config, self.volcan_stream)
            logger.info("Data stream preprocessing completed successfully.")
            logger.debug(f"Processed stream details: {self.volcan_stream}")
        except Exception as e:
            logger.error("Failed to preprocess data stream: %s", e, exc_info=True)
            raise ValueError(f"Error in gui_pre_process_stream: {e}")

    def gui_detection_sta_lta(self):
        """
        Detects triggers using the STA/LTA method.

        Processes the preprocessed data stream to detect triggers, then plots the results.

        Raises:
            ValueError: If trigger detection fails or if the data stream is not available.
        """
        try:
            if not self.volcan_stream:
                raise ValueError("No preprocessed data stream available for STA/LTA detection.")

            logger.info("Performing STA/LTA trigger detection...")
            self.triggers = detect_trigger.get_triggers(self.xaap_config, self.volcan_stream)
            logger.info(f"Trigger detection completed successfully. Number of triggers: {len(self.triggers)}")
            self.plot_triggers()
        except Exception as e:
            logger.error("Failed to detect triggers using STA/LTA: %s", e, exc_info=True)
            raise ValueError(f"Error in gui_detection_sta_lta: {e}")



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

        self.main_splitter = QSplitter(Qt.Horizontal)
        right_splitter = QSplitter(Qt.Vertical)
        self.layout.addWidget(self.main_splitter)




        self.main_splitter.addWidget(right_splitter)



        self.tree = ParameterTree(showHeader=False)
        #self.main_splitter.addWidget(self.tree)

        self.plot_window = GraphicsLayoutWidget()
        self.plot_window.setWindowTitle("XAAP")

        self.log = QTextEdit("Log Window")
        self.log.setReadOnly(True)

        #self.main_splitter.addWidget(self.plot_window)

        right_splitter.addWidget(self.tree)
        right_splitter.addWidget(self.log)


        self.main_splitter.addWidget(self.plot_window)




        """

        self.splitter3 = QSplitter()
        self.splitter3.setOrientation(Qt.Orientation.Vertical)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.splitter3.addWidget(self.log)
        self.main_splitter.addWidget(self.splitter3)
        

        self.splitter2 = QSplitter()
        self.splitter2.setOrientation(Qt.Orientation.Vertical)
        self.main_splitter.addWidget(self.splitter2)

        self.plot_window = GraphicsLayoutWidget()
        self.plot_window.setWindowTitle("XAAP")
        self.splitter2.addWidget(self.plot_window)
        """
        self.datetime_axis_1 = DateAxisItem(orientation='bottom', utcOffset=5)
        self.datetime_axis_2 = DateAxisItem(orientation='bottom', utcOffset=5)

        self.p1 = self.plot_window.addPlot(row=1, col=0, axisItems={'bottom': self.datetime_axis_1})
        self.p2 = self.plot_window.addPlot(row=2, col=0, axisItems={'bottom': self.datetime_axis_2})

        self.window_region = LinearRegionItem()
        self.p2.addItem(self.window_region, ignoreBounds=True)

        ##add log




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

    # Configure logging
    print("Configuring logging...")
    try:
        logger = configure_logging()
        logger.info("Logging configured successfully.")
    except Exception as e:
        print(f"Failed to configure logging: {e}")
        sys.exit(1)


    # Argument parser setup
    parser = argparse.ArgumentParser(
        description='XAAP_GUI will use the default configuration found in ./config/xaap_gui.json'
    )
    parser.add_argument(
        '--detect_datetime',
        type=str,
        default="default",
        help='Datetime to process. Overrides start_datetime and end_datetime if provided.',
    )
    parser.add_argument(
        '--start_datetime',
        type=str,
        help='Start datetime to process. Overrides detect_datetime if provided.',
    )
    parser.add_argument(
        '--end_datetime',
        type=str,
        help='End datetime to process. Defaults to start_datetime + 1 day if not provided.',
    )
    parser.add_argument(
        '--xaap_gui_json',
        type=str,
        default="xaap_gui.json",
        help='JSON config file for XAAP_GUI.',
    )

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        #sys.exit(1)

    # Log argument values
    logger.info(f'detect_datetime: {args.detect_datetime}')
    logger.info(f'start_datetime: {args.start_datetime}')
    logger.info(f'end_datetime: {args.end_datetime}')
    logger.info(f'xaap_gui_json: {args.xaap_gui_json}')

    app = pg.mkQApp()
    app.setStyleSheet("""
    QWidget {font-size: 15px}
    QMenu {font-size: 10px}
    QMenu QWidget {font-size: 10px}
                        """)
    try: 
        win = xaapGUI(args,
            detect_datetime=args.detect_datetime,
            xaap_gui_json=args.xaap_gui_json,
                    )
        win.setWindowTitle("xaap")
        win.show()
        win.resize(1600,900)

    except Exception as e:
        logger.error(f"Failed to initialize XAAP GUI: {e}", exc_info=True)
        sys.exit(1)   

    pg.exec()
