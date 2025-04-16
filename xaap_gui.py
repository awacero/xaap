# -*- coding: utf-8 -*-

"""
LAST CHANGE
2024.05.07. 
Add save json and then  load it via terminal and delete old unused code 
2025.01.14
Improve code, add docs, create UML diagrams
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
        self.params.param("detection_sta_lta").sigActivated.connect(lambda: self.log_change(f"DETECTION_STA_LTA: {len(self.triggers)} detected. Continue"))
 
        self.params.param('detection_deep_learning').sigActivated.connect(lambda: self.gui_detection_deep_learning( ))
        self.params.param("detection_deep_learning").sigActivated.connect(lambda: self.log_change(f"DETECTION_DEEP_LEARNING:  {len(self.detections)} triggers; \
                                                                                                    {len(self.triggers)} coincidence triggers. Continue"))

        self.params.param('classify_detections').sigActivated.connect(self.classify_detections)


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
            #json_path = os.path.join(xaap_config_dir, self.xaap_gui_json)
            json_path = Path(self.xaap_gui_json)
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

        if hasattr(self,"start_datetime"):
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
        Perform detection using a deep learning model.

        - Creates a deep learning model based on the current configuration.
        - Processes each stream in the volcano stream using the model to retrieve detections.
        - Performs coincidence detection on individual detections.
        - Updates the GUI with the resulting detections and triggers.

        Raises:
            Exception: If the model creation or detection process fails.
        """
        try:
            # Step 1: Create the deep learning model
            logger.info("Creating deep learning model...")
            deep_learning_model = process_deep_learning.create_model(self.xaap_config)
            logger.info(f"Deep learning model '{deep_learning_model.name}' created successfully.")
        except Exception as e:
            logger.error("Error creating deep learning model: %s", e, exc_info=True)
            raise Exception(f"Failed to create deep learning model: {e}")

        try:
            # Step 2: Process streams to retrieve individual detections
            logger.info("Running detection with deep learning model...")
            detections = []
            for station_stream in self.volcan_stream:
                stream_detections = process_deep_learning.get_detections(
                    self.xaap_config, Stream(station_stream), deep_learning_model
                )
                detections.extend(stream_detections)

            if detections:
                logger.info(f"Deep learning detection completed. Number of individual detections: {len(detections)}")
                for detection in detections:
                    logger.debug(f"Detection: {detection}")
            else:
                logger.warning("No detections made by the deep learning model.")
        except Exception as e:
            logger.error("Error during detection processing: %s", e, exc_info=True)
            raise Exception(f"Detection process failed: {e}")

        if not detections:
            return  # No detections, no further processing needed

        self.detections = detections

        try:
            # Step 3: Perform coincidence detection
            logger.info("Running coincidence detection...")
            coincidence_detections = process_deep_learning.coincidence_detection_deep_learning(
                self.xaap_config, detections
            )
            self.triggers = coincidence_detections
            self.plot_triggers()

            logger.info(f"Coincidence detection completed. Number of coincidence detections: {len(self.triggers)}")

            for coincidence_detection in coincidence_detections:
                logger.debug(f"Coincidence Detection: {coincidence_detection}")

            # Step 4: Plot picks
            for detection in self.detections:
                logger.debug(f"Detection: {detection}")
                if hasattr(detection.pick_detection, 'phase'):
                    logger.debug(f"Phase: {detection.pick_detection.phase}")
            
            logger.info(f"###Result of detections was: {len(self.detections)}")
            logger.info(f"###Result of coincidence detections (TRIGGERS) was: {len(self.triggers)}")

            self.plot_picks()

        except Exception as e:
            logger.error("Error during coincidence detection: %s", e, exc_info=True)
            raise Exception(f"Coincidence detection failed: {e}")


        
    def plot_triggers(self):
        """
        Plots the trigger windows on the appropriate plot items.

        - Uses the configuration parameter `sta_lta_endtime_buffer` to calculate the extended duration.
        - Selects traces based on `trigger['trace_ids']` and slices the stream for the trigger duration.
        - Plots the sliced data on the corresponding plot item.

        Raises:
            ValueError: If required data or configuration is missing.
        """
        try:
            # Validate configuration
            sta_lta_endtime_buffer = float(self.xaap_config.sta_lta_endtime_buffer)
        except (AttributeError, ValueError) as e:
            logger.error("Invalid or missing `sta_lta_endtime_buffer` in configuration: %s", e)
            raise ValueError("Invalid `sta_lta_endtime_buffer` in configuration") from e

        if not self.triggers:
            logger.warning("No triggers available to plot.")
            return

        if not self.volcan_stream:
            logger.error("No data in the stream to plot triggers.")
            raise ValueError("The stream is empty. Cannot plot triggers.")

        for trigger in self.triggers:
            trace_ids = trigger.get('trace_ids')
            if not trace_ids:
                logger.warning("Trigger missing `trace_ids`: %s", trigger)
                continue

            for trace_id in trace_ids:
                # Match the trace ID to a plot item
                for plot_item in self.plot_items_list:
                    if plot_item.getAxis("left").labelText == trace_id:
                        # Select the trace and slice for the trigger window
                        trigger_trace_temp = self.volcan_stream.select(id=trace_id)
                        if not trigger_trace_temp:
                            logger.warning("No trace found for ID: %s", trace_id)
                            continue

                        try:
                            trigger_window = trigger_trace_temp.slice(
                                trigger['time'], 
                                trigger['time'] + trigger['duration'] * sta_lta_endtime_buffer
                            )
                            if not trigger_window:
                                logger.warning("Empty trigger window for trace ID: %s", trace_id)
                                continue

                            # Plot the trigger window
                            plot_item.plot(
                                trigger_window[0].times(type='timestamp'),
                                trigger_window[0].data,
                                pen='r'
                            )
                            logger.info(f"Plotted trigger for trace ID: {trace_id}")
                        except Exception as e:
                            logger.error("Error processing trigger for trace ID %s: %s", trace_id, e, exc_info=True)


    def plot_picks(self):
        """
        Plots "P" and "S" phase picks on the corresponding plot items.

        For each detection, the function adds a vertical line and label at the pick's timestamp
        based on its phase ("P" or "S").

        Raises:
            ValueError: If detections or necessary attributes are missing.
        """
        if not self.detections:
            logger.warning("No detections available to plot.")
            return

        for detection in self.detections:
            if not hasattr(detection.pick_detection, 'phase'):
                logger.warning("Detection missing 'phase': %s", detection)
                continue

            pick = detection.pick_detection
            try:
                pick_time = UTCDateTime(pick.start_time)
            except Exception as e:
                logger.error("Invalid start time for pick: %s. Error: %s", pick.start_time, e, exc_info=True)
                continue

            if not pick.trace_id:
                logger.warning("Missing trace ID for pick: %s", pick)
                continue

            # Plot for the corresponding trace ID
            for plot_item in self.plot_items_list:
                if plot_item.getAxis("left").labelText.startswith(pick.trace_id):
                    # Consolidated logic for P and S phases
                    phase_color = 'cyan' if pick.phase == 'P' else 'blue' if pick.phase == 'S' else None
                    if phase_color:
                        # Add label and vertical line
                        label = pg.TextItem(text=pick.phase, color='r', anchor=(0.5, 0))
                        y_range = plot_item.viewRange()[1]
                        label_y_pos = y_range[1] - (y_range[1] - y_range[0]) * 0.1
                        label.setPos(pick_time.timestamp, label_y_pos)
                        vertical_line = pg.InfiniteLine(pos=pick_time.timestamp, angle=90, pen=phase_color)

                        # Add to the plot
                        plot_item.addItem(vertical_line)
                        plot_item.addItem(label)
                        ##logger.debug(f"Plotted {pick.phase} pick at {pick_time} for trace ID {pick.trace_id}.")
                    else:
                        logger.warning(f"Unknown phase '{pick.phase}' for trace ID {pick.trace_id}.")


    def plot_stream(self):
        """
        Plots the seismic data streams in the GUI.

        - Clears existing plots and initializes new ones based on the number of traces.
        - Links the X-axis of all plots for synchronized scrolling.

        Raises:
            Exception: If an error occurs during plotting, it is logged and re-raised.
        """
        logger.info("Starting plot_stream. Clearing existing plots...")
        
        try:
            # Clear existing plots
            self.p1.clearPlots()
            self.p2.clearPlots()
            self.plot_window.clear()
            self.plot_items_list = []

            # Setup plots for each trace
            n_plots = len(self.volcan_stream)
            logger.info(f"Number of traces to plot: {n_plots}")

            if n_plots == 0:
                logger.warning("No traces available in the stream to plot.")
                return

            for i, trace in enumerate(self.volcan_stream):
                # Create a new plot with a datetime axis
                datetime_axis = pg.graphicsItems.DateAxisItem.DateAxisItem(orientation='bottom', utcOffset=5)
                plot = self.plot_window.addPlot(
                    row=(3 + i), col=0, axisItems={'bottom': datetime_axis}
                )
                view_box = plot.getViewBox()
                view_box.setMouseMode(pg.ViewBox.RectMode)
                self.plot_items_list.append(plot)

                # Plot the trace data
                logger.info(f"Plotting trace {i + 1}: {trace.id}")
                plot.clearPlots()
                plot.plot(trace.times(type="timestamp"), trace.data, pen='g')
                plot.getAxis("left").setWidth(100)
                plot.setLabel(
                    "left",
                    text=f"{trace.stats['network']}.{trace.stats['station']}.{trace.stats['location']}.{trace.stats['channel']}"
                )

            # Link X-axes for synchronized scrolling
            logger.info("Linking X-axes for synchronized scrolling...")
            vbox_0 = self.plot_items_list[0]
            for plot_item in self.plot_items_list:
                plot_item.setXLink(vbox_0)

            logger.info("Finished plot_stream successfully.")
        except Exception as e:
            logger.error(f"Error in plot_stream: {e}", exc_info=True)
            raise Exception(f"Error during plotting: {e}")









    def classify_detections(self):

        domains = " ".join(self.xaap_config.classification_feature_domains.replace(","," ").split())

        if self.triggers:
            try:
                logger.info("Recover trigger traces")
                self.triggers_traces = detect_trigger.create_trigger_traces(self.xaap_config,self.volcan_stream,self.triggers)
            except Exception as e:
                logger.error(f"Error while retrieving trigger traces: {e}")

            feature_config = {"features_file":self.xaap_config.classification_feature_file,
                              "domains":domains                             
                              }

            model_file = Path(f"{self.xaap_config.classification_model_file}")            
            volcano_classifier = pickle.load(open(model_file,'rb'))
            volcano_classifier_model = volcano_classifier["model"]
            volcano_classifier_labels = volcano_classifier["labels"]

            classified_triggers_file = Path(xaap_dir,"data/classifications") / UTCDateTime.now().strftime("out_xaap_%Y.%m.%d.%H.%M.%S.csv")
            classification_file = open(classified_triggers_file,'a+')
            features = FeatureVector(feature_config, verbatim=2)
            input_data = []

            logger.info("start feature calculation")
            for trace in self.triggers_traces:
                ##Modificar file code para que incluya la ventana de end_time
                trace_window = int(trace.stats.endtime - trace.stats.starttime)
                file_code = "%s.%s.%s.%s.%s.%s" %(trace.stats.network,trace.stats.station,trace.stats.location,trace.stats.channel,trace.stats.starttime.strftime("%Y.%m.%d.%H.%M.%S"),trace_window)
                features.compute(trace.data,trace.stats.sampling_rate)
                row = np.append(file_code, features.featuresValues)
                input_data.append(row)


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

            y_pred = volcano_classifier_model.predict(data_scaled)
            logger.info(f"Classifications made were:{y_pred.shape}")

            for i in range(rows_length):
                #prediction = "%s,%s\n" %(data.iloc[i,0],volcano_classifier_labels[int(y_pred[i])])
                prediction = f"{data.iloc[i,0]},{volcano_classifier_labels[int(y_pred[i])]}\n"
                classification_file.write(prediction)
        else:
            logger.info(f"No triggers: {len(self.triggers)}")
    
    def setupGUI(self):
        """
        Sets up the GUI layout, including the menu bar, parameter tree, plot window, and log window.

        - Initializes the main layout and menu bar with actions for saving parameters and toggling the parameter tree.
        - Sets up a main splitter to organize the parameter tree, log window, and plot window.
        - Adds datetime axes and a linear region item to the plot window for visualization.
        """
        # Initialize main layout
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

        # Create a container for the menu bar
        logger.info("Setting up the menu bar...")
        self.menu_bar_container = QWidget()
        self.menu_bar_container.setFixedHeight(25)
        self.layout.addWidget(self.menu_bar_container)

        self.menu_bar = QMenuBar(self.menu_bar_container)
        self.menu_bar_container_layout = QVBoxLayout(self.menu_bar_container)
        self.menu_bar_container_layout.setContentsMargins(0, 0, 0, 0)
        self.menu_bar_container_layout.addWidget(self.menu_bar)

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
        self.menu_bar.addMenu(self.menu)

        # Main splitter setup
        logger.info("Setting up main splitter...")
        self.main_splitter = QSplitter(Qt.Horizontal)
        right_splitter = QSplitter(Qt.Vertical)
        self.layout.addWidget(self.main_splitter)
        self.main_splitter.addWidget(right_splitter)

        # Parameter tree
        self.tree = ParameterTree(showHeader=False)
        right_splitter.addWidget(self.tree)

        # Log window
        self.log = QTextEdit("Log Window")
        self.log.setReadOnly(True)
        right_splitter.addWidget(self.log)

        # Plot window setup
        logger.info("Setting up plot window...")
        self.plot_window = GraphicsLayoutWidget()
        self.plot_window.setWindowTitle("XAAP")
        self.main_splitter.addWidget(self.plot_window)

        # Create datetime axes and plots
        self.datetime_axis_1 = DateAxisItem(orientation='bottom', utcOffset=5)
        self.datetime_axis_2 = DateAxisItem(orientation='bottom', utcOffset=5)

        self.p1 = self.plot_window.addPlot(row=1, col=0, axisItems={'bottom': self.datetime_axis_1})
        self.p2 = self.plot_window.addPlot(row=2, col=0, axisItems={'bottom': self.datetime_axis_2})

        # Add linear region item for zooming or selection
        self.window_region = LinearRegionItem()
        self.p2.addItem(self.window_region, ignoreBounds=True)

        logger.info("GUI setup completed.")



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
        default="./config/xaap_gui.json",
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
