# -*- coding: utf-8 -*-

import os, sys
from pathlib import Path

from get_mseed_data import get_mseed_utils as gmutils
from get_mseed_data import get_mseed

import pyqtgraph as pg
from pyqtgraph.widgets import MatplotlibWidget
from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph import TableWidget
from pyqtgraph.parametertree import Parameter, ParameterTree
import matplotlib.pyplot as plt


from obspy import UTCDateTime
import pandas as pd
import logging, logging.config


xaap_dir = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])))
xaap_config_dir = Path("%s/%s" %(xaap_dir,"config"))
logging.config.fileConfig(xaap_config_dir / "logging.ini" ,disable_existing_loggers=False)
logger = logging.getLogger('stdout')
logger.setLevel(logging.INFO)


class xaap_check(QtGui.QWidget):

    def __init__(self):

        logger.info("Continuation of all this #$%&")

        QtGui.QWidget.__init__(self)
        
        self.classifications_path = Path(xaap_dir,"data/classifications/")
        
        self.setupGUI()
        self.params = self.create_parameters()
        if os.path.exists(self.classifications_path):
            classified_triggers_files = [os.path.splitext(p)[0] for p in os.listdir(self.classifications_path)]
            self.params.param('Classification file').setLimits(classified_triggers_files)

        self.tree.setParameters(self.params, showTop=True)
        self.connect_to_mseed_server()

        self.table_widget.verticalHeader().sectionDoubleClicked.connect(self.get_trigger)
        self.params.param('Load classifications').sigActivated.connect(self.load_csv_file)


    def connect_to_mseed_server(self):

        try:
            self.mseed_client_id = self.params['Parameters','MSEED','client_id']
            mseed_server_config_file = xaap_config_dir / self.params['Parameters','MSEED','server_config_file']
            mseed_server_param = gmutils.read_config_file(mseed_server_config_file)
            
            self.mseed_client = get_mseed.choose_service(mseed_server_param[self.mseed_client_id])

        except Exception as e:
            raise Exception("Error connecting to MSEED server : %s" %str(e))
        

    def load_csv_file(self):

        try:
            classification_file_path = Path(self.classifications_path , self.params['Classification file']+'.txt')
            predicted_data  = pd.read_csv(classification_file_path,sep=',')
            rows_length,column_length = predicted_data.shape
            column_names = ['trigger_code','prediction']
            predicted_data.columns = column_names
            predicted_data['operator']=''
            ##usar pyqtgraph metaarray para los nombres de columna
            self.table_widget.setData(predicted_data.to_numpy())

        except Exception as e:
            logger.error("Error reading classification file : %s" %str(e))
            raise Exception("Error reading classification file : %s" %str(e))
        
    
    def get_trigger(self):

        logger.info(self.table_widget.currentRow())
        trigger_code = self.table_widget.currentItem().text()
        net,station,location,channel,Y,m,d,H,M,S,window = trigger_code.split(".")
        if not location:
            location = ''
        start_time=UTCDateTime("%s-%s-%sT%s:%s:%s"%(Y,m,d,H,M,S))
        window = int(window)
        end_time = start_time + window

        print(net,station,start_time)
        #poner try, llamar a preprocesar, usar informacion de filtros en parametros, agregar pads?
        self.trigger_stream = get_mseed.get_stream(self.mseed_client_id,self.mseed_client,net,station,location,channel,start_time=start_time,window_size=window)
        self.trigger_times = self.trigger_stream[0].times(type='timestamp')
        print(self.trigger_stream)

        self.trigger_plot.clearPlots()
        self.plot_b.clearPlots()
        self.trigger_plot.plot(self.trigger_times,self.trigger_stream[0].data,pen='g')
        self.plot_b_spectro = self.plot_b.plot(self.trigger_times,self.trigger_stream[0].data, pen='r')
        self.plot_b_spectro.setFftMode(True)

        self.mw_fig = self.mw.getFigure()
        #self.mw_fig.clf()
        self.mw_axes = self.mw_fig.add_subplot(111)
        sampling_rate = self.trigger_stream[0].stats.sampling_rate
        #self.trigger_stream.spectrogram(samp_rate=sampling_rate,cmap=plt.cm.jet,log=False,axes=self.mw_axes)
        self.trigger_stream.spectrogram(wlen=2*sampling_rate, per_lap=0.95,dbscale=True,log=False,axes=self.mw_axes,cmap=plt.cm.jet)
        self.mw.draw()

        pad = 300
        
        self.paded_stream = get_mseed.get_stream(self.mseed_client_id,self.mseed_client,net,station,location,channel,start_time=start_time - pad ,end_time=end_time +  pad)
        
        logger.info("Get paded stream")
        
        self.paded_times = self.paded_stream[0].times(type='timestamp')

        self.paded_plot.clearPlots()
        self.paded_plot.plot(self.paded_times,self.paded_stream[0].data,pen='w')
        self.paded_plot.plot(self.trigger_times,self.trigger_stream[0].data,pen='r')

        


    def setupGUI(self):

        self.layout = QtGui.QVBoxLayout()
        self.layout.setContentsMargins(0,0,0,0)
        self.setLayout(self.layout)

        self.table_widget = TableWidget(editable=True)
        self.tree =  ParameterTree()
        self.datetime_axis_1 = pg.graphicsItems.DateAxisItem.DateAxisItem(orientation = 'bottom',utcOffset=5)
        self.datetime_axis_2 = pg.graphicsItems.DateAxisItem.DateAxisItem(orientation = 'bottom',utcOffset=5)

        self.main_layout = pg.GraphicsLayoutWidget()
        self.side_layout = pg.GraphicsLayoutWidget()      
        
        self.trigger_plot = self.main_layout.addPlot(row=0, col=0,axisItems={'bottom': self.datetime_axis_1})
        self.paded_plot = self.main_layout.addPlot(row=1, col=0,axisItems={'bottom': self.datetime_axis_2})

        self.mw = MatplotlibWidget.MatplotlibWidget()
        self.plot_b = self.side_layout.addPlot(row=1,col=0)
        #self.plot_c = self.side_layout.addPlot(row=2,col=0)


        '''horizontal splitter sides widgets horizontally'''
        self.splitter_horizontal = QtGui.QSplitter()
        self.splitter_horizontal.setOrientation(QtCore.Qt.Orientation.Horizontal)
        '''vertical splitter stacks widgets vertically'''
        self.splitter_vertical = QtGui.QSplitter()
        self.splitter_vertical.setOrientation(QtCore.Qt.Orientation.Vertical)
        
        self.splitter_vertical_side = QtGui.QSplitter()
        self.splitter_vertical_side.setOrientation(QtCore.Qt.Orientation.Vertical)

        self.splitter_vertical.addWidget(self.main_layout)
        self.splitter_vertical.addWidget(self.table_widget)

        self.splitter_vertical_side.addWidget(self.mw)
        self.splitter_vertical_side.addWidget(self.side_layout)

        self.splitter_horizontal.addWidget(self.tree)
        self.splitter_horizontal.addWidget(self.splitter_vertical)
        self.splitter_horizontal.addWidget(self.splitter_vertical_side)
        
        self.splitter_horizontal.setSizes([300,600,200])
        self.splitter_vertical.setSizes([600,200])
        self.layout.addWidget(self.splitter_horizontal)
    

    def create_parameters(self):
        
        xaap_parameters = Parameter.create(
            
            name='xaap configuration',type='group',children=[

            {'name':'Classification file','type':'list','limits':[]},
            {'name':'Parameters','type':'group','children':[

                {'name':'MSEED','type':'group','children':[
                    {'name':'client_id','type':'list','values':['FDSN','SEEDLINK','ARCHIVE','ARCLINK']},
                    {'name':'server_config_file','type':'str','value':'%s' %('server_configuration.json')}
                                                                 ]},

                {'name':'GUI','type':'group','children':[
                    {'name':'zoom_region_size','type':'float','value':0.10,'step':0.05,'limits':[0.01,1] }
                                                            ]},
                                                          ]},
            {'name':'Load classifications','type':'action'},
            {'name':'Save triggers','type':'action'}                
                                                                         ]                                                                         
                                                                         )
        logger.info("End of set parameters") 
        return xaap_parameters




        

if __name__ == '__main__':
    app = pg.mkQApp()
    app.setStyleSheet("""
    QWidget {font-size: 15px}
    QMenu {font-size: 10px}
    QMenu QWidget {font-size: 10px}
""")

    win = xaap_check()
    win.setWindowTitle("xaap_check")
    win.show()
    win.resize(1100,700)

    pg.exec()