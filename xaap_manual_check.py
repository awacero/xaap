# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np

import pyqtgraph as pg
from pyqtgraph.widgets import MatplotlibWidget

###from pyqtgraph.Qt import QtGui, QtCore

from pyqtgraph.Qt import QtGui
from PyQt5.QtWidgets import QWidget
from pyqtgraph.Qt import QtCore
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QSplitter


from pyqtgraph import TableWidget
from pyqtgraph.parametertree import Parameter, ParameterTree
import matplotlib.pyplot as plt
#from mpl_axes_aligner import align

from PyQt5.QtWidgets import QHeaderView , QPushButton, QShortcut, QApplication
from PyQt5.QtGui import QKeySequence

from xaap.models import xaap_check 

##class xaapCheck(QtGui.QWidget):
class xaapCheck(QWidget):


    def __init__(self):

        xaap_check.logger.info("Continuation of all this")

        ##QtGui.QWidget.__init__(self)
        QWidget.__init__(self)
        self.classifications_path = Path(xaap_check.xaap_dir,"data/"+
                                         "classifications/")
        
        self.setupGUI()
        self.params = self.create_parameters()
        if xaap_check.os.path.exists(self.classifications_path):
            classified_triggers_files =\
                [xaap_check.os.path.splitext(p)[0]\
                for p in xaap_check.os.listdir(self.classifications_path)]
            self.params.param('Classification file').\
                              setLimits(classified_triggers_files)

        self.tree.setParameters(self.params, showTop=True)
        xaap_check.connect_to_mseed_server(self)

        self.table_widget.verticalHeader().\
            sectionDoubleClicked.connect(self.get_plots)
        self.table_widget.verticalHeader().sectionClicked.connect(self.get_plots)
        self.params.param('Load classifications').\
            sigActivated.connect(self.load_data_to_tbl_widget)
        

    def load_data_to_tbl_widget(self):

        predicted_data = xaap_check.load_csv_file(self)
        self.table_widget.setColumnCount(6)
        self.table_widget.setFont(QtGui.QFont('Helvetica', 10))
        self.table_widget.appendData(predicted_data[['trigger_code','','',
                                     'Station','Network','Component','Fecha',
                                     'Hora','prediction','','coda','','','',
                                     '','']].to_numpy())
        self.table_widget.setHorizontalHeaderLabels(str("Trigger code;Cod Registro;"+
                                                        "Volcan;Estación;"+
                                                        "Net;Componente;Fecha;Hora;"+
                                                        "Tipo;S-P;Coda;Amp-Cuentas;"+
                                                        "T;f;RMS;;").split(";"))
        self.table_widget.setColumnWidth(0, 0) # Trigger code
        self.table_widget.setColumnWidth(1, 100) # Cod Registro
        self.table_widget.setColumnWidth(2, 70) # Volcan
        self.table_widget.setColumnWidth(3, 70) # Estación
        self.table_widget.setColumnWidth(4, 40) # Net
        self.table_widget.setColumnWidth(5, 100) # Componente
        self.table_widget.setColumnWidth(6, 80) # Fecha
        self.table_widget.setColumnWidth(7, 65) # Hora
        self.table_widget.setColumnWidth(8, 65) # Tipo
        self.table_widget.setColumnWidth(9, 40) # S-P
        self.table_widget.setColumnWidth(10, 50) # Coda
        self.table_widget.setColumnWidth(11, 110) # Amp-Cuentas
        self.table_widget.setColumnWidth(12, 50) # T
        self.table_widget.setColumnWidth(13, 50) # f
        self.table_widget.setColumnWidth(14, 50) # RMS
        self.table_widget.setColumnWidth(15, 50) # Botón

        # create a button column
        for index in range(self.table_widget.rowCount()):
            btn = QPushButton(self.table_widget)
            btn.setText('Ver')
            btn.setStyleSheet("background-color: blue; color: white")
            btn.clicked.connect(self.get_plots)
            self.table_widget.setCellWidget(index, 15, btn)


    def get_plots(self):

        xaap_check.logger.info(self.table_widget.currentRow())
        trigger_code = self.table_widget.item(self.table_widget.currentRow(), 0).text()
        
        self.trigger_times, self.paded_stream, self.trigger_stream,\
        self.paded_times, max_trigger, min_trigger,  amp_max, max_loc,\
            min_loc, rms = xaap_check.get_trigger(self, trigger_code)


        #########################
        # Add data to selected row in table widget
        self.table_widget.item(self.table_widget.currentRow(), 11).setText(str(amp_max))
        self.table_widget.item(self.table_widget.currentRow(), 14).setText(str(rms))

        #########################

        self.trigger_plot.clearPlots()
        self.plot_b.clearPlots()
        self.plot_b.showGrid(x=True, y=True)
        self.trigger_plot.showGrid(x=True, y=True)
        self.trigger_plot.plot([float(self.trigger_times[max_loc])],\
            [max_trigger], symbol = '+')
        self.trigger_plot.plot([float(self.trigger_times[min_loc])],\
            [min_trigger], symbol = '+')
        self.trigger_plot.plot(self.trigger_times,self.trigger_stream[0].data,pen='g')
        self.plot_b_spectro = self.plot_b.plot(self.trigger_times,\
                                               self.trigger_stream[0].data, pen='r')
        self.plot_b_spectro.setFftMode(True)
        self.paded_plot.plot(self.trigger_times,self.trigger_stream[0].data,pen='r')

        self.mw_fig = self.mw.getFigure()
        #self.mw_fig.clf()

        self.mw_axes = self.mw_fig.add_subplot(111)
        """"
        
        sampling_rate = self.trigger_stream[0].stats.sampling_rate
        #print("Sampling rate:", sampling_rate)
        
        # self.trigger_stream.spectrogram(samp_rate=sampling_rate,\
        # cmap=plt.cm.jet,log=False,axes=self.mw_axes)
        self.trigger_stream.spectrogram(wlen=2*sampling_rate,\
                                        per_lap=0.95,dbscale=True,\
                                        log=False,axes=self.mw_axes,cmap=plt.cm.jet)
        
        """
        sampling_rate = self.trigger_stream[0].stats.sampling_rate
        num_samples = len(self.trigger_stream[0].data)
        data_duration = len(self.trigger_stream[0].data) / sampling_rate


        # Calculate window length as 1% of data duration, in seconds
        wlen = data_duration * 0.01

        # If window length is less than 1, set it to 1 (minimum)
        wlen = max(wlen, 1)

        # Calculate nfft as the next power of 2 greater than or equal to window length * sampling rate
        nfft = 2**np.ceil(np.log2(wlen * sampling_rate))


        self.trigger_stream.spectrogram(wlen=wlen, per_lap=0.95, dbscale=True,
                                        log=False, axes=self.mw_axes, cmap=plt.cm.jet)

        
        
        
        
        self.mw.draw()

        self.paded_plot.clearPlots()
        self.paded_plot.showGrid(x=True, y=True)
        self.paded_plot.plot(self.paded_times,self.paded_stream[0].data,pen='w')
        self.paded_plot.plot(self.trigger_times,self.trigger_stream[0].data,pen='r')


    def clearAllPlots(self):
        self.paded_plot.clearPlots()
        self.trigger_plot.clearPlots()
        self.plot_b.clearPlots()


    def setupGUI(self):

        ##self.layout = QtGui.QVBoxLayout()
        
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0,0,0,0)
        self.setLayout(self.layout)

        # Shortcuts for GUI
        # Exit
        self.quitSc = QShortcut(QKeySequence('Ctrl+Q'), self)
        self.quitSc.activated.connect(QApplication.instance().quit)
        # Load csv file
        self.quitSc = QShortcut(QKeySequence('Ctrl+L'), self)
        self.quitSc.activated.connect(self.load_data_to_tbl_widget)
        # Graph selected row
        self.quitSc = QShortcut(QKeySequence('Ctrl+G'), self)
        self.quitSc.activated.connect(self.get_plots)
        # Clear plots
        self.quitSc = QShortcut(QKeySequence('Ctrl+C'), self)
        self.quitSc.activated.connect(self.clearAllPlots)

        self.table_widget = TableWidget(editable=True)

        # Uncomment next line if want to fill all space of the table widget
        #self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.tree =  ParameterTree()
        self.datetime_axis_1 = pg.graphicsItems.DateAxisItem.DateAxisItem(orientation = 'bottom',utcOffset=5)
        self.datetime_axis_2 = pg.graphicsItems.DateAxisItem.DateAxisItem(orientation = 'bottom',utcOffset=5)

        self.main_layout = pg.GraphicsLayoutWidget()
        self.side_layout = pg.GraphicsLayoutWidget()      
        
        self.trigger_plot = self.main_layout.addPlot(row=0, col=0,axisItems={'bottom': self.datetime_axis_1})
        self.paded_plot = self.main_layout.addPlot(row=1, col=0,axisItems={'bottom': self.datetime_axis_2})

        # Zoom with square area using mouse
        view_box_i = self.trigger_plot.getViewBox()
        view_box_i.setMouseMode(pg.ViewBox.RectMode)

        self.mw = MatplotlibWidget.MatplotlibWidget()
        self.plot_b = self.side_layout.addPlot(row=1,col=0)
        #self.plot_c = self.side_layout.addPlot(row=2,col=0)
        view_box_i2 = self.plot_b.getViewBox()
        view_box_i2.setMouseMode(pg.ViewBox.RectMode)

        view_box_i3 = self.paded_plot.getViewBox()
        view_box_i3.setMouseMode(pg.ViewBox.RectMode)

        '''horizontal splitter sides widgets horizontally'''
        self.splitter_horizontal = QSplitter()
        self.splitter_horizontal.setOrientation(QtCore.Qt.Orientation.Horizontal)
        '''vertical splitter stacks widgets vertically'''
        self.splitter_vertical = QSplitter()
        self.splitter_vertical.setOrientation(QtCore.Qt.Orientation.Vertical)
        
        self.splitter_vertical_side = QSplitter()
        self.splitter_vertical_side.setOrientation(QtCore.Qt.Orientation.Vertical)

        self.splitter_vertical.addWidget(self.main_layout)
        self.splitter_vertical.addWidget(self.table_widget)

        self.splitter_vertical_side.addWidget(self.mw)
        self.splitter_vertical_side.addWidget(self.side_layout)

        self.splitter_horizontal.addWidget(self.tree)
        self.splitter_horizontal.addWidget(self.splitter_vertical)
        self.splitter_horizontal.addWidget(self.splitter_vertical_side)

        self.splitter_horizontal.setSizes([265,635,200])
        self.splitter_vertical.setSizes([600,200])
        self.layout.addWidget(self.splitter_horizontal)


    def create_parameters(self):

        xaap_parameters = Parameter.create(

            name='xaap configuration',type='group',children=[

            {'name':'Classification file','type':'list','limits':[]},
            {'name':'Parameters','type':'group','children':[

                {'name':'MSEED','type':'group','children':[
                    {'name':'client_id','type':'list',\
                        'values':['FDSN','SEEDLINK','ARCHIVE','ARCLINK']},
                    {'name':'server_config_file','type':'str',\
                        'value':'%s' %('server_configuration.json')}
                                                                 ]},

                {'name':'GUI','type':'group','children':[
                    {'name':'zoom_region_size','type':'float',\
                        'value':0.10,'step':0.05,'limits':[0.01,1] }
                                                            ]},
                                                          ]},
            {'name':'Load classifications','type':'action'},
            {'name':'Save triggers','type':'action'}
                                                                         ]
                                                                         )
        xaap_check.logger.info("End of set parameters")
        return xaap_parameters




        

if __name__ == '__main__':
    app = pg.mkQApp()
    app.setStyleSheet("""
    QWidget {font-size: 15px}
    QMenu {font-size: 10px}
    QMenu QWidget {font-size: 10px}
    """)

    win = xaapCheck()
    win.setWindowTitle("xaap_check")
    win.showMaximized()
    win.resize(1100,700)

    pg.exec()
