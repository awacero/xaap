# -*- coding: utf-8 -*-

import os,sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.parametertree import types as pTypes
import pyqtgraph.configfile
from obspy.signal import filter  


from obspy import read, UTCDateTime

class xaapGUI(QtGui.QWidget):

    def __init__(self):
        QtGui.QWidget.__init__(self)

        self.setupGUI()


        self.params = Parameter.create(name='params',type='group',children=[
            {'name':'Load Preset..','type':'list','limits':[]},
            {'name':'Save','type':'action'},
            {'name':'Load','type':'action'},
            {'name':'Parameters','type':'group','children':[
                {'name':'Filter','type':'group','children':[
                    {'name':'Filter type','type':'list','values':['highpass','bandpass','lowpass']},
                    {'name':'Freq_A','type':'float','value':0.5,'step':0.1,'limits': [0.1, None ]},
                    {'name':'Freq_B','type':'float','value':1.0,'step':0.1,'limits': [0.1, None ]}    ]},

                    {'name':'STA_LTA','type':'group', 'children': [
                    {'name':'sta','type':'float','value':2,'step':0.5,'limits': [0.1, None ]},                    
                    {'name':'lta','type':'float','value':10,'step':0.5,'limits': [1, None ]}  ]}      ]},
            {'name':'Reprocess','type':'action'}
                                                                            ]       )

        #        self.params.param('Recalculate Worldlines').sigActivated.connect(self.recalculate)
        self.params.param('Reprocess').sigActivated.connect(self.pre_process_stream)
        print(self.params.getValues())
        print(self.params.param('Save'))
        print("##")
        print(self.params['Parameters','Filter','Freq_A'])

        ##cargar aqui las formas de onda por defecto?
        self.read_stream()
        self.pre_process_stream()
        #self.plot_stream()

        self.tree.setParameters(self.params, showTop=False)
        self.params.param('Load Preset..').sigValueChanged.connect(self.loadPreset)

        ## read list of preset configs
        presetDir = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'config')
        if os.path.exists(presetDir):
            presets = [os.path.splitext(p)[0] for p in os.listdir(presetDir)]
            self.params.param('Load Preset..').setLimits(['']+presets)



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
        #self.plot_stream()


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

        self.region = pg.LinearRegionItem()
        self.region.setZValue(10)
        self.p2.addItem(self.region, ignoreBounds=True)


    def read_stream(self):
        
        data_file="/home/wacero/proyectos_codigo/xaap/xaap/data/EC.RETU..SHZ.D.2012.187"
        trace = read(data_file)[0]
        self.stream = read(data_file)      


    def pre_process_stream(self):

        ##VUELVE A FILTRAR LA SENAL, HAY QUE LEER LA ORIGINAL OTRA VEZ
        
        f_a = self.params['Parameters','Filter','Freq_A']
        self.stream.merge(method=1, fill_value="interpolate",interpolation_samples=0)
        sampling_rate = self.stream[0].stats.sampling_rate

        temp_data = filter.highpass(self.stream[0].data,f_a,sampling_rate,4)
        self.stream[0].data = temp_data
        self.plot_stream()
        #print("end pre process")

    def plot_stream(self):

        times = self.stream[0].times(type="timestamp")
        data_1 = self.stream[0].data
        self.p2.plot(times,data_1,pen="w")






























if __name__ == '__main__':
    app = pg.mkQApp()
    #import pyqtgraph.console
    #cw = pyqtgraph.console.ConsoleWidget()
    #cw.show()
    #cw.catchNextException()
    win = xaapGUI()
    win.setWindowTitle("xaap")
    win.show()
    win.resize(1100,700)

    pg.exec()