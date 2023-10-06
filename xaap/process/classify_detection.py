from xaap.process import detect_trigger

import pickle
import os
from obspy import UTCDateTime   
from pathlib import Path
from aaa_features.features import FeatureVector
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import logging
logger = logging.getLogger(__name__)



def classify_detection_SVM(xaap_config, volcan_stream, detections):

    feature_config = {"features_file":f"{xaap_config.xaap_folder_config}/features/features_00.json" ,
                #"domains":"time spectral cepstral"}
                "domains":"spectral cepstral"}
    #tungu_clf= pickle.load(open(os.path.join('%s/data/models' %xaap_dir,'tungurahua_rf_20211007144655.pkl'),'rb'))
    #chiles_rf_20230410092541
    #volcano_classifier_model = pickle.load(open(os.path.join('%s/data/models' %xaap_dir,'chiles_rf_20220902115108.pkl'),'rb'))
    volcano_classifier_model = pickle.load(open(os.path.join(f'{xaap_config.xaap_folder_data}/models','chiles_rf_20230410092541.pkl'),'rb'))

    classified_triggers_file = Path(xaap_config.xaap_folder_data,"classifications") / UTCDateTime.now().strftime("out_xaap_%Y.%m.%d.%H.%M.%S.csv")
    classification_file = open(classified_triggers_file,'a+')
    #Como guardar categorias en  el modelo?

    categories = ['LP', 'VT']
    features = FeatureVector(feature_config, verbatim=2)
    input_data = []


    if detections:

        detections_traces = detect_trigger.create_trigger_traces(xaap_config,volcan_stream,detections)
        logger.info("start feature calculation")
        for trace in detections_traces:
            '''
            print("!####")
            print(trace)
            '''
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


        y_pred=volcano_classifier_model.predict(data_scaled)
        '''
        print(type(y_pred))
        print(y_pred.shape)
        '''

        for i in range(rows_length):
            prediction = "%s,%s\n" %(data.iloc[i,0],categories[int(y_pred[i])])
            logger.info(prediction)
            classification_file.write(prediction)



