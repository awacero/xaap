'''
Created on Oct 1, 2021

@author: wacero
'''
import sys,os
from datetime import datetime
from get_mseed_data import get_mseed_utils as gmutils
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle

def main():

    if len(sys.argv) < 2:
        print(f"Invalid number of arguments. Usage: python {sys.argv[0]} ./config/profile_train_model_XXXX.cfg")
        sys.exit(1)

    unique_id = datetime.now().strftime("%Y%m%d%H%M%S")
    try:
        run_param = gmutils.read_parameters(sys.argv[1])
        print(run_param)
    except Exception as e:
        raise Exception("Error reading configuration file: %s" %str(e))

    try:
        volcano = run_param['ENVIRONMENT']['volcano']
        features_file = run_param['ENVIRONMENT']['features_file']
        model_folder = run_param['ENVIRONMENT']['model_folder']
        volcan_labels = list( map(str,run_param['ENVIRONMENT']['volcan_labels'].split(",")))
        best_features = list(map(int,run_param['ENVIRONMENT']['best_features'].split(",")))

    except Exception as e:
        raise Exception(f"Error reading parameters file:{e}")
    
    try:
        model_path=os.path.join("./" ,"%s" %model_folder)

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        column_names = ['id_code','seismic_type']
        data = pd.read_csv(features_file)
        rows_length,column_length = data.shape

        for i in range(column_length - 2):
            column_names.append("f_%s" %i)

        data.columns = column_names
        data.columns = data.columns.str.replace(' ', '')
        pattern = "|".join(volcan_labels)
        """Take note about spaces""" 
        data = data[data['seismic_type'].str.contains(pattern, regex=True, na=False)]
        ##data = data.loc[(data.seismic_type.str.contains('LP')) | data.seismic_type.str.contains('EXP')| data.seismic_type.str.contains('TREMI') ]
        #data = data[data['seismic_type'].isin(volcan_labels)]
        
        x_no_scaled = data.iloc[:,2:].to_numpy()
        scaler = StandardScaler()
        x = scaler.fit_transform(x_no_scaled)
        data_scaled = pd.DataFrame(x,columns=data.columns[2:])

        ##CATEGORIZAR ORDINAL ENCODER
        from sklearn.preprocessing import OrdinalEncoder
        ordinal_encoder = OrdinalEncoder()
        data_seismic_type = data[['seismic_type']]
        data_seismic_type_encode = ordinal_encoder.fit_transform(data_seismic_type)

        data_scaled_best = data_scaled.iloc[:,best_features].copy()
        #data_scaled_best = data_scaled.iloc[:,:]
        data_scaled_best['seismic_type'] = data_seismic_type_encode

        train_data, test_data = train_test_split(data_scaled_best,test_size=0.2,random_state=42)

        x_train = train_data.loc[:,train_data.columns !='seismic_type']
        #y_train = train_data.loc[:,train_data.columns =='seismic_type']
        y_train = train_data['seismic_type']
        x_test = test_data.loc[:,test_data.columns !='seismic_type']
        #y_test = test_data.loc[:,test_data.columns =='seismic_type']
        y_test = test_data["seismic_type"]
        rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
        rnd_clf.fit(x_train,y_train)

        y_pred = rnd_clf.predict(x_test)

        model_trained = os.path.join(model_path,f"{volcano}_rf_{unique_id}.pkl")
        model_bundle ={
            "model":rnd_clf,
            "labels": volcan_labels,
            "features":best_features
        }
        with open(model_trained,'wb') as f:
            pickle.dump(model_bundle,f)

        print(f"RESULT OF TRAINING: \nAccuracy:",metrics.accuracy_score(y_test, y_pred))
        print(f"Model stored as:{model_path}/{volcano}_rf_{unique_id}.pkl")

    except Exception as e:
        print("Error in training model was: %s" %str(e))     


if __name__ == "__main__": 
    main()
