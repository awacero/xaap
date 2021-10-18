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
    is_error =  False

    if len(sys.argv)==1:
        is_error = True
        
    else:
        unique_id = datetime.now().strftime("%Y%m%d%H%M%S")
        try:
            run_param = gmutils.read_parameters(sys.argv[1])
        except Exception as e:
            raise Exception("Error reading configuration file: %s" %str(e))

        try:
            volcano = run_param['ENVIRONMENT']['volcano']
            features_file = run_param['ENVIRONMENT']['features_file']
            model_folder = run_param['ENVIRONMENT']['model_folder']

        except Exception as e:
            raise Exception("Error reading parameters file: %s" %str(e))
        
        try:
            column_names = ['id_code','seismic_type']
            data = pd.read_csv(features_file)

            rows_length,column_length = data.shape

            for i in range(column_length - 2):
                column_names.append("f_%s" %i)

            data.columns = column_names
            data.columns = data.columns.str.replace(' ', '')

            #Use just 2 types of events
            #data = data.loc[(data.seismic_type.str.contains('LP')) | data.seismic_type.str.contains('TREMI')| data.seismic_type.str.contains('EXP')|(data.seismic_type.str.contains('VT')) ]
            data['seismic_type'].hist() 

            x_no_scaled = data.iloc[:,2:].to_numpy()
            scaler = StandardScaler()
            x = scaler.fit_transform(x_no_scaled)
            data_scaled = pd.DataFrame(x,columns=data.columns[2:])

            ##CATEGORIZAR ORDINAL ENCODER
            from sklearn.preprocessing import OrdinalEncoder
            ordinal_encoder = OrdinalEncoder()
            data_seismic_type = data[['seismic_type']]
            data_seismic_type_encode = ordinal_encoder.fit_transform(data_seismic_type)


            categories = ordinal_encoder.categories_[0].tolist()

            print(categories)

            ##CHOSE BEST FEATURES 
            best_features =  [2, 3, 6, 8, 9, 11, 23, 26, 27, 29, 42, 44, 45, 47, 48, 50, 51, 52, 53, 63, 67, 71, 72, 73, 75, 95]
            #data_scaled_best = data_scaled.iloc[:,best_features]
            data_scaled_best = data_scaled.iloc[:,:]

            data_scaled_best['seismic_type'] = data_seismic_type_encode

            print(data_scaled_best.head())




            train_data, test_data = train_test_split(data_scaled_best,test_size=0.2,random_state=42)

            x_train = train_data.loc[:,train_data.columns !='seismic_type']
            y_train = train_data.loc[:,train_data.columns =='seismic_type']

            x_test = test_data.loc[:,test_data.columns !='seismic_type']
            y_test = test_data.loc[:,test_data.columns =='seismic_type']

            

            rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
            rnd_clf.fit(x_train,y_train)

            y_pred = rnd_clf.predict(x_test)
            print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

            model_path=os.path.join("./" ,"%s" %model_folder)

            if not os.path.exists(model_path):
                os.makedirs(model_path)

            pickle.dump(rnd_clf, open(os.path.join(model_path,'%s_rf_%s.pkl'%(volcano,unique_id)),'wb'), protocol=4)


        except Exception as e:
            print("Error in training model was: %s" %str(e))     

    if is_error:
        print(f'Usage: python {sys.argv[0]} profile_train_model_xxxx.txt ')    


if __name__ == "__main__": 
    main()