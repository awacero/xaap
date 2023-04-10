 
import multiprocessing
from get_mseed_data import get_mseed_utils as gmutils
from get_mseed_data import get_mseed
import pandas as pd
from obspy import UTCDateTime 
from multiprocessing import Pool
from aaa_features.features import FeatureVector 
from datetime import datetime
import sys
import json

###DETECTAR LA RED DE LA ESTACION USANDO EL ARCHIVO STATIONS.JSON 
## CREAR LAS CARPETAS FEATURES  SI NO EXISTEN 
def get_features(sipass_row,mseed_client_id,client,network,location,features,output_feature_file):

    day_month_year  = sipass_row['FECHA']
    day,month,year = day_month_year.split("/")
    month = "%02d" %int(month)
    day = "%02d" %int(day)

    hour_minute_second = sipass_row['HORA']
    hour, minute, second = hour_minute_second.split(":")
    hour = "%02d" %int(hour)
    minute = "%02d" %int(minute)
    second = "%02d" %int(second)


    if second == 60:
        second = 0
        minute +=1

    start_time = UTCDateTime("%s-%s-%s %s:%s:%s" %(year,month,day,hour,minute,second) )
    end_time = UTCDateTime(start_time) + int(sipass_row['CODA'])

    station = sipass_row['ESTACION']
    channel = sipass_row['COMPONENTE']
    event_type = sipass_row['TIPO']

    file_code = "%s.%s.%s.%s.%s" %(network,station,location,channel,start_time.strftime("%Y.%m.%d.%H%M%S")) 
    print(file_code)
    try:
        temp_stream=get_mseed.get_stream(mseed_client_id,client,network,station,location,channel,start_time=start_time,end_time=end_time)
        fs = temp_stream[0].stats['sampling_rate']
        features.compute(temp_stream[0].data,fs)
        features_string = ', '.join(map(str,features.featuresValues)) 
        row = "%s, %s , %s \n"% (file_code,event_type, features_string)
        #print(row)
        feature_file = open(output_feature_file,'a+')
        feature_file.write(row)
        feature_file.flush()
    except Exception as e:
        print("Error in get_stream or write for %s, error was: %s" %(file_code, str(e)))        
        return 0
    print(start_time,end_time)


def callback_function(res):
    
    #print("Result of multiprocess: %s" %res)
    print("")




def main():

    is_error = False

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
            cores = int(run_param['ENVIRONMENT']['cores'])
            sipass_db_file = run_param['ENVIRONMENT']['sipass_db_file']
            features_folder = run_param['ENVIRONMENT']['features_folder']

            network = run_param['STATIONS_INFO']['network']
            location = run_param['STATIONS_INFO']['location']
            if location=='None':
                location = ""
            
            mseed_client_id = run_param['MSEED_SERVER']['mseed_client_id']
            mseed_server_config_file = run_param['MSEED_SERVER']['mseed_server_config_file']

            feature_config = {"features_file":"%s" %run_param['FEATURES_CONFIG']['features_file'],
                                "domains":"%s" %run_param['FEATURES_CONFIG']['domains']}
            
        except Exception as e:
            raise Exception("Error reading parameters file: %s" %str(e))

        try:
            mseed_server_param = gmutils.read_config_file(mseed_server_config_file)
            client = get_mseed.choose_service(mseed_server_param[mseed_client_id])
        except Exception as e:
            raise Exception("Error connecting to MSEED server : %s" %str(e))

        try:
            sipass_data  = pd.read_csv(sipass_db_file,sep=';')
        except Exception as e:
            raise Exception("Error reading SIPASS file : %s" %str(e))
        
        try:        
            output_feature_file = "%s/features_%s_%s.csv" %(features_folder,volcano,unique_id)  

        except Exception as e:
            raise Exception("Error creating feature file : %s" %str(e))

        
        if len(sys.argv)==2:
            start_row = 0
            print(f"####{sipass_data.size}")
            end_row = int(sipass_data.size)
        else:
            start_row = int(sys.argv[2])
            end_row = int(sys.argv[3])

        features = FeatureVector(feature_config, verbatim=2)

        pool = multiprocessing.get_context('spawn').Pool(processes=cores)
        #pool = Pool(processes=cores)

        results = [ pool.apply_async(get_features,args = ([row,mseed_client_id,client,network,location,
                                                            features,output_feature_file]),callback=callback_function) for i,row in sipass_data.iloc[start_row:end_row].iterrows()]

        for i,res in enumerate(results):
            try:
                
                print("#%s: Result of multiprocess: %s" %(i,res.get(timeout = 60)))
            except Exception as e:
                print("Error in result %s was %s" %(i,str(e)))

        pool.close()
        pool.terminate()

    if is_error:
        print(f'Usage: python {sys.argv[0]} configuration_file.txt [start] [end] ')    


if __name__ == "__main__": 
    main()
