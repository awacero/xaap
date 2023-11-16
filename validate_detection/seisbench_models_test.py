 
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

import seisbench.models as sbm
from seisbench.util.annotations import Pick, Detection

###DETECTAR LA RED DE LA ESTACION USANDO EL ARCHIVO STATIONS.JSON 
## CREAR LAS CARPETAS FEATURES  SI NO EXISTEN 
def detect_pick_dl(sipass_row,mseed_client_id,client,network,location,sb_model,output_detection_file):



    start_time = UTCDateTime(sipass_row['FechaHora'])
    end_time = UTCDateTime(start_time) + int(sipass_row['Coda'])

    station = sipass_row['Estacion']
    channel = sipass_row['Canal']
    channel = f"{channel[0:2]}*"
    event_type = sipass_row['Tipo']

    print("#####")
    #print(f"Manual,{sipass_row['Estacion']},   {start_time}, {end_time}, {event_type},{channel}")

    file_code = f"{network}.{station}.{location}.{channel}.{start_time.strftime('%Y.%m.%d.%H%M%S')}"
    try:
        # Get the waveform data stream from the MiniSEED client
        temp_stream=get_mseed.get_stream(mseed_client_id,client,network,station,location,channel,start_time=start_time-30,end_time=end_time + 30)
        #print(temp_stream)
    except:
        print(f"Error in get_stream error was: {e}")

    try:

        classify_result = sb_model.classify(temp_stream)
        print(f"TIPO DE OBJETO:  {type(classify_result)}")

        if isinstance(classify_result, list):
            if len(classify_result) > 0:
                if isinstance(classify_result[0],Pick):
                    print("lista de: PICKS")
                    for pick in classify_result:
                        print(pick)
                        coda = pick.end_time - pick.start_time
                        row = f"{pick.start_time},{pick.end_time},{pick.peak_time},{pick.peak_value},{pick.phase},{pick.trace_id},{coda},1\n"
                        print(row)
                        feature_file = open(output_detection_file,'+a')
                        feature_file.write(row)
                        feature_file.flush()

                elif isinstance(classify_result[0],Detection):
                    print("lista de: DETECTIONS")

                else:
                    print("lista vacia")
        elif isinstance(classify_result,tuple):
            print("Tupla con PICKS y DETECTIONS")
            picks,detections = classify_result
            print("PICKS")
            for pick in picks:
                
                print("PICO PICO PICO")
                print(pick)
                coda = pick.end_time - pick.start_time

                row = f"{pick.start_time},{pick.end_time},{pick.peak_time},{pick.peak_value},{pick.phase},{pick.trace_id},{coda},1\n"
                print(row)
                feature_file = open(output_detection_file,'+a')
                feature_file.write(row)
                feature_file.flush()




            print("DETECTIONS")
            for detection in detections:
                coda = detection.end_time - detection.start_time
                row = f"{detection.start_time},{detection.end_time},0,{detection.peak_value},0,{detection.trace_id},{coda},1\n"
                print(row)
                feature_file = open(f"{output_detection_file}_detection.csv",'+a')
                feature_file.write(row)
                feature_file.flush()


        ''' 
        # Get the sampling rate from the waveform data stream
        fs = temp_stream[0].stats['sampling_rate']
        # Compute the specified features from the waveform data and join them together in a string
        features.compute(temp_stream[0].data,fs)
        features_string = ', '.join(map(str,features.featuresValues)) 
        row = f"{file_code}, {event_type}, {features_string}\n"
        #print(row)
        feature_file = open(output_feature_file,'a+')
        feature_file.write(row)
        feature_file.flush()
        '''
        return True
    except Exception as e:
        print(f"Error in DL detection  for {file_code}, error was: {e}")
        return False
    print(start_time,end_time)





def callback_function(result):
    
    if result:
        print("Result of multiprocess: %s" %result)
    



def main():

    is_error = False

    if len(sys.argv) == 1:
        is_error = True

    else:
        unique_id = datetime.now().strftime("%Y%m%d%H%M%S")
        try:
            run_param = gmutils.read_parameters(sys.argv[1])
        except Exception as e:
            raise Exception(f"Error reading configuration file: {e}")

        try:
            volcano = run_param['ENVIRONMENT']['volcano']
            cores = int(run_param['ENVIRONMENT']['cores'])
            sipass_db_file = run_param['ENVIRONMENT']['sipass_db_file']
            output_detection_folder = run_param['ENVIRONMENT']['output_detection_folder']

            network = run_param['STATIONS_INFO']['network']
            location = run_param['STATIONS_INFO']['location']
            if location == 'None':
                location = ""

            mseed_client_id = run_param['MSEED_SERVER']['mseed_client_id']
            mseed_server_config_file = run_param['MSEED_SERVER']['mseed_server_config_file']

            feature_config = {"features_file": run_param['FEATURES_CONFIG']['features_file'],
                              "domains": run_param['FEATURES_CONFIG']['domains']}

            model_name = run_param['MODEL_DEEP_LEARNING']['model_name']
            model_version = run_param['MODEL_DEEP_LEARNING']['model_version']


            print("###")
            print(run_param)

        except Exception as e:
            raise Exception(f"Error reading parameters file: {e}")

        try:
            mseed_server_param = gmutils.read_config_file(mseed_server_config_file)
            client = get_mseed.choose_service(mseed_server_param[mseed_client_id])
        except Exception as e:
            raise Exception(f"Error connecting to MSEED server: {e}")

        try:
            sipass_data = pd.read_csv(sipass_db_file, sep=',')
        except Exception as e:
            raise Exception(f"Error reading SIPASS file: {e}")

        try:
            output_detection_file = f"{output_detection_folder}/_{volcano}_{model_name}_{model_version}_{unique_id}.csv"

        except Exception as e:
            raise Exception(f"Error creating feature file: {e}")

        if len(sys.argv) == 2:
            start_row = 0
            print(f"####{sipass_data.size}")
            end_row = int(sipass_data.size)
        else:
            start_row = int(sys.argv[2])
            end_row = int(sys.argv[3])

        #features = FeatureVector(feature_config, verbatim=2)

        ###sb_model = sbm.EQTransformer.from_pretrained("ethz")

        try:
            sb_model = getattr(sbm,model_name).from_pretrained(model_version)
            

        except Exception as e:
            print(f"Error while creating deep learning model from seisbench: {model_name} and {model_version}: {e}")
            return []


        pool = multiprocessing.get_context('spawn').Pool(processes=cores)

        results = [pool.apply_async(detect_pick_dl, args=([row, mseed_client_id, client, network, location,
                                                         sb_model,output_detection_file]), callback=callback_function) for i, row in sipass_data.iloc[start_row:end_row].iterrows()]

        for i, res in enumerate(results):
            try:
                print(f"#{i}: Result of multiprocess: {res.get(timeout=60)}")
            except Exception as e:
                print(f"Error in result {i} was {e}")

        pool.close()
        pool.terminate()

    if is_error:
        print(f"Usage: python {sys.argv[0]} configuration_file.txt [start] [end]")   





if __name__ == "__main__": 
    main()
