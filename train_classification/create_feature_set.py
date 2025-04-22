 
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
from matplotlib import pyplot as plt
from pathlib import Path

###DETECTAR LA RED DE LA ESTACION USANDO EL ARCHIVO STATIONS.JSON 
## CREAR LAS CARPETAS FEATURES  SI NO EXISTEN 
def get_features(sipass_row,mseed_client_id,client,network,location,features,output_feature_file):

    """
    This function retrieves a waveform data stream from the given parameters, computes the specified features from it,
    and writes the feature values to a file.

    Args:
        sipass_row (dict): A dictionary containing the SIPASS row information, including 'FECHA', 'HORA',
            'CODA', 'ESTACION', 'COMPONENTE', and 'TIPO'.
        mseed_client_id (str): The ID for the MiniSEED client.
        client (str): The client name.
        network (str): The network code.
        location (str): The location code.
        features (obj): The feature object for computing waveform features.
        output_feature_file (str): The file path for writing the computed features.

    Returns:
        bool: True if the features were successfully computed and written, False otherwise.
    """

    day_month_year  = sipass_row["FECHA"]
    day,month,year = day_month_year.split("/")
    month = f"{int(month):02d}"
    day = f"{int(day):02d}"
    
    hour_minute_second = sipass_row['HORA']
    hour, minute, second = hour_minute_second.split(":")
    hour = f"{int(hour):02d}"
    minute = f"{int(minute):02d}"
    second = f"{int(second):02d}"


    if second == 60:
        second = "00"
        minute = f"{int(minute)+1:02d}"


    #start_time = UTCDateTime("%s-%s-%s %s:%s:%s" %(year,month,day,hour,minute,second) )
    start_time = UTCDateTime(f"{year}-{month}-{day} {hour}:{minute}:{second}")
    end_time = UTCDateTime(start_time) + int(sipass_row['CODA'])

    station = sipass_row['ESTACION']
    channel = sipass_row['COMPONENTE']
    event_type = sipass_row['TIPO']


    file_code = f"{network}.{station}.{location}.{channel}.{start_time.strftime('%Y.%m.%d.%H%M%S')}"
    print(file_code)
    try:
        # Get the waveform data stream from the MiniSEED client
        temp_stream=get_mseed.get_stream(mseed_client_id,client,network,station,location,channel,start_time=start_time,end_time=end_time)
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
        return True
    except Exception as e:
        print(f"Error in get_stream or write for {file_code}, error was: {e}")
        return False
    print(start_time,end_time)





def callback_function(result):
    
    if result:
        print(f"Result of multiprocess: {result}")
    



def main():

    if len(sys.argv) < 2:
        print(f"Invalid number of arguments. Usage: python {sys.argv[0]} configuration_file.txt [start] [end]")
        sys.exit(1)

    unique_id = datetime.now().strftime("%Y%m%d%H%M%S")
    try:
        run_param = gmutils.read_parameters(sys.argv[1])
    except Exception as e:
        raise Exception(f"Error reading configuration file: {e}")

    try:
        volcano = run_param['ENVIRONMENT']['volcano']
        cores = int(run_param['ENVIRONMENT']['cores'])
        sipass_db_file = run_param['ENVIRONMENT']['sipass_db_file']
        features_folder = Path(run_param['ENVIRONMENT']['features_folder'])

        network = run_param['STATIONS_INFO']['network']
        location = run_param['STATIONS_INFO']['location']
        if location == 'None':
            location = ""

        mseed_client_id = run_param['MSEED_SERVER']['mseed_client_id']
        mseed_server_config_file = run_param['MSEED_SERVER']['mseed_server_config_file']

        feature_config = {"features_file": run_param['FEATURES_CONFIG']['features_file'],
                            "domains": run_param['FEATURES_CONFIG']['domains']}

    except Exception as e:
        raise Exception(f"Error reading parameters file: {e}")

    try:
        mseed_server_param = gmutils.read_config_file(mseed_server_config_file)
        client = get_mseed.choose_service(mseed_server_param[mseed_client_id])
    except Exception as e:
        raise Exception(f"Error connecting to MSEED server: {e}")

    try:
        sipass_data = pd.read_csv(sipass_db_file, sep=None)
    except Exception as e:
        raise Exception(f"Error reading SIPASS file: {e}")

    try:
        output_feature_file = features_folder/f"features_{volcano}_{unique_id}.csv"

    except Exception as e:
        raise Exception(f"Error creating feature file: {e}")

    if len(sys.argv) == 2:
        start_row = 0
        print(f"####{sipass_data.size}")
        end_row = int(sipass_data.size)
        sipass_data_sampled = sipass_data.copy()
    else:
        print(f"Select random rows")
        start_row = int(sys.argv[2])
        end_row = int(sys.argv[3])
        n_total = len(sipass_data)
        fraction = end_row/n_total
        sipass_data_sampled = sipass_data.groupby("TIPO",group_keys=False).apply(lambda x: x.sample(frac=fraction, random_state=42))

    print(sipass_data_sampled["TIPO"].value_counts())
    """Plot and histogram using TIPO to show the more frequent events"""
    plt.figure(figsize=(10, 6))
    sipass_data_sampled["TIPO"].value_counts().plot(kind="bar", color="skyblue", edgecolor="black")

    plt.savefig(features_folder/f"{volcano}_events_freq_{unique_id}.png")

    print("Succesfully created the events_frequency plot")

    features = FeatureVector(feature_config, verbatim=2)

    pool = multiprocessing.get_context('spawn').Pool(processes=cores)

    results = [pool.apply_async(get_features, 
                args=([row, mseed_client_id, client, network, location,
                features, output_feature_file]), callback=callback_function) 
                for i, row in sipass_data_sampled.iloc[start_row:end_row].iterrows()]

    for i, res in enumerate(results):
        try:
            print(f"#{i}: Result of multiprocess: {res.get(timeout=60)}")
        except Exception as e:
            print(f"Error in result {i} was {e}")

    pool.close()
    pool.terminate()





if __name__ == "__main__": 
    main()
