 
from get_mseed_data import get_mseed_utils as gmutils
from get_mseed_data import get_mseed
import pandas as pd
from obspy import UTCDateTime 
from multiprocessing import Pool
from aaa_features.features import FeatureVector 

CORES = 10
network = "EC"
location = ""
mseed_client_id="ARCLINK"
mseed_server_config_file = "./server_configuration.json"
tags  = pd.read_csv("./base_tungurahua_2011_2021.csv",sep=';')
output_feature_file = "./feature_tungurahua_igepn_2011_2021.csv"



mseed_server_param = gmutils.read_config_file(mseed_server_config_file)
client = get_mseed.choose_service(mseed_server_param[mseed_client_id])
feature_file = open(output_feature_file,'a+')

config = {'features_file':'./config/features/features_00.json', 'domains':'time spectral cepstral'}


features = FeatureVector(config, verbatim=2)

def get_features(sipass_row):

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
        feature_file.write(row)
    except Exception as e:
        print("Error in get_stream or write: %s" %str(e))        

    print(start_time,end_time)


def callback_function(res):
    
    print("Result of multiprocess: %s" %res)


pool = Pool(processes=CORES)

#for i,row in tags.iloc[1:100].iterrows():


results = [ pool.apply_async(get_features,args = ([row]),callback=callback_function) for i,row in tags.iloc[1:10000].iterrows()]

for res in results:
    print("Result of multiprocess: %s" %res.get())

pool.close()


