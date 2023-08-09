from obspy.core import Stream, read
from obspy.signal.trigger import coincidence_trigger
from pprint import pprint


from get_mseed_data import get_mseed_utils as gmutils
from get_mseed_data import get_mseed
from obspy import read, UTCDateTime


volcanes = { "COTOPAXI":{"COTOPAXI":["ANGU","BMAS","BNAS"]},
            "TUNGURAHUA":{"TUNGURAHUA":["BRTU","BMAS","BPAT","ARA2"]}}
stations = {
            "ANGU":{ "net":"EC", "cod":"ANGU", "loc":["",], "cha":["SHZ"]},
            "ARA2":{ "net":"EC", "cod":"ARA2", "loc":["",], "cha":["SHZ"]},
            "BRTU":{ "net":"EC", "cod":"BRTU", "loc":["",], "cha":["HHZ"]},
            "BMAS":{ "net":"EC", "cod":"BMAS", "loc":["",], "cha":["HHZ"]},
            "BPAT":{ "net":"EC", "cod":"BPAT", "loc":["",], "cha":["BHZ"]},
            "BREF":{ "net":"EC", "cod":"BREF", "loc":["",], "cha":["BHZ"]},
            "BVC2":{ "net":"EC", "cod":"BVC2", "loc":["",], "cha":["BHZ"]}
            }

'''
Tungurahua,ARA2,SHZ,02/11/2021,14:11:30,21.6099
Tungurahua,ARA2,SHZ,02/11/2021,11:26:58,36.5
Tungurahua,ARA2,SHZ,30/10/2021,02:40:23,12.49
Tungurahua,ARA2,SHZ,26/10/2021,18:48:01,29.7099
Tungurahua,ARA2,SHZ,25/10/2021,18:02:44,21.69
Tungurahua,ARA2,SHZ,25/10/2021,03:12:54,33.0999
Tungurahua,ARA2,SHZ,24/10/2021,22:54:00,33.0999
Tungurahua,ARA2,SHZ,24/10/2021,07:51:55,21.7
Tungurahua,ARA2,SHZ,22/10/2021,23:50:17,36.5
Tungurahua,ARA2,SHZ,22/10/2021,04:29:31,24

'''

request_date = UTCDateTime("2021-10-25T00:00:00")
window_size = 86400
volcan = "TUNGURAHUA"

mseed_client_id = "ARCLINK"

mseed_server =        {"ARCLINK":{
                "name":"ARCLINK",
                "user":"aaa",
                "server_ip":"192.168.131.2",
                "port":"18001"
        }}

mseed_server_param = gmutils.read_config_file("xaap/config/server_configuration.json")
mseed_client = get_mseed.choose_service(mseed_server_param[mseed_client_id])

volcan_stations = volcanes[volcan][volcan]

volcan_stations_list = []

for station in volcan_stations:
    volcan_stations_list.append(stations[station])

st =Stream()

for station in volcan_stations_list:
    for cha in station['cha']:
        #print(station['net'],station['cod'],station['loc'][0],cha,request_date ,86400)
        mseed_stream=get_mseed.get_stream(mseed_client_id,mseed_client,station['net'],station['cod'],station['loc'][0],cha,start_time=request_date,window_size=window_size)
        #print(mseed_stream)
        if mseed_stream:
            st+=mseed_stream

st.filter('bandpass', freqmin=10, freqmax=20)  # optional prefiltering

from obspy.signal.trigger import coincidence_trigger

st2 = st.copy()

trig = coincidence_trigger("recstalta", 3.5, 1, st2, 3, sta=0.5, lta=10)

from pprint import pprint

for i,t in enumerate(trig):
    print(i,t)

#st2.plot()







''' Revisar las estaciones q se solicitan
    Hay datos?
    Hay gaps? rellenarlos?
    '''

'''
st = Stream()

files = ["BW.UH1..SHZ.D.2010.147.cut.slist.gz",
         "BW.UH2..SHZ.D.2010.147.cut.slist.gz",
         "BW.UH3..SHZ.D.2010.147.cut.slist.gz",
         "BW.UH4..SHZ.D.2010.147.cut.slist.gz"]


##Agregar streams a un stream? 

for filename in files:
    st += read("https://examples.obspy.org/" + filename)
    print(st)

st.filter('bandpass', freqmin=10, freqmax=20)  # optional prefiltering


st2 = st.copy()

trig = coincidence_trigger("recstalta", 3.5, 1, st2, 3, sta=0.5, lta=10)



pprint(trig)

st2 = st.copy()

trig = coincidence_trigger("recstalta", 3.5, 1, st2, 3, sta=0.5, lta=10,  details=True)

pprint(trig[0])
'''
