import obspy

from matplotlib import pyplot as plt
from pathlib import Path
import sys
'''
current_dir = Path(__file__).parent.resolve()
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))
'''
import obspy
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNException 


import seisbench
import seisbench.data as sbd
import seisbench.util as sbu
from xaap.configuration.xaap_configuration import configure_logging
import argparse
from get_mseed_data import get_mseed_utils as gmutils
from get_mseed_data import get_mseed

def get_event_from_fdsn(fdsn_client,start_time, end_time, min_lat, max_lat,min_lon, max_lon):

    try:
        catalogue = fdsn_client.get_events(starttime=start_time, endtime=end_time, minlatitude=min_lat, maxlatitude=max_lat, minlongitude=min_lon, maxlongitude=max_lon, eventtype='volcanic eruption' ,includearrivals=True)
        return catalogue
    except Exception as e:
        logger.error(f"Error getting catalogue : {e}" )
        raise Exception(f"Error getting catalogue : {e}" )  
    


def get_event_params(event, split_date):
    origin = event.preferred_origin()
    mag = event.preferred_magnitude()

    source_id = str(event.resource_id)

    event_params = {
        "source_id": source_id,
        "source_origin_time": str(origin.time),
        "source_origin_uncertainty_sec": origin.time_errors["uncertainty"],
        "source_latitude_deg": origin.latitude,
        "source_latitude_uncertainty_km": origin.latitude_errors["uncertainty"],
        "source_longitude_deg": origin.longitude,
        "source_longitude_uncertainty_km": origin.longitude_errors["uncertainty"],
        "source_depth_km": origin.depth / 1e3,

        "source_depth_uncertainty_km": float(origin.depth_errors["uncertainty"]) / 1e3 if isinstance(origin.depth_errors["uncertainty"], (float, int)) or (isinstance(origin.depth_errors["uncertainty"], str) and origin.depth_errors["uncertainty"].replace('.', '', 1).isdigit()) else None

        #"source_depth_uncertainty_km": origin.depth_errors["uncertainty"] / 1e3,
    }

    if mag is not None:
        event_params["source_magnitude"] = mag.mag
        event_params["source_magnitude_uncertainty"] = mag.mag_errors["uncertainty"]
        event_params["source_magnitude_type"] = mag.magnitude_type
        event_params["source_magnitude_author"] = mag.creation_info.agency_id
    
        if str(origin.time) < split_date:
            split = "train"
        elif str(origin.time) < split_date:
            split = "dev"
        else:
            split = "test"
        event_params["split"] = split
    
    return event_params


def get_trace_params(pick):
    net = pick.waveform_id.network_code
    sta = pick.waveform_id.station_code

    trace_params = {
        "station_network_code": net,
        "station_code": sta,
        "trace_channel": pick.waveform_id.channel_code[:2],
        "station_location_code": pick.waveform_id.location_code,
    }

    return trace_params

def get_waveforms(client, pick, trace_params, time_before=60, time_after=60):
    t_start = pick.time - time_before
    t_end = pick.time + time_after
    
    try:
        waveforms = client.get_waveforms(
            network=trace_params["station_network_code"],
            station=trace_params["station_code"],
            location="*",
            channel=f"{trace_params['trace_channel']}*",
            starttime=t_start,
            endtime=t_end,
        )
    except FDSNException:
        # Return empty stream
        waveforms = obspy.Stream()
    
    return waveforms



def main(args):

    """Load parameters from plain text config file"""

    '''
    event_parameters = get_event_params(catalogue[0])

    print(event_parameters)

    trace_parameters = get_trace_params(catalogue[0].picks[0])

    print(trace_parameters)

    '''

    configuration_file = args.detection_train_config
    try:
        logger.info(f"Check if configuration file {configuration_file} exists")
        file_path = gmutils.check_file(configuration_file)
    except Exception as e:
        logger.error(f"Error reading configuration  file: {e}" )
        raise Exception(f"Error reading configuration file: {e}" )

    try:
        run_param = gmutils.read_parameters(configuration_file)
    except Exception as e:
        logger.error(f"Error reading configuration sets in file: {e}")
        raise Exception(f"Error reading configuration file: {e}")

    try:
        
        mseed_id = run_param['mseed']['client_id']        
        mseed_server_config_file = run_param['mseed']['server_config_file']
        mseed_server_param = gmutils.read_config_file(mseed_server_config_file)
        mseed_client = get_mseed.choose_service(mseed_server_param[mseed_id])

    except Exception as e:
        logger.error(f"Error getting parameters: {e}")
        raise Exception(f"Error getting parameters: {e}")


    try:
        
        fdsn_id = run_param['fdsn']['client_id']        
        fdsn_server_config_file = run_param['fdsn']['server_config_file']
        fdsn_server_param = gmutils.read_config_file(fdsn_server_config_file)
        fdsn_client = get_mseed.choose_service(fdsn_server_param[fdsn_id])

    except Exception as e:
        logger.error(f"Error creating FDSN client: {e}")
        raise Exception(f"Error creating FDSN client: {e}")

    try:

        #get parameters from file 
        start_time = run_param['catalogue']['start_time']
        end_time = run_param['catalogue']['end_time']
        split_date = run_param['catalogue']['split_date']
        min_lat = float(run_param['catalogue']['min_lat'])
        max_lat = float(run_param['catalogue']['max_lat'])
        min_lon = float(run_param['catalogue']['min_lon'])
        max_lon = float(run_param['catalogue']['max_lon'])


        catalogue = get_event_from_fdsn(fdsn_client,start_time,end_time,min_lat,max_lat,min_lon, max_lon)

        print(catalogue)
    except Exception as e:
        logger.error(f"Error creating FDSN client: {e}")
        raise Exception(f"Error creating FDSN client: {e}")



    try:
        
        pass
        ## GUARDA EL CATALOGO CON UN NOMBRE UNICO DEPENDIENDO DE INPUT PARAMS
    except Exception as e:
        logger.error(f"Error creating FDSN client: {e}")
        raise Exception(f"Error creating FDSN client: {e}")

    print(fdsn_client)
    print(mseed_client)



    
    '''
    client = Client("http://192.168.137.16:8080",timeout=100)

    t0 = UTCDateTime(2022, 10, 16)
    t1 = t0 + 1 * 24 * 60 * 60  # 6 days
    split_date = "2022-10-20"

    catalogue = client.get_events(starttime=t0, endtime=t1, minlatitude=-0.82, maxlatitude=-0.55, minlongitude=-78.57, maxlongitude=-78.30, eventtype='volcanic eruption' ,includearrivals=True)
    #print(catalogue.__str__(print_all=True))
    '''

    catalogue.write("COTO.xml", format="QUAKEML")

    catalogue.write("COTO.json", format="JSON")


    pick = catalogue[0].picks[5]
    trace_params = get_trace_params(pick)
    waveform = get_waveforms(fdsn_client, pick, trace_params)

    print(waveform)


    base_path = Path(".")
    metadata_path = base_path / "metadata.csv"
    waveforms_path = base_path / "waveforms.hdf5"


    # Iterate over events and picks, write to SeisBench format
    with sbd.WaveformDataWriter(metadata_path, waveforms_path) as writer:
        
        # Define data format
        writer.data_format = {
            "dimension_order": "CW",
            "component_order": "ZNE",
            "measurement": "velocity",
            "unit": "counts",
            "instrument_response": "not restituted",
        }
        
        for event in catalogue:
            event_params = get_event_params(event, split_date)
            for pick in event.picks:
                trace_params = get_trace_params(pick)
                waveforms = get_waveforms(fdsn_client, pick, trace_params)
                
                if len(waveforms) == 0:
                    # No waveform data available
                    continue
            
                sampling_rate = waveforms[0].stats.sampling_rate
                # Check that the traces have the same sampling rate
                assert all(trace.stats.sampling_rate == sampling_rate for trace in waveforms)
                
                actual_t_start, data, _ = sbu.stream_to_array(
                    waveforms,
                    component_order=writer.data_format["component_order"],
                )
                
                trace_params["trace_sampling_rate_hz"] = sampling_rate
                trace_params["trace_start_time"] = str(actual_t_start)
                
                sample = (pick.time - actual_t_start) * sampling_rate
                trace_params[f"trace_{pick.phase_hint}_arrival_sample"] = int(sample)
                trace_params[f"trace_{pick.phase_hint}_status"] = pick.evaluation_mode
                
                writer.add_trace({**event_params, **trace_params}, data)

    data = sbd.WaveformDataset(base_path, sampling_rate=100)


    print("Training examples:", len(data.train()))
    print("Development examples:", len(data.dev()))
    print("Test examples:", len(data.test()))

    print("METADATAAA")
    print(data.metadata) 

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    ax.plot(data.get_waveforms(0).T)
    ##ax.axvline(data.metadata["trace_P1_arrival_sample"].iloc[0], color="k", lw=3)

    plt.show()


if __name__ == '__main__':

    logger = configure_logging()
    logger.info("Logging configurated")

    '''Call the program with arguments or use default values'''
    parser = argparse.ArgumentParser(description='run_create_catalogue will use default configuration found in ./config/detection_training.cfg')
    parser.add_argument("--detection_train_config",type=str, default="./config/detection_training.cfg", help='Text config file for RUN_CREATE_CATALOGUE')
    
    args = parser.parse_args()
    main(args)