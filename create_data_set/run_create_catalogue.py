import os
from matplotlib import pyplot as plt
from pathlib import Path
import obspy
from obspy.clients.fdsn.header import FDSNException, FDSNNoDataException

import seisbench.data as sbd
import seisbench.util as sbu
from xaap.configuration.xaap_configuration import configure_logging
import argparse
from get_mseed_data import get_mseed_utils as gmutils
from get_mseed_data import get_mseed

#####
#Adapted from https://colab.research.google.com/github/seisbench/seisbench/blob/main/examples/03b_creating_a_dataset.ipynb
#####

def get_event_from_fdsn(
    fdsn_client,
    start_time,
    end_time,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    eventtype: str = "volcanic eruption",
    includearrivals: bool = True,
):
    """
    Retrieves an event catalogue from an FDSN client based on the given criteria.

    Args:
        fdsn_client (Client): The FDSN client to query.
        start_time (UTCDateTime): Start time of the query.
        end_time (UTCDateTime): End time of the query.
        min_lat (float): Minimum latitude of the region.
        max_lat (float): Maximum latitude of the region.
        min_lon (float): Minimum longitude of the region.
        max_lon (float): Maximum longitude of the region.
        eventtype (str, optional): Type of event to query. Defaults to 'volcanic eruption'.
        includearrivals (bool, optional): Whether to include arrival information. Defaults to True.

    Returns:
        obspy.core.event.Catalog: A catalogue of events that match the query.

    Raises:
        FDSNNoDataException: If no events match the query.
        Exception: If any other error occurs during the query.
    """

    try:
        logger.info(
            f"Querying FDSN client for events: "
            f"start_time={start_time}, end_time={end_time}, "
            f"min_lat={min_lat}, max_lat={max_lat}, "
            f"min_lon={min_lon}, max_lon={max_lon}, "
            f"eventtype={eventtype}, includearrivals={includearrivals}"
        )
        
        # Query the FDSN client for events
        catalogue = fdsn_client.get_events(
            starttime=start_time,
            endtime=end_time,
            minlatitude=min_lat,
            maxlatitude=max_lat,
            minlongitude=min_lon,
            maxlongitude=max_lon,
            eventtype=eventtype,
            includearrivals=includearrivals,
        )
        
        logger.info(f"Successfully retrieved {len(catalogue)} events.")
        return catalogue

    except FDSNNoDataException as e:
        logger.warning(f"No events found for the given criteria: {e}")
        raise FDSNNoDataException(f"No events found for the specified parameters.") from e

    except Exception as e:
        logger.error(f"Unexpected error while querying FDSN client: {e}", exc_info=True)
        raise Exception(f"Error getting catalogue from FDSN client: {e}") from e
 
    

def get_event_params(event, split_date):

    """
    Extracts event parameters from an event object and categorizes it into a data split.

    Args:
        event (obspy.core.event.Event): The event object containing metadata.
        split_date (str): The date to categorize events into 'train', 'dev', or 'test'.

    Returns:
        dict: A dictionary containing event parameters and their associated values.
    """
    try:
        origin = event.preferred_origin()
        mag = event.preferred_magnitude()
        source_id = str(event.resource_id)

        # Compute source depth uncertainty in km
        depth_uncertainty = origin.depth_errors.get("uncertainty", None) if origin.depth_errors else None
        source_depth_uncertainty_km = (
            float(depth_uncertainty) / 1e3
            if isinstance(depth_uncertainty, (float, int)) or 
            (isinstance(depth_uncertainty, str) and depth_uncertainty.replace('.', '', 1).isdigit())
            else None
        )

        # Base event parameters
        event_params = {
            "source_id": source_id,
            "source_origin_time": str(origin.time),
            "source_origin_uncertainty_sec": origin.time_errors.get("uncertainty", None) if origin.time_errors else None,
            "source_latitude_deg": origin.latitude,
            "source_latitude_uncertainty_km": origin.latitude_errors.get("uncertainty", None) if origin.latitude_errors else None,
            "source_longitude_deg": origin.longitude,
            "source_longitude_uncertainty_km": origin.longitude_errors.get("uncertainty", None) if origin.longitude_errors else None,
            "source_depth_km": origin.depth / 1e3 if origin.depth else None,
            "source_depth_uncertainty_km": source_depth_uncertainty_km,
        }

        # Add magnitude-related parameters if available
        if mag:
            event_params.update({
                "source_magnitude": mag.mag,
                "source_magnitude_uncertainty": mag.mag_errors.get("uncertainty", None) if mag.mag_errors else None,
                "source_magnitude_type": mag.magnitude_type,
                "source_magnitude_author": mag.creation_info.agency_id if mag.creation_info else None,
            })



        # Determine the data split
        origin_time_str = str(origin.time)
        if origin_time_str < split_date:
            split = "train"
        elif origin_time_str == split_date:
            split = "dev"
        else:
            split = "test"

        event_params["split"] = split

        return event_params


    except AttributeError as e:
        logger.error(f"Missing attribute in event object: {e}")
        raise ValueError(f"Invalid event object: {e}")
    except Exception as e:
        logger.error(f"Unexpected error while extracting event parameters: {e}", exc_info=True)
        raise Exception(f"Error processing event parameters: {e}")



def get_trace_params(pick):
    """
    Extracts trace parameters from a pick object.

    Args:
        pick (obspy.core.event.Pick): The pick object containing waveform metadata.

    Returns:
        dict: A dictionary containing trace parameters such as station and network codes.

    Raises:
        ValueError: If the pick object or its waveform_id attribute is missing required fields.
    """

    try:
        waveform_id = pick.waveform_id
        if not waveform_id:
            raise ValueError("Pick object is missing 'waveform_id'.")

        trace_params = {
            "station_network_code": waveform_id.network_code or None,
            "station_code": waveform_id.station_code or None,
            "trace_channel": waveform_id.channel_code[:2] if waveform_id.channel_code else None,
            "station_location_code": waveform_id.location_code or None,
        }

        # Optional logging for debugging
        logger.debug(f"Extracted trace parameters: {trace_params}")

        return trace_params

    except AttributeError as e:
        logger.error(f"Error extracting trace parameters: {e}")
        raise ValueError(f"Invalid pick object: {e}") from e




import obspy
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNException

def get_waveforms(client, pick, trace_params, time_before=60, time_after=60):
    """
    Fetches waveform data from an FDSN client for a given pick and trace parameters.

    Args:
        client (obspy.clients.fdsn.Client): The FDSN client to fetch waveform data.
        pick (obspy.core.event.Pick): The pick object containing time information.
        trace_params (dict): Dictionary containing trace parameters, including:
            - "station_network_code": Network code of the station.
            - "station_code": Station code.
            - "trace_channel": Two-character channel prefix.
        time_before (float, optional): Time in seconds before the pick time to include in the waveform. Defaults to 60.
        time_after (float, optional): Time in seconds after the pick time to include in the waveform. Defaults to 60.

    Returns:
        obspy.Stream: The waveform data retrieved for the given parameters.
    """
    try:
        # Calculate start and end times for waveform extraction
        t_start = pick.time - time_before
        t_end = pick.time + time_after

        # Log the request details
        logger.info(
            f"Fetching waveforms: network={trace_params['station_network_code']}, "
            f"station={trace_params['station_code']}, channel={trace_params['trace_channel']}*, "
            f"starttime={t_start}, endtime={t_end}"
        )

        # Fetch waveforms
        waveforms = client.get_waveforms(
            network=trace_params["station_network_code"],
            station=trace_params["station_code"],
            location=trace_params.get("station_location_code", "*"),  # Default to wildcard if location is missing
            channel=f"{trace_params['trace_channel']}*",             # Use wildcard for channel
            starttime=t_start,
            endtime=t_end,
        )

        logger.info(f"Successfully fetched waveforms with {len(waveforms)} traces.")
        return waveforms

    except FDSNException as e:
        logger.warning(f"No waveforms found: {e}")
        return obspy.Stream()  # Return an empty stream

    except KeyError as e:
        logger.error(f"Missing required trace parameter: {e}")
        raise ValueError(f"Invalid trace parameters: {e}") from e

    except Exception as e:
        logger.error(f"Unexpected error while fetching waveforms: {e}", exc_info=True)
        raise Exception(f"Error fetching waveforms: {e}") from e


def load_configuration(config_path):
    """Load and validate the configuration file."""
    try:
        logger.info(f"Validating configuration file: {config_path}")
        file_path = gmutils.check_file(config_path)
        logger.info(f"Configuration file found: {file_path}")
        return gmutils.read_parameters(file_path)
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration file: {e}")
        raise


def create_client(client_type, client_id, config_file):
    """Create a client (MSEED or FDSN) based on configuration."""
    try:
        server_config = gmutils.read_config_file(os.path.expandvars(config_file))
        return get_mseed.choose_service(server_config[client_id])
    except Exception as e:
        logger.error(f"Error creating {client_type} client: {e}")
        raise


def fetch_catalogue(fdsn_client, params):
    """Fetch event catalogue from FDSN client."""
    try:
        start_time = params['start_time']
        end_time = params['end_time']
        min_lat, max_lat = float(params['min_lat']), float(params['max_lat'])
        min_lon, max_lon = float(params['min_lon']), float(params['max_lon'])
        split_date = params['split_date']

        logger.info("Fetching event catalogue...")
        catalogue = get_event_from_fdsn(
            fdsn_client, start_time, end_time, min_lat, max_lat, min_lon, max_lon
        )
        logger.info(f"Fetched {len(catalogue)} events.")
        return catalogue, split_date
    except Exception as e:
        logger.error(f"Error fetching event catalogue: {e}")
        raise


def process_catalogue(catalogue, fdsn_client, split_date, base_path):
    """Process the event catalogue, fetch waveforms, and write data."""
    metadata_path = base_path / "metadata.csv"
    waveforms_path = base_path / "waveforms.hdf5"

    logger.info("Processing catalogue...")
    with sbd.WaveformDataWriter(metadata_path, waveforms_path) as writer:
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
                    logger.info(f"No waveform data available for pick: {pick}")
                    continue

                sampling_rate = waveforms[0].stats.sampling_rate
                if not all(trace.stats.sampling_rate == sampling_rate for trace in waveforms):
                    logger.warning(f"Inconsistent sampling rates in waveforms for pick: {pick}")
                    continue

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

    logger.info("Catalogue processing complete.")


def main(args):
    """Main function to execute the program."""
    try:
        run_param = load_configuration(args.detection_train_config)

        # Create MSEED and FDSN clients
        mseed_client = create_client("MSEED", run_param['mseed']['client_id'], run_param['mseed']['server_config_file'])
        fdsn_client = create_client("FDSN", run_param['fdsn']['client_id'], run_param['fdsn']['server_config_file'])

        # Fetch event catalogue
        catalogue, split_date = fetch_catalogue(fdsn_client, run_param['catalogue'])

        # Process catalogue
        base_path = Path(".")
        process_catalogue(catalogue, fdsn_client, split_date, base_path)

        # Load dataset for training/testing
        data = sbd.WaveformDataset(base_path, sampling_rate=100)
        logger.info(f"Training examples: {len(data.train())}")
        logger.info(f"Development examples: {len(data.dev())}")
        logger.info(f"Test examples: {len(data.test())}")

    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        raise Exception(f"Fatal error in main: {e}")






def main_old(args):

    """Load parameters from plain text config file"""

    '''
    event_parameters = get_event_params(catalogue[0])

    print(event_parameters)

    trace_parameters = get_trace_params(catalogue[0].picks[0])

    print(trace_parameters)

    '''
    configuration_file = args.detection_train_config

    # Check if the configuration file exists
    try:
        logger.info(f"Validating configuration file: {configuration_file}")
        file_path = gmutils.check_file(configuration_file)
        logger.info(f"Configuration file found: {file_path}")
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {configuration_file}")
        raise FileNotFoundError(f"Configuration file does not exist: {configuration_file}") from e
    except Exception as e:
        logger.error(f"Unexpected error while checking configuration file: {e}")
        raise Exception(f"Error checking configuration file: {e}") from e

    # Read and parse the configuration parameters
    try:
        logger.info(f"Reading configuration parameters from file: {configuration_file}")
        run_param = gmutils.read_parameters(configuration_file)
        logger.info(f"Configuration parameters successfully loaded.")
    except ValueError as e:
        logger.error(f"Invalid configuration format in file: {configuration_file}. Error: {e}")
        raise ValueError(f"Invalid configuration format: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error while reading configuration parameters: {e}")
        raise Exception(f"Error reading configuration file: {e}") from e


    try:
        
        mseed_id = run_param['mseed']['client_id']        
        mseed_server_config_file = os.path.expandvars(run_param['mseed']['server_config_file'])
        mseed_server_param = gmutils.read_config_file(mseed_server_config_file)
 
        mseed_client = get_mseed.choose_service(mseed_server_param[mseed_id])

    except Exception as e:
        logger.error(f"Error getting parameters: {e}")
        raise Exception(f"Error getting parameters: {e}")


    try:
        
        fdsn_id = run_param['fdsn']['client_id']        
        fdsn_server_config_file = os.path.expandvars(run_param['fdsn']['server_config_file'])
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


    
    pick = catalogue[0].picks[1]  
    print(pick)
    trace_params = get_trace_params(pick)
    waveform = get_waveforms(fdsn_client, pick, trace_params)

    print("LLEGAAAA")

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

                print("####$$$$")
                print(waveforms)
                
                if len(waveforms) == 0:
                    logger.info("# No waveform data available")

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