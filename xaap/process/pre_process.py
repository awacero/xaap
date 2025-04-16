

from obspy.signal import filter  

import logging
logger = logging.getLogger(__name__)


def pre_process_stream(xaap_config, volcan_stream):

    processed_stream = None
    processed_stream = volcan_stream.copy()
    logger.info(f"pre_process_stream start for:  {volcan_stream}")

    if bool(xaap_config.detrend) == True:
        try:
            logger.info("detrend the stream")
            processed_stream.detrend("demean")
        except Exception as e:
            logger.error(f"Error while detrending: {e}")

    if bool(xaap_config.merge):
        try:
            processed_stream.merge(method=1, fill_value="interpolate",interpolation_samples=0)
            logger.info("Stream merged: %s" %processed_stream)
        
        except Exception as e:
            logger.error(f"Error at pre_process_stream(), merge was : {e}")

    if bool(xaap_config.filter):
        for i,trace in enumerate(processed_stream): 
            try:
                times = trace.times(type="timestamp")
                sampling_rate = trace.stats.sampling_rate

                f_a = float(xaap_config.filter_freq_a)
                f_b = float(xaap_config.filter_freq_b)
                filter_type = xaap_config.filter_type

                if filter_type == 'highpass':
                    logger.info("highpass selected")
                    temp_data = filter.highpass(trace.data,f_a,sampling_rate,4)
                elif filter_type == 'bandpass':
                    logger.info("bandpass selected")
                    temp_data = filter.bandpass(trace.data,f_a,f_b,sampling_rate,4)
                elif filter_type == 'lowpass':
                    logger.info("lowpass selected")
                    temp_data = filter.lowpass(trace.data,f_a,sampling_rate,4)

                processed_stream[i].data = temp_data
                logger.info("Stream filtered: %s" %processed_stream[i])

            except Exception as e:
                logger.error("Error at pre_process_stream() was : %s" %str(e))

    return processed_stream