

from obspy.signal import filter  

import logging
logger = logging.getLogger(__name__)


def pre_process_stream(xaap_config, volcan_stream):

    processed_stream = None
    logger.info("self.processed_stream start :%s" %processed_stream)
    try:
        #self.processed_stream = self.stream.copy()
        processed_stream = volcan_stream.copy()
        processed_stream.merge(method=1, fill_value="interpolate",interpolation_samples=0)
        logger.info("Stream merged: %s" %processed_stream)

        times = processed_stream[0].times(type="timestamp")
        sampling_rate = processed_stream[0].stats.sampling_rate

        f_a = float(xaap_config.filter_freq_a)
        f_b = float(xaap_config.filter_freq_b)
        filter_type = xaap_config.filter_type

        if filter_type == 'highpass':
            logger.info("highpass selected")
            temp_data = filter.highpass(processed_stream[0].data,f_a,sampling_rate,4)
        elif filter_type == 'bandpass':
            logger.info("bandpass selected")
            temp_data = filter.bandpass(processed_stream[0].data,f_a,f_b,sampling_rate,4)
        elif filter_type == 'lowpass':
            logger.info("lowpass selected")
            temp_data = filter.lowpass(processed_stream[0].data,f_a,sampling_rate,4)

        #self.processed_stream[0].data = temp_data
        volcan_stream[0].data = temp_data

        logger.info("Stream filtered: %s" %processed_stream)

        return volcan_stream

    except Exception as e:
        logger.error("Error at pre_process_stream() was : %s" %str(e))
