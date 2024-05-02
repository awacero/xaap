class DetectionXaap:
    """
    This class serves as container for storing detection information.
    Defines an ordering based on start time, end time and trace id.

    :param trace_id: Id of the trace the detection was generated from
    :type trace_id: str
    :param start_time: Onset time of the detection
    :type start_time: UTCDateTime
    :param end_time: End time of the detection
    :type end_time: UTCDateTime
    :param peak_value: Peak value of the characteristic function for the detection
    :type peak_value: float
    """

    '''
    def __init__(self, trace_id, start_time, end_time, peak_value=None, pick=None):
        self.trace_id = trace_id
        self.start_time = start_time
        self.end_time = end_time
        self.peak_value = peak_value
        self.pick_object = pick
    '''

    def __init__(self,pick_detection,stream,padding_start, padding_end):

        self.trace_id = stream[0].get_id()
        self.start_time = pick_detection.start_time - padding_start
        self.end_time = pick_detection.end_time + padding_end
        self.pick_detection = pick_detection



    def __lt__(self, other):
        """
        Compares start time, end time and trace id in this order.
        """
        if self.start_time < other.start_time:
            return True
        if self.end_time < other.end_time:
            return True
        if self.trace_id < other.trace_id:
            return True
        return False

    def __str__(self):
        parts = [self.trace_id, str(self.start_time), str(self.end_time)]

        return "\t".join(parts)

