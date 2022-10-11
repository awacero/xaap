import xaap_cli
import pytest

filter_json2 = {
    "name": "STATION_A",
    "type": "None",
    "frequency_1": "None",
    "frequency_2": "None",
    "order": "None"
}

filter_json = {'filter_0': {'name': 'rms', 'type': 'None', 'frequency_1': 'None', 'frequency_2': 'None', 'order': 'None'}, 'Freq_A': {'name': 'banda1', 'type': 'bandpass', 'frequency_1': '0.5', 'frequency_2': '0.1'}, 'Freq_B': {'name': 'banda2', 'type': 'bandpass', 'frequency_1': '2', 'frequency_2': '8'}, 'filter_4': {'name': 'banda4', 'type': 'highpass', 'frequency_1': '10.0', 'frequency_2': 'None', 'order': '1', 'zerophase': 'True'}}



def test_create_filter_list():

    filter_test = xaap_cli.create_filter_list(filter_json)

    print(filter_test)
