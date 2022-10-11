from numpy import isin
import xaap_cli
import pytest
from models.xaap_filter import XaapFilter

filter_json = {
                'STATION_A': {'name': 'STATION_A', 'type': 'None', 'frequency_1': 'None', 'frequency_2': 'None', 'order': 'None'},
                'STATION_B': {'name': 'STATION_B', 'type': 'bandpass', 'frequency_1': '0.5', 'frequency_2': '0.1'}, 
                'STATION_C': {'name': 'STATION_C', 'type': 'bandpass', 'frequency_1': '2', 'frequency_2': '8'}
                 }



def test_create_filter_list():

    filter_test = xaap_cli.create_filter_list(filter_json)
    assert isinstance(filter_test[0],XaapFilter)
    assert filter_test[0].name == "STATION_A"



