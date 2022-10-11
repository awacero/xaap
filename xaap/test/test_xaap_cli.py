from ast import Import
import xaap_cli
import pytest

filter_json = {
    "name":"STATION_A",
    "type":"None",
    "frequency_1":"None",
    "frequency_2":"None",
    "order":"None"
            }


def test_create_filter_list():

    filter_test = xaap_cli.create_filter_list(filter_json)

    print(filter_test)
