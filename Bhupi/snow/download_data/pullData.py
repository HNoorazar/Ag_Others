# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import zeep
import simplejson

# %%
__file__ = "/Users/hn/Documents/01_research_data/Bhupi/snow/00/pull_Data_NWCC/willBeDropped/"

# %%
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 09:42:48 2021

@author: Beau.Uriona
"""

import sys
from os import path, makedirs, listdir
import zeep
import pandas as pd
import simplejson

SENSOR = "STO"

THIS_DIR = path.dirname(path.abspath(__file__))
DATA_DIR = path.join(THIS_DIR, "data_export")
makedirs(DATA_DIR, exist_ok=True)
EXPORT_DIR = path.join(DATA_DIR, SENSOR)
makedirs(EXPORT_DIR, exist_ok=True)
MINI_DIR = path.join(DATA_DIR, f"{SENSOR}_mini")
makedirs(MINI_DIR, exist_ok=True)


def minify_dir(in_dir, out_dir):
    in_files = listdir(in_dir)
    in_files[:] = [i for i in in_files if i.endswith(".json")]
    for in_file in in_files:
        in_path = path.join(in_dir, in_file)
        out_path = path.join(out_dir, in_file)
        with open(in_path, "r") as j:
            in_json = simplejson.load(j)
        with open(out_path, "w") as j:
            simplejson.dump(in_json, j)

sensor = SENSOR
ordinal = 1
duration = "DAILY"
get_flags = True
begin_date = "1900-10-01"
end_date = "2021-12-28"
return_feb29 = False
url = "http://wcc.sc.egov.usda.gov/awdbWebService/services?wsdl"

transport = zeep.transports.Transport()
client = zeep.Client(wsdl=url, transport=transport)
awdb = client.service

stations = awdb.getStations(networkCds="SCAN", elementCds=sensor, logicalAnd=True)
# stations= sntl + scan

sensor_stations = []
for i, station in enumerate(stations[:2]):
    print(f"Getting elements for station {i + 1} of {len(stations)}...")
    elements = awdb.getStationElements(stationTriplet=station)
    elements = zeep.helpers.serialize_object(elements, target_cls=dict)
    """TODO: might want to add a filter and/or parse out the 
       depths you want and durations you want so you dont pull all of them (esp durations)
    """
    has_sensor = any([True if i["elementCd"] == sensor else False for i in elements])
    if has_sensor:
        sensor_stations.append(
            (station, [i for i in elements if i["elementCd"] == sensor])
        )

print(f"Getting metadata for {len(sensor_stations)} stations...")
meta = awdb.getStationMetadataMultiple(stationTriplets=[i[0] for i in sensor_stations])
meta = zeep.helpers.serialize_object(meta, target_cls=dict)
df = pd.DataFrame().from_dict(meta)
meta_export_path = path.join(EXPORT_DIR, "meta.json")
print(f"  Writting json to {meta_export_path}...")
df.to_json(meta_export_path, indent=4, orient="records")


for i, station_elements in enumerate(sensor_stations):
    trip = station_elements[0]
    element_meta = station_elements[1]
    print(f"Getting data for station {i + 1} of {len(sensor_stations)}...")
    for station_element in element_meta:
        if duration not in ["HOURLY"]:
            element_cd = station_element["elementCd"]
            height_depth = station_element["heightDepth"]
            height = None
            if height_depth:
                height = str(height_depth["value"]).replace("-", "")
            sensor_data = awdb.getData(
                stationTriplets=station,
                elementCd=element_cd,
                heightDepth=height_depth,
                ordinal=ordinal,
                duration=duration,
                getFlags=get_flags,
                beginDate=begin_date,
                endDate=end_date,
                alwaysReturnDailyFeb29=return_feb29,
            )
        else:
            sensor_data = awdb.getHourlyData(
                stationTriplets=station,
                elementCd=element_cd,
                heightDepth=height_depth,
                ordinal=ordinal,
                beginDate=begin_date,
                endDate=end_date,
            )
        sensor_data = zeep.helpers.serialize_object(sensor_data, target_cls=dict)
        df = pd.DataFrame().from_dict(sensor_data)
        filename = f"{station.replace(':', '_')}_{element_cd}"
        if height:
            filename = f"{filename}_{height}"
            """TODO: may want to change the export name and/or directory structure
            """
        data_export_path = path.join(EXPORT_DIR, f"{filename}.json") 
        print(f"  Writting json to {data_export_path}...")
        df.to_json( # TODO: you could make this to_json or to_sql if that's better for your use case.
            data_export_path,
            indent=4,
            orient="records",
            default_handler=simplejson.dump,
        )


# %%
stations

# %%
