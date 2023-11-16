
import obspy

from obspy import read
import sys
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from pyproj import CRS, Transformer
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from gamma.utils import association
import seisbench.models as sbm

sns.set(font_scale=1.2)
sns.set_style("ticks")

HORAS = 1




# Projections
wgs84 = CRS.from_epsg(4326)
# EC 32718
local_crs = CRS.from_epsg(32717)  # SIRGAS-Chile 2016 / UTM zone 19S
transformer = Transformer.from_crs(wgs84, local_crs)


'''
Convertir de utm a lat lon

'''
# Crea un transformador para convertir del sistema UTM 17S (por ejemplo) a WGS84
transformer_utm2geo = Transformer.from_crs("EPSG:32717", "EPSG:4326")

print("INICIO DE PEDIDO DE DATOS")

t0 = UTCDateTime("2023-06-28 04:00:00")
t1 = t0 + HORAS * 60 * 60
url_fdsn_ig = "http://192.168.137.16:8080"

client = Client(url_fdsn_ig)


inv = client.get_stations(network="EC", station="*", location="*", channel="?H?", starttime=t0, endtime=t1)


# t1 = t0 + 24 * 60 * 60   # Full day, requires more memory

#'''
stream = client.get_waveforms(network="EC", station="BREF,CO1V,BVC2,BTAM,BMOR", location="*", channel="?H?", starttime=t0, endtime=t1)
inv = client.get_stations(network="EC", station="*", location="*", channel="?H?", starttime=t0, endtime=t1)
print(stream)
print("DATOS DESCARGADOS")
#stream.write("./cotopaxi_medio_1dia.mseed")

#'''

#stream = read("./cotopaxi_medio_1dia.mseed")


# Gamma
config = {}
config["dims"] = ['x(km)', 'y(km)', 'z(km)']
config["use_dbscan"] = True
config["use_amplitude"] = False
config["x(km)"] = (770.45,800.540)
config["y(km)"] = (9909.28, 9939.140)
config["z(km)"] = (-6, 10)
config["vel"] = {"p": 4.0, "s": 4.0 / 1.75}  # We assume rather high velocities as we expect deeper events



config["method"] = "BGMM"
if config["method"] == "BGMM":
    config["oversample_factor"] = 4
if config["method"] == "GMM":
    config["oversample_factor"] = 1

# DBSCAN
config["bfgs_bounds"] = (
    (config["x(km)"][0] - 1, config["x(km)"][1] + 1),  # x
    (config["y(km)"][0] - 1, config["y(km)"][1] + 1),  # y
    (0, config["z(km)"][1] + 1),  # x
    (None, None),  # t
)
config["dbscan_eps"] = 25  # seconds
config["dbscan_min_samples"] = 3

# Filtering
config["min_picks_per_eq"] = 3
config["max_sigma11"] = 2.0
config["max_sigma22"] = 1.0
config["max_sigma12"] = 1.0




##PICAR
picker = sbm.PhaseNet.from_pretrained("instance")

if torch.cuda.is_available():
    picker.cuda()

# We tuned the thresholds a bit - Feel free to play around with these values
picks = picker.classify(stream, batch_size=256, P_threshold=0.075, S_threshold=0.1)

#print(picks)

# picks = picker.classify(stream, batch_size=256, P_threshold=0.075, S_threshold=0.1).picks

val = Counter([p.phase for p in picks])  # Output number of P and S picks

print(val)


###CONVERTIR A PANDAS PARA QUE GAMMA LEA

pick_df = []
for p in picks:
    pick_df.append({
        "id": p.trace_id,
        "timestamp": p.peak_time.datetime,
        "prob": p.peak_value,
        "type": p.phase.lower()
    })
pick_df = pd.DataFrame(pick_df)

station_df = []
for station in inv[0]:
    station_df.append({
        "id": f"EC.{station.code}.",
        "longitude": station.longitude,
        "latitude": station.latitude,
        "elevation(m)": station.elevation
    })
station_df = pd.DataFrame(station_df)

station_df["x(km)"] = station_df.apply(lambda x: transformer.transform(x["latitude"], x["longitude"])[0] / 1e3, axis=1)
station_df["y(km)"] = station_df.apply(lambda x: transformer.transform(x["latitude"], x["longitude"])[1] / 1e3, axis=1)
station_df["z(km)"] = station_df["elevation(m)"] / 1e3

northing = {station: y for station, y in zip(station_df["id"], station_df["y(km)"])}
station_dict = {station: (x, y) for station, x, y in zip(station_df["id"], station_df["x(km)"], station_df["y(km)"])}


print(pick_df.sort_values("timestamp"))


##ASOCIAR

catalogs, assignments = association(pick_df, station_df, config, method=config["method"])

catalog = pd.DataFrame(catalogs)
assignments = pd.DataFrame(assignments, columns=["pick_idx", "event_idx", "prob_gamma"])

print(catalogs)



#catalog['lat'] = catalog.apply(lambda x: print(f"Transformando: x={x['x(km)']*1e3}, y={x['y(km)']*1e3}") or transformer_utm2geo.transform(x['x(km)']*1e3,x['y(km)']*1e3)[0], axis=1 )
catalog['lat'] = catalog.apply(lambda x: transformer_utm2geo.transform(x['x(km)']*1e3,x['y(km)']*1e3)[0], axis=1 )
catalog['lon'] = catalog.apply(lambda x: transformer_utm2geo.transform(x['x(km)']*1e3,x['y(km)']*1e3)[1], axis=1 )
catalog['depth'] = catalog.apply(lambda x: x['z(km)']*-1,axis=1)
print(catalog.head(10))

catalog.to_csv("./cotopaxi_location.csv")

'''


# Coordenadas UTM ejemplo (Estos son solo valores de ejemplo, reemplácelos con sus propios datos)
utm_x = 500000  # en metros
utm_y = 4649776  # en metros

# Realizar la transformación
lat, lon = transformer_utm2geo.transform(utm_x, utm_y)

# Imprimir las coordenadas transformadas
print(f"Latitud: {lat}, Longitud: {lon}")

'''