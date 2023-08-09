import pandas as pd
import numpy as np

# Cargar los datos

original_csv_file = "./data/sipass/sismos_cotopaxi_revisados.gviracuchaEtiquetas2019.csv"
VOLCAN = 'Cotopaxi'
data = pd.read_csv(original_csv_file)

data = data.rename(columns={
    'Year': 'year',
    'Month': 'month',
    'Day': 'day',
    'HourStart': 'hour',
    'MinStart': 'minute',
    'SegStart': 'second'
})


# Convertir las columnas de fecha y hora a una sola columna de datetime
data['FechaHora'] = pd.to_datetime(data[['year', 'month', 'day', 'hour', 'minute', 'second']])

data = data.rename(columns={
    'Type': 'Tipo',
    'Station': 'Estacion',
    'Duration':'Coda'

})


# Crear las nuevas columnas con valores de 0 o 'cuentas'
data['T(S-P)'] = 0
data['AmpMax'] = 0
data['AmpMin'] = 0
data['AmpUnidad'] = 'cuentas'
data['Periodo(seg)'] = 0
data['Magnitud'] = 0
data['Energia'] = 0
data['Canal'] = 'BHZ'
data['Evol_dr_cm2'] = 0
data['Evol_dr_sm2'] = 0
data['Volcan'] = VOLCAN
data['auxiliar'] = 1

print(data.head)

# Seleccionar las columnas que queremos en el orden que queremos
##data = data[['FechaHora', 'FechaHora','Tipo' ,'T(S-P)', 'AmpMax', 'AmpMin', 'AmpUnidad', 'Coda', 'Periodo(seg)', 'Magnitud', 'Energia', 'Estacion', 'Canal', 'Evol_dr_cm2', 'Evol_dr_sm2', 'Volcan']]

data = data[['FechaHora', 'FechaHora','Tipo' ,'T(S-P)', 'Coda', 'Estacion', 'Canal', 'Volcan','auxiliar']]

# Guardar el nuevo dataframe a un archivo csv
data.to_csv('new_data.csv', index=False)
