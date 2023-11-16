



import pandas as pd
import csv


#sipass_filename = "./data/sipass/cotopaxi.sipass.2023.05.02.csv"
#sipass_filename = "./data/sipass/cotopaxi.sipass.2023.06.28.csv"
sipass_filename = "./data/sipass/cotopaxi.sipass.2023.03.12.csv"


xaap_output = "./data/classifications/sipass_alto.csv"

# Suponiendo que tus datos estén en 'data.csv'
df = pd.read_csv(sipass_filename)

# Transformar la data al formato deseado
xaap_check_df = pd.DataFrame()


df['tiempo_local'] = pd.to_datetime(df['FechaHora']).dt.tz_localize('America/Guayaquil')
df['tiempo'] = pd.to_datetime(df['tiempo_local'].dt.tz_convert('UTC'))

df['FechaHora'] = df['tiempo'].dt.strftime("%Y.%m.%d.%H.%M.%S")

''' 

df['FechaHora'] = df['FechaHora'].str.replace(' ', '.', regex=False).replace(':', '.', regex=False)


df['FechaHora'] = df['FechaHora'].str.replace('-', '.', regex=False)

df['FechaHora'] = df['FechaHora'].str.replace(':', '.', regex=False)
''' 


df['Coda'] = df['Coda'].astype(int)

df['Coda_str'] = df['Coda'].astype(str)



df['stream'] = 'EC.' + df['Estacion'] + '..' + df['Canal'] + '.' + df['FechaHora'] + '.' + df['Coda_str']


xaap_check_df['stream']= df['stream'] 
xaap_check_df['tipo']= df['Tipo'] 

print(xaap_check_df.head())
# Guardar el resultado en 'output.csv'
xaap_check_df.to_csv(xaap_output, index=False, header=False)
