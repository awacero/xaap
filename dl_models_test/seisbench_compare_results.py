import pandas as pd

#human_detections_file = "./data/sipass/cotopaxi.sipass.2023.03.12.csv"
human_detections_file = "./data/sipass/ccn.sipass.2023.08.06.csv"
#human_detections_file = "./data/sipass/cotopaxi_2023.04.02.csv"
model_detections_file = "./data/detections//picks_coincidence_xaap_EQTransformer_original_2023.08.16.23.16.16.csv"
#model_detections_file = "data/detections/picks_coincidence_xaap_EQTransformer_original_2023.08.14.23.19.27.csv"

#model_columns_head = ["start_time", "endtime", "peaktime","peak_value","phase","station","coda","auxiliar"]

#model_columns_head = ['','time','stations','trace_ids','coincidence_sum','similarity','duration']

TOLERANCE = 10

# Cargar los archivos CSV
df_human = pd.read_csv(human_detections_file,delimiter=",")
#df_model = pd.read_csv(model_detections_file,sep=",",names=model_columns_head,header=None)
df_model = pd.read_csv(model_detections_file,sep=",")

TIEMPO_COMPARACION = 'time'


print(df_model.head())

# Asegurarse de que las columnas de tiempo estén en formato datetime
df_human['tiempo_local'] = pd.to_datetime(df_human['FechaHora']).dt.tz_localize('America/Guayaquil')
df_human['tiempo'] = pd.to_datetime(df_human['tiempo_local'].dt.tz_convert('UTC'))

print("###")
print(df_human.head())

df_model['tiempo'] = pd.to_datetime(df_model[TIEMPO_COMPARACION])

# Definir la ventana de tolerancia (por ejemplo, 10 segundos)
tolerancia = pd.Timedelta(seconds=TOLERANCE)

verdaderas_positivas = 0
falsas_positivas = 0
falsas_negativas = 0

"""iterar en el frame de detecciones del operador"""
for index, row in df_human.iterrows():
    tiempo_humano = row['tiempo']
    # Verificar si hay alguna detección del modelo dentro de la ventana de tolerancia
    #compara los tiempos del MODELO para ver si estan dentro de los tiempos del humano + la tolerancia.
    match = df_model[(df_model['tiempo'] >= tiempo_humano - tolerancia) & (df_model['tiempo'] <= tiempo_humano + tolerancia)]
    #print("###")
    #print(match)
    if not match.empty:
        verdaderas_positivas += 1
    else:
        #eventos que no fueron detectados por el modelo pero si por el operador. detecciones perdidas por el modelo
        falsas_negativas += 1

# Las falsas positivas son las detecciones del modelo que no tienen correspondencia en las detecciones humanas
# ITERAR EN EL MODELO
for index, row in df_model.iterrows():
    tiempo_modelo = row['tiempo']
    #compara los tiempos del  HUMANO para ver si estan dentro del tiempo del humano + la tolerancia
    match = df_human[(df_human['tiempo'] >= tiempo_modelo - tolerancia) & (df_human['tiempo'] <= tiempo_modelo + tolerancia)]
    if match.empty:
        falsas_positivas += 1

print(f"Verdaderas Positivas: {verdaderas_positivas}")
print(f"Falsas Positivas (Eventos creados por el modelo, pero que el operador no detectó. Errores del modelo?): {falsas_positivas}")
print(f"Falsas Negativas (Eventos no detectados por el modelo. detecciones perdidas por el modelo): {falsas_negativas}")
print(f"TP:{verdaderas_positivas}")
print(f"FP:{falsas_positivas}")
print(f"FN:{falsas_negativas}")


# Nota: Asegúrate de reemplazar 'tiempo' con el nombre correcto de tu columna.
