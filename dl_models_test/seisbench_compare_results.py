import pandas as pd

human_detections_file = "./data/sipass/cotopaxi_test.csv"
model_detections_file = "./data/detections/_cotopaxi_EQTransformer_scedc_20230808165426.csv_detection.csv"
#model_detections_file = "data/detections/_cotopaxi_EQTransformer_original_20230808164858.csv"

model_columns_head = ["start_time", "endtime", "peaktime","peak_value","phase","station","coda","auxiliar"]
TOLERANCE = 2

# Cargar los archivos CSV
df_human = pd.read_csv(human_detections_file,delimiter=",")
df_model = pd.read_csv(model_detections_file,sep=",",names=model_columns_head,header=None)
TIEMPO_COMPARACION = 'start_time'


print(df_model.head())
# Asegurarse de que las columnas de tiempo estén en formato datetime
df_human['tiempo'] = pd.to_datetime(df_human['FechaHora']).dt.tz_localize(None)
df_model['tiempo'] = pd.to_datetime(df_model[TIEMPO_COMPARACION]).dt.tz_localize(None)

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
