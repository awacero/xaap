import pandas as pd

# Leer el archivo CSV
df = pd.read_csv('./cotopaxi_2022.csv')

# Crear un DataFrame vacío para almacenar las muestras seleccionadas
df_sampled = pd.DataFrame()
print(df.head())

# Para cada categoría, seleccionar 100 muestras aleatoriamente
for category in ['LP', 'VT']:
    df_category = df[df['tipo'] == category]
    df_sampled_category = df_category.sample(n=50, replace=True)
    df_sampled = pd.concat([df_sampled, df_sampled_category])

# Verificar el DataFrame con las muestras seleccionadas
print(df_sampled)

df_sampled.to_csv('./test.cvs')