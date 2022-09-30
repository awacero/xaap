import pandas as pd
import numpy as np

df = pd.read_csv("./data/sipass/chiles_entrenamiento.csv", header=0, sep=';',index_col =0)
conditions = [
    (df['ESTACION'] == "CHL1"),
    (df['ESTACION'] == "CHL2"),
    (df['ESTACION'] == "ECEN"),
    (df['ESTACION'] == "LNGL"),
    (df['ESTACION'] == "CHMA"),
    (df['ESTACION'] != "CHL1"),
    (df['ESTACION'] != "CHL2"),
    (df['ESTACION'] != "ECEN"),
    (df['ESTACION'] != "LNGL"),
    (df['ESTACION'] != "CHMA"),
    ]
# list of the values we want to assign for each condition.
values = ['EC', 'EC', 'EC', 'EC', 'EC', 'CO', 'CO', 'CO', 'CO', 'CO']

# New column
df.insert(loc=2, column='NET', value=np.select( conditions, values))
#df['NET'] = np.select( conditions, values)

# display updated DataFrame
print(df)
# export to csv
df.to_csv("./data/sipass/chiles_entrenamiento2.csv",sep=';',na_rep='null')