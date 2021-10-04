'''
Created on Oct 1, 2021

@author: wacero
'''

import sys
import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
from datetime import datetime
import json 

from sklearn.preprocessing import StandardScaler
from sklearn import svm
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.neighbors import KNeighborsClassifier
from numpy import average

plt.rcParams["figure.figsize"]=[20,10] 
pd.set_option('display.max_columns',None)
data_file="./feature_tungurahua_igepn_2011_2021.csv"
column_names = ['id_code','seismic_type']
unique_id = datetime.now().strftime("%Y%m%d%H%M%S")

data = pd.read_csv(data_file)

rows_length,column_length = data.shape

for i in range(column_length - 2):
    column_names.append("f_%s" %i)

data.columns = column_names
data.columns = data.columns.str.replace(' ', '')

#Use just 2 types of events
#data = data.loc[(data.seismic_type.str.contains('LP')) | (data.seismic_type.str.contains('VT')) ]
data['seismic_type'].hist() 

#Scale the features
x_no_scaled = data.iloc[:,2:].to_numpy()
scaler = StandardScaler()
x = scaler.fit_transform(x_no_scaled)


##CATEGORIZAR EN ONE HOT
"""
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()

data_seismic_type = data[['seismic_type']]
data_seismic_type_onehot = cat_encoder.fit_transform(data_seismic_type)
print(data_seismic_type_onehot.toarray())
y = data_seismic_type_onehot.toarray()
print((y.shape))
"""
##CATEGORIZAR ORDINAL ENCODER
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
data_seismic_type = data[['seismic_type']]
data_seismic_type_encode = ordinal_encoder.fit_transform(data_seismic_type)
y = data_seismic_type_encode
#print(y)
print((y.shape))

data_scaled_categorized = pd.DataFrame(x,columns=data.columns[2:])

data_scaled_categorized['seismic_type'] = data_seismic_type_encode

data_scaled_categorized.head()



"""
sffs = SFS (svm.SVC(), k_features=90,forward=True, floating=False,verbose=2, scoring='f1', cv=0, n_jobs=-1)
sffs = SFS (svm.SVC())

sffs.fit(x,y)
sffs.k_feature_names_

"""

#'''
##84 features x 16 cores = 32 minutes
#scoring = F1 for binary
#
knn = KNeighborsClassifier(n_neighbors = 3)
sfs = SFS(knn, k_features=50,forward=True,floating=False,verbose=2,scoring='accuracy',cv=0, n_jobs=-1)
sfs.fit(x, y.ravel())

#'''




fig1 = plot_sfs(sfs.get_metric_dict(), kind='std_dev') 
plt.title('Sequential Forward Selection (w. StdErr)') 
plt.grid() 
#plt.show()
plt.savefig("./best_features_%s.png" %unique_id) 
#print(sfs.k_feature_names_)

best_features_pd = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
best_features_pd.to_csv("./best_features_%s.csv" %unique_id)
"""
print(type(sfs.subsets_))
print((sfs.subsets_))
best_features_file = open("./best_features_%s.json" %unique_id,"w")
best_features_file.write(str(sfs.subsets_))
"""

