Program to automatic classify volcanic events

# Installation

``` bash
conda config --add channels conda-forge
conda create -n xaap python pyqtgraph=0.12.3 pyqt obspy pandas scikit-learn=1.0 mlxtend  -y
conda activate xaap

pip install aaa_features get_mseed_data

```


# Clone XAAP code
```
git clone https://github.com/awacero/xaap.git

```


# Run XAAP
```
conda activate xaap
cd xaap/xaap
python xaap_gui.py 
```

# Train the model 

## Create the feature set using SIPASS CSV data

```
conda activate xaap 
cd xaap/xaap
python create_feature_set.py config/profile_create_feature_guagua.txt 1 2222
```
The features computed are stored in xaap/data/features/features_guagua_UNIQUEID.csv

## Train the model 

```
conda activate xaap 
cd xaap/xaap
python random_forest_train_model.py config/profile_train_model_guagua.txt
```

The last command generates a PKL model in data/models folder
```
guagua_rf_20211007144655.pkl
```

