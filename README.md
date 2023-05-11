Program to automatic classify volcanic events

# Installation
Execute the commands as normal user NOT as root

## Create the conda environment

``` bash
conda config --add channels conda-forge
conda create --name igxaap python=3.10 
```

## Install the requirements libraries 

``` bash
conda activate igxaap  
pip install torch==1.12.1 seisbench 
conda install pyqtgraph pyqt chardet
pip install scikit-learn 
##Install scikit-learn using pip or LD_PRELOAD generates an error ###
pip install aaa_features get_mseed_data 
```

## Clone XAAP code
```
cd /LOCAL_PATH/
git clone https://github.com/awacero/xaap.git

```
## Copy,modify or create the configuration files (json files)
``` bash
cd /LOCAL_PATH/xaap/config

EXAMPLE.xaap_gui.json
EXAMPLE.server_configuration.json
EXAMPLE.volcanoes.json
EXAMPLE.stations.json

```

# Run XAAP
```
conda activate xaap
cd xaap/xaap
python xaap_gui.py 
```

# Train the model 
To create features for your seismic data, you can use the `create_feature.py` module. This module reads a configuration file with the necessary parameters to create the features, and then uses multiprocessing to process the data in parallel.

To edit the configuration file, open it in a text editor and modify the settings as appropriate. The most important settings to check are:

    volcano: The name of the volcano you are analyzing.
    cores: The number of CPU cores to use for feature extraction.
    sipass_db_file: The path to the SIPASS database file containing information about each event.
    features_folder: The path to the folder where feature files will be saved.
    network: The seismic network code for the station data.
    location: The location code for the station data.
    mseed_client_id: The ID of the MSEED client to use for retrieving data.
    mseed_server_config_file: The path to the configuration file for the MSEED server.
    features_file: The path to the feature configuration file.
    domains: The feature domains to compute (time, spectral, cepstral).

Run the create feature module with a configuration profile and optionally the number of rows to read.

```
conda activate xaap 
cd xaap
python train/create_feature_set.py config/profile_create_feature_guagua.txt 1 2222
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

## Test Seisbench models
```
conda activate xaap 
cd xaap
python dl_models_test/seisbench_models_test.py config/dl_models_test_config/profile_dl_models_test_chiles.txt 1 10
```

