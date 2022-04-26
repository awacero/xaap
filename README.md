Program to automatic classify volcanic events

# Installation

``` bash
conda config --add channels conda-forge
conda create -n xaap python=3.7.10
conda activate xaap

conda install pyqtgraph=0.12.3
conda install pyqt
conda install obspy=1.2.0
conda install pandas
pip install aaa_features
pip install get_mseed_data
conda install scikit-learn=1.0
conda install mlxtend
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
