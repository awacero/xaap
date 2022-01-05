Program to automatic classify volcanic events

# Installation

``` bash
conda config --add channels conda-forge
conda create -n xaap python=3.7.10
conda activate xaap

conda install pyqtgraph=0.12.3
conda install pyqt
conda install obspy
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
