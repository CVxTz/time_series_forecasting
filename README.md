# time_series_forcasting
Code for https://towardsdatascience.com/how-to-use-transformer-networks-to-build-a-forecasting-model-297f9270e630 
### Install (GPU)

```
conda create -n py38 python=3.8
conda activate py38
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c conda-forge
conda install -c conda-forge jupyterlab
conda install -c conda-forge matplotlib
git clone https://github.com/CVxTz/time_series_forecasting
cd time_series_forecasting
pip install .
```

### Run

```
bash pipeline.sh
```
