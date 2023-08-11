# HDC-MiniROCKET
Code to the paper [1]. The approach is based on the time series classification algorithm MiniROCKET [2] and extend it with explicit time encoding by HDC.  

[1] Schlegel, K., Neubert, P. & Protzel, P. (2022) HDC-MiniROCKET: Explicit Time Encoding in Time Series Classification with Hyperdimensional Computing. In Proc. of International Joint Conference on Neural Networks (IJCNN).
[2] A. Dempster, D. F. Schmidt, and G. I. Webb, “MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification,” Proc. ACM SIGKDD Int. Conf. Knowl. Discov. Data Min., pp. 248–257, 2021.

# Usage
We recommend using a virtual environment to run the code.
- Create virtual environment:
```python3 -m venv venv```
- Activate virtual environment:
```source venv/bin/activate```
- Install requirements:
```pip3 install -r requirements.txt```


## 1. Download Dataset:
Dataset download: 
```cd data; wget http://www.timeseriesclassification.com/Downloads/Archives/Univariate2018_ts.zip; unzip Univariate2018_ts.zip; cd ..```

## 2. Run scripts:
### File descriptions 
- main.py contains some arguments (dataset, UCR index, scale, HDC dim, etc.)
- config.py contains more specific parameters for running the experiment 
- main_run.py contains the "trainer" with training and evaluation functions 
- models/HDC_MINIROCKET.py is the backbone of HDC-MiniRocket and contains all necessary functions 
- models/Minirocket_utils is basically the original [implementation of MiniROCKET](https://github.com/angus924/minirocket) extended by the HDC operations 

### Arguments to run
- *--dataset* parameter as argument of main.py to select between datasets (UCR, synthetic, and synthetic_hard).
- *--complete_UCR* parameter as argument of main.py to work with the complete UCR ensemble 
- *--multi_scale* parameter as argument of main.py to run different scales defined in config.py
- *--scale* parameter as argument of main.py to define a specific similarity scale
- *--ensemble_idx* parameter as argument of main.py to run a dataset of UCR
- *--config* parameter as argument of main.py to specify various configuration parameters defined in config.py 
- *--dataset_path* parameter as argument of main.py to specify the path to the dataset (default: data/)

### Run:
#### UCR Datasets:
- run dataset 0 of UCR with scale=0
```python3 main.py --model HDC_MINIROCKET --dataset UCR --ensemble_idx 0 --scale 0 --config Config_orig```
- Run the complete UCR Benchmark ensemble with different scales: 
```python3 main.py --model HDC_MINIROCKET --dataset UCR --complete_UCR --multi_scale --config Config_orig```
- run the complete UCR with automatically selecting the best scale (cross validation)
```python3 main.py --model HDC_MINIROCKET --dataset UCR --complete_UCR --multi_scale --config Config_orig_auto```
#### Synthetic Dataset:
- run normal synthetic datasets
```python3 main.py --model HDC_MINIROCKET --dataset synthetic --scale 1 --config Config_orig```
- run hard synthetic datasets
```python3 main.py --model HDC_MINIROCKET --dataset synthetic_hard --scale 1 --config Config_orig```
#### Time Measurement:
- run time measurement on HDC MiniROCKET
```python3 main.py --model HDC_MINIROCKET --dataset UCR --complete_UCR --scale 1 --config Config_time_measure```
- run time measurement on original MiniROCKET
```python3 main.py --model MINIROCKET --dataset UCR --complete_UCR --config Config_time_measure```

### Results:
#### Accuracies:
- the results will be written and saved in /results in from of Excel spreadsheets and text files
#### Figures:
- to plot the figures as in the paper, run the 'plot_figure.m' MATLAB script 