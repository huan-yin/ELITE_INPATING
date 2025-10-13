# ELITE_INPATING

We have modified the code of <a href="https://github.com/csyxwei/ELITE">ELITE</a> to enable it to be trained via the inpainting pipeline. We use it for anomaly generation.


## How to train

### First: install conda environment
```
conda env create -f environment.yml

conda activate elite_inpainting
```

### Second: prepare dataset
For example, download MVTec AD dataset from this <a href=https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads>website</a>. Then execute the following command to extract the file.



```
mkdir -p datasets/mvtec && tar -xvf mvtec_anomaly_detection.tar.xz -C datasets/mvtec
```

We need a series of images and their corresponding mask images during training. Each object in the test set of MVTec-AD contains various defective images and their corresponding mask images. Execute the following command to integrate the defective images and masks of all objects together, so as to form unified "images" and corresponding "masks" folders.

```
python mvtec_defect_collector.py 
```


### Finally: train generative model
```
bash train_global.sh
```


## How to inference
### Inference without gradient
```
python reference_inpainting_inference.py
```
### Inference with gradient
```
python reference_inpainting_inference_with_gradient.py
```
