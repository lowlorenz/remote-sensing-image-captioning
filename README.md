---

<div align="center">    
 
# Active Learning Criteria for Remote Sensing Image Captioning     
<!--  
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  
--> 
![DEMO](https://lowlorenz-remotesensingwebapp-app-limtid.streamlit.app/)


<!--  
Conference   
-->   
</div>
 
## Description   
Our final submission for the TU Berlin course "Computer Vision for Remote Sensing" (Winter 22/23).
This project consists of a active learning based remote sensing image captioning system which has been trained using different active learning criteria.



## How to run   
```bash
# clone project   
git clone https://git.tu-berlin.de/wallburg/activelearning_ic.git

# install requirements       
pip install -r requirements.txt
 ```   

## Get the dataset
1. install git lfs following  [this](https://github.com/git-lfs/git-lfs/blob/main/INSTALLING.md)
then

```bash
git lfs clone  https://github.com/lowlorenz/NWPU-Captions.git
mkdir NWPU-Captions/NWPU_images
tar -xzf NWPU-Captions/NWPU_images.tar.gz -C NWPU-Captions
```  

##### Train a model on any CPU or GPU (only single GPU is supported)   
 ```bash
# module folder
cd activelearning_ic

# run training  
python src/main.py --sample_method [SAMPLE_METHOD] --cluster_mode [CLUSTER_MODE] --conf_mode [CONF_MODE] --args
```
##### Train a model on HPC Cluster (only single GPU is supported)
 The slurm scripts for all experiments are provided. Inside the script, 
  ```bash
#SBATCH -o /path/to/output/log
#SBATCH -D /home/dir/activelearning_ic/
```
need to be set.
Start training:
 ```bash
# module folder
cd activelearning_ic

# run module via slurm   
sbatch [SCRIPT_NAME].sh

# monitor output
tail -f [OUT].log
```

##### Parameters for the training

| Name | Default | Type | Description |
|------|---------|------|-------------|
| epochs | 10 | int | Number of epochs to train per cycle. |
| max_cycles | 5 | int | Number of active learning cycles to train. |
| init_set_size | 0.05 | float | Initial train set size in percent. |
| new_data_size | 0.05 | float | Percentage of added labels per cycle. |
| learning_rate | 0.0001 | float | Learning rate of the optimizer. |
| batch_size | 4 | int | Batch size. |
| sample_method | "cluster" | str | Sampling method to retrieve more labels. |
| device_type | "cuda" | str | Device to train on. |
| run_name | "test" | str | Name of the run. |
| data_path | "NWPU-Captions/" | str | Path to the NWPU-Captions dataset. |
| debug | False | bool | Debug mode. |
| val_check_interval | 1.0 | float | Validation check interval. |
| num_devices | 1 | int | Number of devices to train on. |
| num_nodes | 1 | int | Number of nodes to train on. |
| ckpt_path | None | str | Path to checkpoint to resume training. |
| mode | "train" | str | Choose between train and test mode. |
| seed | 42 | int | Global random seed. |
| conf_mode | "least" | str | Whether to sample based on "least" confidence or "margin" of confidence. |
| conf_average | "sentence" | str | Whether to sample based on average "sentence" confidence or minimum "word" confidence. |
| cluster_mode | "image" | str | Whether to use the image or text embeddings for clustering. |
| mutliple_sentence_loss | False | bool | Whether to use the image or text embeddings for clustering. |


##### Validate a trained model (CPU or single GPU)
This automatically selects the validation set and predicts on it using a pretrained model
  ```bash
# module folder
cd activelearning_ic

# predict  
python validation_generator.py --ckpt [[NAME]_[SEED]-[TIME]-[CYCLE].ckpt]
```

##### Evaluate results (CPU)
At each validation step, the predicted captions are written to (filename). The METEOR, BLEU and Rouge-L score are calculated inside a jupyter notebook after the whole cycle by comparison of the predicted captions and reference captions.     
 ```bash
 # module folder
cd activelearning_ic

# start notebook
jupyter-notebook
```  
In evaluation.ipynb:

## Structure
