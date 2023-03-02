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
##### Train a model on any CPU or GPU (only single GPU is supported)   
 ```bash
# module folder
cd activelearning_ic

# run training  
python main.py --sample_method [SAMPLE_METHOD] --cluster_mode [CLUSTER_MODE] --conf_mode [CONF_MODE] --args
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

