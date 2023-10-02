# Ensemble Modeling for Multimodal Visual Action Recognition
Official code repo for Ensemble Modeling for Multimodal Visual Action Recognition [ICIAP-W 2023] 
[Project](https://www.crcv.ucf.edu/research/projects/ensemble-modeling-for-multimodal-visual-action-recognition/) and 
[Arxiv](https://arxiv.org/pdf/2308.05430.pdf)

## Installations
````
conda create -n mm python=3.11.4
conda activate mm
conda install pytorch=2.0.1 torchvision=0.15.2 torchaudio=2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c anaconda numpy    
conda install -c conda-forge matplotlib
conda install -c conda-forge tqdm
pip install opencv-python
pip install fvcore
pip install timm
pip install mmcv==1.3.11
pip install einops
pip install scikit-learn
pip install focal-loss-torch
pip install pandas
pip install seaborn
````
## Dataset preparation
Download following components of the Meccano dataset from the official website:
[RGB frames](https://iplab.dmi.unict.it/sharing/MECCANO/MECCANO_RGB_frames.zip) <br>
[Depth frames](https://iplab.dmi.unict.it/sharing/MECCANO/MECCANO_Depth_frames.zip) <br>
[Action annotations](https://iplab.dmi.unict.it/sharing/MECCANO/MECCANO_action_annotations.zip)

## Pre-trained weights
RGB with Something-Something v2 pre-training: [Google Drive](https://drive.google.com/drive/folders/14cUWo31X8PBNY61brvzHs2ORkG9dhILi?usp=drive_link) <br>
Depth with Something-Something v2 pre-training: [Google Drive](https://drive.google.com/drive/folders/1ecY5T4nLv0ztMarPS02oBASTeIzfi2pO?usp=drive_link)
