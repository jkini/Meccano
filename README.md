# Ensemble Modeling for Multimodal Visual Action Recognition
Official code repo for Ensemble Modeling for Multimodal Visual Action Recognition [ICIAP-W 2023 ${\color{red}Competition~Winner}$]
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
Download following components of the Meccano dataset from the [official website](https://iplab.dmi.unict.it/MECCANO/challenge.html): <br>
[RGB frames](https://iplab.dmi.unict.it/sharing/MECCANO/MECCANO_RGB_frames.zip) <br>
[Depth frames](https://iplab.dmi.unict.it/sharing/MECCANO/MECCANO_Depth_frames.zip) <br>
[Action annotations](https://iplab.dmi.unict.it/sharing/MECCANO/MECCANO_action_annotations.zip)

## Pre-trained weights
We use the Swin3D-B backbone, which is pre-trained on the SomethingSomething v2 dataset.<br>
Swin3D-B with Something-Something v2 pre-training: [Google Drive](https://drive.google.com/drive/folders/195ecPNdP_f_ds7aBUeIWf4z714OxpVQu?usp=drive_link) <br>

The RGB frames and Depth maps are passed through two independently trained Swin3D-B encoders. The resultant class probabilities, obtained from each pathway, are averaged to subsequently yield action classes. <br>
**Ours** (RGB) with Something-Something v2 pre-training: [Google Drive](https://drive.google.com/drive/folders/14cUWo31X8PBNY61brvzHs2ORkG9dhILi?usp=drive_link) <br>
**Ours** (Depth) with Something-Something v2 pre-training: [Google Drive](https://drive.google.com/drive/folders/1ecY5T4nLv0ztMarPS02oBASTeIzfi2pO?usp=drive_link)

## We Credit
Thanks to https://github.com/SwinTransformer/Video-Swin-Transformer, for the preliminary implementations.

## Citation
````
@article{kini2023ensemble,
  title={Ensemble Modeling for Multimodal Visual Action Recognition},
  author={Kini, Jyoti and Fleischer, Sarah and Dave, Ishan and Shah, Mubarak},
  journal={arXiv preprint arXiv:2308.05430},
  year={2023}
}

@article{kini2023egocentric,
  title={Egocentric RGB+Depth Action Recognition in Industry-Like Settings},
  author={Kini, Jyoti and Fleischer, Sarah and Dave, Ishan and Shah, Mubarak},
  journal={arXiv preprint arXiv:2309.13962},
  year={2023}
}

````
