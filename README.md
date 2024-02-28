# Automatic_Fire_Scars_Mapping
### Project on automatic recognition of fire scars using LANDSAT's satellite imagery applying the U-Net model

### Abstract

Wildfires are a critical problem among the last years worldwide due to their consequences, such
as the carbon footprint, besides other ecological, economic and social impacts. Correspondingly,
studying the affected areas is necessary, mapping every fire scar, typically using satellite data.
In this paper, we propose a Deep Learning (DL) automate approach, using the U-Net model and
Landsat imagery, that could become a straightforward automate alternative. Thus, two models
were evaluated, each trained with a dataset with a different class balance, produced by cropping
the input images to different sizes, to a determined and variable size: 128 and AllSizes (AS),
including a better and worse class balance respectively. The testing results using 195 represen-
tative images of the study area: Dice Coefficient (DC)=0.93, Omission error (OE)=0.086 and
Commission Error (CE)=0.045- for AS, and DC=0.86, OE=0,12 and CE=0,12 for 128, proving
that a better balanced dataset results on a better performance.

### QuickStart
A sample instance is provided  
Uses GDAL so it's easier if you install [QGIS](https://qgis.org)  
Requires a [pytorch](https://pytorch.org/get-started/locally/); users without GPU should use the cpu flag `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu` on `requirements.txt`  
```bash
# venv with --sys... flag to access QGIS's GDAL
python3 -v venv --system-site-packages venv 
source venv/bin/activate
pip install -r requirements.txt # defaults to GPU, edit accordingly
python <tucodigo>.py
```
There are 3 `.py` files, also notebook (`.ipynb`) versions of the code are more user friendly
*
*
*
The trained models can be found here: https://drive.google.com/drive/folders/13UcuxZ1my6RmIuFPrQYG_J-BP_mLFZUc

### Material and Methods

Two specific datasets were constructed from The Landscape Fire Scars Database, to evaluate the performance using different image sizes. 

<img src="Images/methods_data.jpg" width="615" height="384">

Within the Convolutional Neural Network (CNN), the model U Net was selected for the prediction of the burned areas.

<img src="Images/u_net.jpg" width="755" height="387">

### Results

Finallly, some highlights of the model's performance can be seen:

<img src="Images/performance_sum.jpg" width="732" height="704">