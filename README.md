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

### Material and Methods

Two specific datasets were cropped out from the files of The Landscape Fire Scars Database, to evaluate the performance using different image sizes. These datasets included **1966** fires, dividing the data almost equally for each region, **with 977 events from Valparaíso and 989 from BioBío**. 

<img src="Images/methods_data.jpg" width="615" height="384">

Within the Convolutional Neural Network (CNN), the model U Net was selected for the prediction of the burned areas.

<img src="Images/u_net.jpg" width="755" height="387">

### Results

The results of the models are presented in the following Table:

\begin{table*}[ht]
    \centering
    \caption{Tests for the HPO and the corresponding DC}
    \begin{tabular}{@{}cccccccc@{}}
    \toprule
 Name                                                    & Configuration              & Val DC$^a$                        \\ \midrule
    Baseline performance                                    & Base HPO$^b$                   & 0.948  \\ \midrule
    Filters$^b$ & 32/256                     & 0.937        \\                 
           & 160/1280                   & 0.946
           \\ \midrule
         Learning rate & 1e-5                & 0.92          \\
           & 1e-3                       & 0.942
           \\ \midrule
         Batch size                                          & 10                         & 0.947            \\            
            & 24 & 0.942

\\ \bottomrule
    \label{tab:hpo}
    \end{tabular}
\\

Finallly, some highlights of the model's performance can be seen:

<img src="Images/performance_sum.jpg" width="732" height="704">

