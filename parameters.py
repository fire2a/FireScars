# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

#AS
LS_max_as=[1704.0, 2535.0, 3279.0, 5724.0, 5373.5, 4099.5, 1, 1000, 1784.5, 2602.5, 3291.5,
             6013.5, 5218.5, 3942.0, 1, 1000] 
LI_min_as=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1138, -593.5879,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1009, -385.6182]
mean_as=[410.599, 603.8595, 722.0121, 1684.9663, 1740.0623, 1271.392, 0.3869, 141.2389, 
         412.1512, 657.2068, 776.1463, 2205.528, 1913.4361, 1168.8483, 0.4915, 328.0038] 
std_as=[148.55842, 204.69747, 294.87129, 557.89512, 587.68786, 468.61749, 0.16364, 237.97676, 157.58976,
    223.08017, 343.99229, 504.0643, 625.52707, 460.41598, 0.15405, 190.97934]
'''
min_as=[0.0, 0.0, 0.0, 17.0, 7.0, 0.0, -0.0961538, -597.96850,
        0.0, 0.0, 0.0, 0.0, 8.0, 0.0, -0.0966292, -392.023010]
max_as=[1689.0, 2502.0, 3260.0, 5650.0, 5282.0, 4121.0, 1.0, 1000.0, 1750.0, 
        2559.0, 3325.0, 6065.0, 5224.0, 3903.0, 1.0, 1000.0]
'''

#128
LS_max128=[3660.5,  4303.5,  4832.0,  6956.0,  6174.5,  6234.0, 1,  1000,  3811.0,  4504.0,  4950.5,
            7000.5,  6210.0,  6286.5, 1,  1000]
LI_min128= [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  -0.6903,  -605.3293, 0.0,  0.0,  0.0,  0.0,  0.0,
            0.0,  -0.7970,  -573.0518]
mean_128=[415.8392, 646.4156, 761.339, 2109.5385, 1843.5207, 1189.6269, 0.4677, 301.9405,
        414.9891, 657.8648, 761.992, 2235.3473, 1853.7414, 1160.7475, 0.4924, 341.9075]
std_128=[231.0082, 317.1621, 458.5268, 740.4883, 860.3522, 663.1239, 0.2314, 276.4533,
         229.3083, 310.2317, 446.9167, 702.8047, 823.5283, 634.9575, 0.2226, 254.3484 ]
'''
min_128=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.673796, -605.416321,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.771429, -572.139282]
max_128=[3560.0, 4147.0, 4780.0, 6832.0, 6143.0, 6261.0, 1.0, 1000.0,
        3765.0, 4398.0, 4866.0, 6930.0, 6157.0, 6167.0, 1.0, 1000.0]
'''

#For evaluation.py
photo_results_path = "C:/Users/56965/Documents/TesisIan/agostoy2023november/copia_diego_2023_paper_november/evaluation_results/"

#For train128.py and trainAS.py
dataset1_training_path = "../datasets_csv_11_2023/val_train_684.csv"
dataset1_validation_path = "../datasets_csv_11_2023/val_val_196.csv"

dataset2_training_path = "../datasets_csv_11_2023/bio_train_693.csv"
dataset2_validation_path = "../datasets_csv_11_2023/bio_val_198.csv"

#For train128.py
dataset1_Firescar_tif_128_paths = "../../IanMancilla/firescarvalpo128/FireScar/"
dataset1_PostFire_tif_128_paths = "../../IanMancilla/firescarvalpo128/ImgPosF/"
dataset1_PreFire_tif_128_paths = "../../IanMancilla/firescarvalpo128/ImgPreF/"

dataset2_Firescar_tif_128_paths = "../../IanMancilla/firescarbiobio128/FireScar/"
dataset2_PostFire_tif_128_paths = "../../IanMancilla/firescarbiobio128/ImgPosF/"
dataset2_PreFire_tif_128_paths = "../../IanMancilla/firescarbiobio128/ImgPreF/"

model_128_path="/modelos/ep25_lr1e-04_bs16_014_128_std_25_08_mult3_adam01.model"

#For trainAS.py
dataset1_Firescar_tif_AS_paths = "../../IanMancilla/firescarvalpoallsizes/FireScar/"
dataset1_PostFire_tif_AS_paths = "../../IanMancilla/firescarvalpoallsizes/ImgPosF/"
dataset1_PreFire_tif_AS_paths = "../../IanMancilla/firescarvalpoallsizes/ImgPreF/"

dataset2_Firescar_tif_AS_paths = "../../IanMancilla/firescarbiobioallsizes/FireScar/"
dataset2_PostFire_tif_AS_paths = "../../IanMancilla/firescarbiobioallsizes/ImgPosF/"
dataset2_PreFire_tif_AS_paths = "../../IanMancilla/firescarbiobioallsizes/ImgPreF/"

model_AS_path="/modelos/ep25_lr1e-04_bs16_021__as_std_adam_f01_13_07_x3.model"