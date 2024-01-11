import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import jaccard_score
from model_u_net import DoubleConv, Down, Up, OutConv, UNet, model
from parameters import LS_max_as, LI_min_as, mean_as, std_as, min_as, max_as, LS_max128, LI_min128, mean_128, std_128, min_128, max_128
from train128 import create_dataset128
from trainAS import create_datasetAS
from arguments import get_evaluation_args
import re
#ejecución desde la linea de comandos 
#python evaluation.py -ev1 "../datasets_csv_11_2023/val_test_97.csv" -ev2 "../datasets_csv_11_2023/bio_test_98.csv" -mp "../modelos/ep25_lr1e-04_bs16_021__as_std_adam_f01_13_07_x3.model"

def obtain_model_size(input_str):
    # Define el patrón de búsqueda para '128' y 'as' en las posiciones específicas
    patron_128 = re.compile(r'_\d+_(\d+)_')
    patron_as = re.compile(r'_(as)_')
    # Busca el patrón en la cadena
    coincidencia_128 = patron_128.search(input_str)
    coincidencia_as = patron_as.search(input_str)
    # Asigna los valores a las variables según las coincidencias
    valor_128 = coincidencia_128.group(1) if coincidencia_128 else None
    valor_as = coincidencia_as.group(1) if coincidencia_as else None
    return valor_128, valor_as

if __name__ == '__main__':
    args = get_evaluation_args()
    evald1=evald2=dataset=pd.DataFrame()
    print(f'ev1: {args.ev1}, ev2: {args.ev2}, mp: {args.mp}')

evald1=pd.read_csv(args.ev1)
evald2=pd.read_csv(args.ev2)
dataset=pd.concat([evald1,evald2],axis=0,ignore_index=True)
photo_results_path = "C:/Users/56965/Documents/TesisIan/agostoy2023november/copia_diego_2023_paper_november/evaluation_results/"

if obtain_model_size(args.mp)[0] == "128":
    model_size = "128"
elif obtain_model_size(args.mp)[1] == "as":
    model_size = "AS"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Loads a model of a specific epoch to evaluate
#AS model
#model_path="../modelos/ep25_lr1e-04_bs16_021__as_std_adam_f01_13_07_x3.model"
#model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.load_state_dict(torch.load(args.mp, map_location=torch.device('cpu')))

def evaluation(model_size):
    """
    Evaluates the metrics of the given dataset and plots images for each input comparing the pre and post-fire images and
    the original firescar vs the model's prediction.
    
    dataset (object):  Pandas dataframe with the data's filenames from two different regions. There are 3 columns with the required data filenames for 
    each input. "ImPosF": The image post Fire, "ImgPreF": The image pre Fire, and "FireScar_tif": The label, in a raster file
    model_size (str): "AS" or "128", set depending on the desired dataset, AS or 128. 

    """
    # Adjust these following parameters in the parameter's file:
    # ssx_ey: where x= index' number and y: either a for the AS model or 1 for the 128 model
    # ss1_ey (int): index of the first input from the Dataset 1: Region of Valparaiso
    # ss2_ey (int): index of the last input from the Dataset 1: Region of Valparaiso
    # if there is only one Dataset, set both subset_size3 and subset_size4 as 0. 
    # ss3_ey (int): index of the first input from the Dataset 2: Region of Biobio
    # ss4_ey (int): index of the last input from the Dataset 2: Region of Biobio
    # mult (int): times to input the data
    np.random.seed(3)
    torch.manual_seed(3)
    if model_size=="AS":
        data_eval = create_datasetAS(dataset, 0, len(evald1), len(evald1), len(evald1)+len(evald2), mult=1)
    elif model_size=="128":
        data_eval = create_dataset128(dataset, 0, len(evald1), len(evald1), len(evald1)+len(evald2), mult=1)

    batch_size = 1 # 1 to create diagnostic images, any value otherwise
    all_dl = DataLoader(data_eval, batch_size=batch_size)#, shuffle=True)
    progress = tqdm(enumerate(all_dl), total=len(all_dl))
    
    test_df=pd.DataFrame(columns=["ImgPosF","iou","DC","CE","OE"])
    dicec_eval_acc=[]
    FN_eval=[]
    TP_eval=[]
    FP_eval=[]
    comission=[]
    omission=[]
    cont=0
    model.eval()

    def dice2d(pred, targs):  
        """
        Returns the input's Dice Coefficient metric

        pred (object): Object conformed by the binary output (prediction)   
        targs (object): Object conformed by the binary ground truth (firescar of reference)
        
        """
        pred = pred.squeeze()
        targs = targs.squeeze()
        return 2. * (pred*targs).sum() / (pred+targs).sum()

    # define loss function
    loss_fn = nn.BCEWithLogitsLoss()

    # run through test data
    all_ious = []
    all_accs = []
    for i, batch in progress:
        x, y = batch['img'].float().to(device), batch['fpt'].float().to(device)
        idx = batch['idx']

        output = model(x).cpu()

        # obtain binary prediction map
        pred = np.zeros(output.shape)
        pred[output >= 0] = 1

        # derive Iou score
        cropped_iou = []
        for j in range(y.shape[0]):
            z = jaccard_score(y[j].flatten().cpu().detach().numpy(),
                            pred[j][0].flatten())
            if (np.sum(pred[j][0]) != 0 and
                np.sum(y[j].cpu().detach().numpy()) != 0):
                cropped_iou.append(z)       

        all_ious = [*all_ious, *cropped_iou]

        # derive scalar binary labels on a per-image basis
        y_bin = np.array(np.sum(y.cpu().detach().numpy(),
                                axis=(1,2)) != 0).astype(int)
        prediction = np.array(np.sum(pred,
                                axis=(1,2,3)) != 0).astype(int)
        # derive image-wise accuracy for this batch
        all_accs.append(accuracy_score(y_bin, prediction))

        # derive binary segmentation map from prediction
        output_binary = np.zeros(output.shape)
        output_binary[output.cpu().detach().numpy() >= 0] = 1

        if batch_size == 1:
            if prediction == 1 and y_bin == 1:
                res = 'true_pos'
            elif prediction == 0 and y_bin == 0:
                res = 'true_neg'
            elif prediction == 0 and y_bin == 1:
                res = 'false_neg'
            elif prediction == 1 and y_bin == 0:
                res = 'false_pos'    
            TP_eval.append((output_binary.squeeze()*y.cpu().detach().numpy().squeeze()).sum())
            FN_eval.append(((output_binary.squeeze()==0) & (y.cpu().detach().numpy().squeeze()==1)).sum())
            FP_eval.append(((output_binary.squeeze()==1) & (y.cpu().detach().numpy().squeeze()==0)).sum())
            dicec_eval_acc.append(dice2d(output_binary,y.cpu().detach().numpy()))
            test_df.loc[cont,"OE"]=FN_eval[cont]/(TP_eval[cont]+FN_eval[cont])
            test_df.loc[cont,"CE"]=FP_eval[cont]/(TP_eval[cont]+FP_eval[cont])
            test_df.loc[cont,"DC"]=dice2d(output_binary,y.cpu().detach().numpy()) 
            test_df.loc[cont,"ImgPosF"]=(batch['imgfile'][0].split("/")[-1])
            OE=FN_eval[cont]/(TP_eval[cont]+FN_eval[cont])
            this_iou = jaccard_score(y[0].flatten().cpu().detach().numpy(),
                                    pred[0][0].flatten())
            test_df.loc[i,"iou"]=this_iou        

            # create plot
            f, (ax1, ax2, ax3,ax4) = plt.subplots(1, 4, figsize=(20,20))
            x=x.cpu()
            y=y.cpu()

            # false color plot Image prefire
            ax1.imshow(0.2+1.5*(np.dstack([x[0][12], x[0][11], x[0][10]])-np.min([x[0][12].numpy(),
                                x[0][11].numpy(), x[0][10].numpy()]))/(np.max([x[0][12].numpy(),
                                x[0][11].numpy(), x[0][10].numpy()])-np.min([x[0][12].numpy(),
                                x[0][11].numpy(), x[0][10].numpy()])), origin='upper')

            ax1.set_title("ImgPreF",fontsize=12)
            ax1.set_xticks([])
            ax1.set_yticks([])
            #Image Pos-Fire
            ax2.imshow(0.2+1.5*(np.dstack([x[0][4], x[0][3], x[0][2]])-np.min([x[0][4].numpy(), 
                                x[0][3].numpy(), x[0][2].numpy()]))/(np.max([x[0][4].numpy(),
                                x[0][3].numpy(), x[0][2].numpy()])-np.min([x[0][4].numpy(),
                                x[0][3].numpy(), x[0][2].numpy()])), origin='upper')

            ax2.set_title("ImgPosF",fontsize=12)
            ax2.set_xticks([])
            ax2.set_yticks([])

            # segmentation ground-truth and prediction
            ax3.imshow(y[0], cmap='Greys_r', alpha=1)
            ax4.imshow(pred[0][0], cmap='Greys_r', alpha=1)
            ax3.set_title("Original Scar",fontsize=12)
            ax3.set_xticks([])
            ax3.set_yticks([])
            ax3.annotate("IoU={:.2f}".format(this_iou), xy=(5,15), fontsize=15)

            ax4.set_title({'true_pos': 'Scar Prediction: True Positive \n  -IoU={:.2f},' 
                    '-OE={:.2f}, -CE={:.2f}, -DC={:.2F}'.format(this_iou, test_df.loc[cont,"OE"],test_df.loc[cont,"CE"],test_df.loc[cont,"DC"]),
            'true_neg': 'Scar Prediction: True Negative \n  -IoU={:.2f},' 
            '-OE={:.2f}, -CE={:.2f}, -DC={:.2F}'.format(this_iou,test_df.loc[cont,"OE"],test_df.loc[cont,"CE"],test_df.loc[cont,"DC"]),
            'false_pos': 'Scar Prediction: False Positive   -IoU={:.2f},'
            '-OE={:.2f}, -CE={:.2f}, -DC={:.2F}'.format(this_iou, test_df.loc[cont,"OE"],test_df.loc[cont,"CE"],test_df.loc[cont,"DC"]),
                'false_neg': 'Scar Prediction: False Negative \n  -IoU={:.2f},'
            '-OE={:.2f}, -CE={:.2f}, -DC={:.2F}'.format(this_iou,test_df.loc[cont,"OE"], 0,test_df.loc[cont,"DC"])}[res],
                    fontsize=12)
            cont+=1      

            f.subplots_adjust(0.05, 0.02, 0.95, 0.9, 0.05, 0.05)
            '''
            plt.savefig(photo_results_path+(os.path.split(batch['imgfile'][0])[1]).\
                        replace('.tif', '.extention.png').replace(':', '_'),
                        dpi=200)   
            plt.close()     #comment to display
            '''
    print('DC',test_df["DC"].mean(),'OE', test_df["OE"].mean(),'CE',test_df["CE"].mean())
    print('iou:', len(all_ious), np.average(all_ious))
    return test_df

test_df=evaluation(model_size)

print('DC',test_df["DC"].mean(),'OE', test_df["OE"].mean(),'CE',test_df["CE"].mean())