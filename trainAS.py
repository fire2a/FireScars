import numpy as np
import torch
from torch import nn, optim
from tqdm.autonotebook import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, random_split, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import Normalize
from sklearn.metrics import jaccard_score
from model_u_net import model, device
from parameters import LS_max_as, LI_min_as, mean_as, std_as
from parameters import dataset1_training_path, dataset2_training_path, dataset1_validation_path, dataset2_validation_path, model_AS_path
from parameters import dataset1_Firescar_tif_AS_paths, dataset1_PostFire_tif_AS_paths, dataset1_PreFire_tif_AS_paths, dataset2_Firescar_tif_AS_paths, dataset2_PostFire_tif_AS_paths, dataset2_PreFire_tif_AS_paths   
import pandas as pd
import rasterio as rio 
import os
from osgeo import gdal
from scipy.interpolate import NearestNDInterpolator
from arguments import get_train_args

data_train=data_train1=data_train2=data_val=data_val1=data_val2=pd.DataFrame()  #comment by introducing corresponding data

# +
# when there is one dataset 

#data_train1=pd.read_csv("../datasets_csv_11_2023/val_train_684.csv")
#data_train2=pd.DataFrame()
#data_val1=pd.read_csv("../datasets_csv_11_2023/val_val_196.csv")
#data_val2=pd.DataFrame()


# when there are two datasets to analyze

data_train1=pd.read_csv(dataset1_training_path)
data_train2=pd.read_csv(dataset2_training_path)  
data_train=pd.concat([data_train1,data_train2], axis=0, ignore_index=True)

data_val1=pd.read_csv(dataset1_validation_path)
data_val2=pd.read_csv(dataset2_validation_path)  
data_val=pd.concat([data_val1,data_val2], axis=0, ignore_index=True)

# -

def preprocessing(imgdata):
    """
    Preprocesses each image's data, removing outliers and values out of range. 

    imgdata (object: ndarray): Matrix composed of the 16 concatenated matrices of a pre and post-fire image' bands.

    """
    #the limits are determined according to the Dataset's nature
    for k in range(1,17):
        if (imgdata[k-1]>LS_max_as[k-1]).any():
            if imgdata[k-1].mean()<LS_max_as[k-1]:
                imgdata[k-1][imgdata[k-1]>LS_max_as[k-1]]=imgdata[k-1].mean()
            else:
                imgdata[k-1][imgdata[k-1]>LS_max_as[k-1]]=mean_as[k-1]
        elif (imgdata[k-1]<LI_min_as[k-1]).any():
            if imgdata[k-1].mean()>LI_min_as[k-1]:
                imgdata[k-1][imgdata[k-1]<LI_min_as[k-1]]=imgdata[k-1].mean()
            else: 
                imgdata[k-1][imgdata[k-1]<LI_min_as[k-1]]=mean_as[k-1]
    return imgdata


# +
class firescardataset():
    def __init__(self, dataset, ss1, ss2, ss3, ss4, mult=1, transform=None):
        """
        Processes the data to enter into the net. The data to enter can be selected indicating the subset_size(x), dividing the first two 
        for the data of one region, while the latter two for the one of the other region. 
        The files' paths must be defined for the pre- and post-fire images and firescars files. 
        
        dataset (object):  Pandas dataframe with the data's filenames from two different regions. There are 3 columns with the required data filenames for 
        each input. "ImPosF": The image post Fire, "ImgPreF": The image pre Fire, and "FireScar_tif": The label, in a raster file
        # ssx: stands for the subset size of the dataset and x for the index, defined as follows:
        # ss1, ss2 (int): indexes for the first and last input from the Dataset 1 
        # if there is only one Dataset, set ss3 and ss4 as 0. 
        # ss3, ss4 (int) indexes for the first and last input from the Dataset 2 
        mult (int): times to input the data
        transform: in case there is an aditional transformation to apply to the data, it must be given
    
        """        
        self.transform = transform
        # list of image files (pre and post fire), and labels
        # label vector edge coordinates
        self.imgfiles = []
        self.imgprefiles=[]
        self.labels = []
        self.seglabels = []
        imgposfiles = []
        # read in segmentation label files
       
        for i in range(ss1,ss2):
            self.seglabels.append(os.path.join(dataset1_Firescar_tif_AS_paths, dataset.loc[i,"FireScar_tif"]))
            self.imgfiles.append(os.path.join(dataset1_PostFire_tif_AS_paths,dataset.loc[i,"ImgPosF"]))
            self.imgprefiles.append(os.path.join(dataset1_PreFire_tif_AS_paths,dataset.loc[i,"ImgPreF"]))
        
        for i in range(ss3,ss4):
            self.seglabels.append(os.path.join(dataset2_Firescar_tif_AS_paths,dataset.loc[i,"FireScar_tif"]))
            self.imgfiles.append(os.path.join(dataset2_PostFire_tif_AS_paths,dataset.loc[i,"ImgPosF"]))
            self.imgprefiles.append(os.path.join(dataset2_PreFire_tif_AS_paths,dataset.loc[i,"ImgPreF"]))
        
        self.imgfiles = np.array(self.imgfiles)
        self.imgprefiles=np.array(self.imgprefiles)
        self.labels = np.array(self.labels)
        
        if mult > 1:
            self.imgfiles = np.array([*self.imgfiles] * mult)
            self.imgprefiles = np.array([*self.imgprefiles] * mult)
            self.labels = np.array([*self.labels] * mult)
            self.seglabels = self.seglabels * mult
    def __len__(self):

        return len(self.imgfiles)
    def __getitem__(self, idx):
        idx=idx-1
        imgfile = rio.open(self.imgfiles[idx])
        imgpre=rio.open(self.imgprefiles[idx])
        imgdata1 = np.array([imgfile.read(i) for i in [1,2,3,4,5,6,7,8]])
        imgdatapre=np.array([imgpre.read(i) for i in [1,2,3,4,5,6,7,8]])
        new_array=np.concatenate((imgdata1, imgdatapre), axis=0)

        ds = gdal.Open(self.seglabels[idx])
        myarray = np.array(ds.GetRasterBand(1).ReadAsArray())

        if (np.isfinite(new_array)==False).any():                               #Replace nan for the neighbours mean values
            mask=np.where(np.isfinite(new_array))
            interp=NearestNDInterpolator(np.transpose(mask), new_array[mask])
            new_array=interp(*np.indices(new_array.shape))
            
        new_array=preprocessing(new_array)

        ds = gdal.Open(self.seglabels[idx])
        myarray = np.array(ds.GetRasterBand(1).ReadAsArray())

        x=imgdata1.shape[1]
        y=imgdata1.shape[2]
        imgdata=new_array

        size=128
        if (x<size or y<size):
            if (x%2==1 and y%2==1): #if it's odd
                new_array=np.pad(imgdata, ((0,0),(int((size-x)/2-1/2),int((size-x)/2+1/2)),(int((size-y)/2+1/2),int((size-y)/2-1/2))), "constant") #when it's odd, the padd goes 1 additional space left or down depending on the odd axis
            elif (x%2==1 and y%2==0):
                new_array=np.pad(imgdata, ((0,0),(int((size-x)/2-1/2),int((size-x)/2+1/2)),(int((size-y)/2),int((size-y)/2))), "constant") #when it's odd, the padd goes 1 additional space left or down depending on the odd axis
            elif (x%2==0 and y%2==1):
                new_array=np.pad(imgdata, ((0,0),(int((size-x)/2),int((size-x)/2)),(int((size-y)/2+1/2),int((size-y)/2-1/2))), "constant") #when it's odd, the padd goes 1 additional space left or down depending on the odd axis
            elif (x%2==0 and y%2==0):
                new_array=np.pad(imgdata, ((0,0),(int((size-x)/2),int((size-x)/2)),(int((size-y)/2),int((size-y)/2))), "constant") #wh

        x,y=myarray.shape

        if (x<size or y<size):
            if (x%2==1 and y%2==1): #if it's odd
                myarray=np.pad(myarray, ((int((size-x)/2-1/2),int((size-x)/2+1/2)),(int((size-y)/2+1/2),int((size-y)/2-1/2))), "constant") #when it's odd, the padd goes 1 additional space left or down depending on the odd axis
            elif (x%2==1 and y%2==0):
                myarray=np.pad(myarray, ((int((size-x)/2-1/2),int((size-x)/2+1/2)),(int((size-y)/2),int((size-y)/2))), "constant") #when it's odd, the padd goes 1 additional space left or down depending on the odd axis
            elif (x%2==0 and y%2==1):
                myarray=np.pad(myarray, ((int((size-x)/2),int((size-x)/2)),(int((size-y)/2+1/2),int((size-y)/2-1/2))), "constant") #when it's odd, the padd goes 1 additional space left or down depending on the odd axis
            elif (x%2==0 and y%2==0):
                myarray=np.pad(myarray, ((int((size-x)/2),int((size-x)/2)),(int((size-y)/2),int((size-y)/2))), "constant") #wh
    
        sample = {'idx': idx,
            'img': new_array,
            'fpt': myarray,
            'imgfile': self.imgfiles[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        """
        Returns converted Tensor sample
        
        sample: sample to be converted to Tensor
        
        """
        out = {'idx': sample['idx'],
        'img': torch.from_numpy(sample['img'].copy()),
        'fpt': torch.from_numpy(sample['fpt'].copy()),
        'imgfile': sample['imgfile']}

        return out
    
class Randomize(object):
    """Randomize image orientation including rotations by integer multiples of
    90 deg, (horizontal) mirroring, and (vertical) flipping."""
    def __call__(self, sample):
        """
        Randomizes the sample
        
        sample: sample to be randomized
        
        """
        imgdata = sample['img']
        fptdata = sample['fpt']
        idx=sample["idx"]
        # mirror horizontally
        mirror = np.random.randint(0, 2)
        if mirror:
            imgdata = np.flip(imgdata, 2)
            fptdata = np.flip(fptdata, 1)
        # flip vertically
        flip = np.random.randint(0, 2)
        if flip:
            imgdata = np.flip(imgdata, 1)
            fptdata = np.flip(fptdata, 0)
        # rotate by [0,1,2,3]*90 deg
        rot = np.random.randint(0, 4)
        if rot:
            imgdata = np.rot90(imgdata, rot, axes=(1,2))
            fptdata = np.rot90(fptdata, rot, axes=(0,1))

        return {'idx': sample['idx'],
                'img': imgdata.copy(),
                'fpt': fptdata.copy(),
                'imgfile': sample['imgfile']}
    
class Normalize(object):
    """Normalize pixel values to the range [0, 1] measured using minmax-scaling"""    
    def __init__(self):
        #the limits are determined according to the Dataset's nature
        self.channel_means=np.array(mean_as)
        self.channel_std=np.array(std_as)
    def __call__(self, sample):
        """
        :param sample: sample to be normalized
        :return: normalized sample"""
#         sample['img'] = (sample['img']-self.channel_min.reshape(
#             sample['img'].shape[0], 1, 1))/(self.channel_max.reshape(
#             sample['img'].shape[0], 1, 1)-self.channel_min.reshape(
#             sample['img'].shape[0], 1, 1))
        sample['img'] = (sample['img']-self.channel_means.reshape(
        sample['img'].shape[0], 1, 1))/self.channel_std.reshape(
        sample['img'].shape[0], 1, 1)
        return sample 

def create_datasetAS(*args, apply_transforms=True, **kwargs):
        """
        Create a dataset; uses same input parameters as PowerPlantDataset.
        apply_transforms: if `True`, apply available transformation. Returns the data set
        
        """
        if apply_transforms:
            data_transforms = transforms.Compose([
                Normalize(),
                Randomize(),
                ToTensor()
            ])
        else:
            data_transforms = None

        data = firescardataset(*args, **kwargs,
                                            transform=data_transforms)

        return data

# -

# #### Training

def train_model(model, epochs, opt, loss, batch_size, mult):  
    """
    Trains the model with the data. 
    
    model (model): model instance
    dataset (object: pandas df): dataset 
    epochs (int): number of epochs to be trained
    opt (object): optimizer instance
    loss (object): loss function instance
    batch_size (int): batch size 
    mult (int): times to input the data
    
    """
    # Adjust these following parameters, where :
    # ss1_t, ss2_t (int): indexes Dataset 1 for the trainig
    # if there is only one Dataset, set all ss3_t, ss4_t, ss3_v and ss4_v as 0. 
    # ss3_t, ss4_t (int) indexes Dataset 2 for the training 
    # ss1_v, ss2_v (int): indexes Dataset 1 for the validation
    # ss3_v, ss4_v (int): indexes Datset 2 for the validation
    data_train_ = create_datasetAS(data_train, 0, len(data_train1),
                len(data_train1),len(data_train1)+len(data_train2), mult=1)
    data_val_ = create_datasetAS(data_val, 0, len(data_val1), len(data_val1), len(data_val1)+len(data_val2), mult=1)
    train_dl = DataLoader(data_train_, batch_size, num_workers=0, pin_memory=True) #drop_last=True)
    val_dl = DataLoader(data_val_, batch_size, num_workers=0, pin_memory=True) # drop_last=True)  
    filename=""   # ending of the model filename
    best_model={}
    best_model["val_loss_total"]=100
    best_dc = {}
    best_dc["val_DC"]=0
    i=j=0
    def dice2d(pred, targs):  
        """
        Returns the input's Dice Coefficient metric

        pred (object): Object conformed by the binary output (prediction)   
        targs (object): Object conformed by the binary ground truth (firescar of reference)
        
        """
        pred = pred.squeeze()
        targs = targs.squeeze()
        return 2. * (pred*targs).sum() / (pred+targs).sum()
    # start training
    for epoch in range(epochs):
        model.train()
        #metrics 
        dicec_train_acc=[]
        FN_train=[]
        TP_train=[]
        FP_train=[]
        #train_acc_total=0
        train_loss_total = 0
        train_ious = []
        progress = tqdm(enumerate(train_dl), desc="Train Loss: ",
                        total=len(train_dl))
        for i, batch in progress:
            # try:
            x = batch['img'].float().to(device)
            y = batch['fpt'].float().to(device)
                                                                                
            output = model(x)

            # derive binary segmentation map from prediction
            output_binary = np.zeros(output.shape)
            output_binary[output.cpu().detach().numpy() >= 0] = 1
            
            # derive IoU values            
            for j in range(y.shape[0]):                                       
                z = jaccard_score(y[j].flatten().cpu().detach().numpy(),        
                        output_binary[j][0].flatten())
                if (np.sum(output_binary[j][0]) != 0 and
                    np.sum(y[j].cpu().detach().numpy()) != 0):
                    train_ious.append(z)
                    TP_train.append((output_binary.squeeze()*y.cpu().detach().numpy().squeeze()).sum())
                    FN_train.append(((output_binary.squeeze()==0) & (y.cpu().detach().numpy().squeeze()==1)).sum())
                    FP_train.append(((output_binary.squeeze()==1) & (y.cpu().detach().numpy().squeeze()==0)).sum())
                    dicec_train_acc.append(dice2d(output_binary,y.cpu().detach().numpy()))

            # derive scalar binary labels on a per-image basis
            y_bin = np.array(np.sum(y.cpu().detach().numpy(),
                                    axis=(1,2)) != 0).astype(int)
            pred_bin = np.array(np.sum(output_binary,
                                    axis=(1,2,3)) != 0).astype(int)

            # derive image-wise accuracy for this batch
            #train_acc_total += accuracy_score(y_bin, pred_bin)
            # derive loss                                                       
            loss_epoch = loss(output, y.unsqueeze(dim=1))
            train_loss_total += loss_epoch.item()
            progress.set_description("Train Loss: {:.4f}".format(
                train_loss_total/(i+1)))

            # learning
            opt.zero_grad()
            loss_epoch.backward()
            opt.step()
            
        # logging
        writer.add_scalar("training DC", np.average(dicec_train_acc),epoch)
        writer.add_scalar("training CE",  np.mean(FP_train)/(np.mean(TP_train)+np.mean(FP_train)), epoch)
        writer.add_scalar("training OE",  np.mean(FN_train)/(np.mean(TP_train)+np.mean(FN_train)), epoch)                         
        writer.add_scalar("training loss", train_loss_total/(i+1), epoch)
        writer.add_scalar("training iou", np.average(train_ious), epoch)
        #writer.add_scalar("training acc", train_acc_total/(i+1), epoch)
        writer.add_scalar('learning_rate', opt.param_groups[0]['lr'], epoch)
        torch.cuda.empty_cache()

        # evaluation
        model.eval()
        val_loss_total = 0
        val_ious = []
        #val_acc_total = 0
        
        dicec_eval_acc=[]
        FN_eval=[]
        TP_eval=[]
        FP_eval=[]
        
        progress = tqdm(enumerate(val_dl), desc="val Loss: ",
                        total=len(val_dl))
                        
        for j, batch in progress:
            x = batch['img'].float().to(device)
            y = batch['fpt'].float().to(device)
            output = model(x)

        # derive loss
            loss_epoch = loss(output, y.unsqueeze(dim=1))
            val_loss_total += loss_epoch.item()

        # derive binary segmentation map from prediction
            output_binary = np.zeros(output.shape)
            output_binary[output.cpu().detach().numpy() >= 0] = 1

        # derive IoU values
            ious = []
            for k in range(y.shape[0]):
                z = jaccard_score(y[k].flatten().cpu().detach().numpy(),
                        output_binary[k][0].flatten())
                if (np.sum(output_binary[k][0]) != 0 and 
                    np.sum(y[k].cpu().detach().numpy()) != 0):
                    val_ious.append(z)
                    TP_eval.append((output_binary.squeeze()*y.cpu().detach().numpy().squeeze()).sum())
                    FN_eval.append(((output_binary.squeeze()==0) & (y.cpu().detach().numpy().squeeze()==1)).sum())
                    FP_eval.append(((output_binary.squeeze()==1) & (y.cpu().detach().numpy().squeeze()==0)).sum())
                    dicec_eval_acc.append(dice2d(output_binary,y.cpu().detach().numpy()))
                
        # derive scalar binary labels on a per-image basis
            y_bin = np.array(np.sum(y.cpu().detach().numpy(),
                                axis=(1,2)) != 0).astype(int)
            pred_bin = np.array(np.sum(output_binary,
                                    axis=(1,2,3)) != 0).astype(int)

        # derive image-wise accuracy for this batch
        #val_acc_total += accuracy_score(y_bin, pred_bin)
            
            progress.set_description("val Loss: {:.4f}".format(
            val_loss_total/(j+1)))

        # logging
        writer.add_scalar("val DC", np.average(dicec_eval_acc),epoch)
        writer.add_scalar("val CE",  np.mean(FP_eval)/(np.mean(TP_eval)+np.mean(FP_eval)), epoch)
        writer.add_scalar("val OE",  np.mean(FN_eval)/(np.mean(TP_eval)+np.mean(FN_eval)), epoch)
        writer.add_scalar("val loss", val_loss_total/(j+1), epoch)
        writer.add_scalar("val iou", np.average(val_ious), epoch)
        #writer.add_scalar("val acc", val_acc_total/(j+1), epoch) 


        print(("Epoch {:d}: train loss={:.3f}, val loss={:.3f}, "
            "train iou={:.3f}, val iou={:.3f}, "
            "DC training={:.3f}, val DC={:.3f}").format(
                epoch+1, train_loss_total/(i+1), val_loss_total/(j+1),
                np.average(train_ious), np.average(val_ious),np.average(dicec_train_acc),
                    np.average(dicec_eval_acc)))

        if (val_loss_total/(j+1))<best_model["val_loss_total"]:
            best_model["val_loss_total"]=(val_loss_total/(j+1))
            best_model["epoch"]=epoch
        if (np.average(dicec_eval_acc))>best_dc["val_DC"]:
            best_dc["val_DC"]=np.average(dicec_eval_acc)
            best_dc["epoch"]=epoch
            
        #if epoch % 1 == 0:
        #uncomment to save the model files
            #torch.save(model.state_dict(),
            #'U_Net/runs/ep{:0d}_lr{:.0e}_bs{:02d}_{:03d}_{}.model'.format(
            #   args.ep, args.lr, args.bs, epoch, filename))

        writer.flush()
        scheduler.step(val_loss_total/(j+1))
        torch.cuda.empty_cache()
    print("best model: epoch (file): {}, val loss: {}".format(best_model["epoch"], best_model["val_loss_total"]))
    # print("best model_dc: epoch (file): {}, val dc: {}".format(best_dc["epoch"], best_dc["val_DC"])) #uncomment
    
    return model

# +

if __name__ == '__main__':
    args = get_train_args()
    print(f'ep: {args.ep}, bs: {args.bs}, lr: {args.lr}')

    # setup tensorboard writer
    writer = SummaryWriter('U_Net/runs/'+"ep{:0d}_lr{:.0e}_bs{:03d}/".format(
        args.ep, args.lr, args.bs))

    # initialize loss function
    loss = nn.BCEWithLogitsLoss()

    # initialize optimizer
    # opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mo) #for SGD optimizer
    opt = optim.Adam(model.parameters(), lr=args.lr)

    # initialize scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min',
                                                    factor=0.5, threshold=1e-4,
                                                    min_lr=1e-6)
    # -
    
    model.load_state_dict(torch.load(model_AS_path, map_location=torch.device('cpu')))
    
    # run training

    train_model(model, args.ep, opt, loss, args.bs, 1)
    writer.close()
        
