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

# ## Cropping AS & 128

# ### Cropping to the firescar size

# #### Importing libraries

# +
#AS
import os, csv, geopandas as gpd, rasterio as rio, rioxarray as rxr
from osgeo import gdal 
from pathlib import Path

#128
import os, csv, geopandas as gpd, rasterio as rio, rioxarray as rxr, pandas as pd, numpy as np
from osgeo import gdal 
from pathlib import Path
from geopandas import GeoDataFrame
from shapely.geometry import Point


# -

# #### Cropping AS

def to_shp(filename, destin_folder):
    """
    Transforms the raster to shp to enable the posterior cropping
    
    filename: firescar raster path   
    destin_folder: new shapefile path 
    
    """
    inDs=gdal.Open(filename)            
    outDs = gdal.Translate('{}.xyz'.format(destin_folder+"/"+Path(filename).stem), 
                           inDs, format='XYZ', creationOptions=["ADD_HEADER_LINE=YES"])
    outDs = None 
    try:
      os.remove('{}.csv'.format(destin_folder+"/"+Path(filename).stem))
    except OSError:
      pass
    os.rename('{}.xyz'.format(destin_folder+"/"+Path(filename).stem), '{}.csv'.format(destin_folder+"/"+Path(filename).stem))
    return os.system('ogr2ogr -f "ESRI Shapefile" -oo X_POSSIBLE_NAMES=X* -oo Y_POSSIBLE_NAMES=Y* -oo KEEP_GEOM_COLUMNS=NO {0}_2.shp {0}.csv'.format(destin_folder+"/"+Path(filename).stem))    




def cropping(filename, destin_folder, ipname, out_path): 
    """
    Crops the satellite raster to the firescar size using the firescar binary raster for the georeference
    
    filename (str): firescar raster path   
    destin_folder (str): new shapefile path 
    ipname (str): image pre/post raster filename path
    out_path (str): output path destination
    
    """
    to_shp(filename,destin_folder)
    fire_boundary_path = destin_folder+"/"+Path(filename).stem+"_2.shp"
    ippath=os.path.join(ipname) #ip name of raster file
    fire_boundary = gpd.read_file(fire_boundary_path)
    #Check crs
    ip_crs=rxr.open_rasterio(ipname).rio.crs
    fire_boundary.crs=ip_crs
    #cropping
    ip=rxr.open_rasterio(ippath, masked=False).squeeze()
    clip = rxr.open_rasterio(ippath).rio.clip(
        fire_boundary.geometry,
        from_disk=True).squeeze()
    #Export
    clip.rio.to_raster(out_path+"/"+Path(ipname).stem+"_cropAS.tif", compress='LZMA', dtype="float64")


# ### Cropping to 128x128

# #### Cropping 128

def cropping128(filename, destin_folder, ipname, output, size): 
    """
    Crops the satellite image raster to the desired size using another raster containing only the binary firescar 
    for the georeference. It uses only firescar rasters inferior to size in at least one of the two axis.
    
    filename (str): Path of the firescar raster, containing the geospatial information required for the cropping
    destin_folder (str): is the destination path for the csv file created with the filename geospatial information
    ipname (str): stands for the File name of the raster desired to crop to the size dimensions, in this case 128
    output (str): is the path for the cropped raster
    size (int): number of pixel to crop the image on its both axis to
    
    """
    file_=rxr.open_rasterio(filename)
    if (len(file_.y)<size or len(file_.x)<size):
        inDs=gdal.Open(filename)           
        ulx, xres, xskew, uly, yskew, yres  = inDs.GetGeoTransform()
        df=pd.read_csv(destin_folder+"/"+Path(filename).stem+".csv")
        df2=pd.DataFrame(columns=["X","Y"])
        idx=0
        for i in df.values:
            df2.loc[idx, "X"]=float(i[0].split(' ')[0])
            df2.loc[idx, "Y"]=float(i[0].split(' ')[1])
            idx+=1
        # print(f"initial size: {len(df2.X.unique()), len(df2.Y.unique())}")
        if ((len(df2.X.unique())<size) or (len(df2.Y.unique())<size)):
            if (len(df2.X.unique())<size and len(df2.Y.unique())>=size):
                newX=np.linspace(df2.X.min()-((size-(len(df2.X.unique())))/2)*xres,df2.X.max()+((size-(len(df2.X.unique())))/2)*xres,size)
                newY=np.linspace(df2.Y.min()+(((len(df2.Y.unique())-size))/2)*-yres,df2.Y.max()-(((len(df2.Y.unique())-size))/2)*-yres,size)
            elif (len(df2.Y.unique())<size and len(df2.X.unique())>=size):
                newY=np.linspace(df2.Y.min()-((size-(len(df2.Y.unique())))/2)*-yres,df2.Y.max()+((size-(len(df2.Y.unique())))/2)*-yres,size)
                newX=np.linspace(df2.X.min()+(((len(df2.X.unique())-size))/2)*xres,df2.X.max()-(((len(df2.X.unique())-size))/2)*xres,size)
            elif (len(df2.Y.unique())<size and len(df2.X.unique())<size):
                newX=np.linspace(df2.X.min()-((size-(len(df2.X.unique())))/2)*xres,df2.X.max()+((size-(len(df2.X.unique())))/2)*xres,size)
                newY=np.linspace(df2.Y.min()-((size-(len(df2.Y.unique())))/2)*-yres,df2.Y.max()+((size-(len(df2.Y.unique())))/2)*-yres,size)
        # print(f"new size x,y:{len(newX),len(newY)}")
        xx, yy = np.meshgrid(newX.tolist(), newY.tolist())
        newX = np.array(xx.flatten("C"))
        newY = np.array(yy.flatten("C"))
        df_r=pd.DataFrame(columns=["X", "Y"])
        df_r["X"]=newX
        df_r["Y"]=newY
        geometry = [Point(xy) for xy in zip(df_r.X, df_r.Y)]
        df_f = df_r.drop(['X', 'Y'], axis=1)
        gdf = GeoDataFrame(df_f, crs="EPSG:4326", geometry=geometry)
        #cropping
        ippath=os.path.join(ipname) #name of the raster file 
        fire_boundary= gdf  
        ip=rxr.open_rasterio(ippath, masked=True).squeeze()
        clip = rxr.open_rasterio(ippath).rio.clip(
        fire_boundary.geometry,
        from_disk=True).squeeze()
        print(f"size clip: {len(clip.x), len(clip.y)}")
        if (len(clip.x)==128 and len(clip.y)==128):
            #export 
            clip.rio.to_raster(output+"/"+Path(ipname).stem+"_crop.tif")
        #for issues with the cropping when pixels weren't in the exact border. It clips 1/2 aditional pixel down or left to obtain the right clip.
        elif (len(clip.x)==127 or len(clip.y)==127):
            if (len(clip.x)==127 and len(clip.y)==127):
                newX=np.linspace(df2.X.min()-((size-(len(df2.X.unique())))/2)*xres-xres*1/4,df2.X.max()+((size-(len(df2.X.unique())))/2)*xres-xres*1/4,size)
                newY=np.linspace(df2.Y.min()-((size-(len(df2.Y.unique())))/2)*-yres-(-yres*1/4),df2.Y.max()+((size-(len(df2.Y.unique())))/2)*-yres-(-yres*1/4),size)
            elif (len(clip.x)==127 and len(clip.y)==128):
                newX=np.linspace(df2.X.min()-((size-(len(df2.X.unique())))/2)*xres-xres*1/4,df2.X.max()+((size-(len(df2.X.unique())))/2)*xres-xres*1/4,size)
                newY=np.linspace(df2.Y.min()+(((len(df2.Y.unique())-size))/2)*-yres,df2.Y.max()-(((len(df2.Y.unique())-size))/2)*-yres,size)
            elif (len(clip.y)==127 and len(clip.x)==128):
                newY=np.linspace(df2.Y.min()-((size-(len(df2.Y.unique())))/2)*-yres-(-yres*1/4),df2.Y.max()+((size-(len(df2.Y.unique())))/2)*-yres-(-yres*1/4),size)
                newX=np.linspace(df2.X.min()-((size-(len(df2.X.unique())))/2)*xres,df2.X.max()+((size-(len(df2.X.unique())))/2)*xres,size)
            # print(f"new size x,y:{len(newX),len(newY)}")
            xx, yy = np.meshgrid(newX.tolist(), newY.tolist())
            newX = np.array(xx.flatten("C"))
            newY = np.array(yy.flatten("C"))
            df_r=pd.DataFrame(columns=["X", "Y"])
            df_r["X"]=newX
            df_r["Y"]=newY
            geometry = [Point(xy) for xy in zip(df_r.X, df_r.Y)]
            df_f = df_r.drop(['X', 'Y'], axis=1)
            gdf = GeoDataFrame(df_f, crs="EPSG:4326", geometry=geometry)
            ippath=os.path.join(ipname) #ip name of the raster file
            fire_boundary= gdf  
            # cropping
            ip=rxr.open_rasterio(ippath, masked=True).squeeze()
            clip = rxr.open_rasterio(ippath).rio.clip(
            fire_boundary.geometry,
            from_disk=True).squeeze()
            print(f"size clip_fixed: {len(clip.x), len(clip.y)}")
            #Export
            if (len(clip.x)==128 and len(clip.y)==128):
                clip.rio.to_raster(output+"/"+Path(ipname).stem+"_crop128.tif")

