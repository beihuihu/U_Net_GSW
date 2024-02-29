# Author: Ankit Kariryaa, University of Bremen.
# Modified by Xuehui Pi and Qiuqi Luo

import os

# Configuration of the parameters for the 3-FinalRasterAnalysis.ipynb notebook
class Configuration:
    '''
    Configuration for the notebook where objects are predicted in the image.
    Copy the configTemplate folder and define the paths to input and output data.
    '''
    def __init__(self):
        
        self.input_image_dir = r'/media/nkd/backup/4_GSW/occurrence'
        self.input_image_type = '.tif'
        self.GSW_fn_st =  'occurrence'
        self.trained_model_path =r'/home/nkd/hbh/U_Net_GSW/saved_models/UNet/lakes_20240228-1124_AdaDelta_tversky_01_512.h5'
        print('self.trained_model_path:', self.trained_model_path)
        
        self.output_dir = '/media/nkd/backup/4_GSW/occurrence_result/tif'
        self.output_image_type = '.tif'
        self.output_prefix = 'det'  
        self.output_shapefile_type = '.shp'
        self.overwrite_analysed_files =False
        self.output_dtype='uint8'

        # Variables related to batches and model
        self.BATCH_SIZE =256# Depends upon GPU memory and WIDTH and HEIGHT (Note: Batch_size for prediction can be different then for training.
        self.WIDTH=512 # Should be same as the WIDTH used for training the model
        self.HEIGHT=512 # Should be same as the HEIGHT used for training the model
        self.STRIDE=256 # STRIDE = WIDTH   means no overlap；  STRIDE = WIDTH/2   means 50 % overlap in prediction 