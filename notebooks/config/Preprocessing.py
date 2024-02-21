# Author: Ankit Kariryaa, University of Bremen.
# Modified by Xuehui Pi and Qiuqi Luo

import os

# Configuration of the parameters for the 1-Preprocessing.ipynb notebook
class Configuration:
    '''
    Configuration for the first notebook.
    Copy the configTemplate folder and define the paths to input and output data. Variables such as raw_GSW_image_prefix may also need to be corrected if you are use a different source.
    '''
    def __init__(self):
        # For reading the training areas and polygons 
        self.training_base_dir = r'D:\lakemapping\U_Net_GSW'
        self.training_area_fn = r'SampleAnnotations\NormalSmaple\area.shp'         
        self.training_polygon_fn = r'SampleAnnotations\NormalSmaple\polygons.shp' 

        # For reading images
        self.bands = [0]# If raster has multiple channels, then bands will be [0, 1, ...] otherwise simply [0]
        self.raw_image_base_dir = r'E:\lakemapping\occurrence'
        self.raw_image_file_type = '.tif'
        self.raw_GSW_image_prefix = 'occurrence'

        self.show_boundaries_during_processing = False
        self.extracted_file_type = '.png'
        self.extracted_GSW_filename = 'occurrence'
        self.extracted_annotation_filename = 'annotation'