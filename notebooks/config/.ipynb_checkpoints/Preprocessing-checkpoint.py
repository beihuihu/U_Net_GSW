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
        self.training_base_dir = ''
        self.training_area_fn = ''         
        self.training_polygon_fn = '' 

        # For reading images
        self.bands = [0]# If raster has multiple channels, then bands will be [0, 1, ...] otherwise simply [0]
        self.raw_image_base_dir = ''
        self.raw_image_file_type = '.tif'
        self.raw_GSW_image_prefix = 'occurrence'

        # For writing the extracted images and their corresponding annotations file.
        # The Normal model was consisted of type 1, 2, 3, and 4, while the Floodplain Model was made up of types 1, 3, 4, and 5.
        self.path_to_write1 = os.path.join(self.training_base_dir,'output\output1')
        self.path_to_write2 = os.path.join(self.training_base_dir,'output\output2')
        self.path_to_write3 = os.path.join(self.training_base_dir,'output\output3')
        self.path_to_write4 = os.path.join(self.training_base_dir,'output\output4')
        # self.path_to_write5 = os.path.join(self.training_base_dir,'output\output5')
        
        self.show_boundaries_during_processing = False
        self.extracted_file_type = '.png'
        self.extracted_GSW_filename = 'occurrence'
        self.extracted_annotation_filename = 'annotation'
        
        # Path to write should be a valid directory
        assert os.path.exists(self.path_to_write1)
        if not len(os.listdir(self.path_to_write1)) == 0:
            print('Warning: path_to_write1 is not empty! The old files in the directory may not be overwritten!!')
        assert os.path.exists(self.path_to_write2)
        if not len(os.listdir(self.path_to_write2)) == 0:
            print('Warning: path_to_write2 is not empty! The old files in the directory may not be overwritten!!')
        assert os.path.exists(self.path_to_write3)
        if not len(os.listdir(self.path_to_write3)) == 0:
            print('Warning: path_to_write3 is not empty! The old files in the directory may not be overwritten!!')
        assert os.path.exists(self.path_to_write4)
        if not len(os.listdir(self.path_to_write4)) == 0:
            print('Warning: path_to_write4 is not empty! The old files in the directory may not be overwritten!!')
        # assert os.path.exists(self.path_to_write5)
        # if not len(os.listdir(self.path_to_write5)) == 0:
        #     print('Warning: path_to_write5 is not empty! The old files in the directory may not be overwritten!!')