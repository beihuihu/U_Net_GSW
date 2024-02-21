# Author: Ankit Kariryaa, University of Bremen.
# Modified by Xuehui Pi and Qiuqi Luo

import os

# Configuration of the parameters for the 2-UNetTraining.ipynb notebook
class Configuration:
    def __init__(self):
        # Initialize the data related variables used in the notebook
        # For reading the GSW and annotated images generated in the Preprocessing step.
        # In most cases, they will take the same value as in the config/Preprocessing.py
        
        self.base_dir = '' 
        # The Normal model was consisted of type 1, 2, 3, and 4, while the Floodplain Model was made up of types 1, 3, 4, and 5.
        self.path_to_write1 = os.path.join(self.base_dir,'output\output1')
        self.path_to_write2 = os.path.join(self.base_dir,'output\output2')
        self.path_to_write3 = os.path.join(self.base_dir,'output\output3')
        self.path_to_write4 = os.path.join(self.base_dir,'output\output4')
        # self.path_to_write5 = os.path.join(self.base_dir,'output\output5')
        
        self.image_type = '.png'       
        self.GSW_fn = 'occurrence'
        self.annotation_fn = 'annotation'
        
        # Patch generation; from the training areas (extracted in the last notebook), we generate fixed size patches.
        # random: a random training area is selected and a patch is extracted from a random location inside that training area. Uses a lazy stratergy i.e. batch of patches are extracted on demand.
        # sequential: training areas are selected in the given order and patches extracted from these areas sequential with a given step size. All the possible patches are returned in one call.
        self.patch_generation_stratergy = 'random' # 'random' or 'sequential'    
        self.patch_size = (512,512,2) # Height * Width * (Input or Output) channels  
        # step_size = (128,128)# # When stratergy == sequential, then you need the step_size as well
        
        # The training areas are divided into training, validation and testing set. Note that training area can have different sizes, so it doesn't guarantee that the final generated patches (when using sequential stratergy) will be in the same ratio.
        self.test_ratio = 0.2
        self.val_ratio = 0.2
        
        # Probability with which the generated patches should be normalized  0 -> don't normalize,    1 -> normalize all  
        self.normalize =0.4 
        
        # The split of training areas into training, validation and testing set, is cached in patch_dir.
        # 
        self.patch_dir = os.path.join(self.base_dir,'patches{}'.format(self.patch_size[0]))       
        self.frames_json1 = os.path.join(self.patch_dir,'frames_list1.json') #Filename of the json where data is written.
        self.frames_json2 = os.path.join(self.patch_dir,'frames_list2.json')
        self.frames_json3 = os.path.join(self.patch_dir,'frames_list3.json')
        self.frames_json4 = os.path.join(self.patch_dir,'frames_list4.json')
        # self.frames_json5 = os.path.join(self.patch_dir,'frames_list5.json')

        # Shape of the input data, height*width*channel; Here channel is GSW
        self.input_shape = (512,512,1)
        self.input_image_channel = [0]
        self.input_label_channel = [1]

        # CNN model related variables used in the notebook
        self.BATCH_SIZE = 16 
        self.NB_EPOCHS = 250

        # number of validation images to use
        self.VALID_IMG_COUNT = 300        
        # maximum number of steps_per_epoch in training
        self.MAX_TRAIN_STEPS = 750 #steps_per_epoch=(num_train/batch_size)
        self.model_path = os.path.join(self.base_dir, 'saved_models/UNet') 