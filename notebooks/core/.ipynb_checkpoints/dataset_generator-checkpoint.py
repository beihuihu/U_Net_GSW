#   Author: Ankit Kariryaa, University of Bremen
  
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import rasterio

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.

def imageAugmentationWithIAA():
    sometimes = lambda aug, prob=0.5  : iaa.Sometimes(prob, aug)
    seq = iaa.Sequential([
        # Basic aug without changing any values
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images 
        iaa.Flipud(0.2),  # vertically flip 20% of all images 
        sometimes(iaa.Crop(percent=(0, 0.1))),  # random crops
        
        # # Gaussian blur and gamma contrast
        #sometimes(iaa.GaussianBlur(sigma=(0, 0.3)), 0.3),
        # sometimes(iaa.GammaContrast(gamma=0.5, per_channel=True), 0.3),

        # iaa.CoarseDropout((0.03, 0.25), size_percent=(0.02, 0.05), per_channel=True)#dropout
        # sometimes(iaa.Multiply((0.75, 1.25), per_channel=True), 0.3),
        sometimes(iaa.LinearContrast((0.3, 1.2)), 0.3),
        # iaa.Add(value=(-0.5,0.5),per_channel=True),
        sometimes(iaa.PiecewiseAffine(0.05), 0.3),
        sometimes(iaa.PerspectiveTransform(0.01), 0.1)
    ],
        random_order=True)
    return seq

class DataGenerator():
    """The datagenerator class. Defines methods for generating patches randomly and sequentially from given frames.
    """
    def __init__(self, input_image_channel, patch_size, frame_list, frames, annotation_channel = [1], augmenter=None):
        """Datagenerator constructor
        Args:
            input_image_channel (list(int)): Describes which channels is the image are input channels.   #input_image_channel = [0]
            patch_size (tuple(int,int)): Size of the generated patch.
            frame_list (list(int)): List containing the indexes of frames to be assigned to this generator.
            frames (list(FrameInfo)): List containing all the frames i.e. instances of the frame class. 
            augmenter  (string, optional): augmenter to use. None for no augmentation and iaa for augmentations defined in imageAugmentationWithIAA function.
            annotation_channel:annotation
        """
        self.input_image_channel = input_image_channel
        self.patch_size = patch_size
        self.frame_list = frame_list
        self.frames = frames
        self.annotation_channel = annotation_channel
        self.augmenter = augmenter
        
    # Return all training and label images, generated sequentially with the given step size
    def all_sequential_patches(self, step_size, normalize = 1):
        """Generate all patches from all assigned frames sequentially.

            step_size (tuple(int,int)): Size of the step when generating frames.
            normalize (float): Probability with which a frame is normalized.
        """
        patches = []
        patches_lonlat_ul = [] # new add transform
        for fn in self.frame_list:
            frame = self.frames[fn]
#             print('frame.meta:', frame.meta)
            ps, xys = frame.sequential_patches(self.patch_size, step_size, normalize)
            lonlats =  [rasterio.transform.xy(frame.meta['transform'], xy[0], xy[1], offset='ul') for xy in xys]
            patches.extend(ps)
            patches_lonlat_ul.extend(lonlats) # to restore the latlon of the upper left corner of the patches
        data = np.array(patches)
        img_lonlat_ul = np.array(patches_lonlat_ul)
        #print('img_lonlat_ul:', img_lonlat_ul)
        img = data[..., self.input_image_channel]
        y = data[..., self.annotation_channel] 
        ann = y[...,[0]]
        ann_joint=ann
        return (img, ann_joint, img_lonlat_ul) 

    # Return a batch of training and label images, generated randomly
    def random_patch(self, BATCH_SIZE, normalize):
        """Generate patches from random location in randomly chosen frames.
        Args:
            BATCH_SIZE (int): Number of patches to generate (sampled independently). 8
            normalize (float): Probability with which a frame is normalized.
        """
        patches = []
        for i in range(BATCH_SIZE):
            fn = np.random.choice(self.frame_list)
            frame = self.frames[fn]
            patch = frame.random_patch(self.patch_size, normalize)
            patches.append(patch)
        data = np.array(patches)
        img = data[..., self.input_image_channel]#self.input_image_channel = [0]
        ann_joint = data[..., self.annotation_channel]#[1]
        return (img, ann_joint)
#    print("Wrote {} random patches to {} with patch size {}".format(count,write_dir,patch_size)) #patch_size(256,256,3)
    
    
    # Normalization takes a probability between 0 and 1 that an image will be locally normalized.
    def random_generator(self, BATCH_SIZE, normalize = 1):
        """Generator for random patches, yields random patches from random location in randomly chosen frames.
        Args:
            BATCH_SIZE (int): Number of patches to generate in each yield (sampled independently).  
            normalize (float): Probability with which a frame is normalized.
        """
        seq = imageAugmentationWithIAA()

        while True:
            X, y = self.random_patch(BATCH_SIZE, normalize)
            if self.augmenter == 'iaa':   #augmenter = 'iaa'  
                seq_det = seq.to_deterministic()
                X = seq_det.augment_images(X)
    
                # y would have one channel, i.e. annotations. We need to augmenty for operations such as crop and transform
                y = seq_det.augment_images(y) 
                # Some augmentations can change the value of y, so we re-assign values just to be sure.
                ann =  y[...,[0]]
                ann[ann<0.5] = 0
                ann[ann>=0.5] = 1
                
                ann_joint=ann
                yield X, ann_joint
            else:
    
                ann =  y[...,[0]]
                ann_joint=ann
                yield X, ann_joint