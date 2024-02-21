#   Author: Ankit Kariryaa, University of Bremen
  


import numpy as np

def image_normalize(im, axis = (0,1), c = 1e-8):
    '''
    Normalize to zero mean and unit standard deviation along the given axis'''
    return (im - im.mean(axis)) / (im.std(axis) + c)
   
 
# Each area (GSW, annotation) is represented as an Frame
class FrameInfo:
    """ Defines a frame, includes its constituent images, annotation.
    """
    def __init__(self, img, annotations, meta, dtype=np.float32):
        """FrameInfo constructor.
        Args:
            img: ndarray
                3D array containing various input channels.
            annotations: ndarray
                3D array containing human labels, height and width must be same as img.
            dtype: np.float32, optional
                datatype of the array.
        """
        self.img = img
        self.annotations = annotations

        self.meta = meta # new adding meta
        self.dtype = dtype

    # Normalization takes a probability between 0 and 1 that an image will be locally normalized.
    def getPatch(self, i, j, patch_size, img_size, normalize=1):
        """Function to get patch from the given location of the given size.  
        Args:
            i: int
                Starting location on first dimension (x axis).
            j: int
                Starting location on second dimension (y axis).
            patch_size: tuple(int, int)  
                Size of the patch.
            img_size: tuple(int, int)
                Total size of the images from which the patch is generated. 
        """
        patch = np.zeros(patch_size, dtype=self.dtype)
    
        im = self.img[i:i + img_size[0], j:j + img_size[1]] 
        r = np.random.random(1)#Return random floats in the half-open interval [0.0, 1.0).
        if normalize >= r[0]:
            im = image_normalize(im, axis=(0, 1))
        
        an = self.annotations[i:i + img_size[0], j:j + img_size[1]]
        an = np.expand_dims(an, axis=-1)#  (512,512, 1) 
        comb_img = np.concatenate((im, an), axis=-1)  #(512,512, 2)  
        patch[:img_size[0], :img_size[1], ] = comb_img  # (512,512,2)
        return (patch)
    
    
    
    def sequential_patches(self, patch_size, step_size, normalize):
        """All sequential patches in this frame.

        Args:
            patch_size: tuple(int, int)
                Size of the patch.
            step_size: tuple(int, int)
                Total size of the images from which the patch is generated.
            normalize: float
                Probability with which a frame is normalized.
        """
        img_shape = self.img.shape
        x = range(0, img_shape[0] - patch_size[0], step_size[0])#0,1024,...
        y = range(0, img_shape[1] - patch_size[1], step_size[1])#0,1024,...
        if (img_shape[0] <= patch_size[0]):
            x = [0]
        if (img_shape[1] <= patch_size[1]):
            y = [0]

        ic = (min(img_shape[0], patch_size[0]), min(img_shape[1], patch_size[1]))
        xy = [(i, j) for i in x for j in y]
        img_patches = []
        for i, j in xy:
            img_patch = self.getPatch(i, j, patch_size, ic, normalize)
            img_patches.append(img_patch)
        # print(len(img_patches))
        return (img_patches, xy)
    

    # Returns a single patch, startring at a random image
    def random_patch(self, patch_size, normalize):
        """A random from this frame.
        Args:
            patch_size: tuple(int, int)
                Size of the patch.
            normalize: float
                Probability with which a frame is normalized.
        """
        img_shape = self.img.shape
        if (img_shape[0] <= patch_size[0]):
            x = 0
        else:
            x = np.random.randint(0, img_shape[0] - patch_size[0])
        if (img_shape[1] <= patch_size[1]):
            y = 0
        else:
            y = np.random.randint(0, img_shape[1] - patch_size[1])
        ic = (min(img_shape[0], patch_size[0]), min(img_shape[1], patch_size[1]))
        img_patch = self.getPatch(x, y, patch_size, ic, normalize)
        return (img_patch)
    
        # Returns all patches in a image, sequentially generated
