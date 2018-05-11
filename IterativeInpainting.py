
from ImageProcessing import ImageProcessing
import numpy as np
from q2_2 import estimate_patch

class IterativeInpainting:
    
    def __init__(self, image,patch_sizes=32, step_size=32):
        self.imp = ImageProcessing() # this one shouldn't maybe be class but 'static' file :D
        self.dictionary = self.imp.get_dictionary_patches(image, patch_sizes, step_size)
        
    def inpaint(self, max_iter=10, alpha=1):
        
        # politique le plus simple pour remplir le image : dans l'ordre
        for i in range(max_iter):
            patch = self.dictionary[:, i]
            new_patch = estimate_patch(patch, self.dictionary, alpha)
            
            #this shouldn't happen but dont know why
            if np.count_nonzero(new_patch > 1) > 0: # theres some too big values
                max_values = np.ones(len(new_patch))
                new_patch = np.min([new_patch, max_values],axis=0)
            if np.count_nonzero(new_patch < 0) > 0: # theres some too small values
                min_values = np.zeros(len(new_patch))
                new_patch = np.max([new_patch, min_values],axis=0)
            
            # update missing pixels
            pixels_to_update = patch == 0
            patch[pixels_to_update] = new_patch[pixels_to_update]
            self.dictionary[:,i] = patch
            
        return self.dictionary
            