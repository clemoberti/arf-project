
from ImageProcessing import ImageProcessing
import numpy as np
from q2_2 import estimate_patch

class IterativeInpainting:
    
    def __init__(self, image, patch_sizes=32, step_size=32):
        self.imp = ImageProcessing() # this one shouldn't maybe be class but 'static' file :D
        self.dictionary = self.imp.get_dictionary_patches(image, patch_sizes, step_size)
        
    def inpaint(self, max_iter=10):
        
        # politique le plus simple pour remplir le image : dans l'ordre
        for i in range(max_iter):
            patch = self.dictionary[:, i]
            
            new_patch = estimate_patch(patch, self.dictionary)
            self.dictionary[:,i] = new_patch
            
        
            