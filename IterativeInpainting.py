
from ImageProcessing import ImageProcessing
import numpy as np
from q2_2 import estimate_patch

class IterativeInpainting:
    
    def __init__(self, image,patch_sizes=32, step_size=32):
        self.imp = ImageProcessing() # this one shouldn't maybe be class but 'static' file :D
        self.N = image.shape[0]
        self.patch_sizes = patch_sizes
        self.step_size = step_size
        self.grid, self.dictionary = self.imp.get_dictionary_patches(image, patch_sizes, step_size)
        
    def contains_missing_values(self,patch):
        return (patch == -100).any()
        
    def inpaint(self, max_iter=1, alpha=1):
        columns = self.dictionary.shape[1]
        print("columns : ", columns)
        # politique le plus simple pour remplir le image : dans l'ordre
        for ite in range(max_iter):
            print(ite)
            for i in range(columns):
                patch = self.dictionary[:, i]
                
                if not self.contains_missing_values(patch):
                    continue
                
                # TODO dont calcul this every time, instead update in every iteration
                clean_dico = self.imp.complet_dictionary(self.dictionary)
                new_patch = estimate_patch(patch, clean_dico, alpha)
                

                #this shouldn't happen but dont know why
                if np.count_nonzero(new_patch > 1) > 0: # theres some too big values
                    max_values = np.ones(len(new_patch))
                    new_patch = np.min([new_patch, max_values],axis=0)
                if np.count_nonzero(new_patch < 0) > 0: # theres some too small values
                    min_values = np.zeros(len(new_patch))
                    new_patch = np.max([new_patch, min_values],axis=0)

                
                # update missing pixels
                pixels_to_update = patch == -100
                patch[pixels_to_update] = new_patch[pixels_to_update]
                self.dictionary[:,i] = patch
            
        return self.dictionary
    
    def show_image(self, title=None):
        self.image = self.imp.reconstruct_image_by_grid(self.dictionary,self.grid,self.patch_sizes,self.N)
        self.imp.show_im(self.image, title=title)