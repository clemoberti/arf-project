
from ImageProcessing import ImageProcessing
import numpy as np
from sklearn.linear_model import Lasso

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
        clean_dico_indexes = self.imp.complet_dictionary(self.dictionary)
        print("start painting")
        print("columns: ", str(columns))
        # politique le plus simple pour remplir le image : dans l'ordre
        for ite in range(max_iter):
            print(ite)
            for i in range(columns):
                patch = self.dictionary[:, i]
                
                if not self.contains_missing_values(patch):
                    continue
                
                dictionary = self.dictionary[:, clean_dico_indexes]
                
                ### calcule poids : 
               
                non_null_indexes = np.where(patch != -100)[0]

                # for learning, use only nonzero examples
                Y = patch[non_null_indexes].reshape(-1,1)
                X = dictionary[non_null_indexes, :]

                model = Lasso(alpha=alpha)
                model = model.fit(X, Y) # coefficient sparse
                coef = np.array(model.coef_)
                
                #prediction
                new_patch = dictionary.dot(coef)
                
                ############################
                
                print("non null coefs: ")
                print((coef != 0).sum() / len(coef))
                

                print("fix") 
                # update missing pixels
                pixels_to_update = patch == -100
                patch[pixels_to_update] = new_patch[pixels_to_update]
                self.dictionary[:,i] = patch
                
                clean_dico_indexes = np.append(clean_dico_indexes,i) # this patch is handled, take it to account in next iteration
            
        return self.dictionary
    
    def get_image(self):
        return self.imp.reconstruct_image_by_grid(self.dictionary,self.grid,self.patch_sizes,self.N)
    
    def show_image(self, title=None):
        image = self.imp.reconstruct_image_by_grid(self.dictionary,self.grid,self.patch_sizes,self.N)
        self.imp.show_im(image, title=title)