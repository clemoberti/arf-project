import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from sklearn.feature_extraction.image import extract_patches_2d

class ImageProcessing:
    
    #  permet de lire une image et de la renvoyer sous forme d’un numpy.array
    def read_im(self,fn):
        image = np.array(plt.imread(fn))
        if image.shape[2] != 3: # l'image dovait étre h x h x 3
            image = image[:,:,0:3]
        h = min(image.shape[0:2]) # make sure image is squared
        return rgb_to_hsv(image[:h,:h,:])

    # Affichage de l’image (et des pixels manquants)
    def show_im(self,fn):
        plt.imshow(fn)
        plt.show()

    def get_all_patches_by_pixel(self,h, im):
        window_shape = (h, h)
        return extract_patches_2d(im, window_shape)
        #return B[0][0]
        
    # retourne le patch centré en (i, j) et de longueur h d’une image im
    def get_patch(self, i,j,h,image):
        window_start_i = i - h // 2 # left upper corner of the window
        window_start_j = j - h // 2
        window_end_i = i + h // 2# right lower corner of the window
        window_end_j = j + h // 2
        
        if len(image.shape) == 2: # only for testing with number matrix
            return image[window_start_i:window_end_i, window_start_j:window_end_j]
        
        return image[window_start_i:window_end_i, window_start_j:window_end_j,:]
    
    # window to vector
    def patch_to_vector(self,patch):
        x,y,z = patch.shape
        return patch.reshape(x*y*z)
    
    def vector_to_patch(self,vector):
        z = 3 # there are allways three colors
        n = vector.shape[0]
        x = int(np.sqrt(n // z))
        y = x # let's suppose that given patch is allways square
        return vector.reshape((x,y,z))
    
    def noise(self, image, proportion):
        """
        proportion: quantite des pixels qu'on va mettre à noir, par exemple 0.4
        """
        assert 0 < proportion and proportion < 1, "The proportion must be between 0 and 1"
        n,m,_ = image.shape
        pixels = n*m
        black_pixels = int(pixels * proportion)
        vector_indexes = np.random.choice(pixels, black_pixels, replace=False)
        
        noise_image = image.copy()
        for index in vector_indexes:
            i,j = np.unravel_index(index, (n,m))
            noise_image[i,j,:] = 0 # let's put all in HSV to 0 to get black for sure
        return noise_image
    
    def delete_rect(self,img,i,j,height,width):
        new_image = img.copy()
        new_image[i:i+height, j:j+width,:] = 0
        return new_image
    
    def create_grid(self, min_x,max_x,min_y,max_y, step):
        """
        grid explainded : https://stackoverflow.com/questions/36013063/what-is-purpose-of-meshgrid-in-python
        """
        xvalues = np.arange(min_x, max_x, step);
        yvalues = np.arange(min_y, max_y, step);

        xx, yy = np.meshgrid(xvalues, yvalues)
        return np.array([xx.flatten(), yy.flatten()]).T
    
    def get_incomplete_patches_by_pixel(self, img, h):
        contains_zero  = lambda patch: (patch[:,:] == 0).any() 
        return np.array([patch for patch in self.get_all_patches_by_pixel(h, img) if contains_zero(patch)])
    
    def get_dictionnary_patches_by_pixel(self, img, h):
        not_contain_zero = lambda patch: (patch[:,:] != 0).all() 
        return np.array([patch for patch in self.get_all_patches_by_pixel(h, img) if not_contain_zero(patch)])
    
    def get_all_patches(self, im, h, step):
        margin = h // 2
        N = im.shape[0]
        max_x = N - margin + 1 if N % 2 == 0 else N - margin
        max_y = N - margin + 1 if N % 2 == 0 else N - margin
        grid = self.create_grid(margin,max_x, margin,max_y, step)
        return np.array([self.get_patch(pair[1], pair[0], h, im) for pair in grid])
    
    def get_incomplete_patches(self, img, h, step):
        contains_zero  = lambda patch: (patch[:,:] == 0).any() 
        return np.array([patch for patch in self.get_all_patches(img,h, step) if contains_zero(patch)])
    
    def get_dictionary_patches(self, img, h, step):
        """
        Get all patche and convert to vectors.
        """
        #not_contain_zero = lambda patch: (patch[:,:] != 0).all() filtrage plus tard
        return np.array([self.patch_to_vector(patch) \
                         for patch in self.get_all_patches(img,h, step)]).T #if not_contain_zero(patch)]).T
    
    def reconstruct_image(self,dictionary, h, step):
        patches_n = dictionary.shape[1]
        N = int(np.sqrt(patches_n))
        horizontals = []
        for i in range(N):
            start = i * N
            end = start + N
            #dictionary[:, start:end]
            horizontal = np.hstack([self.vector_to_patch(dictionary[:,i]) for i in range(start, end)])
            horizontals.append(horizontal)
        
        return np.vstack(horizontals)
