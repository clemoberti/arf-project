import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from sklearn.feature_extraction.image import extract_patches_2d

class ImageProcessing:
    def __init__(self): #, patch_width, patch_height):
        # We can extend to non squared patchs later on
        self.missing_pixel_value = -100

    def read_im(self,fn):
        """
        Lit une image et de la renvoyer sous forme d’un numpy.array
        """
        image = np.array(plt.imread(fn))
        if image.shape[2] != 3: # l'image dovait étre h x h x 3
            image = image[:,:,0:3]
        h = min(image.shape[0:2]) # make sure image is squared
        return rgb_to_hsv(image[:h,:h,:])

    def show_im(self, fn, title=None):
        """
        Affichage de l’image (et des pixels manquants)
        """
        image = fn.copy()
        image[image == self.missing_pixel_value] = 0
        plt.imshow(hsv_to_rgb(np.abs(image)))
        if title != None:
            plt.title(title)
        plt.show()

    def get_patch(self, i, j, h, image):
        """
        retourne le patch centré en (i, j) et de longueur h d’une image im
        """
        window_start_i = i - h // 2 # left upper corner of the window
        window_start_j = j - h // 2
        window_end_i = i + h // 2# right lower corner of the window
        window_end_j = j + h // 2

        if len(image.shape) == 2: # only for testing with number matrix
            return image[window_start_i:window_end_i, window_start_j:window_end_j]

        return image[window_start_i:window_end_i, window_start_j:window_end_j,:]

    def patch_to_vector(self,patch):
        return patch.copy().reshape(-1)

    def vector_to_patch(self, vector):
        z = 3 # there are always three colors
        n = vector.shape[0]
        x = int(np.sqrt(n // z))
        y = x # let's suppose that a patch is always square
        return vector.copy().reshape((x,y,z))

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
            noise_image[i,j,:] = self.missing_pixel_value
        return noise_image

    def delete_rect(self,img,i,j,height,width):
        new_image = img.copy()
        new_image[i:i+height, j:j+width,:] = self.missing_pixel_value
        return new_image

    def create_grid(self, min_x,max_x,min_y,max_y, step):
        """
        grid explainded : https://stackoverflow.com/questions/36013063/what-is-purpose-of-meshgrid-in-python
        """
        xvalues = np.arange(min_x, max_x, step);
        yvalues = np.arange(min_y, max_y, step);

        xx, yy = np.meshgrid(xvalues, yvalues)
        return np.array([xx.flatten(), yy.flatten()]).T

    def get_all_patches(self, im, h, step):
        margin = h // 2
        N = im.shape[0]
        max_x = N - margin + 1 if N % 2 == 0 else N - margin
        max_y = N - margin + 1 if N % 2 == 0 else N - margin
        grid = self.create_grid(margin,max_x, margin,max_y, step)
        return np.array([self.get_patch(pair[1], pair[0], h, im) for pair in grid])

    def get_incomplete_patches(self, img, h, step):
        contains_zero  = lambda patch: (patch[:,:] == self.missing_pixel_value).any()
        return np.array([patch for patch in self.get_all_patches(img,h, step) if contains_zero(patch)])

    def get_dictionary_patches(self, img, h, step):
        """
        Get all patches and convert to vectors and scale values.
        """
        patches = self.get_all_patches(img,h, step)
        contains_not_zero  = lambda patch: (patch[:,:] != self.missing_pixel_value).all()
        return np.array([self.patch_to_vector(patch) for patch in patches if contains_not_zero(patch)]).T


    def complet_dictionary(self, dictionary):
        """
        return not-null indexes of patches in dictionary, eg what is the "real" dictionary according the annonce
        """
        # return np.array([dictionary]).T
        return np.array([i for i in range(dictionary.shape[1]) if (dictionary[:,i] != self.missing_pixel_value).all()]).T


    def reconstruct_image_by_grid(self,dictionary, grid, h, N):
        image = np.zeros((N,N,3))
        margin = h // 2
        for i,pair in enumerate(grid):
            x,y = pair
            image[y-margin:y+margin,x-margin:x+margin] = self.vector_to_patch(dictionary[:,i])
        return image

    def get_neighbors(self, i, j, im, N):
        """
        return all the neighbors surrounding
        """
        left = [im[i-1][j]]   if i != 0 else [[-1,-1,-1]]
        right = [im[i+1][j]]  if i + 1 < N[0] else [[-1,-1,-1]]
        top = [im[i][j-1]]    if j != 0 else [[-1,-1,-1]]
        bottom = [im[i][j+1]] if j + 1 < N[0] else [[-1,-1,-1]]
        return np.concatenate((left, right, bottom, top), axis=0)

    def is_near_the_edge(self, pixel, neighbors):
        """
        return true if the pixel is near the edge
        """
        if np.sum(pixel) == 3 * self.missing_pixel_value:
            return False
        return (neighbors == -100).any()
        # return np.count_nonzero(np.sum(neighbors, axis=1)) != np.array(neighbors).shape[0]

    def get_edges(self, im, n):
        """
        Return a list of indices around the edge
        """
        edgeList = []
        print(n)
        for i in range(n[0]):
            for j in range(n[1]):
                if self.is_near_the_edge(im[i][j], self.get_neighbors(i, j, im, n)):
                    edgeList.append((i,j))
        return edgeList

    def priority(self, i, j):
        """
        Return a heuristic based on the patch of the region
        """
        confidenceTerm = self.confidence(i, j)
        dataTerm = self.data(i, j)
        return confidenceTerm * dataTerm

    def data(self, i, j, α = 255):
        """
        α is a normalization factor
        """
        α = 255 #
        pass
