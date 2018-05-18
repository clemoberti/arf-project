
from ImageProcessing import ImageProcessing
import numpy as np
from sklearn.linear_model import Lasso

class IterativeInpainting:

    def __init__(self, image, patch_sizes=32, step_size=32):
        self.imp = ImageProcessing() # this one shouldn't maybe be class but 'static' file :D
        self.N = image.shape[0]
        self.patch_sizes = patch_sizes
        self.step_size = step_size
        # Used for trying something out
        self.im = image.copy()
        # Used to compute confidence
        self.im_width = image.shape[0]
        self.im_height = image.shape[1]
        self.h = patch_sizes // 2
        self.m_confid = None # Matrix of confidence init with 0 for missing pixels and 1s otherwise

    def contains_missing_values(self,patch):
        return (patch == 0).any()

    def inpaint(self, alpha=1):
        # clean_dico_indexes = self.imp.complet_dictionary(self.dictionary)
        print("start painting")
        # politique le plus simple pour remplir le image : dans l'ordre

        while self.some_pixels_are_missing():
            patch_original, i, j = self.get_next_patch()
            patch = self.imp.patch_to_vector(patch_original)
            dictionary = self.imp.get_dictionary_patches(self.im, self.patch_sizes, self.step_size)
            ### calcule poids :

            non_null_indexes = np.where(patch != -100)[0]
            # for learning, use only nonzero examples
            Y = patch[non_null_indexes].reshape(-1,1)
            X = dictionary[non_null_indexes, :]

            model = Lasso(alpha=alpha)
            model = model.fit(X, Y) # coefficient sparse
            coef = np.array(model.coef_)
            #prediction

            new_patch = np.sum([coef[i] * dictionary[:,i] for i in range(len(coef))], axis=0)

            ############################

            pixels_to_update = patch == -100
            patch[pixels_to_update] = new_patch[pixels_to_update]
            new_patch = self.imp.vector_to_patch(new_patch)

            # import pdb; pdb.set_trace()
            # pixels_to_update_patch = patch_original

            for i2 in range(self.patch_sizes):
                for j2 in range(self.patch_sizes):
                    if (patch_original[i2,j2] == -100).all():
                        self.im[i+i2,j+j2] = new_patch[i2,j2]
            # update missing pixels

        return self.im

    def get_image(self):
        return self.imp.reconstruct_image_by_grid(self.dictionary,self.grid,self.patch_sizes,self.N)

    def show_image(self, title=None):
        image = self.imp.reconstruct_image_by_grid(self.dictionary,self.grid,self.patch_sizes,self.N)
        self.imp.show_im(image, title=title)

    def some_pixels_are_missing(self):
        """
        return true if any pixel is still missing from the image
        """
        contains_zero = lambda patch: (patch == -100).any()
        for patch in self.im:
            if (contains_zero(patch)):
                return True
        return False

    def get_next_patch(self):
        edges = self.imp.get_edges(self.im, self.im.shape)
        i = edges[0][0] - self.patch_sizes // 2
        j = edges[0][1] - self.patch_sizes // 2
        return self.imp.get_patch(edges[0][0], edges[0][1], self.patch_sizes, self.im), i, j

    def computeConfidence(self, i, j):
        """
        Calculate the confidence of a given pixel(i, j), usually on the edge
        """
        confidence = 0

        # y_max represents the bottom side of the patch
        y_max = j + self.h if j + self.h < self.im_height - 1 else self.im_height - 1
        # we init y to the top of the patch
        y = j - self.h if j - self.h > 0 else 0
        while (y <= y_max):
            # x_max represents the right side of the patch
            x_max = i + self.h if i + self.h < self.im_width - 1 else self.im_width - 1
             # we init x to the left of the patch
            x = i - self.h if i - self.h > 0 else 0  # init
            while (x <= x_max):
                conf = conf + self.m_confid(y * self.im_width + x) # depends on how the matrix is represented matrix or vector
                x += 1
            y += 1

        return confidence / ((self.h * 2 + 1) * (self.h * 2 + 1))
