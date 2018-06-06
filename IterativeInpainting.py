from ImageProcessing import ImageProcessing
import numpy as np
from sklearn.linear_model import Lasso

class IterativeInpainting:

    def __init__(self, image, h=32, step_size=32):
        self.imp = ImageProcessing() # this one shouldn't maybe be class but 'static' file :D
        self.N = image.shape[0]
        self.step_size = step_size

        # Used for trying something out
        self.im = image.copy()
        # Used to compute confidence
        self.im_width = image.shape[0]
        self.im_height = image.shape[1]
        self.h = h
        image_vector = self.imp.patch_to_vector(image)

        # Matrix of confidence init with 0 for missing pixels and 1s otherwise
        self.m_conf = np.zeros(image_vector.shape)
        self.m_conf[image_vector != -100] = 1

    def contains_missing_values(self,patch):
        return (patch == 0).any()

    def inpaint(self, alpha=1):
        # clean_dico_indexes = self.imp.complet_dictionary(self.dictionary)
        print("start painting")
        # politique le plus simple pour remplir le image : dans l'ordre
        dictionary = self.imp.get_dictionary_patches(self.im, self.h, self.step_size)

        while self.some_pixels_are_missing():
            patch_original, i1, j1 = self.get_next_patch()
            patch = self.imp.patch_to_vector(patch_original)
            dictionary = self.imp.get_dictionary_patches(self.im, self.h, self.step_size)

            # for learning, use only nonzero examples
            non_null_indexes = np.where(patch != -100)[0]
            Y = patch[non_null_indexes].reshape(-1,1)
            X = dictionary[non_null_indexes, :]

            model = Lasso(alpha=alpha, max_iter=1000000)
            model = model.fit(X, Y) # coefficient sparse
            coef = np.array(model.coef_)

            # prediction
            # import pdb; pdb.set_trace()
            sum_coef = np.sum(coef[coef != 0])
            new_patch = np.divide(np.sum([coef[i] * dictionary[:,i] for i in range(len(coef))], axis=0), sum_coef)

            pixels_to_update = patch == -100
            patch[pixels_to_update] = new_patch[pixels_to_update]
            new_patch = self.imp.vector_to_patch(new_patch)

            # import pdb; pdb.set_trace()
            for i2 in range(self.h*2):
                for j2 in range(self.h*2):
                    if (patch_original[i2,j2] == -100).all():
                        self.im[i1+i2,j1+j2] = new_patch[i2,j2]
                        self.m_conf[i2 * self.im_width + j2] = 0.5

        return self.im

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

        max_coord = edges[0]
        max_conf = 0
        for current_point in edges:
            conf = self.computeConfidence(current_point[0], current_point[1])
            if conf > max_conf:
                max_conf = conf
                max_point = current_point
        i1 = max(max_coord[0] - self.h, 0)
        j1 = max(max_coord[1] - self.h, 0)
        return self.imp.get_patch(max_point[0], max_point[1], self.h, self.im), i1, j1

    def computeConfidence(self, i, j):
        """
        Calculate the confidence of a given pixel(i, j), usually on the edge
        """
        conf = 0

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
                conf = conf + self.m_conf[y * self.im_width + x] # depends on how the matrix is represented matrix or vector
                x += 1
            y += 1

        return conf / ((self.h * 2 + 1) * (self.h * 2 + 1))
