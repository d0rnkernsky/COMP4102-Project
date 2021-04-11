"""
    Daniil Kulik 101138752

    reference: http://scarlet.stanford.edu/teach/index.php/Object_Removal#Seam_Carving_for_Object_Removal_2
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve

UPDATED_MASK = 'updated_mask.jpg'
INTERM_IMG = 'itermedite_img.jpg'


class ObjectRemover:
    """
        default patch size is defined in the paper
        expecting mask to have only 1 or 0, where 1 indicates a pixel to remove
    """

    def __init__(self, image, mask, patch_size=9):
        self.orig_image = image.astype('uint8')
        self.orig_mask = mask.astype('uint8')
        self.patch_size = patch_size

        self.normal_x_kernel = np.array([[.25, 0, -.25],
                                         [.5, 0, -.5],
                                         [.25, 0, -.25]])
        self.normal_y_kernel = np.array([[-.25, -.5, -.25],
                                         [0, 0, 0],
                                         [.25, .5, .25]])

        self.img_in_progress = None
        self.active_mask = None
        self.fill_border = None
        self.priority = None
        self.data = None
        self.confidence = None

    def do(self):
        """
            Main loop
        """

        # initialize state
        self.data = np.zeros(self.orig_image.shape[:2])
        self.confidence = (1 - self.orig_mask).astype(float)
        # cv.imshow('Confidence', self.confidence)

        self.active_mask = np.copy(self.orig_mask)
        self.img_in_progress = np.copy(self.orig_image)

        done = False
        while not done:
            self.iteration_update()

            target = self.get_highest_priority()
            source = self.get_source_patch(target)
            self.inpaint_image(target, source)

            done = self.is_complete()
            # cv.imwrite(f'{self.path}/{UPDATED_MASK}', self.active_mask)
            # cv.imwrite(f'{self.path}/{INTERM_IMG}', self.img_in_progress)

        return self.img_in_progress.copy()

    def iteration_update(self):
        """
            Iteration updates
        """
        self.show_progress()
        self.set_fill_border()
        self.update_priorities()

    def show_progress(self):
        """
            Displays image in progress
        """
        mask_indx = np.argwhere(self.active_mask == 1)
        img = self.img_in_progress.copy()

        for i in range(mask_indx.shape[0]):
            x, y = mask_indx[i]
            img[x, y, :] = 255.0

        plt.imshow(img)
        plt.draw()
        plt.pause(0.0001)

    def set_fill_border(self):
        """
            Detects border around the working mask
        """
        border = cv.Laplacian(self.active_mask, cv.CV_64F)
        # detect mask border with Laplacian and save it
        self.fill_border = (border > 0).astype('uint8')

    def update_priorities(self):
        """
            Update priorities for new iteration of inpainting
        """
        self.recalc_confidence()
        self.update_data()
        self.priority = self.confidence * self.data * self.fill_border

    def recalc_confidence(self):
        """
            Recalc confidence values
        """
        updated = self.confidence.copy()
        front = np.argwhere(self.fill_border == 1)

        for p in front:
            patch = self.find_patch(p)
            x, y = p

            updated[x, y] = sum(sum(self.get_patch_data(self.confidence, patch))) \
                            / self.get_patch_area(patch)

        self.confidence = updated

    def get_patch_data(self, source, patch):
        """
            Returns patch pixels from the source
        """
        x, x1 = patch[0]
        y, y1 = patch[1]

        return source[x:x1 + 1, y:y1 + 1]

    def get_patch_area(self, patch):
        """
            Returns patch area based on path coordinates
        """
        px1, px2 = patch[0][0], patch[0][1]
        py1, py2 = patch[1][0], patch[1][1]

        return (1 + px2 - px1) * (1 + py2 - py1)

    def find_patch(self, point):
        """
            Returns patch rectangle
        """
        patch_half = (self.patch_size - 1) // 2

        height = self.img_in_progress.shape[0]
        width = self.img_in_progress.shape[1]

        px1 = max(0, point[0] - patch_half)
        py1 = min(point[0] + patch_half, height - 1)

        px2 = max(0, point[1] - patch_half)
        py2 = min(point[1] + patch_half, width - 1)

        return [[px1, py1], [px2, py2]]

    def update_data(self):
        """
            Updates data based on normal matrix and gradient
        """
        normal_gradient = self.get_normal_matrix() * self.find_gradient_matrix()
        epsilon = 0.001
        self.data = np.sqrt(normal_gradient[:, :, 0] ** 2 + normal_gradient[:, :, 1] ** 2) + epsilon

    def get_normal_matrix(self):
        """
            Returns normal matrix based on active mask
        """
        x_normal = cv.filter2D(self.active_mask.astype(float), -1, self.normal_x_kernel)
        y_normal = cv.filter2D(self.active_mask.astype(float), -1, self.normal_y_kernel)

        normal = np.dstack((x_normal, y_normal))

        height, width = normal.shape[:2]
        norm = np.reshape(np.sqrt(y_normal * y_normal + x_normal * x_normal),
                          (height, width, 1)) \
            .repeat(2, axis=2)
        norm[norm == 0] = 1
        return normal / norm

    def find_gradient_matrix(self):
        """
            Returns gradient matrix
        """
        height, width = self.img_in_progress.shape[:2]

        grey_image = cv.cvtColor(self.img_in_progress, cv.COLOR_BGR2GRAY).astype(float)
        grey_image[self.active_mask == 1] = None

        gradients = np.array(np.gradient(grey_image))
        gradient = np.nan_to_num(gradients)
        grad_val = np.sqrt(gradient[0] ** 2 + gradient[1] ** 2)
        grad = np.zeros([height, width, 2])

        positions = np.argwhere(self.fill_border == 1)

        # traverse at the inpainting region
        for point in positions:
            patch = self.find_patch(point)

            y_gradient = self.get_patch_data(gradient[0], patch)
            x_gradient = self.get_patch_data(gradient[1], patch)
            patch_grad = self.get_patch_data(grad_val, patch)

            patch_pos = np.unravel_index(np.argmax(patch_grad), patch_grad.shape)

            x, y = point
            grad[x, y, 0] = y_gradient[patch_pos]
            grad[x, y, 1] = x_gradient[patch_pos]

        return grad

    def get_highest_priority(self):
        """
            Returns the index of the pixel wit highest priority
        """
        return np.unravel_index(self.priority.argmax(), self.priority.shape)

    def get_source_patch(self, dst, pad=200):
        """
            Returns pixels for patch
        """
        target = self.find_patch(dst)
        height, width = self.img_in_progress.shape[:2]

        ph1, ph2 = target[0][0], target[0][1]
        pw1, pw2 = target[1][0], target[1][1]

        best = None
        best_match_difference = 0
        to_lab_image = cv.cvtColor(self.img_in_progress, cv.COLOR_RGB2LAB)

        patch_height = (ph2 - ph1 + 1)
        patch_width = (pw2 - pw1 + 1)

        for y in range(height - patch_height + 1):
            # we look at most pad pixels to left and right
            left_point = dst[1]
            left = 0 if left_point - pad < 0 else left_point - pad

            right = width - patch_width + 1 if left_point + pad > self.orig_image.shape[1] \
                else left_point + pad - patch_width + 1

            for x in range(left, right):
                source = [[y, y + patch_height - 1], [x, x + patch_width - 1]]
                if self.get_patch_data(self.active_mask, source).sum() != 0:
                    # because the inpainting region is filled with 1 the sum gives us its "area"
                    continue

                diff = self.get_patch_diff(to_lab_image, target, source)

                if best is None or diff < best_match_difference:
                    best_match_difference = diff
                    best = source

        return best

    def inpaint_image(self, target, source):
        """
            Sets patch into the target area
        """
        target_patch = self.find_patch(target)

        tp1, tp2 = target_patch[0][0], target_patch[1][0]
        points = np.argwhere(self.get_patch_data(self.active_mask, target_patch) == 1) + [tp1, tp2]
        # take all pixel with 1s

        x, y = target
        patch_confidence = self.confidence[x, y]
        for point in points:
            p1, p2 = point
            self.confidence[p1, p2] = patch_confidence
        mask = self.get_patch_data(self.active_mask, target_patch)
        to_rgb = mask.reshape(mask.shape[0], mask.shape[1], 1).repeat(3, axis=2)

        source_data = self.get_patch_data(self.img_in_progress, source)
        target_data = self.get_patch_data(self.img_in_progress, target_patch)

        patch = source_data * to_rgb + target_data * (1 - to_rgb)

        self.set_to_patch(self.img_in_progress, target_patch, patch)
        self.set_to_patch(self.active_mask, target_patch, 0)

    def get_patch_diff(self, image, target, source):
        """
            Returns how the distance between target and source patch
        """
        mask = 1 - self.get_patch_data(self.active_mask, target)

        h, w, = mask.shape
        to_rgb_mask = mask.reshape(h, w, 1).repeat(3, axis=2)

        target_data = self.get_patch_data(image, target) * to_rgb_mask
        source_data = self.get_patch_data(image, source) * to_rgb_mask
        d1 = ((target_data - source_data) ** 2).sum()

        t1, t2 = target[0][0], target[1][0]
        s1, s2 = source[0][0], source[1][0]

        d2 = np.sqrt((t1 - s1) ** 2 + (t2 - s2) ** 2)
        return d1 + d2

    def is_complete(self):
        """
            Check if there are pixel to be filled
        """
        # just sum all pixel if 1 is in the mask than the alogrithm is not done
        return self.active_mask.sum() == 0

    def set_to_patch(self, dst, dst_patch, data):
        """
             Set values in dst image at dst patch positions with data
        """
        dx1, dx2 = dst_patch[0][0], dst_patch[0][1]
        dy1, dy2 = dst_patch[1][0], dst_patch[1][1]

        dst[dx1:dx2 + 1, dy1:dy2 + 1] = data
