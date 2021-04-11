# Author:
# Daniil Kulik 101138752
# Erica Warriner 101002942
# COMP 4102 Course Project

import sys
import os
import numpy as np
import cv2 as cv
import remove_helper as rh
import time
import tst
import matplotlib.pyplot as plt

PATH = ''
ESC_KEY = 27

x_init = 0
y_init = 0
drawing = False
top_left = 0
bottom_right = 0
# original image
orig_img = None
# image to draw selected area
img = None


def draw_rect(img, x_init, y_init, x, y):
    """
        Draws a rectangle in the selected area and inverts colors
    """
    # invert colors to show the selected area
    img[y_init:y, x_init:x] = 255 - orig_img[y_init:y, x_init:x]
    cv.rectangle(img, (x_init, y_init), (x, y), (0, 255, 0), 1)


def select_area(ev, x, y, _1, _2):
    """
        Event handler to track selected region
    """
    global x_init, y_init, drawing, top_left, bottom_right, orig_img, img

    if ev == cv.EVENT_LBUTTONDOWN:
        drawing = True
        x_init = x
        y_init = y
    elif ev == cv.EVENT_MOUSEMOVE and drawing:
        draw_rect(img, x_init, y_init, x, y)
    elif ev == cv.EVENT_LBUTTONUP:
        drawing = False
        draw_rect(img, x_init, y_init, x, y)

        # prepare mask
        mask = np.full(img.shape[:2], 0, dtype='uint8')

        print(f'Removed region is {y-y_init} by {x-x_init} pixels (HxW)')
        mask[y_init:y, x_init:x] = 1

        # mask_to_display = mask * 255.0
        # cv.imshow('mask', mask_to_display)

        helper = rh.ObjectRemover(orig_img, mask)
        # helper = tst.Inpainter(orig_img, mask)
        start = time.time()
        result = helper.do()

        cv.imwrite(f'result.jpg', result)
        plt.imshow(result)
        plt.draw()

        end = time.time() - start
        print(f'Total took: {end}')


def clean_up():
    """
        Deletes temp files 
    :return: 
    """
    if os.path.exists(f'{PATH}/{rh.UPDATED_MASK}'):
        os.remove(f'{PATH}/{rh.UPDATED_MASK}')
    if os.path.exists(f'{PATH}/{rh.INTERM_IMG}'):
        os.remove(f'{PATH}/{rh.INTERM_IMG}')


def main():
    if len(sys.argv) < 2:
        print('Usage: python3 object_removal.py /path/to/image')
        exit(-1)

    img_path = sys.argv[1]
    remove(img_path)


def remove(img_path):
    in_img = cv.imread(img_path)
    if in_img is None:
        print(f'Cannot find image {img_path}')
        exit(-1)

    clean_up()

    global img, orig_img

    orig_img = np.copy(in_img)
    img = np.copy(in_img)

    cv.namedWindow('image')
    cv.setMouseCallback('image', select_area)

    while True:
        cv.imshow('image', img)
        key = cv.waitKey(1)
        if key == ESC_KEY:
            break

    cv.destroyAllWindows()
    clean_up()


if __name__ == '__main__':
    main()
