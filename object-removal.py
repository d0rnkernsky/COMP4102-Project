# Author:
# Daniil Kulik 101138752
# Erica Warriner 101002942
# COMP 4102 Course Project

import sys
import numpy as np
import cv2 as cv

x_init = 0
y_init = 0
drawing = False
top_left = 0
bottom_right = 0
orig_img = None
img = None


def draw_rect(img, x_init, y_init, x, y):
    # invert colors to show the selected area
    img[y_init:y, x_init:x] = 255 - orig_img[y_init:y, x_init:x]
    cv.rectangle(img, (x_init, y_init), (x, y), (0, 255, 0), 3)


def select_area(ev, x, y, flags, param):
    global x_init, y_init, drawing, top_left, bottom_right, orig_img

    if ev == cv.EVENT_LBUTTONDOWN:
        drawing = True
        x_init = x
        y_init = y
    elif ev == cv.EVENT_MOUSEMOVE and drawing:
        draw_rect(img, x_init, y_init, x, y)
    elif ev == cv.EVENT_LBUTTONUP:
        drawing = False
        draw_rect(img, x_init, y_init, x, y)
        area = (x_init, y_init, x - x_init, y - y_init)
        print("Implement remove from area")
        print(area)


def remove_selected(in_img):
    cv.imshow('Input', in_img)
    cv.imshow('Output', in_img)
    cv.waitKey()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 object-removal.py /path/to/image")
        exit(-1)

    img_path = sys.argv[1]
    in_img = cv.imread(sys.argv[1])

    if in_img is None:
        print(f"Cannot find image {orig_img}")
        exit(-1)

    orig_img = np.copy(in_img)
    img = np.copy(in_img)

    cv.namedWindow('image')
    cv.setMouseCallback('image', select_area)

    while True:
        cv.imshow('image', img)
        c = cv.waitKey(1)
        if c == 27:
            break

    cv.destroyAllWindows()
