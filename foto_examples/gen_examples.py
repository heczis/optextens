"""
Generate images for numerical experiments.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imsave

def gen_const_marks_horiz(width, height, marks, mark_width=10):
    """
    Black marks on white background, distributed horizontally.
    """
    img = 255 * np.ones((width, height))
    for mark in marks:
        mark_start = mark-mark_width//2
        mark_end = mark_start + mark_width
        img[mark_start:mark_end, :] = np.zeros((mark_width, height))
    return img.T

def gen_sin_marks_horiz(width, height, marks, mark_width=10):
    """
    Greyscale marks with sin^2 intensity on white background, distributed
    horizontally.
    """
    img = 255 * np.ones((width, height))
    for mark in marks:
        mark_start = mark-mark_width//2
        mark_end = mark_start + mark_width
        x = np.linspace(0, np.pi, mark_width)
        y = np.linspace(0, 1, height)
        X, _ = np.meshgrid(x, y)
        img[mark_start:mark_end, :] = 255 * (1 - np.sin(X.T)**2)
    return img.T

def gen_multi_marks_horiz(width, height, marks, mark_width=10):
    """
    Greyscale marks with 1-.5*sin^2 intensity on white background, distributed
    horizontally.
    """
    img = 255 * np.ones((width, height))
    for mark in marks:
        mark_start = mark-mark_width//2
        mark_end = mark_start + mark_width
        x = np.linspace(0, np.pi, mark_width)
        y = np.linspace(0, 1, height)
        X, _ = np.meshgrid(x, y)
        img[mark_start:mark_end, :] = 255 * .5 * np.sin(X.T)**2
    return img.T

def gen_triangle_marks_horiz(width, height, marks, mark_width=10):
    """
    Greyscale marks with triangle intensity on white background, distributed
    horizontally.
    """
    img = 255 * np.ones((width, height))
    for mark in marks:
        mark_start = mark-mark_width//2
        mark_end = mark_start + mark_width
        x = np.linspace(0, 2, mark_width)
        y = np.linspace(0, 1, height)
        X, _ = np.meshgrid(x, y)
        img[mark_start:mark_end, :] = 255 * np.abs(X.T-1)
    return img.T

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--const', '-c', action='store_true',
                        help='Generate the constant-marks example')
    parser.add_argument('--sin', '-s', action='store_true',
                        help='Generate the sine-squared-marks example')
    parser.add_argument('--tria', '-t', action='store_true',
                        help='Generate the triangular-marks example')
    parser.add_argument('--multi', '-m', action='store_true',
                        help='Generate the multiple-extrema example')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    output_dir = os.path.dirname(__file__)

    if args.const:
        img = gen_const_marks_horiz(100, 11, [30, 70])
        plt.imshow(img)
        plt.show()
        imsave(os.sep.join([output_dir, 'horiz_const.png']), img)

    if args.sin:
        img = gen_sin_marks_horiz(100, 12, [20, 80])
        plt.imshow(img)
        plt.show()
        imsave(os.sep.join([output_dir, 'horiz_sin.png']), img)

    if args.tria:
        img = gen_triangle_marks_horiz(100, 12, [25, 85])
        plt.imshow(img)
        plt.show()
        imsave(os.sep.join([output_dir, 'horiz_tria.png']), img)

    if args.multi:
        img = gen_multi_marks_horiz(100, 12, [35, 60])
        plt.imshow(img)
        plt.show()
        imsave(os.sep.join([output_dir, 'horiz_multi.png']), img)
