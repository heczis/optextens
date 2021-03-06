"""
Module for working with visual data from mechanical experiments and computation
of movement and deformation based on high-contrast marks.
"""
from __future__ import print_function
import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
import scipy.optimize as sopt

import functions
import interpolation

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s: %(message)s')

class Channel:
    """
    Represents a single channel of color.
    """
    def __init__(self, data):
        """
        data : 2D array
        """
        self.data = data

    def get_line(self, start, end):
        """
        start, end : coordinates of the starting and ending point
        """
        line = interpolation.Line(start, end)
        slice_fun = interpolation.get_data_slice(self.data.T, line, outside_value=0.)
        return slice_fun

    def get_continuous(self, approx_fun=None):
        """
        Returns a continuous representation of the channel data.

        approx_fun : 2D approximation function
        """
        pass

    def to_unit(self):
        """
        Normalize a channel data to the [0, 1] interval
        """
        data = np.array(self.data, dtype=float)
        new_data = (data - np.min(data)) / (np.max(data) - np.min(data))
        return Channel(new_data)


class Image:
    """
    Represents an image (all channels).
    """
    def __init__(self, channels):
        """
        channels : list of Channel instances
        """
        self.channels = channels

    @staticmethod
    def from_file(file_name, **kwargs):
        """
        Load image from file using `scipy.misc.imread`.
        """
        image_data = imread(file_name, **kwargs)
        if len(image_data.shape) == 2:
            channels = [Channel(image_data),]
        else:
            channels = [Channel(image_data[:, :, ii])
                        for ii in range(image_data.shape[-1])]
        return Image(channels)

    def get_image(self):
        """
        Return image data as MxN[xP] array, where P - if anything - can be 3
        or 4.
        """
        p = len(self.channels)
        if p == 1:
            out = self.channels[0].data
        elif p in [3, 4]:
            m, n = self.channels[0].data.shape
            out = np.zeros((m, n, p), dtype=np.uint8)
            for pi, ch in enumerate(self.channels):
                out[:, :, pi] = ch.data
        else:
            out = None

        return out

    def get_line(self, start, end):
        """
        start, end : coordinates of the starting and ending point
        """
        return [channel.get_line(start, end) for channel in self.channels]

    def get_continuous(self, approx_fun=None):
        """
        Returns a continuous representation of all channels.

        approx_fun : 1D approximation function
        """
        return [channel.get_continuous(approx_fun)
                for channel in self.channels]

def get_marks(fun, threshold=.8, mark_thickness=10):
    """
    Returns position of marks in 1D.
    Assumes dark marks on light surface.

    Minimization method:
    1. Compute the integral of `fun` over the whole line.
    2. Find all intervals, where the values of `fun` are below `threshold*
       integral`.
    3. Use `scipy.minimize_scalar` (method 'Bounded') to find minimum on each
       subinterval.

    fun : piecewise constant function (pixel values)
    thershold : multiple of fun integral to split the interval into sub-intervals
    """
    fun_supp = functions.ConstFunction(
        nodes=[fun.nodes[0], fun.nodes[-1]],
        values=[1.,])
    integral_val = fun.integrate_all() / fun_supp.integrate_all()

    funs = fun.get_clusters_by_max(threshold*integral_val)
    out = []
    for f in funs:
        print('cluster: %f-%f' % (f.nodes[0], f.nodes[-1]))
        smooth_f = functions.smoothe(f, functions.get_window, mark_thickness)
        int_len = f.nodes[-1] - f.nodes[0]
        print(' bounds: %f-%f' % (f.nodes[0] - .5*int_len, f.nodes[-1] + .5*int_len))
        res = sopt.minimize_scalar(
                smooth_f,
                bounds=[f.nodes[0] - .5*int_len, f.nodes[-1] + .5*int_len], method='Bounded')
        out.append(res.x)

    return out

def get_marks_2(fun, mark_thickness, n_pts=101, x0tol=1):
    """
    Returns position of marks in 1D.
    Assumes dark marks on light surface.

    Minimization method:
    1. Split the interval into two halves.
    2. Use brute force to find starting guesses.
    3. Use `scipy.minimize_scalar` to finallize the search

    fun : piecewise constant function (pixel values)
    mark_thickness : float
    n_pts : int, number of points for finding initial guess, x0, using brute-force
    x0tol : float, solution is sought on the interval [x0-x0tol, x0+x0tol]
    """
    smooth_fun = functions.smoothe(fun, functions.get_window, mark_thickness)
    x = np.linspace(fun.nodes[0], fun.nodes[-1], n_pts)
    smooth_vals = smooth_fun(x)
    x0 = [
        x[np.argmin(smooth_vals[:n_pts//2])],
        x[n_pts//2 + np.argmin(smooth_vals[n_pts//2:])]]
    marks = [
        sopt.minimize_scalar(smooth_fun, bracket=[x0i-x0tol, x0i+x0tol]).x
        for x0i in x0]
    return marks

def combine_channels(channels, coefs):
    """
    Return a linear combination of given channels.

    channels : list of Channel instances
    coefs : list of numbers

    Returns a new instance of Channel.
    """
    new_data = np.zeros_like(channels[0].data, dtype=float)
    for ch, c in zip(channels, coefs):
        new_data += c * ch.data
    return Channel(new_data)

def parse_args():
    """
    Uses argparse to return CLI-arguments
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('IMAGE_FILE', nargs='*',
                        default=['IMG_8290.JPG', 'IMG_8299.JPG'],
                        help='Image files to process, default: %(default)s')
    parser.add_argument('--plot', '-p', action='store_true', default=False,
                        help='Do the plots of data and smoothed data')
    parser.add_argument('--line', '-l', metavar='COOR', type=float, nargs=4,
                        default=[1779.5, 1674, 3315.5, 1720],
                        help='Coordinates of end-points of the examined line, '
                        'X_start, Y_start, X_end, Y_end, in pixels,'
                        'default: %(default)s')
    parser.add_argument('--mark-thickness', '-t', type=float, default=40,
                        help='Approximate thicknes of marks in pixels,'
                        'default: %(default)s')
    parser.add_argument('--flatten', '-f', action='store_false', default=True,
                        help='By default, images are flattened into one '
                        'channel. If this option is enabled, the channels are '
                        'treated separately.')
    parser.add_argument('--show-images', '-i', action='store_true', default=False,
                        help='Show all images with the examined line highlighted')
    parser.add_argument('--invert', '-I', action='store_true', default=False,
                        help='Invert the evaluation of photos')
    parser.add_argument('--channels-combination', '-c', type=float, nargs='+',
                        default=[1.0,], help='Combine the color channels of '
                        'each image using the given coefficients (default: '
                        '%(default)s).')
    args = parser.parse_args()
    return args

def main():
    """
    Main function for script-like call.
    Parses command-line arguments and computes distances of marks in given
    images.
    """
    logging.info('Program started')
    args = parse_args()

    help_line = interpolation.Line(
        [args.line[0], args.line[1]], [args.line[2], args.line[3]])

    line_start = [help_line.get_x(-.1), help_line.get_y(-.1)]
    line_end = [help_line.get_x(1.1), help_line.get_y(1.1)]
    line = interpolation.Line(line_start, line_end)

    images = [
        Image.from_file(file_name, flatten=args.flatten)
        for file_name in args.IMAGE_FILE]
    images = [
        Image([combine_channels(img.channels, args.channels_combination),])
        for img in images]

    if args.invert:
        images.reverse()

    _get_fig = lambda imn, chn: '%s, ch. %d' % (args.IMAGE_FILE[imn], chn)
    if args.show_images:
        for imn, image in enumerate(images):
            for chn, channel in enumerate(image.channels):
                plt.figure(_get_fig(imn, chn), figsize=(4,1.5))
                plt.imshow(channel.data, origin='lower', extent=[0, channel.data.shape[1], 0, channel.data.shape[0]])
                plt.plot([line_start[0], line_end[0]],
                         [line_start[1], line_end[1]], '.-k')

    lines_list = [image.get_line(line_start, line_end) for image in images]

    slice_length = np.linalg.norm(np.array(line_start) - np.array(line_end))
    print('slice_length =', slice_length)
    t_vals = np.linspace(args.mark_thickness, slice_length - args.mark_thickness,
                         173)

    strains = []
    length_0 = 0.
    for lln, lines in enumerate(lines_list):
        print('%s:' % args.IMAGE_FILE[lln])
        for chn, fun in enumerate(lines):
            smooth_fun = functions.smoothe(fun, functions.get_window, args.mark_thickness)
            marks = get_marks(fun, mark_thickness=args.mark_thickness, threshold=.8)
            print('marks:', marks)
            mark1, mark2 = marks[0], marks[-1]
            length_i = mark1 - mark2
            if not strains:
                length_0 = length_i
                strains.append(0.)
            else:
                strains.append(length_i / length_0 - 1)
            print(' channel %d: distance of marks: %f pixels'
                  % (chn, length_i))
            if args.plot:
                t_vals = np.linspace(0, slice_length, 201)
                smooth_vals = smooth_fun(t_vals)
                plt.figure('lines' + _get_fig(lln, chn), figsize=(4,1.5))
                fun.plot(label='$p$')
                plt.plot(t_vals, smooth_vals, '--', label='$f$')
                plt.plot(
                    [mark1, mark2],
                    [smooth_fun(mark1), smooth_fun(mark2)], 'xk')
                plt.title('šířka okna %dpx' % (2*args.mark_thickness))
                plt.legend(loc='best')

            if args.show_images:
                plt.figure(_get_fig(lln, chn))
                plt.plot(
                    [line.get_x(mark1 / slice_length),
                     line.get_x(mark2 / slice_length)],
                    [line.get_y(mark1 / slice_length),
                     line.get_y(mark2 / slice_length)],
                    'xr')

    print('strains:\n', strains)

    logging.info('Computation finished')

    if args.plot or args.show_images:
        plt.show()

if __name__ == '__main__':
    main()
