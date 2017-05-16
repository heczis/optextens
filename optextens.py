"""
Module for working with visual data from mechanical experiments and computation
of movement and deformation based on high-contrast marks.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread

import interpolation

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

def get_smooth_fun(t_vals, fun, mark_thickness=40):
    """
    Returns smoothed values of `fun`, evaluated at `t_vals`.
    """
    smooth_fun = interpolation.smooth_int(fun, span=mark_thickness)
    smooth_vals = [val for val in map(smooth_fun, t_vals)]
    return smooth_vals

def get_mark_distance(t_vals, brightness_vals):
    """
    Finds position of marks in 1D, returns their distance in pixels.
    Assumes dark marks on light surface.

    t_vals : array of floats (parameter values)
    brightness_vals : smoothed brightness data
    """
    n_pts = len(brightness_vals)
    mark1_index = np.argmin(brightness_vals[:n_pts//2])
    mark2_index = n_pts // 2 + np.argmin(brightness_vals[n_pts//2:])

    return t_vals[mark2_index] - t_vals[mark1_index]

def parse_args():
    """
    Uses argparse to return CLI-arguments
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('IMAGE_FILE', nargs='*',
                        default=['IMG_8300.JPG', 'IMG_8308.JPG'],
                        help='Image files to process, default: %(default)s')
    parser.add_argument('--plot', '-p', action='store_true', default=False,
                        help='Do the plots of data and smoothed data')
    parser.add_argument('--line', '-l', metavar='COOR', type=float, nargs=4,
                        default=[1869.5, 1698.5, 3306.5, 1576],
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
    args = parser.parse_args()
    return args

def main():
    """
    Main function for script-like call.
    Parses command-line arguments and computes distances of marks in given
    images.
    """
    args = parse_args()

    help_line = interpolation.Line(
        [args.line[0], args.line[1]], [args.line[2], args.line[3]])

    line_start = [help_line.get_x(-.1), help_line.get_y(-.1)]
    line_end = [help_line.get_x(1.1), help_line.get_y(1.1)]

    images = [
        Image.from_file(file_name, flatten=args.flatten)
        for file_name in args.IMAGE_FILE]

    lines_list = [image.get_line(line_start, line_end) for image in images]

    slice_length = np.linalg.norm(np.array(line_start) - np.array(line_end))
    t_vals = np.concatenate([
        np.linspace(args.mark_thickness, .2 * slice_length, int(.2*slice_length)+1),
        np.linspace(.8 * slice_length, slice_length-args.mark_thickness,
                    int(.2*slice_length)+1)])
    plt.figure()
    for lln, lines in enumerate(lines_list):
        print('%s:' % args.IMAGE_FILE[lln])
        for chn, fun in enumerate(lines):
            smooth_vals = get_smooth_fun(t_vals, fun)
            if args.plot:
                funvals = [val for val in map(fun, t_vals)]
                plt.plot(t_vals, funvals, label='%s data, channel %d'
                         % (args.IMAGE_FILE[lln], chn))
                plt.plot(t_vals, smooth_vals, ':', label='%s smoothed, channel %d'
                         % (args.IMAGE_FILE[lln], chn))

            print(' channel %d: distance of marks: %f pixels'
                  % (chn, get_mark_distance(t_vals, smooth_vals)))

    if args.plot:
        plt.legend(loc='best')
        plt.show()

if __name__ == '__main__':
    main()
