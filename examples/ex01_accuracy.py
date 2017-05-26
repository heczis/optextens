"""
Plot accuracy as a function of smoothing window span.
"""
import multiprocessing
import sys
sys.path.append('.')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

import functions
import optextens

def get_position(span, image):
    line_start, line_end = [0, 5], [100, 5]
    line_length = np.linalg.norm(np.array(line_start) - np.array(line_end))

    fun = image.get_line(line_start, line_end)[0]
    smooth_fun = functions.smoothe(
        fun, functions.get_window, span=span)
    mark1 = optextens.get_marks_2(
        fun, span)[0]

    return mark1

if __name__ == '__main__':
    pool = multiprocessing.Pool()
    image_file = 'foto_examples/horiz_multi.png'

    # load the image
    image = optextens.Image.from_file(image_file, flatten=True)

    # display it
    plt.figure()
    plt.imshow(image.channels[0].data)

    fig = plt.figure(figsize=(6, 2.1))

    # find marks for different smoothing window spans
    ax1 = fig.add_subplot(121)
    spans = np.linspace(1, 6, 51)
    args = [[span, image] for span in spans]
    positions = np.array([val for val in pool.starmap(get_position, args)])
    plt.plot(2*spans, positions)

    plt.xlabel('smoothing window span [px]')
    plt.ylabel('mark position [px]')
    plt.grid()

    # plot examples of smoothing windows
    ax2 = fig.add_subplot(122)
    x = np.linspace(0, 10, 10*2+1)
    win_1 = functions.get_window(3)
    win_2 = functions.get_window(7, 1)
    plt.plot(x, win_1(x), label='$x=3$, span 4px')
    plt.plot(x, win_2(x), label='$x=7$, span 2px')
    plt.xlabel('$y$')
    plt.legend(loc='best').draggable()
    plt.grid()

    plt.tight_layout()
    plt.show()
