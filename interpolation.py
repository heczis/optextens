"""
Module for interpolation-related methods.
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as sint
import scipy.interpolate as sitp

DEFAULT_WINDOW = type(
    '', (), {'x_vals' : np.array([-1, 0, 1]),
             'weights' : np.array([.25, .5, .25])})

class Line:
    """
    Represents a straight line in plane.
    """
    def __init__(self, start, end, a=0, b=1):
        """
        start, end: [x, y]
        """
        self.start_point = np.array(start)
        self.end_point = np.array(end)
        self.start_par = a
        self.end_par = b

        # find parameters of x-, y-functions
        fp_mat = np.diag([a, 1, a, 1]) + np.diag([1, 0, 1], 1)\
                 + np.diag([b, 0, b], -1)
        fp_rhs = np.array([start[0], end[0], start[1], end[1]])
        fun_pars = np.linalg.solve(fp_mat, fp_rhs)
        self.fun_pars = type(
            '', (), {'kx' : fun_pars[0], 'qx' : fun_pars[1],
                     'ky' : fun_pars[2], 'qy' : fun_pars[3]})

    def get_bounding_box(self):
        """
        Returns [[x_min, y_min], [x_max, y_max]]
        """
        x_min = min(self.start_point[0], self.end_point[0])
        x_max = max(self.start_point[0], self.end_point[0])
        y_min = min(self.start_point[1], self.end_point[1])
        y_max = max(self.start_point[1], self.end_point[1])
        return [[x_min, y_min], [x_max, y_max]]

    def get_x(self, t_val):
        """
        Return x-values from parameter values.
        """
        return self.fun_pars.kx * t_val + self.fun_pars.qx

    def get_y(self, t_val):
        """
        Return y-values from parameter values.
        """
        return self.fun_pars.ky * t_val + self.fun_pars.qy

    def get_t_from_x(self, x_val):
        """
        Return values of parameter corresponding to given x-values.
        """
        return (x_val - self.fun_pars.qx) / self.fun_pars.kx

    def get_t_from_y(self, y_val):
        """
        Return values of parameter corresponding to given y-values.
        """
        return (y_val - self.fun_pars.qy) / self.fun_pars.ky

def get_intersections_pars(line, x_coors, y_coors, tol=1e-12):
    """
    Returns the coordinates of intersections of line with a rectangular grid.
    """
    x_pars = [
        t for t in line.get_t_from_x(x_coors)
        if (line.start_par - t) * (line.end_par - t) <= 0.0]
    y_pars = [
        t for t in line.get_t_from_y(y_coors)
        if (line.start_par - t) * (line.end_par - t) <= 0.0]
    pars = np.array([p_val for p_val in x_pars])
    if len(pars) == 0:
        pars = np.array([p_val for p_val in y_pars])
    else:
        for p_val in y_pars:
            if np.min(np.abs(pars - p_val)) > tol:
                pars = np.concatenate([pars, [p_val,]])

    pars.sort()
    return pars

def get_intersections_coors(line, x_coors, y_coors, tol=1e-12):
    """
    Returns the coordinates of intersections of line with a rectangular grid.
    """
    pars = get_intersections_pars(line, x_coors, y_coors, tol=tol)
    return [[line.get_x(p_val), line.get_y(p_val)] for p_val in pars]

def get_piecewise_constant_function(intervals, values, outside_value=None, tol=1e-12):
    """
    Returns scalar function of scalar variable; the returned

    intervals : list of interval boundaries, assumed sorted
    values : list of [float] values, len(values) = len(intervals)-1
    """
    def _fun(x_val):
        """
        x_val : float
        """
        out = outside_value
        for x_min, x_max, val in zip(intervals[:-1], intervals[1:], values):
            if (x_min - x_val) * (x_max - x_val) < tol:
                out = val
                break
        return out

    return _fun

def is_in_bounding_box(point, bbox):
    """
    point : [x, y]
    bbox : [xmin, ymin, xmax, ymax]
    """
    x, y = point
    xmin, ymin, xmax, ymax = bbox
    out = False
    if ((x >= xmin) and (x <= xmax) and
        (y >= ymin) and (y <= ymax)):
        out = True
    return out

def get_data_slice(data, line, *args, **kwargs):
    """
    Returns scalar-valued function.

    data : 2D array
    line : Line instance
    """
    data_bbox = [0, 0, data.shape[0]+1, data.shape[1]+1]
    intersection_coors = np.concatenate([
        np.array([line.start_point,]),
        get_intersections_coors(line, np.arange(data.shape[0]), np.arange(data.shape[1])),
        np.array([line.end_point,])])
    intersection_coors = np.array([
        point for point in intersection_coors
        if is_in_bounding_box(point, data_bbox)])
    mid_points = np.array([
        .5 * (point_1 + point_2)
        for point_1, point_2 in zip(intersection_coors[:-1], intersection_coors[1:])])
    values = np.array([
        data[int(pt[0]), int(pt[1])] for pt in mid_points])
    intervals = np.array([
        np.linalg.norm(pt-intersection_coors[0]) for pt in intersection_coors])
    return get_piecewise_constant_function(intervals, values, *args, **kwargs)

def smooth(fun, window=None):
    """
    Returns a smoothed function.
    """
    if window is None: window = DEFAULT_WINDOW

    def _fun(x):
        funvals = np.array([val for val in map(fun, x+window.x_vals)])
        return np.dot(funvals, window.weights)

    return _fun

def default_window(x, x0, span):
    if np.abs(x0-x) < .5 * span:
        return 2. / span * (1 - 2 * np.abs(x0 - x) / span)
    else: return 0.

def get_window_default(x0=0., span=40):
    def _f(x):
        return default_window(x, x0, span)
    return _f

def smooth_int(fun, get_window=None, span=40):
    """
    Returns function f smoothed using the relation
    f_{\rm smooth}(x) = \int_{-\infty}^{+\infty} f(y) g(x; y) \d{y}

    fun : function, f(y)
    get_window : returns function, g(x; .)
    """
    if get_window is None:
        get_window = get_window_default

    def _smf(x):
        y_vals = x + np.linspace(-.5*span, .5*span, 31)
        window_vals =  np.array([val for val in map(get_window(x, span=span), y_vals)])
        return np.dot(
            np.array([val for val in map(fun, y_vals)]),
            window_vals) / np.sum(window_vals)
    return _smf

def main():
    img = np.array([
        [50, 75, 200],
        [150, 80, 5]])
    line = Line([.5, -.1], [2.5, 2.1])
    ints = np.array(
        get_intersections_coors(line, np.arange(img.shape[1]+1),
                                np.arange(img.shape[0]+1))).T
    slice_fun = get_data_slice(img.T, line, outside_value=0.)

    plt.imshow(img, origin='lower', extent=[0, img.shape[1], 0, img.shape[0]])
    plt.colorbar()
    plt.plot([line.start_point[0], line.end_point[0]],
             [line.start_point[1], line.end_point[1]])
    plt.plot(ints[0], ints[1], 'ok')
    plt.grid()

    plt.figure()
    x = np.linspace(-.1, 3.4)
    y = np.array([val for val in map(slice_fun, x)])
    X = np.linspace(-1, 4, 8)
    Y = np.array([val for val in map(slice_fun, X)])

    spline_degree = 5
    int_spline = sitp.make_lsq_spline(X, Y, np.r_[
        [min(X)] * (spline_degree+1), [1.5], [max(X)] * (spline_degree+1)], k=spline_degree)
    plt.plot(x, y)
    plt.plot(x, sitp.splev(x, int_spline))
    plt.plot(x, sitp.splev(x, int_spline, der=1))
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()
