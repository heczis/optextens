"""
Module with definitions of various types of scalar functions.
"""
import matplotlib.pyplot as plt
import numpy as np

class Function:
    """
    Parent of all function-classes
    """
    def eval(self, _):
        pass

    def __call__(self, x):
        """
        Evaluate self for all elements of x.
        """
        try:
            out = np.array([val for val in map(self.eval, x)])
        except:
            out = self.eval(x)

        return out

class ConstFunction(Function):
    """
    Piecewise constant function
    """
    def __init__(self, nodes, values):
        """
        nodes : sorted array of x-values
        values : function values on intervals bounded by nodes, i.e.
          len(values) = len(nodes)-1
        """
        self.nodes = nodes
        self.values = values

    def eval(self, x):
        """
        Return function value at x
        """
        for value, x_i, x_ii in zip(self.values,
                                    self.nodes[:-1], self.nodes[1:]):
            if (x_i <= x) and (x < x_ii):
                return value
        return 0.

    def integrate_all(self):
        """
        Return integral over the whole real axis
        """
        dx = np.diff(self.nodes)
        return np.dot(dx, self.values)

    def __rmul__(self, other):
        return self * other

    def __mul__(self, other):
        """
        Return the product of self and another instance of ConstFunction
        """
        if isinstance(other, int) or isinstance(other, float):
            return ConstFunction(self.nodes, [other * val for val in self.values])
        elif isinstance(other, ConstFunction):
            nodes = combine_nodes(self.nodes, other.nodes)
            mid_points = .5 * (nodes[:-1] + nodes[1:])
            values = np.array([self.eval(x) * other.eval(x) for x in mid_points])
            return ConstFunction(nodes, values)
        elif isinstance(other, LinFunction):
            nodes = combine_nodes(self.nodes, other.nodes)
            mid_points = .5 * (nodes[:-1] + nodes[1:])
            slopes = other.diff()(mid_points)
            const_vals = self(mid_points)

            dx = np.diff(nodes)
            lin_vals = [[other.eval(x_i), other.eval(x_i) + dx_i * slope_i]
                        for x_i, dx_i, slope_i in zip(nodes[:-1], dx, slopes)]
            values = [
                [const_val * lin_val_1, const_val * lin_val_2]
                for const_val, [lin_val_1, lin_val_2] in zip(const_vals, lin_vals)]
            return LinFunction(nodes, values)
        else:
            return NotImplemented

    def max(self, value):
        """
        Truncate the function values from above by `value`.
        Return new function.
        """
        new_values = [min(val, value) for val in self.values]
        return ConstFunction(self.nodes, new_values)

    def min(self, value):
        """
        Truncate the function values from below by `value`.
        Return new function.
        """
        new_values = [max(val, value) for val in self.values]
        return ConstFunction(self.nodes, new_values)

    def get_clusters_by_max(self, max_value):
        """
        Return functions defined on intervals where function value of self is
        below `max_value`.
        """
        out = []
        current_interval = None
        for ii, val in enumerate(self.values):
            if val >= max_value:
                if current_interval is not None:
                    # the preceding interval finished an output function
                    current_interval['nodes'].append(self.nodes[ii])
                    # add the last function to out
                    out.append(
                        ConstFunction(current_interval['nodes'],
                                      current_interval['values']))
                    current_interval = None
            else:
                if current_interval is None:
                    # start storing values
                    current_interval = {
                        'nodes' : [self.nodes[ii],], 'values' : [val,]}
                else:
                    # continue staring values
                    current_interval['nodes'].append(self.nodes[ii])
                    current_interval['values'].append(val)

        # check if anything left in current_interval
        if current_interval is not None:
            current_interval['nodes'].append(self.nodes[-1])
            out.append(
                ConstFunction(current_interval['nodes'],
                              current_interval['values']))

        return out

    def plot(self, *args, **kwargs):
        """
        Plot self using matplotlib, *args and **kwargs are passed to
        `matplotlib.pyplot.plot`.
        """
        x = [self.nodes[0],] + sum([[n, n] for n in self.nodes[1:-1]], []) + [self.nodes[-1],]
        y = sum([[v, v] for v in self.values], [])
        plt.plot(x, y, *args, **kwargs)

def combine_nodes(nodes1, nodes2):
    def __accept(node):
        min_node = max(min(nodes1), min(nodes2))
        max_node = min(max(nodes1), max(nodes2))
        if (node >= min_node) and (node <= max_node):
            return True
        else:
            return False

    out = np.concatenate([
        [n for n in nodes1 if __accept(n)],
        [n for n in nodes2 if __accept(n)]])
    out.sort()
    return out

class LinFunction(Function):
    """
    Piecewise linear function
    """
    def __init__(self, nodes, values):
        """
        nodes : sorted array of x-values
        values : function values at boundaries of each element [[f1min, f1max],
          [f2min, f2max],...]
        """
        self.nodes = nodes
        self.values = values

    def eval(self, x):
        """
        Return function value at x
        """
        for [y_i, y_ii], x_i, x_ii in zip(self.values,
                                    self.nodes[:-1], self.nodes[1:]):
            if (x_i <= x) and (x < x_ii):
                return y_i + (y_ii-y_i) / (x_ii-x_i) * (x-x_i)
        return 0.

    def integrate_all(self):
        """
        Return integral over the whole real axis
        """
        dx = np.diff(self.nodes)
        vals = np.array([.5*(val1 + val2) for val1, val2 in self.values])
        return np.dot(dx, vals)

    def diff(self):
        """
        Return derivative, i.e. instance of ConstFunction
        """
        d_vals = np.array([v2-v1 for v1, v2 in self.values])
        return ConstFunction(self.nodes, d_vals / np.diff(self.nodes))

    def plot(self, *args, **kwargs):
        """
        Plot self using matplotlib, *args and **kwargs are passed to
        `matplotlib.pyplot.plot`.
        """
        x = [self.nodes[0],] + sum([[n, n] for n in self.nodes[1:-1]], []) \
            + [self.nodes[-1],]
        y = sum(self.values, [])
        plt.plot(x, y, *args, **kwargs)

def get_window(x0, span=2):
    nodes = [x0-span, x0, x0+span]
    values = [[0, 1. / span], [1. / span, 0]]
    return LinFunction(nodes, values)

def get_diff_window(x0, span=2):
    nodes = [x0-span, x0, x0+span]
    values = [1. / span**2, -1. / span**2]
    return ConstFunction(nodes, values)

def get_const_window(x0, span=2):
    nodes = [x0-span, x0+span]
    values = [.5 / span,]
    return ConstFunction(nodes, values)

def smoothe(fun, window_getter, span=10):
    def __f(x):
        window = window_getter(x, span=span)
        mult_fun = fun * window
        return mult_fun.integrate_all() \
            / (ConstFunction([fun.nodes[0], fun.nodes[-1]], [1.,]) * window).integrate_all()

    smooth_fun = Function()
    smooth_fun.eval = __f
    return smooth_fun

def diff_smoothe(fun, window_getter, diff_window_getter, span=10):
    def __f(x):
        diff_window = diff_window_getter(x, span=span)
        window = window_getter(x, span=span)
        mult_fun = fun * diff_window
        return mult_fun.integrate_all() \
            / (ConstFunction([fun.nodes[0], fun.nodes[-1]], [1.,]) * window).integrate_all()

    smooth_fun = Function()
    smooth_fun.eval = __f
    return smooth_fun

def test_smoothing():
    """
    Illustrate the use of the `functions` module
    """
    x = np.linspace(0, 10, 101)

    data = ConstFunction([1, 2, 4, 5, 7, 8], [1, 3, 2, 1, 2])
    smooth_fun = smoothe(data, get_window, 2)

    data.plot(label='data')
    plt.plot(x, smooth_fun(x))

    xx = np.linspace(0, 10, 401)
    plt.plot(xx, smooth_fun(xx))
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    test_smoothing()
