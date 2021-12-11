from abc import ABC, abstractmethod
from matplotlib import pyplot as plt, animation
import numpy as np
from matplotlib.colors import get_named_colors_mapping
from utils import format_file_path_extension, get_decimal_format


# region ____________________ Visualizer Interface ____________________
class AnimationIA(ABC):

    def __init__(self, **kwargs):
        plt.clf()
        self.fig = plt.figure(1)

        self.file_path = kwargs.get('file_path', None)
        self.file_path = format_file_path_extension(self.file_path, 'gif')

        self.interval_time = kwargs.get('interval_time', 500)
        self.num_frames_per_sec = kwargs.get('num_frames_per_sec', 3)
        self.num_frames = kwargs.get('num_frames', 1)
        self.dpi = kwargs.get('dpi', 100)

    @abstractmethod
    def init(self):
        """ Initialize configuration for current animation. """
        pass

    @abstractmethod
    def update(self, frame):
        """Update data based on current frame. """
        pass

    def show(self):
        """Start the execution of the animation. """
        v_anim = animation.FuncAnimation(
            fig=self.fig, frames=self.num_frames, interval=self.interval_time,
            blit=False, func=self.update, repeat=True)

        plt.grid(b=True)
        plt.show()

        if self.file_path is not None:
            v_anim.save(filename=self.file_path, writer='pillow', fps=self.num_frames_per_sec)
            plt.close(self.fig)

class PlotterIA(ABC):

    def __init__(self, **kwargs):
        plt.clf()
        self.fig = plt.figure(1)
        self.file_path = kwargs.get('file_path', None)
        self.file_path = format_file_path_extension(self.file_path, 'jpg')

    @abstractmethod
    def init(self, **kwargs):
        """ Initialize configuration for current plotter. """
        pass

    @abstractmethod
    def show(self):
        """Start the visualization of the plotter. """
        pass
# endregion

# region ____________________ Plotter Implementation ____________________
class TwoDimensionDataPlotter(PlotterIA):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.x, self.y = None, None

    def init(self, **kwargs):
        self.x, self.y = kwargs.get('x'), kwargs.get('y')
        self.fig.clear()
        plt.grid(b=True)
        return self

    def show(self):
        plt.scatter(self.x[:, 0], self.x[:, 1], c=self.y)
        plt.title("Data visualization")
        plt.xlabel("x"), plt.ylabel("y")
        plt.show()

class ErrorPlotter(PlotterIA):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.errors = None

    def init(self, **kwargs):
        self.errors = kwargs.get('errors')
        self.fig.clear()
        plt.grid(b=True)
        return self

    def show(self):
        incr_per_epoch = int(self.errors.__len__() / 10) + 1
        range_epochs = [i for i in range(0, self.errors.__len__(), incr_per_epoch)]
        plt.plot(self.errors, 'b-', label="Error SSE")
        plt.title("Error evolution SSE")
        plt.xticks(ticks=range_epochs, labels=range_epochs)
        plt.xlabel("Number of epochs"), plt.ylabel("Error (SSE)")
        plt.grid(b=True), plt.show()

class NeuronPlotter(PlotterIA):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.x, self.y = None, None
        self.model = None

    def init(self, **kwargs):
        self.x, self.y = kwargs.get('x'), kwargs.get('y')
        self.model = kwargs.get('model')
        self.fig.clear(), plt.grid(b=True)
        return self

    def show(self):
        x_min, x_max = self.x[:, 0].min() - 1, self.x[:, 0].max() + 1
        y_min, y_max = self.x[:, 1].min() - 1, self.x[:, 1].max() + 1
        _x, _y = np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02)
        xx, yy = np.meshgrid(_x, _y)

        _w = self.model.history['weights'][-1]
        m, n = _w[1] / -_w[2], _w[0] / -_w[2]
        y_w = m * _x + n

        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=.25)
        curr_axis = plt.axis()
        plt.scatter(self.x[:, 0], self.x[:, 1], c=self.y)
        plt.plot(_x, y_w, 'k--')
        plt.axis(curr_axis)

        plt.title("Optimal model visualization")
        plt.xlabel("x"), plt.ylabel("y")
        plt.grid(b=True), plt.show()

class GrowingNeuralGasPlotter(PlotterIA):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.colors = list(get_named_colors_mapping())
        self.fig = plt.figure(1)
        self.ax = plt.axes()
        self.history = None

    def init(self, **kwargs):
        self.history = kwargs.get('history')
        return self

    def show(self):
        self.ax.clear(), self.ax.grid()

        a, n = self.history['a'][-1], self.history['n'][-1]
        error = self.history['errors'][-1]
        clusters = self.history['clusters'][-1]

        x, y = a[:, 0], a[:, 1]
        num_clusters = len(clusters.keys())

        s_title = "Epoch {}: num_clusters={}, num_units={}, error={}"
        s_title = str.format(s_title, len(self.history['a']) - 2, num_clusters, x.shape[0], format(error, '.4f'))
        self.ax.set_title('Growing Neural Gas Evolution\n\n' + s_title, loc='center')

        connections = {}

        for i in range(0, len(clusters.keys())):
            for j in range(0, len(clusters[i])):
                identifier = n[clusters[i][j]].identifier

                for neighbor in n[clusters[i][j]].neighborhood:
                    key = str((identifier, neighbor)) + str((neighbor, identifier))
                    connections.setdefault(key, False)

                    if neighbor in clusters[i] and not connections[key]:
                        x_i, y_i = [x[identifier], x[neighbor]], [y[identifier], y[neighbor]]
                        self.ax.plot(x_i, y_i, '-o', c=self.colors[i])
                        connections[key] = True

        plt.grid(b=True), plt.show()
# endregion

# region ____________________ Animation Implementation ____________________
class NeuronAnimation(AnimationIA):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.x, self.y = None, None
        self.model = None

    def init(self, **kwargs):
        self.x, self.y = kwargs.get('x'), kwargs.get('y')
        self.model = kwargs.get('model')
        self.num_frames = len(self.model.history['weights'])
        self.fig.clear(), plt.grid(b=True)
        return self

    def update(self, frame):
        if self.num_frames < frame: return

        self.fig.clear()
        _error = self.model.history['errors'][frame]
        _accuracy = self.model.history['accuracy'][frame]

        error_ = get_decimal_format(_error)
        accuracy_ = get_decimal_format(_accuracy)

        x_min, x_max = self.x[:, 0].min() - 1, self.x[:, 0].max() + 1
        y_min, y_max = self.x[:, 1].min() - 1, self.x[:, 1].max() + 1
        _x, _y = np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02)
        xx, yy = np.meshgrid(_x, _y)

        _w = self.model.history['weights'][frame]
        m, n = _w[1] / -_w[2], _w[0] / -_w[2]
        y_w = m * _x + n

        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()], num_epoch=frame)
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=.25)
        plt.scatter(self.x[:, 0], self.x[:, 1], c=self.y)
        curr_axis = plt.axis()
        plt.plot(_x, y_w, 'k--')
        plt.axis(curr_axis)

        plt.title(str.format("Training evolution, Epoch={}, Error={}, Accuracy={}", frame, error_, accuracy_))
        plt.xlabel("x"), plt.ylabel("y")
        plt.grid(b=True), plt.show()

class GrowingNeuralGasAnimation(AnimationIA):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.colors = list(get_named_colors_mapping())
        self.fig = plt.figure(1)
        self.ax = plt.axes()
        self.history = None

    def init(self, **kwargs):
        self.history = kwargs.get('history')
        self.num_frames = len(self.history['a']) - 1
        self.ax.clear()
        self.ax.grid()
        plt.grid(b=True)
        return self

    def update(self, frame):
        if self.num_frames < frame: return

        self.ax.clear()
        self.ax.grid()
        plt.grid(b=True)

        a, n = self.history['a'][frame], self.history['n'][frame]
        error = self.history['errors'][frame]

        x, y = a[:, 0], a[:, 1]
        clusters = self.history['clusters'][frame]
        num_clusters = len(clusters.keys())

        s_title = "Epoch {}: num_clusters={}, num_units={}, error={}"
        s_title = str.format(s_title, frame, num_clusters, x.shape[0], format(error, '.4f'))
        self.ax.set_title('Growing Neural Gas Evolution\n\n' + s_title, loc='center')

        connections = {}

        for i in range(0, len(clusters.keys())):
            for j in range(0, len(clusters[i])):
                identifier = n[clusters[i][j]].identifier

                for neighbor in n[clusters[i][j]].neighborhood:
                    key = str((identifier, neighbor)) + str((neighbor, identifier))
                    connections.setdefault(key, False)

                    if neighbor in clusters[i] and not connections[key]:
                        x_i, y_i = [x[identifier], x[neighbor]], [y[identifier], y[neighbor]]
                        self.ax.plot(x_i, y_i, '-o', c=self.colors[i])
                        connections[key] = True
# endregion
