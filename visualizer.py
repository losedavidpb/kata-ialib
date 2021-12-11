from abc import ABC, abstractmethod
from matplotlib import pyplot as plt, animation
import numpy as np
from matplotlib.colors import get_named_colors_mapping

# region ____________________ Private functions ____________________
def _format_file_path_extension(file_path, extension):
    if file_path is not None:
        if not file_path.endswith('.' + extension):
            file_path = file_path + '.' + extension

    return file_path
# endregion

# region ____________________ Visualizer Interface ____________________
class AnimationIA(ABC):

    def __init__(self, **kwargs):
        plt.clf()
        self.fig = plt.figure(1)

        self.file_path = kwargs.get('file_path', None)
        self.file_path = _format_file_path_extension(self.file_path, 'gif')

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
        self.file_path = _format_file_path_extension(self.file_path, 'jpg')

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
        self.weights, self.x_line = None, None

    def init(self, **kwargs):
        self.x, self.y = kwargs.get('x'), kwargs.get('y')
        self.weights = kwargs.get('weights')
        self.x_line = np.arange(min(self.x[:, 0]), max(self.x[:, 0]) + 1)
        self.fig.clear()
        plt.grid(b=True)
        return self

    def show(self):
        line_data = (lambda m, n, _x: m * _x + n)(
            m=self.weights[1] / -self.weights[2],
            n=self.weights[0] / -self.weights[2],
            _x=self.x_line
        )

        plt.scatter(self.x[:, 0], self.x[:, 1], c=self.y)
        plt.plot(self.x_line, line_data)
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
        self.weights, self.x_line = None, None
        self.errors = None

    def init(self, **kwargs):
        self.x, self.y = kwargs.get('x'), kwargs.get('y')
        self.weights = kwargs.get('weights')
        self.errors = kwargs.get('errors')
        self.x_line = np.arange(min(self.x[:, 0]), max(self.x[:, 0]) + 1)
        self.num_frames = len(self.weights)
        self.fig.clear()
        plt.grid(b=True)
        return self

    def update(self, frame):
        if self.num_frames < frame: return

        self.fig.clear()
        _weights = self.weights[frame]
        _error = self.errors[frame]

        line_data = (lambda m, n, _x: m * _x + n)(
            m=_weights[1] / -_weights[2],
            n=_weights[0] / -_weights[2],
            _x=self.x_line
        )

        plt.scatter(self.x[:, 0], self.x[:, 1], c=self.y)
        plt.plot(self.x_line, line_data)
        plt.title(str.format("Training evolution, Epoch={}, Error={}", frame, format(_error, '.4f')))
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
