"""Interactive data viewer.

Props @shoeffner.
"""
from itertools import cycle
from pathlib import Path

import numpy as np
from scipy import misc as spmisc
from PIL import Image
try:
    import _tkinter  # Just to check the dependency.
    import matplotlib
except ImportError:
    pass
else:
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt
    from matplotlib import image


class Dataviewer:

    def __init__(self, dataset, *,
                 rows=3,
                 keys=['img', 'depth'],
                 cmaps={'depth': 'jet'},
                 name=None):
        """Create a new DataBrowser.

        Args:
            dataset (List): List of samples, each sample can either be a
                List or Tuple containing in order of or object instances
                with attributes matching the values of the `keys` arg.
            rows (int): Number of rows to display.
            cmaps (Dict): Dict specifying colormaps for specific keys.
            name (str): Databrowser name.
        """
        self.dataset = dataset
        self.rows = rows
        self.keys = keys

        self.current = 0

        self.figure, self.axes = plt.subplots(rows, len(self.keys))
        if name:
            self.figure.canvas.set_window_title(name)

        self.axes = self.axes.flatten()
        self.images = []
        for i, axes in enumerate(self.axes):
            cmap = cmaps[keys[i % len(keys)]] if keys[i % len(keys)] in cmaps \
                else None
            img = image.AxesImage(axes, cmap=cmap)
            axes.set_aspect('equal')
            self.images.append(axes.add_image(img))

        self.show_next()

        self.keycb = self.figure.canvas.mpl_connect(
            'key_press_event',
            lambda event: self.__key_press_event(event))

    def show_next(self):
        self.update_axes()

    def show_previous(self):
        self.current = (self.current - 2 * self.rows) % len(self.dataset)
        self.update_axes()

    def update_axes(self):
        first = self.current
        for axes, img, (col, key) in zip(self.axes, self.images,
                                         cycle(enumerate(self.keys))):
            title = key
            sample = self.dataset[self.current]
            if hasattr(sample, key) and getattr(sample, key) is not None:
                data = getattr(sample, key)
                if isinstance(data, str):
                    title = Path(data).name
                    data = spmisc.imread(data)
                else:
                    data = np.squeeze(data)
            elif isinstance(sample, (list, tuple)) and len(sample) > col:
                data = sample[col]
                if isinstance(data, str):
                    title = Path(data).name
                    data = spmisc.imread(data)
                else:
                    data = np.squeeze(data)
            else:
                data = np.array([[1]])
            img.set_data(data)
            axes.set_xlim([0, data.shape[1]])
            axes.set_ylim([data.shape[0], 0])
            axes.set_title(title, fontsize=8)

            if key == self.keys[-1]:
                self.current = (self.current + 1) % len(self.dataset)

        self.figure.suptitle('Showing samples {} to {}'
                             .format(first,
                                     (self.current - 1) % len(self.dataset)))
        self.figure.canvas.draw()

    def __key_press_event(self, event):
        events = {
            'q': lambda event: plt.close(self.figure),
            'escape': lambda event: plt.close(self.figure),
            'cmd+w': lambda event: plt.close(self.figure),
            'right': lambda event: self.show_next(),
            'down': lambda event: self.show_next(),
            'left': lambda event: self.show_previous(),
            'up': lambda event: self.show_previous()
        }
        try:
            events[event.key](event)
        except KeyError:
            print('Key pressed but no action available: {}'.format(event.key))
