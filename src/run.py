import tensorflow as tf

try:
    import matplotlib.pyplot as plt
except ImportError:
    gui = False
else:
    gui = True

from lib import Dataviewer
from datasets import Make3D, Nyu
from models import Simple, Eigen2014, Pix2Pix


dataset = Make3D()
model = Eigen2014(dataset)
model.train(epochs=1, workers=4)

results = model.evaluate(fetch_images=gui)

if gui:
    Dataviewer(results, name='Results', keys=['image', 'depth', 'result'],
               cmaps={'depth': 'gray', 'result': 'gray'})
    plt.show(True)
