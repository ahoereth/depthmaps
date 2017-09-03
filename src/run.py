import tensorflow as tf

try:
    import matplotlib.pyplot as plt
except ImportError:
    gui = False
else:
    gui = True

from datasets import Merged, Make3D, Make3D2, Nyu, Dataviewer
from models import Simple, Eigen2014, Pix2Pix


dataset = Make3D(workers=6)
model = Pix2Pix(dataset)
model.train(epochs=400)

results = model.evaluate(fetch_images=gui)
print(len(results))

if gui:
    Dataviewer(results, name='Results', keys=['image', 'depth', 'result'],
               cmaps={'depth': 'gray', 'result': 'gray'})
    plt.show(True)
