import tensorflow as tf

from lib import Dataviewer
from datasets import Make3D, Nyu
from models import Simple, Eigen2014, Pix2Pix


dataset = Make3D()
model = Pix2Pix(dataset, epochs=10)
model.train(workers=3)

# results = model.evaluate()
# Dataviewer(results, name='Results', keys=['image', 'depth', 'result'],
#            cmaps={'depth': 'gray', 'result': 'gray'})
# plt.show(True)
