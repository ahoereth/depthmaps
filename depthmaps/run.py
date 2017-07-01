import tensorflow as tf

from datasets import Make3D
from models import Simple, Eigen2014


dataset = Make3D()
model = Eigen2014(dataset)

print('First epoch will be slow due to data preprocessing and caching...')
model.train(epochs=2)
