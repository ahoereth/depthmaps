import tensorflow as tf

from datasets import Make3D, Nyu
from models import Simple, Eigen2014


dataset = Nyu()
model = Simple(dataset, workers=2)

print('First epoch will be slow due to data preprocessing and caching...')
model.train(epochs=10)
