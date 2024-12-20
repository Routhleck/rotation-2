import braintools as bts
import brainunit as u
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import brainstate as bst

bst.random.seed(42)

from model_EI import SNN
from utils import plot_data, data_generate_1212

num_inputs = 20
num_hidden = 100
num_outputs = 2
ei_ratio = (0.8, 0.2)

time_step = 1 * u.ms
bst.environ.set(dt=time_step)

stimulate = 500 * u.ms
delay = 1000 * u.ms
response = 1000 * u.ms

num_steps = (stimulate + delay + response) / time_step
freq = 500 * u.Hz

batch_size = 128
epoch = 1000


net = SNN()