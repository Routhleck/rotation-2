import brainstate as bst
import braintools as bts
import brainunit as u
import jax
import jax.numpy as jnp
from jax.numpy import nan

bst.random.seed(43)

from model import SNN_ext
from utils import (data_generate_1212, current_generate,
                   plot_data, predict_and_visualize_net_activity)

num_inputs = 20
num_hidden = 100
num_outputs = 2

time_step = 1 * u.ms
bst.environ.set(dt=time_step)

stimulate = int((500 * u.ms).to_decimal(time_step.unit))
delay = int((1000 * u.ms).to_decimal(time_step.unit))
response = int((1000 * u.ms).to_decimal(time_step.unit))

num_steps = stimulate + delay + response
freq = 300 * u.Hz

batch_size = 8
epoch = 40

net = SNN_ext(num_inputs, num_hidden, num_outputs)

x_data, y_data = data_generate_1212(batch_size, num_steps, net, stimulate, delay, freq)
current = current_generate(batch_size, num_steps, stimulate, delay, 5.0 * u.mA, 10.0 * u.mA)

optimizer = bst.optim.Adam(lr=3e-3, beta1=0.9, beta2=0.999)
optimizer.register_trainable_weights(net.states(bst.ParamState))

# plot_data(x_data)

# total_count_state = bst.State(jnp.zeros((batch_size, num_outputs)))

def loss_fn():
    predictions = bst.compile.for_loop(net.update, x_data, current)

    # for _x_data, _current in zip(x_data, current):
    #     spks = net.update(_x_data, _current)

    predictions = predictions[stimulate + delay:]

    predictions = u.math.mean(predictions, axis=0)

    return bts.metric.softmax_cross_entropy_with_integer_labels(predictions, y_data).mean()


# @bst.compile.jit
def train_fn():
    bst.nn.init_all_states(net, batch_size=batch_size)
    grads, l = bst.augment.grad(loss_fn, net.states(bst.ParamState), return_value=True)()
    optimizer.update(grads)

    return l


if __name__ == "__main__":
    train_losses = []
    for i in range(1, epoch + 1):
        loss = train_fn()
        train_losses.append(loss)
        # if i % 10 == 0:
        print(f"Epoch {i}, Loss: {loss}")

    predict_and_visualize_net_activity(net, batch_size, x_data, y_data, current)