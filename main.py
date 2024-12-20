import brainstate as bst
import braintools as bts
import brainunit as u
import jax
import jax.numpy as jnp
from jax.numpy import nan

bst.random.seed(43)

from model import SNN_ext
from utils import data_generate_1212, current_generate, plot_voltage_traces

num_inputs = 20
num_hidden = 100
num_outputs = 2

time_step = 1 * u.ms
bst.environ.set(dt=time_step)

stimulate = int((500 * u.ms).to_decimal(time_step.unit))
delay = int((1000 * u.ms).to_decimal(time_step.unit))
response = int((1000 * u.ms).to_decimal(time_step.unit))

num_steps = stimulate + delay + response
freq = 500 * u.Hz

batch_size = 1
epoch = 20000

net = SNN_ext(num_inputs, num_hidden, num_outputs)

x_data, y_data = data_generate_1212(batch_size, num_steps, net, stimulate, delay, freq)
current = current_generate(batch_size, num_steps, stimulate, delay, 0.0 * u.mA, 30.0 * u.mA)

optimizer = bst.optim.Adam(lr=3e-3, beta1=0.9, beta2=0.999)
optimizer.register_trainable_weights(net.states(bst.ParamState))

# total_count_state = bst.State(jnp.zeros((batch_size, num_outputs)))

def loss_fn():
    # spks = bst.compile.for_loop(net.update, x_data, current)

    for _x_data, _current in zip(x_data, current):
        spks = net.update(_x_data, _current)

    spks_count = u.math.sum(spks[stimulate + delay:], axis=0)

    total_count = u.math.sum(spks_count, axis=1)

    predictions = spks_count / total_count[:, u.math.newaxis]

    # plot_voltage_traces(outs, y_data)

    # print(predictions)
    for prediction in predictions.primal:
        for _ in prediction:
            if jnp.isnan(_):
                return 100.

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
