import brainstate as bst
import braintools as bts
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.numpy import nan

bst.random.seed(43)

from model import SNN_ext
from utils import (data_generate_1221, current_generate,
                   plot_data, plot_current, predict_and_visualize_net_activity,
                   cal_model_accuracy, plot_accuracy, plot_loss)
from loss import communicability_loss

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

batch_size = 40
epoch = 150

net = SNN_ext(num_inputs, num_hidden, num_outputs)

x_data, y_data = data_generate_1221(batch_size, num_steps, net, stimulate, delay, freq)
common_current = 3.0 * u.mA
go_cue_current = 5.0 * u.mA
current = current_generate(batch_size, num_steps, stimulate, delay, common_current, go_cue_current)

optimizer = bst.optim.Adam(lr=1e-3, beta1=0.9, beta2=0.999)
optimizer.register_trainable_weights(net.states(bst.ParamState))

plot_current(current)

# total_count_state = bst.State(jnp.zeros((batch_size, num_outputs)))

def loss_fn():
    # random select batch_size samples
    predictions, rec_spikes = bst.compile.for_loop(net.update, x_data, current)

    weight_matrix = net.get_weight_matrix()

    delays = predictions[:stimulate + delay]
    predictions = predictions[stimulate + delay:]
    delay_rec_spikes = rec_spikes[:stimulate + delay]

    predictions = u.math.mean(predictions, axis=0)

    ce = bts.metric.softmax_cross_entropy_with_integer_labels(predictions, y_data).mean()
    communicability = communicability_loss(weight_matrix, comms_factor=1)
    activity = delay_rec_spikes.mean()
    delay_activity_loss = u.math.mean(u.math.abs(delays[:, :, 0]) + u.math.abs(delays[:, :, 1]))

    activity_penalty = 1000 * (activity ** 2)
    delay_activity_penalty = 0.01 * (delay_activity_loss ** 2)

    return ce + 0.001 * communicability + activity_penalty + delay_activity_penalty


# @bst.compile.jit
def train_fn():
    bst.nn.init_all_states(net, batch_size=batch_size)
    grads, l = bst.augment.grad(loss_fn, net.states(bst.ParamState), return_value=True)()

    acc = cal_model_accuracy(x_data, y_data, net, current, stimulate, delay)

    optimizer.update(grads)
    return l, acc


if __name__ == "__main__":
    train_losses = []
    accuracies = []
    for i in range(1, epoch + 1):
        loss, accuracy = train_fn()
        train_losses.append(loss)
        accuracies.append(accuracy)
        # if i % 10 == 0:
        print(f"Epoch {i}, Loss: {loss}")
        print(f"Epoch {i}, Accuracy: {accuracy}")

    plot_accuracy(accuracies)
    plot_loss(train_losses)
    predict_and_visualize_net_activity(net, batch_size, x_data, y_data, current)

    # save weight_matrix and conn_matrix
    r2r_conn = net.r2r_conn
    r2r_weight = net.get_weight_matrix()

    np.savez("conn_weight.npz", r2r_conn=r2r_conn, r2r_weight=r2r_weight)