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
                   cal_model_accuracy, plot_accuracy, plot_loss, plot_gevfit_shape, plot_q_coreness)
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

# plot_current(current)

# total_count_state = bst.State(jnp.zeros((batch_size, num_outputs)))

def loss_fn():
    # random select batch_size samples
    predictions, r_V = bst.compile.for_loop(net.update, x_data, current)
    # for _x_data, _current in zip(x_data, current):
    #     predictions, r_V = net.update(_x_data, _current)


    weight_matrix = net.get_weight_matrix()

    # delays = predictions[:stimulate + delay]
    predictions = predictions[stimulate + delay:]
    r_V = jnp.abs(r_V[:stimulate + delay])

    predictions = u.math.mean(predictions, axis=0)

    ce = bts.metric.softmax_cross_entropy_with_integer_labels(predictions, y_data).mean()
    communicability = communicability_loss(weight_matrix, comms_factor=1)
    activity = (r_V.mean(axis=(0,1)) * net.r2r_conn * weight_matrix).sum() / net.r2r_conn.sum()
    # delay_activity_loss = u.math.mean(u.math.abs(delays[:, :, 0]) + u.math.abs(delays[:, :, 1]))

    # delay_activity_penalty = 0.01 * (delay_activity_loss ** 2)

    return ce + 1. * communicability + 1. * activity


@bst.compile.jit
def train_fn():
    bst.nn.init_all_states(net, batch_size=batch_size)
    net.start_spike_count()
    grads, l = bst.augment.grad(loss_fn, net.states(bst.ParamState), return_value=True)()

    acc = cal_model_accuracy(x_data, y_data, net, current, stimulate, delay)

    optimizer.update(grads)
    # 权重最小为0
    net.set_weight_matrix(jnp.clip(net.get_weight_matrix(), 0, None))
    return l, acc, net.get_weight_matrix(), net.get_spike_counts()


if __name__ == "__main__":
    train_losses = []
    accuracies = []
    weight_matrixs = []
    spike_counts = []
    for i in range(1, epoch + 1):
        loss, accuracy, weight_matrix, spike_count = train_fn()
        train_losses.append(loss)
        accuracies.append(accuracy)
        weight_matrixs.append(np.asarray(weight_matrix))
        spike_counts.append(np.swapaxes(np.asarray(spike_count), 0, 1))
        # if i % 10 == 0:
        print(f"Epoch {i}, Loss: {loss}, Accuracy: {accuracy}, Activity: {spike_count.sum()}")

    plot_accuracy(accuracies)
    plot_loss(train_losses)
    predict_and_visualize_net_activity(net, batch_size, x_data, y_data, current)

    r2r_conn = np.asarray(net.r2r_conn)

    # plot gevfit shape and coreness
    plot_gevfit_shape(weight_matrixs, r2r_conn)
    plot_q_coreness(weight_matrixs, r2r_conn)

    # save weight_matrix and conn_matrix
    np.savez("conn_weight.npz", r2r_conn=r2r_conn, r2r_weights=weight_matrixs, spike_counts=spike_counts)