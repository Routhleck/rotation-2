import brainstate as bst
import braintools as bts
import brainunit as u
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

bst.random.seed(43)

from model import SNN_ext
from utils import data_generate_1212, current_generate, plot_data

num_inputs = 10
num_hidden = 100
num_outputs = 2

time_step = 1 * u.ms
bst.environ.set(dt=time_step)

stimulate = int((500 * u.ms).to_decimal(time_step.unit))
delay = int((1000 * u.ms).to_decimal(time_step.unit))
response = int((1000 * u.ms).to_decimal(time_step.unit))

num_steps = stimulate + delay + response
freq = 500 * u.Hz

batch_size = 64
epoch = 20

net = SNN_ext(num_inputs, num_hidden, num_outputs)

x_data, y_data = data_generate_1212(batch_size, num_steps, net, stimulate, delay, freq)
current = current_generate(batch_size, num_steps, stimulate, delay, 20.0 * u.mA, 50.0 * u.mA)

# plot_current = current[:, 1, :]
#
# plt.figure()
# plt.plot(plot_current)
# plt.xlabel("Time")
# plt.ylabel("Current")
# plt.title("Current vs Time")
# plt.margins(x=0)
# plt.show()
# plot_data(x_data)




def plot_voltage_traces(mem, y_data=None, spk=None, dim=(3, 5), spike_height=5, show=True):
    fig, gs = bts.visualize.get_figure(*dim, 3, 3)
    if spk is not None:
        mem[spk > 0.0] = spike_height
    if isinstance(mem, u.Quantity):
        mem = mem.to_decimal(u.mV)
    for i in range(np.prod(dim)):
        if i == 0:
            a0 = ax = plt.subplot(gs[i])
        else:
            ax = plt.subplot(gs[i], sharey=a0)
        ax.plot(mem[:, i])

        # 在横坐标为4和12的位置画竖线
        ax.axvline(x=stimulate, color='r', linestyle='--')
        ax.axvline(x=stimulate + delay, color='r', linestyle='--')
        ax.set_title(f"Neuron {i}, True class: {y_data[i]}") if y_data is not None else ax.set_title(f"Neuron {i}")
    if show:
        plt.show()


def print_classification_accuracy(output, target):
    """一个简易的小工具函数，用于计算分类准确率"""
    # m = u.math.max(output, axis=0)  # 获取最大值
    am = get_model_predict(output)  # 获取最大值的索引
    acc = u.math.mean(target == am)  # 与目标值比较
    print("准确率 %.3f" % acc)


def predict_and_visualize_net_activity(net):
    bst.nn.init_all_states(net, batch_size=batch_size)
    vs, spikes, outs = bst.compile.for_loop(net.predict, x_data, pbar=bst.compile.ProgressBar(10))
    plot_voltage_traces(vs, spk=spikes, spike_height=5 * u.mV, show=False)
    plot_voltage_traces(outs, y_data)
    print_classification_accuracy(outs, y_data)


optimizer = bst.optim.Adam(lr=3e-3, beta1=0.9, beta2=0.999)
optimizer.register_trainable_weights(net.states(bst.ParamState))


def get_model_predict(output):
    m = u.math.max(output, axis=0)  # 获取最大值
    am = u.math.argmax(m, axis=1)  # 获取最大值的索引
    return am


i2r_weight = []
r2o_weight = []
r2r_weight = []
r_V = []
model_predict = []


def loss_fn_1():
    for _x_data, _current in zip(x_data, current):
        spks = net.update(_x_data, _current)
    # predictions, _V = bst.compile.for_loop(net.update, x_data, current)

    predictions = predictions[stimulate + delay:]
    # model_predict.append(get_model_predict(predictions))
    predictions = u.math.mean(predictions, axis=0)

    return bts.metric.softmax_cross_entropy_with_integer_labels(predictions, y_data).mean()


def loss_fn_2():
    predictions, _V = bst.compile.for_loop(net.update, x_data, current)

    delays = predictions[:stimulate + delay]

    delay_loss = u.math.mean(u.math.abs(delays[:, :, 0]) + u.math.abs(delays[:, :, 1]))

    predictions = predictions[stimulate + delay:]
    _model_predict = get_model_predict(predictions)
    model_predict.append(_model_predict)
    r_V.append(_V)
    # spike_count.append(u.math.count_nonzero(spikes, axis=0))

    predictions = u.math.mean(predictions, axis=0)
    return bts.metric.softmax_cross_entropy_with_integer_labels(predictions,
                                                                y_data).mean() + 0.01 * delay_loss


# @bst.compile.jit
def train_fn_1():
    bst.nn.init_all_states(net, batch_size=batch_size)
    grads, l = bst.augment.grad(loss_fn_1, net.states(bst.ParamState), return_value=True)()
    optimizer.update(grads)

    states = net.states()
    _i2r_weight = states['i2r', 'layers', 0, 'weight'].value['weight'].mantissa
    _r2o_weight = states['r2o', 'weight'].value['weight']
    _r2r_weight = states['r2r', 'layers', 0, 'weight'].value['weight'].mantissa

    return l, _i2r_weight, _r2o_weight, _r2r_weight


# @bst.compile.jit
def train_fn_2():
    bst.nn.init_all_states(net, batch_size=batch_size)
    grads, l = bst.augment.grad(loss_fn_2, net.states(bst.ParamState), return_value=True)()
    optimizer.update(grads)

    states = net.states()
    _i2r_weight = states['i2r', 'layers', 0, 'weight'].value['weight'].mantissa
    _r2o_weight = states['r2o', 'weight'].value['weight']
    _r2r_weight = states['r2r', 'layers', 0, 'weight'].value['weight'].mantissa

    return l, _i2r_weight, _r2o_weight, _r2r_weight


train_losses = []
for i in range(1, epoch + 1):
    loss, _i2r_weight, _r2o_weight, _r2r_weight = train_fn_1()
    train_losses.append(loss)

    i2r_weight.append(_i2r_weight)
    r2o_weight.append(_r2o_weight)
    r2r_weight.append(_r2r_weight)
    # if i % 10 == 0:
    print(f'Epoch {i}, Loss = {loss:.4f}')

# for i in range(epoch, epoch*2 + 1):
#     loss = train_fn_2()
#     train_losses.append(loss)
#     # if i % 10 == 0:
#     print(f'Epoch {i}, Loss = {loss:.4f}')

plt.plot(np.asarray(jnp.asarray(train_losses)))
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss vs Epoch")
plt.show()

predict_and_visualize_net_activity(net)

from export import save_input, save_train_states

states_dict = {
    'i2r_weight': i2r_weight,
    'r2o_weight': r2o_weight,
    'r2r_weight': r2r_weight,
    'r_V': [V.primal for V in r_V],
    'model_predict': model_predict
}

save_input(x_data, y_data, filename="../inputs.npz")
save_train_states(states_dict, filename="../states.npz")
