import braintools as bts
import brainunit as u
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import brainstate as bst

bst.random.seed(42)

from model import SNN, SNN_ext
from utils import plot_data, data_generate_1212, current_generate

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

batch_size = 128
epoch = 50

net = SNN_ext(num_inputs, num_hidden, num_outputs)

x_data, y_data = data_generate_1212(batch_size, num_steps, net, stimulate, delay, freq)
current = current_generate(batch_size, num_steps, stimulate, delay, 5.0 * u.mA, 15.0 * u.mA)


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
        ax.axvline(x=stimulate+delay, color='r', linestyle='--')
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

def loss_fn_1():
    predictions = bst.compile.for_loop(net.update, x_data, current)

    predictions = predictions[stimulate+delay:]
    # model_predict.append(get_model_predict(predictions))
    predictions = u.math.mean(predictions, axis=0)

    return bts.metric.softmax_cross_entropy_with_integer_labels(predictions, y_data).mean()

@bst.compile.jit
def train_fn_1():
    bst.nn.init_all_states(net, batch_size=batch_size)
    grads, l = bst.augment.grad(loss_fn_1, net.states(bst.ParamState), return_value=True)()
    optimizer.update(grads)

    return l

train_losses = []
for i in range(1, epoch + 1):
    loss = train_fn_1()
    train_losses.append(loss)
    if i % 10 == 0:
        print(f'Epoch {i}, Loss = {loss:.4f}')

plt.plot(np.asarray(jnp.asarray(train_losses)))
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss vs Epoch")
plt.show()

predict_and_visualize_net_activity(net)