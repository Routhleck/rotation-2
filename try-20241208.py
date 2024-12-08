import braintools as bts
import brainunit as u
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


import brainstate as bst

from model import SNN
from utils import plot_data

num_inputs  = 20   # 输入层神经元个数
num_hidden  = 100     # 隐藏层神经元个数
num_outputs = 2     # 输出层神经元个数

time_step = 1 * u.ms
bst.environ.set(dt=time_step)   # 设置仿真时间步长
num_steps  = 200

batch_size = 128
epoch = 100


net = SNN(num_inputs, num_hidden, num_outputs)

freq = 5 * u.Hz

y_data = u.math.asarray(bst.random.rand(batch_size) < 0.5, dtype=int)
x_data = u.math.zeros((num_steps, batch_size, net.num_in))

middle_index = net.num_in // 2
for i in range(batch_size):
    if y_data[i] == 1:
        x_data = x_data.at[:, i, :middle_index].set(bst.random.rand(num_steps, middle_index) < freq * bst.environ.get_dt())
        x_data = x_data.at[:, i, middle_index:].set(bst.random.rand(num_steps, net.num_in - middle_index) < 0.5 * freq * bst.environ.get_dt())
    else:
        x_data = x_data.at[:, i, :middle_index].set(bst.random.rand(num_steps, net.num_in - middle_index) < 0.5 * freq * bst.environ.get_dt())
        x_data = x_data.at[:, i, middle_index:].set(bst.random.rand(num_steps, middle_index) < freq * bst.environ.get_dt())

# plot_data(x_data)

def plot_voltage_traces(mem, spk=None, dim=(3, 5), spike_height=5, show=True):
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
    if show:
        plt.show()

def print_classification_accuracy(output, target):
    """一个简易的小工具函数，用于计算分类准确率"""
    m = u.math.max(output, axis=0)  # 获取最大值
    am = u.math.argmax(m, axis=1)  # 获取最大值的索引
    acc = u.math.mean(target == am)  # 与目标值比较
    print("准确率 %.3f" % acc)

def predict_and_visualize_net_activity(net):
    bst.nn.init_all_states(net, batch_size=batch_size)
    vs, spikes, outs = bst.compile.for_loop(net.predict, x_data, pbar=bst.compile.ProgressBar(10))
    plot_voltage_traces(vs, spikes, spike_height=5 * u.mV, show=False)
    plot_voltage_traces(outs)
    print_classification_accuracy(outs, y_data)

optimizer = bst.optim.Adam(lr=3e-3, beta1=0.9, beta2=0.999)
optimizer.register_trainable_weights(net.states(bst.ParamState))

def loss_fn():
    predictions = bst.compile.for_loop(net.update, x_data)
    predictions = u.math.mean(predictions, axis=0)
    return bts.metric.softmax_cross_entropy_with_integer_labels(predictions, y_data).mean()

@bst.compile.jit
def train_fn():
    bst.nn.init_all_states(net, batch_size=batch_size)
    grads, l = bst.augment.grad(loss_fn, net.states(bst.ParamState), return_value=True)()
    optimizer.update(grads)
    return l

train_losses = []
for i in range(1, epoch + 1):
    loss = train_fn()
    train_losses.append(loss)
    if i % 10 == 0:
        print(f'Epoch {i}, Loss = {loss:.4f}')

plt.plot(np.asarray(jnp.asarray(train_losses)))
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss vs Epoch")
plt.show()

predict_and_visualize_net_activity(net)