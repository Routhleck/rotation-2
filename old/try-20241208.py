import braintools as bts
import brainunit as u
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import brainstate as bst

from model import SNN
from utils import plot_data, data_generate_1208

num_inputs  = 30   # 输入层神经元个数
go_cue_inputs = 10
num_hidden  = 100     # 隐藏层神经元个数
num_outputs = 2     # 输出层神经元个数

time_step = 4/ 30 * u.second
bst.environ.set(dt=time_step)   # 设置仿真时间步长

scale = 4
stimulate = 4 * scale
delay = 8 * scale
response = 8 * scale
num_steps  = stimulate + delay + response
freq = 5 * u.Hz

batch_size = 128
epoch_1 = 30
epoch_2 = 50

bst.random.seed(42)

net = SNN(num_inputs, num_hidden, num_outputs)

x_data, y_data = data_generate_1208(batch_size, num_steps, net, go_cue_inputs, stimulate, delay, freq)

plot_data(x_data)

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

i2r_weight = []
i2r_bias = []
i2r_g = []
o_g = []
r_V = []
r2o_bias = []
r2o_weight = []
r2r_bias = []
r2r_weight = []
r2r_g = []
model_predict = []

def get_model_predict(output):
    m = u.math.max(output, axis=0)  # 获取最大值
    am = u.math.argmax(m, axis=1)  # 获取最大值的索引
    return am

def loss_fn_1():
    predictions = bst.compile.for_loop(net.update, x_data)

    predictions = predictions[stimulate+delay:]
    model_predict.append(get_model_predict(predictions))
    predictions = u.math.mean(predictions, axis=0)

    return bts.metric.softmax_cross_entropy_with_integer_labels(predictions, y_data).mean()

def loss_fn_2():
    predictions = bst.compile.for_loop(net.update, x_data)

    delays = predictions[:stimulate+delay]
    delay_loss = u.math.mean(u.math.abs(delays[:, :, 0]) + u.math.abs(delays[:, :, 1]))

    predictions = predictions[stimulate + delay:]
    model_predict.append(get_model_predict(predictions))
    predictions = u.math.mean(predictions, axis=0)
    return bts.metric.softmax_cross_entropy_with_integer_labels(predictions, y_data).mean() + 0.001 * delay_loss


@bst.compile.jit
def train_fn_1():
    bst.nn.init_all_states(net, batch_size=batch_size)
    grads, l = bst.augment.grad(loss_fn_1, net.states(bst.ParamState), return_value=True)()
    optimizer.update(grads)

    # states = net.states()
    # i2r_bias.append(np.asarray(states['i2r', 'layers', 0, 'weight'].value['bias'].mantissa))
    # i2r_weight.append(np.asarray(states['i2r','layers', 0, 'weight'].value['weight'].mantissa))
    # i2r_g.append(np.asarray(states['i2r','layers', 1, 'g'].value.mantissa))
    # o_g.append(np.asarray(states['o','g'].value))
    # r_V.append(np.asarray(states['r','V'].value.mantissa))
    # r2o_bias.append(np.asarray(states['r2o', 'weight'].value['bias']))
    # r2o_weight.append(np.asarray(states['r2o','weight'].value['weight']))
    # r2r_bias.append(np.asarray(states['r2r', 'layers', 0, 'weight'].value['bias'].mantissa))
    # r2r_weight.append(np.asarray(states['r2r','layers', 0, 'weight'].value['weight'].mantissa))
    # r2r_g.append(np.asarray(states['r2r','layers', 1, 'g'].value.mantissa))
    return l

@bst.compile.jit
def train_fn_2():
    bst.nn.init_all_states(net, batch_size=batch_size)
    grads, l = bst.augment.grad(loss_fn_2, net.states(bst.ParamState), return_value=True)()
    optimizer.update(grads)

    # states = net.states()
    # i2r_bias.append(np.asarray(states['i2r', 'layers', 0, 'weight'].value['bias'].mantissa))
    # i2r_weight.append(np.asarray(states['i2r','layers', 0, 'weight'].value['weight'].mantissa))
    # i2r_g.append(np.asarray(states['i2r','layers', 1, 'g'].value.mantissa))
    # o_g.append(np.asarray(states['o','g'].value))
    # r_V.append(np.asarray(states['r','V'].value.mantissa))
    # r2o_bias.append(np.asarray(states['r2o', 'weight'].value['bias']))
    # r2o_weight.append(np.asarray(states['r2o','weight'].value['weight']))
    # r2r_bias.append(np.asarray(states['r2r', 'layers', 0, 'weight'].value['bias'].mantissa))
    # r2r_weight.append(np.asarray(states['r2r','layers', 0, 'weight'].value['weight'].mantissa))
    # r2r_g.append(np.asarray(states['r2r','layers', 1, 'g'].value.mantissa))
    return l

train_losses = []
for i in range(1, epoch_1 + 1):
    loss = train_fn_1()
    train_losses.append(loss)
    if i % 10 == 0:
        print(f'Epoch {i}, Loss = {loss:.4f}')

for i in range(epoch_1 + 1, epoch_1 + epoch_2 + 1):
    loss = train_fn_2()
    train_losses.append(loss)
    if i % 10 == 0:
        print(f'Epoch {i}, Loss = {loss:.4f}')

plt.plot(np.asarray(jnp.asarray(train_losses)))
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss vs Epoch")
plt.show()

predict_and_visualize_net_activity(net)

from old.export import save_input, save_train_states

states_dict = {
    # 'i2r_bias': i2r_bias,
    'i2r_weight': i2r_weight,
    # 'i2r_g': i2r_g,
    # 'o_g': o_g,
    'r_V': r_V,
    # 'r2o_bias': r2o_bias,
    'r2o_weight': r2o_weight,
    # 'r2r_bias': r2r_bias,
    'r2r_weight': r2r_weight,
    # 'r2r_g': r2r_g
    'model_predict': model_predict
}

save_input(x_data, y_data, filename="../inputs.npz")
save_train_states(states_dict, filename="../states.npz")