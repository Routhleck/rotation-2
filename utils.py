import brainstate as bst
import brainunit as u
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import braintools as bts


def current_generate(batch_size, num_steps, stimulate, delay, common_volt, go_cue_volt):
    current = u.math.zeros((num_steps, batch_size, 1)) * u.mA
    current[:stimulate + delay, :, :] = common_volt
    current[stimulate + delay:
            stimulate + delay + stimulate,
    :, :] = go_cue_volt
    current[stimulate + delay + stimulate:, :, :] = common_volt

    return current


def data_generate_1212(batch_size, num_steps, net, stimulate, delay, freq):
    y_data = u.math.asarray(bst.random.rand(batch_size) < 0.5, dtype=int)
    x_data = u.math.zeros((num_steps, batch_size, net.num_in))

    middle_index = net.num_in // 2
    for i in range(batch_size):
        if y_data[i] == 1:
            x_data = x_data.at[:stimulate, i, :middle_index].set(
                bst.random.rand(stimulate, middle_index) < freq * bst.environ.get_dt())
            x_data = x_data.at[:stimulate, i, middle_index:net.num_in].set(
                bst.random.rand(stimulate,
                                net.num_in - middle_index) < 0.5 * freq * bst.environ.get_dt())
        else:
            x_data = x_data.at[:stimulate, i, :middle_index].set(
                bst.random.rand(stimulate,
                                net.num_in - middle_index) < 0.5 * freq * bst.environ.get_dt())
            x_data = x_data.at[:stimulate, i, middle_index:net.num_in].set(
                bst.random.rand(stimulate, middle_index) < freq * bst.environ.get_dt())

    return x_data, y_data


def data_generate_1208(batch_size, num_steps, net, go_cue_inputs, stimulate, delay, freq):
    y_data = u.math.asarray(bst.random.rand(batch_size) < 0.5, dtype=int)
    x_data = u.math.zeros((num_steps, batch_size, net.num_in))

    middle_index = (net.num_in - go_cue_inputs) // 2
    for i in range(batch_size):
        if y_data[i] == 1:
            x_data = x_data.at[:stimulate, i, :middle_index].set(
                bst.random.rand(stimulate, middle_index) < freq * bst.environ.get_dt())
            x_data = x_data.at[:stimulate, i, middle_index:net.num_in - go_cue_inputs].set(
                bst.random.rand(stimulate,
                                net.num_in - middle_index - go_cue_inputs) < 0.5 * freq * bst.environ.get_dt())
        else:
            x_data = x_data.at[:stimulate, i, :middle_index].set(
                bst.random.rand(stimulate,
                                net.num_in - middle_index - go_cue_inputs) < 0.5 * freq * bst.environ.get_dt())
            x_data = x_data.at[:stimulate, i, middle_index:net.num_in - go_cue_inputs].set(
                bst.random.rand(stimulate, middle_index) < freq * bst.environ.get_dt())

    x_data = x_data.at[:stimulate, :, net.num_in - go_cue_inputs:].set(
        u.math.ones((stimulate, batch_size, go_cue_inputs)))
    x_data = x_data.at[stimulate + delay: stimulate + delay + stimulate, :, net.num_in - go_cue_inputs:].set(
        u.math.ones((stimulate, batch_size, go_cue_inputs)))

    return x_data, y_data


def plot_data(x_data):
    for data_id in range(5):
        plt.clf()
        plt.imshow(x_data.swapaxes(0, 1)[data_id].transpose(), cmap=plt.cm.gray_r, aspect="auto")
        plt.xlabel("Time (ms)")
        plt.ylabel("Unit")
        sns.despine()

        plt.show()

def plot_voltage_traces(mem, y_data=None, spk=None, dim=(3, 5), spike_height=5, show=True,
                        stimulate=500, delay=1000):
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

def get_model_predict(output):
    m = u.math.max(output, axis=0)  # 获取最大值
    am = u.math.argmax(m, axis=1)  # 获取最大值的索引
    return am

def print_classification_accuracy(output, target):
    """一个简易的小工具函数，用于计算分类准确率"""
    # m = u.math.max(output, axis=0)  # 获取最大值
    am = get_model_predict(output)  # 获取最大值的索引
    acc = u.math.mean(target == am)  # 与目标值比较
    print("准确率 %.3f" % acc)

def predict_and_visualize_net_activity(net, batch_size, x_data, y_data, ext_current):
    bst.nn.init_all_states(net, batch_size=batch_size)
    vs, spikes, outs = bst.compile.for_loop(net.predict, x_data, ext_current, pbar=bst.compile.ProgressBar(10))
    plot_voltage_traces(vs, spk=spikes, spike_height=5 * u.mV, show=False)
    plot_voltage_traces(outs, y_data)
    print_classification_accuracy(outs, y_data)