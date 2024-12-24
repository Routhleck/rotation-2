import os

import brainstate as bst
import brainunit as u
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import braintools as bts
import jax.numpy as jnp
from scipy.stats import genextreme, gamma
import networkx as nx
import cpnet
from networkx.algorithms.community import kernighan_lin_bisection as kl
import matlab.engine

def current_generate(batch_size, num_steps, stimulate, delay, common_volt, go_cue_volt):
    current = u.math.zeros((num_steps, batch_size, 1)) * u.mA
    current[:stimulate + delay, :, :] = common_volt
    current[stimulate + delay:
            stimulate + delay + stimulate,
    :, :] = go_cue_volt
    current[stimulate + delay + stimulate:, :, :] = common_volt

    return current

def plot_current(current):
    plt.plot(current[:, 0, 0])
    plt.xlabel("Time (ms)")
    plt.ylabel("Current (mA)")
    plt.title("Current vs Time")
    plt.show()

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
                                net.num_in - middle_index) < 0.7 * freq * bst.environ.get_dt())
        else:
            x_data = x_data.at[:stimulate, i, :middle_index].set(
                bst.random.rand(stimulate,
                                net.num_in - middle_index) < 0.7 * freq * bst.environ.get_dt())
            x_data = x_data.at[:stimulate, i, middle_index:net.num_in].set(
                bst.random.rand(stimulate, middle_index) < freq * bst.environ.get_dt())

    return x_data, y_data


def data_generate_1221(batch_size, num_steps, net, stimulate, delay, freq):
    y_data = u.math.asarray(bst.random.rand(batch_size) < 0.5, dtype=int)
    x_data = u.math.zeros((num_steps, batch_size, net.num_in))

    middle_index = net.num_in // 2
    for i in range(batch_size):
        if y_data[i] == 1:
            x_data = x_data.at[:stimulate, i, :middle_index].set(
                bst.random.rand(stimulate, middle_index) < freq * bst.environ.get_dt())
            x_data = x_data.at[:stimulate, i, middle_index:net.num_in].set(
                bst.random.rand(stimulate,
                                net.num_in - middle_index) < 0.6 * freq * bst.environ.get_dt())
        else:
            x_data = x_data.at[:stimulate, i, :middle_index].set(
                bst.random.rand(stimulate,
                                net.num_in - middle_index) < 0.6 * freq * bst.environ.get_dt())
            x_data = x_data.at[:stimulate, i, middle_index:net.num_in].set(
                bst.random.rand(stimulate, middle_index) < freq * bst.environ.get_dt())
    # 增加噪声随机添加布尔值
    noise_prob = 0.02
    noise = bst.random.rand(num_steps, batch_size, net.num_in) < noise_prob
    x_data = x_data.at[:, :, :].set(x_data.astype(jnp.int32) | noise)

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
        plt.imshow(x_data.swapaxes(0, 1)[data_id].transpose(), cmap=plt.cm.binary,
                   interpolation='nearest', aspect="auto")
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


def cal_model_accuracy(x_test, y_test, net, ext_current, stimulate=500, delay=1000):
    vs, spikes, outs = bst.compile.for_loop(net.predict, x_test, ext_current)
    outs = outs[stimulate + delay:]
    am = get_model_predict(outs)  # 获取最大值的索引
    acc = u.math.mean(y_test == am)
    return acc, am


def print_classification_accuracy(output, target, stimulate=500, delay=1000):
    """一个简易的小工具函数，用于计算分类准确率"""
    # m = u.math.max(output, axis=0)  # 获取最大值
    output = output[stimulate + delay:]
    am = get_model_predict(output)  # 获取最大值的索引
    acc = u.math.mean(target == am)  # 与目标值比较
    print("准确率 %.3f" % acc)


def predict_and_visualize_net_activity(net, batch_size, x_data, y_data, ext_current):
    bst.nn.init_all_states(net, batch_size=batch_size)
    vs, spikes, outs = bst.compile.for_loop(net.predict, x_data, ext_current, pbar=bst.compile.ProgressBar(10))
    plot_voltage_traces(vs, spk=spikes, spike_height=5 * u.mV, show=False)
    plot_voltage_traces(outs, y_data)
    print_classification_accuracy(outs, y_data)


def plot_loss(train_losses):
    plt.plot(np.asarray(jnp.asarray(train_losses)))
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs Epoch")
    plt.show()

def moving_averge(data, window_size):
    return jnp.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_accuracy(accuracies):
    accuracies = jnp.asarray(accuracies)
    smoothed_accuracies = moving_averge(accuracies, 10)
    plt.plot(np.asarray(smoothed_accuracies))
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.show()

def get_abs_non_nan_weight_matrixs(weight_matrixs, r2r_conn):
    abs_non_nan_weight_matrixs = []
    for weight_matrix in weight_matrixs:
        weight_matrix = weight_matrix * r2r_conn
        weight_matrix[weight_matrix == 0] = jnp.nan
        weight_matrix = np.abs(weight_matrix)
        abs_non_nan_weight_matrixs.append(weight_matrix[~np.isnan(weight_matrix)])
    return abs_non_nan_weight_matrixs

def plot_gevfit_shape(weight_matrixs, r2r_conn):
    abs_non_nan_weight_matrixs = get_abs_non_nan_weight_matrixs(weight_matrixs, r2r_conn)
    shapes = []
    for weight_matrix in abs_non_nan_weight_matrixs:
        shape, loc, scale = genextreme.fit(weight_matrix)
        shapes.append(shape)
    # plot shape
    plt.plot(shapes)
    plt.xlabel("Epoch")
    plt.ylabel("GEV Shape")
    plt.title("Shape vs Epoch")
    plt.show()

def plot_gamfit_alpha_beta(weight_matrixs, r2r_conn):
    abs_non_nan_weight_matrixs = get_abs_non_nan_weight_matrixs(weight_matrixs, r2r_conn)
    alphas = []
    betas = []
    for weight_matrix in abs_non_nan_weight_matrixs:
        alpha, loc, beta = gamma.fit(weight_matrix)
        alphas.append(alpha)
        betas.append(beta)
    # plot alpha
    plt.plot(alphas)
    plt.xlabel("Epoch")
    plt.ylabel("Gamma Alpha")
    plt.title("Alpha vs Epoch")
    plt.show()
    # plot beta
    plt.plot(betas)
    plt.xlabel("Epoch")
    plt.ylabel("Gamma Beta")
    plt.title("Beta vs Epoch")
    plt.show()

    x = np.linspace(0.01, 10, 1000)
    plt.figure(figsize=(10, 6))

    alphas = alphas[-10:]

    for i, (alpha, beta) in enumerate(zip(alphas, betas)):
        pdf = gamma.pdf(x, alpha, scale=beta)

        plt.loglog(x, pdf, label=f'Epoch {len(weight_matrixs) - 10 + i+1}')

    plt.title('Log-log plot of Gamma distribution')
    plt.xlabel('Log of value')
    plt.ylabel('Log of Probability Density')
    plt.legend()
    plt.grid(True, which='both', ls="--")
    plt.show()


def plot_q_coreness(weight_matrixs, r2r_conn):
    # abs_non_nan_weight_matrixs = get_abs_non_nan_weight_matrixs(weight_matrixs, r2r_conn)
    eng = matlab.engine.start_matlab()
    eng.addpath(os.path.dirname(__file__))

    q_coreness = []

    for weight_matrix in weight_matrixs:
        C, q = eng.core_periphery_dir(weight_matrix, nargout=2)

        q_coreness.append(q)

    # plot q_coreness
    plt.plot(q_coreness)
    plt.xlabel("Epoch")
    plt.ylabel("Q Coreness")
    plt.title("Q Coreness vs Epoch")
    plt.show()

def plot_weight_prob_log(weight_matrixs, r2r_conn):
    abs_non_nan_weight_matrixs = get_abs_non_nan_weight_matrixs(weight_matrixs, r2r_conn)

