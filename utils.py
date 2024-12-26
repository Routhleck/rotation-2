import os

import brainstate as bst
import braintools as bts
import brainunit as u
import jax.numpy as jnp
import matlab.engine
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import gamma
from tqdm import tqdm
import networkx as nx
import powerlaw


def current_generate(batch_size, num_steps, stimulate, delay, common_volt, go_cue_volt):
    """
    生成给定批量大小和步数的电流信号。

    参数:
    batch_size (int): 批量大小。
    num_steps (int): 信号的步数。
    stimulate (int): 刺激期的持续时间。
    delay (int): 刺激期开始前的延迟时间。
    common_volt (float): 非刺激期的电压水平。
    go_cue_volt (float): 刺激期的电压水平。

    返回:
    numpy.ndarray: 一个形状为 (num_steps, batch_size, 1) 的3D数组，表示生成的电流信号。
    """
    current = u.math.zeros((num_steps, batch_size, 1)) * u.mA
    current[:stimulate + delay, :, :] = common_volt
    current[stimulate + delay:
            stimulate + delay + stimulate,
    :, :] = go_cue_volt
    current[stimulate + delay + stimulate:, :, :] = common_volt

    return current


def plot_current(current):
    """
    绘制电流随时间变化的图表。

    参数:
    current (numpy.ndarray): 一个三维数组，表示电流数据。假设第一个维度是时间，第二个维度和第三个维度是空间维度。

    返回:
    None
    """
    plt.plot(current[:, 0, 0])
    plt.xlabel("Time (ms)")
    plt.ylabel("Current (mA)")
    plt.title("Current vs Time")
    plt.show()


def data_generate_1221(batch_size, num_steps, net, stimulate, delay, freq):
    """
    生成用于神经网络训练的数据。

    参数:
    batch_size (int): 批处理大小，即一次生成的数据样本数量。
    num_steps (int): 时间步数，即每个样本的时间序列长度。
    net (object): 神经网络对象，包含网络的输入维度信息。
    stimulate (int): 刺激时间步数，即在前多少个时间步内进行刺激。
    delay (int): 延迟时间步数，未在函数中使用。
    freq (float): 刺激频率，用于控制随机生成数据的概率。

    返回:
    tuple: 包含两个元素:
        - x_data (ndarray): 生成的输入数据，形状为 (num_steps, batch_size, net.num_in)。
        - y_data (ndarray): 生成的标签数据，形状为 (batch_size,)。
    """
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


def plot_data(x_data, y_data):
    """
    绘制数据图像。

    参数:
    x_data (numpy.ndarray): 输入数据，假设形状为 (units, time, ...)，
                            其中 units 表示单位数量，time 表示时间步长。

    功能:
    该函数将输入数据的前5个单位的数据绘制为图像。每个图像将显示单位随时间变化的情况。
    使用 plt.imshow() 函数显示图像，并设置颜色映射为二值图 (binary)。
    每个图像的 x 轴表示时间 (ms)，y 轴表示单位。
    使用 sns.despine() 函数去除图像的顶部和右侧脊柱。

    注意:
    该函数假设输入数据的形状为 (units, time, ...)，并且会对数据进行转置和轴交换操作。
    """
    for data_id in range(5):
        plt.clf()
        plt.imshow(x_data.swapaxes(0, 1)[data_id].transpose(), cmap=plt.cm.binary,
                   interpolation='nearest', aspect="auto")
        plt.xlabel("Time (ms)")
        plt.ylabel("Neuron Index", labelpad=10)
        plt.yticks(np.arange(20))
        plt.title(f"class: {y_data[data_id]}")
        sns.despine()

        plt.show()


def plot_voltage_traces(
    mem,
    y_data=None,
    spk=None,
    dim=(3, 5),
    spike_height=5,
    show=True,
    stimulate=500,
    delay=1000
):
    """
    绘制电压轨迹图。

    参数:
    mem (ndarray 或 u.Quantity): 神经元的电压数据。
    y_data (ndarray, 可选): 神经元的真实类别标签。
    spk (ndarray, 可选): 神经元的脉冲数据。
    dim (tuple, 可选): 图形的维度，默认为 (3, 5)。
    spike_height (int, 可选): 脉冲的高度，默认为 5。
    show (bool, 可选): 是否显示图形，默认为 True。
    stimulate (int, 可选): 刺激开始的时间点，默认为 500。
    delay (int, 可选): 刺激结束的时间点，默认为 1000。

    返回:
    无

    示例:
    >>> plot_voltage_traces(mem, y_data, spk)
    """
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
    """
    获取模型预测结果的索引。

    参数:
    output (ndarray): 模型输出的多维数组。

    返回:
    ndarray: 最大值的索引数组。
    """
    m = u.math.max(output, axis=0)  # 获取最大值
    am = u.math.argmax(m, axis=1)  # 获取最大值的索引
    return am


def cal_model_accuracy(x_test, y_test, net, ext_current, stimulate=500, delay=1000):
    """
    计算模型的准确率。

    参数:
    x_test (array-like): 测试集的输入数据。
    y_test (array-like): 测试集的标签数据。
    net (object): 训练好的神经网络模型。
    ext_current (array-like): 外部输入电流。
    stimulate (int, 可选): 刺激时间，默认为500。
    delay (int, 可选): 延迟时间，默认为1000。

    返回:
    tuple: 包含以下两个元素的元组:
        - acc (float): 模型的准确率。
        - am (array-like): 模型预测的标签。
    """
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
    """
    预测并可视化神经网络活动。

    参数:
    net (对象): 神经网络模型。
    batch_size (int): 批处理大小。
    x_data (ndarray): 输入数据。
    y_data (ndarray): 目标数据。
    ext_current (ndarray): 外部电流。

    返回:
    无

    功能:
    1. 初始化神经网络的所有状态。
    2. 使用for循环编译器预测神经网络的输出。
    3. 绘制电压轨迹图，包括电压和尖峰。
    4. 打印分类准确率。
    """
    bst.nn.init_all_states(net, batch_size=batch_size)
    vs, spikes, outs = bst.compile.for_loop(net.predict, x_data, ext_current, pbar=bst.compile.ProgressBar(10))
    plot_voltage_traces(vs, spk=spikes, spike_height=5 * u.mV, show=False)
    plot_voltage_traces(outs, y_data)
    print_classification_accuracy(outs, y_data)


def plot_loss(train_losses):
    """
    绘制训练损失随时间变化的图表。

    参数:
    train_losses (list): 包含每个epoch训练损失的列表。

    返回:
    无
    """
    plt.plot(np.asarray(jnp.asarray(train_losses)))
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs Epoch")
    plt.show()


def moving_averge(data, window_size):
    return jnp.convolve(data, np.ones(window_size) / window_size, mode='valid')


def plot_accuracy(accuracies):
    """
    绘制准确率随时间变化的图表。

    参数:
    accuracies (list or array-like): 包含准确率值的列表或数组。

    返回:
    None
    """
    accuracies = jnp.asarray(accuracies)
    smoothed_accuracies = moving_averge(accuracies, 10)
    plt.plot(np.asarray(smoothed_accuracies))
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.show()


def get_abs_non_nan_weight_matrixs(weight_matrixs, r2r_conn):
    """
    获取绝对值且非NaN的权重矩阵。

    此函数接受一组权重矩阵，并将其与r2r_conn矩阵逐元素相乘。然后，将所有零值替换为NaN，
    并取绝对值。最后，返回一个列表，其中包含所有非NaN值的绝对值权重矩阵。

    参数:
    weight_matrixs (list of ndarray): 权重矩阵的列表。
    r2r_conn (ndarray): 用于逐元素相乘的连接矩阵。

    返回:
    list of ndarray: 绝对值且非NaN的权重矩阵列表。
    """
    abs_non_nan_weight_matrixs = []
    for weight_matrix in weight_matrixs:
        weight_matrix = weight_matrix * r2r_conn
        weight_matrix[weight_matrix == 0] = jnp.nan
        weight_matrix = np.abs(weight_matrix)
        abs_non_nan_weight_matrixs.append(weight_matrix[~np.isnan(weight_matrix)])
    return abs_non_nan_weight_matrixs


def plot_gamfit_alpha_beta(weight_matrixs, r2r_conn):
    """
    绘制权重矩阵的Gamma分布参数alpha和beta随时间变化的图表，并生成Gamma分布的对数对数图。

    参数:
    weight_matrixs (list of np.ndarray): 包含多个权重矩阵的列表。
    r2r_conn (np.ndarray): 用于计算非NaN权重矩阵的连接矩阵。

    返回:
    无返回值。该函数会生成并显示图表。
    """
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

        plt.loglog(x, pdf, label=f'Epoch {len(weight_matrixs) - 10 + i + 1}')

    plt.title('Log-log plot of Gamma distribution')
    plt.xlabel('Log of value')
    plt.ylabel('Log of Probability Density')
    plt.legend()
    plt.grid(True, which='both', ls="--")
    plt.show()


def plot_q_coreness(weight_matrixs):
    """
    根据给定的权重矩阵列表，计算每个矩阵的核心-边缘结构，并绘制Q Coreness随时间变化的图表。

    参数:
    weight_matrixs (list): 权重矩阵的列表，每个矩阵表示一个时间点的连接权重。

    返回:
    tuple: 包含两个元素的元组:
        - C_list (list): 每个权重矩阵对应的核心-边缘分配列表。
        - q_coreness (list): 每个权重矩阵对应的Q Coreness值列表。
    """
    eng = matlab.engine.start_matlab()
    eng.addpath(os.path.dirname(__file__))

    q_coreness = []
    C_list = []

    for weight_matrix in weight_matrixs:
        C, q = eng.core_periphery_dir(weight_matrix, nargout=2)

        q_coreness.append(q)
        C_list.append(C)

    # plot q_coreness
    plt.plot(q_coreness)
    plt.xlabel("Epoch")
    plt.ylabel("Q Coreness")
    plt.title("Q Coreness vs Epoch")
    plt.show()

    return C_list, q_coreness


def calculate_pev(spike_counts, model_predicts, plot_num=4):
    """
    计算PEV（预测解释方差）。

    参数:
    spike_counts (numpy.ndarray): 神经元尖峰计数，形状为 (epoch, batch_size, frame, num_rec)。
    model_predicts (numpy.ndarray): 模型预测值，形状为 (epoch, batch_size)。
    plot_num (int, 可选): 分割数据进行计算的数量，默认为4。

    返回:
    list: 包含PEV数组的列表，每个数组形状为 (frame, num_rec)。

    说明:
    该函数使用MATLAB引擎计算PEV。首先将输入数据重塑为适当的形状，然后将数据分割为多个部分进行计算。
    每个部分的PEV计算结果存储在一个数组中，最终返回包含所有PEV数组的列表。
    """
    eng = matlab.engine.start_matlab()
    eng.addpath(os.path.dirname(__file__))
    # return len=4 list[array shape = (4000, 25, 100)]
    epoch = spike_counts.shape[0]
    batch_size = spike_counts.shape[1]
    frame = spike_counts.shape[2]
    num_rec = spike_counts.shape[3]
    trails = epoch * batch_size

    spike_counts = spike_counts.reshape(spike_counts.shape[0] * spike_counts.shape[1], spike_counts.shape[2],
                                        spike_counts.shape[3])
    model_predicts = model_predicts.reshape(model_predicts.shape[0] * model_predicts.shape[1])
    pev_list = []

    for i in range(plot_num):
        spike_count = spike_counts[trails // plot_num * i: trails // plot_num * (i + 1)]
        model_predict = model_predicts[trails // plot_num * i: trails // plot_num * (i + 1)]
        pev_array = np.zeros((frame, num_rec))
        for j in tqdm(range(frame)):
            for k in range(num_rec):
                spike_count_contiguous = np.ascontiguousarray(spike_count[:, j, k])
                model_predict = np.ascontiguousarray(model_predict)
                pev_array[j, k] = eng.calculate_PEV(spike_count_contiguous, model_predict)
        pev_list.append(pev_array)
    return pev_list


def calculate_mse(data, axis=0):
    """
    计算均方误差（Mean Squared Error, MSE）。

    参数:
    data (numpy.ndarray): 输入数据数组。
    axis (int, 可选): 计算均值和均方误差的轴。默认为0。

    返回:
    float: 计算得到的均方误差。

    示例:
    >>> data = np.array([[1, 2, 3], [4, 5, 6]])
    >>> calculate_mse(data, axis=0)
    array([2.25, 2.25, 2.25])
    >>> calculate_mse(data, axis=1)
    array([0.66666667, 0.66666667])
    """
    mean = np.mean(data, axis=axis)

    # 扩展 mean 的形状以匹配 data 的形状
    if axis == 0:
        mean = mean[np.newaxis, :]  # 将 mean 的形状从 (48,) 扩展为 (1, 48)
    elif axis == 1:
        mean = mean[:, np.newaxis]  # 将 mean 的形状从 (25,) 扩展为 (25, 1)

    mse = np.mean((data - mean) ** 2, axis=axis)
    return mse


def plot_spike_count(spike_counts, C, model_predict, plot_num=4):
    """
    绘制神经元的尖峰计数图。

    参数:
    spike_counts (numpy.ndarray): 尖峰计数数据，形状为 (epochs, neurons)。
    C (numpy.ndarray): 核心和外围的分类标签，形状为 (neurons,)。
    model_predict (numpy.ndarray): 模型预测值，形状为 (epochs, neurons)。
    plot_num (int, 可选): 要绘制的图的数量，默认为 4。

    返回:
    None
    """
    core = C == 1
    periphery = C == 0
    pev_list = calculate_pev(spike_counts, model_predict, plot_num)

    fig, axs = plt.subplots(2, plot_num, figsize=(20, 10))

    # 找到所有 pev_list 中的最小值和最大值，用于统一 colorbar 的范围
    vmin = min(np.min(pev) for pev in pev_list)
    vmax = max(np.max(pev) for pev in pev_list)

    # 统一 y 轴范围
    y_min = np.inf
    y_max = -np.inf

    epoch = spike_counts.shape[0]
    for i in range(plot_num):
        # 绘制热图
        im = axs[0, i].imshow(pev_list[i].transpose(), cmap='viridis', interpolation='nearest',
                              aspect='auto', vmin=vmin, vmax=vmax)
        axs[0, i].set_xlabel("Frame")
        axs[0, i].set_ylabel("Neuron Index")
        axs[0, i].set_title(f"Stage {i + 1}")

        # 计算 core 和 periphery 的 spike count 的均值、最大值和最小值
        core_spike_count_mean = np.mean(pev_list[i][:, core[epoch % plot_num * i]], axis=1)
        periphery_spike_count_mean = np.mean(pev_list[i][:, periphery[epoch % plot_num * i]], axis=1)

        core_spike_count_mse = calculate_mse(pev_list[i][:, core[epoch % plot_num * i]], axis=1)
        periphery_spike_count_mse = calculate_mse(pev_list[i][:, periphery[epoch % plot_num * i]], axis=1)

        # 绘制均值线
        axs[1, i].plot(core_spike_count_mean, color='red', label='Core Mean')
        axs[1, i].plot(periphery_spike_count_mean, color='green', label='Periphery Mean')

        # 绘制 error bar（使用 MSE 作为误差范围）
        # axs[1, i].errorbar(range(len(core_spike_count_mean)), core_spike_count_mean, yerr=core_spike_count_mse,
        #                    color='red', fmt='none', capsize=5, label='Core MSE')
        # axs[1, i].errorbar(range(len(periphery_spike_count_mean)), periphery_spike_count_mean,
        #                    yerr=periphery_spike_count_mse,
        #                    color='green', fmt='none', capsize=5, label='Periphery MSE')
        axs[1, i].fill_between(range(len(core_spike_count_mean)), core_spike_count_mean - core_spike_count_mse,
                               core_spike_count_mean + core_spike_count_mse, color='red', alpha=0.3)
        axs[1, i].fill_between(range(len(periphery_spike_count_mean)),
                               periphery_spike_count_mean - periphery_spike_count_mse,
                               periphery_spike_count_mean + periphery_spike_count_mse, color='green', alpha=0.3)

        # 设置统一的 y 轴范围
        y_min = min(y_min, np.min(core_spike_count_mean - core_spike_count_mse),
                    np.min(periphery_spike_count_mean - periphery_spike_count_mse))
        y_max = max(y_max, np.max(core_spike_count_mean + core_spike_count_mse),
                    np.max(periphery_spike_count_mean + periphery_spike_count_mse))

        # 添加图例和标签
        axs[1, i].legend()
        axs[1, i].set_xlabel('Frame')
        axs[1, i].set_ylabel('PEV')
        axs[1, i].set_title(f'Stage {i + 1}')

    cbar_ax = fig.add_axes([0.92, 0.55, 0.02, 0.35])  # 调整 colorbar 的位置和大小
    fig.colorbar(im, cax=cbar_ax)

    for i in range(plot_num):
        axs[1, i].set_ylim(y_min, y_max)

    # plt.tight_layout()
    # 添加全局标题
    plt.suptitle('PEV value and Core/Periphery PEV value', fontsize=16)
    plt.show()

def plot_modularity(weight_matrixs):
    """
    绘制模块度随时间变化的图表。

    参数:
    weight_matrixs (list of ndarray): 包含多个权重矩阵的列表。
    r2r_conn (ndarray): 用于计算非NaN权重矩阵的连接矩阵。

    返回:
    None
    """
    eng = matlab.engine.start_matlab()
    eng.addpath(os.path.dirname(__file__))
    modularity = []
    for weight_matrix in weight_matrixs:
        M, Q = eng.community_louvain(weight_matrix, 1., nargout=2)
        modularity.append(Q)
    plt.plot(modularity)
    plt.xlabel("Epoch")
    plt.ylabel("Modularity")
    plt.title("Modularity vs Epoch")
    plt.show()


def detect_small_world(weight_matrixs):
    """
    检测给定的加权图（由其邻接矩阵表示）是否具有小世界特性。

    该函数计算输入图的平均路径长度和聚类系数，并将其与等效随机图的值进行比较。
    如果平均路径长度大约是随机图的1.1倍或更少，并且聚类系数至少是随机图的1.5倍，则认为该图具有小世界特性。

    参数:
    - weight_matrixs (List[np.ndarray]): 包含邻接矩阵的列表，其中最后一个元素表示要分析的图的邻接矩阵。邻接矩阵应为NumPy数组，每个条目表示两个节点之间的边权重。

    注意:
    - 确保输入的`weight_matrixs`至少包含一个项目，并且最后一个项目是目标图的矩阵。
    - 该函数使用NetworkX库进行图分析，并与等密度的随机生成图进行比较。

    无返回值。直接将结果输出到控制台，指示网络是否具有小世界特性，并提供原始网络和随机网络的计算指标。
    """

    G = nx.from_numpy_array(weight_matrixs[-1])

    # 计算平均路径长度
    average_path_length = nx.average_shortest_path_length(G)
    print(f"平均路径长度: {average_path_length}")

    # 计算聚类系数
    clustering_coefficient = nx.average_clustering(G)
    print(f"聚类系数: {clustering_coefficient}")

    # 生成一个随机网络进行比较
    random_G = nx.erdos_renyi_graph(len(G), nx.density(G))

    # 计算随机网络的平均路径长度
    random_average_path_length = nx.average_shortest_path_length(random_G)
    print(f"随机网络平均路径长度: {random_average_path_length}")

    # 计算随机网络的聚类系数
    random_clustering_coefficient = nx.average_clustering(random_G)
    print(f"随机网络聚类系数: {random_clustering_coefficient}")

    # 判断是否具有小世界特性
    if (average_path_length <= random_average_path_length * 1.1 and
        clustering_coefficient >= random_clustering_coefficient * 1.5):
        print("该网络具有小世界特性。")
    else:
        print("该网络不具有小世界特性。")

def detect_strength_powerlaw(weigh_matrixs):
    """
    检测一系列加权矩阵中是否存在幂律分布。

    该函数评估提供的最后10个加权矩阵中的幂律行为强度。
    它们垂直聚合，处理零值以防止计算过程中出现无穷大，拟合幂律模型，并与指数分布进行比较，以确定数据是否显著符合幂律模式。

    参数
    ----------
    weigh_matrixs : list of numpy.ndarray
        包含2D numpy数组的列表，表示一系列加权矩阵。分析使用最后10个矩阵。

    异常
    ------
    ValueError
        如果 `weigh_matrixs` 包含少于10个矩阵。

    返回
    -------
    None
        该函数直接打印结果，指示数据是否符合幂律分布，并提供幂律参数（alpha和xmin），以及统计检验结果（R和p值）。

    注意
    ----
    - 输入矩阵应兼容垂直求和（即列大小匹配）。
    - NaN值作为 `np.nansum` 的一部分进行求和，但在后处理时检查并处理它们的存在。
    - p值小于0.05表明强烈反对数据符合指数分布的原假设，意味着支持幂律分布。
    - 将零值替换为1e-10是避免对数计算或幂律拟合时出现数学问题的常见做法。
    """
    # 将 r2r_weight 竖着加起来
    last_r2r_weight = weigh_matrixs[-10:]
    r2r_weight_sum = np.nansum(last_r2r_weight, axis=2)

    # print(np.any([np.isnan(r2r_weight) for r2r_weight in r2r_weight_sum.flatten()]))

    flatten_r2r_weight_sum = r2r_weight_sum.flatten()
    # 将所有的0值替换为1e-10，避免计算时出现无穷大
    flatten_r2r_weight_sum[flatten_r2r_weight_sum == 0] = 1e-10
    fit = powerlaw.Fit(r2r_weight_sum.flatten(), discrete=True)

    # 输出幂律分布的参数
    print(f"Alpha (幂律指数): {fit.power_law.alpha}")
    print(f"Xmin (幂律分布的最小值): {fit.power_law.xmin}")

    # 判断数据是否符合幂律分布
    R, p = fit.distribution_compare('power_law', 'exponential')
    print(f"幂律分布 vs 指数分布的似然比 (R): {R}")
    print(f"p 值: {p}")

    if p < 0.05:
        print("数据显著符合幂律分布。")
    else:
        print("数据不符合幂律分布。")