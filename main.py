import brainstate as bst
import brainunit as u
import jax.numpy as jnp
import numpy as np
import braintools as bts

# 设置随机种子确保可重复性
bst.random.seed(43)

from model import SNN_ext
from utils import (data_generate_1221, current_generate,
                   cal_model_accuracy, plot_accuracy, plot_loss,
                   plot_gamfit_alpha_beta, plot_q_coreness,
                   plot_spike_count)
from loss import communicability_loss

# 网络结构参数
num_inputs = 20  # 输入层神经元数量
num_hidden = 100  # 隐藏层神经元数量
num_outputs = 2  # 输出层神经元数量

# 时间参数设置
time_step = 1 * u.ms
bst.environ.set(dt=time_step)

# 定义各阶段时长
stimulate = int((500 * u.ms).to_decimal(time_step.unit))  # 刺激阶段
delay = int((1000 * u.ms).to_decimal(time_step.unit))  # 延迟阶段
response = int((1000 * u.ms).to_decimal(time_step.unit))  # 响应阶段

num_steps = stimulate + delay + response
freq = 300 * u.Hz  # 输入频率

# 训练参数
batch_size = 40
epoch = 400

# 初始化网络
net = SNN_ext(num_inputs, num_hidden, num_outputs, batch_size=batch_size)

# 生成训练数据
x_data, y_data = data_generate_1221(batch_size, num_steps, net, stimulate, delay, freq)
common_current = 3.0 * u.mA  # 基础电流
go_cue_current = 5.0 * u.mA  # 触发电流
current = current_generate(batch_size, num_steps, stimulate, delay, common_current, go_cue_current)

# 初始化优化器
optimizer = bst.optim.Adam(lr=1e-3, beta1=0.9, beta2=0.999)
optimizer.register_trainable_weights(net.states(bst.ParamState))


def loss_fn():
    """
    定义损失函数
    返回：组合损失值（交叉熵损失 + 通信损失 + 活动损失）
    """
    # 前向传播
    predictions, r_V = bst.compile.for_loop(net.update, x_data, current)
    weight_matrix = net.get_weight_matrix()

    # 只考虑延迟期后的预测
    predictions = predictions[stimulate + delay:]
    r_V = jnp.abs(r_V[:stimulate + delay])
    predictions = u.math.mean(predictions, axis=0)

    # 计算各项损失
    ce = bts.metric.softmax_cross_entropy_with_integer_labels(predictions, y_data).mean()
    communicability = communicability_loss(weight_matrix, comms_factor=1)
    activity = (r_V.mean(axis=(0, 1)) * net.r2r_conn * weight_matrix).sum() / net.r2r_conn.sum()

    # 返回加权组合损失
    return 0.025 * ce + 2. * communicability + 2. * activity


@bst.compile.jit
def train_fn():
    """
    训练函数
    返回：损失值、准确率、权重矩阵、脉冲计数和模型预测
    """
    # 初始化网络状态
    bst.nn.init_all_states(net, batch_size=batch_size)
    net.start_spike_count()

    # 计算梯度和损失
    grads, l = bst.augment.grad(loss_fn, net.states(bst.ParamState), return_value=True)()

    # 计算准确率
    acc, am = cal_model_accuracy(x_data, y_data, net, current, stimulate, delay)

    # 更新参数
    optimizer.update(grads)
    # 确保权重非负
    net.set_weight_matrix(jnp.clip(net.get_weight_matrix(), 0, None))

    return l, acc, net.get_weight_matrix(), net.get_spike_counts(), am


if __name__ == "__main__":
    # 训练循环
    train_losses = []
    accuracies = []
    weight_matrixs = []
    spike_counts = []
    model_predicts = []

    for i in range(1, epoch + 1):
        loss, accuracy, weight_matrix, spike_count, model_predict = train_fn()

        # 记录训练数据
        train_losses.append(loss)
        accuracies.append(accuracy)
        weight_matrixs.append(np.asarray(weight_matrix))
        spike_counts.append(np.swapaxes(np.asarray(spike_count), 0, 1))
        model_predicts.append(model_predict)

        print(f"Epoch {i}, Loss: {loss}, Accuracy: {accuracy}, Activity: {spike_count.sum()}")

    # 绘制训练结果
    plot_accuracy(accuracies)
    plot_loss(train_losses)

    r2r_conn = np.asarray(net.r2r_conn)

    # 分析网络特性
    plot_gamfit_alpha_beta(weight_matrixs, r2r_conn)
    C_list, q_list = plot_q_coreness(weight_matrixs)
    spike_counts = np.asarray(spike_counts)
    C = np.asarray(C_list).reshape(epoch, num_hidden)
    plot_spike_count(spike_counts, C, np.asarray(model_predicts))

    # 保存结果
    np.savez(
        "data.npz",
        r2r_conn=r2r_conn,
        r2r_weights=weight_matrixs,
        spike_counts=spike_counts,
        y_data=y_data,
        model_predicts=model_predicts,
        C=C,
    )
