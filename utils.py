import brainstate as bst
import brainunit as u
import seaborn as sns
from matplotlib import pyplot as plt


def current_generate(batch_size, num_steps, stimulate, delay, common_volt, go_cue_volt):
    current = u.math.zeros((num_steps, batch_size, 1)) * u.mA
    current[:stimulate, :, :] = common_volt
    current[stimulate + delay:
            stimulate + delay + stimulate,
    :, :] = go_cue_volt

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
