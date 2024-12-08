from matplotlib import pyplot as plt
import seaborn as sns

def plot_data(x_data):
    for data_id in range(5):
        plt.clf()
        plt.imshow(x_data.swapaxes(0, 1)[data_id].transpose(), cmap=plt.cm.gray_r, aspect="auto")
        plt.xlabel("Time (ms)")
        plt.ylabel("Unit")
        sns.despine()

        plt.show()