import numpy as np

def save_input(x_data, y_data, filename="inputs.npz"):
    np.savez(filename, x_data=x_data, y_data=y_data)

def save_train_states(states_dict, filename="states.npz"):
    np.savez(filename, **states_dict)

if __name__ == "__main__":
    inputs = np.load("inputs.npz")
    states = np.load("states.npz")

    for key in inputs.keys():
        print(key, inputs[key].shape)

    for key in states.keys():
        print(key, states[key].shape)