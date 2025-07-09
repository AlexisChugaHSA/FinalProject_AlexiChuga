#  Copyright (c) 2025. Departamento de Ingenieria de Sistemas y Computacion.
import numpy as np
import pickle
import os

def main():
    # initial state
    initial_state2 = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1],
    ], dtype=np.uint8)
    #para glider y binker
    initial_state = np.array([
        [1, 1, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1, 0]
    ], dtype=np.uint8)

    # the data dir
    data_dir = os.path.join("..","..", "data")

    # path to pickle archive
    pickle_file = os.path.join(data_dir, "initial.pkl")

    # save initial state
    with open(pickle_file, "wb") as f:
        pickle.dump(initial_state, f)
    print("Saved initial state to pickle file")

if __name__ == "__main__":
    main()
