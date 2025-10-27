import h5py
import numpy as np
from helpers import *
import time

start = time.time()

# Load your data from CSVs (only once)
x_train, x_test, y_train, train_ids, test_ids = load_csv_data("data/dataset", sub_sample=False)

# Save to HDF5 file
with h5py.File("data/dataset/cached_data.h5", "w") as f:
    f.create_dataset("x_train", data=x_train)
    f.create_dataset("x_test", data=x_test)
    f.create_dataset("y_train", data=y_train)
    f.create_dataset("train_ids", data=train_ids)
    f.create_dataset("test_ids", data=test_ids)

end = time.time()
print(f"Saved to cached_data.h5 (runtime: {end - start:.2f} s)")