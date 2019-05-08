import numpy as np

metadata = np.load("/home/xelese/CapstoneProject/predictions/predictions_bl-20190325-141754-600.npy")
preds = metadata['predictions']

print preds