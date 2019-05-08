import sys
import numpy as np
import importlib
import lasagne as lasagne
import theano
from theano import tensor as T
import os
import glob

import Data_Manipulator

if not (2 <= len(sys.argv) <= 3):
    sys.exit("Usage: Python prediction.py <metadata_path> [subset=test]")

sym_y = T.imatrix('target_output')
sym_x = T.tensor3()

metadata_path_all = glob.glob(sys.argv[1] + "*")


# print(len(metadata_path_all))

if len(sys.argv) >= 3:
    subset = sys.argv[2]
    assert subset in ['train', 'valid', 'test', 'train_valid']
else:
    subset = 'test'

if subset == "test":
    x_test, mask, _, num_seq = Data_Manipulator.get_test()
else:
    sys.exit("valid not implemented")

for metadata_path in metadata_path_all:

    print "Loading metadata file %s" % metadata_path

    metadata = np.load(metadata_path)

    config_name = metadata['config_name']

    config = importlib.import_module("configurations.%s" % config_name)

    print "Using configurations: '%s'" % config_name

    print "Build model"

    l_in, l_out = config.build_model()

    print "Build eval function"

    inference = lasagne.layers.get_output(l_out, sym_x, deterministic=True)

    print "Load parameters"

    lasagne.layers.set_all_param_values(l_out, metadata['param_values']) #feck cb513 and cullpdb paramenters

    print "Compile functions"

    predict = theano.function([sym_x], inference)

    print "Predict"

    predictions = []
    batch_size = config.batch_size
    num_batches = np.size(x_test, axis=0) // batch_size

    for i in range(num_batches):
        idx = range(i * batch_size, (i + 1) * batch_size)
        x_batch = x_test[idx]
        mask_batch = mask[idx]
        p = predict(x_batch)
        predictions.append(p)

    predictions = np.concatenate(predictions, axis=0)
    predictions_path = os.path.join("predictions", os.path.basename(metadata_path).replace("dump_", "predictions_").replace(".pkl", ".npy"))

    #print predictions
    #print len(predictions)
    #print (np.shape(predictions))
    #print "Storing predictions in %s" % predictions_path
    np.save(predictions_path, predictions)
