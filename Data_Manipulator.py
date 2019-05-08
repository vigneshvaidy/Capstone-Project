import numpy as np
import theano
import os.path
# import subprocess
import utils
# import protein_vector_loader

TRAIN_PATH = 'data/cullpdb+profile_6133_filtered.npy'
TEST_PATH = 'data/cb513+profile_split1.npy.gz'


# TRAIN DATA #
def get_train():
    if not os.path.isfile(TRAIN_PATH):
        print("No Training Data Available ...")
    else:
        print("Training Data is Available ...")
    print("Loading train data ...")
    x_in = utils.load_gz(TRAIN_PATH)
    x = np.reshape(x_in, (5534, 700, 57))
    del x_in
    x = x[:, :, :]
    labels = x[:, :, 22:30]
    mask = x[:, :, 30] * -1 + 1

    amino_acid_residues = np.arange(0, 21)
    sequence_profile = np.arange(35, 56)
    horizontal_stack = np.hstack((amino_acid_residues, sequence_profile))
    x = x[:, :, horizontal_stack]
    print ('x: ', x)



    # getting meta #
    num_seqs_row = np.size(x, 0)
    seqlen_column = np.size(x, 1)




    # REMAKING LABELS #
    x = x.astype(theano.config.floatX)
    mask = mask.astype(theano.config.floatX)




    # Dummy -> concat
    vals = np.arange(0, 8)
    labels_new = np.zeros((num_seqs_row, seqlen_column))
    for i in xrange(np.size(labels, axis=0)):
        labels_new[i, :] = np.dot(labels[i, :, :], vals)
    labels_new = labels_new.astype('int32')
    labels = labels_new
    print("labels: ", labels)



    print("Loading splits ...")
    # SPLITS #
    seq_names = np.arange(0, num_seqs_row)

    x_train = x[seq_names[0:5278]]
    x_valid = x[seq_names[5278:5534]]
    labels_train = labels[seq_names[0:5278]]
    labels_valid = labels[seq_names[5278:5534]]
    mask_train = mask[seq_names[0:5278]]
    mask_valid = mask[seq_names[5278:5534]]
    num_seq_train = np.size(x_train, 0)
    return x_train, x_valid, labels_train, labels_valid, mask_train, mask_valid, num_seq_train





# TEST DATA #
def get_test():
    if not os.path.isfile(TEST_PATH):
        print("Test Data Unavailable")
    print("Loading test data ...")
    x_test_in = utils.load_gz(TEST_PATH)
    x_test = np.reshape(x_test_in, (514, 700, 57))
    del x_test_in
    x_test = x_test[:, :, :].astype(theano.config.floatX)
    labels_test = x_test[:, :, 22:30].astype('int32')
    mask_test = x_test[:, :, 30].astype(theano.config.floatX) * -1 + 1




    a = np.arange(0, 21)
    b = np.arange(35, 56)
    c = np.hstack((a, b))
    x_test = x_test[:, :, c]




    # getting meta
    seqlen = np.size(x_test, 1)
    d = np.size(x_test, 2)
    num_classes = 8
    num_seq_test = np.size(x_test, 0)
    del a, b, c




    # DUMMY -> CONCAT #
    vals = np.arange(0, 8)
    labels_new = np.zeros((num_seq_test, seqlen))
    for i in xrange(np.size(labels_test, axis=0)):
        labels_new[i, :] = np.dot(labels_test[i, :, :], vals)
    labels_new = labels_new.astype('int32')
    labels_test = labels_new




    # ADDING BATCH PADDING #
    x_add = np.zeros((126, seqlen, d))
    label_add = np.zeros((126, seqlen))
    mask_add = np.zeros((126, seqlen))




    x_test = np.concatenate((x_test, x_add), axis=0).astype(theano.config.floatX)
    labels_test = np.concatenate((labels_test, label_add), axis=0).astype('int32')
    mask_test = np.concatenate((mask_test, mask_add), axis=0).astype(theano.config.floatX)
    return x_test, mask_test, labels_test, num_seq_test
