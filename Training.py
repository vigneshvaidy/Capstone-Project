import numpy as np
import theano
import theano.tensor as T
import lasagne as las
import string
import sys
from datetime import datetime, timedelta
import importlib
import time
import cPickle as pickle
import utils


np.random.seed(1)  # keeping random values same at every iteration

if len(sys.argv) != 2:
    sys.exit("Usage: python Training.py <config_name>")

config_name = sys.argv[1]

config = importlib.import_module("configurations.%s" % config_name)
optimizer = config.optimizer
print "Using configurations: '%s'" % config_name

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
experiment_id = "%s-%s" % (config_name, timestamp)
metadata_path = "metadata/dump_%s" % experiment_id

print "Experiment id: %s" % experiment_id
num_classes = 8


def main():
    def training(num_batches, batch_size, x_train, label_train, mask_train):
        for i in range(num_batches):
            idx = range(i * batch_size, (i + 1) * batch_size)
            x_batch = x_train[idx]
            y_batch = label_train[idx]
            mask_batch = mask_train[idx]
            loss, out, batch_norm = train(x_batch, y_batch, mask_batch)
            norms.append(batch_norm)
            preds.append(out)
            losses.append(loss)

        predictions = np.concatenate(preds, axis=0)
        loss_train = np.mean(losses)
        all_losses_train.append(loss_train)
        acc_train = utils.proteins_acc(predictions, label_train[0:num_batches * batch_size],
                                       mask_train[0:num_batches * batch_size])

        print('acc_train: ', acc_train)
        all_accuracy_train.append(acc_train)
        mean_norm = np.mean(norms)
        all_mean_norm.append(mean_norm)
        print "  average training loss: %.5f" % loss_train
        print "  average training accuracy: %.5f" % acc_train
        print "  average norm: %.5f" % mean_norm

    def testing(num_batches, batch_size, X, y, mask):
        for i in range(num_batches):
            idx = range(i * batch_size, (i + 1) * batch_size)
            x_batch = X[idx]
            y_batch = y[idx]
            mask_batch = mask[idx]
            loss, out = evaluate(x_batch, y_batch, mask_batch)
            preds.append(out)
            losses.append(loss)
        predictions = np.concatenate(preds, axis=0)
        loss_eval = np.mean(losses)
        all_losses.append(loss_eval)

        acc_eval = utils.proteins_acc(predictions, y, mask)
        all_accuracy.append(acc_eval)

        print("Average evaluation loss ({}): {:.5f}".format(subset, loss_eval))
        print("Average evaluation accuracy ({}): {:.5f}".format(subset, acc_eval))
        return i

    global momentum_schedule, momentum, i
    sym_y = T.imatrix('target_output')
    sym_mask = T.matrix('mask')
    sym_x = T.tensor3()

    tol = 1e-5
    num_epochs = config.epochs
    batch_size = config.batch_size

    print("Building network ...")
    # DEBUG #
    l_in, l_out = config.build_model()
    # DEBUG #
    all_layers = las.layers.get_all_layers(l_out)
    num_params = las.layers.count_params(l_out)
    print("  number of parameters: %d" % num_params)
    print("  layer output shapes:")
    # output for debugging (names and dimensions) # InputLayer(None, 700, 42)
    for layer in all_layers:
        name = string.ljust(layer.__class__.__name__, 32)
        print("    %s %s" % (name, las.layers.get_output_shape(layer)))

    print("Creating cost function")
    # lasagne.layers.get_output produces a variable for the output of the net
    out_train = las.layers.get_output(l_out, sym_x, deterministic=False)
    print('out_train: ', out_train)

    print("Creating eval function")
    out_eval = las.layers.get_output(l_out, sym_x, deterministic=True)
    probs_flat = out_train.reshape((-1, num_classes))
    print("probs_flat: ", probs_flat)
    lambda_reg = config.lambda_reg
    params = las.layers.get_all_params(l_out, regularizable=True)

    reg_term = sum(T.sum(p ** 2) for p in params)
    cost = T.nnet.categorical_crossentropy(T.clip(probs_flat, tol, 1 - tol), sym_y.flatten())
    print('cost: ', cost)
    cost = T.sum(cost * sym_mask.flatten()) / T.sum(sym_mask) + lambda_reg * reg_term
    print('cost_2: ', cost)

    # Retrieve all parameters from the network
    all_params = las.layers.get_all_params(l_out, trainable=True)

    # Compute SGD updates for training
    print("Computing updates ...")
    if hasattr(config, 'learning_rate_schedule'):
        learning_rate_schedule = config.learning_rate_schedule  # Import learning rate schedule
    # else:
    #     learning_rate_schedule = {0: config.learning_rate}
    learning_rate = theano.shared(np.float32(learning_rate_schedule[0]))
    all_grads = T.grad(cost, all_params)
    cut_norm = config.cut_grad

    updates, norm_calc = las.updates.total_norm_constraint(all_grads, max_norm=cut_norm, return_norm=True)

    if optimizer == "rmsprop":
        updates = las.updates.rmsprop(updates, all_params, learning_rate)
    else:
        sys.exit("please choose <rmsprop> in configfile")

    # Theano functions for training and computing cost
    print "config.batch_size %d" % batch_size
    print "data.num_classes %d" % num_classes
    if hasattr(config, 'build_model'):
        print("has build model")
    print("Compiling train ...")

    # Use this for training (see deterministic = False above)
    train = theano.function([sym_x, sym_y, sym_mask], [cost, out_train, norm_calc], updates=updates)

    print("Compiling eval ...")
    # use this for eval (deterministic = True + no updates)
    evaluate = theano.function([sym_x, sym_y, sym_mask], [cost, out_eval])

    # Start timers
    start_time = time.time()
    prev_time = start_time
    all_losses_train = []
    all_accuracy_train = []
    all_losses_eval_train = []
    all_losses_eval_valid = []
    all_losses_eval_test = []
    all_accuracy_eval_train = []
    all_accuracy_eval_valid = []
    all_accuracy_eval_test = []
    all_mean_norm = []

    import Data_Manipulator
    x_train, x_valid, label_train, label_valid, mask_train, mask_valid, num_seq_train = Data_Manipulator.get_train()
    # print("y shape")
    # print(label_valid.shape)
    # print("x_test shape")
    # print(x_valid.shape)

    # Start training
    for epoch in range(num_epochs):
        if (epoch % 10) == 0:
            print "Epoch %d of %d" % (epoch + 1, num_epochs)
        if epoch in learning_rate_schedule:
            lr = np.float32(learning_rate_schedule[epoch])
            print "  setting learning rate to %.7f" % lr
            learning_rate.set_value(lr)

        # print "Shuffling data"
        seq_names = np.arange(0, num_seq_train)
        np.random.shuffle(seq_names)
        x_train = x_train[seq_names]
        label_train = label_train[seq_names]
        mask_train = mask_train[seq_names]
        num_batches = num_seq_train // batch_size  # integer division
        losses = []
        preds = []
        norms = []

        training(num_batches, batch_size, x_train, label_train, mask_train)

        sets = [('valid', x_valid, label_valid, mask_valid, all_losses_eval_valid, all_accuracy_eval_valid)]
        for subset, X, y, mask, all_losses, all_accuracy in sets:
            print "  validating: %s loss" % subset
            preds = []
            num_batches = np.size(X, axis=0) // config.batch_size
            testing(num_batches, batch_size, X, y, mask)

        now = time.time()
        time_since_start = now - start_time
        time_since_prev = now - prev_time
        prev_time = now
        est_time_left = time_since_start * num_epochs
        eta = datetime.now() + timedelta(seconds=est_time_left)
        eta_str = eta.strftime("%c")
        print "  %s since start (%.2f s)" % (utils.hms(time_since_start), time_since_prev)
        print "  estimated %s to go (ETA: %s)" % (utils.hms(est_time_left), eta_str)

        if (epoch >= config.start_saving_at) and ((epoch % config.save_every) == 0):
            print "  saving parameters and metadata"
            with open((metadata_path + "-%d" % epoch + ".pkl"), 'w') as f:
                pickle.dump({
                    'config_name': config_name,
                    'param_values': las.layers.get_all_param_values(l_out),
                    'losses_train': all_losses_train,
                    'accuracy_train': all_accuracy_train,
                    'losses_eval_train': all_losses_eval_train,
                    'losses_eval_valid': all_losses_eval_valid,
                    'losses_eval_test': all_losses_eval_test,
                    'accuracy_eval_valid': all_accuracy_eval_valid,
                    'accuracy_eval_train': all_accuracy_eval_train,
                    'accuracy_eval_test': all_accuracy_eval_test,
                    'mean_norm': all_mean_norm,
                    'time_since_start': time_since_start
                }, f, pickle.HIGHEST_PROTOCOL)

            print "  stored in %s" % metadata_path


if __name__ == '__main__':
    main()
