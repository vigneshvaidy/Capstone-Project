import sys
import numpy as np
import glob
import utils

if len(sys.argv) < 2:
    sys.exit("Usage: python eval_avrg.py <predictions_path> [subset=test]")

predictions_path_all = glob.glob(sys.argv[1] + "*")



mybool = False
for predictions_path in predictions_path_all:
    print(predictions_path)
    if not mybool:
        predictions = np.load(predictions_path)
        mybool = True
    else:
        predictions = predictions + np.load(predictions_path)





import Data_Manipulator

if len(sys.argv) == 3:
    subset = sys.argv[2]
    assert subset in ['train', 'valid', 'test', 'test_valid']
else:
    subset = 'test'




if subset == "test":
    _, mask, y, _ = Data_Manipulator.get_test()





acc = utils.proteins_acc(predictions, y, mask)

print "Accuracy (%s) is: %.5f" % (subset, acc)

