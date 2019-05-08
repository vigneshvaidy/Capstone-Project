#!/bin/bash

for i in {0..600..20}
  do
     echo "$i epoch"
     python Argument_Evaluation.py /home/xelese/CapstoneProject/predictions/predictions_bl-20190325-141754-$i.npy
 done