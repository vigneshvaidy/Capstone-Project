#!/bin/bash

for i in {0..600..20}
  do
     echo "$i epoch"
     python Prediction.py /home/xelese/CapstoneProject/metadata/dump_bl-20190325-141754-$i.pkl
 done