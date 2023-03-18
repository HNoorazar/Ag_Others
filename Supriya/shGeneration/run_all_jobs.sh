#!/bin/bash

cd /home/hnoorazar/supriya/qsubs/
for runname in {1..3000}
do
qsub ./q_$runname.sh
done
