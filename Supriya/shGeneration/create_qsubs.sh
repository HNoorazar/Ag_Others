#!/bin/bash
cd /home/hnoorazar/supriya/

row_number=1

while [ $row_number -le 3000 ]
do
  cp template.sh ./qsubs/q_$row_number.sh
  sed -i s/row_number/"$row_number"/g    ./qsubs/q_$row_number.sh
  sed -i s/row_number/"$row_number"/g  ./qsubs/q_$row_number.sh
  let "row_number+=1" 
done
