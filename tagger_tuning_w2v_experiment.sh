#!/bin/bash
for lstm_size in 50 100 200
do
  for crf_type in none simple linear simple_bilinear
  do
    for dropout in 0.25 0.5
    do
      for learning_rate in .01 .001
      do
        for w2v_fname in glove.6B.50d.txt glove.6B.100d.txt glove.6B.200d.txt
        do
          (( i++ ))
          echo "========================================================="
          echo "Experiment" $i ":"\
          python -u ace_segmentation.py -n 1000 -b 400 \
          --lstm_size $lstm_size\
          --crf_type $crf_type\
          --dropout $dropout\
          --learning_rate $learning_rate\
          --w2v_fname data/word_vectors/$w2v_fname
          echo "========================================================="

          # run it
          python -u ace_segmentation.py -n 1000 -b 400 \
          --lstm_size $lstm_size\
          --crf_type $crf_type\
          --dropout $dropout\
          --learning_rate $learning_rate\
          --w2v_fname data/word_vectors/$w2v_fname
        done
      done
    done
  done
done
