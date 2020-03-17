#!/bin/bash
set -e

# python prepare.py
cd detector
maxeps=100
f=3
CUDA_VISIBLE_DEVICES=1 python main.py --model 3in1 -b 8 --save-dir res18/retrft96$f/ --epochs $maxeps --config config_training$f 
echo "process 1 finished"

maxeps1=150
CUDA_VISIBLE_DEVICES=1 python main.py --model 3in1Dropout -b 8 --resume results/res18/retrft96$f/100.ckpt --save-dir res18/retrft96$f/ --epochs $maxeps1 --config config_training$f --start-epoch 101 
echo "process 2 finished"

for (( i=130; i<=$maxeps1; i+=1)) 
do
    echo "process $i epoch"
	
	if [ $i -lt 10 ]; then
	    CUDA_VISIBLE_DEVICES=0,1 python main.py --model 3in1Dropout -b 16 --resume results/res18/retrft96$f/00$i.ckpt --test 1 --save-dir res18/retrft96$f/ --config config_training$f
	elif [ $i -lt 100 ]; then 
	    CUDA_VISIBLE_DEVICES=0,1 python main.py --model 3in1Dropout -b 16 --resume results/res18/retrft96$f/0$i.ckpt --test 1 --save-dir res18/retrft96$f/ --config config_training$f
	elif [ $i -lt 1000 ]; then
	    CUDA_VISIBLE_DEVICES=1 python main.py --model 3in1Dropout -b 16 --resume results/res18/retrft96$f/$i.ckpt --test 1 --save-dir res18/retrft96$f/ --config config_training$f
	else
	    echo "Unhandled case"
    fi

    if [ ! -d "results/res18/retrft96$f/val$i/" ]; then
        mkdir results/res18/retrft96$f/val$i/
    fi
    mv results/res18/retrft96$f/bbox/*.npy results/res18/retrft96$f/val$i/
done 