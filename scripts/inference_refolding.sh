#!/bin/bash

# assumes you have downloaded the flips weights by:
# wget --content-disposition https://keeper.mpdl.mpg.de/f/236758d218a748fe93cd/?dl=1 -P ckpt/paper_weights
# and that you have installed proteinMPNN and ESMFold as described in the README

now=$(date +"%Y-%m-%d_%H-%M-%S-%3N")

# Generate 30 samples conditioned on an exemplary flexibility profile for lengths 100 and 150.
# Then screen the top 10 samples for each length using BackFlip and re-fold these top samples using proteinMPNN and ESMFold
python experiments/inference.py \
 --config-path ../configs \
 --config-name inference_flex \
 inference.ckpt_path="./ckpt/paper_weights/flips_final.ckpt" \
 inference.name=${now}_test_inference_refolding \
 inference.run_self_consistency=True \
 inference.flexibility.backflip_screening=True \
 inference.flexibility.min_length=100 \
 inference.flexibility.max_length=150 \
 inference.flexibility.length_step=50 \
 inference.flexibility.num_samples=30 \
 inference.flexibility.num_top_samples=10 \
 inference.pmpnn_dir=./../ProteinMPNN/ \