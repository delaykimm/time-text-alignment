#!/bin/bash
export PYTHONPATH=/path/to/project_root:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

data_path="ETTh1"
seq_len=96
batch_size=16
stride=8

# pred_len 96 - GPU 0 사용
# pred_len=96
# learning_rate=1e-4
# channel=32   #64
# e_layer=1
# d_layer=1    #2
# dropout_n=0.2   #0.7
# align_weight=1
# align_weight_time=1

# log_path="./Results/${data_path}/"
# mkdir -p $log_path
# align_weight_str=$(echo $align_weight | tr '.' 'p')
# align_weight_time_str=$(echo $align_weight_time | tr '.' 'p')
# log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}_aw${align_weight_str}_awt${align_weight_time_str}.log"
# CUDA_VISIBLE_DEVICES=2 nohup python train.py \
#   --data_path $data_path \
#   --batch_size $batch_size \
#   --num_nodes 7 \
#   --seq_len $seq_len \
#   --pred_len 96 \
#   --epochs 70 \
#   --seed 42 \
#   --channel $channel \
#   --learning_rate $learning_rate \
#   --dropout_n $dropout_n \
#   --e_layer $e_layer \
#   --d_layer $d_layer \
#   --align_weight $align_weight \
#   --align_weight_time $align_weight_time \
#   --stride $stride > $log_file &

# align_weight=0
# align_weight_time=1

# log_path="./Results/${data_path}/"
# mkdir -p $log_path
# align_weight_str=$(echo $align_weight | tr '.' 'p')
# align_weight_time_str=$(echo $align_weight_time | tr '.' 'p')
# log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}_aw${align_weight_str}_awt${align_weight_time_str}.log"
# CUDA_VISIBLE_DEVICES=2 nohup python train.py \
#   --data_path $data_path \
#   --batch_size $batch_size \
#   --num_nodes 7 \
#   --seq_len $seq_len \
#   --pred_len 96 \
#   --epochs 70 \
#   --seed 42 \
#   --channel $channel \
#   --learning_rate $learning_rate \
#   --dropout_n $dropout_n \
#   --e_layer $e_layer \
#   --d_layer $d_layer \
#   --align_weight $align_weight \
#   --align_weight_time $align_weight_time \
#   --stride $stride > $log_file &

# align_weight=1
# align_weight_time=0

# log_path="./Results/${data_path}/"
# mkdir -p $log_path
# align_weight_str=$(echo $align_weight | tr '.' 'p')
# align_weight_time_str=$(echo $align_weight_time | tr '.' 'p')
# log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}_aw${align_weight_str}_awt${align_weight_time_str}.log"
# CUDA_VISIBLE_DEVICES=0 nohup python train.py \
#   --data_path $data_path \
#   --batch_size $batch_size \
#   --num_nodes 7 \
#   --seq_len $seq_len \
#   --pred_len 96 \
#   --epochs 70 \
#   --seed 42 \
#   --channel $channel \
#   --learning_rate $learning_rate \
#   --dropout_n $dropout_n \
#   --e_layer $e_layer \
#   --d_layer $d_layer \
#   --align_weight $align_weight \
#   --align_weight_time $align_weight_time \
#   --stride $stride > $log_file &

# align_weight=0
# align_weight_time=0

# log_path="./Results/${data_path}/"
# mkdir -p $log_path
# align_weight_str=$(echo $align_weight | tr '.' 'p')
# align_weight_time_str=$(echo $align_weight_time | tr '.' 'p')
# log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}_aw${align_weight_str}_awt${align_weight_time_str}.log"
# CUDA_VISIBLE_DEVICES=0 nohup python train.py \
#   --data_path $data_path \
#   --batch_size $batch_size \
#   --num_nodes 7 \
#   --seq_len $seq_len \
#   --pred_len 96 \
#   --epochs 70 \
#   --seed 42 \
#   --channel $channel \
#   --learning_rate $learning_rate \
#   --dropout_n $dropout_n \
#   --e_layer $e_layer \
#   --d_layer $d_layer \
#   --align_weight $align_weight \
#   --align_weight_time $align_weight_time \
#   --stride $stride > $log_file &

# pred_len = 192
pred_len=192
learning_rate=1e-4
channel=32    #64
e_layer=1
d_layer=2
dropout_n=0.2    #0.7

align_weight=1
align_weight_time=1

log_path="./Results/${data_path}/"
mkdir -p $log_path
align_weight_str=$(echo $align_weight | tr '.' 'p')
align_weight_time_str=$(echo $align_weight_time | tr '.' 'p')
log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}_aw${align_weight_str}_awt${align_weight_time_str}.log"
CUDA_VISIBLE_DEVICES=3 nohup python train.py \
  --data_path $data_path \
  --batch_size $batch_size \
  --num_nodes 7 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --epochs 70 \
  --seed 42 \
  --channel $channel \
  --learning_rate $learning_rate \
  --dropout_n $dropout_n \
  --e_layer $e_layer \
  --d_layer $d_layer \
  --align_weight $align_weight \
  --align_weight_time $align_weight_time \
  --stride $stride > $log_file &

align_weight=0
align_weight_time=1

log_path="./Results/${data_path}/"
mkdir -p $log_path
align_weight_str=$(echo $align_weight | tr '.' 'p')
align_weight_time_str=$(echo $align_weight_time | tr '.' 'p')
log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}_aw${align_weight_str}_awt${align_weight_time_str}.log"
CUDA_VISIBLE_DEVICES=3 nohup python train.py \
  --data_path $data_path \
  --batch_size $batch_size \
  --num_nodes 7 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --epochs 70 \
  --seed 42 \
  --channel $channel \
  --learning_rate $learning_rate \
  --dropout_n $dropout_n \
  --e_layer $e_layer \
  --d_layer $d_layer \
  --align_weight $align_weight \
  --align_weight_time $align_weight_time \
  --stride $stride > $log_file &

align_weight=1
align_weight_time=0

log_path="./Results/${data_path}/"
mkdir -p $log_path
align_weight_str=$(echo $align_weight | tr '.' 'p')
align_weight_time_str=$(echo $align_weight_time | tr '.' 'p')
log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}_aw${align_weight_str}_awt${align_weight_time_str}.log"
CUDA_VISIBLE_DEVICES=3 nohup python train.py \
  --data_path $data_path \
  --batch_size $batch_size \
  --num_nodes 7 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --epochs 70 \
  --seed 42 \
  --channel $channel \
  --learning_rate $learning_rate \
  --dropout_n $dropout_n \
  --e_layer $e_layer \
  --d_layer $d_layer \
  --align_weight $align_weight \
  --align_weight_time $align_weight_time \
  --stride $stride > $log_file &

align_weight=0
align_weight_time=0

log_path="./Results/${data_path}/"
mkdir -p $log_path
align_weight_str=$(echo $align_weight | tr '.' 'p')
align_weight_time_str=$(echo $align_weight_time | tr '.' 'p')
log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}_aw${align_weight_str}_awt${align_weight_time_str}.log"
CUDA_VISIBLE_DEVICES=3 nohup python train.py \
  --data_path $data_path \
  --batch_size $batch_size \
  --num_nodes 7 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --epochs 70 \
  --seed 42 \
  --channel $channel \
  --learning_rate $learning_rate \
  --dropout_n $dropout_n \
  --e_layer $e_layer \
  --d_layer $d_layer \
  --align_weight $align_weight \
  --align_weight_time $align_weight_time \
  --stride $stride > $log_file &

# # pred_len 336 - GPU 3 사용
# pred_len=336
# learning_rate=1e-4
# channel=32    #64
# dropout_n=0.2    #0.7
# e_layer=1
# d_layer=2
# align_weight=0.1
# align_weight_time=0.3

# log_path="./Results/${data_path}/"
# mkdir -p $log_path
# align_weight_str=$(echo $align_weight | tr '.' 'p')
# align_weight_time_str=$(echo $align_weight_time | tr '.' 'p')
# log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}_aw${align_weight_str}_awt${align_weight_time_str}.log"
# CUDA_VISIBLE_DEVICES=1 nohup python train.py \
#   --data_path $data_path \
#   --batch_size $batch_size \
#   --num_nodes 7 \
#   --seq_len $seq_len \
#   --pred_len $pred_len \
#   --epochs 70 \
#   --seed 42 \
#   --channel $channel \
#   --learning_rate $learning_rate \
#   --dropout_n $dropout_n \
#   --e_layer $e_layer \
#   --d_layer $d_layer \
#   --align_weight $align_weight \
#   --align_weight_time $align_weight_time \
#   --stride $stride > $log_file &

# # pred_len 720 - GPU 0 사용 (pred_len=96과 공유)
# pred_len=720
# learning_rate=1e-4
# channel=32
# dropout_n=0.2    #0.8
# e_layer=2
# d_layer=2
# align_weight=0.1
# align_weight_time=0.3

# log_path="./Results/${data_path}/"
# mkdir -p $log_path
# align_weight_str=$(echo $align_weight | tr '.' 'p')
# align_weight_time_str=$(echo $align_weight_time | tr '.' 'p')
# log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}_aw${align_weight_str}_awt${align_weight_time_str}.log"
# CUDA_VISIBLE_DEVICES=1 nohup python train.py \
#   --data_path $data_path \
#   --batch_size $batch_size \
#   --num_nodes 7 \
#   --seq_len $seq_len \
#   --pred_len $pred_len \
#   --epochs 70 \
#   --seed 42 \
#   --channel $channel \
#   --head 8 \
#   --learning_rate $learning_rate \
#   --dropout_n $dropout_n \
#   --e_layer $e_layer \
#   --d_layer $d_layer \
#   --align_weight $align_weight \
#   --align_weight_time $align_weight_time \
#   --stride $stride > $log_file &