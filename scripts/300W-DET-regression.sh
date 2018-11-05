# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
CUDA_VISIBLE_DEVICES=2,0 python36 ./exps/basic_main.py \
	--train_lists ./cache_data/lists/300W/300w.train.DET \
	--eval_ilists ./cache_data/lists/300W/300w.test.common.DET \
	              ./cache_data/lists/300W/300w.test.challenge.DET \
	              ./cache_data/lists/300W/300w.test.full.DET \
	--num_pts 68 \
	--model_config ./configs/MobileNetV2Detector.config \
	--opt_config ./configs/SGD-regression.config \
	--save_path ./snapshots/300W-MobileNetV2-DET \
  --crop_height 60 --crop_width 60 \
	--pre_crop_expand 0.2 --sigma 4 --batch_size 128 \
	--crop_perturb_max 8 --rotate_max 20 \
	--scale_prob 1.0 --scale_min 0.9 --scale_max 1.1 --scale_eval 1 \
	--heatmap_type gaussian \
  --regression
