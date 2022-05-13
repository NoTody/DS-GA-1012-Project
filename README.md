DS-GA-1012-Project
======================
NYU Natural Language Understanding Spring22 Project - Self-Supervised Regularization on Improving Robustness of Semantically Fluent Adversarial Texts

baseline_pl.py and baseline_main.py are files for baseline method.

cr_pl.py and cr_main.py are files for our method.

"pl" postfix means the file contains pytorch-lightning model implementation for the given method, "main" postfix means the file is used for training/testing for the given method.

All train/val/test data are in datasets folder. Augmented samples generation code is in augmentation_methods folder. Generate adversarial attack code in attack.py.

# To Install Dependencies
---------------------
```
pip install -r requirement.txt
```

# Run Example for Baseline with AGNews (1 node 1 gpu):

## To Train
---------------------
To train on different dataset, change trainset name on --dataset_name
```
python ./baseline_main.py --accumulate_grad_batches 1 --num_nodes 1 --num_devices 1 --model_name "bert-base-uncased" --dataset_name "agnews" --num_workers 10 --max_epochs 10 --batch_size 32 --max_seq_length 256 --mode "train" --lr 2e-5 --num_labels 4 --scheduler_name "cosine" --tb_save_dir "../"
```

## To Test
---------------------
To test on different dataset, change load path on --load_path and testset name on --dataset_name
```
python ./baseline_main.py --num_nodes 1 --num_devices 1 --model_name "bert-base-uncased" --dataset_name "agnews_eda" --num_workers 10 --batch_size 16 --max_seq_length 256 --mode "test" --load_path '../lightning_logs/agnews_bert_base_eda/checkpoints/epoch=3-step=6876.ckpt' --num_labels 4
```

# Run Example for Proposed Method (1 node 1 gpu):

## To Train
---------------------
To train on different dataset, change trainset name on --dataset_name_ori and augmented datasets on --dataset_name_str_adv and --dataset_name_weak_aug. To change number of layers, change integer on --top_k_layers. To use EMA, use --use_ema. To use projector, use --user_projector with MLP neuron number set by --mlp.
```
python ./cr_main.py --accumulate_grad_batches 1 --num_nodes 1 --gpus 1 --data_dir "../traindata/" --model_name "bert-base-uncased" --num_workers 10 --max_epochs 10 --batch_size 16 --max_seq_length 256 --lr_backbone 5e-5 --lr_projector 1e-3 --mode "train" --num_labels 4 --scheduler_name "cosine" --dataset_name_ori "agnews" --dataset_name_str_adv "agnews_ssmba" --dataset_name_weak_aug "agnews_synonym" --save_dir "./lightning_logs" --loss_func "l1_smooth" --top_k_layers 5 --use_ema --use_projector --mlp "2048-1024-768"
```

## To Test
---------------------
To test on different dataset, change laod path on --load_path and testset name on --testset.
```
python ./cr_main.py --accumulate_grad_batches 1 --num_nodes 1 --gpus 1 --data_dir "../traindata/" --model_name "bert-base-uncased" --num_workers 10 --max_epochs 10 --batch_size 16 --max_seq_length 256 --mode "test" --num_labels 4 --scheduler_name "cosine" --testset "agnews" --save_dir "./lightning_logs" --load_path "./lightning_logs/agnews_l1_smooth_5layers_ssmba_eda_projector2048-1024-768_ema_b16" --use_ema --use_projector --mlp "2048-1024-768" --top_k_layers 5
```
