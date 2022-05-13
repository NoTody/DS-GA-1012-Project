DS-GA-1012-Project
======================
NYU Natural Language Understanding Spring22 Project - On Defending Strong Adversarial Text Attacks with Semi-Supervised Adversarial Learning

baseline_pl.py and baseline_main.py are files for baseline method.

cr_pl.py and cr_main.py are files for our method.

"pl" postfix means the file contains pytorch-lightning model implementation for the given method, "main" postfix means the file is used for training/testing for the given method.

All train/val/test data are in datasets folder. Adversarial and augmented samples generation code is in augmentation_methods folder

# Run Example for Baseline with AGNews (1 node 1 gpu):

## train
---------------------
```
python ./baseline_main.py --accumulate_grad_batches 1 --num_nodes 1 --num_devices 1 --model_name "bert-base-uncased" --dataset_name "agnews" --num_workers 10 --max_epochs 10 --batch_size 32 --max_seq_length 256 --mode "train" --lr 2e-5 --num_labels 4 --scheduler_name "cosine" --tb_save_dir "../"
```
## To Install Dependencies
---------------------
```
pip install -r requirement.txt
```

## test
---------------------
```
python ./baseline_main.py --num_nodes 1 --num_devices 1 --model_name "bert-base-uncased" --dataset_name "augmented_agnews_eda" --num_workers 10 --batch_size 16 --max_seq_length 256 --mode "test" --load_path '../lightning_logs/agnews_bert_base_eda/checkpoints/epoch=3-step=6876.ckpt' --num_labels 4
```

# Run Example for Proposed Method (1 node 1 gpu):

## train
---------------------
```
python ./cr_main.py --accumulate_grad_batches 1 --num_nodes 1 --gpus 1 --model_name "bert-base-uncased" --num_workers 10 --max_epochs 20 --batch_size 16 --max_seq_length 256 --lr_backbone 5e-5 --lr_projector 1e-3 --mode "train" --lr 2e-5 --num_labels 4 --scheduler_name "cosine" --dataset_name_ori "agnews" --dataset_name_str_adv "agnews_ssmba" --dataset_name_weak_aug "agnews_synonym" --tb_save_dir "./" --loss_func "l1_smooth" --top_k_layers 5 --use_ema --use_projector --mlp "2048-1024-768"
```

## test
---------------------
```
python ./cr_main.py --accumulate_grad_batches 1 --num_nodes 1 --gpus 1 --model_name "bert-base-uncased" --num_workers 10 --max_epochs 10 --batch_size 16 --max_seq_length 256 --mode "test" --num_labels 4 --scheduler_name "cosine" --testset "agnews" --tb_save_dir "./" --load_path "./lightning_logs/agnews_l1_smooth_5layers_ssmba_eda_projector2048-1024-768_ema_b1" --use_ema --use_projector --mlp "2048-1024-768" --top_k_layers 5
```
