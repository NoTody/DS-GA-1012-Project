# DS-GA-1012-Project
NYU Natural Language Understanding Spring22 Project - On Defending Strong Adversarial Text Attacks with Semi-Supervised Adversarial Learning

baseline_pl.py and baseline_main.py are files for baseline method.

cr_pl.py and cr_main.py are files for our method.

"pl" postfix means the file contains pytorch-lightning model implementation for the given method, "main" postfix means the file is used for training/testing for the given method.

# Run Example for Baseline with AGNews (1 node 1 gpu):

## train
```
python ./baseline_main.py --accumulate_grad_batches 1 --num_nodes 1 --num_devices 1 --model_name "bert-base-uncased" --dataset_name "agnews" --num_workers 10 --max_epochs 10 --batch_size 32 --max_seq_length 256 --mode "train" --lr 2e-5 --num_labels 4 --scheduler_name "cosine" --tb_save_dir "../"
```

## test
```
python ./baseline_main.py --num_nodes 1 --num_devices 1 --model_name "bert-base-uncased" --dataset_name "augmented_agnews_eda" --num_workers 10 --batch_size 16 --max_seq_length 256 --mode "test" --load_path '../lightning_logs/agnews_bert_base_eda/checkpoints/epoch=3-step=6876.ckpt' --num_labels 4
```

# Run Example for Proposed Method (1 node 1 gpu):

## train
```
python ./cr_main.py --accumulate_grad_batches 1 --num_nodes 1 --gpus 2 --model_name "bert-base-uncased" --num_workers 10 --max_epochs 20 --batch_size 32 --max_seq_length 256 --mode "train" --lr 2e-5 --num_labels 4 --scheduler_name "cosine" --dataset_name_ori "agnews" --dataset_name_str_adv "agnews_ssmba" --dataset_name_weak_aug "agnews_synonym" --tb_save_dir "./" --loss_func "l1_smooth" --top_k_layers 5
```

## test
```
python ./cr_main.py --accumulate_grad_batches 1 --num_nodes 1 --gpus 1 --model_name "bert-base-uncased" --num_workers 10 --max_epochs 10 --batch_size 32 --max_seq_length 256 --mode "test" --num_labels 4 --scheduler_name "cosine" --testset "agnews" --tb_save_dir "./" --load_path "./lightning_logs/agnews_ssmba_random/checkpoints/epoch=4-step=17190.ckpt"
```
