import baseline_pl
from baseline_pl import *
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.strategies.ddp import DDPStrategy
import argparse
from argparse import ArgumentParser

def get_args_parser():
    parser = ArgumentParser('MAIN', add_help=False)
    # config parameters
    parser.add_argument("--tb_save_dir", type=str, default="../")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--num_devices", type=int, default=2)
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--dataset_name", type=str, default="agnews")
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--load_path", type=str, default=None)
    # model parameters
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_labels", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--scheduler_name", type=str, default="cosine")
    return parser

def test_model(args, load_path):
    loaded_model = LitTransformer(args)
    checkpoint = torch.load(load_path)
    loaded_model.eval()
    trainer.test(loaded_model)

if __name__ == '__main__':
    # suppress warning
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    # fix seed
    seed_everything(42)
    AVAIL_GPUS = max(1, torch.cuda.device_count())
    print(f"Available GPUs in current node={AVAIL_GPUS}")
    # parser
    parser = argparse.ArgumentParser('MAIN', parents=[get_args_parser()])
    args = parser.parse_args()
    # Init our model
    model = LitTransformer(args)

    # Define Callbacks
    bar = TQDMProgressBar(refresh_rate=20, process_position=0)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=2, verbose=False, mode="min")
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        save_weights_only=False
    )
    # Define Logger
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.tb_save_dir) 
    # Initialize the trainer
    trainer = Trainer(precision=16, gpus=args.num_devices, accelerator="gpu", num_nodes=args.num_nodes,
                    strategy=DDPStrategy(find_unused_parameters=False), max_epochs=args.max_epochs,
                    log_every_n_steps=20, logger=tb_logger, callbacks=[checkpoint_callback, early_stop_callback, bar]
                    )
    
    if args.mode == "train":
        print("Train mode")
        # Train the model ⚡
        trainer.fit(model)
    elif args.mode == "test":
        print("Test mode")
        # Test the model from loaded checkpoint
        checkpoint = torch.load(args.load_path)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        trainer.test(model)
