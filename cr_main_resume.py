import cr_pl
from cr_pl import *
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.strategies.ddp import DDPStrategy
import argparse
from argparse import ArgumentParser

def main(hparams):
    """
    Main training routine specific for this project
    :param hparams:
    """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = CRTransformer(hparams)
 
    # ------------------------
    # 2 INIT CALLBACKS
    # ------------------------
    bar = TQDMProgressBar(refresh_rate=20, process_position=0)
    #early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=2, verbose=False, mode="min")
    checkpoint_callback = ModelCheckpoint(
        save_top_k=5,
        verbose=True,
        monitor='val_loss',
        mode='min',
        save_weights_only=False,
        dirpath=hparams.save_dir,
    )
    
    # Define Logger
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=hparams.save_dir) 
    # ------------------------
    # 3 INIT TRAINER
    # ------------------------
    trainer = Trainer(precision=hparams.precision, devices=hparams.gpus, accelerator="gpu", num_nodes=hparams.num_nodes,
                    strategy=DDPStrategy(find_unused_parameters=False), max_epochs=hparams.max_epochs, 
                    logger=tb_logger, callbacks=[checkpoint_callback, bar],
                    )

    # ------------------------
    # 4 START TRAINING
    # ------------------------
    print("Train mode from resumed checkpoint")
    trainer.fit(model, ckpt_path=hparams.load_path)

if __name__ == '__main__':
    # suppress warning
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    root_dir = os.path.dirname('./trans_pl')
    parent_parser = argparse.ArgumentParser(add_help=False)

    # gpu args
    parent_parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='how many gpus'
    )
    parent_parser.add_argument(
        '--num_nodes',
        type=int,
        default=1,
        help='how many nodes'
    )
    parent_parser.add_argument(
        '--precision',
        type=int,
        default=16,
        help='default to use mixed precision 16'
    )
    parent_parser.add_argument(
        "--save_dir", 
        type=str,
        default="./",
        help='tensorboard/checkpoints save directory'
    )

    # each LightningModule defines arguments relevant to it
    parser = CRTransformer.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()
    
    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)
 
