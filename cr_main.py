import baseline_pl
from baseline_pl import *
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
    early_stop_callback = EarlyStopping(monitor="loss", min_delta=0.00, patience=2, verbose=False, mode="min")
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor='loss',
        mode='min',
        save_weights_only=False
    )
    
    # Define Logger
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=hparams.tb_save_dir) 

    # ------------------------
    # 3 INIT TRAINER
    # ------------------------
    trainer = Trainer(precision=hparams.precision, gpus=hparams.gpus, accelerator="gpu", num_nodes=hparams.num_nodes,
                strategy=DDPStrategy(find_unused_parameters=False), max_epochs=hparams.max_epochs, 
                logger=tb_logger, callbacks=[checkpoint_callback, early_stop_callback, bar]
                )

    # ------------------------
    # 4 START TRAINING
    # ------------------------
    trainer.fit(model)

if __name__ == '__main__':
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
    parent_parser.add_argument("--tb_save_dir", 
                               type=str, 
                               default="../",
                               help='tensorboard save directory'
                              )

    # each LightningModule defines arguments relevant to it
    parser = CRTransformer.add_model_specific_args(parent_parser)
    hyperparams = parser.parse_args()
  
    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)
