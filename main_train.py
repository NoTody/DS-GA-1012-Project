import pl_module
from pl_module import *
from argparse import ArgumentParser

def get_args_parser():
	parser = ArgumentParser()
	# config parameters
	parser.add_argument("--model_name", type=str, default="bert-base-uncased")
	parser.add_argument("--dataset_name", type=str, default="agnews")
	parser.add_argument("--num_workers", type=int, default=4)
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--max_seq_length", type=int, default=256)
	# model parameters
	parser.add_argument("--lr", type=float, default=2e-5)
	parser.add_argument("--num_labels", type=int, default=4)
	parser.add_argument("--weight_decay", type=float, default=0.0)
	return parser

if __name__ == '__main__':
	# fix seed
	seed_everything(42)
	
	# parser
	parser = argparse.ArgumentParser('ADV', parents=[get_args_parser()])
  	args = parser.parse_args([])

  	# Init our model
  	model = LitTransformer(args)

  	# Define Callbacks
  	bar = TQDMProgressBar(refresh_rate=20, process_position=0)
  	early_stop_callback = EarlyStopping(monitor="val_f1", min_delta=0.00, patience=2, verbose=False, mode="max")
	checkpoint_callback = ModelCheckpoint(
	  # dirpath=os.getcwd(),
	  save_top_k=1,
	  verbose=True,
	  monitor='val_f1',
	  mode='max',
	  save_weights_only=False
	)

	# Initialize the trainer
	trainer = Trainer(
	  precision=16,
	  gpus=AVAIL_GPUS,
	  accelerator="gpu",
	  max_epochs=5,
	  callbacks=[checkpoint_callback, early_stop_callback, bar]
	)

	# Train the model ⚡
	trainer.fit(model)