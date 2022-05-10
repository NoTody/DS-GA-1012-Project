import argparse
import transformers
import textattack
import pandas as pd
from datasets import load_dataset
from csv import reader


class Attack():
	"""
	Generate adversarial examples based on the fine-tuned bert models provided in Textfooler and A2T.
	
	Textfooler: https://arxiv.org/abs/1907.11932
	A2T: https://arxiv.org/abs/2109.00544
	"""
	def __init__(self,
				paper,
				target_model, 
				num_examples, 
				log_to_csv, 
				checkpoint_dir,
				dataset,
				checkpoint_interval=50
				):
		"""
		Args:
			paper: the paper that our attacking method is based on (textfooler/a2t)
			target_model: pretrained model that is about to be attacked.
			num_examples: the number of adversarial examples we want to generate.
			log_to_csv: name or path for saving generated adversarial examples.
			checkpoint_interval: interval for saving checkpoints.
			checkpoint_dir: name or path for saving checkpoints.
			dataset: orginal datasets to be attacked
		"""
		self.paper = paper
		self.target_model = target_model
		self.num_examples = num_examples
		self.log_to_csv = log_to_csv
		self.checkpoint_interval = checkpoint_interval
		self.checkpoint_dir = checkpoint_dir
		self.dataset = dataset


	def attack(self):
		"""
		Execute static attacking process.
		"""
		model = transformers.AutoModelForSequenceClassification.from_pretrained(self.target_model)
		tokenizer = transformers.AutoTokenizer.from_pretrained(self.target_model)

		#Use HuggingFaceModelWrapper class to implement both the forward pass and tokenization
		model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model,tokenizer)

		#Construct attack instance
		if self.paper ==  "textfooler":
			attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)
		elif self.paper ==  "a2t":
			attack = textattack.attack_recipes.a2t_yoo_2021.build(model_wrapper)
		else:
			print("Erroneous paper, not included in this project!")

		#Start attacking...
		attack_args = textattack.AttackArgs(num_examples=self.num_examples, 
											log_to_csv=self.log_to_csv,
											checkpoint_interval=self.checkpoint_interval,
											checkpoint_dir=self.checkpoint_dir,
											disable_stdout=True,
											parallel=True)

		attacker = textattack.Attacker(attack, self.dataset, attack_args)
		attacker.attack_dataset()


def data_preprocess(filepath):
  """
		Preprocess datasets for attacking.

		Parameter:
			filepath - path to the original dataset.
		Returns:
			dataset - preprocessed dataset ready for being attacked.
	"""
  with open(filepath, 'r') as read_obj:
    csv_reader = reader(read_obj)
    list_of_tuples = list(map(tuple, csv_reader))

  my_dataset=[]
  if len(list_of_tuples[0])==2:
    for i in range(1,len(list_of_tuples)):
      my_dataset.append((list_of_tuples[i][0], int(list_of_tuples[i][1])))
  elif len(list_of_tuples[0])==3:
    for i in range(1,len(list_of_tuples)):
      my_dataset.append(((list_of_tuples[i][0], list_of_tuples[i][1]), int(list_of_tuples[i][2])))
      
  dataset = textattack.datasets.Dataset(my_dataset)
  return dataset

def main():
	
	parser  = argparse.ArgumentParser()

	#Required parameters
	parser.add_argument("--paper",
						type=str,
						required=True,
						choices=["textfooler","a2t"],
						help="the paper which our attack method is based on"
						)

	parser.add_argument("--target_model",
						type=str,
						required=True,
						choices=["textattack/bert-base-uncased-imdb", "textattack/bert-base-uncased-ag-news", "textattack/bert-base-uncased-snli"],
						help="the version of pretrained bert model"
						     "For classification task: bert-base-uncased-imdb/bert-base-uncased-ag-news"
						     "For NLI task: bert-base-uncased-snli"
						)

	parser.add_argument("--num_examples",
						type=int,
						required=True,
						help="the number of adversarial examples we want to generate"
						)

	parser.add_argument("--log_to_csv",
						type=str,
						required=True,
						help="the path of file for saving generated examples"
						)

	parser.add_argument("--checkpoint_dir",
						type=str,
						required=True,
						help="the path of folder for saving checkpoints"
						)

	parser.add_argument("--dataset_path",
						type=str,
						required=True,
						help="Which dataset to attack"
						)

	args = parser.parse_args()

	#Get data to attack
	#For classification task, texts will be attacked, while for textual entailment task, we will use SNLI, where hypothesis will be attacked
	dataset = data_preprocess(args.dataset_path)

	attack = Attack(args.paper,
					args.target_model,
					args.num_examples, 
					args.log_to_csv,
					args.checkpoint_dir,
					dataset
		)
	attack.attack()


if __name__ == "__main__":
	main()

