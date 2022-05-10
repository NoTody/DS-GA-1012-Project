#pip install datasets
# !pip install -U nltk
# !python -m nltk.downloader omw

"""
EDA Augmentation method.
"""

from datasets import load_dataset
from eda import *
import pandas as 
import nltk
nltk.download("wordnet")
nltk.download("all")


def gen_eda(train_orig, output_file, alpha_sr, alpha_ri, alpha_rs, alpha_rd, num_aug=1):

	writer = open(output_file, 'w')
	lines = open(train_orig, 'r').readlines()

	for i, line in enumerate(lines):
		parts = line[:-1].split(',',maxsplit=3)
		label = parts[2]
		hypothesis = parts[1]
		premise = parts[0]
		try:
			aug_sentences = eda(hypothesis, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd, num_aug=num_aug)
		except (ValueError, IndexError):
			aug_sentences = hypothesis
		for aug_sentence in aug_sentences:
			writer.write(str(label) + "\t" + aug_sentence + "\t" + premise + '\n')

	writer.close()
	print("generated augmented sentences with eda for " + train_orig + " to " + output_file + " with num_aug=" + str(num_aug))


def main():

	parser = argparse.ArgumentParser()

	#required params
	parser.add_argument("--output_file",
						type=str,
						required=True,
						help="the filepath or filename for generated adversarial texts")

	parser.add_argument("--train_orig",
						type=str,
						required=True,
						help="the filepath or filename for original texts")
	
	parser.add_argument("--alpha_sr",
						type=float,
						required=True,
						help="synonyms replacement ratio")

	parser.add_argument("--alpha_ri",
						type=float,
						required=True,
						help="random insertion ratio")

	parser.add_argument("--alpha_rs",
						type=float,
						required=True,
						help="random swap ratio")

	parser.add_argument("--alpha_rd",
						type=float,
						required=True,
						help="random deletion rate")

	parser.add_argument("--num_aug",
						type=float,
						required=True,
						help="number of augmented examples")

	args = parser.parse_args()
	
	#start generating adversarial examples
	gen_eda(args.train_orig, args.output_file, args.alpha_sr, args.alpha_ri, args.alpha_rs, args.alpha_rd, args.num_aug)


if __name__ == "__main__":
	main()









