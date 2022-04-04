import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup

def model_init(model_name, config):
	"""
	Returns an initialized model
	"""
	return AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
