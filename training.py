from src.baseline import train_and_evaluate
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

training_sizes = [25, 50, 100, 150, 200, 250, 300]

for size in training_sizes:
    result = train_and_evaluate(size)