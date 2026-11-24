.PHONY: install train gradcheck test curves all clean

install:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

train:
	.venv/bin/python train.py

gradcheck:
	.venv/bin/python -m src.attention
	.venv/bin/python -m src.layer_norm
	.venv/bin/python -m src.block

test:
	.venv/bin/python -m unittest discover tests

curves:
	.venv/bin/python -m experiments.plot_loss

all: test gradcheck train

clean:
	rm -rf data/ runs/ checkpoints/ samples/ __pycache__ */__pycache__ */*/__pycache__
