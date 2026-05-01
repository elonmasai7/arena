PYTHON ?= python

.PHONY: install test train validate infer export benchmark lint

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

test:
	pytest

train:
	$(PYTHON) scripts/train.py

validate:
	$(PYTHON) scripts/validate.py

infer:
	$(PYTHON) scripts/infer.py

export:
	$(PYTHON) scripts/export.py

benchmark:
	$(PYTHON) scripts/benchmark.py

lint:
	$(PYTHON) -m compileall src tests scripts
