PYTHON  ?= python
WEIGHTS ?=
ARGS    ?=

.PHONY: help install stage1 stage2

help:
	@echo "make install                      - поставить зависимости"
	@echo "make stage2 [WEIGHTS=<.pth>]      - обучение на stage2/images (с весов stage1, если WEIGHTS задан)"
	@echo "ARGS=\"...\"                      - доп. hydra-оверрайды"

install:
	$(PYTHON) -m pip install -r requirements.txt

stage1:
	$(PYTHON) main.py \
		data.train=stage1/images \
		$(ARGS)

stage2:
	$(PYTHON) main.py \
		data.train=stage2/images \
		$(if $(WEIGHTS),"model.weights='$(WEIGHTS)'") \
		$(ARGS)
