PYTHON  ?= python
WEIGHTS ?=
ARGS    ?=

# официальный код DEIMv2 (модель, loss) — фиксированный коммит
DEIMV2_REPO   := https://github.com/Intellindust-AI-Lab/DEIMv2.git
DEIMV2_COMMIT := e658301c11a29c4e34a8b1cb82b282c9bb8d206d
DEIMV2_DIR    := third_party/DEIMv2

.PHONY: help install stage1 stage2 clean

help:
	@echo "make install                                - скачать код DEIMv2 и поставить зависимости"
	@echo "make stage1                                 - обучение на stage1/images с COCO-весов"
	@echo "make stage2 [WEIGHTS=<папка>]               - дообучение на stage2/images (с весов stage1, если WEIGHTS задан)"
	@echo "models: atto | femto | pico | nano | small | medium | large | xlarge  (ARGS=\"model.size=...\")"
	@echo "ARGS=\"...\"                                - доп. hydra-оверрайды"

install:
	$(if $(wildcard $(DEIMV2_DIR)/engine),,git clone $(DEIMV2_REPO) $(DEIMV2_DIR))
	git -C $(DEIMV2_DIR) checkout $(DEIMV2_COMMIT)
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

ifeq ($(OS),Windows_NT)
    RM_DIR = rmdir /S /Q
else
    RM_DIR = rm -rf
endif

clean:
	-$(RM_DIR) outputs
