#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = GapFilter
PYTHON_VERSION = 3.9
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Download data
.PHONY: data_download
data_download: 
	nohup bash data_download.sh > data/raw/download.log 2>&1 &

## Download pretrained models
.PHONY: model_download
model_download: 
	nohup bash model_download.sh > models/pretrained_models/download.log 2>&1 &

## Preprocess data
.PHONY: data_preprocess
data_preprocess:
	$(PYTHON_INTERPRETER) gapfilter/data.py \
	--overwrite \
	$(if $(from_interim),--from-interim,)
# 从 raw 全量: make data_preprocess
# 从 interim 续跑: make data_preprocess from_interim=1
# 不加 --overwrite 时，若 processed/expr.h5ad 已存在则跳过该样本

## run KNN baseline
.PHONY: knn_baseline
knn_baseline: 
	$(PYTHON_INTERPRETER) gapfilter/baseline/KNN.py \
	$(if $(dataset),dataset=$(dataset),)
# for example: make knn_baseline dataset=visiumhd_hbc

## extract H&E features
.PHONY: extract_features
extract_features:
	$(PYTHON_INTERPRETER) gapfilter/features.py -m \
	experiment=extract \
	$(if $(feature),feature=$(feature),) \
	$(if $(dataset),dataset=$(dataset),)
# make extract_features experiment=extract feature=seal_univ2 dataset=visiumhd_hbc,visiumhd_hpc,xenium_hbc_s1r1,xenium_hbc_s1r2,visium_dlpfc


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
