# Validation-Based LLM Reasoning

Implementation code and experimental resources for the master’s thesis:

**“A Validation-Based Framework for Reasoning Reliability in Large Language Models Without Model Retraining”**

## Overview

This repository contains the implementation of a validation-based reasoning control framework designed to improve the reasoning reliability of Large Language Models (LLMs) without retraining the underlying model.

The framework introduces an external validation layer that:

* structures input information and generated reasoning as graphs,
* applies task-adaptive validation mechanisms,
* supports reasoning revision,
* and rejects unsupported reasoning through explicit refusal behavior.

The experiments were conducted using GPT-5 on the NeuLR and CLUTRR reasoning datasets.


## Repository Structure

```text
analysis/
    Utility scripts for filtering and analyzing experimental outputs.

data/
    raw/
        Original datasets used in the experiments.
    processed/
        Preprocessed datasets used by the framework.

evaluation/
    Scripts for reevaluating and correcting experimental outputs
    under the final evaluation settings.

preprocessing/
    Dataset preprocessing scripts for NeuLR and CLUTRR.

results/
    Full experimental output files.

main.py
    Baseline evaluation pipeline.

main_framework.py
    Validation-based reasoning framework pipeline.

requirements.txt
    Python package dependencies.

config.env
    Environment configuration file for API access.
```


## Datasets

This study uses the following datasets:

* NeuLR
* CLUTRR

The datasets are used for research and evaluation purposes.


## Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Set the OpenAI API key in `config.env`:

```env
OPENAI_API_KEY=your_api_key
```


## Running Experiments

Baseline evaluation:

```bash
python main.py
```

Framework-based evaluation:

```bash
python main_framework.py
```


## Preprocessing

Dataset preprocessing scripts are located in:

```text
preprocessing/
```

These scripts convert the original datasets into the processed JSON format used by the framework.


## Evaluation and Analysis

The repository also includes:

* reevaluation scripts (`evaluation/`)
* analysis utilities for filtering error/refusal samples (`analysis/`)
* complete experimental outputs (`results/`)


## Thesis Context

This repository accompanies a master’s thesis submitted to Stockholm University.

The framework focuses on reasoning reliability rather than only answer accuracy. The goal is to reduce unsupported reasoning by introducing explicit validation and controlled refusal behavior.


## License

This repository is provided for academic and research purposes.
