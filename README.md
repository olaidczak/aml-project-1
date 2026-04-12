# Advanced Machine Learning - Project 1
## Setup

1. **Clone the repository**
```bash
   git clone https://github.com/olaidczak/aml-project-1.git
```

2. **Create and activate a virtual environment**
```bash
   python -m venv .venv
   source .venv/bin/activate      # macOS/Linux
   .venv\Scripts\activate         # Windows
```

3. **Install packages**
```bash
   pip install -e ".[dev]"
```

## Repository Structure

```
├── demo.ipynb              # Demo notebook
├── pyproject.toml          # Project configuration and dependencies
└── src/
    ├── data/               # Data loading and preprocessing
    │   ├── loader.py       # Dataset loading utilities
    │   └── missing_data.py # Missing data generation
    ├── experiments/         # Experiment notebooks and helpers
    │   ├── comparison_lr.ipynb          # Logistic regression comparison
    │   ├── task3.ipynb                  # Task 3 experiments
    │   ├── task3_visualizations.ipynb   # Task 3 visualizations
    │   └── utils.py                    # Experiment utilities
    └── models/             # Model implementations
        ├── fista_lr.py     # FISTA logistic regression
        ├── unlabeled_lr.py # Unlabeled logistic regression
        └── measures.py     # Evaluation measures
```