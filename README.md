# British Airways Data Science Job Simulation

**Portfolio project by Zain Yaseen**  
**Virtual experience provider:** Forage  
**Completed:** October 2026

This repository documents work completed for the British Airways Data Science Job Simulation on Forage. It is an independent learning/portfolio project, not work performed as a British Airways employee.

## Project overview

The simulation covered two business problems:

### Task 1 — Lounge eligibility modelling

- Grouped flights by shared characteristics to create a reusable lounge-eligibility lookup approach.
- Documented assumptions for estimating demand across service tiers.
- Deliverable: `task-1-lounge-eligibility/BA_Lounge_Eligibility_Lookup_Completed.xlsx`

### Task 2 — Predicting customer booking behaviour

- Explored the customer booking dataset and engineered grouped features for purchase lead, length of stay, and departure period.
- Built a preprocessing and classification pipeline using one-hot encoding and a class-weighted Random Forest.
- Used a stratified train/test split and five-fold stratified cross-validation.
- Evaluated the holdout set with accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, average precision, and a confusion matrix.
- Used permutation importance to compare the predictive contribution of original input features.
- Deliverables: the Python script, Jupyter notebook, feature-importance figure, and one-slide PowerPoint summary.

## Results from the recorded local run

The reported metrics can vary slightly across library versions or reruns. In the run reviewed for this portfolio:

| Metric | Holdout result |
|---|---:|
| Accuracy | 69.4% |
| Balanced accuracy | 72.1% |
| Precision | 29.6% |
| Recall | 75.9% |
| F1-score | 42.6% |
| ROC-AUC | 0.784 |
| Average precision | 0.377 |

Five-fold training cross-validation gave a mean ROC-AUC of **0.776** and mean average precision of **0.359**. The target was imbalanced: approximately **14.96%** of records had `booking_complete = 1`.

Permutation importance ranked `booking_origin` and `route` as the strongest original input features in this modelling setup, followed by `length_of_stay_band` and `wants_extra_baggage`. Feature importance indicates predictive association, not causation.

## Business interpretation

The model found a useful ranking signal but its precision was modest: many customers flagged as likely bookers did not complete a booking at the default 0.50 threshold. Before any real campaign use, the decision threshold should be tuned to campaign capacity and the relative costs of false positives and missed bookings. Performance should also be validated on newer booking cohorts.

## Repository contents

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── British_Airways_Customer_Booking_Analysis.ipynb
├── British_Airways_Customer_Booking_Analysis.py
├── task-1-lounge-eligibility/
│   └── BA_Lounge_Eligibility_Lookup_Completed.xlsx
└── reports/
    ├── British_Airways_Customer_Booking_Findings.pptx
    └── booking_feature_importance.png
```

## Run the Python analysis locally

1. Download the `Customer Booking.csv` dataset from the Resources section of the Forage simulation and place a local copy named `customer_booking.csv` in the repository root.
2. Create and activate a Python environment.
3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Run the script:

   ```bash
   python British_Airways_Customer_Booking_Analysis.py
   ```

   Or open `British_Airways_Customer_Booking_Analysis.ipynb` in Jupyter/VS Code and run the cells in order.

## Data-use note

The original customer booking CSV and British Airways flight-schedule source file are intentionally **not included** in this repository because the Forage resource page labels the relevant data as for simulation use only. Download the data from the official simulation resources for personal learning and keep those source files out of Git. The `.gitignore` file excludes the CSV and schedule dataset by name.

## Tools

- Python
- pandas and NumPy
- Matplotlib
- scikit-learn
- Jupyter Notebook

## Disclaimer

British Airways and Forage names and marks belong to their respective owners. This repository records an independent virtual job simulation and learning exercise; it is not an official British Airways product or endorsed analysis.
