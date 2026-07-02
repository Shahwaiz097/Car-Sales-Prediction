# Car Sales Prediction

A simple machine learning project for predicting used car selling prices using Python and scikit-learn.

## Project Structure

```text
Car-Sales-Prediction/
├── data/
│   ├── README.md
│   └── car_data_sample.csv
├── models/
│   └── car_sales_model.pkl
├── notebooks/
│   └── original_notebook.ipynb
├── outputs/
│   ├── actual_vs_predicted.png
│   ├── model_metrics.csv
│   └── predictions.csv
├── src/
│   └── train.py
├── README.md
└── requirements.txt
```

## Dataset

The project uses a used car price dataset with features such as:

- Year
- Present Price
- Kilometers Driven
- Fuel Type
- Seller Type
- Transmission
- Owner
- Selling Price

The script first looks for `data/car_data.csv`. If that file is not available, it tries to download the dataset used in the original notebook. If internet is not available, it uses the sample file in `data/car_data_sample.csv`.

## Models Used

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the training script:

```bash
python src/train.py
```

The script will save:

- Trained model in `models/`
- Metrics in `outputs/model_metrics.csv`
- Predictions in `outputs/predictions.csv`
- Actual vs predicted plot in `outputs/actual_vs_predicted.png`

## GitHub Upload Steps

1. Create a new GitHub repository named:

```text
Car-Sales-Prediction
```

2. Keep these options:

```text
Visibility: Public
Add README: Off
Add .gitignore: No .gitignore
Add license: No license
```

3. Click **Create repository**.

4. Click **uploading an existing file**.

5. Upload the project folders one by one:

```text
data/
models/
notebooks/
outputs/
src/
README.md
requirements.txt
```

6. Click **Commit changes** after each upload.
