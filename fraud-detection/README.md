# Credit Card Fraud Detection

A machine learning project that classifies a credit card transaction as **fraud** or **legitimate**, with a small Streamlit web app to try it out.

*Mini Project (SE Semester 4, A. P. Shah Institute of Technology, 2023-24), team project. My part: training the model and building the web front end.*

## How it works

1. **Data:** the public Kaggle "Credit Card Fraud Detection" dataset (284,807 transactions, only 492 are fraud). Each transaction has `Time`, `V1` to `V28` (anonymised features) and `Amount`.
2. **Balancing:** fraud is very rare, so I keep all fraud rows and randomly pick the same number of legitimate rows (undersampling).
3. **Split:** 80% training, 20% testing, with the fraud/legitimate ratio kept equal in both.
4. **Model:** Logistic Regression from scikit-learn.
5. **App:** paste the 30 feature values of a transaction and the app answers "Fraud Transaction" or "Legitimate Transaction". The sidebar shows the training and test accuracy.

## Files

- `app.py`: trains the model and runs the Streamlit app
- `sample_inputs.txt`: one example fraud row and one legitimate row to paste into the app
- `requirements.txt`: Python packages

## How to run

1. The dataset is not included because it is large. Download `creditcard.csv` from Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud (you need a free Kaggle account), then put it next to `app.py`.
2. `pip install -r requirements.txt`
3. `streamlit run app.py`
4. Paste a row from `sample_inputs.txt` and click Submit.

## Limitations

- After undersampling, the model learns from fewer than 1,000 rows, so it throws away most of the legitimate data.
- Accuracy alone can hide mistakes on a rare class. A fuller evaluation would also report precision, recall and a confusion matrix.
- The app needs all 30 numbers typed in by hand. It is a demo, not a real payment system.
