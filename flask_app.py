import pickle
from flask import Flask, request, jsonify, render_template
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, make_pipeline
import pandas as pd
from preprocessing_utils import map_emp_length
import joblib

model_status = joblib.load('loan_status_classifier.joblib')
model_grade = joblib.load('grade_regressor.joblib')
model_sub_grade = joblib.load('sub_grade_regressor.joblib')
model_int_rate = joblib.load('int_rate_regressor.joblib')

preprocessor_feature_pipeline = joblib.load('preprocessor_feature_pipeline.joblib')

app = Flask(__name__)

addr_state_categories = ['PA', 'SD', 'MN', 'SC', 'RI', 'CA', 'VA',
                         'AZ', 'MD', 'NY', 'TX', 'KS', 'NM', 'AL',
                         'WA', 'OH', 'GA', 'IL', 'FL', 'CO', 'IN',
                         'MI', 'MO', 'DC', 'MA', 'WI', 'NJ', 'DE',
                         'TN', 'NH', 'NE', 'OR', 'NC', 'AR', 'NV',
                         'WV', 'LA', 'HI', 'WY', 'KY', 'OK', 'CT',
                         'VT', 'MS', 'UT', 'ND', 'ME', 'AK', 'MT',
                         'ID', 'IA']

input_features = [
        {'name': 'loan_amnt',
         'explanation': 'Loan amount in USD (int)',
         'type': 'int64'},

        {'name': 'home_ownership',
         'type': 'object',
         'explanation': 'Home ownership status of the borrower',
         'options': ['MORTGAGE', 'RENT', 'OWN', 'OTHER']},

        {'name': 'annual_inc',
         'explanation': 'Annual income of the borrower (int)',
         'type': 'int64'},

        {'name': 'verification_status',
         'type': 'object',
         'explanation': 'Income verification status of the borrower',
         'options': ['Not Verified',
                     'Source Verified',
                     'Verified']},

        {'name': 'purpose',
         'type': 'object',
         'explanation': 'Purpose of the loan',
         'options': ['debt_consolidation', 'small_business', 'major_purchase',
                     'credit_card', 'home_improvement', 'house',
                     'other', 'car', 'medical', 'vacation', 'moving',
                     'renewable_energy', 'wedding', 'educational']},

        {'name': 'addr_state',
         'type': 'object',
         'explanation': 'State of residence of the borrower',
         'options': addr_state_categories},

        {'name': 'dti',
         'type': 'float64',
         'explanation': 'Debt-to-Income ratio (float from 0 to 1000)'},

        {'name': 'delinq_2yrs',
         'type': 'int64',
         'explanation': 'Number of 30+ days delinquencies in the last 2 years (int)'},

        {'name': 'inq_last_6mths',
         'type': 'int64',
         'explanation': 'Number of inquiries in the last 6 months (int)'},

        {'name': 'open_acc',
         'type': 'int64',
         'explanation': 'Number of open credit lines (int)'},

        {'name': 'pub_rec',
         'type': 'int64',
         'explanation': 'Number of derogatory public records (int)'},

        {'name': 'revol_bal',
         'type': 'float64',
         'explanation': 'Total credit revolving balance (float)'},

        {'name': 'initial_list_status',
         'type': 'object',
         'explanation': ' Initial listing status of the loan (w - whole, f - fractional)',
         'options': ['w', 'f']},

        {'name': 'collections_12_mths_ex_med',
         'type': 'int64',
         'explanation': 'Number of collections in 12 months excluding medical collections (int)'},

        {'name': 'application_type',
         'type': 'object',
         'explanation': 'Type of application for the loan',
         'options': ['Individual', 'Joint App']},

        {'name': 'acc_now_delinq',
         'type': 'int64',
         'explanation': 'Number of accounts on which the borrower is now delinquent (int)'},

        {'name': 'tot_coll_amt',
         'type': 'float64',
         'explanation': 'Total collection amounts ever owed (float)'},

        {'name': 'acc_open_past_24mths',
         'type': 'int64',
         'explanation': 'Number of trades opened in the last 24 months (int)'},

        {'name': 'avg_cur_bal',
         'type': 'float64',
         'explanation': 'Average current balance of all accounts (float)'},

        {'name': 'chargeoff_within_12_mths',
         'type': 'int64',
         'explanation': 'Number of charge-offs within 12 months (int)'},

        {'name': 'delinq_amnt',
         'type': 'float64',
         'explanation': 'The past-due amount owed for delinquent accounts (float)'},

        {'name': 'mo_sin_old_il_acct',
         'type': 'int64',
         'explanation': 'Months since oldest installment account was opened (int)'},

        {'name': 'mo_sin_old_rev_tl_op',
         'type': 'int64',
         'explanation': 'Months since oldest revolving account was opened (int)'},

        {'name': 'mo_sin_rcnt_rev_tl_op',
         'type': 'int64',
         'explanation': 'Months since most recent revolving account was opened (int)'},

        {'name': 'mo_sin_rcnt_tl',
         'type': 'int64',
         'explanation': 'Months since most recent account was opened (int)'},

        {'name': 'mort_acc',
         'type': 'int64',
         'explanation': 'Number of mortgage accounts (int)'},

        {'name': 'mths_since_recent_bc',
         'type': 'int64',
         'explanation': 'Months since recent bankcard delinquency (int)'},

        {'name': 'mths_since_recent_inq',
         'type': 'int64',
         'explanation': 'Months since recent inquiry (int)'},

        {'name': 'num_accts_ever_120_pd',
         'type': 'int64',
         'explanation': 'Number of accounts ever 120 or more days past due (int)'},

        {'name': 'num_actv_bc_tl',
         'type': 'int64',
         'explanation': 'Number of currently active bankcard accounts (int)'},

        {'name': 'num_il_tl',
         'type': 'int64',
         'explanation': 'Number of installment accounts (int)'},

        {'name': 'num_rev_accts',
         'type': 'int64',
         'explanation': 'Number of revolving accounts (int)'},

        {'name': 'num_tl_120dpd_2m',
         'type': 'int64',
         'explanation': ' Number of accounts currently 120 days or more past due (int)'},

        {'name': 'num_tl_30dpd',
         'type': 'int64',
         'explanation': 'Number of accounts currently 30 days past due (int)'},

        {'name': 'num_tl_90g_dpd_24m',
         'type': 'int64',
         'explanation': 'Number of accounts 90 or more days past due in last 24 months (int)'},

        {'name': 'num_tl_op_past_12m',
         'type': 'int64',
         'explanation': 'Number of accounts opened in the last 12 months (int)'},

        {'name': 'pct_tl_nvr_dlq',
         'type': 'float64',
         'explanation': 'Percentage of trades never delinquent (float from 0 to 100)'},

        {'name': 'pub_rec_bankruptcies',
         'type': 'int64',
         'explanation': 'Number of public record bankruptcies (int)'},

        {'name': 'tax_liens',
         'type': 'int64',
         'explanation': 'Number of tax liens (int)'},

        {'name': 'total_bc_limit',
         'type': 'float64',
         'explanation': 'Total bankcard high credit/credit limit (float)'},

        {'name': 'total_il_high_credit_limit',
         'type': 'float64',
         'explanation': 'Total installment high credit/credit limit (float)'},

        {'name': 'disbursement_method',
         'type': 'object',
         'explanation': 'Method of disbursement for the loan',
         'options': ['Cash', 'DirectPay']},

        {'name': 'term_months',
         'type': 'int64',
         'explanation': 'Loan term in months (int)'},

        {'name': 'emp_length_years',
         'type': 'object',
         'explanation': 'Employment length in years',
         'options': ['10+', '1-3', '4-6', '7-9', '<1']},

        {'name': 'emp_length_numeric',
         'type': 'int64',
         'explanation': 'Numeric representation of employment length (years, int)'},

        {'name': 'fico_range_avg',
         'type': 'float64',
         'explanation': 'Average FICO score range (float from 0 to 1000)'},

        {'name': 'utilization_rate',
         'type': 'float64',
         'explanation': 'Credit utilization rate (float from 0 to 100)'},

        {'name': 'delinquent',
         'type': 'int64',
         'explanation': 'Whether the borrower is delinquent (0 or 1)',
         'options': [0, 1]},

        {'name': 'high_utilization',
         'type': 'int64',
         'explanation': 'High utilization indicator (0 or 1)',
         'options': [0, 1]},

        {'name': 'last_fico_range_avg',
         'type': 'float64',
         'explanation': 'Last FICO score range average (float from 0 to 1000)'}
]

@app.route('/')
def index():
    return render_template('index.html',
                           input_features=input_features,
                           addr_state_categories=addr_state_categories)


def preprocess_df(df):
    print("Dataframe shape before preprocessing:", df.shape)
    processed_df = preprocessor_feature_pipeline.transform(df)
    print("Dataframe shape after preprocessing:", processed_df.shape)
    return processed_df


def preprocessDataAndPredict(df):
    processed_df = preprocess_df(df)
    print("Processed DataFrame:")
    print(processed_df)

    print("processed_df no reshape", processed_df.shape)

    prediction_status = model_status.predict(processed_df.reshape(1, -1))
    prediction_grade = model_grade.predict(processed_df.reshape(1, -1))
    prediction_sub_grade = model_sub_grade.predict(processed_df.reshape(1, -1))
    prediction_int_rate = model_int_rate.predict(processed_df.reshape(1, -1))

    print("processed_df reshaped and sliced:", processed_df[0, :].reshape(1, -1))
    print("processed_df reshaped:", processed_df.reshape(1, -1))

    return (prediction_status, prediction_grade,
            prediction_sub_grade, prediction_int_rate)


@app.route('/home/')
def home():
    return render_template('index.html')


@app.route('/predict/', methods=['POST'])
def predict():
    if request.method == "POST":
        try:
            form_data = {}

            for feature in input_features:
                feature_name = feature['name']
                feature_type = feature['type']

                if feature_type == 'int64':
                    form_data[feature_name] = int(request.form.get(feature_name))
                elif feature_type == 'float64':
                    form_data[feature_name] = float(request.form.get(feature_name))
                elif feature_type == 'object':
                    form_data[feature_name] = request.form.get(feature_name)

            print("Form Data:", form_data)

            raw_df = pd.DataFrame([form_data])

            print("Raw DataFrame:")
            print(raw_df)
            print("Raw DataFrame Features:", raw_df.columns.tolist())

            status, grade, sub_grade, int_rate = preprocessDataAndPredict(raw_df)

            # Pass prediction to template
            return render_template('predict.html',
                                   prediction=[status, grade, sub_grade, int_rate])

        except ValueError as e:
            print(f"Error: {e}")
            return "Please, enter valid values"

    return render_template('predict.html', prediction=None)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)