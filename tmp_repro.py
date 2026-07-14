import sys
import traceback
sys.path.insert(0, r'c:/Users/BOSS/OneDrive/Desktop/Credit Card Score Builder')
from src.predictor import predict_applicant

data = {
    'loan_amnt': 10000,
    'term': '36 months',
    'int_rate': 10,
    'grade': 'B',
    'sub_grade': 'B1',
    'emp_length': '10+ years',
    'home_ownership': 'RENT',
    'annual_inc': 60000,
    'verification_status': 'Verified',
    'purpose': 'debt_consolidation',
    'dti': 15,
    'fico_range_low': 700,
    'fico_range_high': 704,
    'open_acc': 10,
    'revol_bal': 10000,
    'revol_util': 45,
    'delinq_2yrs': 0,
    'pub_rec': 0,
    'inq_last_6mths': 1,
    'unrate': 4.2,
    'fedfunds': 4.5,
    'cpi': 320,
}

try:
    print(predict_applicant(data))
except Exception:
    traceback.print_exc()
