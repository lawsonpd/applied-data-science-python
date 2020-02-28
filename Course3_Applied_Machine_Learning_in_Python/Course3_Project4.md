## Project 4 - Understanding and Predicting Property Maintenance Fines

This project is based on a data challenge from the Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)). 

The Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)) and the Michigan Student Symposium for Interdisciplinary Statistical Sciences ([MSSISS](https://sites.lsa.umich.edu/mssiss/)) have partnered with the City of Detroit to help solve one of the most pressing problems facing Detroit - blight. [Blight violations](http://www.detroitmi.gov/How-Do-I/Report/Blight-Complaint-FAQs) are issued by the city to individuals who allow their properties to remain in a deteriorated condition. Every year, the city of Detroit issues millions of dollars in fines to residents and every year, many of these fines remain unpaid. Enforcing unpaid blight fines is a costly and tedious process, so the city wants to know: how can we increase blight ticket compliance?

The first step in answering this question is understanding when and why a resident might fail to comply with a blight ticket. This is where predictive modeling comes in. For this project, your task is to predict whether a given blight ticket will be paid on time.

All data for this project has been provided to us through the [Detroit Open Data Portal](https://data.detroitmi.gov/). **Only the data already included in your Coursera directory can be used for training the model for this project.** Nonetheless, we encourage you to look into data from other Detroit datasets to help inform feature creation and model selection. We recommend taking a look at the following related datasets:

* [Building Permits](https://data.detroitmi.gov/Property-Parcels/Building-Permits/xw2a-a7tf)
* [Trades Permits](https://data.detroitmi.gov/Property-Parcels/Trades-Permits/635b-dsgv)
* [Improve Detroit: Submitted Issues](https://data.detroitmi.gov/Government/Improve-Detroit-Submitted-Issues/fwz3-w3yn)
* [DPD: Citizen Complaints](https://data.detroitmi.gov/Public-Safety/DPD-Citizen-Complaints-2016/kahe-efs3)
* [Parcel Map](https://data.detroitmi.gov/Property-Parcels/Parcel-Map/fxkw-udwf)

___

We provide you with two data files for use in training and validating your models: train.csv and test.csv. Each row in these two files corresponds to a single blight ticket, and includes information about when, why, and to whom each ticket was issued. The target variable is compliance, which is True if the ticket was paid early, on time, or within one month of the hearing data, False if the ticket was paid after the hearing date or not at all, and Null if the violator was found not responsible. Compliance, as well as a handful of other variables that will not be available at test-time, are only included in train.csv.

Note: All tickets where the violators were found not responsible are not considered during evaluation. They are included in the training set as an additional source of data for visualization, and to enable unsupervised and semi-supervised approaches. However, they are not included in the test set.

<br>

**File descriptions** (Use only this data for training your model!)

    train.csv - the training set (all tickets issued 2004-2011)
    test.csv - the test set (all tickets issued 2012-2016)
    addresses.csv & latlons.csv - mapping from ticket id to addresses, and from addresses to lat/lon coordinates. 
     Note: misspelled addresses may be incorrectly geolocated.

<br>

**Data fields**

train.csv & test.csv

    ticket_id - unique identifier for tickets
    agency_name - Agency that issued the ticket
    inspector_name - Name of inspector that issued the ticket
    violator_name - Name of the person/organization that the ticket was issued to
    violation_street_number, violation_street_name, violation_zip_code - Address where the violation occurred
    mailing_address_str_number, mailing_address_str_name, city, state, zip_code, non_us_str_code, country - Mailing address of the violator
    ticket_issued_date - Date and time the ticket was issued
    hearing_date - Date and time the violator's hearing was scheduled
    violation_code, violation_description - Type of violation
    disposition - Judgment and judgement type
    fine_amount - Violation fine amount, excluding fees
    admin_fee - $20 fee assigned to responsible judgments
state_fee - $10 fee assigned to responsible judgments
    late_fee - 10% fee assigned to responsible judgments
    discount_amount - discount applied, if any
    clean_up_cost - DPW clean-up or graffiti removal cost
    judgment_amount - Sum of all fines and fees
    grafitti_status - Flag for graffiti violations
    
train.csv only

    payment_amount - Amount paid, if any
    payment_date - Date payment was made, if it was received
    payment_status - Current payment status as of Feb 1 2017
    balance_due - Fines and fees still owed
    collection_status - Flag for payments in collections
    compliance [target variable for prediction] 
     Null = Not responsible
     0 = Responsible, non-compliant
     1 = Responsible, compliant
    compliance_detail - More information on why each ticket was marked compliant or non-compliant


___

## Evaluation

Your predictions will be given as the probability that the corresponding blight ticket will be paid on time.

The evaluation metric for this project is the Area Under the ROC Curve (AUC). 

Your grade will be based on the AUC score computed for your classifier. A model which with an AUROC of 0.7 passes this project, over 0.75 will recieve full points.
___

For this project, create a function that trains a model to predict blight ticket compliance in Detroit using `train.csv`. Using this model, return a series of length 61001 with the data being the probability that each corresponding ticket from `test.csv` will be paid, and the index being the ticket_id.

Example:

    ticket_id
       284932    0.531842
       285362    0.401958
       285361    0.105928
       285338    0.018572
                 ...
       376499    0.208567
       376500    0.818759
       369851    0.018528
       Name: compliance, dtype: float32

## Student work below


```python
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer, OneHotEncoder
from sklearn.metrics import roc_auc_score

# Note:
# For most assignments we were advised to include all of our work inside the main function that
# would return our answer (e.g. in this case `blight_model`). This was because global variables
# and functions would sometimes cause problems for the autograder.

def blight_model():
    train_cols = ['admin_fee',
#              'agency_name',
#              'balance_due',
#              'city',
             'clean_up_cost',
             'compliance_detail',
             'discount_amount',
#              'disposition',
             'fine_amount',
             'hearing_date',
#              'inspector_name',
             'judgment_amount',
             'late_fee',
#              'mailing_address_str_name',
#              'state',
             'state_fee',
             'ticket_id',
             'ticket_issued_date',
#              'violation_code',
#              'violation_description',
#              'violation_street_name',
#              'zip_code'
           ]
    
    test_cols = ['admin_fee',
#              'agency_name',
#              'balance_due',
#              'city',
             'clean_up_cost',
#              'compliance_detail',
             'discount_amount',
#              'disposition',
             'fine_amount',
             'hearing_date',
#              'inspector_name',
             'judgment_amount',
             'late_fee',
#              'mailing_address_str_name',
#              'state',
             'state_fee',
             'ticket_id',
             'ticket_issued_date',
#              'violation_code',
#              'violation_description',
#              'violation_street_name',
#              'zip_code'
       ]

    
    compliance_cats = ['non-compliant by no payment',
                     'compliant by late payment within 1 month',
                     'non-compliant by late payment more than 1 month',
                     'compliant by early payment',
                     'compliant by on-time payment',
                     'compliant by payment with no scheduled hearing',
                     'compliant by payment on unknown date',
                     'compliant by no fine'
                    ]

    compliance_map = dict(zip(compliance_cats, (0, 1, 0, 1, 1, 1, 1, 1)))

    def convert_comp(detail):
        return compliance_map[detail]
    
    def convert_time(d):
        td = pd.to_timedelta(d)
        try:
            days_diff = td.days
        except AttributeError:
            days_diff = X_train['dates_diff'].mean()
        return days_diff
    
    train_df = pd.read_csv('train.csv', encoding='ISO-8859-1', usecols=train_cols, parse_dates=[1,2])
    test_df = pd.read_csv('test.csv', encoding='ISO-8859-1', usecols=test_cols, parse_dates=[1,2])
    
    # drop where party found not responsible
    train_df = train_df.where(train_df['compliance_detail'] != 'not responsible by disposition').dropna(subset=['compliance_detail'])
    train_df = train_df.where(train_df['compliance_detail'] != 'not responsible by pending judgment disposition').dropna(subset=['compliance_detail'])
    
    
    # drop ticket_id column
    train_df = train_df.drop('ticket_id', axis=1)
    
    # add column for difference in dates
    train_df['dates_diff'] = train_df['hearing_date'] - train_df['ticket_issued_date']
    train_df = train_df.drop(['ticket_issued_date', 'hearing_date'], axis=1)
    
    # drop rows where hearing_date was NaT
    train_df = train_df.dropna(how='any', axis=0)
    
    # convert date diff to days count
    train_df['dates_diff'] = train_df['dates_diff'].apply(convert_time)
    
    # make 'compliance' last column
    columns_list = list(train_df.columns)
    columns_list[-2], columns_list[-1] = columns_list[-1], columns_list[-2]
    train_df = train_df.reindex(columns=columns_list)
    
    # split X,y
    X, y = train_df.iloc[:, :-1], train_df.iloc[:, -1:]
    
    # convert y vector to binary
    y['compliance_detail'] = y['compliance_detail'].apply(convert_comp)
    
    clf = tree.DecisionTreeClassifier()
    
    clf.fit(X, y)
    
    ###########
    ### Pre-process test data
    
    # copy and drop ticket_id column
    test_id_col = test_df['ticket_id']
    test_df = test_df.drop('ticket_id', axis=1)
    
    # add column for difference in dates
    test_df['dates_diff'] = test_df['hearing_date'] - test_df['ticket_issued_date']
    test_df = test_df.drop(['ticket_issued_date', 'hearing_date'], axis=1)
    
    # drop rows where hearing_date was NaT
    test_df['dates_diff'] = test_df['dates_diff'].fillna(test_df['dates_diff'].mean())
    
    # convert date diff to days count
    test_df['dates_diff'] = test_df['dates_diff'].apply(convert_time)
    
    ### / Pre-process test data
    ###########
    
    preds = clf.predict_proba(test_df)
    
    results_df = pd.DataFrame(preds, index=test_id_col, columns=[0,1])

    return results_df[1]

```
