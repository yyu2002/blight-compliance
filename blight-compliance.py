import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

def blight_model():
    
    # load train and test data
    train_data = pd.read_csv('train.csv', encoding='ISO-8859-1')
    test_data = pd.read_csv('test.csv')
    
    # remove null values in compliance column
    train_data = train_data[(train_data['compliance']==0) | (train_data['compliance'])==1]
    
    # load address and location data
    addresses = pd.read_csv('addresses.csv')
    latlons = pd.read_csv('latlons.csv')
    addresses = addresses.set_index('address').join(latlons.set_index('address'))
    
    # merge address and location to train and test data
    train_data = train_data.set_index('ticket_id').join(addresses.set_index('ticket_id'))
    test_data = test_data.set_index('ticket_id').join(addresses.set_index('ticket_id'))
    
    # remove null values in hearing date column 
    train_data = train_data[~train_data['hearing_date'].isnull()]
    
    # remove features not available in test data, but keep compliance
    cols_to_remove = [
            'balance_due',
            'collection_status',
            'compliance_detail',
            'payment_amount',
            'payment_date',
            'payment_status'
        ]
    train_data = train_data.drop(cols_to_remove, axis=1)
   
    # remove non-numeric features
    cols_to_remove = ['violator_name', 'zip_code', 'country', 'city',
            'inspector_name', 'violation_street_number', 'violation_street_name',
            'violation_zip_code', 'violation_description',
            'mailing_address_str_number', 'mailing_address_str_name',
            'non_us_str_code', 'agency_name', 'state', 'disposition',
            'ticket_issued_date', 'hearing_date', 'grafitti_status', 'violation_code'
        ]
    train_data = train_data.drop(cols_to_remove, axis=1)
    test_data = test_data.drop(cols_to_remove, axis=1)
    
    # fill null values in lat and long columns
    train_data['lat'] = train_data['lat'].fillna(method='ffill')
    train_data['lon'] = train_data['lon'].fillna(method='ffill')
    test_data['lat'] = test_data['lat'].fillna(method='ffill')
    test_data['lon'] = test_data['lon'].fillna(method='ffill')
    
    # split up data into X train, X test, y train, 
    y_train = train_data['compliance']
    X_train = train_data.drop('compliance', axis=1)
    X_test = test_data
    
    # normalize data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # fit and train mpl classifier
    clf = MLPClassifier(hidden_layer_sizes=[100, 10], alpha=.001, random_state=0, solver='lbfgs').fit(X_train_scaled, y_train)
    
    # predict probabilities
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]
    
    # combine with reloaded test data
    test_df = pd.read_csv('test.csv', encoding = "ISO-8859-1")
    test_df['compliance'] = y_proba
    test_df.set_index('ticket_id', inplace=True)
    
    return test_df['compliance'] 


blight_model()
