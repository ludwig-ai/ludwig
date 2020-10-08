import sys
import requests
import pandas as pd

# Ludwig model server default values
LUDWIG_HOST = '0.0.0.0'
LUDWIG_PORT = '8000'


#
# retrieve data to make predictions
#
test_df = pd.read_csv('../titanic/data/test.csv')
print('retrieved {:d} records for predictions'.format(test_df.shape[0]))


#
# execute REST API /predict for a single record
#

# get a single record from dataframe and convert to list of dictionaries
prediction_request_dict_list = test_df.head(1).to_dict(orient='records')

# extract dictionary for the single record only
prediction_request_dict = prediction_request_dict_list[0]

print('single record for prediciton:\n', prediction_request_dict)

# construct URL
predict_url = ''.join(['http://', LUDWIG_HOST, ':', LUDWIG_PORT, '/predict'])

print('\ninvoking REST API /predict for single record...')
# connect using the default host address and port number
try:
    r = requests.post(
        predict_url,
        data=prediction_request_dict
    )
except requests.exceptions.ConnectionError as e:
    print(e)
    print("REST API /predict failed")
    sys.exit(1)


# check if REST API worked
if r.status_code == 200:
    # REST API successful
    # convert JSON response to panda dataframe
    pred_df = pd.read_json('[' + r.text + ']', orient='records')

    print('\nReceived {:d} predictions'.format(pred_df.shape[0]))
    print('Sample predictions:')
    print(pred_df.head())

else:
    # Error encountered during REST API processing
    print(
        '\nError during predictions, error code: ', r.status_code,
        'reason code: ', r.text
    )

#
# execute REST API /batch_predict on a pandas dataframe
#

# create json representation of dataset for REST API
prediction_request_json = test_df.to_json(orient='split')

print('\ninvoking REST API /batch_predict for entire dataframe...')

# construct URL
batch_predict_url = ''.join(['http://', LUDWIG_HOST, ':', LUDWIG_PORT,
                             '/batch_predict'])

# connect using the default host address and port number
response = requests.post(
    'http://0.0.0.0:8000/batch_predict',
    data={'dataset': prediction_request_json}
)
try:
    r = requests.post(
        batch_predict_url,
        data={'dataset': prediction_request_json}
    )
except requests.exceptions.ConnectionError as e:
    print(e)
    print("REST API /batch_predict failed")
    sys.exit(1)


# check if REST API worked
if response.status_code == 200:
    # REST API successful
    # convert JSON response to panda dataframe
    pred_df = pd.read_json(response.text, orient='split')

    print('\nReceived {:d} predictions'.format(pred_df.shape[0]))
    print('Sample predictions:')
    print(pred_df.head())

else:
    # Error encountered during REST API processing
    print(
        '\nError during predictions, error code: ', r.status_code,
        'reason code: ', r.text
    )
