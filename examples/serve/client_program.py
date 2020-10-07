import requests
import pandas as pd

# retrieve data to make predictions
test_df = pd.read_csv('../titanic/data/test.csv')
print('retrieved {:d} records for predictions'.format(test_df.shape[0]))

# create json representation of dataset for REST API
prediction_request_json = test_df.to_json(orient='split')

# execute REST API /batch_predict
print('\ninvoking REST API /batch_predict...')
# connect using the default host address and port number
r = requests.post(
    'http://0.0.0.0:8000/batch_predict',
    data={'dataset': prediction_request_json}
)

# check if REST API worked
if r.status_code == 200:
    # REST API successful
    # convert JSON response to panda dataframe
    pred_df = pd.read_json(r.text, orient='split')

    print('\nReceived {:d} predictions'.format(pred_df.shape[0]))
    print('Sample predictions:')
    print(pred_df.head())

else:
    # Error encountered during REST API processing
    print(
        '\nError during predictions, error code: ', r.status_code,
        'reason code: ', r.text
    )
