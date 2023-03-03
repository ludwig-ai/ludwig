import yaml
from ludwig.api import LudwigModel
import ray

import dask.dataframe as dd
import pandas as pd

# Should result in a split of 21, 5, 4 which ensures each split has > 3 rows.
df = pd.read_csv("melbourne_airbnb_split.csv", nrows=30)
ddf = dd.from_pandas(df, npartitions=1)

config = yaml.safe_load(
    """
input_features:
  - name: last_scraped
    type: category
    column: last_scraped
  - name: name
    type: text
    column: name
  - name: summary
    type: text
    column: summary
  - name: space
    type: text
    column: space
  - name: description
    type: text
    column: description
  - name: neighborhood_overview
    type: text
    column: neighborhood_overview
  - name: notes
    type: text
    column: notes
  - name: transit
    type: text
    column: transit
  - name: access
    type: text
    column: access
  - name: interaction
    type: text
    column: interaction
  - name: house_rules
    type: text
    column: house_rules
  - name: picture_url
    type: image
    column: picture_url
  - name: host_id
    type: number
    column: host_id
  - name: host_name
    type: category
    column: host_name
  - name: host_since
    type: category
    column: host_since
  - name: host_location
    type: category
    column: host_location
  - name: host_about
    type: text
    column: host_about
  - name: host_response_time
    type: category
    column: host_response_time
  - name: host_response_rate
    type: category
    column: host_response_rate
  - name: host_is_superhost
    type: category
    column: host_is_superhost
  - name: host_thumbnail_url
    type: image
    column: host_thumbnail_url
  - name: host_picture_url
    type: image
    column: host_picture_url
  - name: host_neighborhood
    type: category
    column: host_neighborhood
  - name: host_verifications
    type: category
    column: host_verifications
  - name: host_has_profile_pic
    type: category
    column: host_has_profile_pic
  - name: host_identity_verified
    type: category
    column: host_identity_verified
  - name: street
    type: category
    column: street
  - name: neighborhood
    type: category
    column: neighborhood
  - name: city
    type: category
    column: city
  - name: suburb
    type: category
    column: suburb
  - name: state
    type: category
    column: state
  - name: zipcode
    type: number
    column: zipcode
  - name: smart_location
    type: category
    column: smart_location
  - name: latitude
    type: number
    column: latitude
  - name: longitude
    type: number
    column: longitude
  - name: is_location_exact
    type: binary
    column: is_location_exact
  - name: property_type
    type: category
    column: property_type
  - name: room_type
    type: category
    column: room_type
  - name: accommodates
    type: category
    column: accommodates
  - name: bathrooms
    type: number
    column: bathrooms
  - name: bedrooms
    type: number
    column: bedrooms
  - name: beds
    type: number
    column: beds
  - name: bed_type
    type: category
    column: bed_type
  - name: amenities
    type: text
    column: amenities
  - name: price
    type: number
    column: price
  - name: weekly_price
    type: number
    column: weekly_price
  - name: monthly_price
    type: number
    column: monthly_price
  - name: security_deposit
    type: number
    column: security_deposit
  - name: cleaning_fee
    type: number
    column: cleaning_fee
  - name: guests_included
    type: number
    column: guests_included
  - name: extra_people
    type: number
    column: extra_people
  - name: minimum_nights
    type: number
    column: minimum_nights
  - name: maximum_nights
    type: number
    column: maximum_nights
  - name: calendar_updated
    type: category
    column: calendar_updated
  - name: availability_30
    type: number
    column: availability_30
  - name: availability_60
    type: number
    column: availability_60
  - name: availability_90
    type: number
    column: availability_90
  - name: availability_365
    type: number
    column: availability_365
  - name: calendar_last_scraped
    type: category
    column: calendar_last_scraped
  - name: number_of_reviews
    type: number
    column: number_of_reviews
  - name: first_review
    type: category
    column: first_review
  - name: last_review
    type: category
    column: last_review
  - name: review_scores_rating
    type: number
    column: review_scores_rating
  - name: review_scores_accuracy
    type: number
    column: review_scores_accuracy
  - name: review_scores_cleanliness
    type: category
    column: review_scores_cleanliness
  - name: review_scores_checkin
    type: category
    column: review_scores_checkin
  - name: review_scores_communication
    type: category
    column: review_scores_communication
  - name: review_scores_location
    type: number
    column: review_scores_location
  - name: review_scores_value
    type: category
    column: review_scores_value
  - name: license
    type: category
    column: license
  - name: instant_bookable
    type: binary
    column: instant_bookable
  - name: cancellation_policy
    type: category
    column: cancellation_policy
  - name: require_guest_profile_picture
    type: binary
    column: require_guest_profile_picture
  - name: require_guest_phone_verification
    type: binary
    column: require_guest_phone_verification
  - name: calculated_host_listings_count
    type: number
    column: calculated_host_listings_count
  - name: reviews_per_month
    type: number
    column: reviews_per_month
  - name: host_verifications_jumio
    type: binary
    column: host_verifications_jumio
  - name: host_verifications_government_id
    type: binary
    column: host_verifications_government_id
  - name: host_verifications_kba
    type: binary
    column: host_verifications_kba
  - name: host_verifications_zhima_selfie
    type: binary
    column: host_verifications_zhima_selfie
  - name: host_verifications_facebook
    type: binary
    column: host_verifications_facebook
  - name: host_verifications_work_email
    type: binary
    column: host_verifications_work_email
  - name: host_verifications_google
    type: binary
    column: host_verifications_google
  - name: host_verifications_sesame
    type: binary
    column: host_verifications_sesame
  - name: host_verifications_manual_online
    type: binary
    column: host_verifications_manual_online
  - name: host_verifications_manual_offline
    type: binary
    column: host_verifications_manual_offline
  - name: host_verifications_offline_government_id
    type: binary
    column: host_verifications_offline_government_id
  - name: host_verifications_selfie
    type: binary
    column: host_verifications_selfie
  - name: host_verifications_reviews
    type: binary
    column: host_verifications_reviews
  - name: host_verifications_identity_manual
    type: binary
    column: host_verifications_identity_manual
  - name: host_verifications_sesame_offline
    type: binary
    column: host_verifications_sesame_offline
  - name: host_verifications_weibo
    type: binary
    column: host_verifications_weibo
  - name: host_verifications_email
    type: binary
    column: host_verifications_email
  - name: host_verifications_sent_id
    type: binary
    column: host_verifications_sent_id
output_features:
  - name: price_label
    type: category
    column: price_label
trainer:
  train_steps: 2
preprocessing:
  split:
    type: fixed
    """
)

#   - name: host_verifications_phone
#     type: binary
#     column: host_verifications_phone

model = LudwigModel(config, backend="ray")
model.train(dataset=ddf)

ray.shutdown()
