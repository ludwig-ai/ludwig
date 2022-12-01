#! /usr/bin/env python
# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from datetime import date

from ludwig.api_annotations import DeveloperAPI


@DeveloperAPI
def create_vector_from_datetime_obj(datetime_obj):
    yearday = datetime_obj.toordinal() - date(datetime_obj.year, 1, 1).toordinal() + 1

    midnight = datetime_obj.replace(hour=0, minute=0, second=0, microsecond=0)
    second_of_day = (datetime_obj - midnight).seconds

    return [
        datetime_obj.year,
        datetime_obj.month,
        datetime_obj.day,
        datetime_obj.weekday(),
        yearday,
        datetime_obj.hour,
        datetime_obj.minute,
        datetime_obj.second,
        second_of_day,
    ]
