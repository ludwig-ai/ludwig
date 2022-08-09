import json
import os
import tempfile
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
from starlette.datastructures import UploadFile
from starlette.responses import JSONResponse

from ludwig.utils.data_utils import NumpyEncoder


def serialize_payload(data_source: Union[pd.DataFrame, pd.Series]) -> tuple:
    """
    Generates two dictionaries to be sent via REST API for Ludwig prediction
    service.
    First dictionary created is payload_dict. Keys found in payload_dict:
    raw_data: this is json string created by pandas to_json() method
    source_type: indicates if the data_source is either a pandas dataframe or
        pandas series.  This is needed to know how to rebuild the structure.
    ndarray_dtype:  this is a dictionary where each entry is for any ndarray
        data found in the data_source.  This could be an empty dictioinary if no
        ndarray objects are present in data_source. Key for this dictionary is
        column name if data_source is dataframe or index name if data_source is
        series.  The value portion of the dictionary is the dtype of the
        ndarray.  This value is used to set the correct dtype when rebuilding
        the entry.

    Second dictionary created is called payload_files, this contains information
    and content for files to be sent to the server.  NOTE: if no files are to be
    sent, this will be an empty dictionary.
    Entries in this dictionary:
    Key: file path string for file to be sent to server
    Value: tuple(file path string, byte encoded file content,
                 'application/octet-stream')

    Args:
        data_source: input features to be sent to Ludwig server

    Returns: tuple(payload_dict, payload_files)

    """
    payload_dict = {}
    payload_dict["ndarray_dtype"] = {}
    payload_files = {}
    if isinstance(data_source, pd.DataFrame):
        payload_dict["raw_data"] = data_source.to_json(orient="columns")
        payload_dict["source_type"] = "dataframe"
        for col in data_source.columns:
            if isinstance(data_source[col].iloc[0], np.ndarray):
                # if we have any ndarray columns, record dtype
                payload_dict["ndarray_dtype"][col] = str(data_source[col].iloc[0].dtype)
            elif isinstance(data_source[col].iloc[0], str) and os.path.exists(data_source[col].iloc[0]):
                # if we have file path feature, prepare file for transport
                for v in data_source[col]:
                    payload_files[v] = (v, open(v, "rb"), "application/octet-stream")
    elif isinstance(data_source, pd.Series):
        payload_dict["raw_data"] = data_source.to_json(orient="index")
        payload_dict["source_type"] = "series"
        for col in data_source.index:
            if isinstance(data_source[col], np.ndarray):
                # for ndarrays record dtype for reconstruction
                payload_dict["ndarray_dtype"][col] = str(data_source[col].dtype)
            elif isinstance(data_source[col], str) and os.path.exists(data_source[col]):
                # if we have file path feature, prepare file for transport
                v = data_source[col]
                payload_files[v] = (v, open(v, "rb"), "application/octet-stream")
    else:
        ValueError(
            '"data_source" must be either a pandas DataFrame or Series, '
            "format found to be {}".format(type(data_source))
        )

    return payload_dict, payload_files


def _write_file(v, files):
    # Convert UploadFile to a NamedTemporaryFile to ensure it's on the disk
    suffix = os.path.splitext(v.filename)[1]
    named_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    files.append(named_file)
    named_file.write(v.file.read())
    named_file.close()
    return named_file.name


def deserialize_payload(json_string: str) -> pd.DataFrame:
    """This function performs the inverse of the serialize_payload function and rebuilds the object represented in
    json_string to a pandas DataFrame.

    Args:
        json_string: representing object to be rebuilt.

    Returns: pandas.DataFrame
    """
    payload_dict = json.loads(json_string)

    # extract raw data from json string
    raw_data_dict = json.loads(payload_dict["raw_data"])
    # rebuild based on original data source
    if payload_dict["source_type"] == "dataframe":
        # reconstitute the pandas dataframe
        df = pd.DataFrame.from_dict(raw_data_dict, orient="columns")
    elif payload_dict["source_type"] == "series":
        # reconstitute series into single row dataframe
        df = pd.DataFrame(pd.Series(raw_data_dict)).T
    else:
        ValueError(
            'Unknown "source_type" found.  Valid values are "dataframe" or '
            '"series".  Instead found {}'.format(payload_dict["source_type"])
        )

    # if source has ndarrays, rebuild those from list and set
    # original dtype.
    if payload_dict["ndarray_dtype"]:
        # yes, now covert list representation to ndarray representation
        for col in payload_dict["ndarray_dtype"]:
            dtype = payload_dict["ndarray_dtype"][col]
            df[col] = df[col].apply(lambda x: np.array(x).astype(dtype))

    return df


def deserialize_request(form) -> tuple:
    """This function will deserialize the REST API request packet to create a pandas dataframe that is input to the
    Ludwig predict method and a list of files that will be cleaned up at the end of processing.

    Args:
        form: REST API provide form data

    Returns: tuple(pandas.DataFrame, list of temporary files to clean up)
    """
    files = []
    file_index = {}
    for k, v in form.multi_items():
        if type(v) == UploadFile:
            file_index[v.filename] = _write_file(v, files)

    # reconstruct the dataframe
    df = deserialize_payload(form["payload"])

    # insert files paths of the temporary files in place of the original
    # file paths specified by the user.
    # pd.DataFrame.replace() method is used to replace file path string
    # specified by the user context with the file path string where a
    # temporary file containing the same content.
    # parameters for replace() method:
    #   to_replace: list of file path strings that the user provided
    #   value: list of temporary files created for each input file
    #
    # IMPORTANT: There is a one-to-one correspondence of the to_replace list
    # and the value list. Each list must be the same size.
    df.replace(to_replace=list(file_index.keys()), value=list(file_index.values()), inplace=True)

    return df, files


class NumpyJSONResponse(JSONResponse):
    def render(self, content: Dict[str, Any]) -> str:
        """Override the default JSONResponse behavior to encode numpy arrays.

        Args:
            content: JSON object to be serialized.

        Returns: str
        """
        return json.dumps(
            content, ensure_ascii=False, allow_nan=False, indent=None, separators=(",", ":"), cls=NumpyEncoder
        ).encode("utf-8")
