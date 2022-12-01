from typing import List, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from ludwig.api_annotations import PublicAPI
from ludwig.constants import TEST_SPLIT, TRAIN_SPLIT, VALIDATION_SPLIT
from ludwig.data.dataset.base import Dataset
from ludwig.utils.defaults import default_random_seed


@PublicAPI
def get_repeatable_train_val_test_split(
    df_input, stratify_colname="", random_seed=default_random_seed, frac_train=0.7, frac_val=0.1, frac_test=0.2
):
    """Return df_input with split column containing (if possible) non-zero rows in the train, validation, and test
    data subset categories.

    If the input dataframe does not contain an existing split column or if the
    number of rows in both the validation and test split is 0 and non-empty
    stratify_colname specified, return df_input with split column set according
    to frac_<subset_name> and stratify_colname.

    Else stratify_colname is ignored, and:
     If the input dataframe contains an existing split column and non-zero row
      counts for all three split types, return df_input.
     If the input dataframe contains an existing split column but only one of
      validation and test split has non-zero row counts, return df_input with
      missing split getting rows from train split as per frac_<subset_name>.

    Parameters
    ----------
    df_input : Pandas dataframe
        Input dataframe to be split.
    stratify_colname : str
        The column used for stratification (if desired); usually the label column.
    random_seed : int
        Seed used to get repeatable split.
    frac_train : float
    frac_val   : float
    frac_test  : float
        The ratios with which to split the dataframe into train, val, and test data;
        should sum to 1.0.

    Returns
    -------
    df_split :
        Dataframe containing the three splits.
    """

    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError(f"fractions {frac_train:f}, {frac_val:f}, {frac_test:f} do not add up to 1.0")
    if stratify_colname:
        do_stratify_split = True
        if stratify_colname not in df_input.columns:
            raise ValueError("%s is not a column in the dataframe" % (stratify_colname))
    else:
        do_stratify_split = False
        if "split" not in df_input.columns:
            df_input["split"] = 0  # set up for non-stratified split path

    if "split" in df_input.columns:
        df_train = df_input[df_input["split"] == TRAIN_SPLIT].copy()
        df_val = df_input[df_input["split"] == VALIDATION_SPLIT].copy()
        df_test = df_input[df_input["split"] == TEST_SPLIT].copy()
        if not do_stratify_split or len(df_val) != 0 or len(df_test) != 0:
            if len(df_val) == 0:
                df_val = df_train.sample(frac=frac_val, replace=False, random_state=random_seed)
                df_train = df_train.drop(df_val.index)
            if len(df_test) == 0:
                df_test = df_train.sample(frac=frac_test, replace=False, random_state=random_seed)
                df_train = df_train.drop(df_test.index)
            do_stratify_split = False

    if do_stratify_split:
        # Make sure the `stratify_colname` doesn't have any NaNs.
        df_input = df_input[df_input[stratify_colname].notna()]

        # Split original dataframe into train and temp dataframes.
        y = df_input[[stratify_colname]]  # Dataframe of just the column on which to stratify.
        df_train, df_temp, y_train, y_temp = train_test_split(
            df_input, y, stratify=y, test_size=(1.0 - frac_train), random_state=random_seed
        )
        # Split the temp dataframe into val and test dataframes.
        relative_frac_test = frac_test / (frac_val + frac_test)
        df_val, df_test, y_val, y_test = train_test_split(
            df_temp, y_temp, stratify=y_temp, test_size=relative_frac_test, random_state=random_seed
        )

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)
    df_train["split"] = TRAIN_SPLIT
    df_val["split"] = VALIDATION_SPLIT
    df_test["split"] = TEST_SPLIT
    df_split = pd.concat([df_train, df_val, df_test], ignore_index=True)
    return df_split


def generate_dataset_statistics(
    training_set: Dataset, validation_set: Union[Dataset, None], test_set: Union[Dataset, None]
) -> List[Tuple[str, int, int]]:
    from ludwig.benchmarking.utils import format_memory

    dataset_statistics = [["Dataset", "Size (Rows)", "Size (In Memory)"]]
    dataset_statistics.append(["Training", len(training_set), format_memory(training_set.in_memory_size_bytes)])
    if validation_set is not None:
        dataset_statistics.append(
            ["Validation", len(validation_set), format_memory(validation_set.in_memory_size_bytes)]
        )
    if test_set is not None:
        dataset_statistics.append(["Test", len(test_set), format_memory(test_set.in_memory_size_bytes)])
    return dataset_statistics
