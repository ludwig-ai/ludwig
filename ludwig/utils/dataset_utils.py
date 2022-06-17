import pandas as pd
from sklearn.model_selection import train_test_split

from ludwig.constants import TEST_SPLIT, TRAIN_SPLIT, VALIDATION_SPLIT


def get_repeatable_train_val_test_split(
    df_input, stratify_colname, random_seed, frac_train=0.7, frac_val=0.1, frac_test=0.2
):
    """Return df_input with split column containing (if possible) non-zero rows in the train split, validation
    split, and test split categories.

    If the input dataframe does not contain an existing split column or if the
    number of rows in both the validation and test split is 0, return df_input
    with split column set according to frac_<type> and stratify_colname.

    Else stratify_colname is ignored, and:
     If the input dataframe contains an existing split column and non-zero row
      counts for all three split types, return df_input.
     If the input dataframe contains an existing split column but only one of
      validation and test split has non-zero row counts, return df_input with
      missing split getting rows from train split as per frac_<type>.

    Parameters
    ----------
    df_input : Pandas dataframe
        Input dataframe to be split.
    stratify_colname : str
        The column used for stratification; usually the label column.
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
        raise ValueError("fractions %f, %f, %f do not add up to 1.0" % (frac_train, frac_val, frac_test))
    if stratify_colname not in df_input.columns:
        raise ValueError("%s is not a column in the dataframe" % (stratify_colname))

    do_stratify_split = True
    if "split" in df_input.columns:
        df_train = df_input[df_input["split"] == TRAIN_SPLIT]
        df_val = df_input[df_input["split"] == VALIDATION_SPLIT]
        df_test = df_input[df_input["split"] == TEST_SPLIT]
        if len(df_val) != 0 or len(df_test) != 0:
            if len(df_val) == 0:
                df_val = df_train.sample(frac=frac_val, replace=False, random_state=random_seed)
                df_train = df_train.drop(df_val.index)
            if len(df_test) == 0:
                df_test = df_train.sample(frac=frac_test, replace=False, random_state=random_seed)
                df_train = df_train.drop(df_test.index)
            do_stratify_split = False

    if do_stratify_split:
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
