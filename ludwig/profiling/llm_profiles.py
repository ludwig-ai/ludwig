import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, PreTrainedTokenizer

from ludwig.backend.base import LocalBackend
from ludwig.data.prompt import format_input_with_prompt
from ludwig.utils.llm_utils import set_pad_token


@dataclass
class HistogramBin:
    bin: str = ""
    count: int = 0


@dataclass
class SequenceLengthDistribution:
    min: int = 0
    p1: int = 0
    p5: int = 0
    p10: int = 0
    p25: int = 0
    p50: int = 0
    p75: int = 0
    p90: int = 0
    p95: int = 0
    p99: int = 0
    max: int = 0
    mean: float = 0
    stdev: float = 0
    histogram_bins: List[HistogramBin] = field(default_factory=list)


@dataclass
class SequenceLengthData:
    total_tokens: int
    token_counts: List[int]
    sequence_length_distribution: SequenceLengthDistribution


@dataclass
class SingleRecommendation:
    input_max_sequence_length: int = 0
    output_max_sequence_length: int = 0
    global_max_sequence_length: int = 0


@dataclass
class SequenceLengthRecommendation:
    # p50.
    reserved: SingleRecommendation
    # p95.
    gracious: SingleRecommendation
    # p100.
    max: SingleRecommendation


@dataclass
class DatasetSplitSequenceLengthData:
    tokenizer_name: str = ""
    tokenizer: PreTrainedTokenizer = None
    vocab_size: int = 0
    template: str = ""
    num_tokens_in_template: int = 0
    input_sequence_length_data: SequenceLengthData = None
    output_sequence_length_data: SequenceLengthData = None
    merged_sequence_length_data: SequenceLengthData = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    # total_accounted_tokens: int = 0
    total_tokens: int = 0
    num_examples: int = 0

    num_missing_outputs: int = 0
    percent_missing_outputs: float = 0
    num_missing_inputs: int = 0
    percent_missing_inputs: float = 0


@dataclass
class DatasetSequenceLengthData:
    split_sequence_length_data: Dict[str, DatasetSplitSequenceLengthData] = field(default_factory=dict)
    base_model: str = ""
    tokenizer_name: str = ""
    tokenizer: PreTrainedTokenizer = None
    vocab_size: int = 0
    warnings: List[str] = field(default_factory=list)
    unused_columns: Set[str] = field(default_factory=set)
    template: str = ""
    sequence_length_recommendation: SequenceLengthRecommendation = None
    num_examples: int = 0

    # Summed from all splits.
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    # total_accounted_tokens: int = 0
    total_tokens: int = 0


# AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# AutoTokenizer.from_pretrained(pretrained_model_name_or_path)


def get_sequence_length_recommendations(
    train_sequence_length_data: DatasetSplitSequenceLengthData,
    test_sequence_length_data: DatasetSplitSequenceLengthData,
) -> SequenceLengthRecommendation:
    pass


# @dataclass
# class SingleRecommendation:
#     input_max_sequence_length: int = 0
#     output_max_sequence_length: int = 0
#     global_max_sequence_length: int = 0


# @dataclass
# class SequenceLengthRecommendation:
#     # p50.
#     reserved: SingleRecommendation
#     # p95.
#     gracious: SingleRecommendation
#     # p100.
#     max: SingleRecommendation


def reject_outliers(data, m=2):
    """Returns the data points that are within 2 standard deviations of the median."""
    # From https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list.
    return data[abs(data - np.median(data)) < m * np.std(data)]


def get_sequence_length_distribution(list_of_num_tokens: List[int]) -> SequenceLengthDistribution:
    numbers = pd.Series(list_of_num_tokens)
    real_numbers = pd.Series(numbers.dropna().values)

    sequence_length_distribution = SequenceLengthDistribution()

    # Some items in the pandas_df.describe may be NaN or missing depending on the data.
    pd_description = real_numbers.describe([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.95, 0.99])
    if "mean" in pd_description and not pd.isna(pd_description["mean"]):
        sequence_length_distribution.mean = pd_description["mean"]
    if "std" in pd_description and not pd.isna(pd_description["std"]):
        sequence_length_distribution.stdev = pd_description["std"]
    if "min" in pd_description and not pd.isna(pd_description["min"]):
        sequence_length_distribution.min = pd_description["min"]
    if "max" in pd_description and not pd.isna(pd_description["max"]):
        sequence_length_distribution.max = pd_description["max"]
    if "1%" in pd_description and not pd.isna(pd_description["1%"]):
        sequence_length_distribution.p1 = pd_description["1%"]
    if "5%" in pd_description and not pd.isna(pd_description["5%"]):
        sequence_length_distribution.p5 = pd_description["5%"]
    if "10%" in pd_description and not pd.isna(pd_description["10%"]):
        sequence_length_distribution.p10 = pd_description["10%"]
    if "25%" in pd_description and not pd.isna(pd_description["25%"]):
        sequence_length_distribution.p25 = pd_description["25%"]
    if "50%" in pd_description and not pd.isna(pd_description["50%"]):
        sequence_length_distribution.median = pd_description["50%"]
    if "75%" in pd_description and not pd.isna(pd_description["75%"]):
        sequence_length_distribution.p75 = pd_description["75%"]
    if "95%" in pd_description and not pd.isna(pd_description["95%"]):
        sequence_length_distribution.p95 = pd_description["95%"]
    if "99%" in pd_description and not pd.isna(pd_description["99%"]):
        sequence_length_distribution.p99 = pd_description["99%"]

    # Add histogram bins.
    try:
        counts, bins = np.histogram(real_numbers, bins=30)
        for bin, count in zip(bins, counts):
            sequence_length_distribution.histogram_bins.append(HistogramBin(bin=str(bin), count=count))
    except TypeError:
        # If the numbers are all the same, the stdev is 0, and histogram will fail.
        pass

    # Compute percentage of outliers.
    try:
        without_outliers = reject_outliers(real_numbers)
        sequence_length_distribution.percentage_outliers = 1 - (len(without_outliers) / len(real_numbers))
    except TypeError:
        # If the numbers are all the same, the stdev is 0, and outlier calculation will fail.
        pass
    except Exception as e:
        logging.warning(f"Encountered error {e} in get_number_distribution() for {numbers}")
        pass

    return sequence_length_distribution


def get_sequence_length_data_for_tokens(tokens: List[int]) -> SequenceLengthData:
    total_tokens = sum([len(x) for x in tokens])
    token_counts = [len(x) for x in tokens]

    sequence_length_distribution = SequenceLengthData(
        total_tokens=total_tokens,
        token_counts=token_counts,
        sequence_length_distribution=get_sequence_length_distribution(token_counts),
    )
    return sequence_length_distribution


def merge_sublists(list1, list2):
    # Check if the lengths of the lists are the same
    if len(list1) != len(list2):
        return "Lists should be of the same length."
    # Initialize an empty list to store the merged sublists
    merged_list = []
    # Loop through the lists and merge each corresponding sublist
    for sublist1, sublist2 in zip(list1, list2):
        merged_sublist = sublist1 + sublist2
        merged_list.append(merged_sublist)
    return merged_list


def get_merged_sequence_length_data_for_tokens(input_sequence_length_data, output_sequence_length_data):
    return SequenceLengthData(
        total_tokens=input_sequence_length_data.total_tokens + output_sequence_length_data.total_tokens,
        # sequence_length_distribution=get_merged_sequence_length_distribution(
        #     input_sequence_length_data.sequence_length_distribution,
        #     output_sequence_length_data.sequence_length_distribution,
        # ),
    )


def remove_empty_strings(lst):
    return [item for item in lst if item != ""]


def get_sequence_length_data_for_split(
    data: pd.DataFrame, output_column_name: str, tokenizer: PreTrainedTokenizer, template: str
) -> DatasetSplitSequenceLengthData:
    # Check missing data.
    data_no_na = data.dropna(subset=[output_column_name])
    num_missing_outputs = len(data_no_na) - len(data)
    percent_missing_outputs = num_missing_outputs / len(data)

    num_missing_outputs = len(data) - len(data[output_column_name].dropna())
    percent_missing_outputs = num_missing_outputs / len(code_alpaca_data)

    data = data_no_na

    realized_strings: List[str] = (
        format_input_with_prompt(
            input_col_name="na",
            dataset_df=data,
            backend=LocalBackend(),
            task_str=None,
            template=template,
        )
        .dropna()
        .tolist()
    )
    output_strings: List[str] = data[output_column_name].dropna().tolist()

    input_tokens_with_na = tokenizer.batch_encode_plus(realized_strings)
    output_tokens_with_na = tokenizer.batch_encode_plus(output_strings)
    merged_tokens = merge_sublists(input_tokens_with_na.input_ids, output_tokens_with_na.input_ids)

    # Remove empty strings as these will cause problems for pandas data analysis.
    # realized_strings: List[str] = remove_empty_strings(realized_strings)
    realized_strings = (
        format_input_with_prompt(
            input_col_name="na",
            dataset_df=data,
            backend=LocalBackend(),
            task_str=None,
            template=template,
        )
        .dropna()
        .tolist()
    )
    # output_strings: List[str] = remove_empty_strings(output_strings)
    output_strings = data[output_column_name].dropna().tolist()

    input_tokens = tokenizer.batch_encode_plus(realized_strings)
    output_tokens = tokenizer.batch_encode_plus(output_strings)

    sequence_length_data = DatasetSplitSequenceLengthData()
    sequence_length_data.tokenizer_name = tokenizer.name_or_path
    sequence_length_data.vocab_size = tokenizer.vocab_size
    sequence_length_data.tokenizer = tokenizer
    sequence_length_data.template = template
    sequence_length_data.num_tokens_in_template = len(tokenizer.encode(template))

    sequence_length_data.input_sequence_length_data = get_sequence_length_data_for_tokens(input_tokens.input_ids)
    sequence_length_data.output_sequence_length_data = get_sequence_length_data_for_tokens(output_tokens.input_ids)
    sequence_length_data.merged_sequence_length_data = get_sequence_length_data_for_tokens(merged_tokens)
    # sequence_length_data.merged_sequence_length_data = get_merged_sequence_length_data_for_tokens(
    #     sequence_length_data.input_sequence_length_data, sequence_length_data.output_sequence_length_data
    # )

    sequence_length_data.total_input_tokens = sequence_length_data.input_sequence_length_data.total_tokens
    sequence_length_data.total_output_tokens = sequence_length_data.output_sequence_length_data.total_tokens
    sequence_length_data.total_tokens = sequence_length_data.merged_sequence_length_data.total_tokens

    # TODO: Check that this works? We should remove extra padding tokens.
    # sequence_length_data.total_accounted_tokens = (
    #     sequence_length_data.total_input_tokens + sequence_length_data.total_output_tokens
    # )

    sequence_length_data.num_missing_outputs = num_missing_outputs
    sequence_length_data.percent_missing_outputs = percent_missing_outputs
    # sequence_length_data.num_missing_inputs = num_missing_inputs
    # sequence_length_data.percent_missing_inputs = percent_missing_inputs

    return sequence_length_data


def get_sequence_length_data(
    data: pd.DataFrame, output_column_name: str, base_model: str, template: str
) -> DatasetSequenceLengthData:
    train, test = train_test_split(data, test_size=0.1, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    set_pad_token(tokenizer)

    dataset_sequence_length_data = DatasetSequenceLengthData(
        split_sequence_length_data={
            "train": get_sequence_length_data_for_split(train, output_column_name, tokenizer, template),
            "test": get_sequence_length_data_for_split(train, output_column_name, tokenizer, template),
        },
        base_model=base_model,
        tokenizer_name=tokenizer.name_or_path,
        tokenizer=tokenizer,
        vocab_size=tokenizer.vocab_size,
        template=template,
        num_examples=len(data),
    )

    # Sum up total tokens used across all splits.
    for split_sequence_length_data in dataset_sequence_length_data.split_sequence_length_data.values():
        dataset_sequence_length_data.total_input_tokens += split_sequence_length_data.total_input_tokens
        dataset_sequence_length_data.total_output_tokens += split_sequence_length_data.total_output_tokens
        # dataset_sequence_length_data.total_accounted_tokens += split_sequence_length_data.total_accounted_tokens
        dataset_sequence_length_data.total_tokens += split_sequence_length_data.total_tokens

    # TODO:
    # warnings: List[str]
    # unused_columns: Set[str]
    # sequence_length_recommendation: SequenceLengthRecommendation

    return dataset_sequence_length_data


# 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
# if __name__ == "__main__":
def test_sequence_length_data():
    # logging.basicConfig(level=logging.INFO)
    code_alpaca_data = pd.read_csv("code_alpaca_20k.csv")

    template = """Below is an instruction that describes a task, paired with an input
    that provides further context. Write a response that appropriately
    completes the request.

    ### Instruction: {instruction}

    ### Input: {input}

    ### Response:
    """

    # Fun-fact -- there are 7 examples that are empty!
    # output_strings = code_alpaca_data["output"].dropna().tolist()

    sequence_length_data = get_sequence_length_data(code_alpaca_data, "output", "meta-llama/Llama-2-7b-hf", template)

    # print(sequence_length_data.num_examples)
    # print(sequence_length_data.total_input_tokens)
    # print(sequence_length_data.total_output_tokens)
    # print(sequence_length_data.total_tokens)

    if sequence_length_data.percent_missing_outputs > 0:
        print(
            f"Some data is missing: {sequence_length_data.num_missing_outputs} examples, which is "
            f"{sequence_length_data.percent_missing_outputs*100:.2f}% of the dataset."
        )

    # Recommended sequence lengths.

    # Warnings:
    # - % missing values. (easy)
    # - target distribution falls outside of reference distribution
    # - target distribution mean is more than 1 reference distribution’s stdev away from reference distribution’s mean
    # - aspirational max sequence length chops off too much data (% tokens)
    # - aspirational global max sequence length chops off too much data (% tokens)
    # - input sequence length + output sequence length is smaller than global max sequence length

    # print(get_sequence_length_data(code_alpaca_data, "output", "meta-llama/Llama-2-7b-hf", template))


if __name__ == "__main__":
    code_alpaca_data = pd.read_csv("code_alpaca_20k.csv")

    template = """Below is an instruction that describes a task, paired with an input
    that provides further context. Write a response that appropriately
    completes the request.

    ### Instruction: {instruction}

    ### Input: {input}

    ### Response:
    """

    # Fun-fact -- there are 7 examples that are empty!
    # output_strings = code_alpaca_data["output"].dropna().tolist()

    # sequence_length_data = get_sequence_length_data(code_alpaca_data, "output", "meta-llama/Llama-2-7b-hf", template)

    st.set_page_config(
        page_title="LLM Dataset Profiler",
    )
    # st.write(css, unsafe_allow_html=True)

    st.markdown(
        """
            <h1 style="text-align:center;">LLM Dataset Profiler</h1>
            <p style="text-align:center;">
            <img src="https://app.predibase.com/logos/predibase/predibase.svg" width="25" />
            Powered by <a href="https://predibase.com">Predibase</a>
            </p>
            """,
        unsafe_allow_html=True,
    )

    if "llm_dataset_sequence_length_data" not in st.session_state:
        st.session_state.llm_dataset_sequence_length_data = None

    st.subheader("Your dataset")
    docs = st.file_uploader("Upload your data and click on 'Process'")

    if st.button("Process"):
        with st.spinner("Processing"):
            start_time = time.time()

            # Get the text chunks.
            st.session_state.llm_dataset_sequence_length_data: DatasetSequenceLengthData = get_sequence_length_data(
                code_alpaca_data, "output", "meta-llama/Llama-2-7b-hf", template
            )

            end_time = time.time()

            st.success(f"Profiling took {(end_time - start_time):.1f} seconds.")

            st.write(code_alpaca_data)

            with st.expander("Document statistics"):
                # Distribution of number of pages.
                fig1, ax = plt.subplots()
                # num_pages_array = np.array(list(doc_id_to_num_pages.values()))
                ax.hist(
                    np.array([len(document.page_content) for document in st.session_state.document_chunks]),
                    bins=20,
                )
                st.write("Distribution of chunk sizes")
                st.pyplot(fig1)

                # Distribution of length of texts.
                fig2, ax = plt.subplots()
                ax.hist(
                    np.array([len(document.page_content) for document in st.session_state.documents]),
                    bins=20,
                )
                st.write("Distribution of document lengths (chars)")
                st.pyplot(fig2)
