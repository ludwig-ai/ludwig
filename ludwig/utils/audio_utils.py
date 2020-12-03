#! /usr/bin/env python
# coding=utf-8
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
import numpy as np
from scipy.signal import lfilter
from scipy.signal.windows import get_window


def _pre_emphasize_data(data, emphasize_value=0.97):
    filter_window = np.asarray([1, -emphasize_value])
    pre_emphasized_data = lfilter(filter_window, 1, data)
    return pre_emphasized_data


def get_length_in_samp(sampling_rate_in_hz, length_in_s):
    return int(sampling_rate_in_hz * length_in_s)


def get_group_delay(raw_data, sampling_rate_in_hz, window_length_in_s,
                    window_shift_in_s, num_fft_points, window_type):
    X_stft_transform = _get_stft(raw_data, sampling_rate_in_hz,
                                 window_length_in_s, window_shift_in_s,
                                 num_fft_points, window_type=window_type)
    Y_stft_transform = _get_stft(raw_data, sampling_rate_in_hz,
                                 window_length_in_s, window_shift_in_s,
                                 num_fft_points, window_type=window_type,
                                 data_transformation='group_delay')
    X_stft_transform_real = np.real(X_stft_transform)
    X_stft_transform_imag = np.imag(X_stft_transform)
    Y_stft_transform_real = np.real(Y_stft_transform)
    Y_stft_transform_imag = np.imag(Y_stft_transform)
    nominator = np.multiply(X_stft_transform_real,
                            Y_stft_transform_real) + np.multiply(
        X_stft_transform_imag, Y_stft_transform_imag)
    denominator = np.square(np.abs(X_stft_transform))
    group_delay = np.divide(nominator, denominator + 1e-10)
    assert not np.isnan(
        group_delay).any(), 'There are NaN values in group delay'
    return np.transpose(group_delay)


def get_phase_stft_magnitude(raw_data, sampling_rate_in_hz, window_length_in_s,
                             window_shift_in_s, num_fft_points, window_type):
    stft = _get_stft(raw_data, sampling_rate_in_hz, window_length_in_s,
                     window_shift_in_s, num_fft_points,
                     window_type=window_type)
    abs_stft = np.abs(stft)
    phase = np.angle(stft)
    stft_phase = np.concatenate((phase, abs_stft), axis=1)
    return np.transpose(stft_phase)


def get_stft_magnitude(raw_data, sampling_rate_in_hz, window_length_in_s,
                       window_shift_in_s, num_fft_points, window_type):
    stft = _get_stft(raw_data, sampling_rate_in_hz, window_length_in_s,
                     window_shift_in_s, num_fft_points,
                     window_type=window_type)
    stft_magnitude = np.abs(stft)
    return np.transpose(stft_magnitude)


################################################################################
# The following code for FBank is adapted from jameslyons/python_speech_features
# MIT licensed implementation
# https://github.com/jameslyons/python_speech_features/blob/40c590269b57c64a8c1f1ddaaff2162008d1850c/python_speech_features/base.py#L84################################################################################
################################################################################
def get_fbank(raw_data, sampling_rate_in_hz, window_length_in_s,
              window_shift_in_s, num_fft_points, window_type,
              num_filter_bands):
    stft = _get_stft(raw_data, sampling_rate_in_hz, window_length_in_s,
                     window_shift_in_s, num_fft_points,
                     window_type=window_type,
                     zero_mean_offset=True)
    stft_power = np.abs(stft) ** 2
    upper_limit_freq = int(sampling_rate_in_hz / 2)
    upper_limit_mel = _convert_hz_to_mel(upper_limit_freq)
    lower_limit_mel = 0
    list_mel_points = np.linspace(lower_limit_mel, upper_limit_mel,
                                  num_filter_bands + 2)
    mel_fbank_matrix = _get_mel_fbank_matrix(list_mel_points, num_filter_bands,
                                             num_fft_points,
                                             sampling_rate_in_hz)
    mel_fbank_feature = np.dot(stft_power, np.transpose(mel_fbank_matrix))
    log_mel_fbank_feature = np.log(mel_fbank_feature + 1.0e-10)
    return np.transpose(log_mel_fbank_feature)


def _get_mel_fbank_matrix(list_mel_points, num_filter_bands, num_fft_points,
                          sampling_rate_in_hz):
    num_ess_fft_points = get_non_symmetric_length(num_fft_points)
    freq_scale = (num_fft_points + 1) / sampling_rate_in_hz
    freq_bins_on_mel_scale = np.floor(
        freq_scale * _convert_mel_to_hz(list_mel_points))
    mel_scaled_fbank = np.zeros((num_filter_bands, num_ess_fft_points),
                                dtype=np.float32)
    for filt_idx in range(num_filter_bands):
        start_bin_freq = freq_bins_on_mel_scale[filt_idx]
        middle_bin_freq = freq_bins_on_mel_scale[filt_idx + 1]
        end_bin_freq = freq_bins_on_mel_scale[filt_idx + 2]
        mel_scaled_fbank[filt_idx] = _create_triangular_filter(start_bin_freq,
                                                               middle_bin_freq,
                                                               end_bin_freq,
                                                               num_ess_fft_points)
    return mel_scaled_fbank


def _create_triangular_filter(start_bin_freq, middle_bin_freq, end_bin_freq,
                              num_ess_fft_points):
    filter_window = np.zeros(num_ess_fft_points, dtype=np.float32)
    filt_support_begin = middle_bin_freq - start_bin_freq
    filt_support_end = end_bin_freq - middle_bin_freq
    for freq in range(int(start_bin_freq), int(middle_bin_freq)):
        filter_window[freq] = (freq - start_bin_freq) / filt_support_begin
    for freq in range(int(middle_bin_freq), int(end_bin_freq)):
        filter_window[freq] = (end_bin_freq - freq) / filt_support_end
    return filter_window


def _convert_hz_to_mel(hz):
    return 2595.0 * np.log10(1 + hz / 700.0)


def _convert_mel_to_hz(mel):
    return 700.0 * (10 ** (mel / 2595.0) - 1)


def _get_stft(raw_data, sampling_rate_in_hz, window_length_in_s,
              window_shift_in_s, num_fft_points, window_type,
              data_transformation=None, zero_mean_offset=False):
    pre_emphasized_data = _pre_emphasize_data(raw_data)
    stft = _short_time_fourier_transform(pre_emphasized_data,
                                         sampling_rate_in_hz,
                                         window_length_in_s, window_shift_in_s,
                                         num_fft_points, window_type,
                                         data_transformation, zero_mean_offset)
    non_symmetric_stft = get_non_symmetric_data(stft)
    return non_symmetric_stft


def _short_time_fourier_transform(data, sampling_rate_in_hz,
                                  window_length_in_s,
                                  window_shift_in_s, num_fft_points,
                                  window_type, data_transformation=None,
                                  zero_mean_offset=False):
    window_length_in_samp = get_length_in_samp(window_length_in_s,
                                               sampling_rate_in_hz)
    window_shift_in_samp = get_length_in_samp(window_shift_in_s,
                                              sampling_rate_in_hz)
    preprocessed_data_matrix = _preprocess_to_padded_matrix(data,
                                                            window_length_in_samp,
                                                            window_shift_in_samp,
                                                            zero_mean_offset=zero_mean_offset)
    weighted_data_matrix = _weight_data_matrix(
        preprocessed_data_matrix,
        window_type,
        data_transformation=data_transformation
    )
    fft = np.fft.fft(weighted_data_matrix, n=num_fft_points)
    return fft


def _preprocess_to_padded_matrix(data, window_length_in_samp,
                                 window_shift_in_samp, zero_mean_offset=False):
    num_input = data.shape[0]
    num_output = get_num_output_padded_to_fit_input(num_input,
                                                    window_length_in_samp,
                                                    window_shift_in_samp)
    zero_padded_matrix = np.zeros((num_output, window_length_in_samp),
                                  dtype=np.float)
    for num_output_idx in range(num_output):
        start_idx = window_shift_in_samp * num_output_idx
        is_last_output = num_output_idx == num_output - 1
        end_idx = start_idx + window_length_in_samp if not is_last_output else num_input
        end_padded_idx = window_length_in_samp if not is_last_output else end_idx - start_idx
        window_data = data[start_idx:end_idx]
        if zero_mean_offset:
            window_data = window_data - np.mean(window_data)
        zero_padded_matrix[num_output_idx, :end_padded_idx] = window_data
    return zero_padded_matrix


def get_num_output_padded_to_fit_input(num_input, window_length_in_samp,
                                       window_shift_in_samp):
    num_output_valid = (num_input - window_length_in_samp) / float(
        window_shift_in_samp) + 1
    return int(np.ceil(num_output_valid))


def _weight_data_matrix(data_matrix, window_type, data_transformation=None):
    window_length_in_samp = data_matrix[0].shape[0]
    window = get_window(window_type, window_length_in_samp, fftbins=False)
    if (data_transformation == 'group_delay'):
        window *= np.arange(window_length_in_samp)
    return data_matrix * window


def get_non_symmetric_length(symmetric_length):
    return int(symmetric_length / 2) + 1


def get_non_symmetric_data(data):
    num_fft_points = data.shape[-1]
    num_ess_fft_points = get_non_symmetric_length(num_fft_points)
    return data[:, :num_ess_fft_points]


def get_max_length_stft_based(length_in_samp, window_length_in_s,
                              window_shift_in_s, sampling_rate_in_hz):
    window_length_in_samp = get_length_in_samp(window_length_in_s,
                                               sampling_rate_in_hz)
    window_shift_in_samp = get_length_in_samp(window_shift_in_s,
                                              sampling_rate_in_hz)
    return get_num_output_padded_to_fit_input(length_in_samp,
                                              window_length_in_samp,
                                              window_shift_in_samp)


def calculate_incr_var(var_prev, mean_prev, mean, length):
    return var_prev + (length - mean_prev) * (length - mean)


def calculate_incr_mean(count, mean, length):
    return mean + (length - mean) / float(count)


def calculate_var(sum1, sum2, count):
    return (sum2 - ((sum1 * sum1) / float(count))) / float(count - 1)


def calculate_mean(sum1, count):
    return sum1 / float(count)
