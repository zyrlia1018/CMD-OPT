import numpy as np
import pandas as pd

import configuration.config_default as cfgd

STEP_QED = 0.2
STEP_SMI = 0.2

def encode_property_change(input_data_path, LOG=None):
    property_change_encoder = {}
    for property_name in cfgd.PROPERTIES_scaffold:
        if property_name == 'QED':
            intervals, start_map_interval = build_intervals(input_data_path, step=STEP_QED, LOG=LOG)
        elif property_name in ['2D_Similarity', '3D_Similarity']:
            intervals, start_map_interval = build_smi_intervals(property_name, input_data_path, step=STEP_SMI, LOG=LOG)
        elif property_name in ['LMHuman', 'ClPlasma','T12']:
            intervals = ['higher', 'lower', 'no_change']

        if property_name == 'QED':
            property_change_encoder[property_name] = intervals, start_map_interval
        elif property_name in ['2D_Similarity', '3D_Similarity']:
            property_change_encoder[property_name] = intervals, start_map_interval
        elif property_name in ['LMHuman', 'ClPlasma', 'T12']:
            property_change_encoder[property_name] = intervals

    return property_change_encoder


def value_in_interval(value, start_map_interval):
    start_vals = sorted(list(start_map_interval.keys()))
    return start_map_interval[start_vals[np.searchsorted(start_vals, value, side='right') - 1]]


def interval_to_onehot(interval, encoder):
    return encoder.transform([interval]).toarray()[0]


def build_intervals(input_transformations_path, step=STEP_QED, LOG=None):
    df = pd.read_csv(input_transformations_path)
    delta_QED = df['Delta_QED'].tolist()
    min_val, max_val = min(delta_QED), max(delta_QED)
    if LOG:
        LOG.info("QED min and max: {}, {}".format(min_val, max_val))

    start_map_interval = {}
    interval_str = '({}, {}]'.format(round(-step/2, 2), round(step/2, 2))
    intervals = [interval_str]
    start_map_interval[-step/2] = interval_str

    positives = step/2
    while positives < max_val:
        interval_str = '({}, {}]'.format(round(positives, 2), round(positives+step, 2))
        intervals.append(interval_str)
        start_map_interval[positives] = interval_str
        positives += step
    interval_str = '({}, inf]'.format(round(positives, 2))
    intervals.append(interval_str)
    start_map_interval[float('inf')] = interval_str

    negatives = -step/2
    while negatives > min_val:
        interval_str = '({}, {}]'.format(round(negatives-step, 2), round(negatives, 2))
        intervals.append(interval_str)
        negatives -= step
        start_map_interval[negatives] = interval_str
    interval_str = '(-inf, {}]'.format(round(negatives, 2))
    intervals.append(interval_str)
    start_map_interval[float('-inf')] = interval_str

    return intervals, start_map_interval


# def build_smi_intervals(property_name, input_transformations_path, step=STEP_SMI, LOG=None):
#     df = pd.read_csv(input_transformations_path)
#     SMI_score = df[property_name].tolist()
#     min_val, max_val = min(SMI_score), max(SMI_score)
#     if LOG:
#         LOG.info("{} min and max: {}, {}".format(property_name, min_val, max_val))
#
#     start_map_interval = {}
#     interval_str = '({}, {}]'.format(round(-step/2, 2), round(step/2, 2))
#     intervals = [interval_str]
#     start_map_interval[-step/2] = interval_str
#
#     positives = step/2
#     while positives < max_val:
#         interval_str = '({}, {}]'.format(round(positives, 2), round(positives+step, 2))
#         intervals.append(interval_str)
#         start_map_interval[positives] = interval_str
#         positives += step
#     interval_str = '({}, inf]'.format(round(positives, 2))
#     intervals.append(interval_str)
#     start_map_interval[float('inf')] = interval_str
#
#     negatives = -step/2
#     while negatives > min_val:
#         interval_str = '({}, {}]'.format(round(negatives-step, 2), round(negatives, 2))
#         intervals.append(interval_str)
#         negatives -= step
#         start_map_interval[negatives] = interval_str
#     interval_str = '(-inf, {}]'.format(round(negatives, 2))
#     intervals.append(interval_str)
#     start_map_interval[float('-inf')] = interval_str
#
#     return intervals, start_map_interval


def build_smi_intervals(property_name, input_transformations_path, step=STEP_SMI, LOG=None):
    labels = ['not similar', 'slightly similar', 'similar', 'very similar', 'very similar']
    df = pd.read_csv(input_transformations_path)
    SMI_score = df[property_name].tolist()
    min_val, max_val = min(SMI_score), max(SMI_score)
    if LOG:
        LOG.info("{} min and max: {}, {}".format(property_name, min_val, max_val))

    # 计算每个区间的阈值
    step_size = (max_val - min_val) / 5
    thresholds = [min_val + i * step_size for i in range(5)] + [max_val]

    # 生成区间描述列表
    intervals = [labels[i] for i in range(5)]

    # 创建区间起始值与区间描述的映射字典
    start_map_interval = {}
    start_map_interval[float('-inf')] = intervals[0]  # 负无穷对应第一个区间描述

    for i in range(5):
        start_map_interval[thresholds[i]] = intervals[i]

    return intervals, start_map_interval