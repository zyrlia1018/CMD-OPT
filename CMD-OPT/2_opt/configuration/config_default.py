import math

# Data
DATA_DEFAULT = {
    'max_sequence_length': 128,
    'padding_value': 0
}

# Properties
PROPERTIES = ['QED', 'LMHuman', 'ClPlasma', 'T12']
PROPERTY_THRESHOLD = {
    'LMHuman': 0.1,
    'ClPlasma': 0.5,
    'T12': 0.1
}

PROPERTY_ERROR = {
    'QED': 0.1,
    'LMHuman': 0.1,
    'ClPlasma': 0.1,
    'T12': 0.1 
}

# For Test_Property test
LOD_MIN = 1.0
LOD_MAX = 3.4



