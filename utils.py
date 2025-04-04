#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utils

Created on Thu Nov 28 14:32:53 2024

@author: Anonymous Authors
"""
###############################################################################

import numpy as np

###############################################################################


def numpy_serializer(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")


###############################################################################
