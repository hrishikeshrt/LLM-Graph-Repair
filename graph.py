#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scope Aware Property Graph

Created on Thu Nov 21 18:24:09 2024

@author: Anonymous Authors
"""

###############################################################################

import numpy as np
import networkx as nx

###############################################################################


class PropertyGraph(nx.DiGraph):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return (
            f"{self.__class__.__qualname__}("
            f"n:{self.number_of_nodes()}, "
            f"e:{self.number_of_edges()})"
        )

###############################################################################
