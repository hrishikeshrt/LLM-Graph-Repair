#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load Dataset

@author: Anonymous Authors
"""

###############################################################################

from dataset import GraphDataset
from dataset_synthea import (
    SERVER_URI, USERNAME, PASSWORD,
    SYNTHEA_DATA_DIR,
    SYNTHEA_DATA_QUERIES,
    SYNTHEA_ADDITIONAL_QUERIES
)

neo4j_graph = GraphDataset(SERVER_URI, USERNAME, PASSWORD)
neo4j_graph.run("MATCH (n) DETACH DELETE (n)")
neo4j_graph.load_dataset(
    SYNTHEA_DATA_DIR,
    SYNTHEA_DATA_QUERIES,
    SYNTHEA_ADDITIONAL_QUERIES
)
neo4j_graph.close()

###############################################################################
