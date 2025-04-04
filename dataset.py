#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Class

@author: Anonymous Authors
"""

###############################################################################

import os
import csv
from typing import Dict, List

from tqdm import tqdm

from connect import Graph

###############################################################################


class GraphDataset(Graph):
    def load_csv(self, query: str, filepath: str):
        """Load CSV data using a Neo4j query

        Parameters
        ----------
        query : str
            Neo4j query to load data
        filepath : str
            Path to the CSV file
        """
        with self.driver.session() as session:
            with open(filepath, "r") as file:
                reader = csv.DictReader(file)

                for row in tqdm(reader):
                    session.run(query, **row)

    def load_dataset(self, data_directory: str, data_queries: Dict[str, str], additional_queries: List[str]):
        """Load a dataset containing multiple CSVs and relations

        Parameters
        ----------
        data_directory : str
            Path to the data directory containing CSVs
        data_queries : Dict[str, str]
            Dictionary containing a mapping of
                * CSV file names relative to the `data_directory`
                * Corresponding Neo4j query to load data from that file
        additional_queries : List[str]
            Additional Neo4j queries, typically to create relationships
            among the added nodes and perform any kind of clean-up
        """
        for filename, query in data_queries.items():
            print(f"Loading from CSV: {filename} ...")
            print(f"QUERY: {query}")
            self.load_csv(query, os.path.join(data_directory, filename))

        with self.driver.session() as session:
            for query in additional_queries:
                session.run(query)

###############################################################################
