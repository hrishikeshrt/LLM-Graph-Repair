#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Connection to Neo4j Graph Database

@author: Anonymous Authors
"""

###############################################################################

from neo4j import GraphDatabase, RoutingControl

###############################################################################


class Graph:
    def __init__(self, uri, username, password):
        self._uri = uri
        self._username = username
        self._password = password

        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        self.driver.close()

    def run(self, query):
        with self.driver.session() as session:
            result = session.run(query)
            result_list = list(result)
        return result_list

    def __enter__(self):
        return self

    def __exit__(self):
        self.close()


###############################################################################
