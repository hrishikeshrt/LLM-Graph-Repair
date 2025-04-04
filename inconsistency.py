#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find Inconsistencies

@author: Anonymous Authors
"""

###############################################################################

import pickle
from tqdm import tqdm

from neo4j.graph import Node, Relationship

from dataset import GraphDataset
from dataset_synthea import SERVER_URI, USERNAME, PASSWORD

from graph import PropertyGraph

###############################################################################


def graph_from_cypher(record):
    """Constructs a networkx graph from the results of a neo4j cypher query

    Example of use:
    >>> result = session.run(query)
    >>> g = graph_from_cypher(result.data())

    Nodes have fields 'labels' (frozenset) and 'properties' (dicts). Node IDs correspond to the neo4j graph.
    Edges have fields 'type_' (string) denoting the type of relation, and 'properties' (dict)."""

    G = PropertyGraph()

    def add_node(node):
        # add node id if it hasn't already been added
        u = node.element_id
        if G.has_node(u):
            return
        G.add_node(u, labels=node._labels, properties=dict(node))

    def add_edge(relation):
        # add edge if it hasn't already been added
        # make sure the nodes at both ends are created
        for node in (relation.start_node, relation.end_node):
            add_node(node)
        # check if edge already exists
        u = relation.start_node.element_id
        v = relation.end_node.element_id
        eid = relation.element_id
        if G.has_edge(u, v):
            return
        # if not, create it
        G.add_edge(u, v, key=eid, type_=relation.type, properties=dict(relation))

    for entry in record:
        # is node
        if isinstance(entry, Node):
            add_node(entry)
        # is relationship
        elif isinstance(entry, Relationship):
            add_edge(entry)
        else:
            raise TypeError(f"Unrecognized entry in Record: {entry.__class__.__qualname__}")

    return G


def find_inconsistencies(ic_id, ic_query, ic_template, relevant_variables):
    """
    Identifies semantic inconsistencies from a Neo4j database and stores them as
    ScopeAwarePropertyGraphs in a pickle file.

    Returns:
        list: A list of ScopeAwarePropertyGraphs representing the inconsistencies.
    """
    G = GraphDataset(SERVER_URI, USERNAME, PASSWORD)  # Connect to Neo4j database
    result = G.run(ic_query)
    inconcistencies = {
        "id": ic_id,
        "query": ic_query,
        "template": ic_template,
        "variables": relevant_variables,
        "record": [],
        "graph": []
    }

    for record in tqdm(result):
        print(record)
        dict_record = {key: dict(record[key]) for key in record.keys()}
        # create a ScopeAwarePropertyGraph for each inconsistency
        graph = graph_from_cypher(record)
        inconcistencies["record"].append(dict_record)
        inconcistencies["graph"].append(graph)

    # Store the inconsistent graphs in a pickle file
    with open("inconsistencies.pkl", "wb") as f:
        pickle.dump(inconcistencies, f)

    return inconcistencies


###############################################################################


if __name__ == "__main__":
    # query to find semantic inconsistencies (example: allergy conflict)
    from dataset_synthea import INCONSISTENCY_QUERIES
    ic_id, ic_query, ic_template, relevant_variables = INCONSISTENCY_QUERIES[0]
    inconcistencies = find_inconsistencies(ic_id, ic_query, ic_template, relevant_variables)
    print(f"Found {len(inconcistencies['graph'])} inconsistencies.")


###############################################################################
