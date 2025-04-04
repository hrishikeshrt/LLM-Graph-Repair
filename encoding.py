#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Encoding of Graph as Text

@author: Anonymous Authors
"""

import json
from llm import (
    MODEL_LLAMA,
    MODEL_DEEPSEEK,
    MODEL_MISTRAL,
    MODEL_PHI,
    MODEL_GEMMA,
    MODEL_QWEN,
    create_completion,
    stop_model
)

###############################################################################


def encode_graph_to_text(graph):
    """
    Encodes a ScopeAwarePropertyGraph into a textual representation.

    Args:
        graph (ScopeAwarePropertyGraph): The graph to encode.

    Returns:
        str: A textual representation of the graph.
    """
    text_representation = []

    for node in graph.nodes(data=True):
        node_id, attributes = node
        attributes_text = ", ".join(f"{key}: {value}" for key, value in attributes.items())
        text_representation.append(f"Node {node_id}: {attributes_text}")

    for edge in graph.edges(data=True):
        source, target, attributes = edge
        attributes_text = ", ".join(f"{key}: {value}" for key, value in attributes.items())
        text_representation.append(f"Edge {source} -> {target}: {attributes_text}")

    return "\n".join(text_representation)


def encode_graph_to_text_with_llm(model, graph):
    """
    Encodes a ScopeAwarePropertyGraph into a textual representation.

    Args:
        graph (ScopeAwarePropertyGraph): The graph to encode.

    Returns:
        str: A textual representation of the graph.
    """
    text_representation = encode_graph_to_text(graph)
    system_prompt = (
        "You are an intelligent AI assistant. The user will provide you with "
        "a graph in Node/Edge representation. Your task is to describe it into "
        "natural language sentences. You may use original labels in brackets, "
        "but the sentences should be proper English sentences without too much "
        "syntactic clutter. Node IDs should be preserved (in brackets). "
        "You do not output anything else nor any kind of explanation."
    )

    user_prompt = (
        "Describe the following graph in a natural language:\n"
        "---\n"
        f"{text_representation}\n"
    )
    response = create_completion(model, user_prompt, system_prompt)
    response_json = response.model_dump_json(indent=2)
    human_text_representation = response.content
    return human_text_representation, response_json


def encode_graph_to_text_with_template(record, template, variables):
    variable_values = []
    for nested_key_tuple in variables:
        value = record.copy()
        for key in nested_key_tuple:
            value = value.get(key)
        variable_values.append(value)
    # print(template)
    # print(record)
    # print(variables)
    # print(variable_values)
    return template.format(*variable_values)

###############################################################################


if __name__ == "__main__":
    import pickle
    from graph import PropertyGraph

    with open("inconsistencies.pkl", "rb") as f:
        inconsistencies = pickle.load(f)

    ic_template = inconsistencies["template"]
    graph_record = inconsistencies["record"][0]
    sapg = inconsistencies["graph"][0]

    sapg_text = encode_graph_to_text(sapg)
    sapg_llm_text, response_json = encode_graph_to_text_with_llm(sapg)
    sapg_template_text = encode_graph_to_text_with_template(graph_record, ic_template)
