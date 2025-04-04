#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 13:41:11 2025

@author: Anonymous Authors
"""

import os
import json
import pickle
import logging
from datetime import datetime

from tqdm import tqdm
from stopwatch import Stopwatch

from graph import PropertyGraph
from encoding import (
    encode_graph_to_text,
    encode_graph_to_text_with_template,
    encode_graph_to_text_with_llm
)
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

INCONSISTENCY_DIR = "inconsistency"
INCONSISTENCIES_PICKLE_FILE = "inconsistencies.pkl"

# --------------------------------------------------------------------------- #

response_counter = 0
RESPONSE_PREFIX = "response_"

CONTINUE = False
CONTINUE_RESPONSE_DIR = "response_21"

while True:
    RESPONSE_DIR = f"{RESPONSE_PREFIX}{response_counter}"
    if CONTINUE:
        if not os.path.isdir(CONTINUE_RESPONSE_DIR):
            os.mkdir(CONTINUE_RESPONSE_DIR)
        RESPONSE_DIR = CONTINUE_RESPONSE_DIR
        break
    elif not os.path.isdir(RESPONSE_DIR):
        os.mkdir(RESPONSE_DIR)
        break
    else:
        response_counter += 1

###############################################################################

LOG_FILE = os.path.join(RESPONSE_DIR, "log.txt")

LOGGER = logging.getLogger(__name__)
STREAM_FORMATTER = logging.Formatter(fmt="%(message)s")
STREAM_HANDLER = logging.StreamHandler()
STREAM_HANDLER.setFormatter(STREAM_FORMATTER)
STREAM_HANDLER.setLevel(logging.INFO)

FILE_FORMATTER = logging.Formatter(
    fmt="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
FILE_HANDLER = logging.FileHandler(LOG_FILE)
FILE_HANDLER.setFormatter(FILE_FORMATTER)
FILE_HANDLER.setLevel(logging.DEBUG)

LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(STREAM_HANDLER)
LOGGER.addHandler(FILE_HANDLER)

# --------------------------------------------------------------------------- #

ENCODE_MODES = {
    "graph": {
        "id": "graph",
        "receive": "- A node-edge graph representation of an inconsistency",
    },
    "llm": {
        "id": "llm",
        "receive": (
            "- A Cypher query constraint identifying the inconsistency\n"
            "- A textual description of the graph pattern corresponding to the inconsistency"
        ),
    },
    "cypher": {
        "id": "cypher",
        "receive": "- A Cypher query constraint identifying the inconsistency",
    },
    "template": {
        "id": "template",
        "receive": (
            "- A Cypher query constraint identifying the inconsistency\n"
            "- A textual description of the inconsistency with actual values"
        ),
    },
}

EXAMPLE_MODES = {
    "one_large": {
        "id": "one_large",
        "text": (
"""
EXAMPLE OUTPUT:
<repairs>
DEL_EDGE | (rm) | -
ADD_NODE | (m1:Medicine) | description="medication-name"
ADD_EDGE | (p)-[:TAKES_MEDICATION]-> (m1) | -
</repairs>
"""
        )
    },
    "one_small": {
        "id": "one_small",
        "text": (
"""
EXAMPLE OUTPUT #1: (e.g., if the allergy information is incorrect)
<repairs>
DEL_EDGE | [ra] | -
</repairs>
"""
        )
    },
    "two_small": {
        "id": "two_small",
        "text": (
"""
EXAMPLE OUTPUT #1: (e.g., if the allergy information is incorrect)
<repairs>
DEL_EDGE | [ra] | -
</repairs>

EXAMPLE OUTPUT #2: (e.g., if the ingredient information is incorrect)
<repairs>
DEL_EDGE | [rc] | -
</repairs>
"""
        )
    },
    "two_small_large": {
        "id": "two_small_large",
        "text": (
"""
EXAMPLE OUTPUT #1: (e.g., if you find an alternate medication that treats the patient)
<repairs>
DEL_EDGE | (rm) | -
ADD_NODE | (m1:Medicine) | description="medication-name"
ADD_EDGE | (p)-[:TAKES_MEDICATION]-> (m1) | -
</repairs>

EXAMPLE OUTPUT #2: (e.g., if the ingredient information is incorrect)
<repairs>
DEL_EDGE | [rc] | -
</repairs>
"""
        )
    },
"none": {
        "id": "none",
        "text": ""
    }
}

# - A graph representation of an inconsistency
# - A Cypher query constraint identifying the inconsistency
# - A textual description of the inconsistency with actual values


SYSTEM_PROMPT = """You are an AI assistant for Graph Repair. Your task is to identify factual inconsistencies and suggest corrections using structured graph operations.

You will receive:
{0}

Allowed repair operations:
1. `ADD_NODE` - Add a new node (with optional properties)
2. `ADD_EDGE` - Add a new relationship between two nodes
3. `DEL_EDGE` - Remove an existing relationship
4. `UPD_NODE` - Modify a node's properties
5. `UPD_EDGE` - Modify an edge's properties

Suggest only factually accurate repairs. Use the provided format for output.
Keep the suggested number of repair operations small.
"""

INSTRUCTION = """Based on the following description of an inconsistency
suggest graph repairs to fix it:\n---\n"""

OUTPUT_FORMAT = """\n---\n
OUTPUT FORMAT:

Provide suggested repairs in the following structured format:

<repairs> {op_code} | {target} | {details} </repairs>

where:
- {op_code} is one of (`ADD_NODE`, `ADD_EDGE`, `DEL_EDGE`, `UPD_NODE`, `UPD_EDGE`)
- {target} specifies the affected node or relationship variable
- {details} contains relevant property changes (key-value) or `-` if none

Do NOT add explanations beyond the descriptions in the output.

"""

# Try with changed

# Detect conflict between human repair vs LLM repair on same part of the graph (node / edge) !!
# If conflicted part is human scope, prefer human repair,
# Else prefer LLM repair
# This way, no iterations of repairs
# Apply human repair AFTER LLM, so if the same variable gets updated, human overwrites it
# In case of two repairs by LLM, randomly one may overwrite other (no way to differentiate)

###############################################################################

with open(INCONSISTENCIES_PICKLE_FILE, "rb") as f:
    inconsistencies = pickle.load(f)


IC_QUERY = inconsistencies["query"]
IC_TEMPLATE = inconsistencies["template"]
IC_COUNT = len(inconsistencies["record"])
IC_VARIABLES = inconsistencies["variables"]

MAX_IC_COUNT = 1000


MODELS = [MODEL_LLAMA, MODEL_MISTRAL, MODEL_PHI, MODEL_GEMMA, MODEL_QWEN, MODEL_DEEPSEEK]

# --------------------------------------------------------------------------- #

ENCODE_GRAPH_WITH_LLM = False

if ENCODE_GRAPH_WITH_LLM is True:
    ENCODE_START_TIME = datetime.now()
    ES = Stopwatch()
    ES.start()
    # ----------------------------------------------------------------------- #

    for model in tqdm(MODELS):
        print(f"Encoding with Model: {model}")
        # start_model(model)

        for idx in tqdm(range(min(IC_COUNT, MAX_IC_COUNT))):
            human_encoded_prompt, response_json = encode_graph_to_text_with_llm(
                model, inconsistencies["graph"][idx]
            )

            with open(
                os.path.join(INCONSISTENCY_DIR, f"i{idx}_{model}_encode.txt"), "w"
            ) as f:
                f.write(human_encoded_prompt)

            with open(
                os.path.join(INCONSISTENCY_DIR, f"i{idx}_{model}_encode.json"), "w"
            ) as f:
                f.write(response_json)

        stop_model(model)
        ES.tick(f"MODLE:{model}")

    # ----------------------------------------------------------------------- #
    ES.stop()

    ENCODE_LOG = [f"Timestamp: {ENCODE_START_TIME}"]
    for tick in ES.get_ticks():
        ENCODE_LOG.append(str(tick.__dict__))

    with open(os.path.join(INCONSISTENCY_DIR, "encode_log.txt"), "w") as f:
        f.write("\n".join(ENCODE_LOG))

# --------------------------------------------------------------------------- #

###############################################################################

ENCODE_MODE = "template"
EXAMPLE_MODE = "two_small"

EXAMPLE = EXAMPLE_MODES[EXAMPLE_MODE]["text"]
ENCODING_MODEL = MODEL_LLAMA

SYSTEM_PROMPT = SYSTEM_PROMPT.format(ENCODE_MODES[ENCODE_MODE]["receive"])

###############################################################################

for idx in tqdm(range(min(IC_COUNT, MAX_IC_COUNT))):
    # graph record (result of Cypher query for inconsistency)
    graph_record = inconsistencies["record"][idx]
    with open(f"{INCONSISTENCY_DIR}/record_i{idx}.json", "w") as f:
        json.dump(graph_record, f, ensure_ascii=False, indent=2)

    # build description of SAPG based on template
    sapg = inconsistencies["graph"][idx]
    sapg_text = encode_graph_to_text(sapg)
    sapg_template_text = encode_graph_to_text_with_template(graph_record, IC_TEMPLATE, IC_VARIABLES)
    with open(f"{INCONSISTENCY_DIR}/template_i{idx}.txt", "w") as f:
        f.write(sapg_template_text)

    # build prompt
    if ENCODE_MODE == "graph":
        # prompt with node-edge graph representation
        prompt = INSTRUCTION + sapg_text + OUTPUT_FORMAT + EXAMPLE
    elif ENCODE_MODE == "cypher":
        # prompt with cypher query only
        prompt = INSTRUCTION + IC_QUERY + json.dumps(graph_record) + OUTPUT_FORMAT + EXAMPLE
    elif ENCODE_MODE == "template":
        # prompt with cypher query + template
        prompt = INSTRUCTION + IC_QUERY + sapg_template_text + OUTPUT_FORMAT + EXAMPLE
    elif ENCODE_MODE == "llm":
        # prompt with LLM encoding of a graph record
        with open(os.path.join(INCONSISTENCY_DIR, f"i{idx}_{ENCODING_MODEL}_encode.txt"), "r") as f:
            sapg_llm_text = f.read()
        prompt = INSTRUCTION + IC_QUERY + sapg_llm_text + OUTPUT_FORMAT + EXAMPLE
    else:
        raise RuntimeError("Invalid ENCODE_MODE")

    with open(f"{RESPONSE_DIR}/prompt_i{idx}.txt", "w") as f:
        f.write(prompt)

# --------------------------------------------------------------------------- #

START_TIME = datetime.now()
LOGGER.info(f"Repair Start Timestamp: {START_TIME}")

LOGGER.info(f"Encode Mode: {ENCODE_MODE}")
LOGGER.info(f"Example Mode: {EXAMPLE_MODE}")
LOGGER.info(f"LLM Encoding Model: {ENCODING_MODEL}")

S = Stopwatch()
S.start()
# --------------------------------------------------------------------------- #

for model in tqdm(MODELS):
    LOGGER.info(f"Repairing with Model: {model}")
    # start_model(model)

    for idx in tqdm(range(min(IC_COUNT, MAX_IC_COUNT))):
        if os.path.isfile(os.path.join(RESPONSE_DIR, f"i{idx}_{model}.json")):
            LOGGER.info(f"SKIP: i{idx} for model '{model}'")
            continue

        LOGGER.debug(f"START: i{idx} for model '{model}'")

        with open(f"{RESPONSE_DIR}/prompt_i{idx}.txt", "r") as f:
            prompt = f.read()

        response = create_completion(
            model, prompt, SYSTEM_PROMPT
        )
        with open(os.path.join(RESPONSE_DIR, f"i{idx}_{model}.txt"), "w") as f:
            f.write(response.content)

        with open(os.path.join(RESPONSE_DIR, f"i{idx}_{model}.json"), "w") as f:
            f.write(response.model_dump_json(indent=2))

        LOGGER.debug(f"END: i{idx} for model '{model}'")

    stop_model(model)
    S.tick(f"MODLE:{model}")

# --------------------------------------------------------------------------- #
S.stop()

for tick in S.get_ticks():
    LOGGER.info(str(tick.__dict__))

###############################################################################
