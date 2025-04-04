#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 05 14:37:20 2025

@author: Anonymous Authors
"""

###############################################################################

import os
import re
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from llm import MODEL_LLAMA, MODEL_MISTRAL, MODEL_PHI, MODEL_GEMMA, MODEL_QWEN, MODEL_DEEPSEEK

###############################################################################


def parse_repair_text(repair_text):
    """
    Parse repair text and produce:
        - list of repair operations if correct format
        - None if invalid format
    """
    try:
        operations = []
        repair_lines = repair_text.strip().split("\n")
        for repair_line in repair_lines:
            if not repair_line.strip():
                operations.append(("BLANK_LINE", ""))
                continue
            repair_line_words = [w.strip() for w in repair_line.split("|")]
            if len(repair_line_words) < 2:
                operations.append(("INVALID_LINE", ""))
                continue
            repair_operation = repair_line_words[0]
            repair_target = repair_line_words[1].strip("[]() ")
            operations.append((repair_operation, repair_target))
        return operations
    except Exception:
        pass

def evaluate(response_text, graph_record):
    result = defaultdict(list)

    is_valid_format = True     # check if the repairs are given in correct format
    is_valid_repair = True     # check if repair actually repairs
    is_correct_repair = True   # check if repair matches ground truth


    r =  re.search("<repairs>(.*)</repairs>", response_text, flags=re.DOTALL)
    if r:
        repair_text = r.group(1)
    else:
        is_valid_format = False
        is_valid_repair = False
        is_correct_repair = False
        repair_text = ""

    operation_counter = {
        "ADD_NODE": 0,
        "UPD_NODE": 0,
        "DEL_NODE": 0,
        "ADD_EDGE": 0,
        "UPD_EDGE": 0,
        "DEL_EDGE": 0,
        "INVALID": 0,
    }

    invalid_ingredient = int(graph_record["rc"]["is_error"])
    invalid_allergy = int(graph_record["ra"]["is_error"])

    # best should be to mock apply suggested repair to see if it fixes it
    valid_repairs = [
        ("DEL_EDGE", "rm"),
        ("DEL_EDGE", "ra"),
        ("DEL_EDGE", "rc")
    ]

    correct_repairs = []
    if invalid_ingredient:
        correct_repairs = [("DEL_EDGE", "rc")]
    if invalid_allergy:
        correct_repairs = [("DEL_EDGE", "ra")]

    repair_operations = parse_repair_text(repair_text)
    if not repair_operations:
        is_valid_format = False
        is_valid_repair = False
        is_correct_repair = False
    else:
        for (repair_operation, repair_target) in repair_operations:
            op_items = repair_operation.split("_")
            if len(op_items) == 2 and op_items[0] in ["ADD", "UPD", "DEL"] and op_items[1] in ["NODE", "EDGE"]:
                operation_counter[f"{repair_operation}"] += 1
            else:
                operation_counter["INVALID"] += 1
                is_valid_format = False
                is_valid_repair = False
                is_correct_repair = False
                # print(repair_operation)

        # if operation_counter["DEL_NODE"]:
        #     is_valid_repair = False
        #     is_correct_repair = False

        if not any(vr in repair_operations for vr in valid_repairs):
            is_valid_repair = False
            is_correct_repair = False

        is_correct_repair = (repair_operations == correct_repairs)

    result["char_count"] = len(response_text)
    result["response_words"] = len(response_text.split())
    result["response_lines"] = len(response_text.split("\n"))
    result["repair_chars"] = len(repair_text)
    result["repair_words"] = len(repair_text.split())
    result["repair_lines"] = len(repair_text.strip().split("\n"))
    result.update(operation_counter)
    result["is_valid_format"] = is_valid_format
    result["is_valid_repair"] = is_valid_repair
    result["is_correct_repair"] = is_correct_repair

    assert sum(operation_counter.values()) == result["repair_lines"]

    return result


# --------------------------------------------------------------------------- #

def parse_encoding_performance(record_dir="inconsistency"):
    stats = {}
    for filename in os.listdir(record_dir):
        if filename.endswith("_encode.json"):
            filename_parts = filename.split("_")
            ic_id = filename_parts[0]
            model_name = filename_parts[1]

            with open(os.path.join(record_dir, filename), "r") as f:
                data = json.load(f)

            if model_name not in stats:
                stats[model_name] = {
                    "total_duration": [],
                    "load_duration": [],
                    "prompt_eval_count": [],
                    "prompt_eval_duration": [],
                    "eval_count": [],
                    "eval_duration": [],
                    "char_count": [],
                    "response_words": [],
                    "response_lines": [],
                }

            metadata = data["response_metadata"]
            stats[model_name]["total_duration"].append(metadata["total_duration"])
            stats[model_name]["load_duration"].append(metadata["load_duration"])
            stats[model_name]["prompt_eval_duration"].append(metadata["prompt_eval_duration"])
            stats[model_name]["eval_duration"].append(metadata["eval_duration"])
            stats[model_name]["prompt_eval_count"].append(metadata["prompt_eval_count"])
            stats[model_name]["eval_count"].append(metadata["eval_count"])

            response_text = data["content"]
            stats[model_name]["char_count"].append(len(response_text))
            if "</think>" in response_text:
                response_text = response_text.split("</think>")[1].strip()
            stats[model_name]["response_words"].append(len(response_text.split()))
            stats[model_name]["response_lines"].append(len(response_text.split("\n")))

    model_names = [MODEL_LLAMA, MODEL_MISTRAL, MODEL_PHI, MODEL_GEMMA, MODEL_QWEN, MODEL_DEEPSEEK]
    values = {
        metric: [np.mean(stats[model][metric]) for model in model_names]
        for metric in stats[model_name]
    }
    df = pd.DataFrame(values, index=model_names)
    return stats, df

def parse_and_plot(response_dir, record_dir):
    stats = {}

    # Loop through all JSON files in the directory
    for filename in os.listdir(response_dir):
        if filename.endswith(".json"):
            filename_parts = filename.split("_")
            ic_id = filename_parts[0]
            model_name = filename_parts[-1].replace(".json", "")  # Extract model name

            with open(os.path.join(record_dir, f"record_{ic_id}.json"), "r") as f:
                record_json = json.load(f)
            with open(os.path.join(response_dir, filename), "r") as f:
                data = json.load(f)

            if model_name not in stats:
                stats[model_name] = {
                    "total_duration": [],
                    "load_duration": [],
                    "prompt_eval_count": [],
                    "prompt_eval_duration": [],
                    "eval_count": [],
                    "eval_duration": [],
                    "char_count": [],
                    # response stats
                    "response_words": [],
                    "response_lines": [],
                    "repair_chars": [],
                    "repair_lines": [],
                    "repair_words": [],
                    "ADD_NODE": [],
                    "UPD_NODE": [],
                    "DEL_NODE": [],
                    "ADD_EDGE": [],
                    "UPD_EDGE": [],
                    "DEL_EDGE": [],
                    "INVALID": [],
                    "is_valid_format": [],
                    "is_valid_repair": [],
                    "is_correct_repair": []
                }

            # Extract performance metadata metrics
            metadata = data["response_metadata"]
            stats[model_name]["total_duration"].append(metadata["total_duration"])
            stats[model_name]["load_duration"].append(metadata["load_duration"])
            stats[model_name]["prompt_eval_duration"].append(metadata["prompt_eval_duration"])
            stats[model_name]["eval_duration"].append(metadata["eval_duration"])
            stats[model_name]["prompt_eval_count"].append(metadata["prompt_eval_count"])
            stats[model_name]["eval_count"].append(metadata["eval_count"])

            # Calculate response metrics
            # Response length (words in content)
            if "content" in data:
                response_text = data["content"]
                response_evaluation = evaluate(response_text, record_json)
                for k, v in response_evaluation.items():
                    stats[model_name][k].append(v)

            with open(os.path.join(response_dir, f"{ic_id}.{model_name}.eval"), "w") as f:
                json.dump(response_evaluation, f, indent=2)

    # Plot statistics

    GROUPS = [
        {
            "id": "repair_operations",
            "title": "Repair Operations",
            "metrics": [
                "ADD_NODE", "DEL_NODE", "UPD_NODE",
                "ADD_EDGE", "DEL_EDGE", "UPD_EDGE",
                "INVALID"
            ],
        }
    ]

    ALL_KEYS = stats[model_name].keys()
    EXCLUDE_KEYS = [] + [k for g in GROUPS for k in g["metrics"]]

    for metric in [k for k in ALL_KEYS if k not in EXCLUDE_KEYS]:
        plt.figure(figsize=(10, 5))
        model_names = list(stats.keys())
        values = [np.mean(stats[model][metric]) for model in model_names]

        sns.barplot(x=model_names, y=values, hue=model_names, palette="viridis", legend=False)
        plt.xticks(rotation=15)
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"Comparison of {metric.replace('_', ' ')} Across LLMs")
        plt.savefig(os.path.join(response_dir, f"{metric}.png"))

    for group in GROUPS:
        group_id = group["id"]
        metrics = group["metrics"]
        title = group["title"]

        model_names = list(stats.keys())
        # Prepare data for stacking
        values = {metric: [np.mean(stats[model][metric]) for model in model_names] for metric in metrics}

        plt.figure(figsize=(10, 5))

        bottom = np.zeros(len(model_names))  # start stacking from zero
        colors = sns.color_palette("viridis", len(metrics))  # get distinct colors

        for i, metric in enumerate(metrics):
            sns.barplot(
                x=model_names, y=values[metric],
                color=colors[i], label=metric.replace("_", " ").replace("count", "").strip(),
                bottom=bottom  # Stack on top of previous bars
            )
            bottom += np.array(values[metric])  # Update stacking position

        plt.xticks(rotation=15)
        plt.ylabel(f"Mean {title}")
        plt.title(f"Comparison of {title} Across Models")
        plt.legend()

        plt.savefig(os.path.join(response_dir, f"{group_id}.png"))
        # plt.show()

        # plt.figure(figsize=(10, 5))

        # model_names = list(stats.keys())
        # values = {metric: [np.mean(stats[model][metric]) for model in model_names] for metric in metrics}

        # # Stacked bar plot
        # bottom = np.zeros(len(model_names))  # Start stacking from zero
        # for metric in metrics:
        #     plt.bar(model_names, values[metric], bottom=bottom, label=metric.replace("_", " ").replace("count", "").strip())
        #     bottom += np.array(values[metric])  # Update bottom position

        # plt.xticks(rotation=15)
        # plt.ylabel(f"Mean {title}")
        # plt.title(f"Comparison of {title} Across Models")
        # plt.legend()
        # plt.savefig(os.path.join(response_dir, f"{group_id}.png"))

    values = {metric: [np.mean(stats[model][metric]) for model in model_names] for metric in ALL_KEYS}
    full_df = pd.DataFrame(values, index=model_names)

    for group in GROUPS:
        group_id = group["id"]
        metrics = group["metrics"]
        title = group["title"]
        values = {metric: [np.mean(stats[model][metric]) for model in model_names] for metric in metrics}
        df = pd.DataFrame(values, index=model_names)
        df.plot(kind="bar", stacked=True, figsize=(10, 5), colormap="viridis")

        plt.xticks(rotation=15)
        plt.ylabel(f"Mean {title}")
        plt.title(f"Stacked Comparison of {title} Across Models")
        plt.legend()

        plt.savefig(os.path.join(response_dir, f"{group_id}_pandas.png"))
        # plt.show()

    # print("Summary Statistics:")
    # for model in stats:
    #     print(f"\nModel: {model}")
    #     for key, values in stats[model].items():
    #         print(f"  {key}: Mean={np.mean(values):,.2f}, Std={np.std(values):,.2f}")

    return stats, full_df

###############################################################################

STATS = {}
DFS = {}

RESPONSE_DIRS = [
    "example_mode_none",
    "example_mode_one_small",
    "example_mode_two_small",
    "example_mode_one_large",
    "example_mode_two_mix",
    "encode_mode_graph",
    "encode_mode_cypher",
    "encode_mode_template",
    "encode_mode_llm_llama3.2",
    "encode_mode_llm_mistral",
    "encode_mode_llm_phi4",
    "encode_mode_llm_gemma2",
    "encode_mode_llm_qwen2.5",
    "encode_mode_llm_deepseek-r1",
]
for response_dir in RESPONSE_DIRS:
    STATS[response_dir], DFS[response_dir] = parse_and_plot(response_dir=response_dir, record_dir="inconsistency")

###############################################################################

def print_latex_table(table):
    for row in table:
        print(" & ".join(row), end="\\\\\n")

###############################################################################

TABLE_EXAMPLE_MODE_COMPARE = [
    ["\\bf \\#Examples"],
    ["~"]
]

for model in [MODEL_LLAMA, MODEL_MISTRAL, MODEL_PHI, MODEL_GEMMA, MODEL_QWEN, MODEL_DEEPSEEK]:
    # TABLE_EXAMPLE_MODE_COMPARE[0].append(f"\\multicolumn{{4}}{{c}}{{\\bf {model}}}")
    # TABLE_EXAMPLE_MODE_COMPARE[1].extend(["\\bf T", "\\bf F", "\\bf V", "\\bf A"])
    TABLE_EXAMPLE_MODE_COMPARE[0].append(f"\\multicolumn{{3}}{{c}}{{\\bf {model}}}")
    TABLE_EXAMPLE_MODE_COMPARE[1].extend(["\\bf F", "\\bf V", "\\bf A"])

example_modes = {
    "none": "none",
    "one_small": "1-small",
    "two_small": "2-small",
    "one_large": "1-large",
    "two_mix": "2-mixed"
}
for example_mode, ex_header in example_modes.items():
    TABLE_EXAMPLE_MODE_COMPARE.append([ex_header])
    df = DFS[f"example_mode_{example_mode}"]
    for model in [MODEL_LLAMA, MODEL_MISTRAL, MODEL_PHI, MODEL_GEMMA, MODEL_QWEN, MODEL_DEEPSEEK]:
        model_row = df.loc[model]
        # TABLE_EXAMPLE_MODE_COMPARE[-1].append(f"{model_row['total_duration']/1e9:.2f}")
        TABLE_EXAMPLE_MODE_COMPARE[-1].append(
            f"\\bf {model_row['is_valid_format']:.2f}"
            if model_row['is_valid_format'] > 0.9
            else f"{model_row['is_valid_format']:.2f}"
        )
        TABLE_EXAMPLE_MODE_COMPARE[-1].append(
            f"\\bf {model_row['is_valid_repair']:.2f}"
            if model_row['is_valid_repair'] > 0.9
            else f"{model_row['is_valid_repair']:.2f}"
        )
        TABLE_EXAMPLE_MODE_COMPARE[-1].append(
            f"\\bf {model_row['is_correct_repair']:.2f}"
            if model_row['is_correct_repair'] > 0.3
            else f"{model_row['is_correct_repair']:.2f}"
        )

print_latex_table(TABLE_EXAMPLE_MODE_COMPARE)

###############################################################################

TABLE_ENCODE_MODE_COMPARE = [
    ["\\bf Encoding Mode"],
    ["~"]
]

for model in [MODEL_LLAMA, MODEL_MISTRAL, MODEL_PHI, MODEL_GEMMA, MODEL_QWEN, MODEL_DEEPSEEK]:
    # TABLE_ENCODE_MODE_COMPARE[0].append(f"\\multicolumn{{4}}{{c}}{{\\bf {model}}}")
    # TABLE_ENCODE_MODE_COMPARE[1].extend(["\\bf T", "\\bf F", "\\bf V", "\\bf A"])
    TABLE_ENCODE_MODE_COMPARE[0].append(f"\\multicolumn{{3}}{{c}}{{\\bf {model}}}")
    TABLE_ENCODE_MODE_COMPARE[1].extend(["\\bf F", "\\bf V", "\\bf A"])

encode_modes = {
    "encode_mode_graph": "Graph",
    "encode_mode_cypher":  "Cypher",
    "encode_mode_template": "Template",
    "encode_mode_llm_llama3.2": "LLM (llama3.2)",
    "encode_mode_llm_mistral": "LLM (mistral)",
    "encode_mode_llm_phi4": "LLM (phi4)",
    "encode_mode_llm_gemma2": "LLM (gemma2)",
    "encode_mode_llm_qwen2.5": "LLM (qwen2.5)",
    "encode_mode_llm_deepseek-r1": "LLM (deepseek-r1)",
}

for encode_mode, enc_header in encode_modes.items():
    TABLE_ENCODE_MODE_COMPARE.append([enc_header])
    df = DFS[encode_mode]
    for model in [MODEL_LLAMA, MODEL_MISTRAL, MODEL_PHI, MODEL_GEMMA, MODEL_QWEN, MODEL_DEEPSEEK]:
        model_row = df.loc[model]
        # TABLE_ENCODE_MODE_COMPARE[-1].append(f"{model_row['total_duration']/1e9:.2f}")
        TABLE_ENCODE_MODE_COMPARE[-1].append(
            f"\\bf {model_row['is_valid_format']:.2f}"
            if model_row['is_valid_format'] > 0.9
            else f"{model_row['is_valid_format']:.2f}"
        )
        TABLE_ENCODE_MODE_COMPARE[-1].append(
            f"\\bf {model_row['is_valid_repair']:.2f}"
            if model_row['is_valid_repair'] > 0.9
            else f"{model_row['is_valid_repair']:.2f}"
        )
        TABLE_ENCODE_MODE_COMPARE[-1].append(
            f"\\bf {model_row['is_correct_repair']:.2f}"
            if model_row['is_correct_repair'] > 0.3
            else f"{model_row['is_correct_repair']:.2f}"
        )


print_latex_table(TABLE_ENCODE_MODE_COMPARE)

###############################################################################

DF_LIST = list(DFS.values())
THE_DF = pd.DataFrame(sum(DF_LIST)/len(DF_LIST))
THE_DF = THE_DF.reindex([MODEL_LLAMA, MODEL_MISTRAL, MODEL_PHI, MODEL_GEMMA, MODEL_QWEN, MODEL_DEEPSEEK])

# TABLE_LLM_COMPARISON = [
#     ["~", "\\bf Prompt Time (s)", "\\bf Evaluation Time (s)", "\\bf \\#Tokens", "\\bf \\#Repair Operations", "\\bf Correct Format", "\\bf Valid Repair", "\\bf Correct Repair"]
# ]

TABLE_LLM_COMPARISON = [
    ["\\bf Model", "\\bf Prompt Time", "\\bf Evaluation Time", "\\bf \\#Tokens", "\\bf \\#Characters", "\\bf \\#Characters", "\\bf \\#Repair Operations"],
    ["~", "(sec/prompt)", "(sec/prompt)", "~", "(Response)", "(Repair)", "~"]
]

for idx, row in THE_DF.iterrows():
    TABLE_LLM_COMPARISON.append([idx])
    TABLE_LLM_COMPARISON[-1].extend([
        f'{row["prompt_eval_duration"]/1e9:.2f}',
        f'{row["eval_duration"]/1e9:.2f}',
        f'{row["eval_count"]:.1f}',
        f'{row["char_count"]:.1f}',
        f'{row["repair_chars"]:.1f}',
        f'{row["repair_lines"]:.1f}',
        # f'{row["is_valid_format"]:.2f}',
        # f'{row["is_valid_repair"]:.2f}',
        # f'{row["is_correct_repair"]:.2f}',
    ])

print_latex_table(TABLE_LLM_COMPARISON)

###############################################################################

fig, ax1 = plt.subplots(figsize=(12, 6))

# plot bars on the primary y-axis
THE_DF[["is_valid_format", "is_valid_repair", "is_correct_repair"]].plot(kind="bar", ax=ax1, colormap="viridis")
legend = ["Valid Format", "Valid Repair", "Correct Repair"]

ax1.tick_params(axis='both', which='major', labelsize=18)
ax1.tick_params(axis='both', which='minor', labelsize=16)
ax1.xaxis.label.set_size(18)
ax1.yaxis.label.set_size(18)

plt.xticks(rotation=0)
ax1.set_ylabel("Mean Repair Quality", fontsize=18)
ax1.legend(legend, prop={'size': 18}, loc='upper left')

# create a secondary y-axis
ax2 = ax1.twinx()
ax2.yaxis.get_major_formatter().set_scientific(False)

(THE_DF["eval_duration"]/1e9).plot(kind="line", ax=ax2, marker='o', color='red')

ax2.tick_params(axis='y', labelsize=18)
ax2.yaxis.label.set_size(18)
ax2.set_ylabel("Evaluation Time\n(sec/prompt)", fontsize=18)
ax2.legend(["Evaluation Time"], prop={'size': 18}, loc='upper right')
# ax2.yaxis.label.set_color('red')
# ax2.tick_params(axis='y', colors='red')

# Add eval_duration values as text above the bars
for i, val in enumerate(THE_DF["eval_duration"]):
    ax1.text(i, max(THE_DF["is_valid_format"].max(), THE_DF["is_valid_repair"].max(), THE_DF["repair_lines"].max()) *1.05, f"{val:.2f}", ha='center', va='bottom', fontsize=12, rotation=45)

plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.1)
plt.savefig(os.path.join(f"llm_performance.png"))

###############################################################################

TABLE_OPERATION_COMPARISON = [
    ["\\bf Model", "\\bf ADD\\_NODE", "\\bf DEL\\_NODE", "\\bf UPD\\_NODE", "\\bf ADD\\_EDGE", "\\bf DEL\\_EDGE", "\\bf UPD\\_EDGE", "\\bf INVALID", "\\bf F", "\\bf V", "\\bf A"]
]
for idx, row in THE_DF.iterrows():
    TABLE_OPERATION_COMPARISON.append([idx])
    TABLE_OPERATION_COMPARISON[-1].extend([
        f'{row["ADD_NODE"]:.1f}',
        f'{row["DEL_NODE"]:.1f}',
        f'{row["UPD_NODE"]:.1f}',
        f'{row["ADD_EDGE"]:.1f}',
        f'{row["DEL_EDGE"]:.1f}',
        f'{row["UPD_EDGE"]:.1f}',
        f'{row["INVALID"]:.1f}',
        f'{row["is_valid_format"]:.2f}',
        f'{row["is_valid_repair"]:.2f}',
        f'{row["is_correct_repair"]:.2f}',
    ])

ax = THE_DF[["ADD_NODE", "DEL_NODE", "UPD_NODE", "ADD_EDGE", "DEL_EDGE", "UPD_EDGE", "INVALID"]].plot(kind="bar", stacked=True, figsize=(12, 6), colormap="viridis")
ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='both', which='minor', labelsize=16)
ax.xaxis.label.set_size(18)
ax.yaxis.label.set_size(18)
# ax.title.set_size(20)
plt.xticks(rotation=0)
plt.ylabel(f"Mean #Repair Operations")
plt.subplots_adjust(left=0.06, right=0.99, top=0.99, bottom=0.06)
# plt.title(f"Repair Operations Distribution across Models")
plt.legend(prop={'size': 16})
plt.savefig(os.path.join(f"repair_operations.png"))

###############################################################################

_, ENCODING_EVAL_DF = parse_encoding_performance(record_dir="inconsistency")
TABLE_ENCODING_TIME_COMPARISON = [
    ["\\bf Model", "\\bf \\#Tokens", "\\bf \\#Words", "\\bf \\#Lines", "\\bf Encoding Time"]
]
for idx, row in THE_DF.iterrows():
    TABLE_ENCODING_TIME_COMPARISON.append([idx])
    TABLE_ENCODING_TIME_COMPARISON[-1].extend([
        f'{row["eval_count"]:.1f}',
        f'{row["response_words"]:.1f}',
        f'{row["response_lines"]:.1f}',
        f'{row["eval_duration"]/1e9:.2f} s',
    ])

print_latex_table(TABLE_ENCODING_TIME_COMPARISON)

###############################################################################
