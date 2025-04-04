#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 11:12:03 2025

@author: Anonymous Authors

Requirements
------------
- Ollama needs to be installed
- pip install ollama
"""

import subprocess

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

###############################################################################

MODEL_LLAMA = "llama3.2"
MODEL_PHI = "phi4"
MODEL_GEMMA = "gemma2"
MODEL_MISTRAL = "mistral"
MODEL_DEEPSEEK = "deepseek-r1"
MODEL_QWEN = "qwen2.5"

MODELS = [
    MODEL_LLAMA,
    MODEL_PHI,
    MODEL_GEMMA,
    MODEL_MISTRAL,
    MODEL_QWEN,
    MODEL_DEEPSEEK,
]

TEMPERATURE = 0.4

###############################################################################


def stop_model(model_id):
    try:
        subprocess.run(["ollama", "stop", model_id], check=True)
        print(f"Model {model_id} stopped successfully.")
    except subprocess.CalledProcessError:
        print(f"Failed to stop model {model_id}.")


###############################################################################


def create_completion(
    model_name: str,
    user_prompt: str,
    system_prompt: str = None,
    message_history: list = None,
    temperature: float = TEMPERATURE,
):
    model_parameters = {
        "temperature": temperature
    }
    if message_history:
        messages = message_history.copy()
    else:
        messages = []
        if system_prompt:
            messages.append(("system", system_prompt))

    messages.append(("human", user_prompt))
    llm = ChatOllama(model=model_name, **model_parameters)
    response = llm.invoke(messages)
    return response


###############################################################################


if __name__ == '__main__':
    for model in [MODEL_LLAMA, MODEL_DEEPSEEK]:
        print(model)
        response = create_completion(model, "What is the capital of Bharat?")
        print(response.content)
        stop_model(model)
