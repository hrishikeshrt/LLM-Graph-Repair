#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 10:18:00 2025

@author: Anonymous Authors
"""

###############################################################################

import os
import random
import itertools

import requests_cache

import pandas as pd
import numpy as np

from dataset_synthea import SYNTHEA_DATA_DIR
from tqdm import tqdm

###############################################################################

SEED = os.environ.get("H_SEED", 42)
RNG = np.random.RandomState(SEED)

# For every correct ingredient, an incorrect ingredient will also be added
# with INCORRECT_INGREDIENT_PROBABILITY
INCORRECT_INGREDIENT_PROBABILITY = 0.15

# For every entry in MEDICATION_DF, add allergy/allergies with ALLERGY_PROBABILITY
ALLERGY_PROBABILITY = 0.05
WRONG_ALLERGY_PROBABILITY = 0.25

MEDICATION_DF_SAMPLE_SIZE = 1000

###############################################################################

MEDICATION_DF = pd.read_csv(os.path.join(SYNTHEA_DATA_DIR, "medications.csv"))

ALLERGY_DF = pd.read_csv(os.path.join(SYNTHEA_DATA_DIR, "allergies.csv"))

# Unique Rows only
INGREDIENT_DF = MEDICATION_DF[["CODE", "DESCRIPTION"]].drop_duplicates()

MEDICATION_DATA_RAW = {}
MEDICATION_INFO = {}

# Reduce Medication Data Size
MEDICATION_DF = MEDICATION_DF.sample(MEDICATION_DF_SAMPLE_SIZE, random_state=RNG)

###############################################################################
# Reference: https://www.nlm.nih.gov/research/umls/rxnorm/overview.html

# IN - Ingredient
# SCDC - Semantic Clinical Drug Component - Ingredient + Strength
# SCDF - Semantic Clinical Drug Form - Ingredient + Dose Form
# SCD - Semantic Clinical Drug - Ingredient + Strength + Dose Form
# SBDC - Semantic Branded Drug Component - Ingredient + Strength + Brand Name
# SBD - Semantic Branded Drug - Ingredient + Strength + Dose Form + Brand Name
# BN - Brand Name
# PSN - Prescribable Name
# etc.

SESSION = requests_cache.CachedSession('rxnav_cache')


def get_drug_information(rxcui):
    url = f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/allrelated.json"
    response = SESSION.get(url)

    if response.status_code != 200:
        return f"Error: Unable to fetch data (Status Code: {response.status_code})"

    data = response.json()

    # Store
    MEDICATION_DATA_RAW[rxcui] = data

    extract_keys = ["IN", "PIN", "BN", "SCD", "SBD", "SBDG", "SBDF", "PSN", "BPCK"]
    result = {key: [] for key in extract_keys}

    # Extract ingredient names
    for group in data.get("allRelatedGroup", {}).get("conceptGroup", []):
        for extract_key in extract_keys:
            if group.get("tty") == extract_key:
                for concept in group.get("conceptProperties", []):
                    if concept.get("synonym"):
                        result[extract_key].append(concept.get("synonym"))
                    if concept.get("name"):
                        result[extract_key].append(concept.get("name"))


    if rxcui == 757594:
        result["IN"] = ["norethindrone"]
        result["PSN"] = "Jolivette 28 Day Pack"

    if rxcui == 235389:
        result["IN"] = ["mestranol", "norethynodrel"]
        result["PSN"] = "Enovid Tablet"

    if rxcui == 749882:
        result["IN"] = ["mestranol", "norethisterone"]
        result["PSN"] = "Norinyl 1+50 28 Day Pack"

    if rxcui == 204892:
        result["IN"] = ["clonazepam"]
        result["PSN"] = "Klonopin"

    MEDICATION_INFO[rxcui] = result

    return result


def get_ingredient(medication_code):
    drug_information = get_drug_information(medication_code)
    return drug_information["IN"]

def get_medication_names(medication_code):
    drug_information = MEDICATION_INFO[medication_code]

    if drug_information.get("PSN"):
        return [drug_information["PSN"]]
    elif drug_information.get("BPCK"):
        return drug_information["BPCK"]
    elif drug_information.get("SBD"):
        return drug_information["SBD"]
    elif drug_information.get("SCD"):
        return drug_information["SCD"]
    else:
        print(drug_information)
        raise RuntimeError("WHOOPS!")

def get_medication_name(medication_code):
    # # Random
    # return RNG.choice(get_medication_names(medication_code))
    # Unique ("Best Choice")
    return get_medication_names(medication_code)[0]


###############################################################################

def get_inconsistent_ingredient(medication_code):
    correct_ingredients = get_ingredient(medication_code)
    wrong_ingredients = []
    for _ in correct_ingredients:
        if RNG.rand() < INCORRECT_INGREDIENT_PROBABILITY:
            while True:
                # CAUTION: ALL_INGREDIENTS must be generated BEFORE the call to this function
                wrong_ingredient = RNG.choice(ALL_INGREDIENTS)
                if wrong_ingredient not in correct_ingredients:
                    wrong_ingredients.append(wrong_ingredient)
                    break
    return [(i, 0) for i in correct_ingredients] + [(str(i), 1) for i in wrong_ingredients]

###############################################################################

ALL_INGREDIENTS = set()
for rxcui in INGREDIENT_DF["CODE"]:
    ALL_INGREDIENTS.update(get_ingredient(rxcui))
ALL_INGREDIENTS = list(ALL_INGREDIENTS)

###############################################################################

INGREDIENT_DF["DESCRIPTION"] = INGREDIENT_DF["CODE"].apply(get_medication_name)
# MEDICATION_DF["DESCRIPTION"] = MEDICATION_DF["CODE"].apply(get_medication_name)

INGREDIENT_DF["INGREDIENT_DATA"] = INGREDIENT_DF["CODE"].apply(get_inconsistent_ingredient)
INGREDIENT_DF = INGREDIENT_DF.explode("INGREDIENT_DATA", ignore_index=True)

INGREDIENT_DF[["INGREDIENT", "IS_ERROR"]] = pd.DataFrame(
    INGREDIENT_DF["INGREDIENT_DATA"].tolist(),
    index=INGREDIENT_DF.index
)
INGREDIENT_DF.drop(columns=["INGREDIENT_DATA"], inplace=True)


###############################################################################

ALLERGY_INDEX = itertools.count(420000,7)
ALLERGY_ITEMS = {
    ingredient: [next(ALLERGY_INDEX), f"Allergy to {ingredient}"]
    for ingredient in ALL_INGREDIENTS
}

ALLERGY_DATA = [["START", "STOP", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION", "INGREDIENT", "IS_ERROR"]]

for _, (patient, encounter, code) in tqdm(MEDICATION_DF[["PATIENT", "ENCOUNTER", "CODE"]].iterrows()):
    if RNG.rand() < ALLERGY_PROBABILITY:
        correct_ingredients = list(INGREDIENT_DF.query(f"CODE == {code} and IS_ERROR == 0")["INGREDIENT"])
        incorrect_ingredients = list(INGREDIENT_DF.query(f"CODE == {code} and IS_ERROR == 1")["INGREDIENT"])

        if RNG.rand() < WRONG_ALLERGY_PROBABILITY:
            # if allergy edge is to be wrong: ingredient is real
            ingredients = correct_ingredients
            allergy_is_error = 1
        else:
            # else allergy to "non-ingredient"
            ingredients = incorrect_ingredients
            allergy_is_error = 0

        if ingredients:
            allergy_ingredient = RNG.choice(ingredients)
            allergy_code, allergy_description = ALLERGY_ITEMS[allergy_ingredient]

            # IS_ERROR is redundant information in ALLERGY_DF (since it's dependent on IS_ERROR of INGREDIENT_DF)
            # but it may reduce evaluation effort
            ALLERGY_DATA.append(["", "", patient, encounter, allergy_code, allergy_description, allergy_ingredient, allergy_is_error])

ALLERGY_DF = pd.DataFrame(ALLERGY_DATA[1:], columns=ALLERGY_DATA[0])

###############################################################################


MEDICATION_DF.to_csv(os.path.join(SYNTHEA_DATA_DIR, "custom.medications.csv"), index=False)
INGREDIENT_DF.to_csv(os.path.join(SYNTHEA_DATA_DIR, "custom.medications.ingredients.csv"), index=False)
ALLERGY_DF.to_csv(os.path.join(SYNTHEA_DATA_DIR, "custom.allergies.ingredients.csv"), index=False)
