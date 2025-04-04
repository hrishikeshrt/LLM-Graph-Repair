#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load Synthea Dataset

@author: Anonymous Authors
"""

import os

###############################################################################
# Neo4j Connection

SERVER_URI = "neo4j://localhost:7687"
USERNAME = "neo4j"
PASSWORD = ""

###############################################################################
# Data

HOME_DIR = os.path.expanduser("~")
SYNTHEA_DATA_DIR = os.path.join(
    HOME_DIR, "Datasets", "synthea", "synthea_sample_data_csv_apr2020"
)

###############################################################################

SYNTHEA_DATA_QUERIES = {
    "patients.csv": """
        MERGE (p:Patient {id: $Id})
        ON CREATE SET p.birthdate = $BIRTHDATE,
                      p.deathdate = $DEATHDATE,
                      p.first = $FIRST,
                      p.last = $LAST,
                      p.address = $ADDRESS,
                      p.tau_o = 0,
                      p.tau_k = []
    """,

    # # Patients Full
    # "patients.csv": """
    #     MERGE (p:Patient {id: $Id})
    #     ON CREATE SET p.birthdate = $BIRTHDATE,
    #                   p.deathdate = $DEATHDATE,
    #                   p.ssn = $SSN,
    #                   p.drivers = $DRIVERS,
    #                   p.passport = $PASSPORT,
    #                   p.prefix = $PREFIX,
    #                   p.first = $FIRST,
    #                   p.last = $LAST,
    #                   p.suffix = $SUFFIX,
    #                   p.maiden = $MAIDEN,
    #                   p.marital = $MARITAL,
    #                   p.race = $RACE,
    #                   p.ethnicity = $ETHNICITY,
    #                   p.gender = $GENDER,
    #                   p.birthplace = $BIRTHPLACE,
    #                   p.address = $ADDRESS,
    #                   p.city = $CITY,
    #                   p.state = $STATE,
    #                   p.zip = $ZIP,
    #                   p.lat = $LAT,
    #                   p.lon = $LON,
    #                   p.healthcare_expenses = $HEALTHCARE_EXPENSES,
    #                   p.healthcare_coverage = $HEALTHCARE_COVERAGE,
    #                   p.tau_o = 0,
    #                   p.tau_k = []
    # """,

    # "encounters.csv": """
    #     MERGE (e:Encounter {id: $Id})
    #     ON CREATE SET e.start = $START,
    #                   e.stop = $STOP,
    #                   e.patient_id = $PATIENT,
    #                   e.organization_id = $ORGANIZATION,
    #                   e.provider_id = $PROVIDER,
    #                   e.payer_id = $PAYER,
    #                   e.encounter_class = $ENCOUNTERCLASS,
    #                   e.code = $CODE,
    #                   e.description = $DESCRIPTION,
    #                   e.base_encounter_cost = $BASE_ENCOUNTER_COST,
    #                   e.total_claim_cost = $TOTAL_CLAIM_COST,
    #                   e.payer_coverage = $PAYER_COVERAGE,
    #                   e.reason_code = $REASONCODE,
    #                   e.reason_description = $REASONDESCRIPTION
    #     MERGE (p:Patient {id: $PATIENT})
    #     MERGE (p)-[:HAS_ENCOUNTER]->(e)
    # """,

    # "conditions.csv": """
    #     MERGE (c:Condition {code: $CODE, description: $DESCRIPTION, tau_o: 1})
    #     MERGE (p:Patient {id: $PATIENT})
    #     MERGE (p)-[:HAS_CONDITION {start: $START, stop: $STOP, tau_o: 0}]->(c)
    # """,
    #   MERGE (e:Encounter {id: $ENCOUNTER})

    "custom.medications.ingredients.csv": """
        MERGE (m:Medication {code: $CODE, description: $DESCRIPTION, tau_o: 1})
        MERGE (i:Ingredient {id: $INGREDIENT, tau_o: 1})
        MERGE (m)-[:HAS_INGREDIENT {tau_o: 1, is_error: $IS_ERROR}]->(i)
    """,

    "custom.medications.csv": """
        MERGE (m:Medication {code: $CODE})
        MERGE (p:Patient {id: $PATIENT})
        MERGE (p)-[:TAKES_MEDICATION {start: $START, stop: $STOP, tau_o: 0}]->(m)
    """,
    #   MERGE (e:Encounter {id: $ENCOUNTER})

    # "allergies.csv": """
    #     MERGE (a:Allergy {code: $CODE, description: $DESCRIPTION})
    #     MERGE (p:Patient {id: $PATIENT})
    #     MERGE (e:Encounter {id: $ENCOUNTER})
    #     MERGE (p)-[:HAS_ALLERGY {start: $START, stop: $STOP}]->(a)
    # """,

    # NOTE: Allergy nodes are not being used
    "custom.allergies.ingredients.csv": """
        MERGE (a:Allergy {code: $CODE, description: $DESCRIPTION, tau_o: 1})
        MERGE (p:Patient {id: $PATIENT})
        MERGE (i:Ingredient {id: $INGREDIENT})
        MERGE (p)-[:ALLERGIC_TO {start: $START, stop: $STOP, tau_o: 0, is_error: $IS_ERROR}]->(i)
    """,
    #   MERGE (e:Encounter {id: $ENCOUNTER})

    # "procedures.csv": """MERGE (pr:Procedure {code: $CODE, description: $DESCRIPTION})""",

    # "organizations.csv": """
    #     MERGE (o:Organization {id: $Id})
    #     ON CREATE SET o.name = $NAME,
    #                   o.address = $ADDRESS,
    #                   o.city = $CITY,
    #                   o.state = $STATE,
    #                   o.zip = $ZIP,
    #                   o.lat = $LAT,
    #                   o.lon = $LON,
    #                   o.phone = $PHONE,
    #                   o.revenue = $REVENUE,
    #                   o.utilization = $UTILIZATION
    # """,

    # "providers.csv": """
    #     MERGE (prov:Provider {id: $Id})
    #     ON CREATE SET prov.name = $NAME,
    #                   prov.gender = $GENDER,
    #                   prov.speciality = $SPECIALITY,
    #                   prov.address = $ADDRESS,
    #                   prov.city = $CITY,
    #                   prov.state = $STATE, prov.zip = $ZIP,
    #                   prov.lat = $LAT, prov.lon = $LON, prov.utilization = $UTILIZATION
    #     MERGE (o:Organization {id: $ORGANIZATION})
    #     MERGE (prov)-[:WORKS_AT]->(o)
    # """,

    # "immunizations.csv": """MERGE (i:Immunization {code: $CODE, description: $DESCRIPTION})""",

    # "careplans.csv": """MERGE (cp:CarePlan {code: $CODE, description: $DESCRIPTION})""",

    # "imaging_studies.csv": """MERGE (is:ImagingStudy {id: $Id, body_site_code: $BODYSITE_CODE, body_site_description: $BODYSITE_DESCRIPTION})""",
}

###############################################################################

SYNTHEA_ADDITIONAL_QUERIES = [
    # Clean-up
    """
    MATCH (e:Encounter)
    DETACH DELETE e
    """,
    # Setting Visibility
    """
    MATCH (n1)-[r]->(n2)
    WHERE (n1.tau_o = 0 OR n2.tau_o = 0) AND r.tau_o = 1
    SET r.tau_o = 1
    """
]

###############################################################################
# Find Inconsistencies

INCONSISTENCY_QUERIES = [
    (
        "allergy-inconsistency",
        """
        MATCH (p:Patient)-[rm:TAKES_MEDICATION]->(m:Medication)-[rc:HAS_INGREDIENT]->(i:Ingredient),
            (p)-[ra:ALLERGIC_TO]->(i)
        RETURN *
        """,
        """A person should not be treated with a medicine that contains an ingredient
        that the person is allergic to. However, a person (p) (p.first={0}) takes
        a medicine (m) (m.description={1}) which contains an ingredient (i) (i.id={2})
        and (p) is allergic to (i).""",
        [("p","first"), ("m", "description"), ("i", "id")]
    ),
]


###############################################################################
