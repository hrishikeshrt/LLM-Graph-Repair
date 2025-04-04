# LLMs for Graph Repair

## Components

0. `connect.py`: Defines `Graph` class which manages connection to Neo4j Graph Database
1. `dataset.py`: Defines `GraphDataset` class, building on `Graph`, to provide functions for loading data
- `add_inconsistency_synthea.py`: Add controlled inconsistencies to Synthea Dataset
- `dataset_synthea.py`: Queries for Synthea Dataset
- `load_synthea.py`: Load a dataset
2. `graph.py`: Extends `networkx.DiGraph` to define `PropertyGraph` class,
3. `inconsistency.py`: Find inconsistencies and store them in a pickle with `PropertyGraph` format
4. `encoding.py`: Provides functions for computing text representations of a PG
5. `llm.py`: Provides functions for connecting to LLMs and asking questions and getting answers
6. `machine_repair.py`: Ask LLM to repair the graph
7. `response_statistics.py`: Prepare response statistics

## Pipeline

* Load dataset using `python3 load_synthea.py`
* Find inconsistencies using `python3 inconsistency.py`
* Control repair parameters in `machine_repair.py`
* Query LLMs for graph repair using `python3 machine_repair.py`
* Prepare response statistics (generating tables, plots) using `python3 response_statistics.py`

## License

This project is licensed under the terms of the GNU General Public License v3.0.
See the [LICENSE](./LICENSE) file for details.