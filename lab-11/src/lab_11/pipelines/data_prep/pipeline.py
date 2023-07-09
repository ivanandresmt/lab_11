"""
This is a boilerplate pipeline 'data_prep'
generated using Kedro 0.18.11
"""

# from kedro.pipeline import Pipeline, node, pipeline


# def create_pipeline(**kwargs) -> Pipeline:
#     return pipeline([])
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (create_model_input_table, get_data, preprocess_companies,
                    preprocess_shuttles)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=get_data,
                inputs=None,
                outputs=["companies", "reviews", "shuttles"],
                name="get_data_node",
            ),
            node(
                func=preprocess_companies,
                inputs="companies",
                outputs="preprocessed_companies",
                name="preprocess_companies_node",
            ),
            node(
                func=preprocess_shuttles,
                inputs="shuttles",
                outputs="preprocessed_shuttles",
                name="preprocess_shuttles_node",
            ),
            node(
                func=create_model_input_table,
                inputs=(
                    ["reviews", "preprocessed_companies",
                     "preprocessed_shuttles"]
                ),
                outputs="model_input_table",
                name="create_model_input_table_node",
            ),
        ]
    )
