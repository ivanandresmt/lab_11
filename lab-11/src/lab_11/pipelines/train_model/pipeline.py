"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.18.11
"""

# from kedro.pipeline import Pipeline, node, pipeline


# def create_pipeline(**kwargs) -> Pipeline:
#     return pipeline([])

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, split_data, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs={"data": "model_input_table", "params": "train_model"},
                outputs=[
                    "X_train",
                    "X_valid",
                    "X_test",
                    "y_train",
                    "y_valid",
                    "y_test",
                ],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "X_valid", "y_train", "y_valid"],
                outputs="best_model",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["best_model", "X_test", "y_test"],
                outputs=None,
                name="evaluate_model_node",
            ),
        ]
    )
