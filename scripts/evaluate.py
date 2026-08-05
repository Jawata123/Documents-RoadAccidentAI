"""
RoadAccidentAI Evaluation Script.

Run the complete evaluation pipeline.

Example
-------
python scripts/evaluate.py
"""

from __future__ import annotations

import logging
from pathlib import Path

from road_accident_detection.evaluation.evaluator import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def main() -> None:
    """
    Run the complete evaluation pipeline.
    """

    project_root = Path(__file__).resolve().parent.parent

    dataset_directory = (
        project_root
        / "datasets"
    )

    output_directory = (
        project_root
        / "outputs"
    )

    evaluator = Evaluator(
        dataset_directory=dataset_directory,
        output_directory=output_directory,
    )

    evaluator.run()

    logging.info(
        "Evaluation completed successfully."
    )


if __name__ == "__main__":

    main()