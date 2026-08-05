"""
Evaluation report generation for RoadAccidentAI.

This module exports evaluation results into multiple formats
including JSON, CSV, TXT and Markdown.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from road_accident_detection.evaluation.performance import (
    PerformanceSummary,
)

logger = logging.getLogger(__name__)


class EvaluationReport:
    """
    Export evaluation results.
    """

    def __init__(
        self,
        output_directory: str | Path,
    ) -> None:

        self.output_directory = Path(output_directory)

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        logger.info(
            "Evaluation reports will be saved to %s",
            self.output_directory,
        )

    @staticmethod
    def _convert(
        summary: PerformanceSummary,
    ) -> dict[str, Any]:

        return summary.to_dict()

    def export_json(
        self,
        summary: PerformanceSummary,
        filename: str = "results.json",
    ) -> Path:
        """
        Export evaluation summary to JSON.
        """

        output = self.output_directory / filename

        with output.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self._convert(summary),
                file,
                indent=4,
            )

        logger.info(
            "JSON report saved to %s",
            output,
        )

        return output

    def export_csv(
        self,
        summary: PerformanceSummary,
        filename: str = "results.csv",
    ) -> Path:
        """
        Export evaluation summary to CSV.
        """

        output = self.output_directory / filename

        with output.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(file)

            writer.writerow(
                [
                    "Metric",
                    "Value",
                ]
            )

            for key, value in summary.to_dict().items():

                writer.writerow(
                    [
                        key,
                        value,
                    ]
                )

        logger.info(
            "CSV report saved to %s",
            output,
        )

        return output

    def export_markdown(
        self,
        summary: PerformanceSummary,
        filename: str = "results.md",
    ) -> Path:
        """
        Export Markdown report.
        """

        output = self.output_directory / filename

        with output.open(
            "w",
            encoding="utf-8",
        ) as file:

            file.write(
                "# RoadAccidentAI Evaluation Report\n\n"
            )

            file.write(
                summary.markdown_table()
            )

            file.write("\n")

        logger.info(
            "Markdown report saved to %s",
            output,
        )

        return output

    def export_text(
        self,
        summary: PerformanceSummary,
        filename: str = "results.txt",
    ) -> Path:
        """
        Export plain text report.
        """

        output = self.output_directory / filename

        with output.open(
            "w",
            encoding="utf-8",
        ) as file:

            file.write(
                "RoadAccidentAI Evaluation Report\n"
            )

            file.write("=" * 45)

            file.write("\n\n")

            for key, value in summary.to_dict().items():

                file.write(
                    f"{key:<35}: {value}\n"
                )

        logger.info(
            "Text report saved to %s",
            output,
        )

        return output

    def export_latex(
        self,
        summary: PerformanceSummary,
        filename: str = "results.tex",
    ) -> Path:
        """
        Export LaTeX performance table.
        """

        output = self.output_directory / filename

        with output.open(
            "w",
            encoding="utf-8",
        ) as file:

            file.write(
                summary.latex_table()
            )

        logger.info(
            "LaTeX table saved to %s",
            output,
        )

        return output

    def export_all(
        self,
        summary: PerformanceSummary,
    ) -> dict[str, Path]:
        """
        Export all report formats.
        """

        reports = {

            "json": self.export_json(
                summary,
            ),

            "csv": self.export_csv(
                summary,
            ),

            "markdown": self.export_markdown(
                summary,
            ),

            "text": self.export_text(
                summary,
            ),

            "latex": self.export_latex(
                summary,
            ),

        }

        logger.info(
            "Generated %d report files.",
            len(reports),
        )

        return reports