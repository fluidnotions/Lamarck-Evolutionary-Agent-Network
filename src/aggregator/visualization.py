"""Visualization support for validation reports."""

import json
from typing import Any, Optional
from src.models import ValidationResult


class Visualizer:
    """
    Generates visualization data and charts for validation results.

    Supports:
    - Pass/fail pie charts
    - Quality score radar charts
    - Error distribution histograms
    - Validator execution timeline
    - Confidence score gauge
    """

    def __init__(self, backend: str = "plotly"):
        """
        Initialize visualizer.

        Args:
            backend: Visualization backend ('plotly', 'matplotlib', or 'data')
        """
        self.backend = backend

    def generate_all_visualizations(
        self,
        results: list[ValidationResult],
        confidence: float,
        error_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Generate all visualization data.

        Args:
            results: Validation results
            confidence: Confidence score
            error_analysis: Error analysis data

        Returns:
            Dictionary containing all visualization data
        """
        return {
            "pass_fail_chart": self.generate_pass_fail_chart(results),
            "severity_distribution": self.generate_severity_distribution(error_analysis),
            "validator_performance": self.generate_validator_performance(results),
            "confidence_gauge": self.generate_confidence_gauge(confidence),
            "error_timeline": self.generate_error_timeline(results),
            "coverage_chart": self.generate_coverage_chart(results),
        }

    def generate_pass_fail_chart(
        self, results: list[ValidationResult]
    ) -> dict[str, Any]:
        """
        Generate pass/fail pie chart data.

        Args:
            results: Validation results

        Returns:
            Chart data for pass/fail distribution
        """
        passed = sum(1 for r in results if r.status == "passed")
        failed = sum(1 for r in results if r.status == "failed")
        skipped = sum(1 for r in results if r.status == "skipped")
        error = sum(1 for r in results if r.status == "error")

        data = {
            "type": "pie",
            "title": "Validation Status Distribution",
            "data": {
                "labels": ["Passed", "Failed", "Skipped", "Error"],
                "values": [passed, failed, skipped, error],
                "colors": ["#28a745", "#dc3545", "#6c757d", "#fd7e14"],
            },
        }

        if self.backend == "plotly":
            return self._to_plotly_pie(data)
        elif self.backend == "matplotlib":
            return self._to_matplotlib_instructions(data)
        else:
            return data

    def generate_severity_distribution(
        self, error_analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Generate error severity distribution chart.

        Args:
            error_analysis: Error analysis data

        Returns:
            Chart data for severity distribution
        """
        by_severity = error_analysis.get("by_severity", {})

        labels = []
        values = []
        colors_map = {
            "critical": "#dc3545",
            "error": "#fd7e14",
            "warning": "#ffc107",
            "info": "#17a2b8",
        }
        colors = []

        for severity in ["critical", "error", "warning", "info"]:
            if severity in by_severity:
                labels.append(severity.upper())
                values.append(by_severity[severity]["count"])
                colors.append(colors_map[severity])

        data = {
            "type": "bar",
            "title": "Error Severity Distribution",
            "data": {
                "labels": labels,
                "values": values,
                "colors": colors,
            },
        }

        if self.backend == "plotly":
            return self._to_plotly_bar(data)
        elif self.backend == "matplotlib":
            return self._to_matplotlib_instructions(data)
        else:
            return data

    def generate_validator_performance(
        self, results: list[ValidationResult]
    ) -> dict[str, Any]:
        """
        Generate validator performance comparison chart.

        Args:
            results: Validation results

        Returns:
            Chart data for validator performance
        """
        validators = [r.validator_name for r in results]
        execution_times = [r.execution_time for r in results]
        error_counts = [len(r.errors) for r in results]

        data = {
            "type": "grouped_bar",
            "title": "Validator Performance",
            "data": {
                "validators": validators,
                "execution_times": execution_times,
                "error_counts": error_counts,
            },
        }

        if self.backend == "plotly":
            return self._to_plotly_grouped_bar(data)
        elif self.backend == "matplotlib":
            return self._to_matplotlib_instructions(data)
        else:
            return data

    def generate_confidence_gauge(self, confidence: float) -> dict[str, Any]:
        """
        Generate confidence score gauge.

        Args:
            confidence: Confidence score (0.0 to 1.0)

        Returns:
            Chart data for confidence gauge
        """
        # Determine gauge color based on confidence
        if confidence >= 0.9:
            color = "#28a745"
            level = "Excellent"
        elif confidence >= 0.75:
            color = "#5cb85c"
            level = "Good"
        elif confidence >= 0.5:
            color = "#ffc107"
            level = "Fair"
        elif confidence >= 0.25:
            color = "#fd7e14"
            level = "Poor"
        else:
            color = "#dc3545"
            level = "Critical"

        data = {
            "type": "gauge",
            "title": "Confidence Score",
            "data": {
                "value": confidence,
                "percentage": confidence * 100,
                "level": level,
                "color": color,
            },
        }

        if self.backend == "plotly":
            return self._to_plotly_gauge(data)
        elif self.backend == "matplotlib":
            return self._to_matplotlib_instructions(data)
        else:
            return data

    def generate_error_timeline(
        self, results: list[ValidationResult]
    ) -> dict[str, Any]:
        """
        Generate error timeline showing when errors occurred.

        Args:
            results: Validation results

        Returns:
            Chart data for error timeline
        """
        timeline_data = []

        for result in results:
            for error in result.errors:
                timeline_data.append({
                    "timestamp": error.timestamp.isoformat(),
                    "validator": result.validator_name,
                    "severity": error.severity,
                    "message": error.message,
                })

        # Sort by timestamp
        timeline_data.sort(key=lambda x: x["timestamp"])

        data = {
            "type": "timeline",
            "title": "Error Timeline",
            "data": timeline_data,
        }

        if self.backend == "plotly":
            return self._to_plotly_timeline(data)
        elif self.backend == "matplotlib":
            return self._to_matplotlib_instructions(data)
        else:
            return data

    def generate_coverage_chart(
        self, results: list[ValidationResult]
    ) -> dict[str, Any]:
        """
        Generate coverage chart showing validation coverage per validator.

        Args:
            results: Validation results

        Returns:
            Chart data for coverage
        """
        validators = [r.validator_name for r in results]
        coverage = [r.coverage * 100 for r in results]  # Convert to percentage

        data = {
            "type": "bar",
            "title": "Validation Coverage by Validator",
            "data": {
                "labels": validators,
                "values": coverage,
                "colors": ["#17a2b8"] * len(validators),
            },
        }

        if self.backend == "plotly":
            return self._to_plotly_bar(data)
        elif self.backend == "matplotlib":
            return self._to_matplotlib_instructions(data)
        else:
            return data

    def _to_plotly_pie(self, data: dict[str, Any]) -> dict[str, Any]:
        """Convert to Plotly pie chart format."""
        chart_data = data["data"]
        return {
            "type": "plotly",
            "spec": {
                "data": [
                    {
                        "type": "pie",
                        "labels": chart_data["labels"],
                        "values": chart_data["values"],
                        "marker": {"colors": chart_data["colors"]},
                    }
                ],
                "layout": {
                    "title": data["title"],
                },
            },
        }

    def _to_plotly_bar(self, data: dict[str, Any]) -> dict[str, Any]:
        """Convert to Plotly bar chart format."""
        chart_data = data["data"]
        return {
            "type": "plotly",
            "spec": {
                "data": [
                    {
                        "type": "bar",
                        "x": chart_data["labels"],
                        "y": chart_data["values"],
                        "marker": {"color": chart_data["colors"]},
                    }
                ],
                "layout": {
                    "title": data["title"],
                    "xaxis": {"title": "Category"},
                    "yaxis": {"title": "Count"},
                },
            },
        }

    def _to_plotly_grouped_bar(self, data: dict[str, Any]) -> dict[str, Any]:
        """Convert to Plotly grouped bar chart format."""
        chart_data = data["data"]
        return {
            "type": "plotly",
            "spec": {
                "data": [
                    {
                        "type": "bar",
                        "name": "Execution Time (s)",
                        "x": chart_data["validators"],
                        "y": chart_data["execution_times"],
                    },
                    {
                        "type": "bar",
                        "name": "Error Count",
                        "x": chart_data["validators"],
                        "y": chart_data["error_counts"],
                    },
                ],
                "layout": {
                    "title": data["title"],
                    "barmode": "group",
                    "xaxis": {"title": "Validator"},
                    "yaxis": {"title": "Value"},
                },
            },
        }

    def _to_plotly_gauge(self, data: dict[str, Any]) -> dict[str, Any]:
        """Convert to Plotly gauge chart format."""
        chart_data = data["data"]
        return {
            "type": "plotly",
            "spec": {
                "data": [
                    {
                        "type": "indicator",
                        "mode": "gauge+number+delta",
                        "value": chart_data["percentage"],
                        "title": {"text": data["title"]},
                        "gauge": {
                            "axis": {"range": [0, 100]},
                            "bar": {"color": chart_data["color"]},
                            "steps": [
                                {"range": [0, 25], "color": "#ffe0e0"},
                                {"range": [25, 50], "color": "#fff4e0"},
                                {"range": [50, 75], "color": "#fff9e0"},
                                {"range": [75, 100], "color": "#e0ffe0"},
                            ],
                        },
                    }
                ],
                "layout": {
                    "height": 400,
                },
            },
        }

    def _to_plotly_timeline(self, data: dict[str, Any]) -> dict[str, Any]:
        """Convert to Plotly timeline format."""
        chart_data = data["data"]

        # Create scatter plot with timestamps
        timestamps = [item["timestamp"] for item in chart_data]
        validators = [item["validator"] for item in chart_data]
        severities = [item["severity"] for item in chart_data]

        # Map severities to colors
        color_map = {"critical": "#dc3545", "error": "#fd7e14", "warning": "#ffc107", "info": "#17a2b8"}
        colors = [color_map.get(s, "#6c757d") for s in severities]

        return {
            "type": "plotly",
            "spec": {
                "data": [
                    {
                        "type": "scatter",
                        "mode": "markers",
                        "x": timestamps,
                        "y": validators,
                        "marker": {
                            "color": colors,
                            "size": 10,
                        },
                        "text": [item["message"] for item in chart_data],
                    }
                ],
                "layout": {
                    "title": data["title"],
                    "xaxis": {"title": "Time"},
                    "yaxis": {"title": "Validator"},
                },
            },
        }

    def _to_matplotlib_instructions(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Return instructions for matplotlib rendering.

        Actual matplotlib rendering would be done by the consumer.
        """
        return {
            "type": "matplotlib",
            "instructions": f"Use matplotlib to render {data['type']} chart",
            "data": data,
        }

    def export_for_external_tool(
        self, visualizations: dict[str, Any]
    ) -> str:
        """
        Export visualization data in a format suitable for external tools.

        Args:
            visualizations: Dictionary of visualization data

        Returns:
            JSON string of visualization data
        """
        return json.dumps(visualizations, indent=2, default=str)
