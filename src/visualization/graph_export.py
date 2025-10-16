"""Utilities for exporting and visualizing LangGraph workflows."""
import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def export_graph_to_mermaid(workflow) -> str:
    """Export workflow graph to Mermaid diagram format.

    Args:
        workflow: ValidationWorkflow instance

    Returns:
        Mermaid diagram string
    """
    mermaid = """graph TD
    Start([Start]) --> Supervisor[Supervisor Agent]
    Supervisor --> |Route| SchemaVal[Schema Validator]
    Supervisor --> |Route| BusinessRules[Business Rules]
    Supervisor --> |Route| DataQuality[Data Quality]

    SchemaVal --> |Next| BusinessRules
    SchemaVal --> |Next| DataQuality
    SchemaVal --> |Done| Aggregator[Aggregator Agent]

    BusinessRules --> |Next| SchemaVal
    BusinessRules --> |Next| DataQuality
    BusinessRules --> |Done| Aggregator

    DataQuality --> |Next| SchemaVal
    DataQuality --> |Next| BusinessRules
    DataQuality --> |Done| Aggregator

    Aggregator --> End([End])

    style Supervisor fill:#e1f5ff
    style SchemaVal fill:#fff3e0
    style BusinessRules fill:#fff3e0
    style DataQuality fill:#fff3e0
    style Aggregator fill:#e8f5e9
"""
    return mermaid


def export_graph_to_png(workflow, output_path: str | Path) -> bool:
    """Export workflow graph to PNG image.

    Requires graphviz to be installed.

    Args:
        workflow: ValidationWorkflow instance
        output_path: Path to save PNG file

    Returns:
        True if successful, False otherwise
    """
    try:
        from IPython.display import Image

        # Get the compiled graph
        graph = workflow.get_compiled_graph()

        # Generate graph visualization
        img_data = graph.get_graph().draw_mermaid_png()

        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            f.write(img_data)

        logger.info(f"Graph exported to {output_path}")
        return True

    except ImportError:
        logger.error("graphviz or IPython not installed. Cannot export to PNG.")
        return False
    except Exception as e:
        logger.error(f"Failed to export graph to PNG: {e}")
        return False


def generate_graph_html(workflow, output_path: Optional[str | Path] = None) -> str:
    """Generate interactive HTML visualization of the workflow.

    Args:
        workflow: ValidationWorkflow instance
        output_path: Optional path to save HTML file

    Returns:
        HTML string
    """
    mermaid_diagram = export_graph_to_mermaid(workflow)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>HVAS-Mini Workflow Visualization</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>
        mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
    </script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .mermaid {{
            text-align: center;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>HVAS-Mini Validation Workflow</h1>
        <div class="mermaid">
{mermaid_diagram}
        </div>
        <div>
            <h2>Workflow Description</h2>
            <p>
                The HVAS-Mini validation workflow uses a hierarchical multi-agent architecture:
            </p>
            <ul>
                <li><strong>Supervisor Agent:</strong> Analyzes requests and routes to appropriate validators</li>
                <li><strong>Schema Validator:</strong> Validates data structure and types against JSON schemas</li>
                <li><strong>Business Rules:</strong> Applies domain-specific business logic</li>
                <li><strong>Data Quality:</strong> Checks completeness, accuracy, and consistency</li>
                <li><strong>Aggregator:</strong> Synthesizes results and generates final report</li>
            </ul>
        </div>
    </div>
</body>
</html>"""

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(html)

        logger.info(f"HTML visualization saved to {output_path}")

    return html


def print_graph_ascii(workflow) -> None:
    """Print ASCII representation of the workflow graph.

    Args:
        workflow: ValidationWorkflow instance
    """
    ascii_graph = """
    ┌─────────┐
    │  START  │
    └────┬────┘
         │
    ┌────▼─────────┐
    │  Supervisor  │
    └────┬─────────┘
         │
    ┌────▼──────────────────────────┐
    │  Route to Validators          │
    └──┬────────┬──────────┬────────┘
       │        │          │
    ┌──▼──┐  ┌─▼──┐  ┌───▼────┐
    │Schema│  │Rules│  │Quality │
    └──┬───┘  └─┬──┘  └───┬────┘
       │        │          │
    ┌──▼────────▼──────────▼───┐
    │      Aggregator           │
    └──────────┬─────────────────┘
               │
          ┌────▼────┐
          │   END   │
          └─────────┘
    """
    print(ascii_graph)
