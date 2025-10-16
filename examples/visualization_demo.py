"""Visualization demonstration for HVAS-Mini workflow.

This example demonstrates:
1. Exporting workflow graph to different formats
2. Generating interactive HTML visualizations
3. Viewing workflow structure
"""

from src.graph.workflow import ValidationWorkflow
from src.visualization.graph_export import (
    export_graph_to_mermaid,
    generate_graph_html,
    print_graph_ascii,
)


def main():
    """Run visualization demo."""
    print("=" * 60)
    print("HVAS-Mini: Workflow Visualization Demo")
    print("=" * 60)

    # Create workflow
    print("\n1. Creating validation workflow...")
    workflow = ValidationWorkflow()
    print("   ✓ Workflow created")

    # Display ASCII representation
    print("\n2. ASCII Graph Representation:")
    print("-" * 60)
    print_graph_ascii(workflow)

    # Export to Mermaid format
    print("\n3. Exporting to Mermaid format...")
    mermaid_diagram = export_graph_to_mermaid(workflow)
    with open("workflow_graph.mmd", "w") as f:
        f.write(mermaid_diagram)
    print("   ✓ Mermaid diagram saved to: workflow_graph.mmd")
    print("   (Can be visualized at https://mermaid.live/)")

    # Generate interactive HTML
    print("\n4. Generating interactive HTML visualization...")
    html_viz = generate_graph_html(workflow, "workflow_visualization.html")
    print("   ✓ HTML visualization saved to: workflow_visualization.html")
    print("   (Open in a web browser to view)")

    # Display Mermaid code snippet
    print("\n5. Mermaid Graph Preview:")
    print("-" * 60)
    print(mermaid_diagram[:400] + "...")

    print("\n6. Workflow Structure Information:")
    print("-" * 60)
    print("   Nodes (Agents):")
    print("   - Supervisor: Orchestrates and routes validation tasks")
    print("   - Schema Validator: Validates JSON schema compliance")
    print("   - Business Rules: Applies domain-specific business logic")
    print("   - Data Quality: Checks data completeness and quality")
    print("   - Aggregator: Synthesizes results and generates report")
    print()
    print("   Edges (Transitions):")
    print("   - Conditional routing based on validation requirements")
    print("   - Sequential validator execution")
    print("   - Final aggregation of all results")

    print("\n7. Visualization Options:")
    print("-" * 60)
    print("   ✓ ASCII art (terminal display)")
    print("   ✓ Mermaid diagram (.mmd file)")
    print("   ✓ Interactive HTML (web browser)")
    print("   ✓ PNG export (requires graphviz)")

    # Note about LangGraph Studio
    print("\n8. LangGraph Studio Integration:")
    print("-" * 60)
    print("   To view in LangGraph Studio:")
    print("   1. Ensure LangGraph Studio is installed")
    print("   2. Run: uv run langgraph studio")
    print("   3. Open browser to http://localhost:8000")
    print("   4. Load your HVAS-Mini project")
    print()
    print("   LangGraph Studio provides:")
    print("   - Interactive graph visualization")
    print("   - Step-by-step execution debugging")
    print("   - State inspection at each node")
    print("   - Execution replay and analysis")

    print("\n" + "=" * 60)
    print("Visualization demo complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - workflow_graph.mmd")
    print("  - workflow_visualization.html")
    print()


if __name__ == "__main__":
    main()
