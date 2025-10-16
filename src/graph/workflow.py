"""LangGraph workflow definition for HVAS-Mini."""
from typing import Any, Dict, Optional
import logging

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig

from src.graph.state import ValidationState, create_initial_state
from src.graph.routing import route_from_supervisor, route_from_validator
from src.agents.supervisor import SupervisorAgent
from src.agents.schema_validator import SchemaValidatorAgent
from src.agents.business_rules import BusinessRulesAgent
from src.agents.data_quality import DataQualityAgent
from src.agents.aggregator import AggregatorAgent

logger = logging.getLogger(__name__)


class ValidationWorkflow:
    """Main validation workflow using LangGraph."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize validation workflow.

        Args:
            config: Optional workflow configuration
        """
        self.config = config or {}

        # Initialize agents
        self.supervisor = SupervisorAgent()
        self.schema_validator = SchemaValidatorAgent()
        self.business_rules = BusinessRulesAgent()
        self.data_quality = DataQualityAgent()
        self.aggregator = AggregatorAgent()

        # Build graph
        self.graph = self._build_graph()
        self.compiled_graph = self.graph.compile()

        logger.info("Validation workflow initialized")

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow.

        Returns:
            StateGraph instance
        """
        # Create graph
        graph = StateGraph(ValidationState)

        # Add nodes
        graph.add_node("supervisor", self.supervisor)
        graph.add_node("schema_validator", self.schema_validator)
        graph.add_node("business_rules", self.business_rules)
        graph.add_node("data_quality", self.data_quality)
        graph.add_node("aggregator", self.aggregator)

        # Set entry point
        graph.set_entry_point("supervisor")

        # Add conditional edges from supervisor
        graph.add_conditional_edges(
            "supervisor",
            route_from_supervisor,
            {
                "schema_validator": "schema_validator",
                "business_rules": "business_rules",
                "data_quality": "data_quality",
                "aggregator": "aggregator",
            },
        )

        # Add conditional edges from each validator
        for validator_name in ["schema_validator", "business_rules", "data_quality"]:
            graph.add_conditional_edges(
                validator_name,
                route_from_validator,
                {
                    "schema_validator": "schema_validator",
                    "business_rules": "business_rules",
                    "data_quality": "data_quality",
                    "aggregator": "aggregator",
                    "END": END,
                },
            )

        # Aggregator leads to END
        graph.add_edge("aggregator", END)

        logger.info("Workflow graph built successfully")
        return graph

    def run(
        self,
        data: Dict[str, Any],
        validators: Optional[list[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        workflow_id: Optional[str] = None,
    ) -> Any:
        """Run the validation workflow.

        Args:
            data: Data to validate
            validators: Optional list of validators to run
            config: Optional validation configuration
            workflow_id: Optional workflow identifier

        Returns:
            Final aggregated result
        """
        logger.info(f"Starting validation workflow for data: {list(data.keys())}")

        # Create initial state
        initial_state = create_initial_state(
            input_data=data,
            validators=validators or ["schema", "business", "quality"],
            config=config,
            workflow_id=workflow_id,
        )

        # Run workflow
        try:
            final_state = self.compiled_graph.invoke(initial_state)

            logger.info(
                f"Workflow completed with status: {final_state.get('overall_status')}"
            )

            return final_state.get("final_report")

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}", exc_info=True)
            raise

    async def arun(
        self,
        data: Dict[str, Any],
        validators: Optional[list[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        workflow_id: Optional[str] = None,
    ) -> Any:
        """Run the validation workflow asynchronously.

        Args:
            data: Data to validate
            validators: Optional list of validators to run
            config: Optional validation configuration
            workflow_id: Optional workflow identifier

        Returns:
            Final aggregated result
        """
        logger.info(f"Starting async validation workflow")

        # Create initial state
        initial_state = create_initial_state(
            input_data=data,
            validators=validators or ["schema", "business", "quality"],
            config=config,
            workflow_id=workflow_id,
        )

        # Run workflow asynchronously
        try:
            final_state = await self.compiled_graph.ainvoke(initial_state)

            logger.info(
                f"Async workflow completed with status: {final_state.get('overall_status')}"
            )

            return final_state.get("final_report")

        except Exception as e:
            logger.error(f"Async workflow execution failed: {e}", exc_info=True)
            raise

    def stream(
        self,
        data: Dict[str, Any],
        validators: Optional[list[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        workflow_id: Optional[str] = None,
    ):
        """Stream validation workflow execution.

        Args:
            data: Data to validate
            validators: Optional list of validators to run
            config: Optional validation configuration
            workflow_id: Optional workflow identifier

        Yields:
            State updates as they occur
        """
        logger.info("Starting streaming validation workflow")

        # Create initial state
        initial_state = create_initial_state(
            input_data=data,
            validators=validators or ["schema", "business", "quality"],
            config=config,
            workflow_id=workflow_id,
        )

        # Stream workflow
        try:
            for state in self.compiled_graph.stream(initial_state):
                yield state

        except Exception as e:
            logger.error(f"Streaming workflow failed: {e}", exc_info=True)
            raise

    def get_graph(self) -> StateGraph:
        """Get the underlying graph.

        Returns:
            StateGraph instance
        """
        return self.graph

    def get_compiled_graph(self):
        """Get the compiled graph.

        Returns:
            Compiled graph
        """
        return self.compiled_graph
