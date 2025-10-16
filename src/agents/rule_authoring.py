"""LLM-powered rule authoring system for converting natural language to rules."""
from typing import Any, Dict, List, Optional
import logging
import json
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.validators.rule_engine_v2 import RuleV2
from src.validators.rule_dsl import (
    rule,
    data,
    AND,
    OR,
    NOT,
    ERROR,
    WARNING,
    INFO,
    RuleDSLParser,
)

logger = logging.getLogger(__name__)


class RuleGenerationRequest(BaseModel):
    """Request for generating a rule from natural language."""

    description: str = Field(..., description="Natural language description of the rule")
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional context about data structure"
    )
    examples: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional example data"
    )


class RuleGenerationResponse(BaseModel):
    """Response from rule generation."""

    rule_name: str = Field(..., description="Generated rule name")
    rule_code: str = Field(..., description="Python code for the rule")
    explanation: str = Field(..., description="Explanation of the rule logic")
    test_cases: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Generated test cases"
    )


class TestCaseGenerationRequest(BaseModel):
    """Request for generating test cases for a rule."""

    rule: RuleV2 = Field(..., description="Rule to generate tests for")
    num_cases: int = Field(default=5, description="Number of test cases to generate")


class RuleExplanationRequest(BaseModel):
    """Request for explaining a rule violation."""

    rule: RuleV2 = Field(..., description="Rule that was violated")
    data: Dict[str, Any] = Field(..., description="Data that violated the rule")
    context: Optional[str] = Field(default=None, description="Additional context")


class RuleAuthoringAgent:
    """LLM-powered agent for authoring and managing rules."""

    def __init__(self, llm: BaseChatModel):
        """Initialize rule authoring agent.

        Args:
            llm: Language model for rule generation
        """
        self.llm = llm
        self.parser = RuleDSLParser()

    def generate_rule_from_nl(
        self,
        description: str,
        context: Optional[Dict[str, Any]] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
    ) -> RuleGenerationResponse:
        """Generate a rule from natural language description.

        Args:
            description: Natural language description of the rule
            context: Optional context about data structure
            examples: Optional example data

        Returns:
            RuleGenerationResponse with generated rule

        Example:
            ```python
            response = agent.generate_rule_from_nl(
                "Users must be at least 18 years old",
                context={"fields": {"age": "integer"}},
            )
            ```
        """
        system_prompt = """You are an expert at converting natural language business rules into Python code using a rule DSL.

The DSL supports:
- Field access: data.field_name or data['nested.field']
- Comparisons: ==, !=, <, <=, >, >=
- String operations: contains(), starts_with(), ends_with(), matches()
- Membership: is_in([...]), not_in([...])
- Logical operators: AND(...), OR(...), NOT(...)
- Severity levels: ERROR, WARNING, INFO

Example rule structure:
```python
rule("rule_name")
    .when(data.age >= 18)
    .then("Must be 18 or older")
    .severity(ERROR)
    .tag("age_validation")
    .build()
```

Generate complete, executable Python code using this DSL.
Return response as JSON with: rule_name, rule_code, explanation, test_cases.
"""

        context_str = f"\nData structure context: {json.dumps(context, indent=2)}" if context else ""
        examples_str = f"\nExample data: {json.dumps(examples, indent=2)}" if examples else ""

        user_prompt = f"""Generate a validation rule for this requirement:

{description}
{context_str}
{examples_str}

Generate:
1. A descriptive rule name (snake_case)
2. Complete Python code using the rule DSL
3. Explanation of the rule logic
4. At least 3 test cases (should_pass=True/False, data={...})

Return as JSON:
{{
    "rule_name": "...",
    "rule_code": "...",
    "explanation": "...",
    "test_cases": [
        {{"should_pass": true, "data": {{}}, "description": "..."}},
        ...
    ]
}}
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            content = response.content

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return RuleGenerationResponse(**result)
            else:
                raise ValueError("Could not extract JSON from LLM response")

        except Exception as e:
            logger.error(f"Error generating rule: {e}")
            # Return a fallback response
            return RuleGenerationResponse(
                rule_name="generated_rule",
                rule_code=f"# Error generating rule: {e}",
                explanation=f"Failed to generate rule: {str(e)}",
                test_cases=[],
            )

    def generate_test_cases(self, rule: RuleV2, num_cases: int = 5) -> List[Dict[str, Any]]:
        """Generate test cases for a rule.

        Args:
            rule: Rule to generate tests for
            num_cases: Number of test cases to generate

        Returns:
            List of test case dictionaries

        Example:
            ```python
            test_cases = agent.generate_test_cases(my_rule, num_cases=10)
            for tc in test_cases:
                print(tc['description'], tc['should_pass'])
            ```
        """
        system_prompt = """You are an expert at generating comprehensive test cases for validation rules.

Generate diverse test cases that cover:
- Happy path (valid data)
- Edge cases (boundary values)
- Invalid data (should fail)
- Missing fields
- Wrong data types
- Extreme values

Return test cases as JSON array.
"""

        user_prompt = f"""Generate {num_cases} test cases for this rule:

Rule Name: {rule.name}
Error Message: {rule.error_message}
Severity: {rule.severity}
Tags: {list(rule.tags)}

Generate diverse test cases covering happy path, edge cases, and failure cases.

Return as JSON array:
[
    {{
        "description": "Test case description",
        "should_pass": true/false,
        "data": {{...}},
        "reason": "Why this should pass/fail"
    }},
    ...
]
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            content = response.content

            # Extract JSON array from response
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                test_cases = json.loads(json_match.group())
                return test_cases[:num_cases]  # Limit to requested number
            else:
                logger.warning("Could not extract JSON from LLM response")
                return []

        except Exception as e:
            logger.error(f"Error generating test cases: {e}")
            return []

    def explain_violation(
        self,
        rule: RuleV2,
        data: Dict[str, Any],
        context: Optional[str] = None,
    ) -> str:
        """Explain why a rule was violated in plain language.

        Args:
            rule: Rule that was violated
            data: Data that violated the rule
            context: Optional additional context

        Returns:
            Human-readable explanation

        Example:
            ```python
            explanation = agent.explain_violation(
                my_rule,
                {"age": 15},
                context="User registration"
            )
            print(explanation)
            ```
        """
        system_prompt = """You are an expert at explaining validation failures in clear, user-friendly language.

Provide helpful, actionable explanations that:
- Explain what went wrong
- Why the rule exists
- How to fix the issue
- Are empathetic and professional
"""

        context_str = f"\nContext: {context}" if context else ""

        user_prompt = f"""Explain why this validation rule failed:

Rule: {rule.name}
Error Message: {rule.error_message}
Severity: {rule.severity}

Data that failed:
{json.dumps(data, indent=2)}
{context_str}

Provide a clear, helpful explanation for the user.
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            return response.content.strip()

        except Exception as e:
            logger.error(f"Error explaining violation: {e}")
            return f"Validation failed: {rule.error_message}"

    def suggest_improvements(self, rule: RuleV2) -> List[str]:
        """Suggest improvements for a rule.

        Args:
            rule: Rule to analyze

        Returns:
            List of improvement suggestions

        Example:
            ```python
            suggestions = agent.suggest_improvements(my_rule)
            for suggestion in suggestions:
                print(f"- {suggestion}")
            ```
        """
        system_prompt = """You are an expert at analyzing validation rules and suggesting improvements.

Consider:
- Performance optimizations
- Better error messages
- Missing edge cases
- Overly strict/lenient conditions
- Clarity and maintainability
"""

        user_prompt = f"""Analyze this validation rule and suggest improvements:

Rule Name: {rule.name}
Error Message: {rule.error_message}
Severity: {rule.severity}
Priority: {rule.priority}
Tags: {list(rule.tags)}
Dependencies: {list(rule.dependencies)}

Metrics:
- Evaluations: {rule._evaluation_count}
- Avg Time: {rule.average_time_ms:.2f}ms
- Failure Rate: {rule.failure_rate:.2%}

Provide 3-5 specific, actionable improvement suggestions.
Return as JSON array of strings.
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            content = response.content

            # Extract JSON array
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                suggestions = json.loads(json_match.group())
                return suggestions
            else:
                # Fallback: split by newlines
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                return [line.lstrip('- •*').strip() for line in lines if line]

        except Exception as e:
            logger.error(f"Error suggesting improvements: {e}")
            return []

    def parse_rule_text(self, rule_text: str) -> RuleV2:
        """Parse a text-based rule into a RuleV2 object.

        Args:
            rule_text: Text description of rule

        Returns:
            RuleV2 instance

        Example:
            ```python
            rule = agent.parse_rule_text("age >= 18")
            ```
        """
        try:
            condition = self.parser.parse(rule_text)

            # Generate rule name from text
            rule_name = re.sub(r'[^a-z0-9]+', '_', rule_text.lower())[:50]

            # Create rule
            return RuleV2(
                name=rule_name,
                condition=condition,
                error_message=f"Condition not met: {rule_text}",
                severity="error",
            )

        except Exception as e:
            logger.error(f"Error parsing rule text: {e}")
            raise ValueError(f"Could not parse rule: {rule_text}") from e

    def optimize_rule_set(self, rules: List[RuleV2]) -> Dict[str, Any]:
        """Analyze and suggest optimizations for a set of rules.

        Args:
            rules: List of rules to analyze

        Returns:
            Dictionary with optimization suggestions

        Example:
            ```python
            optimizations = agent.optimize_rule_set(all_rules)
            print(optimizations['summary'])
            ```
        """
        system_prompt = """You are an expert at analyzing and optimizing validation rule sets.

Analyze for:
- Redundant rules
- Conflicting rules
- Performance bottlenecks
- Missing validations
- Logical groupings
- Priority ordering
"""

        # Gather analytics
        total_rules = len(rules)
        enabled_rules = sum(1 for r in rules if r.enabled)
        avg_time = sum(r.average_time_ms for r in rules) / total_rules if total_rules > 0 else 0

        slowest = sorted(rules, key=lambda r: r.average_time_ms, reverse=True)[:5]
        most_failing = sorted(rules, key=lambda r: r.failure_rate, reverse=True)[:5]

        user_prompt = f"""Analyze this rule set and provide optimization recommendations:

Total Rules: {total_rules}
Enabled: {enabled_rules}
Average Evaluation Time: {avg_time:.2f}ms

Slowest Rules:
{chr(10).join(f"- {r.name}: {r.average_time_ms:.2f}ms" for r in slowest)}

Most Failing Rules:
{chr(10).join(f"- {r.name}: {r.failure_rate:.2%}" for r in most_failing)}

Provide:
1. Overall assessment
2. Performance optimization suggestions
3. Rule organization recommendations
4. Potential conflicts or redundancies
5. Priority adjustments

Return as JSON:
{{
    "summary": "...",
    "performance_tips": [...],
    "organization_tips": [...],
    "conflicts": [...],
    "priority_suggestions": [...]
}}
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            content = response.content

            # Extract JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {
                    "summary": content,
                    "performance_tips": [],
                    "organization_tips": [],
                    "conflicts": [],
                    "priority_suggestions": [],
                }

        except Exception as e:
            logger.error(f"Error optimizing rule set: {e}")
            return {
                "summary": f"Error analyzing rules: {str(e)}",
                "performance_tips": [],
                "organization_tips": [],
                "conflicts": [],
                "priority_suggestions": [],
            }
