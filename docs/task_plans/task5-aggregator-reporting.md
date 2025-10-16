# Task 5: Aggregator & Reporting
## Implementation Plan

## Objective
Implement the aggregator agent that collects validation results from all validators, calculates confidence scores, generates comprehensive reports, and provides visualization support.

## Dependencies
- Task 1: Core Infrastructure (requires ValidationState, ValidationResult)
- Task 3: Domain Validators (aggregates their results)
- Task 4: Atomic Validators (results to aggregate)

## Components to Implement

### 5.1 Aggregator Agent Core
**File**: `src/agents/aggregator.py`
**Actions**:
- Extend `BaseAgent` class
- Collect results from all validators
- Merge results from parallel executions
- Calculate overall validation status
- Generate final validation report
- Update state with aggregated results
- Add result deduplication logic

### 5.2 Confidence Scoring System
**File**: `src/aggregator/confidence.py`
**Actions**:
- Implement confidence score calculation algorithm
- Weight validators by reliability and domain importance
- Factor in:
  - Number of passing vs failing validators
  - Severity of errors (critical, error, warning, info)
  - Validator execution success rate
  - Data coverage (% of data validated)
- Support custom confidence score formulas
- Provide confidence score breakdown

**Innovation**: Multi-factor confidence scoring that considers validator reliability and error severity

### 5.3 Report Generator
**File**: `src/aggregator/report_generator.py`
**Actions**:
- Generate comprehensive validation reports
- Support multiple output formats:
  - JSON (structured data)
  - Markdown (human-readable)
  - HTML (interactive)
  - PDF (for formal reports)
- Include sections:
  - Executive summary
  - Validation overview (pass/fail counts)
  - Detailed findings per validator
  - Error analysis with grouping
  - Recommendations for remediation
  - Data quality scores
- Add report templates for common scenarios

### 5.4 Error Analysis & Grouping
**File**: `src/aggregator/error_analysis.py`
**Actions**:
- Group errors by type, path, and severity
- Identify error patterns
- Calculate error statistics
- Prioritize errors by impact
- Generate error remediation suggestions using LLM
- Create error dependency graphs (which errors are related)

### 5.5 Visualization Support
**File**: `src/aggregator/visualization.py`
**Actions**:
- Generate validation result visualizations:
  - Pass/fail pie charts
  - Quality score radar charts
  - Error distribution histograms
  - Validator execution timeline
  - Confidence score gauge
- Support LangGraph workflow visualization integration
- Export visualization data for external tools

### 5.6 Result Merging & Deduplication
**File**: `src/aggregator/result_merger.py`
**Actions**:
- Merge results from parallel validator executions
- Deduplicate identical errors from different validators
- Resolve conflicting results (e.g., different validators disagree)
- Aggregate warnings and info messages
- Preserve result provenance (which validator produced which result)

## Testing Strategy

### Unit Tests
**File**: `tests/test_aggregator.py`
- Test result collection from state
- Test state update with aggregated results
- Test handling of empty results
- Test handling of all-pass and all-fail scenarios

**File**: `tests/test_confidence.py`
- Test confidence score calculation with known inputs
- Test weighting logic
- Test edge cases (no validators, all failed, etc.)
- Test custom confidence formulas

**File**: `tests/test_report_generator.py`
- Test report generation in all formats
- Test report content completeness
- Test report with various result combinations
- Test report templates

**File**: `tests/test_error_analysis.py`
- Test error grouping logic
- Test error pattern detection
- Test error prioritization
- Test LLM-based suggestions

### Integration Tests
**File**: `tests/test_aggregator_integration.py`
- Test aggregator in complete workflow
- Test with real validator results
- Test report generation end-to-end
- Test visualization output

## Technical Specifications

### Aggregator Agent
```python
class AggregatorAgent(BaseAgent):
    """Aggregates validation results and generates reports."""

    def __init__(self, llm, report_generator: ReportGenerator):
        super().__init__(
            name="aggregator",
            description="Aggregates results and generates reports"
        )
        self.llm = llm
        self.report_generator = report_generator
        self.confidence_calculator = ConfidenceCalculator()

    def execute(self, state: ValidationState) -> ValidationState:
        """
        Aggregate all validation results and generate final report.

        Steps:
        1. Collect all validation results from state
        2. Merge and deduplicate results
        3. Calculate confidence score
        4. Analyze errors and patterns
        5. Generate comprehensive report
        6. Update state with final results
        """
        results = state["validation_results"]

        # Merge results from parallel executions
        merged_results = self._merge_results(results)

        # Calculate confidence score
        confidence = self.confidence_calculator.calculate(merged_results)

        # Analyze errors
        error_analysis = self._analyze_errors(merged_results)

        # Generate report
        report = self.report_generator.generate(
            results=merged_results,
            confidence=confidence,
            error_analysis=error_analysis
        )

        # Update state
        new_state = state.copy()
        new_state["overall_status"] = self._determine_status(merged_results)
        new_state["confidence_score"] = confidence
        new_state["final_report"] = report

        return new_state
```

### Confidence Score Calculator
```python
class ConfidenceCalculator:
    """Calculates confidence scores for validation results."""

    def __init__(self, weights: dict = None):
        self.weights = weights or {
            "pass_rate": 0.4,
            "severity": 0.3,
            "coverage": 0.2,
            "reliability": 0.1
        }

    def calculate(self, results: List[ValidationResult]) -> float:
        """
        Calculate overall confidence score (0.0 to 1.0).

        Factors:
        - Pass rate: % of validators that passed
        - Severity: Weighted by error severity (critical > error > warning)
        - Coverage: % of data that was validated
        - Reliability: Historical success rate of validators
        """
        if not results:
            return 0.0

        # Calculate pass rate
        passed = sum(1 for r in results if r.status == "passed")
        pass_rate = passed / len(results)

        # Calculate severity score (1.0 = no errors, 0.0 = many critical errors)
        severity_score = self._calculate_severity_score(results)

        # Calculate coverage (assume 100% for now, could be enhanced)
        coverage = 1.0

        # Calculate reliability (assume 1.0 for now, could track history)
        reliability = 1.0

        # Weighted sum
        confidence = (
            pass_rate * self.weights["pass_rate"] +
            severity_score * self.weights["severity"] +
            coverage * self.weights["coverage"] +
            reliability * self.weights["reliability"]
        )

        return round(confidence, 3)

    def _calculate_severity_score(self, results: List[ValidationResult]) -> float:
        """Calculate score based on error severity distribution."""
        total_errors = 0
        weighted_errors = 0
        severity_weights = {"critical": 1.0, "error": 0.6, "warning": 0.3, "info": 0.1}

        for result in results:
            for error in result.errors:
                total_errors += 1
                weighted_errors += severity_weights.get(error.severity, 0.5)

        if total_errors == 0:
            return 1.0

        # Normalize: fewer weighted errors = higher score
        max_weighted = total_errors * 1.0  # All critical
        score = 1.0 - (weighted_errors / max_weighted)
        return max(0.0, score)
```

### Report Generator
```python
class ReportGenerator:
    """Generates validation reports in multiple formats."""

    def generate(
        self,
        results: List[ValidationResult],
        confidence: float,
        error_analysis: dict,
        format: str = "markdown"
    ) -> dict:
        """
        Generate comprehensive validation report.

        Args:
            results: Validation results to report on
            confidence: Confidence score
            error_analysis: Error analysis data
            format: Output format (json, markdown, html, pdf)

        Returns:
            Report dictionary with content and metadata
        """
        report = {
            "summary": self._generate_summary(results, confidence),
            "overview": self._generate_overview(results),
            "detailed_findings": self._generate_findings(results),
            "error_analysis": error_analysis,
            "recommendations": self._generate_recommendations(results, error_analysis),
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "validator_count": len(results),
                "confidence_score": confidence
            }
        }

        if format == "markdown":
            return self._format_as_markdown(report)
        elif format == "html":
            return self._format_as_html(report)
        elif format == "pdf":
            return self._format_as_pdf(report)
        else:
            return report  # JSON
```

## Innovation Highlights

1. **Multi-Factor Confidence Scoring**: Considers pass rate, error severity, coverage, and validator reliability
2. **LLM-Enhanced Recommendations**: Uses LLM to generate actionable remediation suggestions
3. **Error Pattern Detection**: Automatically identifies related errors and patterns
4. **Multi-Format Reports**: Supports JSON, Markdown, HTML, and PDF outputs
5. **Interactive Visualizations**: Generates charts and graphs for better understanding
6. **Intelligent Deduplication**: Merges duplicate errors from different validators

## Acceptance Criteria

- ✅ Aggregator collects and merges all validation results
- ✅ Confidence score calculation works correctly and intuitively
- ✅ Reports generated in all supported formats
- ✅ Error analysis identifies patterns and prioritizes issues
- ✅ Recommendations are actionable and relevant
- ✅ Visualizations render correctly
- ✅ Deduplication removes duplicate errors
- ✅ All unit tests passing (>85% coverage)
- ✅ Integration tests show complete workflow
- ✅ Reports are comprehensive yet readable

## Implementation Order

1. Implement result merger and deduplication logic
2. Implement confidence score calculator
3. Implement error analysis and grouping
4. Implement report generator (start with JSON/Markdown)
5. Add HTML and PDF report formats
6. Implement visualization generation
7. Implement aggregator agent core
8. Write comprehensive tests
9. Integration testing with full workflow

## Estimated Complexity
**High** - Complex aggregation logic, multiple output formats, visualization, and LLM integration

## Notes
- Confidence score algorithm should be tunable based on use case
- Report templates should be customizable
- Consider performance with large numbers of results (1000+ errors)
- Visualization library choice important (plotly, matplotlib, or export data for frontend)
- LLM prompts for recommendations need careful engineering
- Deduplication logic must preserve important error details
