"""
Data Analysis Agent Example

Demonstrates using a cyllama agent to analyze datasets.
Shows how to build an agent that can:
- Load and inspect data
- Perform statistical analysis
- Generate insights
- Create visualizations (conceptually)

Usage:
    python agent_data_analyst.py [model_path]

    If no model_path is provided, will look for models in ./models/ directory.
"""

import argparse
from pathlib import Path
from cyllama import LLM, GenerationConfig
from cyllama.agents import ConstrainedAgent, tool
from cyllama.utils.color import (
    header, section, subheader, success, error, info,
    bullet, numbered, kv
)
import json
import statistics


# Sample dataset (embedded for demo)
SAMPLE_DATA = {
    "sales_data": [
        {"month": "Jan", "revenue": 12500, "units": 250, "region": "North"},
        {"month": "Feb", "revenue": 15800, "units": 310, "region": "North"},
        {"month": "Mar", "revenue": 18200, "units": 380, "region": "North"},
        {"month": "Jan", "revenue": 9800, "units": 190, "region": "South"},
        {"month": "Feb", "revenue": 11200, "units": 220, "region": "South"},
        {"month": "Mar", "revenue": 13500, "units": 270, "region": "South"},
    ],
    "customer_data": [
        {"id": 1, "age": 25, "purchases": 5, "total_spent": 250},
        {"id": 2, "age": 34, "purchases": 12, "total_spent": 680},
        {"id": 3, "age": 45, "purchases": 8, "total_spent": 420},
        {"id": 4, "age": 29, "purchases": 15, "total_spent": 890},
        {"id": 5, "age": 52, "purchases": 6, "total_spent": 310},
    ]
}


# Define data analysis tools

@tool
def load_dataset(dataset_name: str) -> str:
    """
    Load a dataset by name.

    Args:
        dataset_name: Name of dataset ('sales_data' or 'customer_data')

    Returns:
        JSON string of the dataset
    """
    if dataset_name in SAMPLE_DATA:
        return json.dumps(SAMPLE_DATA[dataset_name], indent=2)
    else:
        return f"Dataset '{dataset_name}' not found. Available: {list(SAMPLE_DATA.keys())}"


@tool
def calculate_statistics(dataset_name: str, column: str) -> dict:
    """
    Calculate basic statistics for a numeric column.

    Args:
        dataset_name: Name of the dataset
        column: Column name to analyze

    Returns:
        Dictionary with mean, median, min, max, stddev
    """
    if dataset_name not in SAMPLE_DATA:
        return {"error": f"Dataset '{dataset_name}' not found"}

    data = SAMPLE_DATA[dataset_name]

    try:
        values = [row[column] for row in data if column in row]

        if not values:
            return {"error": f"Column '{column}' not found or has no values"}

        return {
            "column": column,
            "count": len(values),
            "mean": round(statistics.mean(values), 2),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "stddev": round(statistics.stdev(values), 2) if len(values) > 1 else 0
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def filter_data(dataset_name: str, column: str, operator: str, value: str) -> list:
    """
    Filter dataset rows based on a condition.

    Args:
        dataset_name: Name of the dataset
        column: Column to filter on
        operator: Comparison operator ('>', '<', '=', '>=', '<=')
        value: Value to compare against

    Returns:
        List of matching rows
    """
    if dataset_name not in SAMPLE_DATA:
        return [{"error": f"Dataset '{dataset_name}' not found"}]

    data = SAMPLE_DATA[dataset_name]

    try:
        # Convert value to appropriate type
        try:
            compare_value = float(value)
        except ValueError:
            compare_value = value

        results = []
        for row in data:
            if column not in row:
                continue

            row_value = row[column]

            # Perform comparison
            match = False
            if operator == '>':
                match = row_value > compare_value
            elif operator == '<':
                match = row_value < compare_value
            elif operator == '=':
                match = row_value == compare_value
            elif operator == '>=':
                match = row_value >= compare_value
            elif operator == '<=':
                match = row_value <= compare_value

            if match:
                results.append(row)

        return results

    except Exception as e:
        return [{"error": str(e)}]


@tool
def group_by_and_sum(dataset_name: str, group_column: str, sum_column: str) -> dict:
    """
    Group data by a column and sum another column.

    Args:
        dataset_name: Name of the dataset
        group_column: Column to group by
        sum_column: Column to sum

    Returns:
        Dictionary with grouped sums
    """
    if dataset_name not in SAMPLE_DATA:
        return {"error": f"Dataset '{dataset_name}' not found"}

    data = SAMPLE_DATA[dataset_name]

    try:
        groups = {}
        for row in data:
            if group_column in row and sum_column in row:
                group_key = str(row[group_column])
                if group_key not in groups:
                    groups[group_key] = 0
                groups[group_key] += row[sum_column]

        return groups

    except Exception as e:
        return {"error": str(e)}


@tool
def calculate_correlation(dataset_name: str, column1: str, column2: str) -> dict:
    """
    Calculate correlation between two numeric columns.

    Args:
        dataset_name: Name of the dataset
        column1: First column
        column2: Second column

    Returns:
        Dictionary with correlation coefficient
    """
    if dataset_name not in SAMPLE_DATA:
        return {"error": f"Dataset '{dataset_name}' not found"}

    data = SAMPLE_DATA[dataset_name]

    try:
        values1 = [row[column1] for row in data if column1 in row and column2 in row]
        values2 = [row[column2] for row in data if column1 in row and column2 in row]

        if len(values1) < 2:
            return {"error": "Need at least 2 data points"}

        # Simple correlation calculation
        mean1 = statistics.mean(values1)
        mean2 = statistics.mean(values2)

        numerator = sum((x - mean1) * (y - mean2) for x, y in zip(values1, values2))
        denominator = (
            sum((x - mean1) ** 2 for x in values1) *
            sum((y - mean2) ** 2 for y in values2)
        ) ** 0.5

        correlation = numerator / denominator if denominator != 0 else 0

        return {
            "column1": column1,
            "column2": column2,
            "correlation": round(correlation, 3),
            "interpretation": "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.4 else "weak"
        }

    except Exception as e:
        return {"error": str(e)}


def example_basic_analysis(llm: LLM):
    """Demonstrate basic data analysis."""
    section("BASIC DATA ANALYSIS EXAMPLE")

    # Use ConstrainedAgent for reliable JSON outputs
    agent = ConstrainedAgent(
        llm=llm,
        tools=[load_dataset, calculate_statistics],
        max_iterations=5,
        verbose=True,
        format="json"
    )

    section("Task: Analyze sales data revenue statistics", color='yellow')

    result = agent.run(
        "Load the sales_data dataset and calculate statistics for the revenue column"
    )

    section("RESULT", color='bright_green')
    kv("Success", str(result.success), value_color='green' if result.success else 'red')
    kv("Answer", result.answer)


def example_advanced_analysis(llm: LLM):
    """Demonstrate multi-step analysis."""
    section("ADVANCED ANALYSIS EXAMPLE")

    agent = ConstrainedAgent(
        llm=llm,
        tools=[
            load_dataset,
            calculate_statistics,
            filter_data,
            group_by_and_sum,
            calculate_correlation
        ],
        max_iterations=10,
        verbose=True,
        format="json"
    )

    section("Task: Find high-value customers", color='yellow')

    result = agent.run(
        "Analyze customer_data to find customers who spent more than 500 dollars total"
    )

    section("RESULT", color='bright_green')
    kv("Answer", result.answer)


def example_correlation_analysis(llm: LLM):
    """Demonstrate correlation analysis."""
    section("CORRELATION ANALYSIS EXAMPLE")

    agent = ConstrainedAgent(
        llm=llm,
        tools=[load_dataset, calculate_correlation, calculate_statistics],
        max_iterations=8,
        verbose=True,
        format="json"
    )

    result = agent.run(
        "Calculate the correlation between age and total_spent in customer_data"
    )

    section("RESULT", color='bright_green')
    kv("Answer", result.answer)


def show_manual_approach():
    """Show manual analysis approach for comparison."""
    section("MANUAL APPROACH (WITHOUT AGENT)")

    subheader("Traditional data analysis code", color='cyan')
    print("""
    # Load data
    data = load_dataset('sales_data')

    # Calculate stats
    revenues = [row['revenue'] for row in data]
    avg_revenue = sum(revenues) / len(revenues)

    # Filter data
    high_performers = [row for row in data if row['revenue'] > avg_revenue]

    # Group and aggregate
    by_region = {}
    for row in data:
        region = row['region']
        if region not in by_region:
            by_region[region] = []
        by_region[region].append(row['revenue'])

    # Calculate regional averages
    regional_avg = {region: sum(values)/len(values)
                    for region, values in by_region.items()}
    """)

    subheader("With agent approach", color='cyan')
    print("""
    agent = ConstrainedAgent(llm=llm, tools=[...])

    result = agent.run('''
        Analyze sales data:
        1. Calculate average revenue
        2. Find high-performing months
        3. Compare regions
        ''')

    # Agent figures out the sequence automatically!
    """)


def find_model() -> Path:
    """Find a model in the default locations."""
    ROOT = Path.cwd()

    # Preferred models in order
    candidates = [
        ROOT / 'models' / 'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf',
        ROOT / 'models' / 'Llama-3.2-1B-Instruct-Q8_0.gguf',
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def main():
    """Run all data analysis examples."""
    parser = argparse.ArgumentParser(
        description="Data Analysis Agent Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python agent_data_analyst.py
    python agent_data_analyst.py /path/to/model.gguf
    python agent_data_analyst.py models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
        """
    )
    parser.add_argument(
        'model_path',
        nargs='?',
        type=str,
        help='Path to GGUF model file (optional, will auto-detect if not provided)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose model output'
    )

    args = parser.parse_args()

    # Resolve model path
    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.exists():
            error(f"Model not found: {model_path}")
            return 1
    else:
        model_path = find_model()
        if model_path is None:
            error("No model found. Please provide a model path.")
            info("Usage: python agent_data_analyst.py /path/to/model.gguf")
            return 1

    header("DATA ANALYSIS AGENT EXAMPLES")

    print("\nThis example demonstrates:")
    numbered([
        "Loading and inspecting datasets",
        "Statistical analysis (mean, median, stddev)",
        "Filtering and grouping data",
        "Correlation analysis",
        "Multi-step analytical workflows"
    ])

    # Show comparison
    show_manual_approach()

    # Initialize LLM once and share across examples
    info(f"Loading model: {model_path.name}...")
    config = GenerationConfig(n_batch=4096, n_ctx=8192)
    llm = LLM(str(model_path), config=config, verbose=args.verbose)
    success("Model loaded successfully.")

    # Run examples
    example_basic_analysis(llm)
    example_advanced_analysis(llm)
    example_correlation_analysis(llm)

    return 0


if __name__ == "__main__":
    exit(main())
