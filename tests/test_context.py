import pandas as pd

from data_analyst_agent.context import summarize_dataframe


def test_summary_includes_shape_and_columns():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    summary = summarize_dataframe(df)
    assert "3 rows x 2 columns" in summary
    assert "a: int64" in summary


def test_summary_handles_no_numeric_columns():
    df = pd.DataFrame({"label": ["x", "y"]})
    summary = summarize_dataframe(df)
    assert "no numeric columns" in summary
