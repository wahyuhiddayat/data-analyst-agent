import pandas as pd


def summarize_dataframe(df: pd.DataFrame, sample_rows: int = 5) -> str:
    """
    Build a compact text summary of a dataframe for the model's context.

    Includes shape, per-column dtypes, a few sample rows, and a numeric
    summary. The full table is deliberately withheld so the agent must
    query the data through code rather than reading it directly.
    """
    lines = [f"Shape: {df.shape[0]} rows x {df.shape[1]} columns", "", "Columns and dtypes:"]
    for col, dtype in df.dtypes.items():
        lines.append(f"  - {col}: {dtype}")

    lines += ["", f"First {sample_rows} rows:", df.head(sample_rows).to_string(index=False)]

    numeric = df.select_dtypes("number")
    lines += ["", "Numeric summary:"]
    lines.append(numeric.describe().to_string() if not numeric.empty else "  (no numeric columns)")

    return "\n".join(lines)
