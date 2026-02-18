"""Load MedThink QA data and convert it to a pandas DataFrame."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def get_default_data_path() -> Path:
    """Return the default location of QA_data.json relative to this script."""
    return Path(__file__).resolve().parent.parent / "data" / "medthink-bench" / "QA_data.json"


def get_prompt_template_path() -> Path:
    """Return the default location of the input prompt template."""
    return Path(__file__).resolve().parent / "assets" / "prompt_template.txt"


def get_subset_output_path() -> Path:
    """Return the default output Excel path for subset_df."""
    return Path(__file__).resolve().parent / "assets" / "subset.xlsx"


def load_qa_json(json_path: Path | str | None = None) -> list[dict[str, Any]]:
    """Load QA JSON records from disk."""
    path = Path(json_path) if json_path else get_default_data_path()

    if not path.exists():
        raise FileNotFoundError(f"QA data file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON list, got: {type(data).__name__}")

    return data


def qa_json_to_dataframe(json_path: Path | str | None = None) -> pd.DataFrame:
    """Load QA JSON and convert records into a DataFrame."""
    records = load_qa_json(json_path=json_path)
    return pd.DataFrame(records)


def load_prompt_template(template_path: Path | str | None = None) -> str:
    """Load prompt template text from disk."""
    path = Path(template_path) if template_path else get_prompt_template_path()

    if not path.exists():
        raise FileNotFoundError(f"Prompt template file not found: {path}")

    return path.read_text(encoding="utf-8")


def add_input_prompt_column(df: pd.DataFrame, template_text: str) -> pd.DataFrame:
    """Create input_prompt by inserting question text into template."""
    if "question" not in df.columns:
        raise KeyError("Column 'question' is not present in the dataset.")

    transformed_df = df.copy()
    transformed_df["input_prompt"] = transformed_df["question"].apply(
        lambda q: template_text.format(question=str(q).strip())
    )
    return transformed_df


def add_empty_llm_response_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add an empty LLM_response column."""
    transformed_df = df.copy()
    transformed_df["LLM_response"] = ""
    return transformed_df


def qa_type_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return count/percentage distribution for QA_Type."""
    if "QA_Type" not in df.columns:
        raise KeyError("Column 'QA_Type' is not present in the dataset.")

    distribution = (
        df.groupby("QA_Type", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    distribution["percentage"] = (distribution["count"] / len(df) * 100).round(2)
    return distribution


def stratified_sample_by_qa_type(df: pd.DataFrame, n_per_type: int = 10, seed: int = 42) -> pd.DataFrame:
    """Create a stratified sample with equal rows per QA_Type."""
    if "QA_Type" not in df.columns:
        raise KeyError("Column 'QA_Type' is not present in the dataset.")

    counts = df["QA_Type"].value_counts(dropna=False)
    insufficient = counts[counts < n_per_type]
    if not insufficient.empty:
        raise ValueError(
            "Some QA_Type groups have fewer rows than requested: "
            + ", ".join([f"{k} ({v})" for k, v in insufficient.items()])
        )

    subset = (
        df.groupby("QA_Type", group_keys=False)
        .sample(n=n_per_type, random_state=seed)
        .reset_index(drop=True)
    )
    return subset


def scoring_points_length_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return distribution of number of elements in Scoring_Points per row."""
    if "Scoring_Points" not in df.columns:
        raise KeyError("Column 'Scoring_Points' is not present in the dataset.")

    lengths = df["Scoring_Points"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    distribution = (
        lengths.value_counts()
        .sort_index()
        .rename_axis("num_elements")
        .reset_index(name="count")
    )
    distribution["percentage"] = (distribution["count"] / len(df) * 100).round(2)
    return distribution


def _ensure_trailing_period(text: str) -> str:
    """Trim text and ensure it ends with a period."""
    cleaned = text.strip()
    if not cleaned:
        return ""
    if cleaned.endswith("."):
        return cleaned
    return f"{cleaned}."


def concatenate_scoring_points(df: pd.DataFrame, separator: str = " ") -> pd.DataFrame:
    """Concatenate Scoring_Points list elements into one paragraph per row."""
    if "Scoring_Points" not in df.columns:
        raise KeyError("Column 'Scoring_Points' is not present in the dataset.")

    transformed_df = df.copy()
    transformed_df["Scoring_Points"] = transformed_df["Scoring_Points"].apply(
        lambda x: separator.join(
            [normalized for point in x if (normalized := _ensure_trailing_period(str(point)))]
        )
        if isinstance(x, list)
        else ""
    )
    return transformed_df


def export_subset_to_excel(df: pd.DataFrame, output_path: Path | str | None = None) -> Path:
    """Export subset_df to three sheets and preserve input_prompt line breaks."""
    if "input_prompt" not in df.columns:
        raise KeyError("Column 'input_prompt' is not present in the dataset.")

    from openpyxl.styles import Alignment

    path = Path(output_path) if output_path else get_subset_output_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    export_df = df.copy()
    # Convert escaped '\n' to actual newlines so Excel displays line breaks correctly.
    export_df["input_prompt"] = export_df["input_prompt"].astype(str).str.replace("\\n", "\n", regex=False)

    sheet_names = ["ExpertAI", "OpenEvidence", "DoxGPT"]
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name in sheet_names:
            export_df.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            input_prompt_col = export_df.columns.get_loc("input_prompt") + 1
            for row in worksheet.iter_rows(
                min_row=2, max_row=worksheet.max_row, min_col=input_prompt_col, max_col=input_prompt_col
            ):
                row[0].alignment = Alignment(wrap_text=True)

    return path


if __name__ == "__main__":
    seed = 42
    df = qa_json_to_dataframe()
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    print(df.head(3))
    print("\nQA_Type distribution:")
    print(qa_type_distribution(df).to_string(index=False))

    subset_df = stratified_sample_by_qa_type(df, n_per_type=10, seed=seed)
    print(f"\nCreated subset_df with seed={seed}")
    print(f"subset_df shape: {subset_df.shape}")
    print("subset_df QA_Type distribution:")
    print(qa_type_distribution(subset_df).to_string(index=False))

    scoring_lengths = subset_df["Scoring_Points"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    min_elements = int(scoring_lengths.min())  # Min/max are computed from the current subset_df values.
    max_elements = int(scoring_lengths.max())
    print("\nScoring_Points element-count distribution in subset_df:")
    print(scoring_points_length_distribution(subset_df).to_string(index=False))
    print(f"Min elements in Scoring_Points: {min_elements}")
    print(f"Max elements in Scoring_Points: {max_elements}")

    subset_df = concatenate_scoring_points(subset_df, separator=" ")
    print("\nUpdated subset_df: Scoring_Points list entries concatenated into strings.")
    print(subset_df[["QA_Type", "Scoring_Points"]].head(3).to_string(index=False))

    prompt_template = load_prompt_template()
    subset_df = add_input_prompt_column(subset_df, prompt_template)
    subset_df = add_empty_llm_response_column(subset_df)
    print("\nAdded input_prompt column to subset_df using assets/prompt_template.txt")
    print("Added empty LLM_response column to subset_df")
    print(subset_df[["question", "input_prompt", "LLM_response"]].head(1).to_string(index=False))

    output_file = export_subset_to_excel(subset_df)
    print(f"\nExported subset_df to Excel: {output_file}")
    print("Sheets written: ExpertAI, OpenEvidence, DoxGPT")
