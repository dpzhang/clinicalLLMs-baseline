import json
from pathlib import Path
import pandas as pd
import re


def get_default_data_path():
    return Path(__file__).resolve().parent.parent / "data" / "medthink-bench" / "QA_data.json"


def get_prompt_template_path():
    return Path(__file__).resolve().parent / "assets" / "prompt_template.txt"


def get_subset_output_path():
    return Path(__file__).resolve().parent / "assets" / "subset.xlsx"


def load_qa_json():
    path = get_default_data_path()
    if not path.exists():
        raise FileNotFoundError(f"QA data file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(
            f"Expected top-level JSON list, got: {type(data).__name__}")

    return data


def qa_json_to_dataframe():
    # convert to dataframe
    records = pd.DataFrame(load_qa_json())
    # rename the column
    records = records.rename(columns={"Scoring_Points": "scoring_point",
                                      "question": "raw_question"})
    # get the number of scoring points for each record
    records['num_scoring_point'] = records['scoring_point'].apply(
        lambda x: len(x) if isinstance(x, list) else 0)

    records = add_markdown_table_flag(records)
    return records


def add_markdown_table_flag(df):
    table_re = re.compile(
        r"(?m)^\s*\|.*\|\s*$\n^\s*\|(?:\s*:?-{3,}:?\s*\|)+\s*$"
    )
    out = df.copy()
    out["has_markdown_table"] = out["raw_question"].astype(
        str).str.contains(table_re)
    return out


def load_prompt_template():
    path = get_prompt_template_path()
    if not path.exists():
        raise FileNotFoundError(f"Prompt template file not found: {path}")

    return path.read_text(encoding="utf-8")


def add_prompt_column(df, template_text):
    transformed_df = df.copy()
    # Prompt 1: raw question
    transformed_df["prompt1"] = transformed_df["raw_question"].astype(
        str).str.strip()
    # Prompt 2: template text (same for all rows)
    transformed_df["prompt2"] = template_text.strip()

    return transformed_df


def add_empty_llm_response_column(df):
    transformed_df = df.copy()
    transformed_df["response"] = ""
    transformed_df["llm_answer"] = ""
    return transformed_df


def qa_type_distribution(df):
    summary = (
        df.groupby("QA_Type", dropna=False)
        .agg(
            total_questions=("QA_Type", "size"),
            markdown_table_count=("has_markdown_table", "sum"))
        .reset_index()
    )
    summary["markdown_percentage"] = (
        summary["markdown_table_count"] / summary["total_questions"] * 100
    ).round(2)

    summary = summary.sort_values("total_questions", ascending=False)

    return summary


def stratified_sample_by_qa_type(df, n_per_type, seed=42, exclude_markdown_tables=True):
    working_df = df
    if exclude_markdown_tables:
        working_df = working_df.loc[~working_df["has_markdown_table"]].copy()

    counts = working_df["QA_Type"].value_counts(dropna=False)
    insufficient = counts[counts < n_per_type]
    if not insufficient.empty:
        raise ValueError(
            "Some QA_Type groups have fewer questions than requested size: "
            + ", ".join([f"{k} ({v})" for k, v in insufficient.items()])
        )

    subset = (
        working_df.groupby("QA_Type", dropna=False, group_keys=False)
        .sample(n=n_per_type, random_state=seed)
        .reset_index(drop=True)
    )
    return subset


def _ensure_trailing_period(text):
    """
    Trim text and ensure it ends with a period.
    """
    cleaned = text.strip()
    if not cleaned:
        return ""
    if cleaned.endswith("."):
        return cleaned
    return f"{cleaned}."


def concatenate_scoring_point(df, separator):
    """
    Concatenate scoring_point list elements into one paragraph per row.
    """
    if "scoring_point" not in df.columns:
        raise KeyError(
            "Column 'scoring_point' is not present in the dataset.")

    transformed_df = df.copy()
    transformed_df["scoring_paragraph"] = transformed_df["scoring_point"].apply(
        lambda x: separator.join(
            [normalized for point in x if (
                normalized := _ensure_trailing_period(str(point)))]
        )
        if isinstance(x, list)
        else ""
    )
    return transformed_df


def export_subset_to_excel(df, output_path=None):
    from openpyxl.styles import Alignment

    path = Path(output_path) if output_path else get_subset_output_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    export_df = df.copy()
    export_df = export_df.drop(
        columns=["raw_question", "scoring_point",
                 "num_scoring_point", "has_markdown_table"],
        errors="ignore",
    )

    # Convert escaped '\n' to actual newlines for Excel display
    for col in ["prompt1", "prompt2"]:
        if col in export_df.columns:
            export_df[col] = export_df[col].astype(
                str).str.replace("\\n", "\n", regex=False)

    sheet_names = ["ExpertAI", "OpenEvidence", "DoxGPT"]
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name in sheet_names:
            export_df.to_excel(writer, sheet_name=sheet_name, index=False)
            ws = writer.sheets[sheet_name]

            # Wrap text for both prompt columns (if present)
            for col in ["prompt1", "prompt2"]:
                if col not in export_df.columns:
                    continue
                col_idx = export_df.columns.get_loc(
                    col) + 1  # 1-indexed in openpyxl
                for row in ws.iter_rows(min_row=2, max_row=ws.max_row, min_col=col_idx, max_col=col_idx):
                    row[0].alignment = Alignment(
                        wrap_text=True, vertical="top")

    return path


if __name__ == "__main__":
    seed = 42
    n_per_type = 10
    print(f"=> Loading QA data from JSON...")
    df = qa_json_to_dataframe()
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    print("Distribution of QA_Type and Markdown-tables by clinical domains:")
    print(qa_type_distribution(df).to_string(index=False))

    print(
        f"=> Perform stratified sampling to create subset_df with {n_per_type} rows per QA_Type without markdown-style tables...")
    subset_df = stratified_sample_by_qa_type(
        df, n_per_type=n_per_type, seed=seed)
    print("Distribution of QA_Type subset by clinical domains:")
    print(qa_type_distribution(subset_df).to_string(index=False))

    print(f"=> Count the min and max of the number of scoring point")
    print(f"min: {subset_df['num_scoring_point'].min()}")
    print(f"max: {subset_df['num_scoring_point'].max()}")

    print(f"=> Concatenate scoring points into paragraphs in subset_df...")
    subset_df = concatenate_scoring_point(subset_df, separator=" ")

    print(f"=> Load prompt template and add prompt column to subset_df...")
    prompt_template = load_prompt_template()
    subset_df = add_prompt_column(subset_df, prompt_template)
    subset_df = add_empty_llm_response_column(subset_df)

    print(f"=> Export subset_df to Excel with preserved line breaks in 'prompt'...")
    output_file = export_subset_to_excel(subset_df)
