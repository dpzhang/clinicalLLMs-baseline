from __future__ import annotations
import json
from pathlib import Path
import pandas as pd


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
    return records


def load_prompt_template():
    path = get_prompt_template_path()
    if not path.exists():
        raise FileNotFoundError(f"Prompt template file not found: {path}")

    return path.read_text(encoding="utf-8")


def add_prompt_column(df, template_text):
    transformed_df = df.copy()
    transformed_df["prompt"] = transformed_df["raw_question"].apply(
        lambda q: template_text.format(question=str(q).strip())
    )
    return transformed_df


def add_empty_llm_response_column(df):
    transformed_df = df.copy()
    transformed_df["response"] = ""
    return transformed_df


def qa_type_distribution(df):
    if "QA_Type" not in df.columns:
        raise KeyError("Column 'QA_Type' is not present in the dataset.")

    distribution = (
        df.groupby("QA_Type", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    distribution["percentage"] = (
        distribution["count"] / len(df) * 100).round(2)
    return distribution


def stratified_sample_by_qa_type(df, n_per_type, seed=42):
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
    """
    Export subset_df to three sheets and preserve input_prompt line breaks.
     """
    if "prompt" not in df.columns:
        raise KeyError("Column 'prompt' is not present in the dataset.")

    from openpyxl.styles import Alignment

    path = Path(output_path) if output_path else get_subset_output_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    export_df = df.copy()
    # Convert escaped '\n' to actual newlines so Excel displays line breaks correctly.
    export_df["prompt"] = export_df["prompt"].astype(
        str).str.replace("\\n", "\n", regex=False)

    sheet_names = ["ExpertAI", "OpenEvidence", "DoxGPT"]
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name in sheet_names:
            export_df.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            prompt_col = export_df.columns.get_loc("prompt") + 1
            for row in worksheet.iter_rows(
                min_row=2, max_row=worksheet.max_row, min_col=prompt_col, max_col=prompt_col
            ):
                row[0].alignment = Alignment(wrap_text=True)

    return path


if __name__ == "__main__":
    seed = 42
    n_per_type = 10
    print(f"=> Loading QA data from JSON...")
    df = qa_json_to_dataframe()
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    print("Distribution of QA_Type by clinical domains:")
    print(qa_type_distribution(df).to_string(index=False))

    print(
        f"=> Perform stratified sampling to create subset_df with {n_per_type} rows per QA_Type...")
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
