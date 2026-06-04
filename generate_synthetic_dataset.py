import numpy as np
import pandas as pd


INPUT_FILE = "test.csv"
OUTPUT_FILE = "test_synthetic_15000.csv"
TARGET_ROWS = 15000
SEED = 42
DATE_FMT = "%m/%d/%Y %H:%M:%S"


def read_source(path: str) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "cp1254"):
        try:
            return pd.read_csv(path, sep=";", encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("Unable to decode source CSV with expected encodings")


def sample_positive_seconds(values: pd.Series, rng: np.random.Generator, size: int, minimum: int = 1) -> np.ndarray:
    clean = values.dropna().astype(float).to_numpy()
    clean = clean[clean >= minimum]
    if clean.size == 0:
        return np.full(size, minimum, dtype=int)

    sampled = rng.choice(clean, size=size, replace=True)
    jitter_scale = max(float(np.std(clean)) * 0.08, 1.0)
    jitter = rng.normal(0, jitter_scale, size=size)
    return np.maximum(np.rint(sampled + jitter), minimum).astype(int)


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = read_source(INPUT_FILE)
    original_columns = list(df.columns)

    for col in ("Created At", "Started At", "Completed At"):
        df[col] = pd.to_datetime(df[col], format=DATE_FMT, errors="raise")

    df = df.sort_values(["Created At", "Sequence"]).reset_index(drop=True)
    additional_rows = TARGET_ROWS - len(df)
    if additional_rows < 0:
        raise ValueError(f"Source has {len(df)} rows, which is more than target {TARGET_ROWS}")

    created_deltas = df["Created At"].diff().dt.total_seconds()
    start_offsets = (df["Started At"] - df["Created At"]).dt.total_seconds()
    run_durations = (df["Completed At"] - df["Started At"]).dt.total_seconds()
    sequence_deltas = df["Sequence"].diff()

    synthetic_base = df.sample(n=additional_rows, replace=True, random_state=SEED).reset_index(drop=True)

    created_steps = sample_positive_seconds(created_deltas, rng, additional_rows)
    start_steps = sample_positive_seconds(start_offsets, rng, additional_rows, minimum=0)
    run_steps = sample_positive_seconds(run_durations, rng, additional_rows, minimum=1)
    seq_steps = sample_positive_seconds(sequence_deltas, rng, additional_rows)

    synthetic = synthetic_base.copy()
    synthetic["Sequence"] = int(df["Sequence"].max()) + np.cumsum(seq_steps)
    synthetic["Created At"] = df["Created At"].max() + pd.to_timedelta(np.cumsum(created_steps), unit="s")
    synthetic["Started At"] = synthetic["Created At"] + pd.to_timedelta(start_steps, unit="s")
    synthetic["Completed At"] = synthetic["Started At"] + pd.to_timedelta(run_steps, unit="s")

    combined = pd.concat([df, synthetic], ignore_index=True)
    combined = combined.sort_values(["Created At", "Sequence"]).reset_index(drop=True)

    for col in ("Created At", "Started At", "Completed At"):
        combined[col] = combined[col].dt.strftime(DATE_FMT)

    combined = combined[original_columns]
    combined.to_csv(OUTPUT_FILE, sep=";", index=False, encoding="utf-8-sig")

    print(f"source_rows={len(df)}")
    print(f"synthetic_rows={additional_rows}")
    print(f"output_rows={len(combined)}")
    print(f"output_file={OUTPUT_FILE}")
    print(f"duplicate_sequences={combined['Sequence'].duplicated().sum()}")
    print(f"columns={len(combined.columns)}")


if __name__ == "__main__":
    main()
