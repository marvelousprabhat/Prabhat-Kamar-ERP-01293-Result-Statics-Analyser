import sys
import numpy as np
import pandas as pd

# Optional import for plotting; falls back gracefully if matplotlib isn't installed
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

PASS_MARK = 40


def load_and_validate_results(file_path: str = "results.csv") -> pd.DataFrame | None:
    """
    Reads results.csv using Pandas, handles missing file or empty/invalid rows,
    and skips missing marks or marks outside [0, 100] with warnings.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print("No valid marks found. Nothing to analyse.")
        return None
    except Exception as e:
        print(f"Error reading '{file_path}': {e}")
        return None

    if df.empty or "Marks" not in df.columns:
        print("No valid marks found. Nothing to analyse.")
        return None

    valid_rows = []
    for idx, row in df.iterrows():
        # CSV row number: header is row 1, first data row is row 2
        csv_row_num = int(idx) + 2
        roll_no = row.get("RollNo", idx + 1)
        raw_mark = row["Marks"]

        # Check missing mark
        if pd.isna(raw_mark):
            print(f"Warning: Row {csv_row_num} skipped - missing mark (RollNo: {roll_no}).")
            continue

        # Convert mark to numeric
        try:
            mark_val = float(raw_mark)
            if np.isnan(mark_val) or np.isinf(mark_val):
                print(f"Warning: Row {csv_row_num} skipped - invalid mark.")
                continue
        except (ValueError, TypeError):
            print(f"Warning: Row {csv_row_num} skipped - invalid mark.")
            continue

        # Check range [0, 100]
        if mark_val < 0 or mark_val > 100:
            print(f"Warning: Row {csv_row_num} skipped - marks must be between 0 and 100.")
            continue

        # Store as int if whole number, else float
        clean_mark = int(mark_val) if mark_val.is_integer() else mark_val
        valid_rows.append({"RollNo": roll_no, "Marks": clean_mark})

    if not valid_rows:
        print("No valid marks found. Nothing to analyse.")
        return None

    return pd.DataFrame(valid_rows)


def compute_statistics(df: pd.DataFrame) -> dict:
    """Computes mean, median, mode, highest, lowest, and sample standard deviation (ddof=1)."""
    marks_arr = df["Marks"].to_numpy(dtype=float)
    n = len(marks_arr)

    mean_val = float(np.mean(marks_arr))
    median_val = float(np.median(marks_arr))

    # Mode using Pandas Series.mode() (handles ties or single value cleanly)
    mode_series = df["Marks"].mode()
    if len(mode_series) == 1:
        val = mode_series.iloc[0]
        mode_str = str(int(val)) if float(val).is_integer() else f"{float(val):.2f}"
    else:
        mode_str = ", ".join(
            str(int(v)) if float(v).is_integer() else f"{float(v):.2f}"
            for v in mode_series
        )

    highest_val = float(np.max(marks_arr))
    lowest_val = float(np.min(marks_arr))

    # Sample standard deviation (ddof=1); 0.00 when only 1 student is present
    std_val = float(np.std(marks_arr, ddof=1)) if n > 1 else 0.0

    def fmt_num(x: float) -> str:
        return str(int(x)) if x.is_integer() else f"{x:.2f}"

    return {
        "students": n,
        "mean": f"{mean_val:.2f}",
        "median": f"{median_val:.2f}",
        "mode": mode_str,
        "highest": fmt_num(highest_val),
        "lowest": fmt_num(lowest_val),
        "std": f"{std_val:.2f}",
    }


def show_statistics(df: pd.DataFrame) -> None:
    stats = compute_statistics(df)
    print(f"Students: {stats['students']}")
    print(f"Mean: {stats['mean']}")
    print(f"Median: {stats['median']}")
    print(f"Mode: {stats['mode']}")
    print(f"Highest: {stats['highest']} | Lowest: {stats['lowest']}")
    print(f"Standard deviation: {stats['std']}")


def show_pass_and_distribution(df: pd.DataFrame, show_chart: bool = True) -> None:
    marks_arr = df["Marks"].to_numpy(dtype=float)
    total = len(marks_arr)
    passed = int(np.sum(marks_arr >= PASS_MARK))
    failed = total - passed
    pass_pct = (passed / total) * 100.0

    # Score bands: 0-39, 40-59, 60-79, 80-100
    bins = [-0.001, 39.999, 59.999, 79.999, 100.001]
    labels = ["0-39", "40-59", "60-79", "80-100"]
    counts, _ = np.histogram(marks_arr, bins=bins)

    print(f"Passed: {passed} | Failed: {failed} | Pass percentage: {pass_pct:.2f}%")
    for label, count in zip(labels, counts):
        print(f"{label} : {int(count)}")

    if show_chart:
        if MATPLOTLIB_AVAILABLE:
            plt.figure(figsize=(7, 4.5))
            bars = plt.bar(labels, counts, edgecolor="black")
            plt.title("Student Marks Distribution")
            plt.xlabel("Score Band")
            plt.ylabel("Number of Students")
            plt.ylim(0, max(counts) + 1 if len(counts) > 0 else 1)
            for bar, count in zip(bars, counts):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    str(int(count)),
                    ha="center",
                    va="bottom",
                )
            plt.tight_layout()
            print("(bar chart is displayed)")
            plt.show()
        else:
            print("(matplotlib not installed; bar chart display skipped)")


def main() -> None:
    file_path = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
    df = load_and_validate_results(file_path)
    if df is None:
        return

    print("\n--- Result Statistics Analyzer ---")
    print("1. Show statistics")
    print("2. Pass percentage and distribution")
    print("3. Show both")

    try:
        choice = input("Enter menu choice (1/2/3): ").strip()
    except EOFError:
        choice = "1"

    if choice == "1" or choice.lower() == "show statistics":
        show_statistics(df)
    elif choice == "2" or choice.lower() in {"pass percentage and distribution", "pass percentage and d stribution"}:
        show_pass_and_distribution(df, show_chart=True)
    elif choice == "3":
        show_statistics(df)
        print()
        show_pass_and_distribution(df, show_chart=True)
    else:
        print("Invalid menu choice.")


if __name__ == "__main__":
    main()