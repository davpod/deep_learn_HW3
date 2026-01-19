import pandas as pd

# Files
TEST_CSV = "test.csv"
SOLUTION_CSV = "solution.csv"
OUTPUT_CSV = "test_labels.csv"

# Load files
test_df = pd.read_csv(TEST_CSV, encoding="latin1")
solution_df = pd.read_csv(SOLUTION_CSV, encoding="latin1")

# Keep only labeled rows
solution_eval = solution_df[solution_df["relevance"] != -1].copy()

# Merge on 'id'
labeled_test_df = solution_eval.merge(test_df, on="id", how="left")

# Optional: keep only relevant columns
labeled_test_df = labeled_test_df[["id", "product_uid", "product_title", "search_term", "relevance"]]

# Save
labeled_test_df.to_csv(OUTPUT_CSV, index=False)
print(f"Labeled test file saved: {OUTPUT_CSV}")
print("Rows:", len(labeled_test_df))
