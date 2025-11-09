import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style='whitegrid')
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# 1. Load & inspect
# ----------------------------
df = pd.read_csv("student_feedback.csv")  # change filename if needed
print("Original columns:", df.columns.tolist())

# Clean column names (you already did, but this is safe to run)
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
print("After cleaning:", df.columns.tolist())

# ----------------------------
# 2. Drop unneeded columns & duplicates
# ----------------------------
# Drop the Unnamed index column if it exists
df = df.drop(columns=['unnamed:_0'], errors='ignore')

# Optionally drop student_id if you don't need to identify rows in analysis
# Keep it for per-student inspection, but exclude from numeric calculations.
student_id_col = 'student_id' if 'student_id' in df.columns else None

# Remove exact duplicate rows
before = df.shape[0]
df.drop_duplicates(inplace=True)
print(f"Dropped {before - df.shape[0]} duplicate rows")

# ----------------------------
# 3. Convert rating columns to numeric
# ----------------------------
# Identify question columns (exclude student_id)
non_rating_cols = [student_id_col] if student_id_col else []
rating_cols = [c for c in df.columns if c not in non_rating_cols]

# Coerce ratings to numeric, handle errors -> NaN
df[rating_cols] = df[rating_cols].apply(lambda col: pd.to_numeric(col, errors='coerce'))

# Show missing counts
print("\nMissing values per column:")
print(df[rating_cols].isna().sum())

# ----------------------------
# 4. Basic statistics: average per question
# ----------------------------
avg_per_question = df[rating_cols].mean().sort_values(ascending=False)
print("\nAverage rating per question (highest -> lowest):")
print(avg_per_question)

avg_per_question.to_csv(os.path.join(OUTPUT_DIR, "avg_rating_per_question.csv"))

# ----------------------------
# 5. Overall rating per student
# ----------------------------
# Compute mean across the rating columns (row-wise)
df['overall_rating'] = df[rating_cols].mean(axis=1)

# Summary of overall_rating
print("\nOverall rating summary:")
print(df['overall_rating'].describe())

# Save cleaned dataframe with overall rating
df.to_csv(os.path.join(OUTPUT_DIR, "cleaned_feedback_with_overall.csv"), index=False)

# ----------------------------
# 6. Visualizations
# ----------------------------
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_per_question.values, y=avg_per_question.index)
plt.xlabel("Average Score")
plt.title("Average Rating per Feedback Category")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "avg_rating_per_question.png"))
plt.show()

# Distribution of overall ratings
plt.figure(figsize=(8, 4))
sns.histplot(df['overall_rating'].dropna(), bins=10, kde=True)
plt.xlabel("Overall Rating (mean across questions)")
plt.title("Distribution of Overall Ratings")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "overall_rating_distribution.png"))
plt.show()

# Boxplot of each question (useful to see spread)
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[rating_cols].melt(var_name="question", value_name="score"),
            x="score", y="question")
plt.title("Boxplot of Scores by Question")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "boxplot_scores_by_question.png"))
plt.show()

# Heatmap: correlation between questions
plt.figure(figsize=(10, 8))
corr = df[rating_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
plt.title("Correlation Matrix Between Questions")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "questions_correlation_heatmap.png"))
plt.show()

# ----------------------------
# 7. Key insights (printable)
# ----------------------------
top_question = avg_per_question.idxmax()
bottom_question = avg_per_question.idxmin()
print(f"\nTop-rated aspect: {top_question} -> {avg_per_question.max():.3f}")
print(f"Lowest-rated aspect: {bottom_question} -> {avg_per_question.min():.3f}")

# Students with lowest overall scores (example to inspect)
print("\nBottom 5 students by overall rating:")
if student_id_col:
    print(df[[student_id_col, 'overall_rating']].sort_values('overall_rating').head(5))
else:
    print(df[['overall_rating']].sort_values('overall_rating').head(5))

# ----------------------------
# 8. Optional: normalize scales (if a column uses 1-10 while others 1-5)
# ----------------------------
# This is optional; detect if any column has max>5 and rescale to 1-5
max_values = df[rating_cols].max()
cols_need_scale = max_values[max_values > 5].index.tolist()
if cols_need_scale:
    print("\nColumns with scale >5 detected and will be normalized to 1-5:", cols_need_scale)
    for c in cols_need_scale:
        old_max = df[c].max()
        old_min = df[c].min()
        # linear rescale to 1-5: new = 1 + (val - old_min) * (4 / (old_max - old_min))
        df[c] = 1 + (df[c] - old_min) * (4.0 / (old_max - old_min))
    # Recompute overall_rating and averages after scaling
    df['overall_rating'] = df[rating_cols].mean(axis=1)
    avg_per_question = df[rating_cols].mean().sort_values(ascending=False)
    avg_per_question.to_csv(os.path.join(OUTPUT_DIR, "avg_rating_per_question_scaled.csv"))

# ----------------------------
# 9. Save a small text report
# ----------------------------
report_lines = [
    "Course Feedback Quick Report",
    "----------------------------",
    f"Total responses (cleaned): {df.shape[0]}",
    f"Top-rated aspect: {top_question} -> {avg_per_question.max():.3f}",
    f"Lowest-rated aspect: {bottom_question} -> {avg_per_question.min():.3f}",
    f"Average overall rating: {df['overall_rating'].mean():.3f}"
]
with open(os.path.join(OUTPUT_DIR, "quick_report.txt"), "w") as f:
    f.write("\n".join(report_lines))

print("\nSaved outputs to folder:", OUTPUT_DIR)
