import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# SUPERMARKET SALES ANALYSIS
# ============================================================


# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

file_path = "supermarket_sales.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully.")


# ------------------------------------------------------------
# 2. FIRST 5 ROWS
# ------------------------------------------------------------

print("\n==============================")
print("FIRST 5 ROWS")
print("==============================")

print(df.head())


# ------------------------------------------------------------
# 3. DATASET SHAPE
# ------------------------------------------------------------

print("\n==============================")
print("DATASET SHAPE")
print("==============================")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ------------------------------------------------------------
# 4. COLUMN NAMES
# ------------------------------------------------------------

print("\n==============================")
print("COLUMN NAMES")
print("==============================")

print(df.columns.tolist())


# ------------------------------------------------------------
# 5. DATA TYPES
# ------------------------------------------------------------

print("\n==============================")
print("DATA TYPES")
print("==============================")

print(df.dtypes)


# ------------------------------------------------------------
# 6. MISSING VALUES
# ------------------------------------------------------------

print("\n==============================")
print("MISSING VALUES")
print("==============================")

print(df.isnull().sum())


# ------------------------------------------------------------
# 7. DUPLICATE RECORDS
# ------------------------------------------------------------

print("\n==============================")
print("DUPLICATE RECORDS")
print("==============================")

duplicates = df.duplicated().sum()

print("Duplicate rows:", duplicates)

if duplicates > 0:

    df = df.drop_duplicates()

    print("Duplicate rows removed.")

else:

    print("No duplicate rows found.")


# ------------------------------------------------------------
# 8. DATA TYPE CONVERSION
# ------------------------------------------------------------

if "Date" in df.columns:

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )


for column in ["Quantity", "Total"]:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# ------------------------------------------------------------
# 9. DATA VALIDATION
# ------------------------------------------------------------

if "Quantity" in df.columns:

    df = df[
        df["Quantity"] > 0
    ]


if "Total" in df.columns:

    df = df[
        df["Total"] >= 0
    ]


# ------------------------------------------------------------
# 10. STATISTICAL SUMMARY
# ------------------------------------------------------------

print("\n==============================")
print("STATISTICAL SUMMARY")
print("==============================")

print(df.describe())


# ------------------------------------------------------------
# 11. SALES BY PRODUCT LINE
# ------------------------------------------------------------

product_sales = (
    df.groupby(
        "Product line"
    )["Total"]
    .sum()
    .sort_values(
        ascending=False
    )
)

print("\n==============================")
print("SALES BY PRODUCT LINE")
print("==============================")

print(product_sales)


# ------------------------------------------------------------
# 12. SALES BY BRANCH
# ------------------------------------------------------------

branch_sales = (
    df.groupby(
        "Branch"
    )["Total"]
    .sum()
    .sort_values(
        ascending=False
    )
)

print("\n==============================")
print("SALES BY BRANCH")
print("==============================")

print(branch_sales)


# ------------------------------------------------------------
# 13. QUANTITY BY PRODUCT LINE
# ------------------------------------------------------------

product_quantity = (
    df.groupby(
        "Product line"
    )["Quantity"]
    .sum()
    .sort_values(
        ascending=False
    )
)

print("\n==============================")
print("QUANTITY BY PRODUCT LINE")
print("==============================")

print(product_quantity)


# ------------------------------------------------------------
# 14. AVERAGE SALES BY BRANCH
# ------------------------------------------------------------

average_branch_sales = (
    df.groupby(
        "Branch"
    )["Total"]
    .mean()
    .sort_values(
        ascending=False
    )
)

print("\n==============================")
print("AVERAGE SALES BY BRANCH")
print("==============================")

print(average_branch_sales)


# ------------------------------------------------------------
# 15. TRANSACTIONS BY BRANCH
# ------------------------------------------------------------

branch_transactions = (
    df.groupby(
        "Branch"
    )
    .size()
    .sort_values(
        ascending=False
    )
)

print("\n==============================")
print("TRANSACTIONS BY BRANCH")
print("==============================")

print(branch_transactions)


# ------------------------------------------------------------
# 16. TOP PRODUCT LINE
# ------------------------------------------------------------

top_product = product_sales.idxmax()

print("\n==============================")
print("TOP PRODUCT LINE")
print("==============================")

print("Product Line:", top_product)

print(
    "Total Sales:",
    round(
        product_sales.max(),
        2
    )
)


# ------------------------------------------------------------
# 17. BEST PERFORMING BRANCH
# ------------------------------------------------------------

best_branch = branch_sales.idxmax()

print("\n==============================")
print("BEST PERFORMING BRANCH")
print("==============================")

print("Branch:", best_branch)

print(
    "Total Sales:",
    round(
        branch_sales.max(),
        2
    )
)


# ------------------------------------------------------------
# 18. VISUALIZATION SETTINGS
# ------------------------------------------------------------

sns.set_theme(
    style="whitegrid"
)


# ------------------------------------------------------------
# 19. TOTAL SALES BY PRODUCT LINE
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 5)
)

sns.barplot(
    data=df,
    x="Product line",
    y="Total",
    estimator="sum",
    errorbar=None
)

plt.title(
    "Total Sales by Product Line"
)

plt.xlabel(
    "Product Line"
)

plt.ylabel(
    "Total Sales"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 20. TOTAL SALES BY BRANCH
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

sns.barplot(
    data=df,
    x="Branch",
    y="Total",
    estimator="sum",
    errorbar=None
)

plt.title(
    "Total Sales by Branch"
)

plt.xlabel(
    "Branch"
)

plt.ylabel(
    "Total Sales"
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 21. QUANTITY SOLD BY PRODUCT LINE
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 5)
)

sns.barplot(
    data=df,
    x="Product line",
    y="Quantity",
    estimator="sum",
    errorbar=None
)

plt.title(
    "Quantity Sold by Product Line"
)

plt.xlabel(
    "Product Line"
)

plt.ylabel(
    "Quantity Sold"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 22. DISTRIBUTION OF SALES AMOUNT
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

sns.histplot(
    df["Total"],
    bins=30,
    kde=True
)

plt.title(
    "Distribution of Sales Amount"
)

plt.xlabel(
    "Total Sales"
)

plt.ylabel(
    "Frequency"
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 23. DAILY SALES TREND
# ------------------------------------------------------------

daily_sales = (
    df.groupby("Date")["Total"]
    .sum()
    .reset_index()
    .sort_values("Date")
)

plt.figure(
    figsize=(12, 5)
)

sns.lineplot(
    data=daily_sales,
    x="Date",
    y="Total",
    marker="o"
)

plt.title(
    "Daily Sales Trend"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Total Sales"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 24. SALES BY CUSTOMER TYPE
# ------------------------------------------------------------

if "Customer type" in df.columns:

    plt.figure(
        figsize=(8, 5)
    )

    sns.barplot(
        data=df,
        x="Customer type",
        y="Total",
        estimator="sum",
        errorbar=None
    )

    plt.title(
        "Sales by Customer Type"
    )

    plt.xlabel(
        "Customer Type"
    )

    plt.ylabel(
        "Total Sales"
    )

    plt.tight_layout()

    plt.show()


# ------------------------------------------------------------
# 25. PAYMENT METHOD DISTRIBUTION
# ------------------------------------------------------------

if "Payment" in df.columns:

    plt.figure(
        figsize=(8, 5)
    )

    sns.countplot(
        data=df,
        x="Payment"
    )

    plt.title(
        "Payment Method Distribution"
    )

    plt.xlabel(
        "Payment Method"
    )

    plt.ylabel(
        "Number of Transactions"
    )

    plt.tight_layout()

    plt.show()


# ------------------------------------------------------------
# 26. CORRELATION HEATMAP
# ------------------------------------------------------------

numeric_data = (
    df.select_dtypes(
        include=["int64", "float64"]
    )
)

if numeric_data.shape[1] > 1:

    plt.figure(
        figsize=(8, 6)
    )

    sns.heatmap(
        numeric_data.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title(
        "Correlation Heatmap"
    )

    plt.tight_layout()

    plt.show()


# ------------------------------------------------------------
# 27. PROJECT SUMMARY
# ------------------------------------------------------------

print("\n==============================")
print("PROJECT SUMMARY")
print("==============================")

print(
    "Total transactions:",
    len(df)
)

print(
    "Total sales:",
    round(
        df["Total"].sum(),
        2
    )
)

print(
    "Average transaction:",
    round(
        df["Total"].mean(),
        2
    )
)

print(
    "Total quantity sold:",
    int(
        df["Quantity"].sum()
    )
)

print(
    "\nSupermarket Sales Analysis "
    "completed successfully."
)
