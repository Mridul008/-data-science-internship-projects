import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# MOVIE RATING ANALYSIS
# ============================================================


# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

file_path = "ratings.csv"

ratings = pd.read_csv(
    file_path
)

print(
    "Movie ratings dataset loaded successfully."
)


# ------------------------------------------------------------
# 2. FIRST 5 ROWS
# ------------------------------------------------------------

print("\n==============================")
print("FIRST 5 ROWS")
print("==============================")

print(
    ratings.head()
)


# ------------------------------------------------------------
# 3. DATASET SHAPE
# ------------------------------------------------------------

print("\n==============================")
print("DATASET SHAPE")
print("==============================")

print(
    "Rows:",
    ratings.shape[0]
)

print(
    "Columns:",
    ratings.shape[1]
)


# ------------------------------------------------------------
# 4. COLUMN NAMES
# ------------------------------------------------------------

print("\n==============================")
print("COLUMN NAMES")
print("==============================")

print(
    ratings.columns.tolist()
)


# ------------------------------------------------------------
# 5. DATA TYPES
# ------------------------------------------------------------

print("\n==============================")
print("DATA TYPES")
print("==============================")

print(
    ratings.dtypes
)


# ------------------------------------------------------------
# 6. MISSING VALUES
# ------------------------------------------------------------

print("\n==============================")
print("MISSING VALUES")
print("==============================")

print(
    ratings.isnull().sum()
)


# ------------------------------------------------------------
# 7. DUPLICATE RECORDS
# ------------------------------------------------------------

print("\n==============================")
print("DUPLICATE RECORDS")
print("==============================")

duplicate_count = (
    ratings.duplicated().sum()
)

print(
    "Duplicate rows:",
    duplicate_count
)

if duplicate_count > 0:

    ratings = ratings.drop_duplicates()

    print(
        "Duplicate rows removed."
    )


# ------------------------------------------------------------
# 8. CLEAN RATING COLUMN
# ------------------------------------------------------------

ratings["rating"] = pd.to_numeric(
    ratings["rating"],
    errors="coerce"
)

ratings = ratings.dropna(
    subset=["rating"]
)


# ------------------------------------------------------------
# 9. RATING STATISTICS
# ------------------------------------------------------------

print("\n==============================")
print("RATING STATISTICS")
print("==============================")

print(
    ratings["rating"].describe()
)


# ------------------------------------------------------------
# 10. RATING RANGE
# ------------------------------------------------------------

print("\n==============================")
print("RATING RANGE")
print("==============================")

print(
    "Minimum rating:",
    ratings["rating"].min()
)

print(
    "Maximum rating:",
    ratings["rating"].max()
)

print(
    "Average rating:",
    round(
        ratings["rating"].mean(),
        2
    )
)


# ------------------------------------------------------------
# 11. DISTRIBUTION OF MOVIE RATINGS
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

sns.histplot(
    ratings["rating"],
    bins=10,
    kde=True
)

plt.title(
    "Distribution of Movie Ratings"
)

plt.xlabel(
    "Rating"
)

plt.ylabel(
    "Frequency"
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 12. RATING COUNTS
# ------------------------------------------------------------

rating_counts = (
    ratings["rating"]
    .value_counts()
    .sort_index()
)

print("\n==============================")
print("RATING COUNTS")
print("==============================")

print(
    rating_counts
)


# ------------------------------------------------------------
# 13. NUMBER OF RATINGS FOR EACH SCORE
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

sns.barplot(
    x=rating_counts.index,
    y=rating_counts.values
)

plt.title(
    "Number of Ratings for Each Score"
)

plt.xlabel(
    "Rating"
)

plt.ylabel(
    "Number of Ratings"
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 14. USER ANALYSIS
# ------------------------------------------------------------

if "userId" in ratings.columns:

    unique_users = (
        ratings["userId"].nunique()
    )

    print("\n==============================")
    print("USER ANALYSIS")
    print("==============================")

    print(
        "Number of unique users:",
        unique_users
    )


# ------------------------------------------------------------
# 15. MOVIE ANALYSIS
# ------------------------------------------------------------

if "movieId" in ratings.columns:

    unique_movies = (
        ratings["movieId"].nunique()
    )

    print(
        "Number of unique movies:",
        unique_movies
    )


# ------------------------------------------------------------
# 16. MOST ACTIVE USERS
# ------------------------------------------------------------

if "userId" in ratings.columns:

    user_activity = (
        ratings.groupby("userId")
        .size()
        .sort_values(
            ascending=False
        )
    )

    print("\n==============================")
    print("MOST ACTIVE USERS")
    print("==============================")

    print(
        user_activity.head(10)
    )


# ------------------------------------------------------------
# 17. MOST RATED MOVIES
# ------------------------------------------------------------

if "movieId" in ratings.columns:

    movie_activity = (
        ratings.groupby("movieId")
        .size()
        .sort_values(
            ascending=False
        )
    )

    print("\n==============================")
    print("MOST RATED MOVIES")
    print("==============================")

    print(
        movie_activity.head(10)
    )


# ------------------------------------------------------------
# 18. MOVIE RATING SUMMARY
# ------------------------------------------------------------

if "movieId" in ratings.columns:

    movie_summary = (
        ratings.groupby("movieId")
        .agg(
            Mean_Rating=(
                "rating",
                "mean"
            ),

            Rating_Count=(
                "rating",
                "count"
            )
        )
        .reset_index()
    )

    print("\n==============================")
    print("MOVIE RATING SUMMARY")
    print("==============================")

    print(
        movie_summary.head()
    )


# ------------------------------------------------------------
# 19. TOP-RATED MOVIES
# ------------------------------------------------------------

if "movieId" in ratings.columns:

    minimum_count = 20

    ranked_movies = (
        movie_summary[
            movie_summary[
                "Rating_Count"
            ] >= minimum_count
        ]
        .sort_values(
            [
                "Mean_Rating",
                "Rating_Count"
            ],
            ascending=[
                False,
                False
            ]
        )
    )

    print("\n==============================")
    print("TOP-RATED MOVIES")
    print("==============================")

    print(
        ranked_movies.head(20)
    )


# ------------------------------------------------------------
# 20. MOST POPULAR MOVIES
# ------------------------------------------------------------

if "movieId" in ratings.columns:

    popular_movies = (
        movie_summary
        .sort_values(
            "Rating_Count",
            ascending=False
        )
    )

    print("\n==============================")
    print("MOST POPULAR MOVIES")
    print("==============================")

    print(
        popular_movies.head(20)
    )


# ------------------------------------------------------------
# 21. USER RATING SUMMARY
# ------------------------------------------------------------

if "userId" in ratings.columns:

    user_summary = (
        ratings.groupby("userId")
        .agg(
            Average_Rating=(
                "rating",
                "mean"
            ),

            Rating_Count=(
                "rating",
                "count"
            )
        )
        .reset_index()
    )

    print("\n==============================")
    print("USER RATING SUMMARY")
    print("==============================")

    print(
        user_summary.head()
    )


# ------------------------------------------------------------
# 22. TOP 10 MOST ACTIVE USERS GRAPH
# ------------------------------------------------------------

if "userId" in ratings.columns:

    top_users = (
        user_activity
        .head(10)
        .sort_values()
    )

    plt.figure(
        figsize=(8, 5)
    )

    top_users.plot(
        kind="barh"
    )

    plt.title(
        "Top 10 Most Active Users"
    )

    plt.xlabel(
        "Number of Ratings"
    )

    plt.ylabel(
        "User ID"
    )

    plt.tight_layout()

    plt.show()


# ------------------------------------------------------------
# 23. TOP 10 MOST RATED MOVIES GRAPH
# ------------------------------------------------------------

if "movieId" in ratings.columns:

    top_movies = (
        movie_activity
        .head(10)
        .sort_values()
    )

    plt.figure(
        figsize=(8, 5)
    )

    top_movies.plot(
        kind="barh"
    )

    plt.title(
        "Top 10 Most Rated Movies"
    )

    plt.xlabel(
        "Number of Ratings"
    )

    plt.ylabel(
        "Movie ID"
    )

    plt.tight_layout()

    plt.show()


# ------------------------------------------------------------
# 24. AVERAGE RATING VS RATING COUNT
# ------------------------------------------------------------

if "movieId" in ratings.columns:

    plt.figure(
        figsize=(8, 5)
    )

    sns.scatterplot(
        data=movie_summary,
        x="Rating_Count",
        y="Mean_Rating"
    )

    plt.title(
        "Average Rating versus Rating Count"
    )

    plt.xlabel(
        "Number of Ratings"
    )

    plt.ylabel(
        "Average Rating"
    )

    plt.tight_layout()

    plt.show()


# ------------------------------------------------------------
# 25. BOX PLOT OF MOVIE RATINGS
# ------------------------------------------------------------

plt.figure(
    figsize=(7, 5)
)

sns.boxplot(
    x=ratings["rating"]
)

plt.title(
    "Box Plot of Movie Ratings"
)

plt.xlabel(
    "Rating"
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 26. TOP 10 HIGHLY RATED MOVIES
# ------------------------------------------------------------

if "movieId" in ratings.columns:

    top_10_movies = (
        ranked_movies
        .head(10)
        .sort_values(
            "Mean_Rating"
        )
    )

    plt.figure(
        figsize=(9, 6)
    )

    sns.barplot(
        data=top_10_movies,
        x="Mean_Rating",
        y="movieId"
    )

    plt.title(
        "Top 10 Highly Rated Movies"
    )

    plt.xlabel(
        "Average Rating"
    )

    plt.ylabel(
        "Movie ID"
    )

    plt.tight_layout()

    plt.show()


# ------------------------------------------------------------
# 27. CORRELATION HEATMAP
# ------------------------------------------------------------

numeric_columns = (
    ratings.select_dtypes(
        include=["int64", "float64"]
    )
)

if numeric_columns.shape[1] > 1:

    plt.figure(
        figsize=(7, 5)
    )

    sns.heatmap(
        numeric_columns.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title(
        "Correlation Between Numerical Variables"
    )

    plt.tight_layout()

    plt.show()


# ------------------------------------------------------------
# 28. PROJECT INSIGHTS
# ------------------------------------------------------------

print("\n==============================")
print("PROJECT INSIGHTS")
print("==============================")

print(
    "Average rating:",
    round(
        ratings["rating"].mean(),
        2
    )
)


if "userId" in ratings.columns:

    print(
        "Total users:",
        ratings["userId"].nunique()
    )


if "movieId" in ratings.columns:

    print(
        "Total movies:",
        ratings["movieId"].nunique()
    )

    print(
        "Most rated movie ID:",
        movie_activity.idxmax()
    )

    print(
        "Highest-rated movie ID:",
        ranked_movies.iloc[0]["movieId"]
    )

    print(
        "Highest average rating:",
        round(
            ranked_movies.iloc[0][
                "Mean_Rating"
            ],
            2
        )
    )


print(
    "\nMovie Rating Analysis "
    "completed successfully."
)
