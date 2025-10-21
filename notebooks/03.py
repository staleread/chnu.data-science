# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Any
from dateutil.parser import parse as parse_date
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# %%
df = pd.read_csv("data/owid-covid-data.csv")

# %%
df.info()

# %%
cols_to_convert = ["iso_code", "continent", "location", "tests_units"]
df[cols_to_convert] = df[cols_to_convert].apply(lambda s: s.astype("category"))
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# %%
df.info()

# %%
continent_counts = df["continent"].value_counts(dropna=False)
continent_counts.to_frame()

# %%
tests_units_counts = df["tests_units"].value_counts(dropna=False)
tests_units_counts.to_frame()

# %%
pd.options.display.float_format = "{:,.2f}".format

# %%
continent_sum = df.groupby(["continent"], observed=False)["total_cases"].sum()
continent_sum.sort_values(ascending=False)

# %%
tests_units_sum = df.groupby(["tests_units"], observed=False)["total_cases"].sum()
tests_units_sum.sort_values(ascending=False)

# %%
pivot = df.pivot_table(
    values="total_cases",
    columns="location",
    index="date",
    aggfunc="sum",
    observed=False,
)
pivot.shape

# %%
countries = ["Albania", "North Macedonia", "Serbia", "Croatia"]
pivot = pivot[countries].dropna(how="any")
pivot.to_csv("data/cases_by_country.csv")
pivot.shape

# %%
pivot.plot()
plt.savefig("images/cases_by_country.png")

# %%
pivot.corr(method="pearson")
# pivot.corr(method="spearman")
# pivot.corr(method="kendall")

# %%
def compare_columns(df: pd.DataFrame, col_x: str, col_y: str) -> dict[str, Any]:
    if col_x not in df.columns or col_y not in df.columns:
        raise KeyError(f"Both '{col_x}' and '{col_y}' must be in DataFrame columns")
    x = df[col_x].astype(float).values.reshape(-1, 1)
    y = df[col_y].astype(float).values
    regressor = LinearRegression().fit(x, y)
    y_pred = regressor.predict(x)
    r2 = r2_score(y, y_pred)
    corr = np.corrcoef(x.flatten(), y)[0, 1]
    print(f"R^2: {r2_score(y, y_pred):.2f}")
    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, label="Data points", alpha=0.7)
    plt.plot(x, y_pred, color="tab:orange", linewidth=2, label="Fitted line")
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    plt.title(f"Relation between {col_x} and {col_y}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"images/relation_{col_x.lower()}_{col_y.lower()}.png")

# %%
compare_columns(pivot, "Albania", "North Macedonia")
compare_columns(pivot, "Albania", "Serbia")
compare_columns(pivot, "Albania", "Croatia")

# %%
def plot_regression(df: pd.DataFrame, column: str, test_size: float) -> dict[str, Any]:
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")
    series = df[column].copy()
    dates = series.index
    X = np.array([parse_date(d).toordinal() for d in dates]).reshape(-1, 1)
    y = series.values.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    regressor = LinearRegression().fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    print(f"R^2: {r2_score(y_test, y_pred):.2f}")
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5), sharex=True, sharey=True)
    ax[0].scatter(X_train, y_train, label="Train data points")
    ax[0].plot(
        X_train,
        regressor.predict(X_train),
        linewidth=3,
        color="tab:orange",
        label="Model predictions",
    )
    ax[0].set(xlabel="Date", ylabel="Infected count", title="Train set")
    ax[0].legend()
    ax[1].scatter(X_test, y_test, label="Test data points")
    ax[1].plot(X_test, y_pred, linewidth=3, color="tab:orange", label="Model predictions")
    ax[1].set(xlabel="Feature", ylabel="Target", title="Test set")
    ax[1].legend()
    fig.suptitle("Linear Regression")
    plt.savefig(f"images/predict_{column.lower()}.png")

# %%
plot_regression(pivot, "Albania", 30)
plot_regression(pivot, "North Macedonia", 30)
plot_regression(pivot, "Serbia", 30)
plot_regression(pivot, "Croatia", 30)
