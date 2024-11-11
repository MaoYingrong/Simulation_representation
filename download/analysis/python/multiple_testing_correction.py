"""Applying Benjamini/Hochberg correction to account for false discovery rate."""

import typer
from pyspark.sql import SparkSession
from statsmodels.stats.multitest import fdrcorrection


def apply_fdr_correction(df):
    """Apply Benjamini/Hochberg correction to account for false discovery rate."""
    pvals = df["pvalue"].values
    _, pvals_corrected = fdrcorrection(
        pvals, alpha=0.05, method="indep", is_sorted=False
    )
    df["pvalue_corrected"] = pvals_corrected
    return df


def plot_pvalue_scatter(df, figure_name, username, api_key):
    """Plot a scatter plot of the observed p-values vs. the adjusted p-values after multiple testing correction.

    The plot is saved to the plotly cloud and can be accessed via the link in the terminal output.

    Args:
        df (pandas.DataFrame): DataFrame containing the results of the enrichment analysis.
        figure_name (str): Name of the plotly figure.
        username (str): Username for the plotly account.
        api_key (str): API key for the plotly account.
    """
    import chart_studio
    import chart_studio.plotly as py
    import plotly.express as px

    chart_studio.tools.set_credentials_file(username=username, api_key=api_key)

    df["significance"] = "Remains Significant"
    df.loc[
        (df["pvalue"] < 0.05) & (df["pvalue_corrected"] >= 0.05), "significance"
    ] = "No Longer Significant"
    df.loc[
        (df["pvalue"] >= 0.05) & (df["pvalue_corrected"] < 0.05), "significance"
    ] = "Now Significant"
    df.loc[
        (df["pvalue"] >= 0.05) & (df["pvalue_corrected"] >= 0.05), "significance"
    ] = "Never Significant"

    fig = px.scatter(
        df,
        x="pvalue",
        y="pvalue_corrected",
        hover_data=["comparison", "prediction", "or_result"],
        color="significance",
        color_discrete_sequence=["green", "red", "blue", "black"],
        log_x=True,
        log_y=True,
        title="Comparison of df and Empirical P-Values",
    )
    fig.add_shape(
        type="line",
        x0=df["pvalue"].min(),
        x1=df["pvalue"].max(),
        y0=0.05,
        y1=0.05,
        line=dict(color="Red", dash="dash"),
    )

    fig.add_shape(
        type="line",
        x0=0.05,
        x1=0.05,
        y0=df["pvalue_corrected"].min(),
        y1=df["pvalue_corrected"].max(),
        line=dict(color="Red", dash="dash"),
    )
    fig.update_layout(xaxis_title="Observed P-Values", yaxis_title="Corrected P-Values")
    py.plot(fig, filename=figure_name, auto_open=True)


def main(data_path):
    """Apply Benjamini/Hochberg correction to account for false discovery rate in every predictionType/comparisonType pair.

    Args:
        data_path (str): Path to the parquet file containing the results of the enrichment analysis.
    """
    spark = SparkSession.builder.getOrCreate()
    enrichments = spark.read.parquet(data_path).toPandas()

    enrichments = enrichments.groupby(["predictionType", "comparisonType"]).apply(
        apply_fdr_correction
    )
    output_data_path = f"{data_path}_corrected"
    (
        spark.createDataFrame(enrichments)
        .write.mode("overwrite")
        .parquet(output_data_path)
    )
    print(f"Adjusted data saved to {output_data_path}.")


if __name__ == "__main__":
    typer.run(main)
