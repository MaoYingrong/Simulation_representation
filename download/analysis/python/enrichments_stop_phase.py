"""Functions to calculate ORs between the stop reason category of a clinical trial and its clinical phase. Results available at Supplementary Table 7."""

from functools import reduce

import pyspark.sql.functions as f
import typer
from omegaconf import OmegaConf
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType, StringType, StructField, StructType

from analysis.python.utils import _compute_fisher_and_or, aggregations


def prepare_comparisons_df() -> list:
    """Return list of all comparisons to be used in the analysis.
    In this case, we only want to look at the comparison between the stop reason category and the clinical phase.
    """
    comparisons = spark.createDataFrame(
        data=[
            ("reason", "reason"),
        ],
        schema=StructType(
            [
                StructField("comparison", StringType(), True),
                StructField("comparisonType", StringType(), True),
            ]
        ),
    )

    predictions = spark.createDataFrame(
        data=[
            ("phase4", "clinical"),
            ("phase3", "clinical"),
            ("phase32", "clinical"),
            ("phase2", "clinical"),
            ("phase21", "clinical"),
            ("phase1", "clinical"),
            ("phase1early", "clinical"),
            ("phaseOther", "clinical"),
        ],
        schema=StructType(
            [
                StructField("prediction", StringType(), True),
                StructField("predictionType", StringType(), True),
            ]
        ),
    )
    return comparisons.join(predictions, how="full").collect()


def prepare_studies(studies):
    """Processes the studies dataset to keep only the stopped ones in a suitable format to run enrichments on."""
    formatted_studies = (
        studies.filter(f.col("why_stopped").isNotNull())
        .withColumnRenamed("prediction", "reason")
        .withColumn("id", f.monotonically_increasing_id())
        .withColumn("phase4", f.when(f.col("phase") == "Phase 4", f.lit("Phase 4")))
        .withColumn("phase3", f.when(f.col("phase") == "Phase 3", f.lit("Phase 3")))
        .withColumn(
            "phase32",
            f.when(f.col("phase") == "Phase 2/Phase 3", f.lit("Phase 2/Phase 3")),
        )
        .withColumn("phase2", f.when(f.col("phase") == "Phase 2", f.lit("Phase 2")))
        .withColumn(
            "phase21",
            f.when(f.col("phase") == "Phase 1/Phase 2", f.lit("Phase 1/Phase 2")),
        )
        .withColumn("phase1", f.when(f.col("phase") == "Phase 1", f.lit("Phase 1")))
        .withColumn(
            "phase1early",
            f.when(f.col("phase") == "Early Phase 1", f.lit("Early Phase 1")),
        )
        .withColumn("phaseOther", f.when(f.col("phase") == "nan", f.lit("Other")))
    )
    total_cts = formatted_studies.count()
    return formatted_studies.withColumn("total", f.lit(total_cts))


def main(config):
    """Functions to calculate ORs between the stop reason category of a clinical trial and its clinical phase."""
    # Load
    conf = OmegaConf.load(config)
    studies = spark.read.csv(conf.default.input.predictions_freeze_path, header=True)

    # Process
    formatted_studies = prepare_studies(studies)
    agg_setups = prepare_comparisons_df()

    all_comparisons = []
    for row in agg_setups:
        out = aggregations(formatted_studies, *row)
        all_comparisons.append(out)

    schema = StructType(
        [
            StructField("or_result", FloatType(), nullable=False),
            StructField("lower_ci", FloatType(), nullable=False),
            StructField("upper_ci", FloatType(), nullable=False),
            StructField("pvalue", FloatType(), nullable=False),
        ]
    )
    compute_fisher_and_or_udf = f.udf(_compute_fisher_and_or, schema)
    results = (
        reduce(lambda x, y: x.unionByName(y), all_comparisons)
        .coalesce(200)
        .withColumn("b", f.col("predictionTotal") - f.col("a"))
        .withColumn("c", f.col("comparisonTotal") - f.col("a"))
        .withColumn("d", f.col("total") - f.col("a") - f.col("b") - f.col("c"))
        .withColumn("result", compute_fisher_and_or_udf("a", "b", "c", "d"))
        .select("*", "result.*")
        .drop("result")
    )
    results.write.parquet(
        conf.default.stop_vs_phase_enrichments_template,
        mode="overwrite",
    )


if __name__ == "__main__":
    sparkConf = (
        SparkConf()
        .set("spark.hadoop.fs.gs.requester.pays.mode", "AUTO")
        .set("spark.hadoop.fs.gs.requester.pays.project.id", "open-targets-eu-dev")
    )
    spark = SparkSession.builder.getOrCreate()
    typer.run(main)
