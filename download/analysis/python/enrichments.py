"""Analysis of the results on all clinical trials."""
from enum import Enum
from functools import reduce

import pyspark.sql.functions as F
import typer
from omegaconf import OmegaConf
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType, StringType, StructField, StructType
from pyspark.sql.window import Window
from scipy.stats import fisher_exact
from statsmodels.stats.contingency_tables import Table2x2

CLINVAR_VALIDS = [
    # ClinVar evidence we are interested
    "affects",
    "risk factor",
    "pathogenic",
    "likely pathogenic",
    "protective",
    "drug response",
]

NON_NEUTRAL_PREDICTIONS = [
    # Stop predictions from Olesya
    "Negative",
    "Safety_Sideeffects",
    "Success",
    "Business_Administrative",
    "Invalid_Reason",
]

STOPPED_STATUS = ["Terminated", "Withdrawn", "Suspended"]


class ModeByTA(str, Enum):
    """Mode of analysis by therapeutic area."""

    ALL = "all"
    ONCOLOGY = "oncology"
    NON_ONCOLOGY = "non_oncology"


def expand_disease_index(disease):
    """Expand disease index to include ancestors to account for differences in granularity in the mapping."""
    return (
        disease.select(
            F.col("id").alias("diseaseId"),
            F.explode("ancestors").alias("propagatedDiseaseId"),
        )
        .union(
            disease.select(
                F.col("id").alias("diseaseId"), F.col("id").alias("propagatedDiseaseId")
            )
        )
        .distinct()
    )


def prepare_l2g(evidence):
    """Prepare l2g evidence with some arbitrary cut-offs."""
    return (
        evidence.filter(F.col("datasourceId") == "ot_genetics_portal")
        .groupBy("targetId", "diseaseId")
        .agg(F.max("score").alias("max_l2g"))
        .withColumn(
            "l2g_075",
            F.when(F.col("max_l2g") > 0.75, "l2g_0.75"),
        )
        .withColumn("l2g_05", F.when(F.col("max_l2g") > 0.5, "l2g_0.5"))
        .withColumn(
            "l2g_025",
            F.when(F.col("max_l2g") > 0.25, "l2g_0.25"),
        )
        .withColumn("l2g_01", F.when(F.col("max_l2g") > 0.1, "l2g_0.1"))
        .withColumn(
            "l2g_005",
            F.when(F.col("max_l2g") > 0.05, "l2g_0.05"),
        )
    )


def extract_therapeutic_area(disease):
    """Assigns the most severe therapeutic are to a disease."""
    taDf = spark.createDataFrame(
        data=[
            ("MONDO_0045024", "cell proliferation disorder", "Oncology"),
            ("EFO_0005741", "infectious disease", "Other"),
            ("OTAR_0000014", "pregnancy or perinatal disease", "Other"),
            ("EFO_0005932", "animal disease", "Other"),
            ("MONDO_0024458", "disease of visual system", "Other"),
            ("EFO_0000319", "cardiovascular disease", "Other"),
            ("EFO_0009605", "pancreas disease", "Other"),
            ("EFO_0010282", "gastrointestinal disease", "Other"),
            ("OTAR_0000017", "reproductive system or breast disease", "Other"),
            ("EFO_0010285", "integumentary system disease", "Other"),
            ("EFO_0001379", "endocrine system disease", "Other"),
            ("OTAR_0000010", "respiratory or thoracic disease", "Other"),
            ("EFO_0009690", "urinary system disease", "Other"),
            ("OTAR_0000006", "musculoskeletal or connective tissue disease", "Other"),
            ("MONDO_0021205", "disease of ear", "Other"),
            ("EFO_0000540", "immune system disease", "Other"),
            ("EFO_0005803", "hematologic disease", "Other"),
            ("EFO_0000618", "nervous system disease", "Other"),
            ("MONDO_0002025", "psychiatric disorder", "Other"),
            ("OTAR_0000020", "nutritional or metabolic disease", "Other"),
            ("OTAR_0000018", "genetic, familial or congenital disease", "Other"),
            ("OTAR_0000009", "injury, poisoning or other complication", "Other"),
            ("EFO_0000651", "phenotype", "Other"),
            ("EFO_0001444", "measurement", "Other"),
            ("GO_0008150", "biological process", "Other"),
        ],
        schema=StructType(
            [
                StructField("taId", StringType(), True),
                StructField("taLabel", StringType(), True),
                StructField("taLabelSimple", StringType(), True),
            ]
        ),
    ).withColumn("taRank", F.monotonically_increasing_id())
    wByDisease = Window.partitionBy("diseaseId")
    return (
        disease.withColumn("taId", F.explode("therapeuticAreas"))
        .select(F.col("id").alias("diseaseId"), "taId")
        .join(taDf, on="taId", how="left")
        .withColumn("minRank", F.min("taRank").over(wByDisease))
        .filter(F.col("taRank") == F.col("minRank"))
        .drop("taRank", "minRank")
    )


def prepare_genetic_constraint(target):
    """Prepare genetic constraint data for a given gene."""
    return target.withColumn("gc", F.explode("constraint.upperBin6")).select(
        F.col("id").alias("targetId"),
        F.col("gc").cast("string"),
    )


def prepare_pli(target):
    """Prepare pLI (predicted loss of function intolerant) data for a given gene."""
    return (
        target.withColumn("gc", F.explode("constraint"))
        .filter(F.col("gc.constraintType") == "lof")
        .select(F.col("id").alias("targetId"), F.col("gc.score").alias("pLI"))
        .withColumn(
            "lof_tolerance",
            F.when(F.col("pLI") > 0.9, F.lit("LoF intolerant")).otherwise(
                F.when(F.col("pLI") < 0.1, F.lit("LoF tolerant"))
            ),
        )
        .drop("pLI")
    )


def extract_target_partners(interaction):
    """Extract all partners of a given target."""
    allInteractions = (
        interaction.filter(F.col("sourceDatabase") == "intact")
        .filter(F.col("scoring") > 0.42)
        .filter(F.col("targetB").isNotNull())
        .select("targetA", "targetB")
        .distinct()
    )
    return (
        allInteractions.union(
            allInteractions.select(
                F.col("targetA").alias("targetB"), F.col("targetB").alias("targetA")
            )
        )
        .distinct()
        .groupBy("targetA")
        .agg(F.count("targetB").alias("partners"))
        .withColumn(
            "partnersBin",
            F.when(F.col("partners") > 20, F.lit("greaterThan20"))
            .when(
                (F.col("partners") > 10) & (F.col("partners") <= 20),
                F.lit("from11to20"),
            )
            .when(
                (F.col("partners") > 0) & (F.col("partners") <= 10), F.lit("from1to10")
            )
            .otherwise(F.lit("none")),
        )
        .select(F.col("targetA").alias("targetId"), F.col("partnersBin"))
    )


def prepare_hpa_expression(hpa):
    """Prepare HPA expression data for a given target."""
    return hpa.select(
        F.col("Ensembl").alias("targetId"),
        F.col("RNA tissue distribution").alias("rnaDistribution"),
        F.col("RNA tissue specificity").alias("rnaSpecificity"),
    )


def prepare_associations(evidence, disease_ancestors):
    """Prepare a pseudo-associations dataset that consists of propagating the ontology across the evidence dataset and extract the maximum score per data source."""
    return (
        # Cleaned evidence (exclude "benign" clinvar genetic evidence)
        evidence.withColumn("evaValids", F.array([F.lit(x) for x in CLINVAR_VALIDS]))
        .withColumn("evaFilter", F.arrays_overlap("evaValids", "clinicalSignificances"))
        .filter((F.col("evaFilter").isNull()) | (F.col("evaFilter")))
        # pseudo-associations: ontology propagation + max datasource score
        .join(disease_ancestors, on="diseaseId", how="left")
        .drop("diseaseId")
        .withColumnRenamed("propagatedDiseaseId", "diseaseId")
        .select("targetId", "diseaseId", "datasourceId", "datatypeId")
        .distinct()
    )


def prepare_comparisons_df() -> list:
    """Return list of all comparisons to be used in the analysis."""
    comparisons = spark.createDataFrame(
        data=[
            ("datasourceId", "byDatasource"),
            ("datatypeId", "byDatatype"),
            ("taLabelSimple", "ta"),
            ("gc", "geneticConstraint"),
            ("lof_tolerance", "lof_tolerance"),
            ("rnaDistribution", "rnaDistribution"),
            ("rnaSpecificity", "rnaSpecificity"),
            ("partnersBin", "interactions"),
            ("l2g_075", "l2g"),
            ("l2g_05", "l2g"),
            ("l2g_025", "l2g"),
            ("l2g_01", "l2g"),
            ("l2g_005", "l2g"),
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
            ("reason", "reason"),
            ("metareason", "metareason"),
            ("stopStatus", "stopStatus"),
            ("isStopped", "isStopped"),
            ("phase4", "clinical"),
            ("phase3", "clinical"),
            ("phase2", "clinical"),
        ],
        schema=StructType(
            [
                StructField("prediction", StringType(), True),
                StructField("predictionType", StringType(), True),
            ]
        ),
    )
    return comparisons.join(predictions, how="full").collect()


def aggregations(
    df, comparisonColumn, comparisonType, predictionColumn, predictionType
):
    """Aggregate data to compute enrichment analysis."""
    comparison_counts = df.groupBy(comparisonColumn).agg(
        F.countDistinct("id").alias("comparisonTotal")
    )
    prediction_counts = df.groupBy(predictionColumn).agg(
        F.countDistinct("id").alias("predictionTotal")
    )
    intersection_counts = df.groupBy(comparisonColumn, predictionColumn).agg(
        F.countDistinct("id").alias("a")
    )

    return (
        df.filter(
            F.col(comparisonColumn).isNotNull() & F.col(predictionColumn).isNotNull()
        )
        .withColumn("comparisonType", F.lit(comparisonType))
        .withColumn("predictionType", F.lit(predictionType))
        .join(comparison_counts, on=comparisonColumn, how="left")
        .join(prediction_counts, on=predictionColumn, how="left")
        .join(intersection_counts, on=[comparisonColumn, predictionColumn], how="left")
        .select(
            F.col(predictionColumn).alias("prediction"),
            F.col(comparisonColumn).alias("comparison"),
            "comparisonType",
            "predictionType",
            "a",
            "predictionTotal",
            "comparisonTotal",
            "total",
        )
        .distinct()
    )


def _compute_fisher_and_or(a, b, c, d) -> tuple:
    """Compute Fisher's exact test by shaping each row into a 2x2 contingency table."""
    table = [[a, b], [c, d]]
    _, pvalue = fisher_exact(table)
    table2x2 = Table2x2(table)
    or_result = table2x2.oddsratio
    lower_ci, upper_ci = table2x2.oddsratio_confint()
    return (float(or_result), float(lower_ci), float(upper_ci), float(pvalue))


def main(
    config,
    stratify_therapeutic_area: ModeByTA = ModeByTA.ALL,
):
    """Run enrichment analysis."""
    conf = OmegaConf.load(config)
    print("Therapeutic area analysis mode:", stratify_therapeutic_area.value)
    print("Config", conf)
    # Load raw datasets
    evidence = spark.read.parquet(conf.default.input.evidence_path)
    disease = spark.read.parquet(conf.default.input.disease_path)
    target = spark.read.parquet(conf.default.input.target_path)
    interaction = spark.read.parquet(conf.default.input.interactions_path)
    hpa = spark.read.json(conf.default.input.hpa_path)
    stop_predictions = (
        spark.read.csv(conf.default.input.predictions_freeze_path, header=True)
        # add prediction metaclass
        .withColumn(
            "metareason",
            F.when(
                F.col("prediction").isin(NON_NEUTRAL_PREDICTIONS), F.col("prediction")
            ).otherwise(F.lit("Neutral")),
        )
    ).withColumnRenamed("prediction", "reason")

    # Prepare processed datasets
    disease_ancestors = expand_disease_index(disease)
    l2g = prepare_l2g(evidence)
    disease_ta = extract_therapeutic_area(disease)
    target_gc = prepare_genetic_constraint(target)
    target_pli = prepare_pli(target)
    partners = extract_target_partners(interaction)
    hpa_expr = prepare_hpa_expression(hpa)
    associations = prepare_associations(evidence, disease_ancestors)

    # Prepare clinical information
    clinical = (
        evidence.filter(F.col("sourceId") == "chembl")
        .withColumn("urls", F.explode("urls"))
        .withColumn(
            "nctid", F.regexp_extract(F.col("urls.url"), "(.+)(id=%22)(.+)(%22)", 3)
        )
        .withColumn(
            "nctid", F.when(F.col("nctid") != "", F.col("nctid")).otherwise(None)
        )
        .withColumn(
            "stopStatus",
            F.when(
                F.col("clinicalStatus").isin(STOPPED_STATUS), F.col("clinicalStatus")
            ),
        )
        .withColumn(
            "isStopped",
            F.when(F.col("clinicalStatus").isin(STOPPED_STATUS), F.lit("stopped")),
        )
        .withColumn(
            "phase4",
            F.when(F.col("clinicalPhase") == 4, F.lit("Phase IV")),
        )
        .withColumn(
            "phase3",
            F.when(F.col("clinicalPhase") >= 3, F.lit("Phase III+")),
        )
        .withColumn(
            "phase2",
            F.when(F.col("clinicalPhase") >= 2, F.lit("Phase II+")),
        )
        .select(
            "targetId",
            "diseaseId",
            "nctid",
            "clinicalStatus",
            "clinicalPhase",
            "studyStartDate",
            "stopStatus",
            "isStopped",
            "phase4",
            "phase3",
            "phase2",
        )
        .distinct()
        # Create ID
        .withColumn("id", F.xxhash64("targetId", "diseaseId", "nctid"))
        # Bring reason and metareason for stoppage
        .join(stop_predictions, on="nctid", how="left")
        # L2G cut-offs
        .join(l2g, on=["targetId", "diseaseId"], how="left")
        # Disease therapeutic area (only one by disease)
        .join(disease_ta, on="diseaseId", how="left")
        # Target genetic constraint
        .join(target_gc, on="targetId", how="left")
        # Target lof tolerance
        .join(target_pli, on="targetId", how="left")
        # Expression specificity
        .join(hpa_expr, on="targetId", how="left")
        # Physical interaction partners
        .join(partners, on="targetId", how="left")
        .withColumn("partnersBin", F.coalesce(F.col("partnersBin"), F.lit("none")))
        # Datasources and Datatypes
        .join(associations, on=["targetId", "diseaseId"], how="left")
    )
    # Run analysis split by oncology/non_oncology data
    if stratify_therapeutic_area.value == "oncology":
        clinical = clinical.filter(F.col("taLabelSimple") == "Oncology")
    elif stratify_therapeutic_area.value == "non_oncology":
        clinical = clinical.filter(F.col("taLabelSimple") != "Oncology")
    # Define total number
    uniqIds = clinical.select("id").distinct().count()
    clinical = clinical.withColumn("total", F.lit(uniqIds)).persist()

    ## Compute aggregations
    agg_setups = prepare_comparisons_df()
    all_comparisons = []
    for row in agg_setups:
        out = aggregations(clinical, *row)
        all_comparisons.append(out)

    ## Compute Fisher's exact test and odds ratio
    schema = StructType(
        [
            StructField("or_result", FloatType(), nullable=False),
            StructField("lower_ci", FloatType(), nullable=False),
            StructField("upper_ci", FloatType(), nullable=False),
            StructField("pvalue", FloatType(), nullable=False),
        ]
    )
    compute_fisher_and_or_udf = F.udf(_compute_fisher_and_or, schema)
    analysis_result = (
        reduce(lambda x, y: x.unionByName(y), all_comparisons)
        .coalesce(200)
        .withColumn("b", F.col("predictionTotal") - F.col("a"))
        .withColumn("c", F.col("comparisonTotal") - F.col("a"))
        .withColumn("d", F.col("total") - F.col("a") - F.col("b") - F.col("c"))
        .withColumn("result", compute_fisher_and_or_udf("a", "b", "c", "d"))
        .select("*", "result.*")
        .drop("result")
    )
    analysis_result.write.parquet(
        f"{conf.default.enrichments_template}_{stratify_therapeutic_area.value}",
        mode="overwrite",
    )
    return analysis_result


if __name__ == "__main__":
    sparkConf = (
        SparkConf()
        .set("spark.hadoop.fs.gs.requester.pays.mode", "AUTO")
        .set("spark.hadoop.fs.gs.requester.pays.project.id", "open-targets-eu-dev")
    )
    spark = SparkSession.builder.config(conf=sparkConf).getOrCreate()

    typer.run(main)
