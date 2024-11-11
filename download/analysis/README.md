# Interpretable classification of the premature stoppage of clinical trials

Supplementary material to produce the results of the analysis of the classified clinical trials.

This pipeline consists of the following steps:
- Enrichment analysis.
To launch the script as a Pyspark job on a cluster:
```bash
gcloud dataproc clusters create il-big-stop-reasons-2  \
        --image-version=2.1 \
        --project=open-targets-eu-dev \
        --region=europe-west1 \
        --metadata 'PIP_PACKAGES=statsmodels==0.14.0 omegaconf==2.3.0 typer==0.7.0' \
        --initialization-actions gs://goog-dataproc-initialization-actions-europe-west1/python/pip-install.sh \
        --master-machine-type=n1-highmem-96 \
        --enable-component-gateway \
        --single-node \
        --max-idle=10m

# With `stratify-therapeutic-area`, run the analysis choosing `oncology`, `non_oncology`, or `all` records
gcloud dataproc jobs submit pyspark  \
        --cluster=il-big-stop-reasons \
        --files=config.yml \
        --region=europe-west1 \
        --project=open-targets-eu-dev \
        analysis/python/enrichments.py -- config.yml --stratify-therapeutic-area non_oncology

# To determine a baseline between all sets of comparisons, run the analysis with a randomly generated set of associations
gcloud dataproc jobs submit pyspark  \
        --cluster=il-big-stop-reasons \
        --files=config.yml \
        --region=europe-west1 \
        --project=open-targets-eu-dev

# Adjust the p values for multiple testing using the Benjamini-Hochberg procedure by providing the path to the file with the results of the enrichment analysis
gcloud dataproc jobs submit pyspark  \
        --cluster=il-big-stop-reasons \
        --region=europe-west1 \
        --project=open-targets-eu-dev \
        analysis/python/multiple_testing_correction.py  -- "gs://ot-team/irene/stop_reasons/predictions_aggregations_non_oncology"
```
- Visualization of the results.