# Why has a clinical trial stopped early?
Implementation of a NLP classifier based on [BERT](https://huggingface.co/bert-base-uncased) that assigns one or multiple classes to the reason why a clinical trial has stopped early.

This provides a structured representation of why a clinical trial has stopped early.

## Usage
- `stop_reasons/pt_trainer.py`: fine tuning of a BERT model on the stop reasons dataset. This is using the Trainer API and produces weights in Pytorch format. See usage instructions below.
- `stop_reasons/tf_trainer.py`: analogous to the previous script, but instantiating the Tensorflow classifier model to export the weights in TF format.
- `inference_local.py`: inference script to run locally. It loads the model and runs inference on a given input.
- `inference_spark.py`: inference script to run on a Spark cluster. It loads the model and runs inference on a given dataset using Spark NLP as a framework.
- `push_to_hub.py`, `evaluate.py`, and `utils.py`: helper functions to do specific tasks to test the outputs.

## Development

1. Create environment and install dependencies. You need to have Git LFS installed in your system.

```bash

```bash
poetry env use 3.10.8
poetry install
```

2. Bundle package.

```bash

VERSION_NO=0.3.0

rm -rf ./dist
poetry build
cp ./stop_reasons/*.py ./dist
gsutil cp ./dist/stopreasons-${VERSION_NO}-py3-none-any.whl gs://ot-team/irene/bert/initialisation/
gsutil cp ./utils/initialise_cluster.sh gs://ot-team/irene/bert/initialisation/
```

3. Set up virtual machine (dataproc cluster, in my case).

```bash
gcloud dataproc clusters create il-stop-reasons \
    --image-version=2.1 \
    --project=open-targets-eu-dev \
    --region=europe-west1 \
    --master-machine-type=n1-highmem-32 \
    --metadata="PACKAGE=gs://ot-team/irene/bert/initialisation/stopreasons-${VERSION_NO}-py3-none-any.whl" \
    --initialization-actions=gs://ot-team/irene/bert/initialisation/initialise_cluster.sh \
    --enable-component-gateway \
    --single-node \
    --max-idle=10m
```

4. Run training step.

You can parametrise the training step by passing arguments to the script. These are the available options:
    
```bash

python classifier/stop_reasons/pt_trainer.py --help
Usage: pt_trainer.py [OPTIONS] EPOCHS OUTPUT_MODEL_NAME

Main logic of the fine-tuning process: this function loads the dataset,
tokenizes it, splits it into train and validation sets, loads the model,
trains it, and saves it

Args:   subset_data (bool): flag to indicate whether to use a subset of the
data for testing purposes   epochs (int): number of epochs to train for
output_path (str): the path to the directory where the model will be saved.
push_to_hub (bool): flag to indicate whether to push the model to the hub
personal_token (str | None): your personal Hugging Face Hub token

Arguments:
  EPOCHS             [required]
  OUTPUT_MODEL_NAME  [required]

Options:
  --subset-data / --no-subset-data
                                  [default: no-subset-data]
  --push-to-hub / --no-push-to-hub
                                  [default: no-push-to-hub]
  --personal-token TEXT
  --help                          Show this message and exit.
```


a) Submitting a job to the cluster.
```bash
gcloud dataproc jobs submit pyspark ./dist/train.py \
    --cluster=il-stop-reasons \
    --py-files=gs://ot-team/irene/bert/initialisation/stopreasons-${VERSION_NO}-py3-none-any.whl \
    --project=open-targets-eu-dev  \
    --region=europe-west1
```

b) Running the job locally.
```bash
poetry run python ./dist/pt_trainer.py \
    --subset-data \
    1 \
    "stop_reasons_classificator_multi_label_test"
```