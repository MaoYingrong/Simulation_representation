from datasets import load_dataset, DatasetDict, Dataset
import evaluate
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from typing import Union

def explode_label_columns(
    data: Union[Dataset, DatasetDict],
    split_names: list,
    label2id: dict):
    def fix_dataset(dataset):
        pdf = dataset.to_pandas().explode("label_descriptions").rename({"label_descriptions": "label_description"}, axis=1).reset_index()
        pdf["label"] = pdf["label_description"].map(label2id)
        pdf = pdf[["text", "label", "label_description"]]
        good_dataset = Dataset.from_pandas(pdf, preserve_index=False)
        return good_dataset

    ds = DatasetDict()
    if isinstance(data, DatasetDict):
        for split in split_names:
            ds[split] = fix_dataset(data[split])
    
    elif isinstance(data, Dataset):
        split = split_names[0]
        ds[split] = fix_dataset(data)

    return ds

def main():
    # TO-DO: parametrise local model
    dataset_agg = load_dataset("opentargets/clinical_trial_reason_to_stop", split="all")
    model = TFAutoModelForSequenceClassification.from_pretrained("./model_3_epochs_classificator_tf", local_files_only=True)

    dataset = explode_label_columns(dataset_agg, ["all"], model.config.label2id)

    # Evaluate from a local model
    tokenizer = AutoTokenizer.from_pretrained("./model_3_epochs_classificator_tf", local_files_only=True, from_pt=False)
    task_evaluator = evaluate.evaluator("text-classification")
    eval_results = task_evaluator.compute(
        model_or_pipeline=model,
        data=dataset["all"],
        label_mapping=model.config.label2id,
        tokenizer=tokenizer,
    )

    print(eval_results)

if __name__ == '__main__':
    main()