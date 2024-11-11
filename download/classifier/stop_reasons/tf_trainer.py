import logging
from typing import Optional

from datasets import load_dataset
from huggingface_hub import login
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, DefaultDataCollator, PushToHubCallback, TFAutoModelForSequenceClassification
import typer

from stop_reasons.utils import (convert_to_tf_dataset, get_label_metadata, prepare_splits_for_training)

def main(
    epochs: int = typer.Argument(...),
    output_path: str = typer.Argument(...),
    subset_data: bool = typer.Option(False),
    push_to_hub: bool = typer.Option(False),
    personal_token: Optional[str] = typer.Argument(None),
) -> None:
    """
    Main logic of the fine-tuning process: this function loads the dataset, tokenizes it,
    splits it into train and validation sets, loads the model, trains it, and saves it
    
    Args:
      subset_data (bool): flag to indicate whether to use a subset of the data for testing purposes
      epochs (int): number of epochs to train for
      output_path (str): the path to the directory where the model will be saved.
      push_to_hub (bool): flag to indicate whether to push the model to the hub
      personal_token (str | None): your personal Hugging Face Hub token
    """

    logging.basicConfig(level=logging.INFO)

    logging.info("Download dataset from HF's datasets...")
    dataset = load_dataset("opentargets/clinical_trial_reason_to_stop", split='train').train_test_split(test_size=0.1, seed=42)
    global labels
    labels, id2label, label2id = get_label_metadata(dataset)

    logging.info("Tokenizing dataset...")
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    encoded_dataset = dataset.map(encode_text, batched=True)

    logging.info("Splitting dataset into train and validation sets...")
    train_dataset, eval_dataset = prepare_splits_for_training(encoded_dataset, subset_data)
    data_collator = DefaultDataCollator(return_tensors="tf")
    tf_train_dataset = convert_to_tf_dataset(train_dataset, data_collator, shuffle_flag=True, batch_size=32)
    tf_validation_dataset = convert_to_tf_dataset(eval_dataset, data_collator, shuffle_flag=False, batch_size=32)

    model = TFAutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(labels),
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=tf.keras.metrics.BinaryAccuracy(),
    )
    callback = None
    if push_to_hub:
        login(token=personal_token)
        callback = PushToHubCallback(
            output_dir=output_path, tokenizer=tokenizer, hub_model_id="opentargets/stop_reasons_multi_direct"
        )
    
    logging.info("BERT loaded. Starting fine-tuning classifier...")
    model.fit(tf_train_dataset, epochs=epochs, validation_data=tf_validation_dataset, callbacks=callback)

    model.save_pretrained(output_path, saved_model=True, save_format="tf")
    tokenizer.save_pretrained(f"{output_path}_tokenizer")
    logging.info("Model and tokenizer saved. Exiting.")

def encode_text(dataset_split):
    """Tokenize texts and align labels with them."""
    text = dataset_split["text"]
    encoding = tokenizer(text, padding="max_length", truncation=True)
    labels_batch = {k: dataset_split[k] for k in dataset_split.keys() if k in labels}
    # create numpy array of shape (batch_size, num_labels)
    labels_matrix = np.zeros((len(text), len(labels)))
    # fill numpy array
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]

    encoding["labels"] = labels_matrix.tolist()
    return encoding

if __name__ == "__main__":
    typer.run(main)