import logging
from typing import Optional

from datasets import load_dataset
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer
import typer

from stop_reasons.utils import compute_metrics, get_label_metadata, prepare_splits_for_training, preprocess_text

def tokenize(batch):
    """Tokenises the text and creates a numpy array with its assigned labels."""
    text = [preprocess_text(text) for text in batch["text"]]
    encoding = tokenizer(text, max_length=177, padding="max_length", truncation=True)

    labels_batch = {k: batch[k] for k in batch.keys() if k in labels}
    labels_matrix = np.zeros((len(text), len(labels)))
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]
    encoding["labels"] = labels_matrix.tolist()
    return encoding

class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss function calculation using BCEWithLogitsLoss, it returns the loss and the outputs if the
        return_outputs flag is set to True
        This function is used during training, evaluation, and prediction; specifically every time a batch is processed.
        The default loss function is here https://github.com/huggingface/transformers/blob/820c46a707ddd033975bc3b0549eea200e64c7da/src/transformers/trainer.py#L2561
        
        Args:
          model: the model we're training
          inputs: a dictionary of input tensors
          return_outputs: if True, the loss and the model outputs are returned. If False, only the loss is
        returned. Defaults to False
        
        Returns:
          The loss and the outputs of the model.
        """
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        # compute custom loss
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), 
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss


def instantiate_classifier(labels, id2label, label2id):
    """
    We're instantiating a BERT model, and then replacing the classification layer with a custom one for our task.
    
    Args:
      labels: a list of all the labels in the dataset
      id2label: a dictionary mapping from label ids to label names
      label2id: a dictionary mapping labels to integers
    
    Returns:
      A model with a classifier that has 3 layers.
    """

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        problem_type="multi_label_classification",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )
    model.classifier = nn.Sequential(
        nn.Linear(768, 50),
        nn.ReLU(),
        nn.Linear(50, len(labels))
    )
    return model


def main(
    epochs: int = typer.Argument(...),
    output_model_name: str = typer.Argument(...),
    subset_data: bool = typer.Option(False),
    push_to_hub: bool = typer.Option(False),
    personal_token: Optional[str] = typer.Option(None),
) -> Trainer:
    """
    Main logic of the fine-tuning process: this function loads the dataset, tokenizes it,
    splits it into train and validation sets, loads the model, trains it, and saves it
    
    Args:
      epochs (int): number of epochs to train for
      output_model_name (str): filename and path to the directory where the model will be saved.
      subset_data (bool): flag to indicate whether to use a subset of the data for testing purposes
      push_to_hub (bool): flag to indicate whether to push the model to the hub
      personal_token (str | None): your personal Hugging Face Hub token
    """
    
    logging.basicConfig(level=logging.INFO)

    dataset = load_dataset("opentargets/clinical_trial_reason_to_stop", split='train').train_test_split(test_size=0.1, seed=42)
    global labels
    labels, id2label, label2id = get_label_metadata(dataset)

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    dataset_cols = [col for col in dataset["train"].column_names if col not in ["text", "input_ids", "attention_mask", "labels"]]
    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset_cols)

    train_dataset, test_dataset = prepare_splits_for_training(tokenized_dataset, subset_data)
    logging.info(f"Train dataset length: {len(train_dataset)}")
    logging.info(f"Test dataset length: {len(test_dataset)}")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=output_model_name,
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        data_seed=42,
        num_train_epochs=epochs,
        metric_for_best_model="f1",
        save_total_limit=2,
        save_strategy="no",
        load_best_model_at_end=False,
        push_to_hub=push_to_hub,
        hub_model_id=output_model_name,
        hub_token=personal_token,
        hub_private_repo=False,
    )
    trainer = MultilabelTrainer(
        model=instantiate_classifier(labels, id2label, label2id),
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)
    predictions = trainer.predict(test_dataset)
    print(predictions)
    trainer.save_model(output_model_name)
    if push_to_hub:
        trainer.push_to_hub()

    return trainer

if __name__ == '__main__':
    typer.run(main)


