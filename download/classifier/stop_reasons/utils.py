import re
import shutil
from pathlib import Path

import sklearn.metrics
import torch
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

def export_labels_to_model(model_name: str, model) -> None:
    """
    Reads from a model configuration to export the labels of the class target to a file in the model's assets folder.
    
    Args:
      model_name (str): The name of the model. This is used to create a directory for the model.
      model: The model to export.
    """
    labels = model.config.label2id
    labels = sorted(labels, key=labels.get)

    model_assets_path = f'models/{model_name}/saved_model/1/assets'

    with open(f'{model_assets_path}/labels.txt', 'w') as f:
        f.write('\n'.join(labels))

def save_model_from_hub(model_name: str) -> None:
    """
    We load the model and tokenizer from the HuggingFace hub, save them to the `models` directory, and then export
    the labels of the model to the directory that contains all the assets.
    
    Args:
      model_name (str): The name of the model you want to save.
    """

    model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.save_pretrained(f'models/{model_name}', from_tf=True, save_format='tf', saved_model=True)
    tokenizer.save_pretrained(f'models/{model_name}_tokenizer', from_tf=True, save_format='tf')
    export_labels_to_model(model_name, model)

    print(f"Model {model_name} saved.")

def copy_tokenizer_vocab_to_model(model_name):
    """
    We copy the tokenizer's vocabulary to the model's directory, so that we can use the model for
    predictions.

    Args:
        model_name (str): The name of the model you want to use.
    """

    tokenizer_vocab_path = f'models/{model_name}_tokenizer/vocab.txt'
    model_assets_path = f'models/{model_name}/saved_model/1/assets'

    shutil.copyfile(tokenizer_vocab_path, f'{model_assets_path}/vocab.txt')
    

def prepare_model_from_hub(model_name: str, model_dir:str) -> None:
    """
    If the model directory doesn't exist, download the model from the HuggingFace Hub, and copy the tokenizer
    vocab to the model directory so that the format can be digested by Spark NLP.
    
    Args:
      model_name (str): The name of the model you want to use.
      model_dir (str): The directory where the model will be saved.
    """

    model_path = f'{model_dir}/{model_name}'

    if not Path(model_path).is_dir():
        save_model_from_hub(model_name)
        copy_tokenizer_vocab_to_model(model_name)

def get_label_metadata(dataset):
  """
  It takes a dataset and returns a list of labels, a dictionary mapping label ids to labels, and a
  dictionary mapping labels to label ids
  
  Args:
    dataset: the dataset object
  """
  labels = [label for label in dataset['train'].features.keys() if label not in ['text', 'label_descriptions']]
  id2label = dict(enumerate(labels))
  label2id = {label:idx for idx, label in enumerate(labels)}
  return labels, id2label, label2id

def compute_metrics(eval_pred):
  """
  It takes in the predictions and labels from the model, and returns a dictionary of metrics.
  Logits are converted into probabilities following a sigmoid function; then, the predictions are
  converted into binary values by comparing the probabilities to a threshold.
  
  Args:
    eval_pred: a tuple of (predictions, labels)
  
  Returns:
    A dictionary with the accuracy, f1_micro and f1_macro
  """
  sigmoid_threshold = 0.3
  predictions, labels = eval_pred
  accuracy = accuracy_thresh(predictions, labels, sigmoid_threshold)
  f1_micro = sklearn.metrics.f1_score(labels, (predictions > sigmoid_threshold), average="micro")
  f1_macro = sklearn.metrics.f1_score(labels, (predictions > sigmoid_threshold), average="macro")
  confusion_matrix = sklearn.metrics.confusion_matrix(labels.flatten(), (predictions > 0.5).flatten().astype(int))
  return {
      "accuracy_thresh": accuracy,
      "f1_micro": f1_micro,
      "f1_macro": f1_macro,
      "confusion_matrix": confusion_matrix
  }

def accuracy_thresh(y_pred, y_true, thresh): 
    """
    It takes in a predicted probability and a true label, and returns the accuracy of the prediction
    
    Args:
      y_pred: the predicted values
      y_true: the ground truth labels
      thresh: the threshold for the prediction to be considered a positive prediction.
    
    Returns:
      The mean of the accuracy of the predictions.
    """
    y_pred = torch.from_numpy(y_pred).sigmoid()
    y_true = torch.from_numpy(y_true)
    return ((y_pred>thresh)==y_true.bool()).float().mean().item()


def prepare_splits_for_training(dataset, subset_data):
  """Splits and shuffles the dataset into train and test splits.

  Args:
      dataset (DatasetDict): The dataset to split. 
      subset_data (bool, optional): Flag to use a subset of the data.

  Returns:
      Tuple[Dataset]: One dataset object per train, test split.
  """
  fraction = 0.05 if subset_data else 1
  splits = [dataset["train"], dataset["test"]]

  return [
    split.shuffle(seed=42).select(range(int(len(split) * fraction)))
    for split in splits
  ]

def convert_to_tf_dataset(dataset, data_collator, shuffle_flag, batch_size):
  """
  We convert the dataset to a tf.data.Dataset object, which is a TensorFlow object that can be used
  to train a model
  
  Args:
    dataset: The dataset to convert to a tf.data.Dataset.
    data_collator: This is a function that takes in a list of tensors and returns a single tensor.
    shuffle_flag: Whether to shuffle the dataset or not.
    batch_size: The number of samples per batch.
  
  Returns:
    A tf.data.Dataset object
  """
  return (
      dataset.to_tf_dataset(
          columns=["attention_mask", "input_ids", "token_type_ids"],
          label_cols=["labels"],
          shuffle=shuffle_flag,
          collate_fn=data_collator,
          batch_size=batch_size
      )
  )

def preprocess_text(text: str):
    """Cleans and removes special characters from the text."""

    replacements = [
        (r"what's", "what is "),
        (r"won't", "will not "),
        (r"\'s", " "),
        (r"\'ve", " have "),
        (r"can't", "can not "),
        (r"n't", " not "),
        (r"i'm", "i am "),
        (r"\'re", " are "),
        (r"\'d", " would "),
        (r"\'ll", " will "),
        (r"\'scuse", " excuse "),
        (r"\'\n", " "),
        (r"-", " "),
        (r"\'\xa0", " "),
        (r"(@.*?)[\s]", " "),
        (r"&amp;", "&"),
    ]
    
    text = text.lower()
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)

    text = re.sub(r"\s+", " ", text).strip()
    return text

