from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf


def main(text):
    # TO-DO: parametrise local model
    # TO-DO: extract method to run prediction on pdf and apply it here
    tokenizer = AutoTokenizer.from_pretrained("./model_3_epochs_classificator_tf", local_files_only=True, from_pt=False)
    model = TFAutoModelForSequenceClassification.from_pretrained("./model_3_epochs_classificator_tf", local_files_only=True)

    encoded_input = tokenizer(text, return_tensors="tf")
    logits = model(**encoded_input).logits
    probs = tf.nn.softmax(logits)
    print("All probabilities:", probs)

    print("Top 3 classes:")
    top_3_classes = tf.math.top_k(probs, k=3)
    for i in range(3):
        index = int(top_3_classes.indices[0][i])
        prob = float(top_3_classes.values[0][i])
        print(index)
        print("... Class:", model.config.id2label[index])
        print("... Probability:", prob)
        print("--------------------")


if __name__ == '__main__':
    text = input("Enter text: ")
    main(text)
