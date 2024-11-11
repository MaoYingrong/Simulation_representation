from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

if __name__ == '__main__':

    model = TFAutoModelForSequenceClassification.from_pretrained('stop_reasons_classificator_multi_label', local_files_only=True, from_pt=False)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    repo_url = 'https://huggingface.co/opentargets/stop_reasons_classificator_multi_label'
    commit_message = 'feat: add multi-label model - all data 7 epochs'
    model.push_to_hub(repo_path_or_name="stop_reasons_classificator_multi_label", repo_url=repo_url, organization="opentargets", commit_message=commit_message)
    tokenizer.push_to_hub(repo_path_or_name="stop_reasons_classificator_multi_label", repo_url=repo_url, organization="opentargets", commit_message=commit_message)
