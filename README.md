---
dataset_info:
  features:
  - name: id
    dtype: string
  - name: category
    dtype: string
  - name: source_language
    dtype: string
  - name: target_language
    dtype: string
  - name: input_text
    dtype: string
  - name: expected_output
    dtype: string
  - name: model_output
    dtype: string
  - name: error_type
    dtype: string
  - name: explanation
    dtype: string
  - name: is_error
    dtype: bool
  - name: error_note
    dtype: string
  - name: model_name
    dtype: string
  splits:
  - name: train
    num_bytes: 9364
    num_examples: 12
  download_size: 15096
  dataset_size: 9364
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
