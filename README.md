# Proactive Hearing Assistants that Isolate Egocentric Conversations

## More Information

For more information, please refer to our website: [https://proactivehearing.cs.washington.edu/](https://proactivehearing.cs.washington.edu/).


## Training and Evaluation

### 1. Installing Requirements

Before training or evaluating the model, please create an environment and install all dependencies:

```
pip install -r requirements.txt
```

### 2. Dataset (Required Before Training)

The synthetic libri conversation dataset for this project is provided through the publicly available **LibriConversation** dataset:

Dataset link: https://huggingface.co/datasets/guilinhu/libri_conversation

The dataset files are distributed as `.tar` archives. After downloading the dataset (for example `train.tar.gz` and `val.tar.gz`), extract them using:

```
tar -xvf train.tar.gz -C /path/to/output_directory/
tar -xvf val.tar.gz -C /path/to/output_directory/
```

This will produce directory structures containing the audio mixtures, target signals, and metadata used for model training. If you create your own dataset, ensure it follows the same format.

Once extracted, provide the directory paths in your model config file:

```
"train_data_args": {
  "input_dir": ["/absolute/path/to/train_directory"]
},

"val_data_args": {
  "input_dir": ["/absolute/path/to/validation_directory"]
}
```

### 3. Model Training

To train the model, run:

```
python src/train_joint.py --config <path_to_config> --run_dir <path_to_model_checkpoint>
```

To resume training, make sure that <path_to_model_checkpoint> points to the same directory used previously, and rerun the command above.


### 4. Model Evaluation

To evaluate the model, run:

```
python eval.py <path to testing dataset> <path to model checkpoint> --use_cuda --save
```


## Citation

If you use our work, please cite:

```
@inproceedings{hu2025proactive,
  title={Proactive Hearing Assistants that Isolate Egocentric Conversations},
  author={Hu, Guilin and Itani, Malek and Chen, Tuochao and Gollakota, Shyamnath},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages={25377--25394},
  year={2025}
}
```
