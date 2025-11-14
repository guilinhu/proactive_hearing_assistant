# Proactive Hearing Assistants that Isolate Egocentric Conversations

## More Information

For more information, please refer to our website: [https://proactivehearing.cs.washington.edu/](https://proactivehearing.cs.washington.edu/).

## Abstract

We introduce proactive hearing assistants that automatically identify and separate the wearer’s conversation partners, without requiring explicit prompts. Our system operates on egocentric binaural audio and uses the wearer’s self-speech as an anchor, leveraging turn-taking behavior and dialogue dynamics to infer conversational partners and suppress others. To enable real-time, on-device operation, we propose a dual-model architecture: a lightweight streaming model runs every 12.5 ms for low-latency extraction of the conversation partners, while a slower model runs less frequently to capture longer-range conversational dynamics. Results on real-world 2- and 3-speaker conversation test sets, collected with binaural egocentric hardware from 11 participants totaling 6.8 hours, show generalization in identifying and isolating conversational partners in multi-conversation settings. Our work marks a step toward hearing assistants that adapt proactively to conversational dynamics and engagement.


## Training and Evaluation

### 1. Installing Requirements

Before training or evaluating the model, please create an environment and install all dependencies:

```
pip install -r requirements.txt
```

### 2. Model Training

To train the model, run:

```
python src/train_joint.py --config <path to config> --run_dir <path to model checkpoint>
```

### 3. Model Evaluation

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
