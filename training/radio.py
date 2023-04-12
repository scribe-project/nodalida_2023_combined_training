from datasets import load_dataset, load_metric
import json
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import torch
import pickle
from pathlib import Path

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import wandb
import numpy as np

import sys
# Usage:
# python3 rundkast_trainer_kblab_bm.py [data_dir]
#
# Per Erik runs:
# python3 rundkast_trainer_kblab_bm.py /media/pers/elements/
# Giampiero runs:
# python3 rundkast_trainer_kblab_bm.py /NOBACKUP/giampi/nodalida2023/
if len(sys.argv)>1:
    data_dir = Path(sys.argv[1])
else:
    data_dir = Path('.')


print("Train wav2vec model on Rundkast using the KBLab pretrained model")

print("Connect to wanb")

wandb.init(project="rundkast_nb_model_voxrex")

print("Load datasets")
cache_dir = (data_dir / "huggingface_cache/")
dataset_train = load_dataset(
    "scribe-project/rundkast_nb",
    split="train",
    use_auth_token=True,
    cache_dir=cache_dir,
)
dataset_valid = load_dataset(
    "scribe-project/rundkast_nb",
    split="validation",
    use_auth_token=True,
    cache_dir=cache_dir,
)
dataset_test = load_dataset(
    "scribe-project/rundkast_nb",
    split="test",
    use_auth_token=True,
    cache_dir=cache_dir,
)


print("Remove unneccessary columns")
dataset_train = dataset_train.remove_columns(
    ["speaker_id", "utterance_id", "language", "raw_text", "duration", "start", "end"]
)
dataset_train = dataset_train.rename_column("utterance_audio_file", "audio")
dataset_train = dataset_train.rename_column("standardized_text", "sentence")

dataset_valid = dataset_valid.remove_columns(
    ["speaker_id", "utterance_id", "language", "raw_text", "duration", "start", "end"]
)
dataset_valid = dataset_valid.rename_column("utterance_audio_file", "audio")
dataset_valid = dataset_valid.rename_column("standardized_text", "sentence")

dataset_test = dataset_test.remove_columns(
    ["speaker_id", "utterance_id", "language", "raw_text", "duration", "start", "end"]
)
dataset_test = dataset_test.rename_column("utterance_audio_file", "audio")
dataset_test = dataset_test.rename_column("standardized_text", "sentence")

#print("First record:")
#print(dataset_train[0])

print("Make vocabulary")


def extract_all_chars(batch):
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


vocab_train = dataset_train.map(
    extract_all_chars,
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=dataset_train.column_names,
)
vocab_valid = dataset_valid.map(
    extract_all_chars,
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=dataset_valid.column_names,
)
vocab_test = dataset_test.map(
    extract_all_chars,
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=dataset_test.column_names,
)

vocab_list = list(
    set(vocab_train["vocab"][0])
    | set(vocab_valid["vocab"][0])
    | set(vocab_test["vocab"][0])
)
vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}

print("Vocabulary:")
print(vocab_dict)

# Adding CTCs blank token and unknown token (not sure why the unknown...)
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

# Save vocabulary in json for later use
with open("vocab.json", "w") as vocab_file:
    json.dump(vocab_dict, vocab_file)

from transformers import Wav2Vec2CTCTokenizer

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
    "./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token=" "
)


# In order for this to work you have to run huggingface-cli login in a terminal or huggingface_hub.login()
# Apparently, you need to create a directory for temporary files even if you use use_temp_dir=False, check why
Path("wav2vec2-large-voxrex-300m-rundkast_nb_2").mkdir(parents=True, exist_ok=True)
print("Push to hub")
tokenizer.push_to_hub(
    #"wav2vec2-large-voxrex-300m-rundkast_nb_2",
    repo_id="scribe-project/wav2vec2-large-voxrex-300m-rundkast_nb_2",
    use_temp_dir=False,
)

print("Get feature extractor and processor")
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True,
)

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
processor.push_to_hub(
    #"wav2vec2-large-voxrex-300m-rundkast_nb_2",
    repo_id="scribe-project/wav2vec2-large-voxrex-300m-rundkast_nb_2",
    use_temp_dir=False,
)

print("Prepare dataset for training")

# batch here is actually a single example.
def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=16000).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch


# remove columns from datasets. Do this only first time, next time read from pickle
#picklepath = (data_dir / "processed_datasets/")
#picklepath.mkdir(parents=True, exist_ok=True)
#if((picklepath / "rundkast_train.pickle").exists() and (picklepath / "rundkast_valid.pickle").exists()):
#    with open(picklepath / "rundkast_train.pickle", "rb") as picklefile:
#        dataset_train = pickle.load(picklefile)
#    with open(picklepath / "rundkast_valid.pickle", "rb") as picklefile:
#        dataset_valid = pickle.load(picklefile)
#else:
print('preparing training data')
dataset_train = dataset_train.map(
    prepare_dataset, remove_columns=dataset_train.column_names
)
print('preparing validation data')
dataset_valid = dataset_valid.map(
    prepare_dataset, remove_columns=dataset_valid.column_names
)
#    with open(picklepath / "rundkast_train.pickle", "wb") as picklefile:
#        pickle.dump(dataset_train, picklefile)
#    with open(picklepath / "rundkast_valid.pickle", "wb") as picklefile:
#        pickle.dump(dataset_valid, picklefile)

repo_name = "wav2vec2-large-voxrex-300m-rundkast_nb_2"


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features, padding=self.padding, return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features, padding=self.padding, return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels

        return batch


data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

wer_metric = load_metric("wer")


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


print("Load model")

# Settings similar to
# NbAiLab/nb-wav2vec2-300m-bokmaal

model = Wav2Vec2ForCTC.from_pretrained(
    "KBLab/wav2vec2-large-voxrex",
    attention_dropout=0.094,
    activation_dropout=0.055,
    hidden_dropout=0.047,
    feat_proj_dropout=0.04,  # was set to 0.004 before, while nbailab had 0.04
    mask_time_prob=0.082,
    mask_time_length=10,
    mask_feature_prob=0.25,
    mask_feature_length=64,
    layerdrop=0.041,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
    cache_dir=cache_dir,
)

model.freeze_feature_encoder()

training_args = TrainingArguments(
    output_dir = (data_dir / "wav2vec_models" / repo_name),
    group_by_length=True,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    length_column_name="input_length",
    evaluation_strategy="steps",
    num_train_epochs=30,
    gradient_checkpointing=True,
    fp16=True,
    save_steps=500,
    eval_steps=500,
    logging_steps=100,
    learning_rate=1e-4,
    warmup_steps=2000,
    report_to="wandb",
    save_total_limit=3,
    seed=42,
    push_to_hub=True,
    hub_model_id=f"scribe-project/{repo_name}",
    load_best_model_at_end = True,
    metric_for_best_model = 'wer'
)

print("Initialize trainer")
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=dataset_train,
    eval_dataset=dataset_valid,
    tokenizer=processor.feature_extractor,
    #callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    # this was removed for the moment because it caused the training to stop right after
    # the 2000 warmup steps. Need to find out the problem. Also, in this configuration
    # it is not sure the best models are saved: need to find out how to add this.
)

print("start training")
trainer.train()

print("Evaluate:")
trainer.evaluate()

