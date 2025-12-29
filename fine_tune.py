import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import time
import datetime
from pathlib import Path
import random
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from datasets import Dataset, Features, Value  # (not used but you had it)
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
    set_seed,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report, f1_score

# ----------------------------
# Config
# ----------------------------
epochs = 4
fine_tuning_runs = 1
model_path = "runs/microsoft__deberta-v3-large/best"   # folder containing a RoBERTa checkpoint
dataset = pd.read_json("CR_ECSS_dataset.json")
batch_num = 8
seed_val = 42

# ----------------------------
# Helpers
# ----------------------------
def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round(elapsed))))

def unpack_loss_logits(outputs):
    """
    Works for both HF ModelOutput and tuple outputs (e.g., under DataParallel).
    Returns (loss_scalar, logits_tensor).
    """
    if isinstance(outputs, (tuple, list)):
        loss, logits = outputs[:2]
    else:
        loss = outputs.loss
        logits = outputs.logits

    if isinstance(loss, torch.Tensor) and loss.dim() > 0:
        loss = loss.mean()  # <-- reduce per-GPU losses to a single scalar

    return loss, logits

# ----------------------------
# Label mapping
# ----------------------------
tag_vals = dataset["labels"].unique()
tag2idx = {tag: i for i, tag in enumerate(tag_vals)}
tag2name = {v: k for k, v in tag2idx.items()}
tag2name[-100] = "None"

def tokenize_and_align_labels(examples, labels, tokenizer):
    """
    Align word-level labels to wordpiece tokens.
    """
    tokenized = tokenizer(
        [ex for ex in examples],
        padding=True,
        truncation=False,
        is_split_into_words=True,
    )

    word_piece_labels = []
    label_all_tokens = True
    for i, label in enumerate(labels):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        word_piece_labels.append(label_ids)

    tokenized["labels"] = word_piece_labels
    return tokenized

# ----------------------------
# Reproducibility
# ----------------------------
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_val)
set_seed(seed_val)
torch.backends.cudnn.benchmark = True  # speed hint

# ----------------------------
# Device & (optional) multi-GPU
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print(f"Device: {device} | GPUs: {n_gpu}")

# ----------------------------
# Fine-tuning
# ----------------------------
for run_idx in range(fine_tuning_runs):
    print(f"\n==================== Fine Tuning Round {run_idx + 1} ====================")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, add_prefix_space=True)

    # build sentences/labels per sentence_id
    sentence_ids = dataset["sentence_id"].unique()
    sentences = [[w for w in dataset[dataset["sentence_id"] == sid]["words"].values] for sid in sentence_ids]
    labels_list = [[tag2idx[lbl] for lbl in dataset[dataset["sentence_id"] == sid]["labels"].values] for sid in sentence_ids]

    encoded = tokenize_and_align_labels(sentences, labels_list, tokenizer)
    input_ids = encoded["input_ids"]
    attention_masks = encoded["attention_mask"]
    labels = encoded["labels"]

    # sample inspect
    for j in range(min(5, len(input_ids))):
        toks = tokenizer.tokenize(tokenizer.decode(input_ids[j]))
        print(f"No.{j}, len:{len(input_ids[j])}")
        print("texts:", " ".join(toks))
        print(f"No.{j}, len:{len(labels[j])}")
        print("labels:", " ".join(tag2name.get(x, "UNK") for x in labels[j]))

    tr_inputs, val_inputs, tr_tags, val_tags, tr_masks, val_masks = train_test_split(
        input_ids, labels, attention_masks, random_state=4, test_size=0.213
    )

    tr_inputs = torch.as_tensor(tr_inputs)
    val_inputs = torch.as_tensor(val_inputs)
    tr_tags = torch.as_tensor(tr_tags)
    val_tags = torch.as_tensor(val_tags)
    tr_masks = torch.as_tensor(tr_masks)
    val_masks = torch.as_tensor(val_masks)

    # Dataloaders
    train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
    valid_data = TensorDataset(val_inputs, val_masks, val_tags)

    train_dataloader = DataLoader(
        train_data,
        sampler=RandomSampler(train_data),
        batch_size=batch_num,
        drop_last=False,
        num_workers=max(1, os.cpu_count() // 4),
        pin_memory=True,
    )
    valid_dataloader = DataLoader(
        valid_data,
        sampler=SequentialSampler(valid_data),
        batch_size=batch_num,
        drop_last=False,
        num_workers=max(1, os.cpu_count() // 4),
        pin_memory=True,
    )

    # Model
    model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=len(tag2idx))
    model.to(device)
    if n_gpu > 1:
        # Single-process, multi-GPU
        model = torch.nn.DataParallel(model)

    # Optim + sched
    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = []  # you can list substrings to exclude from weight decay, e.g., ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay_rate": 0.00,
                "correct_bias": False,  # harmless extra key for torch.optim.AdamW
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay_rate": 0.0,
                "correct_bias": False,
            },
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    training_stats = []
    train_loss_hist = []
    total_t0 = time.time()

    for epoch_i in range(epochs):
        print(f"\n======== Epoch {epoch_i + 1} / {epochs} ========")
        print("Training...")
        t0 = time.time()
        total_train_loss = 0.0

        model.train()
        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device, non_blocking=True)
            b_input_mask = batch[1].to(device, non_blocking=True)
            b_labels = batch[2].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss, logits = unpack_loss_logits(outputs)

            total_train_loss += float(loss.detach())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / max(1, len(train_dataloader))
        training_time = format_time(time.time() - t0)
        print(f"\n  Average training loss: {avg_train_loss:.4f}")
        print(f"  Training epoch took: {training_time}")
        train_loss_hist.append(avg_train_loss)

        # ----------------------------
        # Validation
        # ----------------------------
        print("\nRunning Validation...")
        t0 = time.time()
        model.eval()

        total_eval_loss = 0.0
        y_true, y_pred = [], []

        with torch.no_grad():
            for batch in valid_dataloader:
                b_input_ids = batch[0].to(device, non_blocking=True)
                b_input_mask = batch[1].to(device, non_blocking=True)
                b_labels = batch[2].to(device, non_blocking=True)

                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                vloss, vlogits = unpack_loss_logits(outputs)
                total_eval_loss += float(vloss.detach())

                # Get predicted labels
                if isinstance(outputs, (tuple, list)):
                    logits = outputs[1]
                else:
                    logits = outputs.logits

                logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.detach().cpu().numpy()
                input_mask = b_input_mask.detach().cpu().numpy()

                for i_m, m in enumerate(input_mask):
                    t1, t2 = [], []
                    for j, flag in enumerate(m):
                        if flag:
                            if tag2name.get(label_ids[i_m][j], "None") != "None":
                                t1.append(tag2name[label_ids[i_m][j]])
                                t2.append(tag2name[logits[i_m][j]])
                        else:
                            break
                    y_true.append(t1)
                    y_pred.append(t2)

        # Flatten & metrics
        y_true_words = [w for sent in y_true for w in sent]
        y_pred_words = [w for sent in y_pred for w in sent]

        # avoid including 'O' in per-class F1s; keep 'weighted' overall
        labels_for_scores = [lab for lab in set(y_true_words) if lab != "O"]
        prfs = precision_recall_fscore_support(y_true_words, y_pred_words, labels=labels_for_scores)[2:]
        f1_scores = {lab: prfs[0][i] for i, lab in enumerate(labels_for_scores)}
        examples = {lab: prfs[1][i] for i, lab in enumerate(labels_for_scores)}
        f1_scores["weighted"] = f1_score(
            y_true_words, y_pred_words, average="weighted", labels=labels_for_scores
        )
        examples["sum"] = np.sum([examples[k] for k in examples.keys()]) if examples else 0

        avg_val_loss = total_eval_loss / max(1, len(valid_dataloader))
        print(f"  F1_score (weighted): {f1_scores['weighted']:.4f}")
        print(f"  Validation Loss: {avg_val_loss:.4f}")
        validation_time = format_time(time.time() - t0)
        print(f"  Validation took: {validation_time}")

        training_stats.append(
            {
                "epoch": epoch_i + 1,
                "Training Loss": avg_train_loss,
                "Valid. Loss": avg_val_loss,
                "F1 score": f1_scores["weighted"],
                "examples_sum": examples["sum"],
                "Label_F1_scores": f1_scores,
                "examples": examples,
                "Training Time": training_time,
                "Validation Time": validation_time,
            }
        )
        with open("Train_results.json", "w+", encoding="utf-8") as file:
            pd.DataFrame(training_stats).to_json(file, orient="records", force_ascii=False)

    print("\nTraining complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

    # Final classification report
    y_true_words = [w for sent in y_true for w in sent]
    y_pred_words = [w for sent in y_pred for w in sent]
    report_dict = classification_report(
        y_true_words,
        y_pred_words,
        digits=3,
        labels=[label for label in set(y_true_words) if label != "O"],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    base = Path("report.csv")
    filename = base
    idx = 1
    while filename.exists():
        filename = base.with_name(f"{base.stem}_{idx}{base.suffix}")
        idx += 1
    report_df.to_csv(filename, encoding="utf-8")
    print(f"Report saved to: {filename}")
