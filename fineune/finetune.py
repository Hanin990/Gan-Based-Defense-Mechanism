import argparse
import os
import random
import pandas as pd
import torch
import wandb
from datasets import load_dataset, load_from_disk
from evaluate import load as load_metric
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, AdamW, get_scheduler

from tqdm import tqdm

num_labels =  {"custom_dataset_arabert_hard_128": 2}


def get_tokenize_function(tokenizer, max_length, dataset):
    tokenize_function = None
    def tokenize_function(examples):
        input_dict = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
        )
        examples.update(input_dict)
        return examples
    return tokenize_function


def prepare_data(args):
    raw_datasets = load_dataset(args.dataset)

    # To reduce computational cost, we only work on a subset of the original yelp dataset
    if args.dataset == "yelp_polarity":
        random.seed(42)
        l = len(raw_datasets["train"])
        raw_datasets["train"] = raw_datasets["train"].select(random.sample(range(l), 25000))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        model_max_length=args.model_max_length,
    )
    tokenize_function = get_tokenize_function(tokenizer, args.model_max_length, args.dataset)
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets.set_format("torch")

    # Data splitting 
    train_dataset, eval_dataset = tokenized_datasets["train"].train_test_split(0.1, seed=42).values()
    test_dataset = tokenized_datasets["test"]

    # Make dataloaders
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=data_collator,
        pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        num_workers=2,
        collate_fn=data_collator,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        num_workers=2,
        collate_fn=data_collator,
        pin_memory=True
    )

    return tokenizer, train_dataloader, eval_dataloader, test_dataloader


def rename(k):
    if k == "label":
        return "labels"
    else:
        return k
    
    
def prepare_data(args):
    import pandas as pd
    from transformers import AutoTokenizer, DataCollatorWithPadding
    
    # Load the custom dataset
    data_df = pd.read_csv(args.data_path)
    
    # Split the data into training and validation sets
    train_df = data_df.sample(frac=0.7, random_state=42)
    val_df = data_df.drop(train_df.index)
    
    # Tokenize the datasets using the specified tokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_encodings = tokenizer(train_df['cleaned_text'].astype(str).values.tolist(), truncation=True, padding='max_length', max_length=args.max_length, return_tensors='pt')
    val_encodings = tokenizer(val_df['cleaned_text'].astype(str).values.tolist(), truncation=True, padding='max_length', max_length=args.max_length, return_tensors='pt')
    
    # Ensure encodings are PyTorch tensors and squeeze if necessary
    train_input_ids = torch.tensor(train_encodings.input_ids).squeeze() if isinstance(train_encodings.input_ids, list) else train_encodings.input_ids.squeeze()
    train_attention_mask = torch.tensor(train_encodings.attention_mask).squeeze() if isinstance(train_encodings.attention_mask, list) else train_encodings.attention_mask.squeeze()
    
    val_input_ids = torch.tensor(val_encodings.input_ids).squeeze() if isinstance(val_encodings.input_ids, list) else val_encodings.input_ids.squeeze()
    val_attention_mask = torch.tensor(val_encodings.attention_mask).squeeze() if isinstance(val_encodings.attention_mask, list) else val_encodings.attention_mask.squeeze()

    # Convert the encodings to PyTorch datasets
    train_dataset = torch.utils.data.TensorDataset(train_input_ids, train_attention_mask, torch.tensor(train_df['rating'].tolist()))
    val_dataset = torch.utils.data.TensorDataset(val_input_ids, val_attention_mask, torch.tensor(val_df['rating'].tolist()))
    
    # # Define the data collator to handle padding
    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Define dataloaders with the data collator
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False)
    
    return tokenizer, train_dataloader, val_dataloader

def load_custom_model(args):
    from transformers import AutoModelForSequenceClassification
    # Load the pretrained model
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_labels)
    # model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_labels, ignore_mismatched_sizes=True)
    return model

def main(args):
    tokenizer, train_dataloader, eval_dataloader = prepare_data(args)
    model = load_custom_model(args)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    if args.ckpt_path is not None:
        model.from_pretrained(args.ckpt_path)
        tokenizer.from_pretrained(args.ckpt_path)

    if args.mode == "train":
        optimizer = AdamW(model.parameters(), lr=args.lr)
        num_training_steps = args.num_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        progress_bar = tqdm(range(num_training_steps))

        model.train()
        global_step = 0
        for epoch in range(args.num_epochs):
            wandb.log({"epoch": epoch})
            for batch in train_dataloader:
                # Unpack the batch from the DataLoader
                input_ids, attention_mask, labels = batch
                # Create a dictionary with the appropriate tensor names and transfer to device
                batch = {'input_ids': input_ids.to(device), 'attention_mask': attention_mask.to(device), 'labels': labels.to(device)}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

                if global_step % 100 == 0:
                    with torch.no_grad():
                        logits = outputs.logits
                        preds = torch.argmax(logits, dim=-1)
                        labels = batch["labels"]

                        loss = loss.item()
                        acc = (preds == labels).float().mean().item()

                        wandb.log({"train/loss": loss})
                        wandb.log({"train/acc": acc})
                        progress_bar.set_postfix({"loss": loss, "acc": acc})

                global_step += 1

            # Evaluate on val set
            metric = load_metric("accuracy")
            model.eval()
            for batch in eval_dataloader:
                # Unpack the batch from the DataLoader
                input_ids, attention_mask, labels = batch
                # Create a dictionary with the appropriate tensor names and transfer to device
                batch = {'input_ids': input_ids.to(device), 'attention_mask': attention_mask.to(device), 'labels': labels.to(device)}
                with torch.no_grad():
                    outputs = model(**batch)

                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                metric.add_batch(predictions=predictions, references=batch["labels"])
            acc = metric.compute()["accuracy"]
            wandb.log({"validate/acc": acc})

            # Save checkpoint
            model.save_pretrained(args.output_dir + f"/epoch_{epoch}")
            tokenizer.save_pretrained(args.output_dir + f"/epoch_{epoch}")

    # Evaluate on test set
    metric = load_metric("accuracy")
    model.eval()
    for batch in eval_dataloader:
        # Unpack the batch from the DataLoader
        input_ids, attention_mask, labels = batch
        # Create a dictionary with the appropriate tensor names and transfer to device
        batch = {'input_ids': input_ids.to(device), 'attention_mask': attention_mask.to(device), 'labels': labels.to(device)}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    acc = metric.compute()["accuracy"]
    if args.mode == "train":
        wandb.log({"test/acc": acc})
    print(f"Accuracy on test set: {acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="custom_dataset_arabert_hard_128")
    parser.add_argument("--data_path", type=str, default="/notebooks/Gan-Based-Defense-Mechanism/data/hard-1/final_data.csv") #from the datasets that exist in Data file
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--model_name", type=str, default="aubmindlab/bert-base-arabertv2")
    parser.add_argument("--lr", type=float, default=3e-05)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--project", type=str)
    parser.add_argument("--entity", type=str)
    parser.add_argument("--tags", type=str, nargs="+", default=["finetune"])
    args = parser.parse_args()
    args.num_labels = num_labels[args.dataset]
    print(args)

    if args.mode == "train":
        run = wandb.init(
            project=args.project,
            entity=args.entity,
            config=args,
        )
        args.output_dir = f"{args.dataset}/{run.id}"
        os.makedirs(args.output_dir, exist_ok=True)

    main(args)
