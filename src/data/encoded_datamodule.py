# DataModule for loading text datasets, encoding them into language-model embeddings,
# caching the encoded representations, and serving embedding–label batches
# for models training.
import os
import sys
import warnings
import random

import pytorch_lightning as pl
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset

from . import DataModule

sys.path.append("/notebooks/Gan-Based-Defense-Mechanism/src")
from models import AutoLanguageModel


class PEDataset(Dataset):
    def __init__(self, clean_embs, labels):
        self.clean_embs = clean_embs
        self.labels = labels

    def __len__(self):
        return len(self.clean_embs)

    def __getitem__(self, idx):
        return {"clean_emb": self.clean_embs[idx], "rating": self.labels[idx]}


class EncodedDataModule(DataModule):
    """
    Note to the future me:
    If you want to load the data from cache, you can just provide the 'data' and 'lm' arguments.
    If you want to create new encoded data, you must provide specifically which pretrained LM to be used.
    """

    def __init__(
        self,
        data="hard",
        data_root="/notebooks/Gan-Based-Defense-Mechanism/data/encoded-data",
        lm="bert",
        lm_path="path to fine tuned model",
        save_name: str = None,
        is_encoded: bool = True,
        use_cache: bool = True,
        setup_batch_size: int = 128,
        train_batch_size: int = 128,
        val_batch_size: int = 128,
        test_batch_size: int = 128,
        num_workers: int = 4,
        setup_device: str = "cuda",
        layer: int = -1,
        max_length: int = 128,
        num_samples_for_pe: int = 1000000,
        seed=42,
        **kwargs,
    ):
        super().__init__()
        if data.replace("_", "").replace("-", "") not in lm_path.replace(
            "_", ""
        ).replace("-", ""):
            warnings.warn(
                f"There maybe a mismatch between dataset ({data}) and pretrained language model ({lm_path})."
            )
        self.save_hyperparameters()
        pl.seed_everything(seed)
        
        # Initialize text data storage
        self._text_samples = None
        self._text_embeddings = None

    def get_encode_fn(self, lm):
        def encode_fn(examples):
            with torch.no_grad():
                text_pairs = None
                texts = examples["cleaned_text"]
                
                if self.hparams.layer == 1:
                    print("inside hpaars")
                    sentence_embs = lm.text2emb(texts, text_pairs)
                    examples["clean_emb"] = sentence_embs.cpu().detach().numpy()
                else:
                    inputs = lm.tokenizer(
                        texts,
                        padding="max_length",
                        truncation=True,
                        max_length=self.hparams.max_length,
                        return_tensors="pt",
                    )
                    print("3")
   
                    inputs = {
                        k: v.to(self.hparams.setup_device) for k, v in inputs.items()
                    }
                    outputs = lm(**inputs, output_hidden_states=True)
                    ss = outputs.hidden_states[self.hparams.layer].cpu().detach().numpy()

                    print(f"Numpy array shape: {ss.shape}")  # Debugging# Assuming you want the last layer's output
#                     clean_emb = outputs.hidden_states[self.hparams.layer]
        
#                     # Example post-processing: Mean pooling across the token embeddings
#                     # Adjust this according to your actual needs
#                     clean_emb = clean_emb.mean(dim=1)  # Mean pooling across tokens

#                     examples["clean_emb"] = clean_emb.cpu().detach().numpy()
                    examples["clean_emb"] = (
                        outputs.hidden_states[self.hparams.layer].cpu().detach().numpy()
                    )
                return examples

        return encode_fn

    def encode(self, dataset, lm):
        # Encode the data using a LM
        encode_fn = self.get_encode_fn(lm)
        encoded_dataset = dataset.map(
            encode_fn, batched=True, batch_size=self.hparams.setup_batch_size
        )
        return encoded_dataset

    def setup(self):
        if self.hparams.is_encoded:
            loaded = False
            if self.hparams.use_cache:
                try:
                    print(f"Load data from cache ({self.cache_path})")
                    datasets = load_from_disk(self.cache_path)
                    loaded = True
                    # Extract text samples from cached data BEFORE setting torch format
                    self._extract_text_samples(datasets)
                except:
                    print("Cache not found.")

            if not loaded:
                print("Setup data from scratch.")
                lm = AutoLanguageModel.get_class_name(
                    self.hparams.lm
                ).from_pretrained(self.hparams.lm_path)
                lm.eval()
                lm.to(self.hparams.setup_device)
                
                print("csvvvv")
                # Load custom dataset from CSV
                datasets = load_dataset('csv', data_files={'train': '/notebooks/Gan-Based-Defense-Mechanism/data/hard-1/final_data.csv'})
                print("after_dataset")
                # To reduce computational cost, we only work on a subset of the original yelp dataset
                if self.hparams.data == "yelp_polarity":
                    random.seed(self.hparams.seed)
                    l = len(datasets["train"])
                    datasets["train"] = datasets["train"].select(random.sample(range(l), 25000))

                # Process PE case (subsample)
                if self.hparams.layer != -1:
                    random.seed(self.hparams.seed)
                    for k, v in datasets.items():
                        num_chosen_ids = min(len(v), self.hparams.num_samples_for_pe // self.hparams.max_length)
                        chosen_ids = random.sample(range(len(v)), num_chosen_ids)
                        datasets[k] = v.select(chosen_ids)
                print("1")
                datasets = self.encode(datasets, lm)
                print("2")

                # Process PE case. For each split:
                # 1. Merge all token embedding into a single tensor, then split into separate vectors
                # 2. Construct new Dataset
                if self.hparams.layer != -1:
                    for k, v in datasets.items():
                        breakpoint()
                        embs = torch.cat([torch.vstack(e) for e in v["clean_emb"]])
                        labels = (
                            v["rating"]
                            .unsqueeze(-1)
                            .expand(v["rating"].shape[0], self.hparams.max_length)
                            .reshape(-1)
                        )
                print(self.cache_path)

                datasets.save_to_disk(self.cache_path)

            # Extract text samples BEFORE setting torch format (which removes other columns)
            self._extract_text_samples(datasets)
        else:
            # Load custom dataset directly if not encoded
            datasets = load_dataset('csv', data_files={'train': '/notebooks/Gan-Based-Defense-Mechanism/data/hard-1/final_data.csv'})
            # Extract text samples for non-encoded data too
            self._extract_text_samples(datasets)

        # Set torch format for both cached and non-cached data AFTER text extraction
        if self.hparams.is_encoded:
            datasets.set_format(type="torch", columns=["clean_emb", "rating"])

        if self.hparams.layer != -1:
            tmp = {}
            for k, v in datasets.items():
                embs = torch.cat([torch.vstack(e) for e in v["clean_emb"]])
                labels = (
                    v["rating"]
                    .unsqueeze(-1)
                    .expand(v["rating"].shape[0], self.hparams.max_length)
                    .reshape(-1)
                )
                embs = embs[torch.randperm(len(labels))][:self.hparams.num_samples_for_pe].clone()
                labels = labels[torch.randperm(len(labels))][:self.hparams.num_samples_for_pe].clone()

                tmp[k] = PEDataset(embs, labels)
            datasets = tmp
        else:
            train_dataset, val_dataset = (
                datasets["train"]
                .train_test_split(0.2, seed=self.hparams.seed)
                .values()
                )
            datasets["train"] = train_dataset
            datasets["validation"] = val_dataset

        self.train_dataset = datasets["train"]
        self.val_dataset = datasets["validation"]
        # self.test_dataset = datasets["test"]
    
    def _extract_text_samples(self, datasets):
        """Extract text samples and embeddings for InfoGAN logging"""
        print(f"Available columns in train dataset: {datasets['train'].column_names}")
        
        # Try different text column names
        text_column = None
        for col_name in ["cleaned_text", "text", "sentence"]:
            if col_name in datasets["train"].column_names:
                text_column = col_name
                break
        
        # Extract text samples
        if text_column:
            try:
                print(f"Extracting text samples using column: {text_column}")
                # Sample some texts and embeddings for logging
                sample_size = min(1000, len(datasets["train"]))
                sample_indices = list(range(0, len(datasets["train"]), max(1, len(datasets["train"]) // sample_size)))[:sample_size]
                
                self._text_samples = [datasets["train"][i][text_column] for i in sample_indices]
                # Convert embeddings from lists to tensors before stacking
                embeddings = [datasets["train"][i]["clean_emb"] for i in sample_indices]
                self._text_embeddings = torch.stack([torch.tensor(emb) for emb in embeddings])
                print(f"Successfully extracted {len(self._text_samples)} text samples")
                print(f"Sample text: {self._text_samples[0][:100] if len(self._text_samples) > 0 else 'No samples'}")
                print(f"Text embeddings shape: {self._text_embeddings.shape}")
            except Exception as e:
                print(f"Could not extract text samples: {e}")
                import traceback
                traceback.print_exc()
                self._text_samples = None
                self._text_embeddings = None
        else:
            print("No text column found in dataset for InfoGAN logging")
            print(f"Available columns were: {datasets['train'].column_names}")
            self._text_samples = None
            self._text_embeddings = None
    
    def get_text_samples(self):
        """Return text samples and embeddings for InfoGAN logging"""
        print(f"DataModule get_text_samples: texts={self._text_samples is not None}, embeddings={self._text_embeddings is not None}")
        if self._text_samples:
            print(f"Number of text samples: {len(self._text_samples)}")
        if self._text_embeddings is not None:
            print(f"Embeddings shape: {self._text_embeddings.shape}")
        return {
            'texts': self._text_samples,
            'embeddings': self._text_embeddings
        }

    @property
    def cache_path(self):
        if self.hparams.save_name is None:
            if self.hparams.layer == -1:
                filename = (
                    f"{self.hparams.lm}_encoded_{self.hparams.data.replace('-', '_')}"
                )
                print("file_name----------->", filename)
                
            else:
                filename = f"{self.hparams.lm}_encoded_{self.hparams.data.replace('-', '_')}_layer{self.hparams.layer}"
            path = os.path.join(self.hparams.data_root, filename)
            print("path", path)
            return path
        else:
            print("hparams--->",self.hparams.save_name )
            return os.path.join(self.hparams.data_root, self.hparams.save_name)
