import torch
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from args import ProgramArgs, string_to_bool
args = ProgramArgs.parse(True)

args.build_environment()

import logging
import traceback
from tqdm import tqdm
from torch.nn.modules.loss import CrossEntropyLoss
import numpy as np
from data.reader import ClassificationReader
import wandb
from utils.config import MODEL_CLASSES, DATASET_LABEL_NUM
from utils.metrics import ClassificationMetric, SimplifidResult
from utils.my_utils import convert_batch_to_bert_input_dict
from utils.public import auto_create, check_and_create_path
from utils.textattack_utils import build_english_attacker, CustomTextAttackDataset

from textattack.loggers import AttackLogManager
from textattack.models.wrappers import HuggingFaceModelWrapper

import sys
sys.path.append("/notebooks/Gan-Based-Defense-Mechanism/src")
from models import AutoLanguageModel, AutoGenerativeModel

from time import time


class AttackBenchmarkTask(object):
    """
    Simplified Attack Benchmark Task focused on InfoGAN-based defense and attack evaluation.

    Supported modes:
    - attack: Run adversarial attacks against the model
    - evaluate: Evaluate model performance
    """

    def __init__(self, args: ProgramArgs):
        self.methods = {
            'attack': self.attack,
            'evaluate': self.evaluate,
        }
        assert args.mode in self.methods, f'Mode {args.mode} not found. Supported modes: {list(self.methods.keys())}'

        # Initialize tokenizer and dataset reader
        self.tokenizer = self._build_tokenizer(args)
        self.dataset_reader = ClassificationReader(model_type=args.model_type, max_seq_len=args.max_seq_len)

        # Load datasets
        self.train_raw, self.eval_raw, self.test_raw = auto_create(
            f'{args.model_type}/{args.dataset_name}_raw_datasets',
            lambda: self._build_raw_dataset(args),
            True,
            path=args.cache_path
        )

        self.train_dataset, self.eval_dataset, self.test_dataset = auto_create(
            f'{args.model_type}/{args.dataset_name}_tokenized_datasets',
            lambda: self._build_tokenized_dataset(args),
            True,
            path=args.cache_path
        )

        # Build data loaders
        self.data_loader, self.eval_data_loader, self.test_data_loader = self._build_dataloader(args)

        # Build model with InfoGAN defense
        self.model = self._build_model(args)

        # Build loss function
        self.loss_function = self._build_criterion(args)

        logging.info(f"Initialized AttackBenchmarkTask with mode: {args.mode}")
        logging.info(f"Model type: {args.model_type}")
        logging.info(f"Training type: {args.training_type}")
        logging.info(f"Dataset: {args.dataset_name}")

    @torch.no_grad()
    def evaluate(self, args: ProgramArgs, is_training: bool = False) -> ClassificationMetric:
        """
        Evaluate model performance on test/eval dataset.

        Args:
            args: Program arguments
            is_training: If True, evaluate on eval set, else on test set

        Returns:
            ClassificationMetric with evaluation results
        """
        if is_training:
            logging.info('Evaluating on validation set')
            epoch_iterator = tqdm(self.eval_data_loader, desc="Evaluating")
        else:
            logging.info('Evaluating on test set')
            epoch_iterator = tqdm(self.test_data_loader, desc="Evaluating")

        self.model.eval()

        metric = ClassificationMetric(compare_key=args.compare_key)
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        start_time = time()

        for batch in epoch_iterator:
            batch = tuple(t.to(args.device) for t in batch)
            golds = batch[3]
            inputs = convert_batch_to_bert_input_dict(batch, args.model_type)
            logits = self.model.forward(**inputs)[0]
            losses = self.loss_function(
                logits.view(-1, DATASET_LABEL_NUM[args.dataset_name]),
                golds.view(-1)
            )

            # Calculate metrics
            total_loss += torch.sum(losses).item()
            total_correct += torch.sum(torch.argmax(logits, dim=1) == golds).item()
            total_samples += golds.numel()

            metric(losses, logits, golds)
            epoch_iterator.set_description(f'loss: {torch.mean(losses):.4f}')

        accuracy = total_correct / total_samples if total_samples > 0 else 0
        metric_dict = metric.get_metric(reset=False)

        stop_time = time()
        elapsed_time = stop_time - start_time

        logging.info(f"Evaluation Results:")
        logging.info(f"  Total Loss: {total_loss:.4f}")
        logging.info(f"  Accuracy: {accuracy:.4f}")
        logging.info(f"  F1 Score: {metric_dict['f1']:.4f}")
        logging.info(f"  Elapsed Time: {elapsed_time:.2f}s")

        if not is_training:
            wandb.log({
                'eval/loss': total_loss,
                'eval/accuracy': accuracy,
                'eval/f1': metric_dict['f1']
            })

        print(metric)
        print(f"Elapsed time: {elapsed_time:.2f}s")
        return metric

    def attack(self, args: ProgramArgs):
        """
        Run adversarial attacks against the model.

        Args:
            args: Program arguments containing attack configuration
        """
        logging.info("=" * 80)
        logging.info("Starting Attack Evaluation")
        logging.info("=" * 80)

        self.model.eval()
        attacker = self._build_attacker(args)

        # Select dataset (dev or test)
        if args.evaluation_data_type == 'dev':
            dataset = self.eval_raw
            logging.info("Using development set for attacks")
        else:
            dataset = self.test_raw
            logging.info("Using test set for attacks")

        # Setup attack logging
        attacker_log_path = os.path.join(
            f"{args.workspace}/log/{args.dataset_name}_{args.model_type}",
            args.build_logging_path()
        )
        attacker_log_manager = AttackLogManager()
        attacker_log_manager.add_output_file(
            os.path.join(
                attacker_log_path,
                f'{args.attack_method}_{args.neighbour_vocab_size}_{args.modify_ratio}_{args.get_file_creation_time}.txt'
            )
        )

        # Filter instances with more than 4 words
        test_instances = [x for x in dataset if len(x.text_a.split(' ')) > 4]
        logging.info(f"Total instances available: {len(test_instances)}")
        logging.info(f"Attack method: {args.attack_method}")
        logging.info(f"Attack numbers per iteration: {args.attack_numbers}")
        logging.info(f"Attack iterations: {args.attack_times}")

        # Run multiple attack iterations for average success rate
        for iteration in range(args.attack_times):
            logging.info(f"\n{'=' * 60}")
            logging.info(f"Attack Iteration: {iteration + 1}/{args.attack_times}")
            logging.info(f"{'=' * 60}")

            # Sample instances for attack
            if len(test_instances) > args.attack_numbers:
                choice_instances = np.random.choice(
                    test_instances,
                    size=(args.attack_numbers,),
                    replace=False
                )
                logging.info(f"Randomly sampled {len(choice_instances)} instances")
            else:
                choice_instances = test_instances
                logging.info(f"Using all {len(choice_instances)} available instances")

            # Apply index slicing if specified
            choice_instances = choice_instances[args.start_index:args.end_index]
            logging.info(f"Final attack set size (after slicing): {len(choice_instances)}")

            # Create TextAttack dataset
            attack_dataset = CustomTextAttackDataset.from_instances(
                args.dataset_name,
                choice_instances,
                self.dataset_reader.get_labels()
            )

            # Run attacks
            results_iterable = attacker.attack_dataset(attack_dataset)
            result_statistics = SimplifidResult()
            exceptions = []

            identifier = f"{args.attack_method}: [{args.start_index}:{args.end_index}]"
            description = tqdm(results_iterable, total=len(choice_instances), desc=identifier)

            for idx, result in enumerate(description):
                try:
                    attacker_log_manager.log_result(result)
                    result_statistics(result)
                    description.set_description(f"{identifier} | {result_statistics.__str__()}")
                    logging.info(f"{identifier} ---> {idx+1}/{len(choice_instances)} | {result_statistics.__str__()}")

                except Exception as e:
                    logging.warning(f"Exception during attack at index {idx}: {str(e)}")
                    logging.warning(traceback.format_exc())

                    try:
                        # Reduce batch size and retry
                        logging.info(f"Reducing batch size to 32 and retrying...")
                        original_batch_size = self.model.batch_size
                        self.model.batch_size = 32

                        attacker_log_manager.log_result(result)
                        result_statistics(result)
                        description.set_description(f"{identifier} | {result_statistics.__str__()}")
                        logging.info(f"{identifier} ---> {idx+1}/{len(choice_instances)} | {result_statistics.__str__()}")

                        self.model.batch_size = original_batch_size
                    except Exception as retry_error:
                        logging.error(f"{identifier} ---> {idx}: Failed even with reduced batch size: {str(retry_error)}")
                        exceptions.append(idx)
                        continue

        # Log final statistics
        logging.info(f"\n{'=' * 80}")
        logging.info(f"Attack Evaluation Complete")
        logging.info(f"{'=' * 80}")

        if exceptions:
            logging.info(f"{identifier} ---> Exceptions at indices: {exceptions}")
            print(f"{identifier} ---> Exceptions at indices: {exceptions}")

        attacker_log_manager.enable_stdout()
        attacker_log_manager.log_summary()

    def _build_model(self, args: ProgramArgs):
        """
        Build model with InfoGAN-based Text Manifold Defense.

        Args:
            args: Program arguments

        Returns:
            Model with InfoGAN reconstructor attached
        """
        logging.info(f"Building model with training_type: {args.training_type}")

        if args.training_type == 'tmd':
            # Load Language Model
            logging.info(f"Loading Language Model: {args.model_name_or_path}")
            model = AutoLanguageModel.get_class_name(args.model_type).from_pretrained(
                args.model_name_or_path,
                load_tokenizer=False
            )
            model.eval()
            model.to(args.device)

            # Load InfoGAN Generative Model
            logging.info(f"Loading Generative Model (InfoGAN): {args.gm}")
            logging.info(f"Checkpoint path: {args.gm_path}")

            gm = AutoGenerativeModel.get_class_name(args.gm).load_from_checkpoint(args.gm_path)
            gm.eval()
            gm.to(args.device)

            # Attach InfoGAN reconstructor to language model
            model.set_reconstructor(gm, **vars(args))
            logging.info("InfoGAN reconstructor successfully attached to language model")

            return model

        elif args.training_type == 'none':
            # Load model without defense
            logging.info(f"Loading Language Model without defense: {args.model_name_or_path}")
            model = AutoLanguageModel.get_class_name(args.model_type).from_pretrained(
                args.model_name_or_path,
                load_tokenizer=False
            )
            model.eval()
            model.to(args.device)

            return model
        else:
            raise ValueError(
                f"Unsupported training_type: {args.training_type}. "
                f"Supported types: 'tmd' (InfoGAN defense), 'none' (no defense)"
            )

    def _build_tokenizer(self, args: ProgramArgs):
        """Build tokenizer for the specified model type."""
        _, _, tokenizer_class = MODEL_CLASSES[args.model_type]
        tokenizer = tokenizer_class.from_pretrained(
            args.model_name_or_path,
            do_lower_case=string_to_bool(args.do_lower_case)
        )
        logging.info(f"Loaded tokenizer for {args.model_type}")
        return tokenizer

    def _build_raw_dataset(self, args: ProgramArgs):
        """Load raw datasets from disk."""
        train_raw = self.dataset_reader.read_from_file(
            file_path=f"{args.workspace}/dataset/{args.dataset_name}",
            split='train'
        )
        eval_raw = self.dataset_reader.read_from_file(
            file_path=f"{args.workspace}/dataset/{args.dataset_name}",
            split='dev'
        )
        test_raw = self.dataset_reader.read_from_file(
            file_path=f"{args.workspace}/dataset/{args.dataset_name}",
            split='test'
        )

        logging.info(f"Loaded datasets:")
        logging.info(f"  Train: {len(train_raw)} samples")
        logging.info(f"  Dev: {len(eval_raw)} samples")
        logging.info(f"  Test: {len(test_raw)} samples")

        return train_raw, eval_raw, test_raw

    def _build_tokenized_dataset(self, args: ProgramArgs):
        """Tokenize raw datasets."""
        train_dataset = self.dataset_reader.get_dataset(self.train_raw, self.tokenizer)
        eval_dataset = self.dataset_reader.get_dataset(self.eval_raw, self.tokenizer)
        test_dataset = self.dataset_reader.get_dataset(self.test_raw, self.tokenizer)

        logging.info("Datasets tokenized successfully")
        return train_dataset, eval_dataset, test_dataset

    def _build_dataloader(self, args: ProgramArgs):
        """Build data loaders for train/eval/test sets."""
        train_data_loader = self.dataset_reader.get_dataset_loader(
            dataset=self.train_dataset,
            tokenized=True,
            batch_size=args.batch_size,
            shuffle=string_to_bool(args.shuffle)
        )
        eval_data_loader = self.dataset_reader.get_dataset_loader(
            dataset=self.eval_dataset,
            tokenized=True,
            batch_size=args.batch_size,
            shuffle=False
        )
        test_data_loader = self.dataset_reader.get_dataset_loader(
            dataset=self.test_dataset,
            tokenized=True,
            batch_size=args.batch_size,
            shuffle=False
        )

        logging.info(f"Data loaders created with batch size: {args.batch_size}")
        return train_data_loader, eval_data_loader, test_data_loader

    def _build_criterion(self, args: ProgramArgs):
        """Build loss function."""
        return CrossEntropyLoss(reduction='none')

    def _build_attacker(self, args: ProgramArgs):
        """
        Build TextAttack attacker with HuggingFace model wrapper.

        Args:
            args: Program arguments

        Returns:
            TextAttack attacker instance
        """
        model_wrapper = HuggingFaceModelWrapper(
            self.model,
            self.tokenizer,
            batch_size=args.batch_size
        )

        attacker = build_english_attacker(args, model_wrapper)
        logging.info(f"Built attacker: {args.attack_method}")

        return attacker


if __name__ == '__main__':
    logging.info("=" * 80)
    logging.info("Adversarial Attack Benchmark with InfoGAN Defense")
    logging.info("=" * 80)
    logging.info(f"Configuration: {args}")

    # Initialize task
    task = AttackBenchmarkTask(args)

    # Run selected mode
    logging.info(f"\nRunning mode: {args.mode}")
    task.methods[args.mode](args)

    logging.info("\nTask completed successfully!")
