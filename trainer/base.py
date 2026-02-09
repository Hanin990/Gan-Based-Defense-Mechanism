
# this commants lines to be updted



import logging

import torch
from tqdm import tqdm
from typing import Tuple
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
import wandb
from utils.my_utils import convert_batch_to_bert_input_dict
from itertools import islice
from utils.metrics import Metric, ClassificationMetric

class BaseTrainer:
    def __init__(self,
                 data_loader: DataLoader,
                 model: nn.Module,
                 loss_function: _Loss,
                 optimizer: Optimizer,
                 lr_scheduler: _LRScheduler = None,
                 writer: SummaryWriter = None):
        self.data_loader = data_loader
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.writer = writer
        self.global_step = 0

    @classmethod
    def add_args(cls, parser):
        group = parser.add_argument_group("Trainer")
        # group.add_argument('--epochs', default=10, type=int, 
        #     help='training epochs')
        group.add_argument('--gradient_accumulation_steps', default=1, type=int,
            help='Number of updates steps to accumulate before performing a backward/update pass')
        group.add_argument('--learning-rate', default=2e-5, type=float,
            help='The initial learning rate for Adam')
        group.add_argument('--weight-decay', default=1e-6, type=float,
            help='weight decay')
        group.add_argument('--adam-epsilon', default=1e-8, type=float,
            help='epsilon for Adam optimizer')
        group.add_argument('--max-grad-norm', default=1.0, type=float,
            help='max gradient norm')
        group.add_argument('--learning-rate-decay', default=0.1, type=float,
            help='Proportion of training to perform linear learning rate warmup for')
        # group.add_argument('--compare-key', type=str, default='+accuracy',
        #     help="the key to compare when choosing the best modeling to be saved, default is '-loss'"+
        #     "where '+X' means the larger the X is, the better the modeling."+
        #     "where '-X' means the smaller the X is, the better the modeling."+
        #     "e.g., when X == '-loss', using loss to compare which epoch is best")
        
    def train_epoch(self, args, epoch: int, limit_data: int = None) -> dict:
        print("Epoch {}:".format(epoch))
        logging.info("Epoch {}:".format(epoch))
        self.model.train()

        epoch_iterator = tqdm(islice(self.data_loader, 5000))
        total_steps = len(self.data_loader) if not limit_data else min(len(self.data_loader), limit_data)
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        metric = ClassificationMetric(compare_key='+accuracy')

        oom_number = 0
        for batch in tqdm(epoch_iterator):
            try:
                # Forward pass
                batch = tuple(t.to(self.model.device) for t in batch)
                output = self.forward(args, batch)

                # Extract logits and labels
                logits = output[0]  # Assuming the first output is logits
                golds = batch[3]  # Assuming labels are in the fourth position

                # Compute loss
                losses = self.loss_function(logits, golds.view(-1))
                loss = torch.mean(losses)
                total_loss += loss.item()

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
                self.optimizer.step()

                # Calculate metrics
                total_correct += torch.sum(torch.argmax(logits, dim=1) == golds).item()
                total_samples += golds.size(0)
                metric(losses , logits, golds)

                epoch_iterator.set_description('loss: {:.4f}'.format(loss.item()))
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logging.warning('oom in batch forward / backward pass, attempting to recover from OOM')
                    print('oom in batch forward / backward pass, attempting to recover from OOM')
                    self.model.zero_grad()
                    self.optimizer.zero_grad()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    oom_number += 1
                else:
                    raise e

        accuracy = total_correct / total_samples if total_samples > 0 else 0
        f1 = metric.get_metric(reset=False)['f1']


        logging.info(f"Training loss: {total_loss:.4f}")
        logging.info(f"Training accuracy: {accuracy:.4f}")
        logging.info(f"Training F1: {f1:.4f}")
        logging.warning('oom number : {}, oom rate : {:.2f}%'.format(oom_number, oom_number / len(self.data_loader) * 100))

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "loss": total_loss,
            "accuracy": accuracy,
            "f1": f1
        })



        # Return metrics for the epoch
        return {'loss': total_loss, 'accuracy': accuracy, 'f1': f1}


        
        
        

#     def train_epoch(self, args, epoch: int, limit_data: int = None) -> None:
#         print("Epoch {}:".format(epoch))
#         logging.info("Epoch {}:".format(epoch))
#         self.model.train()

#         epoch_iterator = tqdm(islice(self.data_loader, 5000))
#         total_steps = len(self.data_loader) if not limit_data else min(len(self.data_loader), limit_data)
#         total_loss = 0.0
#         total_correct = 0
#         total_samples = 0
#         metric = ClassificationMetric(compare_key='+accuracy')

#         # Initialize training log dict
#         # train_log = {
#         #     'train/loss': 0.0,
#         #     'train/learning_rate': self.optimizer.param_groups[0]['lr'],
#         #     'train/epoch': epoch,
#         #     'train/global_step': global_step,
#         #     'train/progress': 0.0
#         # }

#         oom_number = 0
#         for batch in tqdm(epoch_iterator):
#             # if limit_data and step >= limit_data:
#             #     break

#             try:

#                 loss, logits, golds = self.train_batch(args, batch)
#                 total_loss += loss

#                 # Evaluate batch predictions
#                 total_correct += torch.sum(torch.argmax(logits, dim=1) == golds).item()
#                 total_samples += golds.size(0)

#                 metric(loss, logits, golds)

#                 epoch_iterator.set_description('loss: {:.4f}'.format(loss))
#             except RuntimeError as e:
#                 if "out of memory" in str(e):
#                     logging.warning('oom in batch forward / backward pass, attempting to recover from OOM')
#                     print('oom in batch forward / backward pass, attempting to recover from OOM')
#                     self.model.zero_grad()
#                     self.optimizer.zero_grad()
#                     if torch.cuda.is_available():
#                         torch.cuda.empty_cache()
#                     oom_number += 1
#                 else:
#                     raise e

        
#         accuracy = total_correct / total_samples if total_samples > 0 else 0
#         f1 = metric.get_metric(reset=False)['f1']

#         logging.info(f"Training loss: {total_loss}")
#         logging.info(f"Training accuracy: {accuracy}")
#         logging.info(f"Training F1: {f1}")
#         logging.warning('oom number : {}, oom rate : {:.2f}%'.format(oom_number, oom_number / len(self.data_loader) * 100))
#         return {'loss': total_loss, 'accuracy': accuracy, 'f1': f1}

#     def train_epoch(self, args, epoch: int) -> None:
#         print("Epoch {}:".format(epoch))
#         logging.info("Epoch {}:".format(epoch))
#         self.model.train()

#         epoch_iterator = tqdm(self.data_loader)
#         total_steps = len(self.data_loader)
#         global_step = epoch * total_steps
    
#     # Initialize training log dict
#         train_log = {
#         'train/loss': 0.0,
#         'train/learning_rate': self.optimizer.param_groups[0]['lr'],
#         'train/epoch': epoch,
#         'train/global_step': global_step,
#         'train/progress': 0.0
#     }
#         oom_number = 0
#         for step, batch in enumerate(epoch_iterator):
#             try:
#                 loss = self.train_batch(args, batch)
#                 current_step = global_step + step
#                 progress = (current_step) / (args.epochs * total_steps) * 100
            
#             # Update training logs
#                 train_log.update({
#                 'train/loss': loss,
#                 'train/learning_rate': self.optimizer.param_groups[0]['lr'],
#                 'train/global_step': current_step,
#                 'train/progress': progress
#             })
            
#                 # Log to wandb
#                 wandb.log(train_log, step=current_step)
            
#                 epoch_iterator.set_description('loss: {:.4f}'.format(loss))
#             except RuntimeError as e:
#                 if "out of memory" in str(e):
#                     logging.warning('oom in batch forward / backward pass, attempting to recover from OOM')
#                     print('oom in batch forward / backward pass, attempting to recover from OOM')
#                     self.model.zero_grad()
#                     self.optimizer.zero_grad()
#                     if torch.cuda.is_available():
#                         torch.cuda.empty_cache()
#                     oom_number += 1
#                 else:
#                     raise e
#         logging.warning('oom number : {}, oom rate : {:.2f}%'.format(oom_number, oom_number / len(self.data_loader) * 100))
#         return


    def train_batch(self, args, batch: Tuple) -> Tuple[float, torch.Tensor, torch.Tensor]:
        # print(f"Batch structure: {batch}")  # Debugging batch structure
        self.model.zero_grad()
        self.optimizer.zero_grad()
        result = self.train(args, batch)
        if not isinstance(result, tuple) or len(result) != 3:
            raise ValueError("The train method must return a tuple of (loss, logits, golds). Received: {}".format(result))

        loss, logits, golds = result
        # if isinstance(result, tuple) and len(result) == 3:
        #     loss, logits, golds = result
        # else:
        #     raise ValueError("The train method must return a tuple of (loss, logits, golds)")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self.lr_scheduler:
            self.lr_scheduler.step()
        self.write_tensorboard(loss)
        self.global_step += 1
        return loss

    def train(self, args, batch: Tuple) -> Tuple[float, torch.Tensor, torch.Tensor]:
        assert isinstance(batch[0], torch.Tensor)
        batch = tuple(t.to(self.model.device)for t in batch)
        output = self.forward(args, batch)
        if not isinstance(output, tuple) or len(output) < 1:
            raise ValueError("Model output must be a tuple with logits as the first element.")
        logits = output[0]
        golds = batch[3]
        print(f"Loss: {loss}, Logits: {logits.shape}, Golds: {golds.shape}")

        print(f"Logits: {logits}, Golds: {golds}")  # Debugging logits and golds
        losses = self.loss_function(logits, golds.view(-1))
        loss = torch.mean(losses)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
        self.optimizer.step()
        return loss.item()
    
    def forward(self, args, batch: Tuple) -> Tuple:
        '''
        for Bert-like model, batch_input should contains "input_ids", "attention_mask","token_type_ids" and so on
        '''
        inputs = convert_batch_to_bert_input_dict(batch, args.model_type)
        # print(f"Converted inputs: {inputs}")  # Debugging converted inputs
        output = self.model(**inputs)
        return output  # No changes here, returns full model output
    def write_tensorboard(self, loss: float, **kwargs):
        # if self.writer is not None:
        #     self.writer.add_scalar('loss', loss, self.global_step)
        pass