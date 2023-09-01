# Handles model training (optimizer), loading, saving

import argparse
import os
import shutil
from copy import deepcopy

import multiprocessing
import numpy as np
import pandas as pd
import torch
from torch.nn import DataParallel
from torch.nn.modules import BatchNorm2d
from tqdm import tqdm


import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.DEBUG)


class ModelWrapper():
    def __init__(self, args, train_dataset_length):
        self.scheduler = None
        self.args = args
        self.args.gradient_accumulation_steps = args.get("gradient_accumulation_steps", 1)
        self.args.fp16 = args.get("fp16", False)
        self.initialize_model(args)
        self.initialize_opimizer(args, train_dataset_length)

        self.global_step = 0
        self.called_time = 0

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def step(self, batch, eval_mode=False):
        if eval_mode:
            with torch.no_grad():
                output_dict = self.model(**batch)

                if output_dict['loss'] is not None:
                    loss = output_dict['loss'].mean()

                    output_dict['loss'] = loss

                return output_dict

        self.optimizer.zero_grad()

        output_dict = self.model(**batch)

        loss = output_dict['loss']

        cnn_loss = output_dict.get("cnn_regularization_loss", None)
        if cnn_loss is not None and self.model.module.cnn_loss_ratio != 0:
            loss = loss + cnn_loss * self.model.module.cnn_loss_ratio
            output_dict['cnn_regularization_loss'] = cnn_loss.mean().item()

        loss = loss.mean() # This is because on MultiGPU, loss is a tensor of size GPU_NUM

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.get("fp16", False):
            self.optimizer.backward(loss)
        else:
            loss.backward()

        if (self.called_time + 1) % self.args.gradient_accumulation_steps == 0:
            if self.args.fp16:
                # modify learning rate with special warm up BERT uses
                # if args.fp16 is False, BertAdam is used and handles this automatically
                lr_this_step = self.args.learning_rate * self.warmup_linear.get_lr(self.global_step, self.args.warmup_proportion)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr_this_step

            self.optimizer.step()
            self.global_step += 1

        self.called_time += 1

        return output_dict

    def state_dict(self):
        if isinstance(self.model, DataParallel):
            save_dict = {"model":self.model.module.state_dict(),
                     "optimizer":self.optimizer.state_dict()}
        else:
            save_dict = {"model":self.model.state_dict(),
                     "optimizer":self.optimizer.state_dict()}
        return save_dict

    def save_checkpoint(self, serialization_dir, epoch, val_metric_per_epoch, is_best = False):
        assert(serialization_dir)
        model_path = os.path.join(serialization_dir, "model_state_epoch_{}.th".format(epoch))
        model_state = self.model.module.state_dict() if isinstance(self.model, DataParallel) else self.model.state_dict()
        torch.save(model_state, model_path)

        training_state = {'epoch': epoch,
                          'val_metric_per_epoch': val_metric_per_epoch,
                          'optimizer': self.optimizer.state_dict()
                          }
        training_path = os.path.join(serialization_dir,
                                     "training_state_epoch_{}.th".format(epoch))
        torch.save(training_state, training_path)

        if is_best:
            print("Best validation performance so far. Copying weights to '{}/best.th'.".format(serialization_dir))
            shutil.copyfile(model_path, os.path.join(serialization_dir, "best.th"))

    def save_checkpoint_step(self, serialization_dir, step, epoch, is_best = False):
        
        assert(serialization_dir)
        model_path = os.path.join(serialization_dir, "model_step_{}_epoch_{}.th".format(step, epoch))
        model_state = self.model.module.state_dict() if isinstance(self.model, DataParallel) else self.model.state_dict()
        torch.save(model_state, model_path)

        training_state = {'step': step,
                          'epoch': epoch,
                          'val_metric_per_epoch': None,
                          'optimizer': self.optimizer.state_dict()
                          }
        training_path = os.path.join(serialization_dir,
                                     "training_step_{}_epoch_{}.th".format(step, epoch))
        torch.save(training_state, training_path)



    def freeze_detector(self):
        if hasattr(self.model.module, "detector"):
            detector = self.model.module.detector
            for submodule in detector.backbone.modules():
                if isinstance(submodule, BatchNorm2d):
                    submodule.track_running_stats = False
                for p in submodule.parameters():
                    p.requires_grad = False
        else:
            print("No detector found.")

    @staticmethod
    def read_and_insert_args(args, confg=None):
        import commentjson
        from attrdict import AttrDict
        dict_args = vars(args)
        if confg is not None:
            with open(confg) as f:
                config_json = commentjson.load(f)
            config_json.update(dict_args)
            args = AttrDict(config_json)
        else:
            args = AttrDict(dict_args)
        # args.model.bert_model_name = args.text_encoder
        return args




