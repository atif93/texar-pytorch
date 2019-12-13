# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example of building XLNet language model for classification/regression.
"""

import os
import argparse
import functools
import importlib
import logging
from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F
import texar.torch as tx
from texar.torch.run import *  # pylint: disable=wildcard-import
from texar.torch.modules import (BERTClassifier, XLNetClassifier,
                                 RoBERTaClassifier, GPT2Classifier)
from texar.torch.data import (BERTTokenizer, XLNetTokenizer,
                              RoBERTaTokenizer, GPT2Tokenizer)

from utils import dataset, model_utils
from utils.processor import get_processor_class


MODEL_CLASSES = {
    'bert': (BERTClassifier, BERTTokenizer),
    'xlnet': (XLNetClassifier, XLNetTokenizer),
    'gpt2': (GPT2Classifier, GPT2Tokenizer),
    'roberta': (RoBERTaClassifier, RoBERTaTokenizer)
}

ALL_MODEL_NAMES = []
for model_class in MODEL_CLASSES.values():
    ALL_MODEL_NAMES.extend(model_class[0].available_checkpoints())


def load_config_into_args(config_path: str, args, is_dict=False):
    config_module_path = config_path.replace('/', '.').replace('\\', '.')
    if config_module_path.endswith(".py"):
        config_module_path = config_module_path[:-3]
    config_data = importlib.import_module(config_module_path)
    for key in dir(config_data):
        if not key.startswith('__') and key != "hyperparams":
            if is_dict:
                args[key] = getattr(config_data, key)
            else:
                setattr(args, key, getattr(config_data, key))


def parse_args():
    parser = argparse.ArgumentParser()

    # configs
    parser.add_argument(
        "--config-data", default=None,
        help="Path to the dataset configuration file.")
    parser.add_argument(
        "--config-model", default=None,
        help="Configuration of the downstream part of the model")

    parser.add_argument(
        '--model-type', type=str, required=True,
        choices=MODEL_CLASSES.keys(),
        help="Name of the pre-trained checkpoint to load.")
    parser.add_argument(
        '--pretrained-model-name', type=str,
        choices=ALL_MODEL_NAMES,
        help="Name of the pre-trained checkpoint to load.")

    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a saved checkpoint file to load")
    parser.add_argument(
        "--save-dir", type=str, default='output',
        help="Directory to save model checkpoints")

    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to the directory containing raw data. "
             "Defaults to 'data/<task name>'")
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="Path to the directory to cache processed data. "
             "Defaults to 'processed_data/<task name>'")
    parser.add_argument(
        "--uncased", type=bool, default=False,
        help="Whether the pretrained model is an uncased model")

    parser.add_argument(
        "--do-train", action="store_true", help="Whether to run training.")
    parser.add_argument(
        "--do-eval", action="store_true",
        help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--do-test", action="store_true",
        help="Whether to run test on the test set.")

    args = parser.parse_args()
    return args


def construct_datasets(args) -> Dict[str, tx.data.RecordData]:
    cache_prefix = f"length{args.max_seq_len}"

    tokenizer = tx.data.XLNetTokenizer(
        pretrained_model_name=args.pretrained_model_name)
    tokenizer.do_lower_case = args.uncased

    processor_class = get_processor_class(args.task)
    data_dir = args.data_dir or f"data/{processor_class.task_name}"
    cache_dir = args.cache_dir or f"processed_data/{processor_class.task_name}"
    task_processor = processor_class(data_dir)
    dataset.construct_dataset(
        task_processor, cache_dir, args.max_seq_len,
        tokenizer, file_prefix=cache_prefix)

    datasets = dataset.load_datasets(
        args.task, cache_dir, args.max_seq_len, args.batch_size,
        file_prefix=cache_prefix, eval_batch_size=args.eval_batch_size,
        shuffle_buffer=None)
    return datasets


class RegressorWrapper(tx.modules.XLNetRegressor):
    def forward(self,  # type: ignore
                batch: tx.data.Batch) -> Dict[str, torch.Tensor]:
        preds = super().forward(inputs=batch.input_ids,
                                segment_ids=batch.segment_ids,
                                input_mask=batch.input_mask)
        loss = (preds - batch.label_ids) ** 2
        loss = loss.sum() / len(batch)
        return {"loss": loss, "preds": preds}


class ClassifierWrapper(tx.modules.XLNetClassifier):
    def forward(self,  # type: ignore
                batch: tx.data.Batch) -> Dict[str, torch.Tensor]:
        logits, preds = super().forward(inputs=batch.input_ids,
                                        segment_ids=batch.segment_ids,
                                        input_mask=batch.input_mask)
        loss = F.cross_entropy(logits, batch.label_ids, reduction='none')
        loss = loss.sum() / len(batch)
        return {"loss": loss, "preds": preds}


def main(args) -> None:
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    isbert = True if args.model_type == 'bert' else False
    isxlnet = True if args.model_type == 'xlnet' else False
    isgpt2 = True if args.model_type == 'gpt2' else False
    isroberta = True if args.model_type == 'roberta' else False

    if args.config_model is None:
        if isbert:
            raise ValueError("Model config needs to be passed for this "
                             "model type")
    else:
        config_model = {}
        load_config_into_args(args.config_model, config_model, is_dict=True)
        load_config_into_args(args.config_model, args)

    if args.config_data is None:
        if isxlnet or isbert:
            raise ValueError("Data and model config need to be passed for this "
                             "model type")
    else:
        load_config_into_args(args.config_data, args)

    if isxlnet and args.seed != -1:
        make_deterministic(args.seed)
        print(f"Random seed set to {args.seed}")

    if isbert:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tx.utils.maybe_create_dir(args.save_dir)

    if isxlnet:
        datasets = construct_datasets(args)
        print("Dataset constructed")

        processor_class = get_processor_class(args.task)
        is_regression = processor_class.is_regression
        model: Union[RegressorWrapper, ClassifierWrapper]
        if is_regression:
            model = RegressorWrapper(
                pretrained_model_name=args.pretrained_model_name)
        else:
            model = ClassifierWrapper(
                pretrained_model_name=args.pretrained_model_name,
                hparams={"num_classes": len(processor_class.labels)})
        print("Model constructed")

        optim = torch.optim.Adam(
            model.param_groups(args.lr, args.lr_layer_decay_rate), lr=args.lr,
            eps=args.adam_eps, weight_decay=args.weight_decay)
        lambda_lr = model_utils.warmup_lr_lambda(
            args.train_steps, args.warmup_steps, args.min_lr_ratio)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda_lr)

        bps = args.backwards_per_step

        def get_condition(steps: int) -> Optional[cond.Condition]:
            if steps == -1:
                return None
            return cond.iteration(steps * bps)

        if is_regression:
            valid_metric: metric.Metric = metric.PearsonR(
                pred_name="preds", label_name="label_ids")
        else:
            valid_metric = metric.Accuracy(
                pred_name="preds", label_name="label_ids")
        executor = Executor(
            model=model,
            train_data=datasets["train"],
            valid_data=datasets["dev"],
            test_data=datasets.get("test", None),
            checkpoint_dir=args.save_dir or f"saved_models/{args.task}",
            save_every=get_condition(args.save_steps),
            max_to_keep=1,
            train_metrics=[
                ("loss", metric.RunningAverage(args.display_steps * bps)),
                metric.LR(optim)],
            optimizer=optim,
            lr_scheduler=scheduler,
            grad_clip=args.grad_clip,
            num_iters_per_update=args.backwards_per_step,
            log_every=cond.iteration(args.display_steps * bps),
            validate_every=get_condition(args.eval_steps),
            valid_metrics=[valid_metric, ("loss", metric.Average())],
            stop_training_on=cond.iteration(args.train_steps * bps),
            log_format="{time} : Epoch {epoch} @ {iteration:5d}it "
                       "({speed}), LR = {LR:.3e}, loss = {loss:.3f}",
            test_mode='eval',
            show_live_progress=True,
        )

        if args.checkpoint is not None:
            executor.load(args.checkpoint)

        if args.mode == 'train':
            executor.train()
            executor.save()
            executor.test(tx.utils.dict_fetch(datasets, ["dev", "test"]))
        else:
            if args.checkpoint is None:
                executor.load(load_training_state=False)  # load previous best model
            executor.test(tx.utils.dict_fetch(datasets, ["dev", "test"]))

    if isbert:
        # Loads data
        num_train_data = args.num_train_data

        # Builds BERT
        model = model_class(
            pretrained_model_name=args.pretrained_model_name,
            hparams=config_model)
        model.to(device)

        num_train_steps = int(num_train_data / args.train_batch_size *
                              args.max_train_epoch)
        num_warmup_steps = int(num_train_steps * args.warmup_proportion)

        # Builds learning rate decay scheduler
        static_lr = 2e-5

        vars_with_decay = []
        vars_without_decay = []
        for name, param in model.named_parameters():
            if 'layer_norm' in name or name.endswith('bias'):
                vars_without_decay.append(param)
            else:
                vars_with_decay.append(param)

        opt_params = [{
            'params': vars_with_decay,
            'weight_decay': 0.01,
        }, {
            'params': vars_without_decay,
            'weight_decay': 0.0,
        }]
        optim = tx.core.BertAdam(
            opt_params, betas=(0.9, 0.999), eps=1e-6, lr=static_lr)

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optim, functools.partial(model_utils.get_lr_multiplier,
                                     total_steps=num_train_steps,
                                     warmup_steps=num_warmup_steps))

        train_dataset = tx.data.RecordData(hparams=args.train_hparam,
                                           device=device)
        eval_dataset = tx.data.RecordData(hparams=args.eval_hparam,
                                          device=device)
        test_dataset = tx.data.RecordData(hparams=args.test_hparam,
                                          device=device)

        iterator = tx.data.DataIterator(
            {"train": train_dataset, "eval": eval_dataset, "test": test_dataset}
        )

        def _compute_loss(logits, labels):
            r"""Compute loss.
            """
            if model.is_binary:
                loss = F.binary_cross_entropy(
                    logits.view(-1), labels.view(-1), reduction='mean')
            else:
                loss = F.cross_entropy(
                    logits.view(-1, model.num_classes),
                    labels.view(-1), reduction='mean')
            return loss

        def _train_epoch():
            r"""Trains on the training set, and evaluates on the dev set
            periodically.
            """
            iterator.switch_to_dataset("train")
            model.train()

            for batch in iterator:
                optim.zero_grad()
                input_ids = batch["input_ids"]
                segment_ids = batch["segment_ids"]
                labels = batch["label_ids"]

                input_length = (1 - (input_ids == 0).int()).sum(dim=1)

                logits, _ = model(input_ids, input_length, segment_ids)

                loss = _compute_loss(logits, labels)
                loss.backward()
                optim.step()
                scheduler.step()
                step = scheduler.last_epoch

                dis_steps = config_data.display_steps
                if dis_steps > 0 and step % dis_steps == 0:
                    logging.info("step: %d; loss: %f", step, loss)

                eval_steps = config_data.eval_steps
                if eval_steps > 0 and step % eval_steps == 0:
                    _eval_epoch()

        @torch.no_grad()
        def _eval_epoch():
            """Evaluates on the dev set.
            """
            iterator.switch_to_dataset("eval")
            model.eval()

            nsamples = 0
            avg_rec = tx.utils.AverageRecorder()
            for batch in iterator:
                input_ids = batch["input_ids"]
                segment_ids = batch["segment_ids"]
                labels = batch["label_ids"]

                input_length = (1 - (input_ids == 0).int()).sum(dim=1)

                logits, preds = model(input_ids, input_length, segment_ids)

                loss = _compute_loss(logits, labels)
                accu = tx.evals.accuracy(labels, preds)
                batch_size = input_ids.size()[0]
                avg_rec.add([accu, loss], batch_size)
                nsamples += batch_size
            logging.info("eval accu: %.4f; loss: %.4f; nsamples: %d",
                         avg_rec.avg(0), avg_rec.avg(1), nsamples)

        @torch.no_grad()
        def _test_epoch():
            """Does predictions on the test set.
            """
            iterator.switch_to_dataset("test")
            model.eval()

            _all_preds = []
            for batch in iterator:
                input_ids = batch["input_ids"]
                segment_ids = batch["segment_ids"]

                input_length = (1 - (input_ids == 0).int()).sum(dim=1)

                _, preds = model(input_ids, input_length, segment_ids)

                _all_preds.extend(preds.tolist())

            output_file = os.path.join(args.save_dir, "test_results.tsv")
            with open(output_file, "w+") as writer:
                writer.write("\n".join(str(p) for p in _all_preds))

        if args.checkpoint:
            ckpt = torch.load(args.checkpoint)
            model.load_state_dict(ckpt['model'])
            optim.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])

        if args.do_train:
            for _ in range(config_data.max_train_epoch):
                _train_epoch()
            states = {
                'model': model.state_dict(),
                'optimizer': optim.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            torch.save(states, os.path.join(args.save_dir, 'model.ckpt'))

        if args.do_eval:
            _eval_epoch()

        if args.do_test:
            _test_epoch()


if __name__ == '__main__':
    logging.root.setLevel(logging.INFO)
    _args = parse_args()
    main(_args)
