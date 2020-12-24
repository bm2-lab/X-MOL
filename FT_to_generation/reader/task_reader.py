#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import csv
csv.field_size_limit(1024 * 1024)
import json
import numpy as np
from collections import namedtuple

import tokenization
from batching import pad_batch_data, mask
import random

import paddle.fluid as fluid

class Seq2SeqReader(object):
    def __init__(self, args):
        self.tokenizer = getattr(tokenization, args.tokenizer)(
                vocab_file=args.vocab_path, do_lower_case=args.do_lower_case)
        self.vocab = self.tokenizer.vocab
        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.tgt_type_id = args.tgt_type_id
        self.max_src_len = args.max_src_len
        self.max_tgt_len = args.max_tgt_len
        self.max_dec_len = args.max_dec_len
        self.tokenized_input = args.tokenized_input
        self.in_tokens = args.in_tokens
        self.mask_prob = args.mask_prob
        self.continuous_position = args.continuous_position
        self.two_stream =  args.two_stream
        self.random_noise = args.random_noise
        self.is_dialogue_task = (args.task_type == "dialog")
        self.is_trans_task = (args.task_type == "trans")
        self.turn_type_size = args.turn_type_size

        if self.is_trans_task:
            self.src_tokenizer = getattr(tokenization, args.src_tokenizer)(
                vocab_file=args.src_vocab_path, do_lower_case=args.src_do_lower_case)

        # random_seed must be set for data slicing when using multi-gpu
        if args.random_seed:
            np.random.seed(args.random_seed)
        else:
            np.random.seed(0)

        self.trainer_id = 0
        self.trainer_nums = 1
        if os.getenv("PADDLE_TRAINER_ID"):
            self.trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))
        if os.getenv("PADDLE_NODES_NUM"):
            self.trainer_nums = int(os.getenv("PADDLE_TRAINERS_NUM"))

        self.current_example = 0
        self.current_epoch = 0
        self.num_examples = 0

        self.features = {}

    def get_train_progress(self):
        """Gets progress for training phase."""
        return self.current_example, self.current_epoch

    def get_num_examples(self, input_file):
        examples = self._read_tsv(input_file)
        return len(examples)

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        data_id = 0
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            headers = next(reader)
            src_indices = [
                index for index, h in enumerate(headers) if h != "tgt" and h != "knowledge"
            ]
            assert len(src_indices) <= self.tgt_type_id, "len(src_indices) > self.tgt_type_id"
            assert len(src_indices) > 0, "len(src_indices) <= 0"

            Example = namedtuple('Example', ["src", "tgt", "knowledge", "data_id"])

            examples = []
            for line in reader:
                src = []
                tgt = None
                knowledge = None
                assert len(line) == len(headers), "len(line) != len(headers)"
                for index, text in enumerate(line):
                    if index in src_indices:
                        src.append(text)
                    elif headers[index] == "tgt":
                        tgt = text
                    else:
                        knowledge = text

                examples.append(Example(src=src, tgt=tgt, knowledge=knowledge, data_id=data_id))
                data_id += 1

            return examples

    def _convert_dialogue_example_to_record(self, example, tokenizer, do_dec=False):
        turn_split = " __eou__ " 
        srcs = example.src[0].split(turn_split) 
        if len(srcs) > self.turn_type_size - 1:
            srcs = srcs[len(srcs) - (self.turn_type_size - 1):]
        cur_role_type = len(srcs) % 2 
        cur_turn_type = len(srcs)
        
        token_ids = [self.cls_id]
        role_type_ids = [cur_role_type]
        turn_type_ids = [cur_turn_type]
        position_ids = [0]
        
        if example.knowledge:
            text = example.knowledge
            if not self.tokenized_input:
                cur_tokens = tokenizer.tokenize(tokenization.convert_to_unicode(text))
            else:
                cur_tokens = tokenization.convert_to_unicode(text).split(" ")
            if len(cur_tokens) > self.max_src_len - 2:
                cur_tokens = cur_tokens[:self.max_src_len - 2]
            cur_token_ids = tokenizer.convert_tokens_to_ids(cur_tokens) + [self.sep_id]
            token_ids += cur_token_ids
            role_type_ids += [2] * len(cur_token_ids)
            turn_type_ids += [0] * len(cur_token_ids)
            position_ids += range(1, len(cur_token_ids) + 1)

        for text in srcs:
            if not self.tokenized_input:
                cur_tokens = tokenizer.tokenize(tokenization.convert_to_unicode(text))
            else:
                cur_tokens = tokenization.convert_to_unicode(text).split(" ")
            if len(cur_tokens) > self.max_src_len - 2:
                cur_tokens = cur_tokens[:self.max_src_len - 2]
            cur_token_ids = tokenizer.convert_tokens_to_ids(cur_tokens) + [self.sep_id]
            token_ids += cur_token_ids
            role_type_ids += [cur_role_type] * len(cur_token_ids)
            turn_type_ids += [cur_turn_type] * len(cur_token_ids)
            position_ids += range(1, len(cur_token_ids) + 1)
            cur_turn_type -= 1
            cur_role_type = (cur_role_type + 1) % 2

        if self.continuous_position and len(token_ids) > self.max_src_len:
            token_ids = token_ids[-self.max_src_len:]
            role_type_ids = role_type_ids[-self.max_src_len:]
            turn_type_ids = turn_type_ids[-self.max_src_len:]

        tgt_start_idx = len(token_ids)

        if not do_dec:
            assert example.tgt, "example.tgt is None"
            token_ids.append(self.cls_id)
            role_type_ids.append(0)
            turn_type_ids.append(0)
            position_ids.append(0)

            if not self.tokenized_input:
                tgt_tokens = tokenizer.tokenize(tokenization.convert_to_unicode(example.tgt))
            else:
                tgt_tokens = tokenization.convert_to_unicode(example.tgt).split(" ")
            tgt_token_ids = tokenizer.convert_tokens_to_ids(tgt_tokens)
            tgt_token_ids.append(self.sep_id)

            if len(tgt_token_ids) > self.max_tgt_len - 1:
                tgt_token_ids = tgt_token_ids[:self.max_tgt_len - 1]
            token_ids += tgt_token_ids
            role_type_ids += [0] * len(tgt_token_ids)
            turn_type_ids += [0] * len(tgt_token_ids)
            position_ids += range(1, len(tgt_token_ids) + 1)

        if self.continuous_position:
            position_ids = range(len(token_ids))

        assert len(token_ids) == len(position_ids) == len(role_type_ids) == len(turn_type_ids), \
            "not len(token_ids) == len(position_ids) == len(role_type_ids) == len(turn_type_ids)"

        Record = namedtuple(
            'Record',
            ['token_ids', 'position_ids', 'role_ids', 'turn_ids', 'tgt_start_idx', 'data_id'])
        record = Record(
            token_ids=token_ids,
            position_ids=position_ids,
            role_ids=role_type_ids,
            turn_ids=turn_type_ids,
            tgt_start_idx=tgt_start_idx,
            data_id=example.data_id)

        return record

    def _convert_example_to_record(self, example, tokenizer, do_dec=False):
        """Converts a single `Example` into a single `Record`."""
        if self.is_dialogue_task:
            return self._convert_dialogue_example_to_record(example, tokenizer, do_dec=do_dec)

        token_ids = [self.cls_id]
        text_type_ids = [0]
        position_ids = [0]
        text_type = 0
        
        for text in example.src:
            if not self.tokenized_input:
                cur_tokens = tokenizer.tokenize(tokenization.convert_to_unicode(text))
            else:
                cur_tokens = tokenization.convert_to_unicode(text).split(" ")

            if len(cur_tokens) > self.max_src_len - 2:
                #cur_tokens = cur_tokens[:self.max_src_len - 2]
                cur_tokens = cur_tokens[-(self.max_src_len - 2):]
            if self.is_trans_task:
                cur_token_ids = self.src_tokenizer.convert_tokens_to_ids(cur_tokens) + [self.sep_id]
            else:
                cur_token_ids = tokenizer.convert_tokens_to_ids(cur_tokens) + [self.sep_id]
            token_ids += cur_token_ids
            text_type_ids += [text_type] * len(cur_token_ids)
            position_ids += range(1, len(cur_token_ids) + 1) 
            text_type += 1 

        tgt_start_idx = len(token_ids)

        if not do_dec:
            assert example.tgt, "example.tgt is None"
            token_ids.append(self.cls_id)
            text_type_ids.append(self.tgt_type_id)
            position_ids.append(0)

            if not self.tokenized_input:
                tgt_tokens = tokenizer.tokenize(tokenization.convert_to_unicode(example.tgt))
            else:
                tgt_tokens = tokenization.convert_to_unicode(example.tgt).split(" ")

            tgt_token_ids = tokenizer.convert_tokens_to_ids(tgt_tokens)
            tgt_token_ids.append(self.sep_id)
            if len(tgt_token_ids) > self.max_tgt_len - 1:
                tgt_token_ids = tgt_token_ids[:self.max_tgt_len - 1]
            token_ids += tgt_token_ids
            text_type_ids += [self.tgt_type_id] * len(tgt_token_ids)
            position_ids += range(1, len(tgt_token_ids) + 1)

        if self.continuous_position:
            position_ids = range(len(token_ids))

        assert len(token_ids) == len(position_ids) == len(text_type_ids), \
            "not len(token_ids) == len(position_ids) == len(text_type_ids)"

        Record = namedtuple(
            'Record',
            ['token_ids', 'text_type_ids', 'position_ids', 'tgt_start_idx', 'data_id'])
        record = Record(
            token_ids=token_ids,
            text_type_ids=text_type_ids,
            position_ids=position_ids,
            tgt_start_idx=tgt_start_idx,
            data_id=example.data_id)

        return record

    def _prepare_batch_data(self, examples, batch_size, phase=None, do_dec=False, place=None):
        """generate batch records"""
        batch_records, max_len = [], 0
        for index, example in enumerate(examples):
            if phase == "train":
                self.current_example = index
            record = self._convert_example_to_record(example, self.tokenizer, do_dec)

            max_len = max(max_len, len(record.token_ids))
            if self.in_tokens:
                to_append = (len(batch_records) + 1) * max_len <= batch_size
            else:
                to_append = len(batch_records) < batch_size
            if to_append:
                batch_records.append(record)
            else:
                yield self._pad_batch_records(batch_records, do_dec, place)
                batch_records, max_len = [record], len(record.token_ids)

        if batch_records:
            yield self._pad_batch_records(batch_records, do_dec, place)

    def get_features(self, phase):
        return self.features.get(phase, None)

    def data_generator(self,
                       input_file,
                       batch_size,
                       epoch,
                       dev_count=1,
                       shuffle=True,
                       phase=None,
                       do_dec=False,
                       place=None):
        examples = self._read_tsv(input_file)
        if do_dec:
            features = {}
            for example in examples:
                features[example.data_id] = example
            self.features[phase] = features

        def wrapper():
            all_dev_batches = []
            for epoch_index in range(epoch):
                if phase == "train":
                    self.current_example = 0
                    self.current_epoch = epoch_index

                trainer_id = self.trainer_id
                if shuffle:
                    np.random.shuffle(examples)
                for batch_data in self._prepare_batch_data(
                        examples, batch_size, phase=phase, do_dec=do_dec, place=place):
                    if len(all_dev_batches) < dev_count:
                        all_dev_batches.append(batch_data)
                    if len(all_dev_batches) == dev_count:
                        yield all_dev_batches[trainer_id]
                        all_dev_batches = []
                if phase != "train":
                    if trainer_id < len(all_dev_batches):
                        yield all_dev_batches[trainer_id]

        return wrapper

    def _to_lodtensor(self, data, place, lod=None):
        data_tensor = fluid.LoDTensor()
        data_tensor.set(data, place)
        if lod is not None:
            data_tensor.set_lod(lod)
        return data_tensor

    def _pad_batch_records(self, batch_records, do_dec, place):
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_position_ids = [record.position_ids for record in batch_records]
        batch_tgt_start_idx = [record.tgt_start_idx for record in batch_records]

        padded_token_ids, input_mask = pad_batch_data(
            batch_token_ids,
            sent_b_starts=batch_tgt_start_idx,
            pad_idx=self.pad_id,
            is_unidirectional=True,
            return_input_mask=True)
        padded_position_ids = pad_batch_data(
            batch_position_ids, pad_idx=self.pad_id)

        if self.is_dialogue_task:  
            batch_role_ids = [record.role_ids for record in batch_records]
            batch_turn_ids = [record.turn_ids for record in batch_records]
            padded_role_ids = pad_batch_data(
                batch_role_ids, pad_idx=self.pad_id)
            padded_turn_ids = pad_batch_data(
                batch_turn_ids, pad_idx=self.pad_id)
            return_list = [
                padded_token_ids, padded_role_ids, padded_turn_ids, padded_position_ids,
                input_mask
            ]

        else:
            batch_text_type_ids = [record.text_type_ids for record in batch_records]
            padded_text_type_ids = pad_batch_data(
                batch_text_type_ids, pad_idx=self.pad_id)

            return_list = [
                padded_token_ids, padded_text_type_ids, padded_position_ids,
                input_mask
            ]
       
        max_len = padded_token_ids.shape[1]
        if do_dec:
            batch_data_ids = [record.data_id for record in batch_records]
            batch_data_ids = np.array(batch_data_ids).astype("int64").reshape([-1, 1])
            tgt_word = np.array([[self.cls_id]] * len(batch_token_ids), dtype="int64")
            tgt_word = tgt_word.reshape(-1, 1, 1)
            tgt_pos_id = np.array(batch_tgt_start_idx, dtype="int64")
            tgt_pos_id = tgt_pos_id.reshape(-1, 1, 1)

            init_score = self._to_lodtensor(np.zeros_like(tgt_word, dtype="float32").reshape(-1, 1),
                    place, [range(tgt_word.shape[0] + 1)] * 2)
            tgt_word = self._to_lodtensor(tgt_word, place, [range(tgt_word.shape[0] + 1)] * 2)
            tgt_pos_id = self._to_lodtensor(tgt_pos_id, place, [range(tgt_pos_id.shape[0] + 1)] * 2)

            init_idx = np.array(range(len(batch_token_ids)), dtype="int32")
            tgt_src_attn_bias = np.tile(input_mask[:,::max_len,:],
                    [1, 1, 1]).astype("float32")
            
            return_list += [tgt_word, tgt_pos_id, init_score, init_idx,
                    tgt_src_attn_bias, batch_data_ids]

        else:
            #batch_tgt_start_idx = [idx + 1 for idx in batch_tgt_start_idx]
            batch_token_ids, tgt_label, tgt_pos = mask(
                    batch_tokens=batch_token_ids,
                    seg_labels=None,
                    mask_word_tags=None,
                    total_token_num=-1,
                    vocab_size=-1,
                    sent_b_starts=batch_tgt_start_idx,
                    is_unidirectional=True)
            return_list += [tgt_label, tgt_pos]

        return return_list


if __name__ == '__main__':
    pass
