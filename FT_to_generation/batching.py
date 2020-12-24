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
"""Mask, padding and batching."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random

from six.moves import xrange


def mask(batch_tokens,
         seg_labels,
         mask_word_tags,
         total_token_num,
         vocab_size,
         CLS=1,
         SEP=2,
         MASK=3,
         sent_b_starts = None,
         is_unidirectional=False):
    """
    Add mask for batch_tokens, return out, mask_label, mask_pos;
    Note: mask_pos responding the batch_tokens after padded;
    """
    max_len = max([len(sent) for sent in batch_tokens])
    mask_label = []
    mask_pos = []

    if is_unidirectional:
        assert sent_b_starts is not None, \
            "[FATAL] For unidirectional lanugae model loss," \
            " sent_b_starts should not be None"
        for sent_index, sent in enumerate(batch_tokens):
            sent_b_index = sent_b_starts[sent_index]
            mask_label.extend(sent[sent_b_index + 1:])
            # For chinese characters, it may be more reasonable
            # to predict more than one position (sent_b_index - 1) in advance
            mask_pos.extend([
                sent_index * max_len + i
                    for i in range(sent_b_index, len(sent) - 1)
            ])
        mask_label = np.array(mask_label).astype("int64").reshape([-1, 1])
        mask_pos = np.array(mask_pos).astype("int64").reshape([-1, 1])
        return batch_tokens, mask_label, mask_pos

    return None

def prepare_batch_data(insts,
                       total_token_num,
                       voc_size=0,
                       pad_id=None,
                       cls_id=None,
                       sep_id=None,
                       mask_id=None,
                       tgt_sent_id=None,
                       return_input_mask=True,
                       return_max_len=True,
                       return_num_token=False,
                       is_unidirectional=False):

    batch_src_ids = [inst[0] for inst in insts]
    batch_sent_ids = [inst[1] for inst in insts]
    batch_pos_ids = [inst[2] for inst in insts]
    labels = [inst[3] for inst in insts]
    labels = np.array(labels).astype("int64").reshape([-1, 1])
    seg_labels = [inst[4] for inst in insts]
    mask_word_tags = [inst[5] for inst in insts]
    batch_del_pos = []

    if is_unidirectional:
        sent_b_starts = [sent.index(cls_id, 1) for sent in batch_src_ids]

    # First step: do mask without padding
    assert mask_id >= 0, "[FATAL] mask_id must >= 0"
    out, mask_label, mask_pos = mask(
        batch_src_ids,
        seg_labels,
        mask_word_tags,
        total_token_num,
        vocab_size=voc_size,
        CLS=cls_id,
        SEP=sep_id,
        MASK=mask_id,
        sent_b_starts = sent_b_starts,
        is_unidirectional=is_unidirectional)
    # Second step: padding
    src_id, self_input_mask = pad_batch_data(
        out,
        sent_b_starts=sent_b_starts,
        is_unidirectional=is_unidirectional,
        pad_idx=pad_id,
        return_input_mask=True)

    pos_id = pad_batch_data(batch_pos_ids, pad_idx=pad_id)
    sent_id = pad_batch_data(batch_sent_ids, pad_idx=pad_id)

    return_list = [
        src_id, pos_id, sent_id, self_input_mask, mask_label, mask_pos, labels
    ]

    return return_list

def get_query_input(token_ids, max_len, sent_b_starts, mask_id):
    bsz = len(sent_b_starts)
    dec_len = map(lambda i:len(token_ids[i]) - sent_b_starts[i], range(bsz))
    max_len_query = max(dec_len)
    mask_datas = np.zeros((bsz, max_len_query, max_len + max_len_query))
    mask_ids = np.ones((bsz, max_len_query, 1)) * mask_id
    tgt_pos = sum(map(lambda i:list(range(max_len_query * i + 1, max_len_query * i + dec_len[i])), range(bsz)),[])
    for index, mask_data in enumerate(mask_datas):
        for i in range(dec_len[index]):
            mask_data[i, :sent_b_starts[index] + i] = 1.0
            mask_data[i, max_len + i] = 1.0
    return mask_datas.astype('float32'), mask_ids.astype('int64'), np.array(tgt_pos).reshape([-1,1]).astype('int64')


def pad_batch_data(insts,
                   pad_idx=0,
                   sent_b_starts=None,
                   is_unidirectional=False,
                   return_pos=False,
                   return_input_mask=False,
                   return_max_len=False,
                   return_num_token=False,
                   return_seq_lens=False):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []
    max_len = max(len(inst) for inst in insts)
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.

    inst_data = np.array(
        [inst + list([pad_idx] * (max_len - len(inst))) for inst in insts])
    return_list += [inst_data.astype("int64").reshape([-1, max_len, 1])]

    # position data
    if return_pos:
        inst_pos = np.array([
            list(range(0, len(inst))) + [pad_idx] * (max_len - len(inst))
            for inst in insts
        ])

        return_list += [inst_pos.astype("int64").reshape([-1, max_len, 1])]

    if return_input_mask:
        if is_unidirectional:
            assert sent_b_starts is not None, \
                "[FATAL] For unidirectional lanugae model loss," \
                " sent_b_starts should not be None"
            # This is used to avoid attention on paddings and subsequent words.
            input_mask_data = np.zeros((inst_data.shape[0], max_len, max_len))
            for index, mask_data in enumerate(input_mask_data):
                start = sent_b_starts[index]
                end = len(insts[index])
                mask_data[:end, :start] = 1.0
                # Generate the lower triangular matrix using the slice of matrix
                b = np.tril(np.ones([end - start, end - start]), 0)
                mask_data[start:end, start:end] = b
            input_mask_data = np.array(input_mask_data).reshape([-1, max_len, max_len])
        else:
            # This is used to avoid attention on paddings.
            input_mask_data = np.array([[1] * len(inst) + [0] *
                                        (max_len - len(inst)) for inst in insts])
            input_mask_data = np.expand_dims(input_mask_data, axis=-1)
        return_list += [input_mask_data.astype("float32")]

    if return_max_len:
        return_list += [max_len]

    if return_num_token:
        num_token = 0
        for inst in insts:
            num_token += len(inst)
        return_list += [num_token]

    if return_seq_lens:
        seq_lens = np.array([len(inst) for inst in insts])
        return_list += [seq_lens.astype("int64").reshape([-1, 1])]

    return return_list if len(return_list) > 1 else return_list[0]


if __name__ == "__main__":

    pass
