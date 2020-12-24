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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import numpy as np
import multiprocessing
import commands
import math

import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers

from six.moves import xrange

from model.ernie_gen import ErnieModel

from tokenization import BasicTokenizer
from utils.bleu import compute_bleu

def cal_logit(enc_out, tgt_pos, args, ernie_config):
    enc_out = fluid.layers.reshape(x=enc_out,
            shape=[-1, ernie_config["hidden_size"]])
    if tgt_pos:
        tgt_pos = fluid.layers.cast(x=tgt_pos, dtype='int32')
        tgt_feat = fluid.layers.gather(input=enc_out, index=tgt_pos)
    else:
        tgt_feat = enc_out

    tgt_trans_feat = fluid.layers.fc(
        input=tgt_feat,
        size=ernie_config["emb_size"] if ernie_config["emb_size"] is not None else ernie_config["hidden_size"],
        act=ernie_config["hidden_act"],
        param_attr=fluid.ParamAttr(
            name="mask_lm_trans_fc.w_0",
            initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
        bias_attr=fluid.ParamAttr(
            name="mask_lm_trans_fc.b_0",
            initializer=fluid.initializer.Constant(0.)))

    tgt_trans_feat = fluid.layers.layer_norm(
            tgt_trans_feat,
            begin_norm_axis=len(tgt_trans_feat.shape) - 1,
            param_attr=fluid.ParamAttr(
                name='mask_lm_trans_layer_norm_scale',
                initializer=fluid.initializer.Constant(1.)),
            bias_attr=fluid.ParamAttr(
                name='mask_lm_trans_layer_norm_bias',
                initializer=fluid.initializer.Constant(1.)))


    seq2seq_out_bias_attr = fluid.ParamAttr(
        name="mask_lm_out_fc.b_0",
        initializer=fluid.initializer.Constant(value=0.0))

    if args.weight_sharing:
        fc_out = fluid.layers.matmul(
            x=tgt_trans_feat,
            y=fluid.default_main_program().global_block().var(
                "word_embedding"),
            transpose_y=True)
        fc_out += fluid.layers.create_parameter(
            shape=[ernie_config['vocab_size']],
            dtype="float32",
            attr=seq2seq_out_bias_attr,
            is_bias=True)
    else:
        out_size = ernie_config["tgt_vocab_size"]
        if not out_size:
            out_size = ernie_config['vocab_size']
        fc_out = fluid.layers.fc(input=tgt_trans_feat,
                size=out_size,
                param_attr=fluid.ParamAttr(
                    name="mask_lm_out_fc.w_0",
                    initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
                bias_attr=seq2seq_out_bias_attr)

    return fc_out


def create_model(args, pyreader_name, ernie_config, is_prediction=False):

    if is_prediction:
        return fast_decode(args, pyreader_name, ernie_config)

    if args.task_type == "dialog":
        pyreader = fluid.layers.py_reader(
            capacity=50,
            shapes=[[-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1], [-1, args.max_seq_len, args.max_seq_len],
                [-1, 1], [-1, 1]],
            dtypes=['int64', 'int64', 'int64', 'int64', 'float32',
                'int64', 'int64'],
            lod_levels=[0, 0, 0, 0, 0, 0, 0],
            name=pyreader_name,
            use_double_buffer=True)

        (src_ids, role_ids, turn_ids, pos_ids, input_mask,
         tgt_labels, tgt_pos) = fluid.layers.read_file(pyreader)
        sent_ids = None

    else:
        pyreader = fluid.layers.py_reader(
            capacity=50,
            shapes=[[-1, args.tgt_type_id], [-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1], [-1, args.max_seq_len, args.max_seq_len], 
                [-1, 1], [-1, 1]],
            dtypes=['float32', 'int64', 'int64', 'int64', 'float32',
                'int64', 'int64'],
            lod_levels=[0, 0, 0, 0, 0, 0, 0],
            name=pyreader_name,
            use_double_buffer=True)

        (gen_tgt, src_ids, sent_ids, pos_ids, input_mask,
         tgt_labels, tgt_pos) = fluid.layers.read_file(pyreader)
        role_ids = None
        turn_ids = None

    ernie = ErnieModel(
        label_values= gen_tgt,
        src_ids=src_ids,
        position_ids=pos_ids,
        sentence_ids=sent_ids,
        role_ids=role_ids,
        turn_ids=turn_ids,
        input_mask=input_mask,
        config=ernie_config,
        use_fp16=args.use_fp16,
        is_unidirectional=True,
        two_stream=False)

    enc_out = ernie.get_sequence_output()
    fc_out = cal_logit(enc_out, tgt_pos, args, ernie_config)

    if args.label_smooth:
        out_size = ernie_config["tgt_vocab_size"]
        if not out_size:
            out_size = ernie_config['vocab_size']
        labels = fluid.layers.label_smooth(
            label=fluid.layers.one_hot(
                input=tgt_labels, depth=out_size),
            epsilon=0.1)

        ce_loss = layers.softmax_with_cross_entropy(
            logits=fc_out, label=labels, soft_label=True)
        #probs = fluid.layers.log(fluid.layers.softmax(fc_out))
        #ce_loss = fluid.layers.kldiv_loss(probs, labels, reduction='batchmean')
    else:
        ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
            logits=fc_out, label=tgt_labels, return_softmax=True)

    loss = fluid.layers.mean(x=ce_loss)

    if args.use_fp16 and args.loss_scaling > 1.0:
        loss *= args.loss_scaling

    graph_vars = {
        "loss": loss,
    }
    for k, v in graph_vars.items():
        v.persistable = True

    return pyreader, graph_vars

def fast_decode(args, pyreader_name, ernie_config):
    if args.task_type == "dialog":
        pyreader = fluid.layers.py_reader(
            capacity=50,
            shapes=[[-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1], [-1, args.max_seq_len, args.max_seq_len],
                [-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1], [-1, 1], [-1],
                [-1, 1, args.max_seq_len], [-1, 1]],
            dtypes=['int64', 'int64', 'int64', 'int64', 'float32', 'int64', 'int64', 'float32',
                'int32', 'float32', 'int64'],
            lod_levels=[0, 0, 0, 0, 0, 2, 2, 2, 0, 0 ,0],
            name=pyreader_name,
            use_double_buffer=True)

        (src_ids, role_ids, turn_ids, pos_ids, input_mask, tgt_ids, tgt_pos, init_scores, parent_idx,
            tgt_input_mask, data_ids) = fluid.layers.read_file(pyreader)
        sent_ids = None
    else:
        pyreader = fluid.layers.py_reader(
            capacity=50,
            shapes=[[-1,args.tgt_type_id], [-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1], [-1, args.max_seq_len, args.max_seq_len], 
                [-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1], [-1, 1], [-1],
                [-1, 1, args.max_seq_len], [-1, 1]],
            dtypes=['float32', 'int64', 'int64', 'int64', 'float32', 'int64', 'int64', 'float32',
                'int32', 'float32', 'int64'],
            lod_levels=[0, 0, 0, 0, 0, 2, 2, 2, 0, 0 ,0],
            name=pyreader_name,
            use_double_buffer=True)

        (gen_tgt, src_ids, sent_ids, pos_ids, input_mask, tgt_ids, tgt_pos, init_scores, parent_idx,
            tgt_input_mask, data_ids) = fluid.layers.read_file(pyreader)
        role_ids = None
        turn_ids = None

    # if args.decoding_strategy.endswith("rerank"):
    #     def repeat(x):
    #         expand_times = [1] * len(x.shape)
    #         expand_times[0] = args.num_samples
    #         y = layers.expand(x, expand_times=expand_times)
    #         y = layers.cast(y, x.dtype)
    #         return y
    #     src_ids = repeat(src_ids)
    #     if sent_ids is not None:
    #         sent_ids = repeat(sent_ids)
    #     if role_ids is not None:
    #         role_ids = repeat(role_ids)
    #     if turn_ids is not None:
    #         turn_ids = repeat(turn_ids)
    #     pos_ids = repeat(pos_ids)
    #     input_mask = repeat(input_mask)
    #     tgt_ids = repeat(tgt_ids)
    #     tgt_pos = repeat(tgt_pos)
    #     init_scores = repeat(init_scores)
    #     parent_idx = repeat(parent_idx)
    #     tgt_input_mask = repeat(tgt_input_mask)
    #     data_ids = repeat(data_ids)

    ernie = ErnieModel(
        label_values=gen_tgt,
        src_ids=src_ids,
        position_ids=pos_ids,
        sentence_ids=sent_ids,
        role_ids=role_ids,
        turn_ids=turn_ids,
        input_mask=input_mask,
        config=ernie_config,
        use_fp16=args.use_fp16,
        is_unidirectional=True,
        decoding=True,
        gather_idx=parent_idx)

    max_len = layers.fill_constant(
            shape=[1],
            dtype=tgt_ids.dtype,
            value=args.max_dec_len,
            force_cpu=True)
    step_idx = layers.fill_constant(
            shape=[1], dtype=tgt_ids.dtype, value=0, force_cpu=True)
    cond = layers.less_than(x=step_idx, y=max_len)  
    while_op = layers.While(cond)

    ids = layers.array_write(
        layers.reshape(tgt_ids, (-1, 1)), step_idx)
    pos_biases = layers.array_write(layers.reshape(tgt_pos, (-1, 1)), step_idx)
    scores = layers.array_write(init_scores, step_idx)
    tgt_masks = layers.array_write(tgt_input_mask, step_idx)

    if args.decoding_strategy == "beam_search":
        beam_size = args.beam_size
    else:
        beam_size = 1
    flag = 1
    with while_op.block():
        pre_ids = layers.array_read(array=ids, i=step_idx)
        pre_ids = layers.reshape(pre_ids, (-1, 1, 1), inplace=True)
        pre_scores = layers.array_read(array=scores, i=step_idx)
        pos_bias = layers.array_read(array=pos_biases, i=step_idx)
        pos_bias = layers.gather(input=pos_bias, index=parent_idx)
        tmp_tgt_input_mask = layers.array_read(tgt_masks, i=step_idx)
        append_mask = layers.fill_constant_batch_size_like(
                input=tmp_tgt_input_mask,
                value=1.0,
                shape=[-1, 1, 1],
                dtype=tmp_tgt_input_mask.dtype)
        tmp_tgt_input_mask = layers.concat([tmp_tgt_input_mask, append_mask], axis=2)
        pre_mask = layers.gather(input=tmp_tgt_input_mask, index=parent_idx)
        if args.continuous_position:
            pre_pos = layers.elementwise_mul(
                x=layers.fill_constant_batch_size_like(
                    input=pre_mask,
                    value=1,
                    shape=[-1, 1, 1],
                    dtype=pre_ids.dtype), y=step_idx, axis=0) + pos_bias
        else:
            pre_pos = layers.elementwise_mul(
                x=layers.fill_constant_batch_size_like(
                    input=pre_mask,
                    value=1,
                    shape=[-1, 1, 1],
                    dtype=pre_ids.dtype), y=step_idx, axis=0)

        type_ids = layers.fill_constant_batch_size_like(
                input=pre_mask,
                value=args.tgt_type_id,
                shape=[-1, 1, 1],
                dtype=pre_ids.dtype)
        role_type_ids = layers.fill_constant_batch_size_like(
                input=pre_mask,
                value=0,
                shape=[-1, 1, 1],
                dtype=pre_ids.dtype)
        turn_type_ids = layers.fill_constant_batch_size_like(
                input=pre_mask,
                value=0,
                shape=[-1, 1, 1],
                dtype=pre_ids.dtype)

        dec_out = ernie.encode(gen_tgt, pre_ids, pre_pos, type_ids, pre_mask, parent_idx, store=True, role_ids=role_type_ids, turn_ids=turn_type_ids, dec=2)
        
        fc_out = cal_logit(dec_out, None, args, ernie_config)
        probs_softmax = layers.softmax(fc_out / args.temperature)

        if args.T > 0:
            probs_softmax = layers.exp(probs_softmax/args.T)
            fc_out_basep = layers.reduce_sum(probs_softmax, dim=-1)
            fc_out_basep = layers.reshape(fc_out_basep, shape=(-1,1))
            probs = probs_softmax/fc_out_basep
        else:
            probs = probs_softmax        


        if args.decoding_strategy == "beam_search":
            topk_scores, topk_indices = layers.topk(
                input=probs, k=beam_size)
        else:
            if args.decoding_strategy.startswith("sampling"):
                sampling_ids = layers.sampling_id(probs, dtype="int")
            elif args.decoding_strategy.startswith("topk_sampling"):
                topk_probs, _ = layers.topk(input=probs, k=args.topk)
                ge_cond = layers.cast(
                    layers.greater_equal(
                        probs,
                        layers.unsqueeze(topk_probs[:, -1], [1])),
                    "float")
                probs = probs * ge_cond / layers.reduce_sum(topk_probs, dim=-1, keep_dim=True)
                sampling_ids = layers.sampling_id(probs, dtype="int")
            else:
                raise ValueError(args.decoding_strategy)

            sampling_scores = layers.one_hot(
                layers.unsqueeze(sampling_ids, [1]), probs.shape[1]
            )
            sampling_scores = sampling_scores * probs - (1 - sampling_scores) * 1e3
            topk_scores, topk_indices = layers.topk(
                input=sampling_scores, k=1)

        accu_scores = layers.elementwise_add(
            x=layers.log(topk_scores), y=pre_scores, axis=0)
        topk_indices = layers.lod_reset(topk_indices, pre_ids)
        accu_scores = layers.lod_reset(accu_scores, pre_ids)
        selected_ids, selected_scores, gather_idx = layers.beam_search(
            pre_ids=pre_ids,
            pre_scores=pre_scores,
            ids=topk_indices,
            scores=accu_scores,
            beam_size=beam_size,
            end_id=args.eos_idx,
            return_parent_idx=True)

        layers.increment(x=step_idx, value=1.0, in_place=True)
        layers.array_write(selected_ids, i=step_idx, array=ids)
        layers.array_write(selected_scores, i=step_idx, array=scores)
        layers.array_write(pre_mask, i=step_idx, array=tgt_masks)
        layers.array_write(pos_bias, i=step_idx, array=pos_biases)

        layers.assign(gather_idx, parent_idx)

        length_cond = layers.less_than(x=step_idx, y=max_len)
        finish_cond = layers.logical_not(layers.is_empty(x=selected_ids))
        layers.logical_and(x=length_cond, y=finish_cond, out=cond)

    finished_ids, finished_scores = layers.beam_search_decode(
        ids, scores, beam_size=beam_size, end_id=args.eos_idx)

    graph_vars = {
        "finished_ids": finished_ids,
        "finished_scores": finished_scores,
        "data_ids": data_ids
    }

    for k, v in graph_vars.items():
        v.persistable = True

    return pyreader, graph_vars


def post_process_seq(seq, eos_idx):
    """
    Post-process the beam-search decoded sequence. Truncate from the first
    <eos> and remove the <bos> and <eos> tokens currently.
    """
    eos_pos = len(seq)
    for i, idx in enumerate(seq):
        if idx == eos_idx:
            eos_pos = i
            break
    seq = seq[1:eos_pos]
    return seq


def evaluate_bleu(refs, preds, bleu_n=4):
    eval_res = compute_bleu(refs, preds, max_order=bleu_n)
    return eval_res[0]

def evaluate(exe,
             program,
             pyreader,
             graph_vars,
             eval_phase,
             dev_count=1,
             do_dec=False,
             vocab_path=None,
             features=None,
             length_average=False,
             length_penalty=0,
             eval_bleu=True,
             output_path=False,
             eval_script=None,
             gpu_id=None,
             merge_subword=None,
             decoding_strategy="beam_search"):

    if do_dec and not hasattr(evaluate, 'trg_idx2word'):
        evaluate.trg_idx2word = {}
        fin = open(vocab_path)
        id = 0
        for line in fin:
            vk = line.strip().decode("utf8").split("\t")
            v = vk[0]
            if len(vk) == 2:
                k = int(vk[1])
            else:
                k = id
            evaluate.trg_idx2word[k] = v
            id += 1
            if v == "[SEP]":
                evaluate.eos_idx = k 

    if eval_phase == "train":
        fetch_list = [
            graph_vars["loss"].name
        ]  
        if "learning_rate" in graph_vars:
            fetch_list.append(graph_vars["learning_rate"].name)
        outputs = exe.run(fetch_list=fetch_list)
        np_loss = outputs[0]
        ret = {
            "loss": np.mean(np_loss),
            "ppl": np.exp(np.mean(np_loss))
        }
        if "learning_rate" in graph_vars:
            ret["learning_rate"] = float(outputs[1][0])
        return ret

    if not do_dec:
        fetch_list = [
            graph_vars["loss"].name
        ]
    else:
        fetch_list = [
            graph_vars["finished_ids"].name,
            graph_vars["finished_scores"].name,
            graph_vars["data_ids"].name,
        ]


    if do_dec:
        return_numpy = False
        dec_out = {}
    else:
        steps = 0
        cost = 0.0
        return_numpy = True

    time_begin = time.time()
    pyreader.start()

    while True:
        try:
            outputs = exe.run(program=program, fetch_list=fetch_list,
                    return_numpy=return_numpy)
            if not do_dec:
                np_loss = outputs[0]
                cost += np.mean(np_loss)
                steps += 1
            else:
                seq_ids, seq_scores, data_ids = outputs
                seq_ids_list, seq_scores_list = [seq_ids], [
                    seq_scores] if isinstance(
                        seq_ids, paddle.fluid.core.LoDTensor) else (seq_ids, seq_scores)

                data_ids = np.array(data_ids).reshape(-1).tolist()
                data_idx = 0

                for seq_ids, seq_scores in zip(seq_ids_list, seq_scores_list):
                # How to parse the results:
                #   Suppose the lod of seq_ids is:
                #     [[0, 3, 6], [0, 12, 24, 40, 54, 67, 82]]
                #   then from lod[0]:
                #     there are 2 source sentences, beam width is 3.
                #   from lod[1]:
                #     the first source sentence has 3 hyps; the lengths are 12, 12, 16
                #     the second source sentence has 3 hyps; the lengths are 14, 13, 15
                    #hyps = [[] for i in range(len(seq_ids.lod()[0]) - 1)]
                    #scores = [[] for i in range(len(seq_scores.lod()[0]) - 1)]
                    for i in range(len(seq_ids.lod()[0]) -1):  # for each source sentence
                        start = seq_ids.lod()[0][i]
                        end = seq_ids.lod()[0][i + 1]
                        max_cand = None
                        for j in range(end - start):  # for each candidate
                            sub_start = seq_ids.lod()[1][start + j]
                            sub_end = seq_ids.lod()[1][start + j + 1]
                            tokens = [evaluate.trg_idx2word[idx]
                                for idx in post_process_seq(
                                    np.array(seq_ids)[sub_start:sub_end], evaluate.eos_idx)
                            ]
                            score = np.array(seq_scores)[sub_end - 1]
                            if length_average:
                                score = score / len(tokens)
                            elif length_penalty > 0:
                                score =  score / math.pow((5 + len(tokens)) / 6, length_penalty)
                            if (not max_cand) or score > max_cand[1]:
                                max_cand = (tokens, score)

                        data_id = data_ids[data_idx]
                        data_idx += 1
                        if data_id not in dec_out or dec_out[data_id][1] < max_cand[1]:
                            dec_out[data_id] = max_cand

        except fluid.core.EOFException:
            pyreader.reset()
            break

    eval_result = "Empty"
    if not do_dec:
        eval_result = "loss: " + str(cost / steps) + ", ppl: " + str(np.exp(cost / steps))
        time_end = time.time()
    else:
        tk = BasicTokenizer()
        keys = features.keys()
        writer = None
        if output_path:
            outfile = output_path + "/" + eval_phase
            outfile_part = outfile + ".part" + str(gpu_id)
            writer = open(outfile_part, "w")

        for i in keys:
            if i not in dec_out:
                continue
            pred = merge_subword(dec_out[i][0])
            if writer:
                writer.write(str(i) + "\t" + " ".join(pred).encode("utf8") + "\t"+ str(dec_out[i][1]) + "\n")

        if writer:
            writer.close()
            tmp_writer = open(output_path + "/" + eval_phase + "_dec_finish." + str(gpu_id), "w")
            tmp_writer.close()
            if gpu_id == 0:
                while True:
                    _, ret = commands.getstatusoutput('find ' + output_path + \
                                ' -maxdepth 1 -name ' + eval_phase + '"_dec_finish.*"')
                    ret = ret.split("\n")
                    if len(ret) != dev_count:
                        time.sleep(1)
                        continue
                    time_end = time.time()
                    commands.getstatusoutput("sort -t $'\t' -k 1 -n " + outfile + ".part* | " + \
                                            "awk -F \"\t\" '{print $2 \"\t\" $3}'> " + outfile)
                    commands.getstatusoutput("rm " + outfile + ".part*")
                    commands.getstatusoutput("rm " + output_path + "/" + eval_phase + "_dec_finish.*")
                    break

        if eval_script and gpu_id == 0:
            if eval_phase.startswith("dev"):
                eval_split = "dev"
            elif eval_phase.startswith("test"):
                eval_split = "test"
            else:
                eval_split = "pred"
           
            cmd = "sh " + eval_script + " " + outfile + " " + eval_split
            retcode, eval_result = commands.getstatusoutput(cmd) 
            if retcode != 0:
                eval_result = "Error in evaluation"

        elif eval_bleu and gpu_id == 0:
            fin = open(outfile)
            preds = []
            refs = []
            for line in fin:
                preds.append(line.strip().decode("utf8").split(" "))
            fin.close()
            for i in keys: 
                refs.append([tk.tokenize(features[i].tgt)])
            bleu = evaluate_bleu(refs, preds)
            eval_result = "bleu-4: " + str(bleu)

    if gpu_id == 0:    
        print("[%s evaluation] %s, elapsed time: %f s"
            % (eval_phase, eval_result, time_end - time_begin))
