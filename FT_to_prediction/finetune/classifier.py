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
"""Model for classifier."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np

from six.moves import xrange
import paddle.fluid as fluid

from model.ernie import ErnieModel
import time
from sklearn.metrics import roc_auc_score

def create_model(args, pyreader_name, ernie_config, is_prediction=False):
    pyreader = fluid.layers.py_reader(
        capacity=50,
        shapes=[[-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1], [-1, 1],
                [-1, 1]],
        dtypes=['int64', 'int64', 'int64', 'float32', 'int64', 'int64'],
        lod_levels=[0, 0, 0, 0, 0, 0],
        name=pyreader_name,
        use_double_buffer=True)

    (src_ids, sent_ids, pos_ids, input_mask, labels,
     qids) = fluid.layers.read_file(pyreader)

    ernie = ErnieModel(
        src_ids=src_ids,
        position_ids=pos_ids,
        sentence_ids=sent_ids,
        input_mask=input_mask,
        config=ernie_config,
        use_fp16=args.use_fp16)

    cls_feats = ernie.get_pooled_output()
    cls_feats = fluid.layers.dropout(
        x=cls_feats,
        dropout_prob=0.1,
        dropout_implementation="upscale_in_train")
    logits = fluid.layers.fc(
        input=cls_feats,
        size=args.num_labels,
        param_attr=fluid.ParamAttr(
            name="cls_out_w",
            initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
        bias_attr=fluid.ParamAttr(
            name="cls_out_b", initializer=fluid.initializer.Constant(0.)))
    
    if is_prediction:
        probs = fluid.layers.softmax(logits)
        feed_targets_name = [
            src_ids.name, pos_ids.name, sent_ids.name, input_mask.name
        ]
        return pyreader, probs, feed_targets_name

    ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
        logits=logits, label=labels, return_softmax=True)
    loss = fluid.layers.mean(x=ce_loss)

    num_seqs = fluid.layers.create_tensor(dtype='int64')
    accuracy = fluid.layers.accuracy(input=probs, label=labels, total=num_seqs)
    
    auc_type = 'micro'
    class_num = args.num_labels
    labels_oh = fluid.one_hot(labels,depth=class_num)
    labels_oh = fluid.layers.cast(labels_oh, 'int64')
    
    if args.num_labels <= 2:
        auc = fluid.layers.auc(input=probs, label=labels)
    elif auc_type == 'macro':
        # macro method
        auc_macro = fluid.layers.zeros(shape=(1,), dtype='float32')
        for i in range(class_num):
            prob_0 = fluid.layers.reshape(probs[:,i], (-1,1))
            prob_0_comp = 1 - prob_0
            prob_0 = fluid.layers.concat([prob_0_comp, prob_0], axis=-1)
            label_0 = fluid.layers.reshape(labels_oh[:,i], (-1,1))

            auc_0 = fluid.layers.auc(input=prob_0, label=label_0)
            auc_macro = auc_macro + auc_0[0]
        auc = auc_macro/class_num
    elif auc_type == 'micro':
        # micro method
        prob_1 = fluid.layers.reshape(probs, (-1,1))
        prob_1_comp = 1 - prob_1
        prob_1 = fluid.layers.concat([prob_1_comp, prob_1], axis=-1)
        label_1 = fluid.layers.reshape(labels_oh, (-1,1))
        auc = fluid.layers.auc(input=prob_1, label=label_1)[0]
    
    graph_vars = {
        "loss": loss,
        "probs": probs,
        "accuracy": accuracy,
        "labels": labels,
        "num_seqs": num_seqs,
        "qids": qids,
    }

    if args.num_labels <= 2: 
        graph_vars['auc'] = auc[0]
    else:
        graph_vars['auc'] = auc

    for k, v in graph_vars.items():
        v.persistable = True

    return pyreader, graph_vars

def create_model_reg(args, pyreader_name, ernie_config, is_prediction=False):
    pyreader = fluid.layers.py_reader(
        capacity=50,
        shapes=[[-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1], [-1, 1],
                [-1, 1]],
        dtypes=['int64', 'int64', 'int64', 'float32', 'float32', 'int64'],
        lod_levels=[0, 0, 0, 0, 0, 0],
        name=pyreader_name,
        use_double_buffer=True)

    (src_ids, sent_ids, pos_ids, input_mask, labels,
     qids) = fluid.layers.read_file(pyreader)

    ernie = ErnieModel(
        src_ids=src_ids,
        position_ids=pos_ids,
        sentence_ids=sent_ids,
        input_mask=input_mask,
        config=ernie_config,
        use_fp16=args.use_fp16)

    cls_feats = ernie.get_pooled_output()
    cls_feats = fluid.layers.dropout(
        x=cls_feats,
        dropout_prob=0.1,
        dropout_implementation="upscale_in_train")
    logits = fluid.layers.fc(
        input=cls_feats,
        size=1,
        param_attr=fluid.ParamAttr(
            name="cls_out_w",
            initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
        bias_attr=fluid.ParamAttr(
            name="cls_out_b", initializer=fluid.initializer.Constant(0.)))
    
    if is_prediction:
        probs = logits
        feed_targets_name = [
            src_ids.name, pos_ids.name, sent_ids.name, input_mask.name
        ]
        return pyreader, probs, feed_targets_name

    mse_loss = fluid.layers.mse_loss(input=logits, label=labels)
    loss = fluid.layers.mean(x=mse_loss)
    #mae = fluid.layers.sqrt(loss)
    mae = fluid.layers.mean(fluid.layers.abs(logits-labels))

    num_seqs = fluid.layers.create_tensor(dtype='int64')
    ph0 = fluid.layers.fill_constant_batch_size_like(logits,shape=[-1,2],dtype="float32",value=0.5)
    ph1 = fluid.layers.fill_constant_batch_size_like(logits,shape=[-1,1],dtype="int64",value=1)
    acc = fluid.layers.accuracy(input=ph0, label=ph1, total=num_seqs)
    # cad.layers.accuracy(input=probs, label=labels, total=num_seqs)lculate R-Square
    ssr = mse_loss
    labels_a = fluid.layers.mean(labels)
    sst = fluid.layers.mse_loss(input=labels, label=labels_a)
    r2 = 1-ssr/sst

    graph_vars = {
        "loss": loss,
        "labels": labels,
        "num_seqs": num_seqs,
        "qids": qids,
        "Rsquare": r2,
        "MAE": mae,
        "acc":acc,
        "outputs":logits,
    }

    for k, v in graph_vars.items():
        v.persistable = True

    return pyreader, graph_vars

def evaluate_mrr(preds):
    last_qid = None
    total_mrr = 0.0
    qnum = 0.0
    rank = 0.0
    correct = False
    for qid, score, label in preds:
        if qid != last_qid:
            rank = 0.0
            qnum += 1
            correct = False
            last_qid = qid

        rank += 1
        if not correct and label != 0:
            total_mrr += 1.0 / rank
            correct = True

    return total_mrr / qnum


def evaluate_map(preds):
    def singe_map(st, en):
        total_p = 0.0
        correct_num = 0.0
        for index in xrange(st, en):
            if int(preds[index][2]) != 0:
                correct_num += 1
                total_p += correct_num / (index - st + 1)
        if int(correct_num) == 0:
            return 0.0
        return total_p / correct_num

    last_qid = None
    total_map = 0.0
    qnum = 0.0
    st = 0
    for i in xrange(len(preds)):
        qid = preds[i][0]
        if qid != last_qid:
            qnum += 1
            if last_qid != None:
                total_map += singe_map(st, i)
            st = i
            last_qid = qid

    total_map += singe_map(st, len(preds))
    return total_map / qnum


def evaluate_reg(exe, test_program, test_pyreader, 
        graph_vars, eval_phase, use_multi_gpu_test=False, flag=''):

    train_fetch_list = [
        graph_vars["loss"].name,
        graph_vars["MAE"].name
    ]

    if eval_phase == "train":
        if "learning_rate" in graph_vars:
            train_fetch_list.append(graph_vars["learning_rate"].name)
        outputs = exe.run(fetch_list=train_fetch_list)
        ret = {"loss": np.mean(outputs[0]),"MAE":np.mean(outputs[1])}
        if "learning_rate" in graph_vars:
            ret["learning_rate"] = float(outputs[3][0])
        return ret

    test_pyreader.start()
    total_cost, total_r2, total_mae ,total_num_seqs = 0.0, 0.0, 0.0, 0.0
    r2_nums = 0.0
    qids, labels = [], []
    time_begin = time.time()

    fetch_list = [
        graph_vars["loss"].name, graph_vars["Rsquare"].name,
        graph_vars["MAE"].name, graph_vars["labels"].name,
        graph_vars["num_seqs"].name, graph_vars["qids"].name,
        graph_vars["outputs"].name
    ]
    val_outputs = []
    val_labels = []
    while True:
        try:
            np_loss, np_r2, np_mae, np_labels, np_num_seqs, np_qids, np_outputs = exe.run(
                program=test_program, fetch_list=fetch_list) \
                        if not use_multi_gpu_test else exe.run(fetch_list=fetch_list)
            if val_outputs == []:
                val_outputs = np.copy(np_outputs)
            else:
                val_outputs = np.concatenate((val_outputs, np_outputs), axis=0)
            if val_labels ==[]:
                val_labels = np.copy(np_labels)
            else:
                val_labels = np.concatenate((val_labels, np_labels), axis=0)
            total_cost += np.sum(np_loss * np_num_seqs)
            if np_num_seqs[0] > 1:
                total_r2 += np.sum(np_r2 * np_num_seqs)
                r2_nums += np.sum(np_num_seqs)
            total_mae += np.sum(np_mae * np_num_seqs)
            total_num_seqs += np.sum(np_num_seqs)
            labels.extend(np_labels.reshape((-1)).tolist())
            if np_qids is not None:
                qids.extend(np_qids.reshape(-1).tolist())
        except fluid.core.EOFException:
            test_pyreader.reset()
            break
    time_end = time.time()

    print(
        "[%s evaluation] ave loss: %f, ave r2: %f, ave mae: %f, data_num: %d, elapsed time: %f s"
        % (eval_phase, total_cost / total_num_seqs, total_r2 /
            r2_nums, total_mae / total_num_seqs, total_num_seqs, time_end - time_begin))
    tag = time.time()
    np.save('./checkpoints/outputs_{0}_{1}_{2}.npy'.format(eval_phase, tag, flag), val_outputs)
    np.save('./checkpoints/labels_{0}_{1}_{2}.npy'.format(eval_phase, tag, flag), val_labels)


def evaluate(exe, test_program, test_pyreader, 
        graph_vars, eval_phase, use_multi_gpu_test=False, flag=''):

    train_fetch_list = [
        graph_vars["loss"].name, graph_vars["accuracy"].name,
        graph_vars["num_seqs"].name
    ]

    if eval_phase == "train":
        if "learning_rate" in graph_vars:
            train_fetch_list.append(graph_vars["learning_rate"].name)
        outputs = exe.run(fetch_list=train_fetch_list)
        ret = {"loss": np.mean(outputs[0]), "accuracy": np.mean(outputs[1])}
        if "learning_rate" in graph_vars:
            ret["learning_rate"] = float(outputs[3][0])
        return ret

    test_pyreader.start() 
    total_cost, total_acc, total_auc ,total_num_seqs, total_label_pos_num, total_pred_pos_num, total_correct_num = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    qids, labels, scores = [], [], []
    time_begin = time.time()

    fetch_list = [
        graph_vars["loss"].name, graph_vars["accuracy"].name,
        graph_vars["probs"].name, graph_vars["labels"].name,
        graph_vars["num_seqs"].name, graph_vars["qids"].name,
    ]
    do_auc = True
    if do_auc:
        fetch_list.append(graph_vars["auc"].name)
    
    val_probs = []
    val_labels = []
    while True:
        try:
            if do_auc:
                np_loss, np_acc, np_probs, np_labels, np_num_seqs, np_qids, np_auc = exe.run(
                    program=test_program, fetch_list=fetch_list) \
                            if not use_multi_gpu_test else exe.run(fetch_list=fetch_list)
            else:
                np_loss, np_acc, np_probs, np_labels, np_num_seqs, np_qids = exe.run(
                    program=test_program, fetch_list=fetch_list) \
                            if not use_multi_gpu_test else exe.run(fetch_list=fetch_list)

            val_probs += np_probs[:, 1].tolist()
            val_labels += np_labels.reshape(-1).tolist()
        except fluid.core.EOFException:
            test_pyreader.reset()
            break
    time_end = time.time()

    if len(qids) == 0:
        if do_auc:
            auc = roc_auc_score(np.array(val_labels), np.array(val_probs))
            print(
                "[%s evaluation] auc: %f, data_num: %d, elapsed time: %f s"
                % (eval_phase, auc, len(val_labels), time_end - time_begin))
        else:
            print(
                "[%s evaluation] ave loss: %f, ave acc: %f, data_num: %d, elapsed time: %f s"
                % (eval_phase, total_cost / total_num_seqs, total_acc /
                   total_num_seqs, total_num_seqs, time_end - time_begin))
    else:
        r = total_correct_num / total_label_pos_num
        p = total_correct_num / total_pred_pos_num
        f = 2 * p * r / (p + r)

        assert len(qids) == len(labels) == len(scores)
        preds = sorted(
            zip(qids, scores, labels), key=lambda elem: (elem[0], -elem[1]))
        mrr = evaluate_mrr(preds)
        map = evaluate_map(preds)
        if do_auc:
            print(
                "[%s evaluation] ave loss: %f, ave_acc: %f, ave auc: %f, mrr: %f, map: %f, p: %f, r: %f, f1: %f, data_num: %d, elapsed time: %f s"
                % (eval_phase, total_cost / total_num_seqs,
                   total_acc / total_num_seqs, total_auc / total_num_seqs, mrr, map, p, r, f, total_num_seqs,
                   time_end - time_begin))
        else:
            print(
                "[%s evaluation] ave loss: %f, ave_acc: %f, mrr: %f, map: %f, p: %f, r: %f, f1: %f, data_num: %d, elapsed time: %f s"
                % (eval_phase, total_cost / total_num_seqs,
                   total_acc / total_num_seqs, mrr, map, p, r, f, total_num_seqs,
                   time_end - time_begin))
