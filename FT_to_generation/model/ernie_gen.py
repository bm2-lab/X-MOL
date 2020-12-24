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
"""Ernie model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
from time import time

import six
import paddle.fluid as fluid

from model.transformer_encoder import encoder, two_stream_encoder, pre_process_layer


class ErnieConfig(object):
    def __init__(self, config_path):
        self._config_dict = self._parse(config_path)

    def _parse(self, config_path):
        try:
            with open(config_path) as json_file:
                config_dict = json.load(json_file)
        except Exception:
            raise IOError("Error in parsing Ernie model config file '%s'" %
                          config_path)
        else:
            return config_dict

    def __getitem__(self, key):
        return self._config_dict.get(key, None)

    def __setitem__(self, key, value):
        self._config_dict[key] = value

    def print_config(self):
        for arg, value in sorted(six.iteritems(self._config_dict)):
            print('%s: %s' % (arg, value))
        print('------------------------------------------------')


class ErnieModel(object):
    def __init__(self,
                 label_values,
                 src_ids,
                 position_ids,
                 sentence_ids,
                 input_mask,
                 config,
                 role_ids=None,
                 turn_ids=None,
                 weight_sharing=True,
                 use_fp16=False,
                 is_unidirectional=False,
                 two_stream=False,
                 decoding=False,
                 gather_idx=None):

        self._emb_size = config['emb_size'] if config['emb_size'] is not None else config['hidden_size']
        self._hidden_size = config['hidden_size']
        self._n_layer = config['num_hidden_layers']
        self._n_head = config['num_attention_heads']
        self._voc_size = config['vocab_size']
        self._max_position_seq_len = config['max_position_embeddings']
        self._sent_types = config['type_vocab_size']
        
        self._hidden_act = config['hidden_act']
        self._prepostprocess_dropout = config['hidden_dropout_prob']
        self._attention_dropout = config['attention_probs_dropout_prob']
        self._weight_sharing = weight_sharing
        self._is_unidirectional = is_unidirectional
        self._word_emb_name = "word_embedding"
        self._pos_emb_name = "pos_embedding"
        self._sent_emb_name = "sent_embedding"
        if config["is_dialogue_task"]:
            self._role_type_size = config["role_type_size"]
            self._turn_type_size = config["turn_type_size"]
            self._role_emb_name = "role_embedding"
            self._turn_emb_name = "turn_embedding"
        self._is_dialogue_task = config["is_dialogue_task"]

        self._dtype = "float16" if use_fp16 else "float32"
        self._emb_dtype = "float32"
        self._two_stream = two_stream

        self._epsilon = config['epsilon'] if config['epsilon'] is not None else 1e-5
        self._param_share = config['param_share'] if config['param_share'] is not None else "normal"
        self._n_layer_per_block = config['n_layer_per_block'] if config['n_layer_per_block'] is not None else 1
        self._pre_encoder_cmd = config['pre_encoder_cmd'] if config['pre_encoder_cmd'] is not None else 'nd'
        self._preprocess_cmd = config['preprocess_cmd'] if config['preprocess_cmd'] is not None else ''
        self._postprocess_cmd= config['postprocess_cmd'] if config['postprocess_cmd'] is not None else 'dan'
        if self._hidden_size != self._emb_size:
            self._emb_mapping_in = True
        else:
            self._emb_mapping_in = config['emb_mapping_in'] if config['emb_mapping_in'] is not None else False
        
        self.print_n = 0
        if decoding:
            self.caches = [{
                "k":
                fluid.layers.fill_constant_batch_size_like(
                    input=src_ids,
                    shape=[-1, 0, self._hidden_size],
                    dtype=self._dtype,
                    value=0),
                "v":
                fluid.layers.fill_constant_batch_size_like(
                    input=src_ids,
                    shape=[-1, 0, self._hidden_size],
                    dtype=self._dtype,
                    value=0),
            } for i in range(self._n_layer)]
        else:
            self.caches = None

        # Initialize all weigths by truncated normal initializer, and all biases
        # will be initialized by constant zero by default.
        self._param_initializer = fluid.initializer.TruncatedNormal(
            scale=config['initializer_range'])

        self._build_model(label_values, src_ids, position_ids, sentence_ids, input_mask, gather_idx, role_ids, turn_ids)
   
    def _gen_input(self, label_values, src_ids, position_ids, sentence_ids, input_mask, \
            query_stream=False, role_ids=None, turn_ids=None, dec=0):
        fluid.layers.Print(src_ids, summarize=-1, message='inputSMILES@'+ str(round(time(),2))[4:])
        emb_out = fluid.layers.embedding(
            input=src_ids,
            size=[self._voc_size, self._emb_size],
            dtype=self._emb_dtype,
            param_attr=fluid.ParamAttr(
                name=self._word_emb_name, initializer=self._param_initializer),
            is_sparse=False)
        conditional_generation = True
        rep_loc = 0
        if conditional_generation:
            if dec in (0,1):
                """
                # random method v0
                # label_emb ~ N(label, 1)
                random_label = fluid.layers.gaussian_random_batch_size_like(emb_out, shape=[-1,1,self._emb_size])
                #fluid.layers.Print(fluid.layers.mean(random_label[0,0,:]),print_tensor_type=False,print_tensor_shape=False,print_tensor_lod=False)
                #fluid.layers.Print(label_values[0],print_tensor_type=Flase,print_tensor_shape=False,print_tensor_lod=False)
                src_basic = fluid.layers.fill_constant_batch_size_like(random_label, shape=[-1,1,self._emb_size], value=1, dtype='float32')
                src_labels = fluid.layers.cast(label_values, "float32")
                src_labels = src_basic * src_labels
                random_label_emb = random_label+src_labels
                # v0 end
                
                # random method v1
                # label_emb ~ N(0, 1), the last item is label
                fluid.layers.Print(label_values)
                random_label = fluid.layers.gaussian_random_batch_size_like(emb_out, shape=[-1,1,self._emb_size-1])
                src_basic = fluid.layers.fill_constant_batch_size_like(random_label, shape=[-1,1,1], value=1, dtype='float32')
                src_labels = fluid.layers.cast(label_values, "float32")
                src_labels = src_basic * src_labels
                fluid.layers.Print(random_label[0,0,-10:])
                    #fluid.layers.Print(label_values,print_tensor_type=False,print_tensor_shape=False,print_tensor_lod=False)
                random_label_emb = fluid.layers.concat([random_label, src_labels], axis=-1)
                # v1 end
                
                # random method v2
                # no random, label_emb = float value
                src_basic = fluid.layers.fill_constant_batch_size_like(emb_out, shape=[-1,1,self._emb_size], value=1, dtype='float32')
                src_labels = fluid.layers.cast(label_values, "float32")
                random_label_emb = src_basic * src_labels
                # v2 end
                
                # random method v3
                # no random, only replace the last dim of original embedding
                src_basic = fluid.layers.fill_constant_batch_size_like(emb_out, shape=[-1,1,128], value=1, dtype='float32')
                src_labels = fluid.layers.cast(label_values, "float32")
                src_labels = src_basic * src_labels
                emb_out_p = fluid.layers.reshape(emb_out[:,rep_loc,128:], (-1,1,self._emb_size-128))
                random_label_emb = fluid.layers.concat([src_labels, emb_out_p], axis=-1)
                # v3 end
                """
                # random method v4
                # no random, scaling in original embedding
                src_labels = fluid.layers.cast(label_values, "float32")
                random_label_emb =  fluid.layers.reshape(emb_out[:,rep_loc,:], (-1,1,self._emb_size)) * src_labels
                # v4 end
                """
                # random method v5
                # no random,  add in original embedding
                src_basic = fluid.layers.fill_constant_batch_size_like(emb_out, shape=[-1,1,self._emb_size], value=1, dtype='float32')
                src_labels = fluid.layers.cast(label_values, "float32")
                src_labels = src_basic * src_labels
                random_label_emb =  fluid.layers.reshape(emb_out[:,rep_loc,:], (-1,1,self._emb_size)) + src_labels
                # v5 end
                """
                if rep_loc == 1:
                    emb_out_a = emb_out[:,0,:]
                    emb_out_a = fluid.layers.reshape(emb_out_a, (-1,1,self._emb_size))
                    random_label_emb = fluid.layers.concat([emb_out_a, random_label_emb], axis=1)
                    if emb_out.shape[1] > 2: 
                        emb_out_c = emb_out[:,2:,:]
                        emb_out = fluid.layers.concat([random_label_emb, emb_out_c], axis=1)
                    else:
                        emb_out = random_label_emb
                else:
                    if emb_out.shape[1] == 1:
                        emb_out = random_label_emb
                    else:
                        emb_out_c = emb_out[:,1:,:]
                        emb_out = fluid.layers.concat([random_label_emb, emb_out_c], axis=1)
                #fluid.layers.Print(emb_out[0,0,-10:])
        position_emb_out = fluid.layers.embedding(
            input=position_ids,
            size=[self._max_position_seq_len, self._emb_size],
            dtype=self._emb_dtype,
            param_attr=fluid.ParamAttr(
                name=self._pos_emb_name, initializer=self._param_initializer))

        if not self._is_dialogue_task:
            sent_emb_out = fluid.layers.embedding(
                sentence_ids,
                size=[self._sent_types, self._emb_size],
                dtype=self._emb_dtype,    
                param_attr=fluid.ParamAttr(
                    name=self._sent_emb_name, initializer=self._param_initializer))
            emb_out = emb_out + position_emb_out + sent_emb_out

        else:
            role_emb_out = fluid.layers.embedding(
                input=role_ids,
                size=[self._role_type_size, self._emb_size],
                dtype=self._emb_dtype,
                param_attr=fluid.ParamAttr(
                    name=self._role_emb_name, initializer=self._param_initializer))
            turn_emb_out = fluid.layers.embedding(
                input=turn_ids,
                size=[self._turn_type_size, self._emb_size],
                dtype=self._emb_dtype,
                param_attr=fluid.ParamAttr(
                    name=self._turn_emb_name, initializer=self._param_initializer))
            emb_out = emb_out + position_emb_out + role_emb_out + turn_emb_out

        emb_out = pre_process_layer(
            emb_out,
            self._pre_encoder_cmd,
            self._prepostprocess_dropout,
            name="pre_encoder",
            epsilon=self._epsilon)
        if self._emb_mapping_in:
            emb_out = fluid.layers.fc(input=emb_out,
                          num_flatten_dims=2,
                          size=self._hidden_size,
                          param_attr=fluid.ParamAttr(
                              name='emb_hidden_mapping',
                              initializer=self._param_initializer),
                          bias_attr='emb_hidden_mapping_bias')


        if self._dtype is "float16":
            emb_out = fluid.layers.cast(x=emb_out, dtype=self._dtype)
            input_mask = fluid.layers.cast(x=input_mask, dtype=self._dtype)

        self_attn_mask = input_mask
        if not self._is_unidirectional:
            self_attn_mask = fluid.layers.matmul(
                x=input_mask, y=input_mask, transpose_y=True)
        self_attn_mask = fluid.layers.scale(
            x=self_attn_mask, scale=1e4, bias=-1.0, bias_after_scale=False)
        n_head_self_attn_mask = fluid.layers.stack(
            x=[self_attn_mask] * self._n_head, axis=1)
        n_head_self_attn_mask.stop_gradient = True

        return emb_out, n_head_self_attn_mask


    def encode(self, label_values, src_ids, position_ids, sentence_ids, input_mask, gather_idx=None, store=False, role_ids=None, turn_ids=None, dec=0):
        # padding id in vocabulary must be set to 0
        
        if self._two_stream:
            emb_out, n_head_self_attn_mask = self._gen_input(src_ids[0],
                position_ids[0], sentence_ids[0], input_mask[0], query_stream=False, role_ids=role_ids[0], turn_ids=turn_ids[0])
            #g_ids = fluid.layers.zeros_like(position_ids[1])
            g_emb_out, n_head_query_attn_mask = self._gen_input(src_ids[1],
                position_ids[1], sentence_ids[1], input_mask[1], query_stream=True, role_ids=role_ids[1], turn_ids=turn_ids[1])

            self._enc_out_context, self._enc_out_query = two_stream_encoder(
                enc_input_context=emb_out,
                enc_input_query=g_emb_out,
                attn_bias_context=n_head_self_attn_mask,
                attn_bias_query=n_head_query_attn_mask,
                n_layer=self._n_layer,
                n_head=self._n_head,
                d_key=self._hidden_size // self._n_head,
                d_value=self._hidden_size // self._n_head,
                d_model=self._hidden_size,
                d_inner_hid=self._hidden_size * 4,
                prepostprocess_dropout=self._prepostprocess_dropout,
                attention_dropout=self._attention_dropout,
                relu_dropout=0,
                hidden_act=self._hidden_act,
                preprocess_cmd=self._preprocess_cmd,
                postprocess_cmd=self._postprocess_cmd,
                param_initializer=self._param_initializer,
                epsilon=self._epsilon,
                param_share=self._param_share,
                n_layer_per_block=self._n_layer_per_block,
                name='encoder')
            return self._enc_out_query
        else:
            emb_out, n_head_self_attn_mask = self._gen_input(label_values, src_ids,
                position_ids, sentence_ids, input_mask, query_stream=False, role_ids=role_ids, turn_ids=turn_ids, dec=dec)
            self._enc_out_context = encoder(
                enc_input=emb_out,
                attn_bias=n_head_self_attn_mask,
                n_layer=self._n_layer,
                n_head=self._n_head,
                d_key=self._hidden_size // self._n_head,
                d_value=self._hidden_size // self._n_head,
                d_model=self._hidden_size, 
                d_inner_hid=self._hidden_size * 4,
                prepostprocess_dropout=self._prepostprocess_dropout,
                attention_dropout=self._attention_dropout,
                relu_dropout=0,
                hidden_act=self._hidden_act,
                preprocess_cmd=self._preprocess_cmd,
                postprocess_cmd=self._postprocess_cmd,
                param_initializer=self._param_initializer,
                epsilon=self._epsilon,
                param_share=self._param_share,
                n_layer_per_block=self._n_layer_per_block,
                name='encoder',
                caches=self.caches,
                gather_idx=gather_idx,
                store=store)
            return self._enc_out_context

    def _build_model(self, label_values, src_ids, position_ids, sentence_ids, input_mask, gather_idx=None, role_ids=None, turn_ids=None):
        store = False
        if self.caches:
            store = True
        self._enc_out = self.encode(label_values, src_ids, position_ids, sentence_ids, input_mask, gather_idx, store=store, role_ids=role_ids, turn_ids=turn_ids)

    def get_sequence_output(self):
        return self._enc_out

    def get_pooled_output(self):
        """Get the first feature of each sequence for classification"""
        next_sent_feat = fluid.layers.slice(
            input=self._enc_out, axes=[1], starts=[0], ends=[1])
        if self._dtype == "float16":        
            next_sent_feat = fluid.layers.cast(
                x=next_sent_feat, dtype=self._emb_dtype)
        next_sent_feat = fluid.layers.fc(
            input=next_sent_feat,
            size=self._hidden_size,
            act="tanh",
            param_attr=fluid.ParamAttr(
                name="pooled_fc.w_0", initializer=self._param_initializer),
            bias_attr="pooled_fc.b_0")
        return next_sent_feat

    def get_pretraining_output(self, mask_label, mask_pos, labels):
        """Get the loss & accuracy for pretraining"""

        mask_pos_context = fluid.layers.cast(x=mask_pos, dtype='int32')

        reshaped_emb_out_context = fluid.layers.reshape(
            x=self._enc_out_context, shape=[-1, self._hidden_size])
        # extract masked tokens' feature

        mask_feat_context = fluid.layers.gather(input=reshaped_emb_out_context, \
            index=mask_pos_context)

        if self._dtype == "float16":
            mask_feat_context = fluid.layers.cast(x=mask_feat_context, dtype=self._emb_dtype)

        # transform: fc
        mask_trans_feat_context = fluid.layers.fc(
            input=mask_feat_context,
            size=self._emb_size,
            act=self._hidden_act,
            param_attr=fluid.ParamAttr(
                name='mask_lm_trans_fc.w_0',
                initializer=self._param_initializer),
            bias_attr=fluid.ParamAttr(name='mask_lm_trans_fc.b_0'))
        
        # transform: layer norm 
        mask_trans_feat_context = fluid.layers.layer_norm(
            mask_trans_feat_context,
            begin_norm_axis=len(mask_trans_feat_context.shape) - 1,
            param_attr=fluid.ParamAttr(
                name='mask_lm_trans_layer_norm_scale',
                initializer=fluid.initializer.Constant(1.)),
            bias_attr=fluid.ParamAttr(
                name='mask_lm_trans_layer_norm_bias',
                initializer=fluid.initializer.Constant(1.)))
        # transform: layer norm 
        #mask_trans_feat = pre_process_layer(
        #    mask_trans_feat, 'n', name='mask_lm_trans')

        mask_lm_out_bias_attr = fluid.ParamAttr(
            name="mask_lm_out_fc.b_0",
            initializer=fluid.initializer.Constant(value=0.0))


        if self._weight_sharing:
            fc_out_context = fluid.layers.matmul(
                x=mask_trans_feat_context,
                y=fluid.default_main_program().global_block().var(
                    self._word_emb_name),
                transpose_y=True)
            fc_out_context += fluid.layers.create_parameter(
                shape=[self._voc_size],
                dtype=self._emb_dtype,
                attr=mask_lm_out_bias_attr,
                is_bias=True)

        else:
            fc_out = fluid.layers.fc(input=mask_trans_feat_context,
                                     size=self._voc_size,
                                     param_attr=fluid.ParamAttr(
                                         name="mask_lm_out_fc.w_0",
                                         initializer=self._param_initializer),
                                     bias_attr=mask_lm_out_bias_attr)

        mask_lm_loss_context = fluid.layers.softmax_with_cross_entropy(
            logits=fc_out_context, label=mask_label)

        if self._is_unidirectional:
            mask_lm_loss_context = fluid.layers.reduce_sum(mask_lm_loss_context, dim=1)
        mean_mask_lm_loss_context = fluid.layers.mean(mask_lm_loss_context)
        
        # extract the first token feature in each sentence
        next_sent_feat = self.get_pooled_output()
        next_sent_fc_out = fluid.layers.fc(
            input=next_sent_feat,
            size=2,
            param_attr=fluid.ParamAttr(
                name="next_sent_fc.w_0", initializer=self._param_initializer),
            bias_attr="next_sent_fc.b_0")

        next_sent_loss, next_sent_softmax = fluid.layers.softmax_with_cross_entropy(
            logits=next_sent_fc_out, label=labels, return_softmax=True)

        next_sent_acc = fluid.layers.accuracy(
            input=next_sent_softmax, label=labels)

        mean_next_sent_loss = fluid.layers.mean(next_sent_loss)
        if self._is_unidirectional:
            next_sent_weight = 0.0
            context_weight = 1.0
        else:
            next_sent_weight = 1.0
            context_weight = 1.0

        loss = mean_next_sent_loss * next_sent_weight + mean_mask_lm_loss_context * context_weight
        return next_sent_acc, mean_mask_lm_loss_context, loss
