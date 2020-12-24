import os
import numpy as np
import argparse
import paddle.fluid as fluid
import sys
import json

def init_emb(size, dim):
    program = fluid.Program()
    global_block = program.global_block()
    global_block.create_parameter(name="add_emb",
        shape=[size, dim],
        dtype='float32',
        initializer=fluid.initializer.TruncatedNormal(scale=0.02))
    place = fluid.core.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(program)
    np_value = np.array(fluid.global_scope().find_var("add_emb").get_tensor())
    return np_value

def get_emb(path, param_name="pos_embedding", org_dict_size=1, size=None):
    from_dir = path
    param_name = param_name

    program = fluid.Program()
    global_block = program.global_block()
  
    global_block.create_parameter(name=param_name, 
        shape=[org_dict_size, 768], 
        dtype='float32',
        initializer=fluid.initializer.Constant(value=0.00)) 

    place = fluid.core.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(program)

    param_path = os.path.join(from_dir, param_name)
    if os.path.exists(param_path):
        print('Load param from %s' % param_path)
        fluid.io.load_params(exe, from_dir, main_program=program, filename=param_name)
    else:
        raise IOError("%s doesn't exist" % param_path)

    np_value = np.array(fluid.global_scope().find_var(param_name).get_tensor())

    if size:
        return np_value[:size]

    return np_value

def set_emb(value, path, param_name):
    program = fluid.Program()
    global_block = program.global_block()

    global_block.create_parameter(name=param_name,
                               shape=[1, 1],
                               dtype='float32',
                               initializer=fluid.initializer.Constant(value=0.00))

    place = fluid.core.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(program)

    fluid.global_scope().find_var(param_name).get_tensor().set(value, place)

    fluid.io.save_params(exe, path, main_program=program, filename=param_name)
    print("\nWrite param to %s" % os.path.join(path, param_name))

def extend_pos_emb(config_path, init_path, new_size):

    #scope = fluid.core.Scope()
    #with fluid.scope_guard(scope):
    fin = open(config_path)
    config = json.load(fin)
    fin.close()

    if new_size <= config["max_position_embeddings"]:
        return
    
    add_size = new_size - config["max_position_embeddings"]
    add_emb = init_emb(add_size, config["hidden_size"])
    org_emb = get_emb(init_path)[:config["max_position_embeddings"]]
    new_emb = np.concatenate((org_emb, add_emb), axis=0) 

    set_emb(new_emb, init_path, "pos_embedding")

def extend_word_emb(config_path, init_path):

    #scope = fluid.core.Scope()
    #with fluid.scope_guard(scope):
    fin = open(config_path)
    config = json.load(fin)
    fin.close()

    new_dict_length = config["vocab_size"]
    org_dict_length = 112
    

    if new_dict_length <= org_dict_length:
        print('wrong dict')

    add_size = new_dict_length - org_dict_length
    add_emb = init_emb(add_size, config["hidden_size"])
    org_emb = get_emb(init_path, param_name="word_embedding", org_dict_size=org_dict_length)[:org_dict_length]
    new_emb = np.concatenate((org_emb, add_emb), axis=0)

    set_emb(new_emb, init_path, "word_embedding")


def extend_sent_emb(config_path, init_path):

    #scope = fluid.core.Scope()
    #with fluid.scope_guard(scope):
    fin = open(config_path)
    config = json.load(fin)
    fin.close()

    new_dict_length = config["type_vocab_size"]
    org_dict_length = 4


    if new_dict_length <= org_dict_length:
        print('wrong dict')

    add_size = new_dict_length - org_dict_length
    add_emb = init_emb(add_size, config["hidden_size"])
    org_emb = get_emb(init_path, param_name="sent_embedding", org_dict_size=org_dict_length)[:org_dict_length]
    new_emb = np.concatenate((org_emb, add_emb), axis=0)

    set_emb(new_emb, init_path, "sent_embedding")

#if __name__ == "__main__":
    #extend_pos_emb(sys.argv[1], sys.argv[2], int(sys.argv[3]))

