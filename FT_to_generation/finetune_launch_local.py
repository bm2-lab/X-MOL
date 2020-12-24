import sys
import subprocess
import commands
import os
import six
import copy
import argparse
import time
import random

from utils.stream import stream_by_running as get_stream_m
from utils.args import ArgumentGroup, print_arguments, inv_arguments
from finetune_args import parser as finetuning_parser
from extend_pos import extend_pos_emb, extend_word_emb, extend_fc_out_b

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
multip_g = ArgumentGroup(parser, "multiprocessing", 
        "start paddle training using multi-processing mode.")
multip_g.add_arg("node_ips", str, None, 
        "paddle trainer ips")
multip_g.add_arg("node_id", int, None, 
        "the trainer id of the node for multi-node distributed training.")
multip_g.add_arg("print_config", bool, True, 
        "print the config of multi-processing mode.")
multip_g.add_arg("current_node_ip", str, None, 
        "the ip of current node.")
multip_g.add_arg("split_log_path", str, "log",
        "log path for each trainer.")
multip_g.add_arg("log_prefix", str, "",
        "the prefix name of job log.")
multip_g.add_arg("nproc_per_node", int, 1, 
        "the number of process to use on each node.")
multip_g.add_arg("selected_gpus", str, "0", 
        "the gpus selected to use.")
multip_g.add_arg("training_script", str, None, "the program/script to be lauched "
        "in parallel followed by all the arguments", positional_arg=True)
multip_g.add_arg("training_script_args", str, None,
        "training script args", positional_arg=True, nargs=argparse.REMAINDER)
multip_g.add_arg("init_new_sent_embedding", bool, True, "init_new_sent_embedding")

# yapf: enable


def start_procs(args):
    procs = []
    log_fns = []

    default_env = os.environ.copy()

    node_id = args.node_id
    node_ips = [x.strip() for x in args.node_ips.split(',')]
    current_ip = args.current_node_ip
    num_nodes = len(node_ips)
    selected_gpus = [x.strip() for x in args.selected_gpus.split(',')]
    selected_gpu_num = len(selected_gpus)

    all_trainer_endpoints = ""
    
    if selected_gpu_num < args.nproc_per_node:
        for ip in node_ips:
            for i in range(selected_gpu_num):
                if all_trainer_endpoints != "":
                    all_trainer_endpoints += ","
                all_trainer_endpoints += "%s:617%d" % (ip, int(selected_gpus[i]))

        nranks = num_nodes * selected_gpu_num
    else:
        for ip in node_ips:
            for i in range(args.nproc_per_node):
                if all_trainer_endpoints != "":
                    all_trainer_endpoints += ","
                all_trainer_endpoints += "%s:617%d" % (ip, i)

        nranks = num_nodes * args.nproc_per_node
    
    gpus_per_proc = args.nproc_per_node % selected_gpu_num 
    if gpus_per_proc == 0:
        if selected_gpu_num < args.nproc_per_node:
            gpus_per_proc = 1
        else:
            gpus_per_proc =  selected_gpu_num / args.nproc_per_node
    else:
        gpus_per_proc =  selected_gpu_num / args.nproc_per_node + 1

    selected_gpus_per_proc = [selected_gpus[i:i + gpus_per_proc] for i in range(0, len(selected_gpus), gpus_per_proc)]

    if args.print_config:
        print("all_trainer_endpoints: ", all_trainer_endpoints, 
              ", node_id: ", node_id,
              ", current_ip: ", current_ip,
              ", num_nodes: ", num_nodes,
              ", node_ips: ", node_ips,
              ", gpus_per_proc: ", gpus_per_proc,
              ", selected_gpus_per_proc: ", selected_gpus_per_proc,
              ", nranks: ", nranks)

    current_env = copy.copy(default_env)
    procs = []
    cmds = []
    log_fns = []
    for i in range(0, args.nproc_per_node):
        trainer_id = node_id * args.nproc_per_node + i
        current_env.update({
            "FLAGS_selected_gpus": "%s" % ",".join([str(s) for s in selected_gpus_per_proc[i]]),
            "PADDLE_TRAINER_ID" : "%d" % trainer_id,
            "PADDLE_CURRENT_ENDPOINT": "%s:617%d" % (current_ip, i),
            "PADDLE_TRAINERS_NUM": "%d" % nranks,
            "PADDLE_TRAINER_ENDPOINTS": all_trainer_endpoints,
            "PADDLE_NODES_NUM": "%d" % num_nodes
        })

        cmd = [sys.executable, "-u",
               args.training_script] + args.training_script_args
        cmds.append(cmd)

        if args.split_log_path:
            fn = open("%s/%sjob.log.%d" % (args.split_log_path, args.log_prefix, trainer_id), "a")
            log_fns.append(fn)
            process = subprocess.Popen(cmd, env=current_env, stdout=fn, stderr=fn)
        else:
            process = subprocess.Popen(cmd, env=current_env)
        procs.append(process)

    for i in range(len(procs)):
        proc = procs[i]
        proc.wait()
        if len(log_fns) > 0:
            log_fns[i].close()
        if proc.returncode != 0:    
            raise subprocess.CalledProcessError(returncode=procs[i].returncode,
                                                cmd=cmds[i])
        else:
            print("proc %d finsh" % i)


def stream(args, lanch_args):
    #stream model list
    #stream_m = get_stream_m(args.stream_job, args.stream_cluster)
    stream_m = get_stream_m(args.stream_job)
    while len(stream_m) == 0:
        time.sleep(600)
        stream_m = get_stream_m(args.stream_job)
    
    download, tar, init_path = stream_m[-1]
    retcode, ret = commands.getstatusoutput(download)
    if not os.path.exists(init_path):
        retcode, ret = commands.getstatusoutput(tar)
    if not args.use_fp16:
        retcode, ret = commands.getstatusoutput(
                'cd ' + init_path + ' && ls *master | grep -v layer_norm | xargs rename ".master" ""'
                )
    if lanch_args.init_new_sent_embedding:
        retcode, ret = commands.getstatusoutput(
            'rm ' + init_path + '/sent_embedding'
        )

    arg_name = '--init_pretraining_params'
    val_index = -1
    if arg_name in lanch_args.training_script_args:
        val_index = lanch_args.training_script_args.index(arg_name) + 1
        lanch_args.training_script_args[val_index] = init_path
    else:
        lanch_args.training_script_args += [arg_name, init_path]
        val_index = lanch_args.training_script_args.index(arg_name) + 1
    
    main(lanch_args)

    while True:
        #updated_m = get_stream_m(args.stream_job, args.stream_cluster)
        updated_m = get_stream_m(args.stream_job)
        download, tar, init_path = updated_m[-1]
        if len(updated_m) > len(stream_m):
            retcode, ret = commands.getstatusoutput(download)
            if not os.path.exists(init_path):
                retcode, ret = commands.getstatusoutput(tar)
            if not args.use_fp16:
                retcode, ret = commands.getstatusoutput(
                        'cd ' + init_path + ' && ls *master | grep -v layer_norm | xargs rename ".master" ""'
                        )
        
            if lanch_args.init_new_sent_embedding:
                retcode, ret = commands.getstatusoutput(
                    'rm ' + init_path + '/sent_embedding'
                )
        lanch_args.training_script_args[val_index] = init_path
        main(lanch_args)
        #update
        stream_m = updated_m


def main(args):

    extend_vocab = False
    extend_fc = False

    def get_param(name):
        key = "--" + name
        if key not in args.training_script_args:
            return None
        index = args.training_script_args.index(key) + 1
        return args.training_script_args[index]

    if extend_vocab: 
        extend_word_emb(get_param("ernie_config_path"), get_param("init_pretraining_params"))
    if extend_fc:
        extend_fc_out_b(get_param("ernie_config_path"), get_param("init_pretraining_params"))

    if get_param("init_pretraining_params") and get_param("pos_emb_size"):
        extend_pos_emb(get_param("ernie_config_path"), \
            get_param("init_pretraining_params"), int(get_param("pos_emb_size")))

    rs_index = -1
    rs_name = "--random_seed"
    has_rs = False
    if rs_name in args.training_script_args:
        rs_index = args.training_script_args.index(rs_name) + 1
        if args.training_script_args[rs_index] != '-1':
            has_rs = True
    else:
        args.training_script_args += [rs_name, '-1']
        rs_index = args.training_script_args.index(rs_name) + 1

    if not has_rs:
        args.training_script_args[rs_index] = str(random.randint(0, 100000))
    if args.print_config:
        print_arguments(args)
    start_procs(args)
    if not has_rs:
        args.training_script_args[rs_index] = '-1'

if __name__ == "__main__":
    rec = 0
    lanch_args = parser.parse_args()
    finetuning_args = finetuning_parser.parse_args(
            lanch_args.training_script_args)
    
    if finetuning_args.stream_job and finetuning_args.stream_job != "":
        stream(finetuning_args, lanch_args)
    else:
        init_path = finetuning_args.init_pretraining_params 
        print("init model: %s" % init_path)
        if not finetuning_args.use_fp16:
            retcode, ret = commands.getstatusoutput(
                    'cd ' + init_path + ' && ls *master | grep -v layer_norm | xargs rename ".master" ""'
                    )

        if lanch_args.init_new_sent_embedding:
            retcode, ret = commands.getstatusoutput(
                'rm ' + init_path + '/sent_embedding'
            )

        while rec<1:
            main(lanch_args)
            if not finetuning_args.do_train:
                break
            rec += 1
