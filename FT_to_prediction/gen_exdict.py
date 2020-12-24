import sys

def split_smi_1(smi, rm_d=False):
    """
        split SMILES to tokens
        function type 1
        according to the character in SMILES, character within [] and two-digit number %** is regarded as one token
        :param smi: SMILES to be splitted
        :param rm_d: True when just extract tokens, the duplicate token will be removed, and False when coding the SMILES
        :return: tokens of the SMILES
    """
    tokens = []
    dc_a = ('l','r')
    marker = 0 # 1,2 indicates that the last token is not complete
    for c in smi:
        if marker == 0:
            if c in dc_a:
                tokens[-1] += c
            else:
                tokens.append(c)
                if c == '[':
                    marker = 1
                elif c == '%':
                    marker = 2
                    marker_l = 2 #indicates that remain length of %**
        else:
            tokens[-1] += c
            if marker == 1:
                if c == ']':
                    marker = 0
            elif marker == 2:
                marker_l -= 1
                if marker_l == 0:
                    marker = 0
    if rm_d:
        tokens = list(set(tokens))

    return tokens

def read_file(ft_data_dir):
    data = []
    with open(ft_data_dir+'/all_data','r') as f:
        for l in f:
            data.append(l.strip().split(','))
    return data

def get_tokens(data):
    smis = [i[0] for i in data]
    tk_lib = []
    for smi in smis:
        tk = split_smi_1(smi, rm_d=True)
        tk_lib += tk
        tk_lib = list(set(tk_lib))
    return sorted(tk_lib)

def read_token():
    with open('./package/mol/molecule_dict_f','r') as f:
        pre_token = []
        for l in f:
            pre_token.append(l.split('\t')[0])
    return pre_token

def gen_vocab_dict(task):
    ft_data_dir = './package/task_data/'+task
    ft_data = read_file(ft_data_dir)
    ft_tokens = get_tokens(ft_data)
    pre_tokens = read_token()
    add_tokens = [i for i in ft_tokens if i not in pre_tokens]
    new_tokens = pre_tokens + sorted(add_tokens)
    with open('./package/mol/molecule_dict_ft','w') as f:
        for idx, tk in enumerate(new_tokens):
            f.write('{0}\t{1}\n'.format(tk, idx))
    print('new dict amount : {0}'.format(len(new_tokens)))


if __name__ == '__main__':
   task = sys.stdin.readline().strip()
   gen_vocab_dict(task)
