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


class token():
    """
    define the non_atom tokens and numeric tokens in the SMILES
    """
    def __init__(self):
        self.non_atom = ('',' ','(',')','.','-',':','=','/','\\','#','1','2','3','4','5','6','7','8','9','%10','%11','%12','%13','%14')
        self.bond = ('.','-',':','=','/','\\','#')
        self.num = ('1','2','3','4','5','6','7','8','9','%10','%11','%12','%13','%14')
        self.illegal = ('',' ','[',']')
        self.branch = ('(',')')
        self.ss = ('[SEP]','[CLS]','[UNK]','[MASK]','[PAD]')


def find_last_atom(smis, idx):
    """
    find the last linked atom before atom at the location idx of splitted SMILES
    :param smis: splitted SMILES
    :param idx: the index of focused atom
    :return: last atom index which link the atom at the location idx of splitted SMILES
    """
    tokens = token()
    non_atom = tokens.non_atom
    if idx == 0:
        return None
    else:
        flag = 0
        res = None
        for f_idx in range(1,idx+1):
            j_tk = smis[idx-f_idx]
            if j_tk == ')':
                flag += 1
            elif j_tk == '(' and flag != 0:
                flag -= 1
            if flag == 0 and j_tk not in non_atom:
                res = idx - f_idx
                break

        return res


def link_emb(smi):
    """
    build link embedding for SMILES
    return: a list that link the token in a branch to the atom where the branch starts, tokens in main chain is labeled as '*'
    """
    smis = split_smi_1(smi)
    link_emb_store = []
    pointers = ['*']
    c_pointers = '*'
    for idx, tk in enumerate(smis):
        if tk not in ('(',')'):
            link_emb_store.append(pointers[-1])
            c_pointers = '*'
        elif tk == '(':
            if idx == 0:# any SMILES should not starts with '('
                link_emb_store.append('inv')
                break
            if c_pointers == '*':
                pointers.append(str(idx-1))
                link_emb_store.append(pointers[-1])
            else:
                pointers.append(c_pointers)
                link_emb_store.append(pointers[-1])
        else :
            if len(pointers) == 1:# the number of '(' and ')' should be equivalent
                link_emb_store.append('inv')
                break
            link_emb_store.append(pointers[-1])
            c_pointers = pointers[-1]
            pointers = pointers[:-1]

    if link_emb_store[-1] == 'inv':
        return 'invalid SMILES'
    else:
        return link_emb_store


def link_emb_all(smi):
    """
    build link embedding for SMILES
    return: a list that link the token in a branch to the atom where the branch starts, tokens in main chain is labeled as '*'
    """
    smis = split_smi_1(smi)
    link_emb_store = []
    pointers = ['*']
    c_pointers = '*'
    for idx, tk in enumerate(smis):
        if tk not in ('(',')'):
            link_emb_store.append(pointers[-1])
            c_pointers = '*'
        elif tk == '(':
            if idx == 0:# any SMILES should not starts with '('
                link_emb_store.append('inv')
                break
            if c_pointers == '*':
                pointers.append(str(idx-1))
                link_emb_store.append(pointers[-1])
            else:
                pointers.append(c_pointers)
                link_emb_store.append(pointers[-1])
        else :
            if len(pointers) == 1:# the number of '(' and ')' should be equivalent
                link_emb_store.append('inv')
                break
            link_emb_store.append(pointers[-1])
            c_pointers = pointers[-1]
            pointers = pointers[:-1]

    if link_emb_store[-1] == 'inv':
        return 'invalid SMILES'
    else:
        return link_emb_store


def find_link_atom(smi, return_num=False):
    """
    find the linked atoms for each atom in the given SMILES
    :param smi: input SMILES
    :param return_num: if True, return the num of linked atoms of each atom; if False, return the index of linked atoms of each atom
    """

    smis = split_smi_1(smi)
    tokens = token()
    non_atom = tokens.non_atom
    num_pool = tokens.num
    if return_num:
        link_n = ['*' for _ in smis]
    else:
        link_n = [['*'] for _ in smis]
    pointers = {idx:[] for idx, tk in enumerate(smis) if tk not in non_atom}
    ring_close = {i:'*' for i in num_pool}
    ring_pointers = {idx:[] for idx, tk in enumerate(smis) if tk not in non_atom}
    for idx, tk in enumerate(smis):
        if tk in num_pool:
            a_idx = find_last_atom(smis, idx)
            if ring_close[tk] == '*':
                ring_close[tk] = a_idx
            else:
                ring_pointers[a_idx].append(ring_close[tk])
                ring_pointers[ring_close[tk]].append(a_idx)
            
        elif tk not in non_atom:
            if idx != 0:
                a_idx = find_last_atom(smis, idx)
                pointers[idx].append(a_idx)
                pointers[a_idx].append(idx)
    if return_num:
        for idx in pointers:
            l_n = len(pointers[idx])
            if l_n != 0:
                link_n[idx] = l_n
        for idx in ring_pointers:
            l_n = len(ring_pointers[idx])
            link_n[idx] += l_n
    else:
        for idx in pointers:
            l_n = len(pointers[idx])
            if l_n != 0:
                link_n[idx] = pointers[idx]
        for idx in ring_pointers:
            link_n[idx] += ring_pointers[idx]


    return link_n


def ring_emb_n(smi):
    """
    build ring embedding for SMILES
    the return index points to the ring closing number
    """
    num_pool = ('1','2','3','4','5','6','7','8','9','%10','%11','%12','%13')
    pointers = {i:'*' for i in num_pool}
    smis = split_smi_1(smi)
    ring_emb_store = ['*' for _ in smis]
    for idx,tk in enumerate(smis):
        if tk in num_pool:
            if pointers[tk] == '*':
                pointers[tk] = idx
            elif pointers[tk] != '*':
                ring_emb_store[pointers[tk]] = str(idx)
                ring_emb_store[idx] = str(pointers[tk])
                pointers[tk] = '*'
    
    return ring_emb_store


def ring_emb_a(smi, padding=False, token_in=False):
    """
    build ring embedding for SMILES
    the return index points to the ring closing atom
    """
    tokens = token()
    num_pool = tokens.num
    pointers = {i:'*' for i in num_pool}
    if token_in:
        smis = smi
    else:
        smis = split_smi_1(smi)
    ring_emb_store = ['*' for _ in smis]
    for idx,tk in enumerate(smis):
        if tk in num_pool:
            if pointers[tk] == '*':
                pointers[tk] = idx
            elif pointers[tk] != '*':
                ring_emb_store[pointers[tk]] = str(find_last_atom(smis,idx))
                ring_emb_store[idx] = str(find_last_atom(smis,pointers[tk]))
                pointers[tk] = '*'

    if padding:
        ring_emb_store_ex = []
        for idx, e in enumerate(ring_emb_store):
            if e == '*':
                ring_emb_store_ex.append(idx)
            else:
                ring_emb_store_ex.append(int(e))
        ring_emb_store = ring_emb_store_ex
    """
    # trick algorithm
    ring_emb = ring_emb_n(smi)
    p = 1
    for i in range(len(ring_emb)):
        if ring_emb[i].isalnum():
            ring_emb[i] = str(int(ring_emb[i])-p)
            p += 1
        else:
            p = 1
    ring_emb_store = ring_emb
    """

    return ring_emb_store


def show(smi,func):
    smis = split_smi_1(smi)
    link_e = func(smi)
    for i in range(len(smis)):
        print('{2},{0},{1}'.format(smis[i],link_e[i],i))


def type_emb(smi, token_in=False):
    """
    build type embedding for SMILES
    return: a list that indicate the character type of each token in the splited SMILES
    system symbol : 0    @ [CLS], [SEP] ...
    atom : 1    @ C, O, N, S ...
    aromatic_atom : 2    @ c, o, n, s, p, [nH] ...
    bond : 3    @ -, =, /, # ...
    num : 4    @ 1, 2, 3, %10 ...
    charge : 5    @ [N+] ...
    stereo : 6    @ [C@H], [C@@H] ...
    branch : 7    @ (, )
    illegal symbol: 8    @ [, ] ...
    """
    tokens = token()

    if token_in:
        smis = smi
    else:
        smis = split_smi_1(smi)
    type_emb_store = []
    for tk in smis:
        if tk in tokens.bond:
            type_emb_store.append(3)
        elif tk in tokens.branch:
            type_emb_store.append(7)
        elif tk in tokens.illegal:
            type_emb_store.append(8)
        elif tk in tokens.num:
            type_emb_store.append(4)
        elif ('+' in tk) or ('-' in tk):
            type_emb_store.append(5)
        elif '@' in tk:
            type_emb_store.append(6)
        elif tk.strip('[H]') in ('c','n','o','p','s'):
            type_emb_store.append(2)
        else:
            type_emb_store.append(1)
    
    return type_emb_store
