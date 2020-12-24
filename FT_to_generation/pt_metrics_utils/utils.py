from rdkit import Chem
from rdkit.Chem import MACCSkeys
from random import shuffle
import numpy as np
import pandas as pd
#import torch
import scipy
from scipy import sparse
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as Morgan
from functools import partial
import os
#print(os.listdir())
_mcf = pd.read_csv('pt_metrics_utils/mcf.csv')
_pains = pd.read_csv('pt_metrics_utils/wehi_pains.csv', names=['smarts', 'names'])
_filters = [Chem.MolFromSmarts(x) for x in _mcf.append(_pains, sort=True)['smarts'].values]




def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    else:
        return smiles_or_mol


"""
def average_agg_tanimoto(stock_vecs, gen_vecs,
                         batch_size=5000, agg='max',
                         device='cpu', p=1):
    #
    #For each molecule in gen_vecs finds closest molecule in stock_vecs.
    #Returns average tanimoto score for between these molecules

    #Parameters:
        #stock_vecs: numpy array <n_vectors x dim>
        #gen_vecs: numpy array <n_vectors' x dim>
        #agg: max or mean
        #p: power for averaging: (mean x^p)^(1/p)
    #
    assert agg in ['max', 'mean'], "Can aggregate only max or mean"
    agg_tanimoto = np.zeros(len(gen_vecs))
    total = np.zeros(len(gen_vecs))
    for j in range(0, stock_vecs.shape[0], batch_size):
        x_stock = torch.tensor(stock_vecs[j:j + batch_size]).to(device).float()
        for i in range(0, gen_vecs.shape[0], batch_size):
            y_gen = torch.tensor(gen_vecs[i:i + batch_size]).to(device).float()
            y_gen = y_gen.transpose(0, 1)
            tp = torch.mm(x_stock, y_gen)
            jac = (tp / (x_stock.sum(1, keepdim=True) +
                         y_gen.sum(0, keepdim=True) - tp)).cpu().numpy()
            jac[np.isnan(jac)] = 1
            if p != 1:
                jac = jac**p
            if agg == 'max':
                agg_tanimoto[i:i + y_gen.shape[1]] = np.maximum(
                    agg_tanimoto[i:i + y_gen.shape[1]], jac.max(0))
            elif agg == 'mean':
                agg_tanimoto[i:i + y_gen.shape[1]] += jac.sum(0)
                total[i:i + y_gen.shape[1]] += jac.shape[0]
    if agg == 'mean':
        agg_tanimoto /= total
    if p != 1:
        agg_tanimoto = (agg_tanimoto)**(1/p)
    return np.mean(agg_tanimoto)
    """

def fingerprint(smiles_or_mol, fp_type='maccs', dtype=None, morgan__r=2,
                morgan__n=1024, *args, **kwargs):
    """
    Generates fingerprint for SMILES
    If smiles is invalid, returns None
    Returns numpy array of fingerprint bits

    Parameters:
        smiles: SMILES string
        type: type of fingerprint: [MACCS|morgan]
        dtype: if not None, specifies the dtype of returned array
    """

    fp_type = fp_type.lower()
    molecule = get_mol(smiles_or_mol, *args, **kwargs)
    if molecule is None:
        return None
    if fp_type == 'maccs':
        keys = MACCSkeys.GenMACCSKeys(molecule)
        keys = np.array(keys.GetOnBits())
        fingerprint = np.zeros(166, dtype='uint8')
        if len(keys) != 0:
            fingerprint[keys - 1] = 1  # We drop 0-th key that is always zero
    elif fp_type == 'morgan':
        fingerprint = np.asarray(Morgan(molecule, morgan__r, nBits=morgan__n),
                                 dtype='uint8')
    else:
        raise ValueError("Unknown fingerprint type {}".format(fp_type))
    if dtype is not None:
        fingerprint = fingerprint.astype(dtype)
    return fingerprint


def fingerprints(smiles_mols_array, already_unique=False, *args,
                 **kwargs):
    '''
    Computes fingerprints of smiles np.array/list/pd.Series with n_jobs workers
    e.g.fingerprints(smiles_mols_array, type='morgan', n_jobs=10)
    Inserts np.NaN to rows corresponding to incorrect smiles.
    IMPORTANT: if there is at least one np.NaN, the dtype would be float
    Parameters:
        smiles_mols_array: list/array/pd.Series of smiles or already computed
            RDKit molecules
        n_jobs: number of parralel workers to execute
        already_unique: flag for performance reasons, if smiles array is big
            and already unique. Its value is set to True if smiles_mols_array
            contain RDKit molecules already.
    '''
    if isinstance(smiles_mols_array, pd.Series):
        smiles_mols_array = smiles_mols_array.values
    else:
        smiles_mols_array = np.asarray(smiles_mols_array)
    if not isinstance(smiles_mols_array[0], str):
        already_unique = True

    if not already_unique:
        smiles_mols_array, inv_index = np.unique(smiles_mols_array,
                                                 return_inverse=True)

    fps = [fingerprint(i, *args, **kwargs) for i in smiles_mols_array]
    length = 1
    for fp in fps:
        if fp is not None:
            length = fp.shape[-1]
            first_fp = fp
            break
    fps = [fp if fp is not None else np.array([np.NaN]).repeat(length)[None, :]
           for fp in fps]
    if sparse.issparse(first_fp):
        fps = sparse.vstack(fps).tocsr()
    else:
        fps = np.vstack(fps)
    if not already_unique:
        return fps[inv_index]
    else:
        return fps


def mol_passes_filters(smi,
                       isomericSmiles=False):
    """
    Checks if mol
    * passes MCF and PAINS filters,
    * has only allowed atoms
    * is not charged
    """

    #allowed = {'C', 'N', 'S', 'O', 'F', 'Cl', 'Br', 'H'}
    try:
        mol = Chem.MolFromSmiles(smi)
    except:
        return False
    else:
        if mol == None:
            return False
    
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() != 0 and any(
            len(x) >= 8 for x in ring_info.AtomRings()
    ):
        return False
    h_mol = Chem.AddHs(mol)
    if any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()):
        return False
    """
    if any(atom.GetSymbol() not in allowed for atom in mol.GetAtoms()):
        return False
    """
    if any(h_mol.HasSubstructMatch(smarts) for smarts in _filters):
        return False
    smiles = Chem.MolToSmiles(mol, isomericSmiles=isomericSmiles)
    if smiles is None or len(smiles) == 0:
        return False
    if Chem.MolFromSmiles(smiles) is None:
        return False
    return True
