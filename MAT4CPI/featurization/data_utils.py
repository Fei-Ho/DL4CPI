import logging
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles
from sklearn.metrics import pairwise_distances
from torch.utils.data import Dataset


def load_data_from_df(
        dataset_path, add_dummy_node=True, one_hot_formal_charge=False):
    """Load and featurize data stored in a CSV file.

    Args:
        dataset_path (str): A path to the CSV file containing the data.
                            It should have two columns:
                            the first one contains SMILES strings of the
                            compounds,
                            the second one contains protein string
                            the second one contains labels.
        add_dummy_node (bool):
            If True, a dummy node will be added to the molecular graph.
            Defaults to True.
        one_hot_formal_charge (bool): If True, formal charges on atoms are
            one-hot encoded. Defaults to False.

    Returns:
        A tuple (X, y, z)
        in which X is a list of graph descriptors
        (node features, adjacency matrices, distance matrices),
        and y is a list of the corresponding labels.
        z is a list of protein embedding generate by ESM1b
    """
    data_df = pd.read_csv(dataset_path)

    data_x = data_df.iloc[:, 0].values
    data_y = data_df.iloc[:, 1].values
    data_z = data_df.iloc[:, 2].values

    if data_y.dtype == np.float64:
        data_y = data_y.astype(np.float32)

    x_all, y_all = load_data_from_smiles(
        data_x, data_y, add_dummy_node=add_dummy_node,
        one_hot_formal_charge=one_hot_formal_charge)

    return x_all, y_all, data_z


def load_data_from_smiles(x_smiles, labels,
                          add_dummy_node=True, one_hot_formal_charge=False):
    """Load and featurize data from lists of SMILES strings and labels.

    Args:
        x_smiles (list[str]): A list of SMILES strings.
        labels (list[float]): A list of the corresponding labels.
        add_dummy_node (bool): If True, a dummy node will be added to the
                               molecular graph. Defaults to True.
        one_hot_formal_charge (bool): If True, formal charges on atoms are
                               one-hot encoded. Defaults to False.

    Returns:
        A tuple (X, y) in which X is a list of graph descriptors
        (node features, adjacency matrices, distance matrices),
        and y is a list of the corresponding labels.
    """
    x_all, y_all = [], []

    for smiles, label in zip(x_smiles, labels):
        try:
            mol = MolFromSmiles(smiles)
            try:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, maxAttempts=5000)
                AllChem.UFFOptimizeMolecule(mol)
                mol = Chem.RemoveHs(mol)
            except:  # noqa
                AllChem.Compute2DCoords(mol)

            afm, adj, dist = featurize_mol(
                mol, add_dummy_node, one_hot_formal_charge)
            x_all.append([afm, adj, dist])
            y_all.append([label])
        except ValueError as e:
            logging.warning(
                'the SMILES ({}) can not be converted to a graph.\n'
                'REASON: {}'.format(smiles, e))

    return x_all, y_all


def featurize_mol(mol, add_dummy_node, one_hot_formal_charge):
    """Featurize molecule.

    Args:
        mol (rdchem.Mol): An RDKit Mol object.
        add_dummy_node (bool):
            If True, a dummy node will be added to the molecular graph.
        one_hot_formal_charge (bool):
            If True, formal charges on atoms are one-hot encoded.
    Returns:
        A tuple of molecular graph descriptors
        (node features, adjacency matrix, distance matrix).
    """
    node_features = np.array([get_atom_features(atom, one_hot_formal_charge)
                              for atom in mol.GetAtoms()])

    adj_matrix = np.eye(mol.GetNumAtoms())
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom().GetIdx()
        end_atom = bond.GetEndAtom().GetIdx()
        adj_matrix[begin_atom, end_atom] = adj_matrix[end_atom, begin_atom] = 1

    conf = mol.GetConformer()
    pos_matrix = np.array([[conf.GetAtomPosition(k).x,
                            conf.GetAtomPosition(k).y,
                            conf.GetAtomPosition(k).z]
                           for k in range(mol.GetNumAtoms())])
    dist_matrix = pairwise_distances(pos_matrix)

    if add_dummy_node:
        m = np.zeros((node_features.shape[0] + 1, node_features.shape[1] + 1))
        m[1:, 1:] = node_features
        m[0, 0] = 1.
        node_features = m

        m = np.zeros((adj_matrix.shape[0] + 1, adj_matrix.shape[1] + 1))
        m[1:, 1:] = adj_matrix
        adj_matrix = m

        m = np.full((dist_matrix.shape[0] + 1, dist_matrix.shape[1] + 1), 1e6)
        m[1:, 1:] = dist_matrix
        dist_matrix = m

    return node_features, adj_matrix, dist_matrix


def get_atom_features(atom, one_hot_formal_charge=True):
    """Calculate atom features.

    Args:
        atom (rdchem.Atom): An RDKit Atom object.
        one_hot_formal_charge (bool): If True,
        formal charges on atoms are one-hot encoded.

    Returns:
        A 1-dimensional array (ndarray) of atom features.
    """
    attributes = []

    attributes += one_hot_vector(
        atom.GetAtomicNum(),
        [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999]
    )

    attributes += one_hot_vector(
        len(atom.GetNeighbors()),
        [0, 1, 2, 3, 4, 5]
    )

    attributes += one_hot_vector(
        atom.GetTotalNumHs(),
        [0, 1, 2, 3, 4]
    )

    if one_hot_formal_charge:
        attributes += one_hot_vector(
            atom.GetFormalCharge(),
            [-1, 0, 1]
        )
    else:
        attributes.append(atom.GetFormalCharge())

    attributes.append(atom.IsInRing())
    attributes.append(atom.GetIsAromatic())

    return np.array(attributes, dtype=np.float32)


def one_hot_vector(val, lst):
    """Converts a value to a one-hot vector based on options in lst"""
    if val not in lst:
        val = lst[-1]
    return map(lambda x: x == val, lst)


class Molecule:
    """
        Class that represents a train/validation/test datum
        - self.label: 0 neg, 1 pos -1 missing for different target.
    """

    def __init__(self, x, y, z, index):
        self.node_features = x[0]
        self.adjacency_matrix = x[1]
        self.distance_matrix = x[2]
        self.protein_embed = y
        self.label = z
        self.index = index


class MolDataset(Dataset):
    """
    Class that represents a train/validation/test dataset
    that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, data_path, embed_dir, partition,
                 add_dummy_node, one_hot_formal_charge):
        """
        @param data_list: list of [smile, protein, label]
        smile: str,
        protein: str,
        label: float
        """
        data_path = data_path + '_' + partition + '.pt'
        assert partition in ['train', 'val'], \
            "please specify partition(train, val)"
        self.dataset = torch.load(data_path, map_location='cpu')
        self.embed_dir = embed_dir
        self.id_list = list(self.dataset.keys())
        self.add_dummy_node = add_dummy_node
        self.one_hot_formal_charge = one_hot_formal_charge

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_id = self.id_list[index]
        smile = self.dataset[data_id]['smiles']
        embeds = torch.load(self.embed_dir + '/' + str(data_id) + '_embeds',
                            map_location='cpu')
        protein_feat = torch.mean(embeds['pro_embed'], axis=0)
        label = torch.tensor(self.dataset[data_id]['label']).float()
        # generate node_feat, adjacency_matrix, distance_matrix
        mol = MolFromSmiles(smile)
        try:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, maxAttempts=5000)
            AllChem.UFFOptimizeMolecule(mol)
            mol = Chem.RemoveHs(mol)
        except:  # noqa
            AllChem.Compute2DCoords(mol)

        node_features, adjacency_matrix, distance_matrix = featurize_mol(
                mol, self.add_dummy_node, self.one_hot_formal_charge)

        return node_features, adjacency_matrix, distance_matrix, \
            protein_feat, label


def pad_array(array, shape, dtype=np.float32):
    """Pad a 2-dimensional array with zeros.

    Args:
        array (ndarray): A 2-dimensional array to be padded.
        shape (tuple[int]): The desired shape of the padded array.
        dtype (data-type): The desired data-type for the array.

    Returns:
        A 2-dimensional array of the given shape padded with zeros.
    """
    padded_array = np.zeros(shape, dtype=dtype)
    padded_array[:array.shape[0], :array.shape[1]] = array
    return padded_array


def data_collate_func(batch):
    """Create a padded batch of molecule features.
                       batch of protein Embeddings.

    Args:
        batch (list[Molecule], list[protein embedding]):
        A batch of raw molecules. A batch of protein embedding.
    Returns:
        A list of FloatTensors with padded molecule features:
        adjacency matrices, node features, distance matrices, and labels.
        A list of protein embedding
    """
    adjacency_list, distance_list, features_list, protein_list = [], [], [], []
    labels = []
    max_size = 0

    for node_features, adjacency_matrix, distance_matrix, protein_feat, label \
            in batch:
        labels.append(label)
        if adjacency_matrix.shape[0] > max_size:
            max_size = adjacency_matrix.shape[0]

    for node_features, adjacency_matrix, distance_matrix, protein_feat, label \
            in batch:
        adjacency_list.append(
            pad_array(adjacency_matrix, (max_size, max_size)))
        distance_list.append(
            pad_array(distance_matrix, (max_size, max_size)))
        features_list.append(
            pad_array(node_features, (max_size, node_features.shape[1])))
        protein_list.append(protein_feat)

    return torch.Tensor(adjacency_list), torch.Tensor(features_list), \
        torch.Tensor(distance_list), torch.stack(protein_list), \
        torch.Tensor(labels)


def construct_loader(data_path, embed_dir, partition, batch_size,
                     add_dummy_node=True, one_hot_formal_charge=False,
                     shuffle=True):
    """Construct a data loader for the provided data.

    Args:
        batch_size (int): The batch size.
        shuffle (bool): If True the data will be loaded in a random order.
            Defaults to True.

    Returns:
        A DataLoader object that yields batches of padded molecule features.
    """
    data_set = MolDataset(
        data_path, embed_dir, partition,
        add_dummy_node, one_hot_formal_charge)

    loader = torch.utils.data.DataLoader(
        dataset=data_set, batch_size=batch_size,
        collate_fn=data_collate_func, shuffle=shuffle, num_workers=30)
    return loader
