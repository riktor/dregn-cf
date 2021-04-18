from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm


def parse_line(l):
    if len(l) > 0:
        l = l.strip("\n").split(" ")
        try:
            items = [int(i) for i in l[1:]]
        except:
            return None, None
        user = int(l[0])
        return user, items
    else:
        return None, None


def load_ds(cfg):
    dat_split_dict = {"train": list(), "test": list()}
    pos_items_dict = {"train": dict(), "test": dict()}
    for split, data_file in zip(dat_split_dict.keys(), [cfg.train_path, cfg.test_path]):
        with open(data_file) as f:
            for l in tqdm(f.readlines()):
                user, items = parse_line(l)
                if user is not None:
                    dat_split_dict[split].append([[user] * len(items), items])
                    pos_items_dict[split][user] = (
                        pos_items_dict[split].get(user, []) + items
                    )
    train_data = np.concatenate(dat_split_dict["train"], 1).T
    test_data = np.concatenate(dat_split_dict["test"], 1).T

    return train_data, test_data, pos_items_dict


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.data_dir = data_dir
        self.split = cfg.environment.a_split
        self.folds = cfg.environment.num_a_fold
        self.mode_dict = {"train": 0, "test": 1}
        self.mode = self.mode_dict["train"]

        train_data, test_data, pos_items_dict = load_ds(cfg)
        for split in pos_items_dict:
            d = pos_items_dict[split]
            for u in tqdm(d):
                d[u] = list(set(d[u]))

        self.train_data = train_data
        self.test_data = test_data

        train_users, train_items = set(list(train_data[:, 0])), set(
            list(train_data[:, 1])
        )
        test_users, test_items = set(list(test_data[:, 0])), set(list(test_data[:, 1]))
        self.train_unique_users = list(set(train_users))
        self._n_users = len(train_users | test_users)
        self._n_items = len(train_items | test_items)
        print(self._n_users, self._n_items)

        self.Graph = None
        logger.info(f"{len(self.train_data)} interactions for training")
        logger.info(f"{len(self.test_data)} interactions for testing")
        logger.info(
            f"{cfg.dataset} Sparsity : {(len(self.train_data) + len(self.test_data)) / self.n_users / self.n_items}"
        )

        # (users,items), bipartite graph
        self.bipartite_graph = sp.csr_matrix(
            (np.ones(len(train_data)), (train_data[:, 0], train_data[:, 1])),
            shape=(self._n_users, self._n_items),
        )
        self.users_D = np.array(self.bipartite_graph.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.0] = 1
        self.items_D = np.array(self.bipartite_graph.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.0] = 1.0

        # set of positive items for each user
        self._user_positive_items_dict = pos_items_dict["train"]
        self._test_items_dict = pos_items_dict["test"]

    @property
    def n_users(self):
        return self._n_users

    @property
    def n_items(self):
        return self._n_items

    @property
    def train_users(self):
        return self.train_data[:, 0]

    @property
    def train_items(self):
        return self.train_data[:, 1]

    @property
    def train_data_size(self):
        return self.train_data.shape[0]

    @property
    def user_positive_items_dict(self):
        return self._user_positive_items_dict

    @property
    def test_items_dict(self):
        return self._test_items_dict

    def get_sparse_graph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A =
            |I,   R|
            |R^T, I|
        """
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.data_dir / "s_pre_adj_mat.npz")
                norm_adj = pre_adj_mat
            except:
                adj_mat = sp.dok_matrix(
                    (self.n_users + self.n_items, self.n_users + self.n_items),
                    dtype=np.float32,
                )
                adj_mat = adj_mat.tolil()
                R = self.bipartite_graph.tolil()
                adj_mat[: self.n_users, self.n_users :] = R
                adj_mat[self.n_users :, : self.n_users] = R.T
                adj_mat = adj_mat.todok()
                adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.0
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                sp.save_npz(self.data_dir / "s_pre_adj_mat.npz", norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.to(self.cfg.environment.device)
        return self.Graph

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.n_users + self.n_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(
                self._convert_sp_mat_to_sp_tensor(A[start:end]).to(
                    self.cfg.environment.device
                )
            )
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape)).coalesce()


class PointwiseDataset(BaseDataset):
    def __init__(self, data_dir, cfg, logger):
        BaseDataset.__init__(self, data_dir, cfg, logger)
        self.all_users = np.arange(self.n_users, dtype=np.int64)

    def __getitem__(self, idx):
        user = self.train_unique_users[idx]
        pos_items = self.user_positive_items_dict[user]
        return user, pos_items

    def __len__(self, *args):
        return len(self.train_unique_users)

    def get_loader(self):
        def collate_fn(batch):
            pos_items = np.concatenate([i for u, i in batch])

            users = np.array([u for u, i in batch])
            items = np.unique(pos_items)
            item_position_dict = dict(zip(items, range(len(items))))
            get_item_position = lambda i: item_position_dict.get(int(i), np.nan)
            item_pos_inds = np.vectorize(get_item_position)(pos_items)
            user_pos_inds = np.concatenate(
                [[u_idx] * len(pos_items) for u_idx, (_, pos_items) in enumerate(batch)]
            )

            user_pos_inds = user_pos_inds[~np.isnan(item_pos_inds)]
            item_pos_inds = item_pos_inds[~np.isnan(item_pos_inds)]

            users = torch.from_numpy(users)
            items = torch.from_numpy(items)
            user_pos_inds = torch.from_numpy(user_pos_inds)
            item_pos_inds = torch.from_numpy(item_pos_inds)
            return users, items, user_pos_inds, item_pos_inds

        loader = InfiniteDataLoader(
            self,
            shuffle=True,
            batch_size=self.cfg.optimize.batch_size,
            num_workers=self.cfg.environment.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )
        return loader


class InfiniteDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        while True:
            yield next(self.iterator)


class _RepeatSampler(object):
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
