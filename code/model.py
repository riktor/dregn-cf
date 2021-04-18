import random
from collections import Counter

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

torch.manual_seed(2021)


class BaseModel(nn.Module):
    def __init__(self, dataset, cfg, logger):
        super(BaseModel, self).__init__()
        self.cfg = cfg
        self.logger = logger
        self.dataset = dataset
        self.device = cfg.environment.device

    def get_users_rating(self, users):
        raise NotImplementedError


class LightGCN(BaseModel):
    def __init__(self, dataset, cfg, logger):
        super(LightGCN, self).__init__(dataset, cfg, logger)
        self._init_weight()

    def _init_weight(self):
        self.latent_dim = self.cfg.emb_dim
        self.n_layers = self.cfg.n_layers
        self.keep_prob = self.cfg.keep_prob
        self.A_split = self.cfg.environment.a_split
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.dataset.n_users, embedding_dim=self.latent_dim
        )
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.dataset.n_items, embedding_dim=self.latent_dim
        )
        nn.init.xavier_uniform_(self.embedding_user.weight, gain=0.1)
        nn.init.xavier_uniform_(self.embedding_item.weight, gain=0.1)

        self.f = nn.Sigmoid()
        self.graph = self.dataset.get_sparse_graph()

    @torch.jit.export
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    @torch.jit.export
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.graph, keep_prob)
        return graph

    @torch.jit.export
    def get_user_item_embeddings(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.cfg.dropout:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.graph
        else:
            g_droped = self.graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(
            light_out, [self.dataset.n_users, self.dataset.n_items]
        )
        return users, items

    def get_user_rating(self, users):
        all_users, all_items = self.get_user_item_embeddings()
        users_emb = all_users[users.long()]
        items_emb = all_items

        rating = torch.matmul(users_emb, items_emb.t())
        return rating

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma


class DREGN_CF(LightGCN):
    def __init__(self, dataset, cfg, logger):
        super(DREGN_CF, self).__init__(dataset, cfg, logger)
        self._estimate_constants()

    def _estimate_constants(self):
        user_freq = Counter(self.dataset.train_users)
        users = list(user_freq.keys())
        u_freqs = np.array(list(user_freq.values()))
        self.p_u = torch.from_numpy(u_freqs / u_freqs.sum())

        # compute p(i)
        item_freq = Counter(self.dataset.train_items)
        items = list(range(self.dataset.n_items))
        i_freqs = np.array([item_freq[i] for i in items])
        self.p_i = torch.from_numpy(i_freqs / i_freqs.sum())

        not_pos_users = self.dataset.n_users - i_freqs
        batch_size = self.cfg.optimize.batch_size
        p_not_sampled = np.ones(self.dataset.n_items)
        for i in range(batch_size):
            p_not_sampled *= (not_pos_users - i) / (self.dataset.n_users - i)
        p_not_sampled[p_not_sampled < 0] = 0.0
        item_sampling_prob = torch.from_numpy(1 - p_not_sampled)

        self.item_inv_prob = 1 / item_sampling_prob

    @torch.jit.export
    def get_batch_embeddings(self, users, items):
        all_users, all_items = self.get_user_item_embeddings()
        users_emb = all_users[users]
        items_emb = all_items[items]

        # for regularisation
        users_emb_ego = self.embedding_user(users)
        items_emb_ego = self.embedding_item(items)

        return (users_emb, items_emb, users_emb_ego, items_emb_ego)

    @torch.jit.export
    def get_losses(self, users, items, pos_mask, item_inv_prob, pi_p_user):
        (
            users_emb,
            items_emb,
            users_emb_ego,
            items_emb_ego,
        ) = self.get_batch_embeddings(users, items)
        reg_loss = (1 / 2) * users_emb_ego.norm(2).pow(2).div(float(len(users)))
        reg_loss += (1 / 2) * items_emb_ego.norm(2).pow(2).div(float(len(items)))

        # construct estimated desnity ratio
        r = torch.matmul(users_emb, items_emb.t())

        if self.cfg.ablation.mode == "full":
            r = torch.nn.functional.softplus(r)
            base_loss = self.get_rankingDRE_loss(r, pos_mask, item_inv_prob, pi_p_user)
        elif self.cfg.ablation.mode == "nonn":
            r = torch.nn.functional.softplus(r)
            base_loss = self.get_rankingDRE_loss_nonn(
                r, pos_mask, item_inv_prob, pi_p_user
            )
        elif self.cfg.ablation.mode == "nois":
            r = torch.nn.functional.softplus(r)
            base_loss = self.get_rankingDRE_loss_nois(r, pos_mask, pi_p_user)
        else:
            raise ValueError(f"unknown abltion mode {self.cfg.ablation.mode}")

        return base_loss, reg_loss

    @torch.jit.export
    def normalized_weighted_mean(self, elements, weights):
        s = torch.sum(elements * weights, 1)
        return s.div_(torch.sum(weights, 1))

    @torch.jit.export
    def get_rankingDRE_loss(self, r, pos_mask, item_inv_prob, pi_p_user):
        # Non-negative risk correction:  enable
        # Importance sampling estimator: enable

        # weights and masks
        item_rank_w_neg = r.detach().clone()
        item_rank_w_pos = r.detach().clone().pow_(-1.0)

        item_w_nu_pos = item_rank_w_pos.mul_(pos_mask)
        item_w_nu_neg = pos_mask.mul_(item_rank_w_neg)

        # term for non-negative risk correction
        r2 = r.pow(2)
        C = 1 / self.cfg.dr_upper_bound
        corr = self.normalized_weighted_mean(r2, item_w_nu_pos).mul_(C * (1 / 2))

        # risk for samples from p(i|u,y=+1) (p_nu)
        term_a = self.normalized_weighted_mean(r, item_w_nu_pos).neg_()
        term_a_neg = self.normalized_weighted_mean(r2, item_w_nu_pos)
        term_a_neg = term_a_neg.sub_(self.normalized_weighted_mean(r2, item_w_nu_neg))
        term_a = term_a.add_(term_a_neg.mul_(pi_p_user))
        term_a = term_a.add_(corr)

        # risk for samples from p(i|u) (p_de)
        term_b = self.normalized_weighted_mean(
            r2, item_rank_w_neg.mul_(item_inv_prob)
        ).mul_((1 / 2))
        term_b = term_b.sub_(corr)

        base_loss = torch.mean(term_a + torch.clamp(term_b, min=0))

        return base_loss

    @torch.jit.export
    def get_rankingDRE_loss_nois(self, r, pos_mask, pi_p_user):
        # Non-negative risk correction:  enable
        # Importance sampling estimator: disable
        # default setting

        # weights and masks
        item_rank_w_neg = r.detach().clone()
        item_rank_w_pos = r.detach().clone().pow_(-1)
        item_w_nu_pos = item_rank_w_pos.mul(pos_mask)
        item_w_nu_neg = pos_mask.mul_(item_rank_w_neg)

        # term for non-negative risk correction
        r2 = r.pow(2)
        C = 1 / self.cfg.dr_upper_bound
        corr = self.normalized_weighted_mean(r2, item_w_nu_pos).mul_(C * (1 / 2))

        # risk for samples from p(i|u,y=+1) (p_nu)
        term_a = self.normalized_weighted_mean(r, item_w_nu_pos).neg_()
        term_a_neg = self.normalized_weighted_mean(r2, item_w_nu_pos)
        term_a_neg = term_a_neg.sub_(self.normalized_weighted_mean(r2, item_w_nu_neg))
        term_a = term_a.add_(term_a_neg.mul_(pi_p_user))
        term_a = term_a.add_(corr)

        # risk for samples from p(i|u) (p_de)
        term_b = self.normalized_weighted_mean(r2, item_rank_w_neg).mul_((1 / 2))
        term_b = term_b.sub_(corr)
        base_loss = torch.mean(term_a + torch.clamp(term_b, min=0))

        return base_loss

    @torch.jit.export
    def get_rankingDRE_loss_nonn(self, r, pos_mask, item_inv_prob, pi_p_user):
        # Non-negative risk correction:  disable
        # Importance sampling estimator: enable

        # weights and masks
        item_rank_w_neg = r.detach().clone()
        item_rank_w_pos = r.detach().clone().pow(-1)
        item_w_nu_pos = item_rank_w_pos.mul(pos_mask)
        item_w_nu_neg = pos_mask.mul_(item_rank_w_neg)

        # term for non-negative risk correction
        # risk for samples from p(i|u,y=+1) (p_nu)
        r2 = r.pow(2)
        term_a = self.normalized_weighted_mean(r, item_w_nu_pos).neg_()
        term_a = term_a.add_(
            self.normalized_weighted_mean(r2, item_w_nu_pos).mul_(pi_p_user)
        )
        term_a = term_a.sub_(
            self.normalized_weighted_mean(r2, item_w_nu_neg).mul_(pi_p_user)
        )

        # risk for samples from p(i|u) (p_de)
        term_b = self.normalized_weighted_mean(
            r2, item_rank_w_neg.mul_(item_inv_prob)
        ).mul_((1 / 2))

        base_loss = torch.mean(term_a + term_b)

        return base_loss

    @torch.jit.export
    def compute_loss(self, epoch, n_iter, samples):
        users, items, user_pos_inds, item_pos_inds = samples

        users = users.to(self.device, non_blocking=True)
        items = items.to(self.device, non_blocking=True)

        pos_mask = torch.cuda.FloatTensor(len(users), len(items)).fill_(0)
        pos_mask[user_pos_inds, item_pos_inds] = 1.0

        item_inv_prob = self.item_inv_prob[items].to(self.device, non_blocking=True)
        pi_p_user = self.p_u[users].to(self.device, non_blocking=True)

        base_loss, reg_loss = self.get_losses(
            users, items, pos_mask, item_inv_prob, pi_p_user
        )
        loss = base_loss + reg_loss * self.cfg.reg_weight

        _base_loss = base_loss.cpu().item()
        _reg_loss = reg_loss.cpu().item()
        summary = {
            "step": f"{epoch}-{n_iter}",
            "step_name": "epoch-iter",
            "base_loss": _base_loss,
            "reg_loss": _reg_loss,
        }
        return loss, summary
