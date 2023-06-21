import numpy as np

import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score
import os

from tqdm import tqdm

#----------------------------------------------------------------------
EPS_ = 1e-16

mat_mul = torch.matmul
mat_add = torch.add

reduce_sum = torch.sum
reduce_mean = torch.mean

multiply = torch.multiply
divide = torch.divide

transpose = torch.transpose
squeeze = torch.squeeze

exp = torch.exp
log = lambda x: torch.log(x + EPS_)
logsumexp = torch.logsumexp

square = torch.square
sqrt = lambda x: torch.sqrt(x + EPS_)

sigmoid = torch.sigmoid
logistic_loss = lambda x: log(1 + exp(-x))

# https://pytorch.org/docs/stable/generated/torch.nn.ParameterList.html
torch.set_default_dtype(torch.float32)

device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#----------------------------------------------------------------------


# --------------------------------------------------------------------------
# Deep Gaussian process with random feature expansion
# --------------------------------------------------------------------------
dropout_prob = 0.2
dropout_1mp = 1.0 - dropout_prob

class dgp_rf_(torch.nn.Module):
    def __init__(self, cnn_output_dim, setting, \
                 log_lw2=np.log(1.0), log_lb2=np.log(1.0), **kwargs):
        super(dgp_rf_, self).__init__(**kwargs)

        self.nMCsamples = setting.nMCsamples
        self.p_keep = dropout_1mp

        self.n_GPlayers = setting.n_layers
        self.n_RFs = setting.n_RFs

        self.flag_norm = True

        self.ker_type = setting.ker_type

        self.input_dim = cnn_output_dim

        self.log_lw2_val = log_lw2
        self.log_lb2_val = log_lb2

        self.flag_dropconnect = setting.flag_dropconnect

        if self.ker_type == 'rbf':
            n_factor = 2
        else:
            n_factor = 1

        omega_input_dims = np.array([self.input_dim] * self.n_GPlayers)
        omega_input_dims[1:] = 2 * self.n_RFs

        w_dims = np.int_(np.append([self.n_RFs] * (self.n_GPlayers - 1), setting.dim_output))

        self.Omegas = []
        self.W = []
        for idx in range(self.n_GPlayers):
            O_layer = nn.Linear(in_features=omega_input_dims[idx], out_features=self.n_RFs, bias=False)
            W_layer = nn.Linear(in_features=n_factor * self.n_RFs, out_features=w_dims[idx], bias=True)

            torch.nn.init.xavier_uniform_(O_layer.weight)
            torch.nn.init.xavier_uniform_(W_layer.weight)
            W_layer.bias.data.fill_(0.01)

            self.Omegas.append(O_layer)
            self.W.append(W_layer)

        self.Omegas = nn.ModuleList(self.Omegas)
        self.W = nn.ModuleList(self.W)

        self.W_skip = nn.Linear(in_features=self.input_dim, out_features=self.n_RFs, bias=True)

        self.log_sigma2 = nn.Parameter(self.log_lw2_val * torch.ones(self.n_GPlayers), requires_grad=False)
        self.log_omega2 = nn.Parameter(self.log_lb2_val * torch.ones(self.n_GPlayers), requires_grad=False)

        self.dropout = nn.Dropout(p=dropout_prob, inplace=True)
        self.ReLU = nn.ReLU(inplace=True)

        return None

    def forward(self, x):
        for mn_i in range(self.n_GPlayers):
            if mn_i == 0:
                x_in = x.repeat(self.nMCsamples, 1, 1)

                x_skip = self.W_skip(self.p_keep * self.dropout(x_in))
            else:
                x_in = F_out

            x_mdc = self.p_keep * self.dropout(x_in)
            Omega_x_ = self.Omegas[mn_i](x_mdc)

            if self.flag_norm and (mn_i > 2):
                mean_a = reduce_mean(Omega_x_, dim=-1, keepdims=True)
                std_a = sqrt(reduce_mean(square(Omega_x_ - mean_a), axis=-1, keepdims=True))
                Omega_x_ = divide(Omega_x_ - mean_a, std_a)

            if self.ker_type == 'rbf':
                factor_ = exp(self.log_sigma2[mn_i]) / np.sqrt(self.n_RFs)
                phi = factor_ * torch.concat((torch.cos(Omega_x_), torch.sin(Omega_x_)), axis=-1)
            else:
                factor_ = exp(self.log_sigma2[mn_i]) * (np.sqrt(2 / self.n_RFs))
                phi = factor_ * self.ReLU(Omega_x_)

            phi_mcd = self.p_keep * self.dropout(phi)
            F_out = self.W[mn_i](phi_mcd)

            if (mn_i < self.n_GPlayers - 1):
                F_out = torch.concat((F_out, x_skip), dim=-1)

        return F_out

    # - Rregularization
    def regularization(self):
        regu_loss = 0.0
        for mn_i in range(self.n_GPlayers):
            regu_loss += (self.p_keep * exp(self.log_omega2[mn_i]) * reduce_sum(square(self.Omegas[mn_i].weight)))
            regu_loss += (self.p_keep * reduce_sum(square(self.W[mn_i].weight)))

            regu_loss += reduce_sum(square(self.W[mn_i].bias))

        regu_loss += (self.p_keep * reduce_sum(square(self.W_skip.weight)))

        return regu_loss

    def set_model_hypParams(self, nMCsamples=None):
        if nMCsamples is not None:
            self.nMCsamples = nMCsamples

        return None


# --------------------------------------------------------------------------
# Ensemble module: run a base classifier
# --------------------------------------------------------------------------
class bin_classifier:
    def __init__(self, dataX, data_Y, setting, idx_trn=None, str_filepath=''):
        # - data loader
        self.setting = setting

        self.X = dataX
        self.Y = data_Y

        self.idx_trn = idx_trn

        self.str_costsenstive = setting.str_costsenstive

        self.flag_ensemble = setting.exp_ensemble
        if self.flag_ensemble:
            self.n_runs = setting.n_runs

            str_ensemble = 'ensemble_models/'
        else:
            self.n_runs = 1

            str_ensemble = 'single_best_models'

            if self.idx_trn is None:
                print('single best model:: the training index should be given')

        # model save
        str_path = os.path.join(str_filepath, 'Triplet_loss', str_ensemble,\
                                'n_layers_' + str(setting.n_layers))

        if not os.path.exists(str_path):
            os.makedirs(str_path)
        self.save_model_path = str_path

        return None

    def model_fit(self, ratio_trn=0.75):
        for ith_run in range(self.n_runs):
            print("--------------------------------------------------------------------------")
            if self.n_runs > 1:
                print(str(ith_run) + 'th run')

            if self.flag_ensemble:
                if self.idx_trn is None:
                    trn_index_pos = np.random.choice(\
                        np.where(self.Y == 1)[0], np.int_(ratio_trn*np.sum(self.Y == 1)), replace=False)
                    trn_index_neg = np.random.choice(\
                        np.where(self.Y == 0)[0], np.int_(ratio_trn*np.sum(self.Y == 0)), replace=False)

                    trn_index = np.sort(np.concatenate((trn_index_pos, trn_index_neg)))

                    val_index = np.setdiff1d(np.array(range(len(self.Y))), trn_index)
                else:
                    Y_ = np.reshape(self.Y[self.idx_trn], [-1])

                    trn_index_pos = np.random.choice(\
                        np.where(Y_ == 1)[0], np.int_(ratio_trn*np.sum(Y_ == 1)), replace=False)
                    trn_index_neg = np.random.choice(\
                        np.where(Y_ == 0)[0], np.int_(ratio_trn*np.sum(Y_ == 0)), replace=False)

                    trn_index = self.idx_trn[np.sort(np.concatenate((trn_index_pos, trn_index_neg)))]

                    val_index = np.setdiff1d(self.idx_trn, trn_index)
            else:
                trn_index = self.idx_trn
                val_index = None

            # train each model
            if not os.path.exists(self.save_model_path):
                os.makedirs(self.save_model_path)

            str_model_save_path = os.path.join(self.save_model_path, "dgp_vi_" + str(ith_run))

            self.each_run = Classifier_TripletLoss(self.X, self.Y, self.setting, \
                                           trn_index=trn_index, val_index=val_index, str_filepath=str_model_save_path)

            self.each_run.model_fit()

            if self.flag_ensemble:
                np.save(str_model_save_path + '_trn_idx.py', trn_index)

        return None

    def predict(self, X_tst, idx_tst, save_pred_path=''):
        if (save_pred_path != '') and (not os.path.exists(save_pred_path)):
            os.makedirs(save_pred_path)

        self.each_run = Classifier_TripletLoss(self.X, self.Y, self.setting)

        self.backup_nMCsamples = self.each_run.base_model.nMCsamples
        self.each_run.base_model.set_model_hypParams(nMCsamples=50)

        F_out1 = []
        F_out2 = []
        for ith_run in range(self.n_runs):
            str_model_load_path = os.path.join\
                (self.save_model_path, "dgp_vi_" + str(ith_run))
            self.each_run.load_model(str_model_load_path)

            F_sub1, F_sub2 = self.each_run.predict(idx_tst, X_tst)

            F_out1.append(F_sub1)
            F_out2.append(F_sub2)

        return torch.hstack(F_out1).cpu().detach().numpy(), torch.hstack(F_out2).cpu().detach().numpy()


# --------------------------------------------------------------------------
# the base classifier using DGP-RFE
# --------------------------------------------------------------------------
log_loss = lambda x: -log(torch.clamp(sigmoid(x), 1e-7, 1.0))
def approx_alpha_lik(F_out, Y_true, NpNm, alpha=0.5, loss_type='Npair_loss'):
    idx_pos = np.where(Y_true == 1.0)[0]
    idx_pos = idx_pos[1:]

    idx_neg = np.where(Y_true == 0.0)[0]

    n_pos, n_neg = np.size(idx_pos), np.size(idx_neg)

    if loss_type == 'Npair_loss':
        mat_A = torch.unsqueeze(F_out[:, 0, :], dim=-1)
        mat_P = F_out[:, idx_pos, :]
        mat_N = F_out[:, idx_neg, :]

        corr_P_A = torch.bmm(mat_P, mat_A)
        corr_N_A = torch.bmm(mat_N, mat_A).permute(0, 2, 1)

        mat_zeros = torch.zeros_like(corr_P_A)
        mat_diff = torch.concat((mat_zeros, corr_N_A - corr_P_A), dim=2)

        const_ = NpNm / (alpha * n_pos * n_neg)

        # https://discuss.pytorch.org/t/use-class-weight-with-binary-cross-entropy-loss/125265/3
        log_probs = logsumexp(-alpha*logsumexp(mat_diff, dim=-1), dim=0)

    else:
        idx_pos_ex = np.repeat(idx_pos, repeats=n_neg)
        idx_neg_ex = np.tile(idx_neg, reps=n_pos)

        mat_A = F_out[:, 0, :]
        mat_P = F_out[:, idx_pos_ex, :]
        mat_N = F_out[:, idx_neg_ex, :]

        dist_P_A = torch.squeeze(torch.cdist(mat_P, mat_A), dim=-1)
        dist_N_A = torch.squeeze(torch.cdist(mat_N, mat_A), dim=-1)

        const_ = NpNm / (alpha * n_pos * n_neg)

       # https://discuss.pytorch.org/t/use-class-weight-with-binary-cross-entropy-loss/125265/3
        log_probs = logsumexp(-alpha*log_loss(dist_N_A - dist_P_A - 1.0), dim=0)

    return -const_ * reduce_sum(log_probs)

class Classifier_TripletLoss:
    def __init__(self, data_X, data_Y, setting, trn_index=None, val_index=None, str_filepath=None):
        self.max_epoch = setting.max_epoch
        self.minimum_epoch = np.minimum(self.max_epoch - 2, setting.minimum_epoch)

        self.batch_size = setting.batch_size

        self.ker_type = setting.ker_type
        self.n_RFs = setting.n_RFs

        self.regul_const = setting.regul_const
        self.alpha = setting.alpha

        #- data loader
        self.data_X = data_X

        self.trn_index = trn_index
        self.val_index = val_index

        self.model_save_full_path = str_filepath

        # - define the model
        # __init__(self, fea_dims, num_RF, num_Att=4):
        self.base_model = dgp_rf_(data_X.shape[1], setting).to(device_)

        if self.trn_index is None:
            return None

        self.Y = np.reshape(data_Y, [-1])
        self.Ytrn = self.Y[self.trn_index]

        self.pos_idx = np.intersect1d(np.argwhere(self.Y == 1.0), self.trn_index)
        self.neg_idx = np.intersect1d(np.argwhere(self.Y == 0.0), self.trn_index)

        N_pos = np.sum(data_Y[self.trn_index]==1)
        self.NpNm = 0.5*(N_pos*(N_pos-1)*np.sum(data_Y[self.trn_index]==0))

        # optimizer
        self.optimizer = torch.optim.Adam\
            (self.base_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-07)

        self.mat_trn_est = None

        self.str_selection = 'simple'

        return None

    def model_fit(self):
        n_pos = np.int_(np.maximum(np.rint(self.batch_size / 2), 8))
        n_neg = np.minimum(self.batch_size - n_pos, 8)

        iters_Pos = len(self.pos_idx)
        mn_repeat = 5

        best_auc_val = 0.0
        for epoch in tqdm(range(0, self.max_epoch), desc='Training Epochs'):  #

            mr_sumojb = 0.0
            for iter in range(iters_Pos):
                anc_idx = self.pos_idx[iter]

                pos_idx = np.random.choice\
                    (np.setdiff1d(self.pos_idx, anc_idx), n_pos, replace=False)
                pos_idx = np.concatenate(([anc_idx], pos_idx))

                for mn_i in range(mn_repeat):
                    neg_idx = np.random.choice(self.neg_idx, n_neg, replace=False)

                    # load np data matrics
                    index_vec = np.concatenate((pos_idx, neg_idx))
                    X_, Y_ = torch.Tensor(self.data_X[index_vec, :]).to(device_), self.Y[index_vec]

                    # Zero gradients, perform a backward pass, and update the weights.
                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(True):

                        F_out = self.base_model(X_)

                        loss = approx_alpha_lik(F_out, Y_, self.NpNm)
                        mr_regul = self.base_model.regularization()

                        objective_fnc = (loss + (self.regul_const * mr_regul))

                        objective_fnc.backward()
                        self.optimizer.step()

                    mr_sumojb += objective_fnc.item()

            if self.val_index is not None:  # ((epoch % 10) == 0) | epoch > 0) & (( |
                Y_prob1, Y_prob2 = self.predict(self.val_index)
                Y_prob1 = reduce_mean(Y_prob1, dim=1).cpu().detach().numpy()
                Y_prob2 = reduce_mean(Y_prob2, dim=1).cpu().detach().numpy()

                auc_val1 = roc_auc_score(self.Y[self.val_index], Y_prob1)
                auc_val2 = roc_auc_score(self.Y[self.val_index], Y_prob2)
                print('%3d:: trn obj = (%.3f) & val_auc = (%.3f, %.3f)' % \
                      (epoch, mr_sumojb/iters_Pos, auc_val1, auc_val2))

                if (epoch > self.minimum_epoch) and (auc_val1 > best_auc_val):
                    best_auc_val = auc_val1
                    self.save_model(self.model_save_full_path)

        return None

    def load_model(self, str_filefullpath):
        print('load the trained model: ' + str_filefullpath)

        self.base_model.load_state_dict(torch.load(str_filefullpath))

        self.trn_idx_save = np.load(str_filefullpath + '_trn_idx.py.npy', allow_pickle=True)

        tmp = np.load(str_filefullpath + 'trn_embedding.npz', allow_pickle=True)
        self.Ytrn_save = tmp['Y_trn_save']
        return None

    def save_model(self, str_filefullpath):
        print('save the trained model: ' + str_filefullpath)

        torch.save(self.base_model.state_dict(), str_filefullpath)
        np.save(str_filefullpath + 'Y_trn_save.npy', Y_trn_save=self.Ytrn)
        return None

    def predict(self, tst_index, data_set_=None):
        if data_set_ is None:
            # data_filenames = self.data_filenames
            data_set_ = self.data_X

        # calcuate embeddings
        if self.trn_index is None:
            trn_index_save, Y_trn = self.trn_idx_save, self.Ytrn_save
        else:
            trn_index_save = self.trn_index

        Fout_trn = self.model_eval(trn_index_save, data_set_=self.data_X)

        #
        idx_pos = np.where(Y_trn == 1.0)[0]
        idx_neg = np.where(Y_trn == 0.0)[0]

        n_pos, n_neg = np.size(idx_pos), np.size(idx_neg)

        idx_pos_ex = np.repeat(idx_pos, repeats=n_neg)
        idx_neg_ex = np.tile(idx_neg, reps=n_pos)

        # the test data points' embeddings
        Fout_tst = self.model_eval(tst_index, data_set_=data_set_)

        with torch.no_grad():
            #- KNN classifier
            mn_K = 30

            Y_trn = torch.Tensor(Y_trn).to(device_)

            dist_mat = torch.cdist(Fout_tst, Fout_trn)

            idx_sorted = torch.argsort(dist_mat, dim=2, descending=False)
            Y_probs1 = reduce_mean(Y_trn[idx_sorted[:, :, 0:mn_K]], axis=2)

            #- AUC
            corr_mat = torch.bmm(Fout_tst, Fout_trn.permute(0, 2, 1))
            Y_probs2 = reduce_sum(corr_mat[:, :, idx_pos_ex] > corr_mat[:, :, idx_neg_ex], dim=-1)/(n_pos*n_neg)

        return Y_probs1.permute(1, 0), Y_probs2.permute(1, 0)

    def model_eval(self, tst_index, data_set_, batch_size=128):
        N_runs = np.int_(np.ceil(len(tst_index)/batch_size))

        with torch.no_grad():
            for cnt in range(N_runs):
                idx = range((cnt*batch_size), np.minimum(len(tst_index),(cnt+1)*batch_size))

                Fout_sub = self.base_model\
                    (torch.Tensor(data_set_[tst_index[idx], :]).to(device_))

                if cnt == 0:
                    Fout = Fout_sub
                else:
                    Fout = torch.concat((Fout, Fout_sub), dim=1)

        return Fout


# for mn_i in range(dist_mat.shape[0]):

# def cal_pred_probs(mat_A, mat_P, mat_N, n_select=8, n_repeat=50):
#     corr_P_A = torch.bmm(mat_P, mat_A)
#
#     n_pos, n_neg = mat_P.shape[1], mat_N.shape[1]
#     n_select = np.int_(np.minimum(n_select, n_neg/2))
#
#     prob_est = torch.zeros((n_pos, n_repeat), dtype=torch.float32)
#     for mn_i in range(n_repeat):
#         idx_sub = np.random.choice(range(n_neg), n_select, replace=False)
#
#         corr_N_A = torch.bmm(mat_N[:, idx_sub, :], mat_A).permute(0, 2, 1)
#
#         mat_zeros = torch.zeros_like(corr_P_A)
#         mat_diff = torch.concat((mat_zeros, corr_N_A - corr_P_A), dim=2)
#
#         prob_est[:, mn_i] = logsumexp(-logsumexp(mat_diff, dim=-1), dim=0)
#
#     return reduce_mean(prob_est, dim=1)

# def approx_alpha_lik(F_out, Y_true, NpNm, alpha=1.0):
#     idx_pos = np.where(Y_true == 1.0)[0]
#     idx_pos = idx_pos[1:]
#
#     idx_neg = np.where(Y_true == 0.0)[0]
#
#     n_pos, n_neg = np.size(idx_pos), np.size(idx_neg)
#
#     mat_A = torch.unsqueeze(F_out[:, 0, :], dim=-1)
#     mat_P = F_out[:, idx_pos, :]
#     mat_N = F_out[:, idx_neg, :]
#
#     corr_P_A = torch.bmm(mat_P, mat_A)
#     corr_N_A = torch.bmm(mat_N, mat_A).permute(0, 2, 1)
#
#     mat_zeros = torch.zeros_like(corr_P_A)
#     mat_diff = torch.concat((mat_zeros, corr_N_A - corr_P_A), dim=2)
#
#     const_ = NpNm / (alpha * n_pos * n_neg)
#
#    # https://discuss.pytorch.org/t/use-class-weight-with-binary-cross-entropy-loss/125265/3
#     log_probs = logsumexp(-alpha*logsumexp(mat_diff, dim=-1), dim=0)
#
#     return -const_ * reduce_sum(log_probs)


# def predict(self, tst_index, data_set_=None, flag_trndata=False):
#     if data_set_ is None:
#         # data_filenames = self.data_filenames
#         data_set_ = self.data_X
#
#     # calcuate embeddings
#     Fout_trn = self.model_eval(self.trn_index, data_set_=self.data_X)
#
#     # embeddings from the test data
#     idx_pos = np.where(self.Ytrn == 1.0)[0]
#     idx_neg = np.where(self.Ytrn == 0.0)[0]
#
#     n_pos, n_neg = np.size(idx_pos), np.size(idx_neg)
#
#     idx_pos_ex = np.repeat(idx_pos, repeats=n_neg)
#     idx_neg_ex = np.tile(idx_neg, reps=n_pos)
#
#     #
#     Y_probs = np.zeros((len(tst_index)), dtype=np.float32)
#
#     if flag_trndata:
#         # for training data points
#         for mn_i in range(len(tst_index)):
#             mat_A = torch.unsqueeze(Fout_trn[:, mn_i, :], dim=1)
#
#             if self.Ytrn[mn_i] == 1.0:
#                 idx = np.sum(self.Ytrn[range(mn_i + 1)] == 1)
#                 idx_selected = ~(idx_pos_ex == (idx - 1))
#             else:
#                 idx = np.sum(self.Ytrn[range(mn_i + 1)] == 0)
#                 idx_selected = ~(idx_neg_ex == (idx - 1))
#
#             # prediction
#             idx_pos_ex_new = idx_pos_ex[idx_selected]
#             idx_neg_ex_new = idx_neg_ex[idx_selected]
#
#             mat_P = Fout_trn[:, idx_pos_ex_new, :]
#             mat_N = Fout_trn[:, idx_neg_ex_new, :]
#
#             dist_P_A = torch.squeeze(torch.cdist(mat_P, mat_A), dim=-1)
#             dist_N_A = torch.squeeze(torch.cdist(mat_N, mat_A), dim=-1)
#
#             # https://discuss.pytorch.org/t/use-class-weight-with-binary-cross-entropy-loss/125265/3
#             prob_est = reduce_mean(sigmoid(dist_N_A - dist_P_A), dim=0)
#             Y_probs[mn_i] = \
#                 reduce_mean(filter_out_rows(prob_est)).cpu().detach().numpy()
#
#         return Y_probs
#     else:
#         # for test data points
#         Fout_tst = self.model_eval(tst_index, data_set_=data_set_)
#
#         #
#         mat_P = Fout_trn[:, idx_pos_ex, :]
#         mat_N = Fout_trn[:, idx_neg_ex, :]
#
#         for mn_i in range(len(tst_index)):
#             mat_A = torch.unsqueeze(Fout_tst[:, mn_i, :], dim=1)
#
#             dist_P_A = torch.squeeze(torch.cdist(mat_P, mat_A), dim=-1)
#             dist_N_A = torch.squeeze(torch.cdist(mat_N, mat_A), dim=-1)
#
#             prob_est = reduce_mean(sigmoid(dist_N_A - dist_P_A), dim=0)
#             Y_probs[mn_i] = \
#                 reduce_mean(filter_out_rows(prob_est)).cpu().detach().numpy()
#     return Y_probs


# def predict(self, tst_index, data_set_=None, flag_trndata=False):
#     if data_set_ is None:
#         # data_filenames = self.data_filenames
#         data_set_ = self.data_X
#
#     # calcuate embeddings
#     Fout_trn = self.model_eval(self.trn_index, data_set_=self.data_X)
#
#     # embeddings from the test data
#     idx_pos = np.where(self.Ytrn == 1.0)[0]
#     idx_neg = np.where(self.Ytrn == 0.0)[0]
#
#     Y_probs = torch.zeros((len(tst_index)), dtype=torch.float32)
#
#     if flag_trndata:
#         # for training data points
#         for mn_i in range(len(tst_index)):
#             mat_A = torch.unsqueeze(Fout_trn[:, mn_i, :], dim=-1)
#
#             if self.Ytrn[mn_i] == 1.0:
#                 idx_pos_new = np.setdiff1d(idx_pos, mn_i)
#                 idx_neg_new = idx_neg
#             else:
#                 idx_pos_new = idx_pos
#                 idx_neg_new = np.setdiff1d(idx_neg, mn_i)
#
#             # prediction
#             mat_P = Fout_trn[:, idx_pos_new, :]
#             mat_N = Fout_trn[:, idx_neg_new, :]
#
#             Y_probs[mn_i] = reduce_mean(cal_pred_probs(mat_A, mat_P, mat_N, n_select=self.Nneg_ref))
#     else:
#         # for test data points
#         Fout_tst = self.model_eval(tst_index, data_set_=data_set_)
#
#         #
#         mat_P = Fout_trn[:, idx_pos, :]
#         mat_N = Fout_trn[:, idx_neg, :]
#
#         for mn_i in range(len(tst_index)):
#             mat_A = torch.unsqueeze(Fout_tst[:, mn_i, :], dim=-1)
#             Y_probs[mn_i] = reduce_mean(cal_pred_probs(mat_A, mat_P, mat_N, n_select=self.Nneg_ref))
#
#     return Y_probs.cpu().detach().numpy()}
# def filter_out_rows(vec_in, mr_rate=0.025):
#     N_vec = len(vec_in)
#     vec_in_ = torch.sort(vec_in)[0]
#     vec_out = vec_in_[np.int_(N_vec*mr_rate):np.int_(N_vec*(1-mr_rate))]
#     return vec_out
#
# def KNN_classifier(Fout_trn, Y_trn, Fout_tst, mn_K=20):
#     dist_mat = torch.cdist(Fout_trn, Fout_tst)
#
#     for mn_i in range(dist_mat.shape[0]):
#         dist_mat_cur = dist_mat[mn_i, :, :]
#
#         idx_sorted = torch.argsort(dist_mat_cur, dim=0, descending=False)
#
#         if mn_i == 0:
#             Y_probs = reduce_sum(Y_trn[idx_sorted[0:mn_K, :]], dim=0)
#         else:
#             Y_probs += reduce_sum(Y_trn[idx_sorted[0:mn_K, :]], dim=0)
#
#     Y_probs /= (mn_K * dist_mat.shape[0])
#
#     return Y_probs
#         if flag_trndata:
#             # for training data points
#             Y_probs = np.zeros((len(tst_index)), dtype=np.float32)
#
#             for mn_i in range(len(tst_index)):
#                 mat_A = torch.unsqueeze(Fout_trn[:, mn_i, :], dim=1)
#
#                 if Y_trn[mn_i] == 1.0:
#                     idx_selected = ~np.isin(idx_pos_ex, mn_i)
#                 else:
#                     idx_selected = ~np.isin(idx_neg_ex, mn_i)
#
#                 # prediction
#                 idx_pos_ex_new = idx_pos_ex[idx_selected]
#                 idx_neg_ex_new = idx_neg_ex[idx_selected]
#
#                 mat_P = Fout_trn[:, idx_pos_ex_new, :]
#                 mat_N = Fout_trn[:, idx_neg_ex_new, :]
#
#                 dist_P_A = torch.squeeze(torch.cdist(mat_P, mat_A), dim=-1)
#                 dist_N_A = torch.squeeze(torch.cdist(mat_N, mat_A), dim=-1)
#
#                 # https://discuss.pytorch.org/t/use-class-weight-with-binary-cross-entropy-loss/125265/3
#                 prob_est = reduce_mean(sigmoid(dist_N_A - dist_P_A), dim=0)
#                 Y_probs[mn_i] = \
#                     reduce_mean(filter_out_rows(prob_est)).cpu().detach().numpy()
#
#             return Y_probs
#         else:
#             # for test data points
#             Y_trn = torch.Tensor(Y_trn).to(device_)
#
#             Fout_tst = self.model_eval(tst_index, data_set_=data_set_)
#
#             dist_mat = torch.cdist(Fout_tst, Fout_trn)
#
#             mn_K = 20
#             for mn_i in range(dist_mat.shape[0]):
#                 dist_mat_cur = dist_mat[mn_i, :, :]
#
#                 idx_sorted = torch.argsort(dist_mat_cur, dim=1, descending=False)
#
#                 Y_distMat = reduce_sum(Y_trn[idx_sorted[:, 0:mn_K]], dim=1)
#
#                 if mn_i == 0:
#                     Y_probs = Y_distMat
#                 else:
#                     Y_probs += Y_distMat
#
#             Y_probs /= (mn_K * dist_mat.shape[0])
#
#             return Y_probs.cpu().detach().numpy()