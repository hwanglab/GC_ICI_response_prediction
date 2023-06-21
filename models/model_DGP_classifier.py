import numpy as np

import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score
import copy

import os

from tqdm import tqdm

# --------------------------------------------------------------------------
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

device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --------------------------------------------------------------------------


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

        w_dims = np.int_(np.append([self.n_RFs] * (self.n_GPlayers - 1), 1))

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

    # - Forward
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

        return torch.squeeze(F_out, dim=-1).t()

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


# --------------------------------------------------------------------------
# Ensemble module: run a base classifier
# --------------------------------------------------------------------------
class bin_classifier:
    def __init__(self, dataX, data_Y, setting, idx_trn=None, str_trndata=''):
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
        str_path = os.path.join(setting.save_directory, \
                                str_ensemble, 'n_layers_' + str(setting.n_layers))

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

            self.each_run = dgp_classifier(self.X, self.Y, self.setting, \
                                           trn_index=trn_index, val_index=val_index, str_filepath=str_model_save_path)

            self.each_run.model_fit()

            if self.flag_ensemble:
                np.save(str_model_save_path + '_trn_idx.py', trn_index)

        return None

    def predict(self, X_tst, idx_tst , sub_Ni=None, save_pred_path=''):
        if (save_pred_path != '') and (not os.path.exists(save_pred_path)):
            os.makedirs(save_pred_path)

        self.each_run = dgp_classifier(self.X, self.Y, self.setting)

        self.backup_nMCsamples = self.each_run.nMCsamples
        self.each_run.base_model.set_model_hypParams(nMCsamples=50)

        F_out = []
        for ith_run in range(self.n_runs):
            str_model_load_path = os.path.join\
                (self.save_model_path, "dgp_vi_" + str(ith_run))
            self.each_run.load_model(str_model_load_path)

            F_sub = self.each_run.predict(X_tst, idx_tst)
            F_out.append(F_sub)

        # -
        # n_runs = len(F_out)
        F_out = sigmoid(torch.hstack(F_out)).cpu().detach().numpy()
        return F_out, F_out


# --------------------------------------------------------------------------
# the base classifier using DGP-RFE
# --------------------------------------------------------------------------
loss_fun = lambda x: -log(torch.clamp(x, min=1e-7, max=1.0))

def approx_alpha_lik(f_est, Y_true, N, w_pos=1.0, alpha=0.5):
    f_est_singed = sigmoid(multiply(f_est, Y_true))

    if w_pos > 1.0:
        # https://discuss.pytorch.org/t/use-class-weight-with-binary-cross-entropy-loss/125265/3
        # Yes, it would be calculated as nb_neg/nb_pos = 80/20 = 4.
        class_weights = torch.ones_like(Y_true).to(device_)
        class_weights[Y_true == 1] = w_pos

        log_probs = logsumexp(-alpha*multiply(class_weights, loss_fun(f_est_singed)), dim=1)
    else:
        log_probs = logsumexp(-alpha*loss_fun(f_est_singed), dim=1)

    return -(N / (alpha*len(Y_true))) * reduce_sum(log_probs)


class dgp_classifier:
    def __init__(self, data_X, data_Y, setting, trn_index=None, val_index=None, str_filepath=None):
        self.max_epoch = setting.max_epoch

        self.batch_size = setting.batch_size

        self.regul_const = setting.regul_const

        self.nMCsamples = setting.nMCsamples
        self.alpha = setting.alpha

        # - data loader
        self.X = data_X
        self.Y = data_Y

        self.trn_index = trn_index
        self.val_index = val_index

        self.model_save_full_path = str_filepath

        self.w_pos = setting.w_pos

        # - define the model
        self.base_model = dgp_rf_(data_X.shape[1], setting)
        self.base_model.to(device_)

        if self.trn_index is not None:
            self.N = len(trn_index)
        else:
            return None

        # optimizer
        self.minimum_epoch = np.minimum(self.max_epoch - 2, setting.minimum_epoch)
        self.optimizer = torch.optim.Adam(self.base_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-07)

        return None

    def load_model(self, str_filefullpath):
        print('load the trained model: ' + str_filefullpath)

        self.base_model.load_state_dict(torch.load(str_filefullpath))
        return None

    def save_model(self, str_filefullpath):
        print('save the trained model: ' + str_filefullpath)

        torch.save(self.base_model.state_dict(), str_filefullpath)
        return None

    def model_fit(self):
        num_rep = np.int_(np.ceil(self.trn_index.size / self.batch_size))

        best_auc_val = 0.0
        for epoch in tqdm(range(self.max_epoch), desc='Training Epochs'):
            mv_trn_index = copy.deepcopy(self.trn_index)
            np.random.shuffle(mv_trn_index)

            mr_sumloss = 0.0
            mr_sumojb = 0.0
            for iter in range(num_rep):
                # load np data matrics
                index_vec = mv_trn_index \
                    [iter * self.batch_size:np.minimum(self.trn_index.size, (iter + 1) * self.batch_size)]

                X = self.X[index_vec, :]
                Y = (2 * (self.Y[index_vec] - 0.5)).astype(np.float32)

                X, Y = torch.Tensor(X).to(device_), torch.Tensor(np.reshape(Y, [-1, 1])).to(device_)

                # Zero gradients, perform a backward pass, and update the weights.
                self.optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    F_est = self.base_model(X)

                    loss = approx_alpha_lik(F_est, Y, self.N, self.w_pos)

                    mr_regul = self.base_model.regularization()
                    objective_fnc = loss + (self.regul_const * mr_regul)

                    objective_fnc.backward()
                    self.optimizer.step()

                mr_sumloss += loss.item()
                mr_sumojb += objective_fnc.item()

            if self.val_index is not None:
                F_out = self.predict(self.X, self.val_index)
                Y_est_val = reduce_mean(sigmoid(F_out), dim=1).cpu().detach().numpy()

                auc_val = roc_auc_score(self.Y[self.val_index], Y_est_val)

                print('%3d:: trn obj = (%.3f, %.3f) & val_auc = (%.3f)' % \
                      (epoch, mr_sumloss / num_rep, mr_sumojb / num_rep, auc_val))

                if (epoch > self.minimum_epoch) and (auc_val > best_auc_val):
                    best_auc_val = auc_val
                    self.save_model(self.model_save_full_path)

        # the end of the epoch loop
        if self.val_index is None:
            self.save_model(self.model_save_full_path)

        return None

    def predict(self, X_tst, idx_tst):
        with torch.no_grad():
            F_out = self.base_model(torch.Tensor(X_tst[idx_tst, :]).to(device_))

        return F_out