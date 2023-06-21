# https://discuss.pytorch.org/t/how-to-print-output-shape-of-each-layer/1587/3
import os

import numpy as np

import pandas as pd
import pickle

import rpy2.robjects as robjects
readRDS = robjects.r['readRDS']

from rpy2.robjects import pandas2ri
pandas2ri.activate()

import argparse

# input argument
parser = argparse.ArgumentParser()

parser.add_argument('--gpu_id', type=str, default='0')

parser.add_argument('--flag_dropconnect', type=bool, default=True)

parser.add_argument('--model_', type=int, default=1)

# training options
parser.add_argument('--regul_const', type=float, default=1.0)

parser.add_argument('--batch_size', type=int, default=16)

parser.add_argument('--max_epoch', type=int, default=50)
parser.add_argument('--minimum_epoch', type=int, default=30)

# Variational dropout
parser.add_argument('--nMCsamples', type=int, default=10)
parser.add_argument('--alpha', type=float, default=0.5)

# DGP + Random feature expansion (RF)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--n_RFs', type=int, default=50)
parser.add_argument('--dim_output', type=int, default=16)

parser.add_argument('--ker_type', type=str, default='arccosin')  # arccosin

parser.add_argument('--save_directory', type=str, default='./results')

parser.add_argument('--flag_costsensitive', type=bool, default=True)

# multi-swag or deep ensemble learning
parser.add_argument('--exp_ensemble', type=bool, default=True)
parser.add_argument('--str_ensemble', type=str, default='full_training_data')
parser.add_argument('--n_runs', type=int, default=10)

# setting = parser.parse_args()
setting, unknown = parser.parse_known_args()

# https://discuss.pytorch.org/t/running-on-specific-gpu-device/89841/3
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# Yes, but you need to make sure it is set before initializing the CUDA context. See the code below:# select a GPU
os.environ["CUDA_VISIBLE_DEVICES"] = setting.gpu_id

from sklearn.metrics import roc_auc_score

def seed_everything(seed: int):
    import random
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# DGP + CNN model
if setting.model_ == 0:
    from models.model_DGP_classifier import bin_classifier as DGP_RF_ens

    setting.n_layers = 2

    setting.max_epoch = 50
    setting.minimum_epoch = 30

    str_model = 'DGP_base_vi'

    seed_everything(11111)
else:
    from models.model_dgp_rf_WSL_triplet_MCDrop import bin_classifier as DGP_RF_ens
    # from models.model_dgp_rf_WSL_triplet_TF_VBP import bin_classifier as DGP_RF_ens

    str_model = 'DGP_Triletloss'
    setting.max_epoch = 75
    setting.minimum_epoch = 50

    setting.n_layers = 3

DATA_PATH_prefix = '../data/outputs/stomach_cancer_immunotherapy/'

def cal_unc_quntities(p_hat):
    p_bar = np.mean(p_hat, axis=1, keepdims=True)

    unc_aleat = np.mean(p_hat - np.square(p_hat), axis=1, keepdims=True)
    unc_epist = np.mean(np.square(p_hat - p_bar), axis=1, keepdims=True)

    return p_bar, unc_aleat, unc_epist

# from Saehoon Kim 11:29 AM
def write_to_pickle(fname, data, verbose=False):
    with open(fname, 'wb') as fopen:
        pickle.dump(data, fopen)

    if verbose:
        print("Saving %s done..."%fname)

    return None

def read_pickle(fname, verbose=False):
    with open(fname, 'rb') as fopen:
        data = pickle.load(fopen)

    if verbose:
        print("Loading %s done..."%fname)

    return data

def loading_data():
    df_GEmat = readRDS("./temp/ICI_GEmat.RDS")
    gene_names = list(readRDS("./temp/ICI_GEmat_rownames.RDS"))
    sample_ids = list(readRDS("./temp/ICI_GEmat_colnames.RDS"))

    df_32genes = list(readRDS("./temp/GC_32genes.RDS"))

    str_genes_old = ["FAM96A", "NCOA6IP", "TP73L"]
    str_genes_new = ["CIAO2A", "TGS1", "TP63"]

    for mn_i in range(len(str_genes_old)):
        idx_rep = df_32genes.index(str_genes_old[mn_i])
        df_32genes[idx_rep] = str_genes_new[mn_i]

    idx_genes = [gene_names.index(elm) for elm in df_32genes]
    # [gene_names[idx] for idx in idx_genes]

    Xmat = np.transpose(df_GEmat[np.array(idx_genes), :])

    # meta data
    df_meta = readRDS("./temp/ICI_meta.RDS")
    df_meta = pandas2ri.rpy2py_dataframe(df_meta)

    idx_meta = [df_meta["sample"].tolist().index(elm) for elm in sample_ids]
    df_meta_new = df_meta.iloc[idx_meta, :]

    df_tmp= pd.read_csv('./GC_ICI_bulkRNAseq_Foursubtypes_ACTA2_updated.csv')
    idx_meta_2 = [df_tmp["sample"].tolist().index(elm) for elm in sample_ids]

    df_meta_ref = pd.merge(df_meta_new, df_tmp.iloc[idx_meta_2, :], \
                           how="left", left_on="sample", right_on="sample")

    return Xmat, df_meta_ref, df_32genes

if __name__ == "__main__":
    # --- experiment settings:
    Xmat, df_meta_ref, df_32genes = loading_data()

    num_Exps = 5 # repetition

    print('the prediction model is ' + str_model)

    # evaluation
    RAUC_micro_DGP1 = np.zeros((num_Exps), dtype=float)

    Y_ = np.zeros((len(df_meta_ref)), dtype=np.int32)
    sel_idx = [idx for idx, str in enumerate(df_meta_ref["resp_x"], 0) if (str == 'R') ]
    Y_[sel_idx] = 1

    str_save_path = os.path.join('./results', 'predictions')
    if not os.path.exists(str_save_path):
        os.makedirs(str_save_path)

    if setting.flag_costsensitive:
        setting.str_costsenstive = 'cost_sens_loss'
        print("class imbalnaced: costsensitive loss")

        n_pos = np.sum(Y_ == 1)
        n_neg = len(Y_) - n_pos

        setting.w_pos = n_neg / n_pos
    else:
        setting.str_costsenstive = 'cost_equal_loss'
        print("binary classification: BCEloss")

        setting.w_pos = 1.0

    N_total = len(Y_)

    idx_tst = np.where((df_meta_ref["seqrun_x"] == "mkwon_srun1").to_numpy())[0]
    idx_trn = np.setdiff1d(np.array(range(N_total)), idx_tst)

    # train the (ensemble) models
    model_sepbest = DGP_RF_ens(Xmat, Y_, setting, idx_trn=idx_trn)
    # model_sepbest.model_fit()

    # make predictions
    Ytst_probs_V1, Ytst_probs_V2 = model_sepbest.predict(Xmat, idx_tst)

    [Ytst_bar_V1, unc_aleat_V1, unc_epist_V1] = cal_unc_quntities(Ytst_probs_V1)
    Yest_sep_bcs_V1 = 1 - 2 * np.sqrt(unc_aleat_V1 + unc_epist_V1)

    [Ytst_bar_V2, unc_aleat_V2, unc_epist_V2] = cal_unc_quntities(Ytst_probs_V2)
    Yest_sep_bcs_V2 = 1 - 2 * np.sqrt(unc_aleat_V2 + unc_epist_V2)
    # ------------------------------------------------------------------------------

    # for cnt, str_tst_data in enumerate(data_set_full):
    #     print("%s & %d & %d & %.1f" % (str_tst_data, mat_info[cnt, 0], mat_info[cnt, 1], mat_info[cnt, 2]))
    auc_val1 = roc_auc_score(Y_[idx_tst], Ytst_bar_V1)
    auc_val2 = roc_auc_score(Y_[idx_tst], Ytst_bar_V2)

    print('test AUC = (%.3f, %.3f)' % (auc_val1, auc_val2))
    print('-------')
else:
    print("End: no result")

    # df_tmp1 = pd.read_csv('D:/Download/ViT_CTransPath_CRC_STAD_Kather_TCGA_STAD_label_info.txt', delimiter='\t')
    # df_tmp1["pID_new"] = df_tmp1["pID"].str[:12]
    #
    # df_tmp2 = pd.read_csv('D:/Download/stad_tcga_pan_can_atlas_2018_clinical_data.tsv', delimiter='\t')
    # df_tmp2_sub = df_tmp2[["Patient ID", "Subtype"]]
    #
    # df_labels = df_tmp1.merge(df_tmp2_sub, how="left", left_on="pID_new", right_on="Patient ID")
    #
    # label_mat = np.zeros((len(df_labels), 3), dtype=np.int32)
    # label_mat[df_labels["label"] == "MSIMUT",     0] = 1
    # label_mat[df_labels["Subtype"] == "STAD_GS",  1] = 1
    # label_mat[df_labels["Subtype"] == "STAD_CIN", 2] = 1
    #
    # df_labels_ = pd.concat([df_labels, pd.DataFrame(label_mat)], axis=1)
    # df_labels_.columns = df_labels.columns.tolist() + ["MSI", "GS", "CIN"]
    # df_labels_.to_csv("./results/TCGA_STAD_MSI_GS_CIN_labels.txt", sep="\t", index=False)
    #
    # idx_select = np.sum(label_mat, axis=1) > 0
    #
    # df_labels_subset = df_labels_[idx_select]
    # df_labels_subset.to_csv("./results/TCGA_STAD_MSI_GS_CIN.txt", sep="\t", index=False)
