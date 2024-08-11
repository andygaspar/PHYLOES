
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300


fig_size = plt.rcParams["figure.figsize"]
scale = 1
plt.rcParams["figure.figsize"] = fig_size[0] * scale, fig_size[1]*2.5*scale


df = pd.read_csv('results/complete_dist.csv')
df['Dataset'] = df['Taxa'].astype(str) + '_' + df.dataset + ' ' + df.distance
df.sort_values(by=['Dataset'], inplace=True)
sub_p = 1
for taxa in [100, 200, 300]:
    # sub_p = 1
    df_taxa = df[df.Taxa==taxa]
    for dataset in df_taxa.Dataset.unique():
        df_dist = df[df.Dataset == dataset]
        nnis = []
        sprs = []
        nni_len = 0
        for i in range(10):
            nnis.append(literal_eval(df_dist.p_nni.iloc[i]))
            sprs.append(literal_eval(df_dist.p_spr.iloc[i]))
            nni_len = nni_len if len(nnis[i]) <= nni_len else len(nnis[i])

        nni_mat = np.zeros((10, nni_len), dtype=int)
        sprs_mat = np.zeros((10, nni_len), dtype=int)
        for i in range(10):
            nni_mat[i, :len(nnis[i])] = nnis[i]
            sprs_mat[i, :len(sprs[i])] = sprs[i]

        plt.subplot(6, 2, sub_p)
        # plt.subplot(2, 2, sub_p)
        plt.plot(nni_mat.T, color='r', linewidth=1)
        plt.plot(sprs_mat.T, color='b', linewidth=1)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(4, 4))
        plt.title(dataset)
        if taxa == 100:
            k = plt.xlim()
            locs, labels = plt.xticks()
            for lab in labels[1:]:
                lab._text = str(int(float(lab._text) + 1))
            plt.xticks(locs, labels)
            plt.xlim(k)
            pass
        else:
            locs, labels = plt.xticks()
            k = plt.xlim()
            print(nni_mat.shape, k)
            labels[1]._text = '1'
            plt.xticks(locs[1:], labels[1:])
            k = plt.xlim(k)

        plt.tight_layout()
        sub_p += 1
    # plt.savefig('results/figures/' + str(taxa) + '.eps', format='eps')
plt.savefig('results/figures/all_1200dpi.png', format='png')
# plt.show()
#
#
# for dataset in df.Dataset.unique():
#     df_dist = df[df.Dataset == dataset]
#     bests = []
#     worst = []
#     len_list = 0
#     for i in range(10):
#         bests.append(literal_eval(df_dist.best_list.iloc[i]))
#         worst.append(literal_eval(df_dist.worst_list.iloc[i]))
#         len_list = len_list if len(bests[i]) <= len_list else len(bests[i])
#
#     best_mat = np.zeros((10, len_list))
#     worst_mat = np.zeros((10, len_list))
#     for i in range(10):
#         best_mat[i, :len(bests[i])] = bests[i]
#         worst_mat[i, :len(worst[i])] = worst[i]
#
#     bm = best_mat.mean(axis=0)
#     wm = worst_mat.mean(axis=0)
#
#     plt.plot(bm, color='r')
#     plt.plot(wm, color='b')
#     plt.tight_layout()
#     plt.show
#
#
#     ()