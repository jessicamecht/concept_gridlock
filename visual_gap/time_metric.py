import os
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(font_scale = 2.5)
plt.rcParams['text.usetex'] = True
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['lines.linewidth'] = 4
matplotlib.rcParams['axes.linewidth'] = 2
matplotlib.rcParams['lines.markersize'] = 14
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['axes.xmargin'] = 0.1
matplotlib.rcParams['axes.ymargin'] = 0.1
matplotlib.rc('font', **{'weight': 'normal', 'size': 28})

from get_best_result import get_best

def get_x_y(all_logs, kind, time_horizon = 10, sampling_freq = 10.0):
    top_ten, _ = get_best(all_logs, kind, time_horizon)
    
    x = list(map(lambda x: x / sampling_freq, list(range(1, time_horizon + 1))))
    y = [ [] for _ in x ]

    for i in top_ten:
        fl = 0
        for k in range(time_horizon): 
            if 'RMSE_{}'.format(k + 1) not in i: fl += 1
        if fl > 0: continue
        
        for k in range(time_horizon):
            y[k].append(i['RMSE_{}'.format(k + 1)])

    return x, y

if __name__ == "__main__":
    base_kind = 'copy'

    for time_horizon in [ 1, 2, 5, 10 ]: # Seconds

        plt.clf()
        fig, ax = plt.subplots(1, 3, figsize = (30, 8))

        for at, (dataset, sampling_freq, lower_lim, upper_lim) in enumerate([ 
            [ "openacc", 10.0, -5, 1 ],
            [ "toyota", 10.0, -2, 1 ], 
            [ "once", 2.0, -5, 1 ]
        ]):

            BASE_PATH = "../results/{}/logs/".format(dataset)
            all_logs = os.listdir(BASE_PATH)
            all_logs = list(map(lambda x: BASE_PATH + x, all_logs))

            base_x, base_y = get_x_y(all_logs, base_kind, time_horizon = int(time_horizon * sampling_freq), sampling_freq = sampling_freq)

            for kind, fancy in [ 
                [ 'copy', 'Copy' ],
                [ 'linear', 'Linear' ],
                [ 'MLP', 'MLP' ],
                [ 'RNN', 'RNN' ],
                [ 'Transformer', 'Transformer' ],
                [ 'GapFormer', r'\textsc{GapFormer}' ],
                
				[ 'MLP_Horizon', r'MLP-\textsc{Horizon}' ],
                [ 'RNN_Horizon', r'RNN-\textsc{Horizon}' ],
                [ 'Transformer_Horizon', r'Transformer-\textsc{Horizon}' ],
                [ 'GapFormer_Horizon', r'\textsc{GapFormer-Horizon}' ],
            ]:

                x, y = get_x_y(all_logs, kind, time_horizon = int(time_horizon * sampling_freq), sampling_freq = sampling_freq)

                if kind == base_kind:
                    ax[at].plot(x, [ i[0] - base_y[at][0] for at, i in enumerate(y) ], '--', label = fancy if at == 0 else None)
                    continue

                if x is None or y is None: continue

                if len(y[0]) == 0: continue
                ax[at].plot(x, [ i[0] - base_y[at][0] for at, i in enumerate(y) ], '-', label = fancy if at == 0 else None)
                # ax[at].fill_between(
                #     x, 
                #     [ np.min(i) for i in y ],
                #     [ np.max(i) for i in y ],
                #     alpha = 0.2
                # )

            ax[at].set_xlabel("Time-Horizon (seconds)")
            ax[at].set_ylabel(r"$\Delta$ RMSE@k")
            # ax[at].yscale("log")
            ax[at].set_ylim(lower_lim, upper_lim)

            ax[at].set_title({
                "openacc": "Open-ACC",
                "toyota": "Proprietary",
                "once": "ONCE"
            }[dataset])
        
        fig.tight_layout()
        plt.subplots_adjust(bottom=0.32)
        fig.legend(loc = 8, ncol = 5)
        fig.savefig("../plots/time_metric_comparison_horizon_{}.png".format(time_horizon))
        fig.savefig("../plots/time_metric_comparison_horizon_{}.pdf".format(time_horizon))
