if __name__ == '__main__':
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from GenerateMPCAE import mpcsigmoid
import matplotlib.pyplot as plt
import seaborn as sns
import torch, pickle

if __name__ == '__main__':
    plot_type = 'neuron_heat_map'
    if plot_type == 'sigmoid':
        import crypten
        x = np.linspace(-5, 5, 1000)
        x1 = torch.tensor(x, dtype=torch.float)
        crypten.init()
        x1 = crypten.cryptensor(x1)
        y1 = crypten.common.functions.approximations.sigmoid(x1).get_plain_text()
        y1 = y1.numpy()
        y2 = 1.0/(1.0 + np.exp(-x))
        err = np.abs(y2-y1).sum()
        print("error is:", err)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        plt.plot(x,y1, label='Sigmoid in MPC Compiler CrypTen')
        plt.plot(x,y2, label='Sigmoid in PyTorch')
        plt.legend(loc="upper left")
        plt.xlabel("x")
        plt.ylabel("Sigmoid(x)")
        plt.savefig('./results/Sigmoid_recp_4.png', format='png')

    elif plot_type == 'neuron_heat_map':
        with open('./analyze/layer3_err.pkl', 'rb') as fp:
            err = pickle.load(fp).cpu().detach().numpy()
        err = err.flatten()
        err = np.sort(err, axis=None)[::-1]
        err = err.reshape((12, 10))
        print(err.shape)
        sns.set(font_scale=2)
        ax = sns.heatmap(err, linewidth=0.5, cmap="YlGnBu", cbar_kws={'label': 'Neuron Errors'})
        ax.set_xlabel('Neuron Index')
        ax.set_ylabel('Neuron Index')
        plt.savefig('./analyze/neuron_error.png', bbox_inches='tight')
    
    elif plot_type == 'neuron_votes':
        with open('./analyze/layer3_votes_max.pkl', 'rb') as fp:
            votes_max = pickle.load(fp)
        with open('./analyze/layer3_votes_min.pkl', 'rb') as fp:
            votes_min = pickle.load(fp)

        votes = votes_max - votes_min
        votes = votes.numpy()
        votes = np.sort(votes, axis=None)[::-1]
        votes = votes.reshape((12, 10))
        sns.set(font_scale=2)
        ax = sns.heatmap(votes, linewidth=0.5, cbar_kws={'label': 'Importance Weights'})
        ax.set_xlabel('Neuron Index')
        ax.set_ylabel('Neuron Index')
        plt.savefig('./analyze/neuron_weights.png', bbox_inches='tight')