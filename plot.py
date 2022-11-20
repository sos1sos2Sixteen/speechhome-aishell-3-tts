import matplotlib.pyplot as plt 
import numpy as np 
from numpy import ndarray
from matplotlib.figure import Figure


def plot_alignment(align_matrix: ndarray, title: str) -> Figure: 
    '''align_matrix: (ntext, nframe)'''

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(
        align_matrix, aspect='auto', origin='lower', interpolation='none', cmap='coolwarm'
    )
    fig.colorbar(im, ax=ax)
    plt.xlabel('Decoder Steps')
    plt.ylabel('Encoder Steps')
    plt.title(title)
    plt.tight_layout()
    return fig

def plot_spectrogram(pred: ndarray, target: ndarray, title: str) -> Figure: 
    fig = plt.figure(figsize=(9, 8))
    fig.text(0.5, 0.18, title, horizontalalignment='center', fontsize=16)

    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)

    ax1.set_title('Target Mel-spectrogram')
    im = ax1.imshow(target, aspect='auto', interpolation='none', cmap='coolwarm')
    fig.colorbar(mappable=im, shrink=0.65, orientation='horizontal', ax=ax1)
    
    ax2.set_title('Predicted Mel-spectrogram')
    im = ax2.imshow(pred, aspect='auto', interpolation='none', cmap='coolwarm')
    fig.colorbar(mappable=im, shrink=0.65, orientation='horizontal', ax=ax2)

    plt.tight_layout()
    return fig

def plot_gate(gate: ndarray, title: str) -> Figure: 
    '''align_matrix: (nframe, )'''

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(gate)
    plt.xlabel('Decoder Steps')
    plt.ylabel('gate value')
    plt.title(title)
    plt.tight_layout()
    return fig

def get_data_from_figure(fig: Figure) -> ndarray: 
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    plt.close()
    return data

