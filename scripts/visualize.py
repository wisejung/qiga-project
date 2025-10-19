# %%
# Import Libraries
import os

import networkx as nx
import matplotlib.pyplot as plt

"""
# %%
# Display instance graph and save if needed
def chromosome_graph(chromosome_idx: int,
                   display_flag: bool = True,
                   save_flag: bool = False):

    node_colors = list()
    for node in self.graph.nodes:
        if self.graph.nodes[node]['terminal']:
            if node not in list(nx.isolates(self.graph)):
                node_colors.append('blue')
            else:
                node_colors.append('red')
        else:
            node_colors.append('grey')
    # edge_labels = nx.get_edge_attributes(instance.graph, 'weight')

    nx.draw(self.graph, 
            self.instance.pos,
            node_color=node_colors,
            node_size=20,
            font_size=2,)

    if save_flag:
        results_dir = os.path.join(os.getcwd(), 'results')
        file_name = f'{self.instance.timestamp}_graph_{self.instance.data_name}_{chromosome_idx:03d}.png'
        plt.savefig(os.path.join(results_dir, file_name), dpi=300, bbox_inches="tight")

    if display_flag:
        plt.show()
    """

import pandas as pd
import matplotlib.pyplot as plt

def plot_2d_scatter(file_path: str,
                    x_col: str, y_col: str):

    df = pd.read_csv(file_path)
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Columns '{x_col}' and/or '{y_col}' not found in file.")

    x = df[x_col]
    y = df[y_col]

    plt.style.use('seaborn-v0_8-whitegrid')  # clean base
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.0,
        'grid.color': '#E0E0E0',
        'grid.linewidth': 0.6,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
        'figure.dpi': 150,
    })

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y,
               s=30,
               color='tab:blue',
               edgecolor='black',
               linewidth=0.5,
               alpha=0.8)

    # Labels and title
    ax.set_xlabel(x_col, labelpad=6)
    ax.set_ylabel(y_col, labelpad=6)
    ax.set_title(f"{x_col} vs {y_col}", pad=10)

    # Grid and layout
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.show()


plot_2d_scatter(file_path="/Users/jjung404/Projects/qiga-nc/results/se03_20251017-0212/chromosome.csv",
                x_col="gen", y_col="edge_density")