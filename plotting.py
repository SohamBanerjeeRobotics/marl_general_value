from multiprocessing.pool import ThreadPool
import plotly.express as px
import wandb
import tqdm
import pandas as pd
import os
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt


px.defaults.template = "seaborn"
sns.set_theme()

csv_path = "plots/llm.csv"

def reduce(df, x='Train Epoch', y='Distance'):
    """Reduce a bunch of runs to a mean and ci"""
    grouped = df.groupby(x)[y].agg(['mean', 'std', 'count'])
    grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])
    confidence = 0.95
    degrees_freedom = grouped['count'] - 1
    critical_value = stats.t.ppf((1 + confidence) / 2, degrees_freedom)
    grouped['ci95'] = critical_value * grouped['sem']
    grouped['Distance'] = grouped['mean']
    grouped['lower'] = grouped['mean'] - grouped['ci95']
    grouped['upper'] = grouped['mean'] + grouped['ci95']
    return grouped


def set_opacity(color_str, alpha):
    # e.g., 'rgb(100, 200, 300) -> 'rgba(100, 200, 300, 0.5)'
    return color_str.replace('rgb', 'rgba').replace(')', f', {alpha})')

def smooth(df, ys, window=10):
    df[ys] = df.groupby('run_name')[ys].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )
    return df


def plot_llm():
    def process_llm_run(run):
        df = pd.DataFrame(run.scan_history())
        df['run_name'] = run.name
        df['run_id'] = run.id
        return df


    if not os.path.exists(csv_path):
        api = wandb.Api(timeout=90)
        project = api.runs("morlmarl-llm")
        pool = ThreadPool(100)
        runs = [run for run in project]
        result = tqdm.tqdm(pool.imap_unordered(process_llm_run, runs), total=len(runs))
        # Block until all done
        result = list(result)
        df = pd.concat(result)
        df.to_csv(csv_path)
    else:
        df = pd.read_csv(csv_path)


    # Sometimes wandb will return duplicates, no clue why...
    df = df.drop_duplicates().reset_index(drop=True)
    df['Validation Loss'] = df.groupby('run_name')['eval_loss'].transform(
        lambda x: x.rolling(window=50, min_periods=1).mean()
    )
    df = df.rename(columns={"epoch": "Epoch", "run_name": "LLM"})
    fig = plt.figure(figsize=(7, 3))
    g = sns.lineplot(data=df, x='Epoch', y='Validation Loss', hue='LLM', errorbar=None)
    g.set_yscale("log")
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig("plots/llm.pdf")
    plt.show()
    #fig = px.line(df, x='Epoch', y='Validation Loss', color='LLM', log_y=True, width=1000, height=450)
    #fig.show()
    #fig.write_image("plots/llm.pdf")


def plot_loss_fn():
    ## Compare loss functions
    csv_path = "plots/loss_functions.csv"
    def process_loss_run(run):
        df = pd.DataFrame(run.scan_history())
        # Only get 1k epoch intervals
        df = df[df['eval/best_return'].notnull()]
        df['run_name'] = run.name
        df['run_id'] = run.id
        df['Loss'] = run.config['loss']
        loss_kwargs = run.config.get('loss_kwargs')
        if loss_kwargs is None:
            loss_kwargs = {}
        df['Alpha'] = loss_kwargs.get('cql_alpha', 0)
        df['Tau'] = loss_kwargs.get('weighted_tau', 0)
        return df

    if not os.path.exists(csv_path):
        api = wandb.Api(timeout=90)
        project = api.runs("morlmarl-loss")
        pool = ThreadPool(100)
        runs = [run for run in project]
        result = tqdm.tqdm(pool.imap_unordered(process_loss_run, runs), total=len(runs))
        # Block until all done
        result = list(result)
        df = pd.concat(result)
        df.to_csv(csv_path)
    else:
        df = pd.read_csv(csv_path)

    metric_keys={
        "eval/best_return": "Best Return",
        "eval/mean_return": "Return",
        "eval/mean_distance": "Distance",
        "eval/best_distance": "Best Distance",
        "eval/collisions": "Collisions per Timestep",
        "train/epoch": "Train Epoch",
    }
    df = df.rename(columns=metric_keys)
    # Subtract of velocity distance -- set goals to be 30cm radius
    df['Distance'] = np.maximum(df['Distance'] - 0.3, 0)
    # Count collisions per timestep, 5 episodes each 50s long
    df['Collisions per Timestep'] = df['Collisions per Timestep'] / (50 * 5)
    # Otherwise hue interpolates as float
    df['Tau'] = df['Tau'].astype(str)
    df['Alpha'] = df['Alpha'].astype(str)
    groups = df.groupby(['Loss'])
    weighted = groups.get_group(('weighted',)).sort_values('Tau')
    plt.figure()
    ax = sns.lineplot(weighted, x='Train Epoch', y='Distance', hue='Tau')
    ax.set_title('Soft Q Distance to Target')
    ax.set_ylim(-0.1, 2.0)
    plt.savefig("plots/sim_weighted_distance.pdf")
    plt.figure()
    ax = sns.lineplot(weighted, x='Train Epoch', y='Collisions per Timestep', hue='Tau')
    ax.set_ylim(-0.2, 5)
    ax.set_title('Soft Q Number of Collisions per Timestep')
    plt.savefig("plots/sim_weighted_collision.pdf")

    cql = groups.get_group(('cql',))
    cql = groups.get_group(('cql',)).sort_values('Alpha')
    plt.figure()
    ax = sns.lineplot(cql, x='Train Epoch', y='Distance', hue='Alpha')
    ax.legend(loc='lower right',ncol=2)
    ax.set_title('CQL Distance to Target')
    ax.set_ylim(-0.1, 2.0)
    plt.savefig("plots/sim_cql_distance.pdf")
    plt.figure()
    ax = sns.lineplot(cql, x='Train Epoch', y='Collisions per Timestep', hue='Alpha')
    ax.set_title('CQL Number of Collisions per Timestep')
    ax.set_ylim(-0.2, 5)
    plt.savefig("plots/sim_cql_collision.pdf")

    # Now plot the best cql and weighted against mean/max
    groups = df.groupby(['Loss', 'Tau', 'Alpha'])
    best_cql = groups.get_group(('cql', '0.0', '0.4'))
    #best_weighted = groups.get_group(('weighted', '2.0', '0.0'))
    best_weighted = groups.get_group(('weighted', '5.0', '0.0'))
    mean = groups.get_group(('meanq', '0.0', '0.0'))
    max = groups.get_group(('maxq', '0.0', '0.0'))
    best_df = pd.concat([best_cql, best_weighted, mean, max]).reset_index(drop=True)
    best_df = best_df.replace({'meanq': 'Mean Q', 'maxq': 'Max Q', 'cql': 'CQL', 'weighted': 'Soft Q'})
    plt.figure()
    ax = sns.lineplot(best_df, x='Train Epoch', y='Distance', hue='Loss')
    ax.set_title('Distance to Target')
    ax.set_ylim(-0.1, 2.0)
    plt.savefig("plots/sim_distance.pdf")

    plt.figure()
    ax = sns.lineplot(best_df, x='Train Epoch', y='Collisions per Timestep', hue='Loss')
    ax.set_title('Number of Collisions')
    ax.set_ylim(-0.2, 5)
    plt.savefig("plots/sim_collision.pdf")

    plt.show()
    breakpoint()

def plot_data_ablate():
    ## Compare with less data
    csv_path = "plots/data_efficiency.csv"
    def process_loss_run(run):
        df = pd.DataFrame(run.scan_history())
        # Only get 1k epoch intervals
        df = df[df['eval/best_return'].notnull()]
        df['run_name'] = run.name
        df['run_id'] = run.id
        df['Loss'] = run.config['loss']
        df['dataset'] = run.config['dataset']
        loss_kwargs = run.config.get('loss_kwargs')
        if loss_kwargs is None:
            loss_kwargs = {}
        df['Alpha'] = loss_kwargs.get('cql_alpha', 0)
        df['Tau'] = loss_kwargs.get('weighted_tau', 0)
        return df

    if not os.path.exists(csv_path):
        api = wandb.Api(timeout=90)
        project = api.runs("morlmarl-limited")
        pool = ThreadPool(100)
        runs = [run for run in project]
        result = tqdm.tqdm(pool.imap_unordered(process_loss_run, runs), total=len(runs))
        # Block until all done
        result = list(result)
        df = pd.concat(result)
        df.to_csv(csv_path)
    else:
        df = pd.read_csv(csv_path)

    metric_keys={
        "eval/best_return": "Best Return",
        "eval/mean_return": "Return",
        "eval/mean_distance": "Distance",
        "eval/best_distance": "Best Distance",
        "eval/collisions": "Collisions per Timestep",
        "train/epoch": "Train Epoch",
        "dataset": "Dataset Size (Mins)"
    }
    df = df.rename(columns=metric_keys)
    # Subtract of velocity distance -- set goals to be 30cm radius
    df['Distance'] = np.maximum(df['Distance'] - 0.3, 0)
    # Count collisions per timestep, 5 episodes each 50s long
    df['Collisions per Timestep'] = df['Collisions per Timestep'] / (50 * 5)
    # Total dataset length in mins
    datasize = 90
    df = df.sort_values('Dataset Size (Mins)')
    df = df.replace({
        'dataset-05.h5': str(datasize * 0.05),
        'dataset-10.h5': str(datasize * 0.10),
        'dataset-25.h5': str(datasize * 0.25),
        'dataset-50.h5': str(datasize * 0.50),
        'dataset-75.h5': str(datasize * 0.75),
        'dataset.h5': str(datasize)
    })
    groups = df.groupby(['Loss'])
    #weighted = groups.get_group(('weighted',)).sort_values('Dataset Size (Mins)').reset_index()
    mean = groups.get_group(('meanq',))
    plt.figure()
    ax = sns.lineplot(mean, x='Train Epoch', y='Distance', hue='Dataset Size (Mins)')
    ax.set_title('Distance to Target')
    ax.set_ylim(-0.1, 2.0)
    plt.savefig("plots/data_distance.pdf")
    plt.show()

    plt.figure()
    ax = sns.lineplot(mean, x='Train Epoch', y='Collisions per Timestep', hue='Dataset Size (Mins)')
    ax.set_title('Number of Collisions')
    ax.set_ylim(-0.2, 5)
    plt.savefig("plots/data_collision.pdf")
    plt.show()


def plot_real():
    soft_train = pd.read_csv("data/real_csv/robomaster_eval_1716209490.csv")
    dists = soft_train.groupby('robot_idx').apply(
        # Compute the distance between the state and goal per robot
        lambda x: np.maximum(0, np.linalg.norm(x[['state.pn', 'state.pe']].values - x[['goals.pn', 'goals.pe']].values, axis=-1) - 0.3)
    )
    # Now compute mean over all agents, for each timestep
    dists = dists.mean()
    print(dists)
    pos = np.stack([soft_train['state.pn'], soft_train['state.pe']], axis=-1)
    goal = np.stack([soft_train['goals.pn'], soft_train['goals.pe']], axis=-1)
    breakpoint()



if __name__ == '__main__':
    #plot_llm()
    #plot_loss_fn()
    #plot_data_ablate()
    plot_real()