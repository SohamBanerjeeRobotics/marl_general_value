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
sns.set_context("talk")


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
    csv_path = "plots/llm2.csv"
    def process_llm_run(run):
        df = pd.DataFrame(run.scan_history())
        df['run_name'] = run.name
        df['run_id'] = run.id
        return df


    if not os.path.exists(csv_path):
        api = wandb.Api(timeout=90)
        project = api.runs("morlmarl-llm2")
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
    df['Test Distance to Goal (m)'] = df.groupby('run_name')['eval_loss'].transform(
        lambda x: x.rolling(window=50, min_periods=1).mean()
    )
    df = df.rename(columns={"epoch": "Epoch", "run_name": "LLM"})
    fig = plt.figure(figsize=(10, 5))
    g = sns.lineplot(data=df, x='Epoch', y='Test Distance to Goal (m)', hue='LLM', errorbar=None)
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
    ax.set_title('Soft Q Test Distance to Goal (m)')
    ax.set_ylim(-0.1, 2.0)
    ax.legend(loc="upper left", ncol=2)
    plt.tight_layout()
    plt.savefig("plots/sim_weighted_distance.pdf")
    plt.figure()
    ax = sns.lineplot(weighted, x='Train Epoch', y='Collisions per Timestep', hue='Tau')
    ax.set_ylim(-0.2, 5)
    ax.set_title('Soft Q Number of Collisions per Timestep')
    plt.tight_layout()
    plt.savefig("plots/sim_weighted_collision.pdf")

    cql = groups.get_group(('cql',))
    cql = groups.get_group(('cql',)).sort_values('Alpha')
    plt.figure()
    ax = sns.lineplot(cql, x='Train Epoch', y='Distance', hue='Alpha')
    ax.legend(loc='lower right',ncol=2)
    ax.set_title('CQL Test Distance to Goal (m)')
    ax.set_ylim(-0.1, 2.0)
    plt.tight_layout()
    plt.savefig("plots/sim_cql_distance.pdf")
    plt.figure()
    ax = sns.lineplot(cql, x='Train Epoch', y='Collisions per Timestep', hue='Alpha')
    ax.set_title('CQL Number of Collisions per Timestep')
    ax.set_ylim(-0.2, 5)
    plt.tight_layout()
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
    ax.set_title('Test Distance to Goal (m)')
    ax.set_ylim(-0.1, 2.0)
    plt.tight_layout()
    ax.legend(loc='upper left', ncol=2)
    plt.savefig("plots/sim_distance.pdf")

    plt.figure()
    ax = sns.lineplot(best_df, x='Train Epoch', y='Collisions per Timestep', hue='Loss')
    ax.set_title('Number of Collisions')
    ax.set_ylim(-0.2, 5)
    plt.tight_layout()
    plt.savefig("plots/sim_collision.pdf")

    plt.show()

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
        'dataset-05.h5': str(int(datasize * 0.05)) + ' mins',
        'dataset-10.h5': str(int(datasize * 0.10)) + ' mins',
        'dataset-25.h5': str(int(datasize * 0.25)) + ' mins',
        'dataset-50.h5': str(int(datasize * 0.50)) + ' mins',
        'dataset-75.h5': str(int(datasize * 0.75)) + ' mins',
        'dataset.h5': str(int(datasize)) + ' mins'
    })
    groups = df.groupby(['Loss'])
    #weighted = groups.get_group(('weighted',)).sort_values('Dataset Size (Mins)').reset_index()
    mean = groups.get_group(('meanq',))
    plt.figure()
    ax = sns.lineplot(mean, x='Train Epoch', y='Distance', hue='Dataset Size (Mins)')
    ax.set_title('Test Distance to Goal (m)')
    ax.set_ylim(-0.1, 2.0)
    ax.legend(ncol=2)
    plt.tight_layout()
    plt.savefig("plots/data_distance.pdf")
    plt.show()

    plt.figure()
    ax = sns.lineplot(mean, x='Train Epoch', y='Collisions per Timestep', hue='Dataset Size (Mins)')
    ax.set_title('Number of Collisions')
    ax.set_ylim(-0.2, 5)
    plt.tight_layout()
    plt.savefig("plots/data_collision.pdf")
    plt.show()


def plot_real():
    #soft_train = pd.read_csv("data/real_csv/robomaster_eval_1716209490.csv")
    csv_path = "/Users/smorad/code/corl_2024/processed_rosbags/"
    for trial, title, title2 in [
        ("soft_2_0", 'Soft Q Real World Distance (3 Agents)', "Soft Q Real World Collisions (3 Agents)"),
        ("mean", "Mean Q Real World Distance (3 Agents)", "Mean Q Real World Collisions (3 Agents)"),
        ("max", "Max Q Real World Distance (3 Agents)", "Max Q Real World Collisions (3 Agents)"),
        ("cql", "CQL Real World Distance (3 Agents)", "CQL Q Real World Collisions (3 Agents)"),
        ("5agent_soft_2_0", 'Soft Q Real World Distance (5 Agents)', "Soft Q Real World Collisions (5 Agents)"),
    ]:
        train_dists = [
            np.maximum(0, pd.read_csv(csv_path + trial + f"_train/robomaster_{i}/dist_to_goal/float.csv")['float'][:-1] - 0.3) for i in [1, 2, 3]
        ]
        mean_train_dists = np.mean(train_dists, axis=0)
        eval_dists = [
            np.maximum(0, pd.read_csv(csv_path + trial + f"_eval/robomaster_{i}/dist_to_goal/float.csv")['float'][:-1] - 0.3) for i in [1, 2, 3]
        ]
        mean_eval_dists = np.mean(eval_dists, axis=0)
        plt.figure()
        ax = sns.lineplot(x=np.arange(len(mean_train_dists)), y=mean_train_dists, label='Train')
        ax = sns.lineplot(x=np.arange(len(mean_eval_dists)), y=mean_eval_dists, label='Test')
        ax.legend(loc="upper left", ncol=2)
        ax.set_ylim(-0.1, 3.0)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Test Distance to Goal (m)')
        if len(mean_train_dists) > 300:
            ax.set_xticks(np.arange(0, len(mean_train_dists) + 1, 60))
        else:
            ax.set_xticks(np.arange(0, len(mean_train_dists) + 1, 30))
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(f'plots/real_world_{trial}.pdf')

        # Collisions
        train_states = [
            pd.read_csv(csv_path + trial + f"_train/robomaster_{i}/current_state/pos.csv") for i in [1, 2, 3]
        ]
        eval_states = [
            pd.read_csv(csv_path + trial + f"_eval/robomaster_{i}/current_state/pos.csv") for i in [1, 2, 3]
        ]
        plt.figure()
        for name, state in {"Train": train_states, "Test": eval_states}.items():
            common_times = np.sort(np.concatenate([df['t_msg'] for df in state]))
            state = [df.set_index('t_msg').reindex(common_times).interpolate(method='nearest', limit_direction='both').reset_index() for df in state]
            pos = np.array([[s['x'], s['y']] for s in state]).transpose(2, 0, 1)
            dist = np.linalg.norm(pos[:, :, None] - pos[:, None], axis=-1)
            # Set self dist to zero
            for i in range(len(state)):
                dist[:, i, i] = np.inf
            dist = dist.min(axis=(1,2))
            common_times = (common_times - common_times[0]) / 1e9 # To relative seconds
            ax = sns.lineplot(x=common_times, y=dist, label=name)
        ax.set_ylim(0, 3.0)
        ax.set_title(title2)
        ax.set_ylabel('Minimum Distance (m)')
        ax.set_xlabel('Time (s)')
        ax.hlines(0.3, 0, common_times[-1], linestyles='dashed', color='red')
        plt.tight_layout()
        plt.savefig(f'plots/real_world_collision_{trial}.pdf')

    plt.legend(loc='upper left', ncol=2)

    plt.show()



if __name__ == '__main__':
    #plot_llm()
    #plot_loss_fn()
    #plot_data_ablate()
    plot_real()