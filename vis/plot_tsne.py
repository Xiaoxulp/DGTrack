import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import matplotlib.colors as mcolors
from collections import defaultdict
DATA_ROOT = r"D:\Domain_mot\Datatest\LoadMOTData\data"
SAMPLES_PER_ID = 1  
OUTPUT_IMG = "tsne_balanced_id_clip_dann.png"


def collect_balanced_data(data_root):
   
    search_pattern = os.path.join(data_root, "Domain*", "images", "embeddings_feature", "*.npz")
    npz_files = glob.glob(search_pattern)

    
    data_pool = defaultdict(lambda: defaultdict(list))

    print(f"Loading {len(npz_files)} files...")
    for file_path in npz_files:
        try:
            data = np.load(file_path)
            features = data['features']
            pids = data['pids']
            dids = data['dids']

            for f, p, d in zip(features, pids, dids):
                data_pool[d][p].append(f)
        except:
            continue

   
    final_feats = []
    final_pids = []
    final_dids = []

    
    for did, pid_dict in data_pool.items():
        print(f"Processing Domain {did}: Found {len(pid_dict)} IDs")

        
        for pid, feats_list in pid_dict.items():
            
            n = len(feats_list)
            if n <= 0: continue

            
            if n > SAMPLES_PER_ID:
                indices = np.random.choice(n, SAMPLES_PER_ID, replace=False)
                selected_feats = [feats_list[i] for i in indices]
            else:
                selected_feats = feats_list         
            for f in selected_feats:
                final_feats.append(f)
                final_pids.append(pid)
                final_dids.append(did)

    final_feats = np.array(final_feats)
    final_pids = np.array(final_pids)
    final_dids = np.array(final_dids)

    print(f"Total sampled: {final_feats.shape[0]}")
    return final_feats, final_pids, final_dids


def plot_tsne(features, pids, dids):
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42)
    X_tsne = tsne.fit_transform(features)
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X_tsne)

    plt.figure(figsize=(16, 7))

    
    plt.subplot(1, 2, 1)
    unique_dids = np.unique(dids)
    colors = ['r', 'g', 'b', 'c', 'm', 'y']  

    for i, domain_id in enumerate(unique_dids):
        mask = dids == domain_id
        plt.scatter(
            X_norm[mask, 0], X_norm[mask, 1],
            c=colors[i % len(colors)],
            label=f"Domain {domain_id}",
            s=15, alpha=0.6  
        )
    plt.title("Domain Distribution (Should be mixed)")
    plt.legend()

    
    plt.subplot(1, 2, 2)
    
    unique_pids = np.unique(pids)
    target_pids = np.random.choice(unique_pids, min(20, len(unique_pids)), replace=False)

    
    cmap = plt.cm.tab20

    for i, pid in enumerate(target_pids):
        mask = pids == pid
        plt.scatter(
            X_norm[mask, 0], X_norm[mask, 1],
            color=cmap(i / 20),
            label=f"ID {pid}",
            s=30, alpha=0.9  
        )
    plt.title("ID Clustering (Should be compact)")

    plt.savefig(OUTPUT_IMG, dpi=300)
    plt.show()


if __name__ == "__main__":
    feats, pids, dids = collect_balanced_data(DATA_ROOT)
    if feats is not None:
        plot_tsne(feats, pids, dids)