
import torch
import numpy as np
import pandas as pd
import random
import os

from build_graph import load_graph
from gnn_model import UrbanGNN

# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================
SCAN_POOL_SIZE = 800       # Scan 800 nodes (High capture rate)
TOP_N_TO_OPTIMIZE = 3      # Top 3 Champions per category
ENSEMBLE_RUNS = 3          # Run optimization 3 times per node for stability

# GA Settings
POPULATION_SIZE = 100     
GENERATIONS = 50
MUTATION_RATE = 0.3       
ELITISM_COUNT = 5

# BOUNDS (Unified Synergistic Strategy)
BOUNDS = {
    "NDVI": (1.0, 4.0),      # Target Greenery
    "Nb_AHI": (0.5, 3.5)     # Neighbor Height
}

FEATURE_MAP = {"NDVI": 8, "NB_AHI": 5}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# ==============================================================================
# 1. OPTIMIZER CLASS
# ==============================================================================
class GeneticOptimizer:
    def __init__(self, model, data, target_node_idx, feature_indices, mode="Hub"):
        self.model = model
        self.data = data
        self.target_idx = target_node_idx
        self.feat_indices = feature_indices 
        self.mode = mode 
        
        self.model.eval()
        with torch.no_grad(): 
            preds = self.model(self.data.x, self.data.edge_index).cpu().numpy()
        
        # Baselines
        self.base_epi = preds[target_node_idx, 0]
        self.base_spi = preds[target_node_idx, 1]
        
        # Neighbor Baselines
        edge_index = data.edge_index.cpu().numpy()
        self.neighbors = edge_index[1][edge_index[0] == target_node_idx]
        
        if len(self.neighbors) > 0:
            self.n_epi_old = preds[self.neighbors, 0]
            self.n_spi_old = preds[self.neighbors, 1]
        else:
            self.n_epi_old = np.array([])
            self.n_spi_old = np.array([])

    def _calculate_fitness(self, genome):
        x_mod = self.data.x.clone()
        x_mod[self.target_idx, self.feat_indices["NDVI"]] += genome[0]
        x_mod[self.target_idx, self.feat_indices["NB_AHI"]] += genome[1]

        with torch.no_grad():
            pred_new = self.model(x_mod, self.data.edge_index).cpu().numpy()

        d_epi = self.base_epi - pred_new[self.target_idx, 0] # Pos=Good
        d_spi = pred_new[self.target_idx, 1] - self.base_spi # Pos=Good
        
        spill_epi = 0
        spill_spi = 0
        if len(self.neighbors) > 0:
            spill_epi = np.sum(self.n_epi_old - pred_new[self.neighbors, 0])
            spill_spi = np.sum(pred_new[self.neighbors, 1] - self.n_spi_old)

        # Fitness Logic
        if self.mode == "Hub":
            # Maximize Total Cooling (Local + Spillover)
            total_cooling = d_epi + spill_epi
            score = (total_cooling * 10.0) + (d_spi * 1.0)
        else:
            # Maximize Total Vitality
            total_vitality = d_spi + spill_spi
            score = (total_vitality * 10.0) + (d_epi * 2.0)

        return score, d_epi, d_spi, spill_epi, spill_spi

    def run(self):
        population = []
        for _ in range(POPULATION_SIZE):
            ind = [
                random.uniform(BOUNDS["NDVI"][0], BOUNDS["NDVI"][1]),
                random.uniform(BOUNDS["Nb_AHI"][0], BOUNDS["Nb_AHI"][1])
            ]
            population.append(ind)

        best_genome = None
        best_stats = None 
        
        for gen in range(GENERATIONS):
            scores = []
            valid_results = []
            
            for ind in population:
                fit, le, ls, se, ss = self._calculate_fitness(ind)
                scores.append(fit)
                valid_results.append((le, ls, se, ss))
            
            ranked_indices = np.argsort(scores)[::-1]
            population = [population[i] for i in ranked_indices]
            
            best_genome = population[0]
            best_stats = valid_results[ranked_indices[0]]
            
            survivors = population[:POPULATION_SIZE // 2]
            new_pop = survivors[:ELITISM_COUNT]
            
            while len(new_pop) < POPULATION_SIZE:
                parent = random.choice(survivors)
                child = list(parent)
                if random.random() < MUTATION_RATE:
                    idx = random.randint(0, 1)
                    child[idx] += random.gauss(0, 0.5)
                    key = "NDVI" if idx == 0 else "Nb_AHI"
                    child[idx] = max(BOUNDS[key][0], min(BOUNDS[key][1], child[idx]))
                new_pop.append(child)
            population = new_pop

        return best_genome, best_stats

# ==============================================================================
# 2. CHAMPION SCANNER
# ==============================================================================
def scan_champions(model, data, pool, feat_indices, mode):
    candidates = []
    x_mod = data.x.clone()
    probe_val = 3.0
    
    for idx in pool:
        x_probe = x_mod.clone()
        x_probe[idx, feat_indices["NDVI"]] += probe_val
        x_probe[idx, feat_indices["NB_AHI"]] += probe_val
        
        with torch.no_grad():
            pred_old = model(data.x, data.edge_index).cpu().numpy()
            pred_new = model(x_probe, data.edge_index).cpu().numpy()
        
        d_epi = -(pred_new[idx, 0] - pred_old[idx, 0])
        d_spi = pred_new[idx, 1] - pred_old[idx, 1]
        
        edge_index = data.edge_index.cpu().numpy()
        neighbors = edge_index[1][edge_index[0] == idx]
        
        spill_epi = 0
        spill_spi = 0
        if len(neighbors) > 0:
            spill_epi = np.sum(pred_old[neighbors, 0] - pred_new[neighbors, 0])
            spill_spi = np.sum(pred_new[neighbors, 1] - pred_old[neighbors, 1])
            
        metric = (d_epi + spill_epi) if mode == "Hub" else (d_spi + spill_spi)
        candidates.append((idx, metric))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in candidates[:TOP_N_TO_OPTIMIZE]]

# ==============================================================================
# 3. HELPER: PCT
# ==============================================================================
def calc_pct(orig, delta):
    if abs(orig) < 0.001: return 0.0
    return (delta / orig) * 100

# ==============================================================================
# 4. MAIN WORKFLOW
# ==============================================================================
def run_robust_optimization():
    print("[1/4] Loading Data & Model...")
    data, df_nodes, feature_cols, target_cols = load_graph()
    feat_indices = {k: feature_cols.index(k) for k in FEATURE_MAP.keys()}
    
    model = UrbanGNN(num_features=data.x.shape[1], hidden_dim=64, num_targets=2)
    try:
        model.load_state_dict(torch.load("urban_gnn_model.pth", weights_only=True))
    except FileNotFoundError:
        print("Error: model not found.")
        return
    model.eval()

    # --- CATEGORIZE ---
    print("[2/4] Identifying Champions...")
    row, col = data.edge_index
    deg = torch.bincount(row, minlength=data.num_nodes).numpy()
    with torch.no_grad(): preds = model(data.x, data.edge_index).cpu().numpy()
    
    deg_top20 = np.percentile(deg, 80)
    deg_median = np.median(deg)
    spi_top30 = np.percentile(preds[:, 1], 70)
    spi_bot30 = np.percentile(preds[:, 1], 30)
    
    hub_pool = [i for i in range(data.num_nodes) if deg[i] >= deg_top20 and preds[i, 1] >= spi_top30]
    lag_pool = [i for i in range(data.num_nodes) if deg[i] > deg_median and deg[i] < deg_top20 and preds[i, 1] <= spi_bot30]
    
    # Shuffle before scan to ensure random sampling of the pool
    random.shuffle(hub_pool)
    random.shuffle(lag_pool)
    
    top_hubs = scan_champions(model, data, hub_pool[:SCAN_POOL_SIZE], feat_indices, "Hub")
    top_lags = scan_champions(model, data, lag_pool[:SCAN_POOL_SIZE], feat_indices, "Lagging")

    # --- OPTIMIZE (ENSEMBLE) ---
    print(f"\n[3/4] Running Robust Optimization ({ENSEMBLE_RUNS} runs per node)...")
    results = []
    
    def process_group(candidates, type_name, mode):
        for idx in candidates:
            runs = []
            for r in range(ENSEMBLE_RUNS):
                # Set different seed for each run implicitly by random state
                optimizer = GeneticOptimizer(model, data, idx, feat_indices, mode=mode)
                best_genome, stats = optimizer.run()
                runs.append((best_genome, stats))
            
            # Average the runs
            avg_ndvi = np.mean([r[0][0] for r in runs])
            avg_nbh = np.mean([r[0][1] for r in runs])
            avg_d_epi = np.mean([r[1][0] for r in runs])
            avg_d_spi = np.mean([r[1][1] for r in runs])
            avg_spill_epi = np.mean([r[1][2] for r in runs])
            avg_spill_spi = np.mean([r[1][3] for r in runs])
            
            # Original Baselines (Same for all runs)
            orig_epi = optimizer.base_epi
            orig_spi = optimizer.base_spi
            
            results.append({
                "Type": type_name, "ID": idx,
                "d_NDVI": avg_ndvi, "d_NbH": avg_nbh,
                "d_EPI": avg_d_epi, "pct_EPI": calc_pct(orig_epi, -avg_d_epi)*-1,
                "d_SPI": avg_d_spi, "pct_SPI": calc_pct(orig_spi, avg_d_spi),
                "Spill_EPI": avg_spill_epi, "Spill_SPI": avg_spill_spi
            })

    process_group(top_hubs, "Hub", "Hub")
    process_group(top_lags, "Lagging", "Lagging")

    # --- FINAL TABLE ---
    print("\n" + "="*155)
    print(f"{'Type':<8} | {'ID':<5} | {'d_NDVI':<8} {'d_NbH':<8} | {'Loc EPI':<18} {'Loc SPI':<18} | {'Spill EPI':<14} {'Spill SPI':<14}")
    print(f"{'':<8} | {'':<5} | {'(Avg σ)':<8} {'(Avg σ)':<8} | {'Gain (%)':<18} {'Gain (%)':<18} | {'(Total)':<14} {'(Total)':<14}")
    print("-" * 155)
    
    for r in results:
        epi_str = f"{r['d_EPI']:+.2f} ({r['pct_EPI']:+.1f}%)"
        spi_str = f"{r['d_SPI']:+.2f} ({r['pct_SPI']:+.1f}%)"
        
        print(f"{r['Type']:<8} | {r['ID']:<5} | {r['d_NDVI']:+.2f}     {r['d_NbH']:+.2f}     | {epi_str:<18} {spi_str:<18} | {r['Spill_EPI']:+.2f}          {r['Spill_SPI']:+.2f}")
        
    print("="*155)
    print("Note: Results are averaged over 3 independent optimization runs for robustness.")

if __name__ == "__main__":
    run_robust_optimization()