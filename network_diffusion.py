# =========================================================
# COMPLEXITY ECONOMICS MODEL
# Network Diffusion of Economic Shocks
# =========================================================

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# -------------------------
# 1. PARAMETERS
# -------------------------

NUM_AGENTS = 50          # number of nodes in the network
TIME_STEPS = 30          # simulation length
SHOCK_SIZE = -5.0        # size of initial negative shock
SHOCK_NODE = 0           # node where shock originates
DECAY = 0.6              # strength of spillover

np.random.seed(42)

# -------------------------
# 2. CREATE NETWORK
# -------------------------

# Scale-free network (Barabási–Albert)
G = nx.barabasi_albert_graph(NUM_AGENTS, m=2)

# Initial wealth for each agent
wealth = {node: 10.0 for node in G.nodes()}
wealth_history = []

# -------------------------
# 3. SHOCK MECHANISM
# -------------------------

def apply_initial_shock(wealth, node):
    """
    Applies an exogenous shock to one node
    """
    wealth[node] += SHOCK_SIZE


def diffusion_step(G, wealth):
    """
    Diffuses wealth shocks through the network
    """
    new_wealth = wealth.copy()

    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if neighbors:
            spillover = sum(
                DECAY * (wealth[n] - wealth[node])
                for n in neighbors
            ) / len(neighbors)
            new_wealth[node] += spillover

    return new_wealth

# -------------------------
# 4. SIMULATION
# -------------------------

apply_initial_shock(wealth, SHOCK_NODE)

for t in range(TIME_STEPS):
    wealth = diffusion_step(G, wealth)
    wealth_history.append(list(wealth.values()))

# -------------------------
# 5. ANALYSIS
# -------------------------

average_wealth = np.mean(wealth_history, axis=1)

plt.figure(figsize=(10, 5))
plt.plot(average_wealth)
plt.title("Average Wealth After Network Shock")
plt.xlabel("Time")
plt.ylabel("Average Wealth")
plt.tight_layout()
plt.show()

# -------------------------
# 6. NETWORK VISUALIZATION
# -------------------------

plt.figure(figsize=(8, 8))
pos = nx.spring_layout(G, seed=42)
node_colors = [wealth[n] for n in G.nodes()]

nx.draw(
    G,
    pos,
    node_color=node_colors,
    cmap=plt.cm.coolwarm,
    with_labels=False,
    node_size=200
)

plt.title("Final Wealth Distribution on Network")
plt.show()
