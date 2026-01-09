# =========================================================
# GRAD-LEVEL PYTHON PROJECT
# Agent-Based Model (Micro → Macro Economics)
# =========================================================

import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# 1. MODEL PARAMETERS
# -------------------------

NUM_AGENTS = 500          # number of economic agents
TIME_PERIODS = 200        # simulation length

BETA = 0.95               # discount factor
INTEREST_RATE = 0.03      # savings return
INCOME_VOLATILITY = 0.3   # income risk
INITIAL_WEALTH = 10.0

np.random.seed(42)

# -------------------------
# 2. AGENT CLASS
# -------------------------

class EconomicAgent:
    """
    Represents one household/agent in the economy
    """

    def __init__(self, wealth):
        self.wealth = wealth
        self.consumption = 0.0

    def income_shock(self):
        """
        Lognormal income shock
        """
        return np.random.lognormal(mean=0, sigma=INCOME_VOLATILITY)

    def choose_consumption(self, income):
        """
        Simple forward-looking consumption rule
        """
        total_resources = self.wealth + income
        self.consumption = (1 - BETA) * total_resources
        return self.consumption

    def update_wealth(self, income):
        savings = self.wealth + income - self.consumption
        self.wealth = (1 + INTEREST_RATE) * savings

# -------------------------
# 3. ECONOMY CLASS
# -------------------------

class Economy:
    """
    Collection of agents + macro dynamics
    """

    def __init__(self):
        self.agents = [
            EconomicAgent(INITIAL_WEALTH)
            for _ in range(NUM_AGENTS)
        ]

        self.aggregate_consumption = []
        self.wealth_history = []

    def step(self):
        total_consumption = 0.0
        current_wealths = []

        for agent in self.agents:
            income = agent.income_shock()
            consumption = agent.choose_consumption(income)
            agent.update_wealth(income)

            total_consumption += consumption
            current_wealths.append(agent.wealth)

        self.aggregate_consumption.append(total_consumption)
        self.wealth_history.append(current_wealths)

    def run(self):
        for t in range(TIME_PERIODS):
            self.step()

# -------------------------
# 4. INEQUALITY MEASURE
# -------------------------

def gini_coefficient(x):
    """
    Computes Gini coefficient
    """
    x = np.array(x)
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * x)) / (n * np.sum(x)) - (n + 1) / n

# -------------------------
# 5. ANALYSIS & PLOTS
# -------------------------

def analyze_results(economy):
    final_wealth = economy.wealth_history[-1]
    gini = gini_coefficient(final_wealth)

    print("====================================")
    print("SIMULATION RESULTS")
    print("====================================")
    print(f"Number of agents: {NUM_AGENTS}")
    print(f"Time periods: {TIME_PERIODS}")
    print(f"Gini coefficient: {gini:.3f}")
    print("====================================")

    # Plot aggregate consumption
    plt.figure(figsize=(10, 5))
    plt.plot(economy.aggregate_consumption)
    plt.title("Aggregate Consumption Over Time")
    plt.xlabel("Time")
    plt.ylabel("Consumption")
    plt.show()

    # Plot wealth distribution
    plt.figure(figsize=(8, 5))
    plt.hist(final_wealth, bins=40)
    plt.title("Final Wealth Distribution")
    plt.xlabel("Wealth")
    plt.ylabel("Number of Agents")
    plt.show()

# -------------------------
# 6. RUN MODEL
# -------------------------

if __name__ == "__main__":
    economy = Economy()
    economy.run()
    analyze_results(economy)
