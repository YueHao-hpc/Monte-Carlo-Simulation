
import pandas as pd
import matplotlib.pyplot as plt

# Plot a few simulated paths
try:
    df = pd.read_csv("build/paths.csv")
    t = df["step"].values
    cols = [c for c in df.columns if c.startswith("path")]
    plt.figure()
    for c in cols:
        plt.plot(t, df[c].values, linewidth=1)
    plt.xlabel("Step")
    plt.ylabel("Price")
    plt.title("Sample Simulated Paths")
    plt.tight_layout()
    plt.savefig("paths_plot.png", dpi=150)
    print("Saved paths_plot.png")
except Exception as e:
    print("Skipping paths plot:", e)

# Plot convergence of running mean
try:
    dc = pd.read_csv("build/convergence.csv")
    plt.figure()
    plt.plot(dc["num_paths"].values, dc["mean_discounted_payoff"].values)
    plt.xlabel("Approx. Num Paths (thread 0)")
    plt.ylabel("Mean Discounted Payoff")
    plt.title("Convergence (illustrative)")
    plt.tight_layout()
    plt.savefig("convergence_plot.png", dpi=150)
    print("Saved convergence_plot.png")
except Exception as e:
    print("Skipping convergence plot:", e)
