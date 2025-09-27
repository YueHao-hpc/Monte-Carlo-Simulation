
# Monte Carlo European Call (C++ OpenMP) + Python Viz

## Build
```bash
cd mc_option_project
mkdir -p build && cd build
cmake ..
cmake --build . -j
```

## Run
```bash
./mc_option \
  paths=2000000 steps=252 S0=100 K=100 T=1 r=0.03 q=0 sigma=0.2 seed=42 threads=8
```

Outputs:
- `paths.csv` (first N=50 paths, configurable) -> for plotting
- `convergence.csv` (approx running mean) -> for plotting

## Visualize
```bash
# from the build directory after running the binary
python3 ../visualize.py
# generates paths_plot.png and convergence_plot.png
```

## Notes
- Change threads with `threads=...` (or OMP env vars).
- This demo uses GBM under risk-neutral drift (r - q).
- For barrier/Asian options, change the payoff and (for Asian) track running averages per path.
