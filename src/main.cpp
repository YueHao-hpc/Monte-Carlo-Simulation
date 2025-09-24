
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

// Simple European Call Monte Carlo with OpenMP.
// Outputs: price/stderr/time to stdout, and optional CSVs:
//   - paths.csv: a small sample of simulated price paths
//   - convergence.csv: running mean of discounted payoff (for convergence plot)

struct Params
{
    long long n_paths = 2'000'000; // total paths
    int n_steps = 252;             // timesteps
    double S0 = 100.0;
    double K = 100.0;
    double T = 1.0;
    double r = 0.03;
    double q = 0.00;
    double sigma = 0.20;
    unsigned int seed = 42u;
    int n_threads = 0;     // 0 => use OMP default
    int sample_paths = 50; // how many paths to dump to CSV (0 to disable)
    bool dump_convergence = true;
};

static inline void write_paths_csv(const std::string &path, const std::vector<std::vector<double>> &paths)
{
    std::ofstream out(path);
    // header
    out << "step";
    if (!paths.empty())
    {
        for (size_t i = 0; i < paths.size(); ++i)
            out << ",path" << (i + 1);
    }
    out << "\n";
    int steps = paths.empty() ? 0 : (int)paths[0].size();
    for (int t = 0; t < steps; ++t)
    {
        out << t;
        for (const auto &p : paths)
            out << "," << std::setprecision(10) << p[t];
        out << "\n";
    }
}

static inline void write_convergence_csv(const std::string &path, const std::vector<double> &running_mean)
{
    std::ofstream out(path);
    out << "num_paths,mean_discounted_payoff\n";
    for (size_t i = 0; i < running_mean.size(); ++i)
    {
        out << (i + 1) << "," << std::setprecision(10) << running_mean[i] << "\n";
    }
}
// --- Black–Scholes European Call closed-form (with dividend yield q) ---
static inline double norm_cdf(double x)
{
    // Φ(x) = 0.5 * erfc(-x / sqrt(2))
    static const double inv_sqrt2 = 0.7071067811865475;
    return 0.5 * std::erfc(-x * inv_sqrt2);
}

static inline double bs_call_price(double S0, double K, double r, double q, double sigma, double T)
{
    if (sigma <= 0 || T <= 0)
    {
        return std::max(0.0, S0 * std::exp(-q * T) - K * std::exp(-r * T));
    }
    double sqrtT = std::sqrt(T);
    double d1 = (std::log(S0 / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    double d2 = d1 - sigma * sqrtT;
    return S0 * std::exp(-q * T) * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}
int main(int argc, char **argv)
{
    Params P;
    // Read simple overrides from env/args if needed (basic parsing)
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        auto get = [&](const std::string &key) -> const char *
        {
            if (a.rfind(key + "=", 0) == 0)
                return a.c_str() + key.size() + 1;
            return nullptr;
        };
        if (const char *v = get("paths"))
            P.n_paths = std::atoll(v);
        else if (const char *v = get("steps"))
            P.n_steps = std::atoi(v);
        else if (const char *v = get("S0"))
            P.S0 = std::atof(v);
        else if (const char *v = get("K"))
            P.K = std::atof(v);
        else if (const char *v = get("T"))
            P.T = std::atof(v);
        else if (const char *v = get("r"))
            P.r = std::atof(v);
        else if (const char *v = get("q"))
            P.q = std::atof(v);
        else if (const char *v = get("sigma"))
            P.sigma = std::atof(v);
        else if (const char *v = get("seed"))
            P.seed = (unsigned)std::stoul(v);
        else if (const char *v = get("threads"))
            P.n_threads = std::atoi(v);
        else if (a == "no_paths_csv")
            P.sample_paths = 0;
        else if (a == "no_convergence_csv")
            P.dump_convergence = false;
    }

    if (P.n_threads > 0)
    {
#ifdef _OPENMP
        omp_set_num_threads(P.n_threads);
#endif
    }

    const double dt = P.T / P.n_steps;
    const double drift = (P.r - P.q - 0.5 * P.sigma * P.sigma) * dt;
    const double vol_sqrt_dt = P.sigma * std::sqrt(dt);

    auto t0 = std::chrono::high_resolution_clock::now();

// per-thread RNG and local accumulators
#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
#else
    int nthreads = 1;
#endif

    std::vector<double> local_sum(nthreads, 0.0);
    std::vector<double> local_sumsq(nthreads, 0.0);

    // optional sampled paths for CSV (recorded by thread 0)
    std::vector<std::vector<double>> sampled_paths;
    if (P.sample_paths > 0)
    {
        sampled_paths.assign(P.sample_paths, std::vector<double>(P.n_steps + 1, 0.0));
        for (int i = 0; i < P.sample_paths; ++i)
            sampled_paths[i][0] = P.S0;
    }

    // optional convergence running mean (downsampled to ~1e5 points max)
    std::vector<double> running_mean;
    if (P.dump_convergence)
    {
        running_mean.reserve((size_t)std::min<long long>(100000LL, P.n_paths));
    }

#pragma omp parallel
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        std::mt19937_64 gen(P.seed + 9973u * (unsigned)tid);
        std::normal_distribution<double> nd(0.0, 1.0);

// Thread-local state for sampled paths
// Only thread 0 will write sampled paths to avoid locks.
// It will simulate the first P.sample_paths paths explicitly storing all steps.
#pragma omp for schedule(static)
        for (long long i = 0; i < P.n_paths; ++i)
        {
            double S = P.S0;
            if (P.sample_paths > 0 && i < P.sample_paths && tid == 0)
            {
                for (int t = 0; t < P.n_steps; ++t)
                {
                    double Z = nd(gen);
                    S *= std::exp(drift + vol_sqrt_dt * Z);
                    sampled_paths[(size_t)i][(size_t)t + 1] = S;
                }
            }
            else
            {
                for (int t = 0; t < P.n_steps; ++t)
                {
                    double Z = nd(gen);
                    S *= std::exp(drift + vol_sqrt_dt * Z);
                }
            }
            double payoff = std::max(S - P.K, 0.0);
            double disc_payoff = std::exp(-P.r * P.T) * payoff;
            local_sum[tid] += disc_payoff;
            local_sumsq[tid] += disc_payoff * disc_payoff;

            if (P.dump_convergence && tid == 0)
            {
                // downsample to about 1e5 points
                long long stride = std::max(1LL, P.n_paths / 100000LL);
                static thread_local long long count0 = 0;
                ++count0;
                if (count0 % stride == 0)
                {
                    double total = 0.0;
                    for (double v : local_sum)
                        total += v;              // approx; other threads not included
                    long long n_approx = count0; // only paths handled by thread 0, used for rough curve
                    running_mean.push_back(total / std::max(1LL, n_approx));
                }
            }
        }
    } // end parallel

    double sum = 0.0, sumsq = 0.0;
    for (int t = 0; t < nthreads; ++t)
    {
        sum += local_sum[t];
        sumsq += local_sumsq[t];
    }
    double mean = sum / (double)P.n_paths;
    double var = std::max(0.0, (sumsq / (double)P.n_paths) - mean * mean);
    double stderr = std::sqrt(var / (double)P.n_paths);

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Price: " << mean << "\n";
    std::cout << "StdErr: " << stderr << "\n";
    std::cout << "Paths: " << P.n_paths << ", Steps: " << P.n_steps << "\n";
#ifdef _OPENMP
    std::cout << "Threads: " << nthreads << "\n";
#else
    std::cout << "Threads: " << 1 << "\n";
#endif
    std::cout << "Time_ms: " << ms << "\n";
    // Black–Scholes closed-form comparison
    double price_bs = bs_call_price(P.S0, P.K, P.r, P.q, P.sigma, P.T);
    double abs_err = std::fabs(mean - price_bs);
    double rel_err = abs_err / (price_bs == 0.0 ? 1.0 : price_bs);

    std::cout << "BS_ClosedForm: " << price_bs << "\n";
    std::cout << "AbsError: " << abs_err << "\n";
    std::cout << "RelError: " << rel_err * 100.0 << "%\n";

    if (P.sample_paths > 0)
    {
        write_paths_csv("paths.csv", sampled_paths);
    }
    if (P.dump_convergence && !running_mean.empty())
    {
        write_convergence_csv("convergence.csv", running_mean);
    }
    return 0;
}
