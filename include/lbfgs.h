// Limited-memory BFGS minimizer (Nocedal & Wright, Algorithm 7.5) with a
// backtracking Armijo line search. Used by logistic regression on the CPU
// and on the GPU: the optimizer runs on the host over the (small) parameter
// vector, and only the objective callback differs between the two.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace ml {

struct LbfgsOptions {
    int max_iter = 100;
    int history = 10;        // correction pairs kept
    double tol = 1e-4;       // stop when the largest gradient component is below this
};

struct LbfgsResult {
    int iterations = 0;
    int evaluations = 0;     // objective + gradient evaluations
    double value = 0;
    double gradient_max = 0;
    bool converged = false;
};

// eval(x, g) returns f(x) and writes the gradient into g.
template <typename Eval>
LbfgsResult lbfgs_minimize(std::vector<double>& x, Eval&& eval, const LbfgsOptions& opt) {
    const size_t n = x.size();
    auto dot = [n](const std::vector<double>& a, const std::vector<double>& b) {
        double s = 0;
        for (size_t i = 0; i < n; ++i) s += a[i] * b[i];
        return s;
    };
    auto max_abs = [](const std::vector<double>& a) {
        double m = 0;
        for (double v : a) m = std::max(m, std::fabs(v));
        return m;
    };

    LbfgsResult r;
    std::vector<double> g(n), g_new(n), x_new(n), dir(n);
    std::vector<std::vector<double>> s_hist, y_hist;
    std::vector<double> rho_hist, alpha;

    double f = eval(x, g);
    ++r.evaluations;
    r.gradient_max = max_abs(g);

    while (r.iterations < opt.max_iter && r.gradient_max > opt.tol) {
        // Two-loop recursion: dir = -H g.
        const size_t m = s_hist.size();
        dir = g;
        alpha.assign(m, 0.0);
        for (size_t k = m; k-- > 0;) {
            alpha[k] = rho_hist[k] * dot(s_hist[k], dir);
            for (size_t i = 0; i < n; ++i) dir[i] -= alpha[k] * y_hist[k][i];
        }
        if (m > 0) {
            const double gamma = 1.0 / (rho_hist[m - 1] * dot(y_hist[m - 1], y_hist[m - 1]));
            for (size_t i = 0; i < n; ++i) dir[i] *= gamma;
        }
        for (size_t k = 0; k < m; ++k) {
            const double beta = rho_hist[k] * dot(y_hist[k], dir);
            for (size_t i = 0; i < n; ++i) dir[i] += (alpha[k] - beta) * s_hist[k][i];
        }
        for (size_t i = 0; i < n; ++i) dir[i] = -dir[i];

        double slope = dot(g, dir);
        if (!(slope < 0)) {   // not a descent direction: restart from steepest descent
            s_hist.clear();
            y_hist.clear();
            rho_hist.clear();
            for (size_t i = 0; i < n; ++i) dir[i] = -g[i];
            slope = -dot(g, g);
        }

        // Backtracking line search (sufficient decrease). The first iteration
        // has no curvature information, so its step is scaled by the gradient.
        double step = s_hist.empty() ? 1.0 / std::max(1.0, std::sqrt(dot(g, g))) : 1.0;
        double f_new = f;
        bool accepted = false;
        for (int tries = 0; tries < 40; ++tries) {
            for (size_t i = 0; i < n; ++i) x_new[i] = x[i] + step * dir[i];
            f_new = eval(x_new, g_new);
            ++r.evaluations;
            if (std::isfinite(f_new) && f_new <= f + 1e-4 * step * slope) {
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if (!accepted) break;   // no further progress possible at this precision

        std::vector<double> s(n), y(n);
        for (size_t i = 0; i < n; ++i) {
            s[i] = x_new[i] - x[i];
            y[i] = g_new[i] - g[i];
        }
        const double sy = dot(s, y);
        if (sy > 1e-12 * dot(y, y)) {   // keep the pair only if it carries positive curvature
            if (static_cast<int>(s_hist.size()) == opt.history) {
                s_hist.erase(s_hist.begin());
                y_hist.erase(y_hist.begin());
                rho_hist.erase(rho_hist.begin());
            }
            s_hist.push_back(std::move(s));
            y_hist.push_back(std::move(y));
            rho_hist.push_back(1.0 / sy);
        }
        x.swap(x_new);
        g.swap(g_new);
        f = f_new;
        ++r.iterations;
        r.gradient_max = max_abs(g);
    }
    r.value = f;
    r.converged = r.gradient_max <= opt.tol;
    return r;
}

}  // namespace ml
