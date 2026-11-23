# Reproduction dependencies

- python: 3.12.14
- numpy: 2.5.5
- matplotlib: 3.11.2

Statistics use Python stdlib only. CP endpoints invert finite binomial tails (64 bisection steps); no scipy/statsmodels runtime dependency. NumPy/Matplotlib are used only for the figure.

Method reference: https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html
