# Project 3: Aggregate loss simulation in R Markdown

Simulates the total annual loss of an insurance portfolio (compound Poisson with lognormal
claim sizes) and studies how risk behaves.

## Files
- `src/aggregate_loss.Rmd`: the full analysis (code, tables, plots, conclusions)
- `reports/aggregate_loss.html`: the rendered report

## Reproduce (from this folder)
    Rscript -e 'rmarkdown::render("src/aggregate_loss.Rmd", output_dir = "reports")'

## What it covers
1. Simulate total annual loss and check the mean and variance against E[S] = lambda*E[X] and
   Var(S) = lambda*E[X^2].
2. Show the coefficient of variation falling like 1/sqrt(n) as the portfolio grows.
3. Fit the severity distribution by maximum likelihood, with QQ plots, a Gamma comparison and a
   parametric-bootstrap KS test.
4. Apply a per-claim deductible and an aggregate stop-loss; compare mean, SD and 99th percentile.
5. Bootstrap the 99th percentile from histories of 20 to 10,000 years against a 100,000-year reference.

See the Conclusions section of the report for results and limitations.
