# School Funding vs. Academic Outcomes: A Value-Added Analysis

A pandas/numpy pipeline that goes beyond "does funding correlate with results?" 
to ask which schools beat the results their funding and deprivation profile 
would predict — the same "value-added" logic behind the UK DfE's Progress 8 metric.

## The question
Raw correlations between school funding and exam results are misleading: 
schools with more disadvantaged pupils receive more funding (via pupil premium) 
while facing steeper headwinds to attainment. This project controls for that 
by building an expected-outcome model, then flags schools whose actual results 
are well above or below prediction — the ones worth asking "what are they doing differently?"

## Key finding
Deprivation (% pupils on Free School Meals) explains far more variation in 
outcomes than funding level does. The model's low R² (~0.10) is itself a 
finding: most of the variation between schools comes from unmeasured factors 
— leadership, teaching quality — which the value-added residuals surface.

## Pipeline
1. **Generate/Load** — synthetic dataset modelled on real DfE funding/outcomes patterns
2. **Clean** — dedupe, numpy z-score outlier detection, median/mode imputation
3. **Feature engineer** — funding quartiles (`pd.qcut`), deprivation bands (`pd.cut`)
4. **Model** — multiple linear regression from scratch via `numpy.linalg.lstsq`
5. **Analyse** — residuals identify over/under-performing schools
6. **Visualise & report** — seaborn chart + auto-generated markdown report

## Skills demonstrated
- **pandas**: groupby/agg, qcut/cut binning, markdown report generation
- **numpy**: vectorised z-score outlier detection, OLS regression via linear algebra (not a black-box ML library)
- **Statistics**: correlation vs. regression, controlling for confounders, R², residual analysis

## Running it
\`\`\`bash
pip install pandas numpy matplotlib seaborn tabulate
python generate_data.py
python analysis.py
\`\`\`

## Next steps
- Merge in real DfE School Funding + Key Stage 4 performance tables
- Add school size and prior attainment as additional model controls
