# Documentation

This project estimates a noisy binary transition contour in a bounded 2-D
parameter space. The generic contour is:

```text
y_c(x)
```

where:

```text
p(id = 1 | x, y) = 0.5
```

The method is Bayesian active learning / level-set estimation for a binary
classifier. It is not ordinary Bayesian optimization because there is no scalar
objective to maximize.

## Files

- [workflow.md](workflow.md): how to run iterative sweeps.
- [model-and-acquisition.md](model-and-acquisition.md): model assumptions and scoring.
- [drop-injection-mapping.md](drop-injection-mapping.md): how `Rr,Oh,id` maps into generic `x,y,id`.

## Required Input Shape

Use completed data with:

```text
caseId,x,y,id
```

For physical projects, pass column mappings:

```bash
--x-col Rr --y-col Oh --label-col id
```

Rows with `id = -1` are treated as stale or pending proposals and are ignored
during fitting.

## Main Output Shape

```text
caseId,x,y,id,proposal_type,score,p_positive_pred,y_c_pred,
y_c_q05,y_c_q95,y_c_std,n_existing,k_existing,reason
```

Use `--legacy-columns-only` to write only:

```text
caseId,x,y,id
```
