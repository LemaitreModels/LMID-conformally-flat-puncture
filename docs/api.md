# API reference

The entry points for the three workflows:

| task | call |
|---|---|
| get a shipped model | {func}`models.load <lemaitre.initial_data.conformally_flat_puncture.models.load>` |
| interpolate | `model.evaluate(theta)`, `model.evaluate_jax(theta)` ({class}`~lemaitre.initial_data.conformally_flat_puncture.parametric.hermite_smolyak_pod_cross.PODHermiteSmolyakCross`) |
| refine and certify | `model.evaluate_polished(theta)` |
| the model's grid | {func}`models.problem <lemaitre.initial_data.conformally_flat_puncture.models.problem>` |
| parameters to a binary | {func}`~lemaitre.initial_data.conformally_flat_puncture.parametric.parametric_nd_3d.theta_to_slice3d` |
| solve one binary | {func}`~lemaitre.initial_data.conformally_flat_puncture.solver.solver_3d_nk.newton_solve_nk` |
| enforce the gate on a raw solve | {func}`~lemaitre.initial_data.conformally_flat_puncture.parametric.certification.certified_return` |
| evaluate $u$ at points | {func}`~lemaitre.initial_data.conformally_flat_puncture.solver.solver_3d.evaluate_field` |
| certified tangent $\partial U/\partial\theta$ | {func}`~lemaitre.initial_data.conformally_flat_puncture.applications.sensitivity_3d_qc.certified_tangent_3d_qc` |
| export to GRTeclyn | {func}`~lemaitre.initial_data.conformally_flat_puncture.validation.export_grteclyn.from_solution` |

## The shipped model class

```{eval-rst}
.. autoclass:: lemaitre.initial_data.conformally_flat_puncture.parametric.hermite_smolyak_pod_cross.PODHermiteSmolyakCross
   :members: evaluate, evaluate_jax, evaluate_polished, coeffs, save
   :inherited-members:
   :no-index:
```

## Modules

The model download, and the four subpackages that make up the package. The
producers in `pipeline/` are scripts, one per figure or model build; they are
described on [Reproduce the paper](DATA.md).

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   lemaitre.initial_data.conformally_flat_puncture.models
   lemaitre.initial_data.conformally_flat_puncture.solver
   lemaitre.initial_data.conformally_flat_puncture.parametric
   lemaitre.initial_data.conformally_flat_puncture.applications
   lemaitre.initial_data.conformally_flat_puncture.validation
```
