Adaptive-dynamics reduction
===========================

This page derives the mean-field model implemented in
:mod:`brainevo.ode`.  It is the Stage-2 witness of the project: if the
agent-based result survives the reduction, it is structural rather than
an artefact of stochasticity or implementation.

1. The environment is one number
--------------------------------

Under uniform foraging (Experiment 1) every organism draws resource
types uniformly, so the probability that a roll yields an item is the
same scalar :math:`p` for everyone, set by how depleted the standing
stock is.  The expected yearly energy balance of an organism with traits
:math:`x = (E_f, E_b)` is

.. math::

   \Delta E(x; p) = R\,e\,p\,F(x) - C(x), \qquad C(x) = b + c + E_f + E_b,

where :math:`F = 1 + F_f(E_f) + F_b(E_b)`, :math:`R` is the number of
foraging rounds, :math:`e` the energy per item, :math:`b` the basal cost
and :math:`c` the per-capita surplus that demography absorbs at
equilibrium (reproductive spending net of the energy injected at birth
and carried off by the dead).  The agent-based model measures
:math:`c \approx 1.35` at :math:`P = 2`.

2. Resident equilibrium and the pessimisation principle
-------------------------------------------------------

A resident population :math:`x` grows until its surplus is absorbed,
which fixes the environment:

.. math::

   p^*(x) = \frac{C(x)}{R\,e\,F(x)}.

A rare mutant :math:`x'` in that environment has surplus

.. math::

   s(x' \mid x) = R\,e\,p^*(x)\,F(x') - C(x')
                = C(x)\,\frac{F(x')}{F(x)} - C(x'),

so :math:`s(x'|x) > 0 \iff F(x')/C(x') > F(x)/C(x)`.  Because the
strategy affects fitness only through the one-dimensional, monotone
feedback :math:`p`, evolution obeys an optimisation principle (Metz,
Mylius & Diekmann 2008): it maximises

.. math::

   Q(x) = \frac{F(x)}{C(x)} = \frac{1 + F_f(E_f) + F_b(E_b)}{b + c + E_f + E_b},

the fidelity bought per unit of total cost, and equivalently minimises
:math:`p^*`, the resource level at which the population can persist.
:math:`Q` is a Lyapunov function of the evolutionary dynamics.

3. Singular strategies
----------------------

Setting :math:`\partial Q/\partial E_f = 0` and
:math:`\partial Q/\partial E_b = 0` gives

.. math::

   F_f'(E_f^*) = Q(x^*), \qquad F_b'(E_b^*) = Q(x^*) \quad\text{or}\quad E_b^* = 0.

Invest at each end until the marginal fidelity per unit energy equals the
organism's average fidelity per unit cost.  Because :math:`F_f` is
concave the front-end condition has one solution.  The back-end has two
candidates:

* **front-only**, :math:`E_b = 0`.  With a hard switch
  :math:`F_b'(0) = 0 < Q`, so :math:`\partial Q/\partial E_b < 0` at the
  boundary and :math:`E_b = 0` is a convergence-stable attractor **for every
  parameter value**.
* **mixed**, the interior solution of :math:`F_b'(E_b) = Q` above the
  switch, which exists whenever the log channel's initial slope
  :math:`B_b\,\mathrm{SNR}/\ln 2` exceeds :math:`Q`; for the thesis
  parameters it always does.

Hence the hard-switch model is **bistable everywhere**: the two
architectures are both local attractors, separated by the fitness valley
:math:`0 < E_b \lesssim e_\mathrm{switch} + \delta` in which cost is paid
and nothing returned.  Which is *globally* fitter is decided by
:math:`Q_\mathrm{mixed} \gtrless Q_\mathrm{front}`, the evolutionary
analogue of a Maxwell line.

4. What this says about the thesis results
------------------------------------------

* **Fig. 7 boundary.**  The mean field puts the Maxwell line at
  :math:`M F_{\max} \approx 0.5` for :math:`P = 2`, far below the
  simulated boundary at :math:`M F_{\max} \approx 2.5`.  Agent-based
  runs seeded above the switch confirm that the back-end is retained,
  with loss mutations enabled, throughout the region where unseeded runs
  never acquire it.  The Fig. 7 boundary is therefore an
  **accessibility** boundary (how fast a fitness valley is crossed in
  6000 years; Weissman et al. 2009), not an **affordability** boundary.
  The valley is crossed by a large-effect mutant whose immediate fitness
  deficit is :math:`\approx e_\mathrm{switch}`; its descendants must
  climb :math:`E_b` until :math:`Q` recovers.  The slope they climb at,
  :math:`B_b\,\mathrm{SNR}/\ln 2 \propto r_b M F_{\max}`, is what makes
  the boundary run along constant :math:`M F_{\max}`.
* **Evolved investments.**  The interior optimum predicts the seeded
  runs' :math:`E_b` to within a few percent and :math:`E_f` to within
  10-20 % (the latter still creeping upward at 2000 years).
* **"Back-end displaces front-end."**  A functional back-end raises
  :math:`F`, hence :math:`Q`, hence the marginal return the front-end
  must match; :math:`E_f^*` falls.  This is the mean-field reading of
  thesis §3.3.

5. Where the genuine bifurcation lives
--------------------------------------

Replace the hard kink by a smooth switch-on of width :math:`w`,

.. math::

   \max(0, E_b - e_\mathrm{switch}) \;\to\; w \log\!\left(1 + e^{(E_b - e_\mathrm{switch})/w}\right),

so that :math:`w \to 0` recovers the thesis model (``switch_shape="ramp"``,
``switch_width=w`` in :class:`brainevo.abm.cost_laws.CostLawParams`).  Now
:math:`F_b'(0) > 0`, and the front-only state is an attractor only while
:math:`F_b'(0) < Q`.  As :math:`w` grows (or :math:`P` falls) this
inequality fails and the front-only attractor disappears in a
**saddle-node bifurcation**; above a cusp point in the :math:`(P, w)`
plane only one attractor remains and allocation is a continuum.
Ramping :math:`P` forward and back under the canonical equation traces a
hysteresis loop whose width shrinks to zero at the cusp.  This is the
pre-registered test that separates "discreteness built in by the hard
switch" from "discreteness that survives a graded switch-on".

**Ramp vs saturation gate.**  The alternative family is a Hill gate of
flank steepness :math:`n`,

.. math::

   F_b = \mathrm{softcap}\!\left[B_b \log_2(1 + \mathrm{SNR}\,E_b)\right]\,
         \frac{E_b^n}{E_b^n + e_\mathrm{switch}^n}
   \qquad (\texttt{switch\_shape="hill"},\ \texttt{switch\_steepness}=n).

Its outcome differs qualitatively from the ramp: because the gate has
zero slope at :math:`E_b = 0` for every :math:`n > 0`, the front-only
state is a local attractor at *any* flank steepness; sweeping :math:`n`
never produces a fold.  What changes is the depth of the valley
(``MeanField.valley_deficit``), which falls from the hard-switch value
to effectively zero below :math:`n \approx 2`.  Whether the two states are
"distinct" is then a question of valley depth relative to the
mutational step, not of attractor count.  ``brainevo-ode fold-map
--shape ramp|hill`` draws both maps.

6. The canonical equation
-------------------------

With :math:`\mu` the mutation rate, :math:`\sigma^2` the mutational
variance and :math:`N^*(x)` the resident population size (Dieckmann & Law
1996),

.. math::

   \frac{dx}{dt} = \tfrac12\,\mu\,\sigma^2\,N^*(x)\,
   \left.\nabla_{x'} s(x' \mid x)\right|_{x'=x}
   = \tfrac12\,\mu\,\sigma^2\,N^*(x)\,C(x)\,\nabla \log Q(x).

The prefactor is positive, so trajectories ascend :math:`Q` and end at a
local maximum.  :func:`brainevo.ode.canonical.integrate_canonical`
integrates this flow with :math:`E_f, E_b \ge 0` enforced;
:func:`brainevo.ode.canonical.ramp_hysteresis` performs the quasi-static
forward/reverse ramps.

7. Evolving niche (Experiment 2): :mod:`brainevo.ode.niche`
------------------------------------------------------------

With niche-based foraging the environment is the vector of per-type
success probabilities :math:`h_z` set by the resident's encounter
distribution :math:`\pi_z(\mu,\sigma)`; no optimisation principle
exists.  :class:`brainevo.ode.niche.NicheMeanField` implements the
deterministic skeleton:

* **Stationary stocks.**  A yearly quota :math:`A_z = N R \pi_z` on a
  logistic stock has a stable upper fixed point only while
  :math:`A_z < rK_z/4`; above it the type collapses to its refuge and
  yields only the regrowth from the refuge.  Yield is therefore
  discontinuous in :math:`A_z` (overexploitation collapse), exactly as in
  the agent model.
* **Resident equilibrium.**  :math:`N^*` solves intake = cost; where the
  surplus jumps across zero without a root, the bracket closes on the
  jump and fitness is measured relative to the resident's realised
  surplus.
* **Invasion fitness** :math:`s(x'|x) = R e \sum_z \pi_z' h_z^* F(z;x') - C(x')`
  in coordinates :math:`(E_f, E_b, \log\sigma)`; selection gradient,
  mutant Hessian, Jacobian of the gradient, canonical equation with the
  thesis mutational variances, and the Geritz et al. (1998)
  classification (CSS / branching point / repellor).
* **PIP** in niche breadth at fixed architecture.

First result (thesis Exp-2 parameters, :math:`F_{\max}=0.8`, :math:`M=3`,
:math:`P=3.75`, :math:`w_\mathrm{tune}=3.75`): from either a front-only or a
mixed founder the flow converges to :math:`\sigma^* \approx 14.5`
(close to the resource SD of 15) with all Jacobian eigenvalues negative,
and the mutant Hessian has one positive eigenvalue aligned with
:math:`\log\sigma`.  The singular niche breadth is a **branching
point**: the resident at :math:`\sigma^*` is invadable by both narrower
and broader mutants (``brainevo-ode niche-pip``).  Investments stay
convergence- and evolutionarily stable (:math:`E_f^* \approx 0.37` front-only,
:math:`E_f^* \approx 0.25`, :math:`E_b^* \approx 4.5` mixed).  The prediction
for the agent model is a bimodal :math:`\sigma` distribution within a
run, i.e. coexisting specialists and generalists sharing one
architecture, to be checked against ``sigma_sd`` in the Exp-2 logs.

References
----------

Dieckmann U, Law R (1996) J Math Biol 34:579-612.
Geritz SAH, Kisdi É, Meszéna G, Metz JAJ (1998) Evol Ecol 12:35-57.
Metz JAJ, Mylius SD, Diekmann O (2008) Evol Ecol Res 10:629-654.
Weissman DB, Desai MM, Fisher DS, Feldman MW (2009) Theor Popul Biol 75:286-300.
Ackermann M, Doebeli M (2004) Evolution 58:2599-2612.
