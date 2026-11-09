The FEBE model
==============

FEBE (Front-End / Back-End Evolution) is the eco-evolutionary agent-based
model of Dodd (2026).  This page is the ODD-style summary; the thesis
holds the full description and the biological motivation.

Entities
--------

**Organisms** carry four heritable traits: front-end investment
:math:`E_f`, back-end investment :math:`E_b` (both in energy units per
year), niche position :math:`\mu` and niche breadth :math:`\sigma` (in
resource types).  Non-heritable state: age and energy reserve.

**Resources** are :math:`Z = 100` discrete types on one axis with a
Gaussian carrying capacity (peak 400 items, SD 15).  Stock is an item
count; a successful encounter removes one item if the type is above its
refuge (10 % of capacity).  Stocks regrow logistically once a year and
may fluctuate year to year by a mean-preserving log-normal shock with
correlation length :math:`L` along the axis.

Cost laws
---------

Both ends convert investment to fidelity through a Shannon-type channel
with a soft cap,

.. math::

   F(E) = \mathrm{softcap}\!\left[B \log_2 (1 + \mathrm{SNR}\,E)\right],
   \qquad B = F_{\max}/k,
   \qquad \mathrm{softcap}(x) = \frac{x}{\left[1 + (x/F_{\max})^s\right]^{1/s}}.

The back-end differs in three ways: its ceiling is :math:`M F_{\max}`,
its bandwidth is :math:`B_b = r_b M F_{\max} / k`, and it returns
nothing until investment exceeds :math:`e_\mathrm{switch} = P\,E_{f,\mathrm{sat}}`:

.. math::

   F_b(E_b) = \mathrm{softcap}\!\left[B_b \log_2\!\left(1 + \mathrm{SNR}\,\max(0, E_b - e_\mathrm{switch})\right)\right].

Total fidelity is additive, :math:`F = 1 + F_f + F_b`, and multiplies
the energy extracted from every item.  With a matched-filter front-end
(Wehner 1987) the front-end contribution on an item of type :math:`z` is
discounted by a Gaussian tuning curve of width :math:`w` centred on
:math:`\mu`.

Life cycle (one year)
---------------------

1. Carrying capacities fluctuate (if volatility is on).
2. :math:`R = 32` foraging rounds; in each round every organism, in
   random order, draws one type from its encounter distribution (or
   uniformly in Experiment 1) and takes one item if available.
3. Energy settles: :math:`\Delta E = \sum_\text{items} F(z) - b - E_f - E_b`
   with basal cost :math:`b = 8`.
4. Reproduction: each organism enters the mating pool with probability
   :math:`\min(1, E/60)`; the pool is paired at random, each pair pays 8
   per parent and produces one offspring with 5 energy.  Traits are
   inherited uniparentally and mutated: small Gaussian steps (SD 0.03 on
   investments), plus rare large-effect gain (to a minimal functional
   back-end) and loss (to :math:`E_b = 0`) of the back-end.
5. Mortality: starvation (:math:`E \le 0`) or age :math:`> 20`.
6. Logistic regrowth of every resource type.

Experiments
-----------

The thesis presets are in :mod:`brainevo.abm.presets`.

* ``exp1-cost-sweep``: niche fixed, uniform foraging; sweep
  :math:`F_{\max}`, :math:`M`, :math:`P`, :math:`r_b`.
* ``exp1-overhead-sweep``: :math:`F_{\max}=1.2`, :math:`M=2.5`; sweep :math:`P`.
* ``exp2-tuning``: matched-filter front-end (:math:`w = 3.75`); niche evolves.
* ``exp2-relative-control``: tuning width scales with niche breadth.
* ``exp3-volatility``: ``exp2-tuning`` with :math:`v = 0.25`, :math:`L = \infty`.
