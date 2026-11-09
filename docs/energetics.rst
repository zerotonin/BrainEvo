Grounded cost laws
==================

:mod:`brainevo.abm.energetics` replaces the phenomenological Shannon law
with laws whose shapes and scales come from measurements.  Costs are
fractions of the organism's basal metabolic rate (the model's
:math:`b = 8`), because that is how the literature reports neural costs.

Front-end
---------

.. math::

   F_f(E) = \mathrm{softcap}\!\left[F_{\max,f}\,(E/E_{f,\max})^{1/\alpha},\; F_{\max,f}\right],
   \qquad \alpha = 1.7,\quad E_{f,\max} = 0.08\,b.

The exponent is the design-level scaling of photoreceptor cost with peak
information rate across fly species (Niven, Anderson & Laughlin 2007);
the energy scale is the retina's share of resting metabolism (Howard et
al. 1987; Niven & Laughlin 2008).

Back-end
--------

.. math::

   E_b(n) = E_{b,\max}\,\frac{n\,(1 + w\,n^{\gamma})}{1+w},\qquad
   F_b = F_{\max,b}\,\frac{g(\mathrm{softcap}(n,1))}{g(1)},\qquad
   g(n) = \frac{(n-n_{\min})_+}{(n-n_{\min})_+ + n_{1/2}},

with :math:`n` the neuron count relative to the ceiling the back-end share
can buy.  Cost per neuron is constant across species (Herculano-Houzel
2011), the wire tax exponent :math:`\gamma = 1/3` follows white matter
:math:`\propto` grey matter\ :sup:`4/3` (Zhang & Sejnowski 2000; 0 for
insects, Polilov & Makarova 2020), pooled information saturates under
information-limiting correlations (Zohary et al. 1994; Moreno-Bote et al.
2014), and the minimal functional circuit :math:`n_{\min}` is the
switch-on (Chittka & Niven 2009; Caron et al. 2013).

Every constant is a :class:`~brainevo.abm.energetics.GroundedConstant`
with value, range and the sentence it comes from;
:func:`~brainevo.abm.energetics.constants_table` prints the methods table.

Capsule and oxygen cap
----------------------

With a :class:`~brainevo.abm.energetics.CapsuleLaw` the housing of central
tissue becomes a third heritable trait :math:`E_c`.  Capacity costs
support-tissue maintenance (Attwell & Laughlin 2001; Karbowski 2011),
carriage (Taylor et al. 1980; Nagy 2005) and amortised construction
(Kuzawa et al. 2014), and the neurons an organism can use are
:math:`\min(n, \text{capacity}, \text{hard cap})`.  The hard cap is the
lineage's oxygen supply: diffusive (tracheal) supply scales as
:math:`(p\mathrm{O}_2 / 21.2\,\mathrm{kPa})^{1.5}` (Harrison, Kaiser &
VandenBrooks 2010), convective supply follows arterial haemoglobin
saturation (Severinghaus 1979) and is nearly flat above 12 kPa.  The
mean field pairs every :math:`E_b` with the smallest capsule that houses
it, so the reduction stays two-dimensional.

Implications
------------

With these laws the thesis' front-end calibration is confirmed (saturation
at 12 % of basal versus the retina's 8-10 %), but the thesis back-end is
about ten times too expensive (66 % of basal at saturation versus 2-20 %
for real brains) and its overhead :math:`P` is 20-300 times larger than
the cost of a minimal circuit.  Under grounded costs the fitness valley
is shallower than one mutational step and the back-end establishes within
tens of years in the agent model, so within-lineage bistability rests on
the one unmeasured quantity, :math:`n_{\min}`, or on a developmental cost
that running-ATP budgets do not contain.
