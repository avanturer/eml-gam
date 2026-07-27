# Submission plan

The manuscript (`paper/paper.tex`, compiled `paper/paper.pdf`) is
self-contained: inline bibliography, all numbers generated from
versioned JSONs in this repository, all figures reproducible. What
remains for the author is administrative: name/affiliation, venue
formatting, and the submission forms.

## Recommended path

1. **arXiv first** (cs.LG, cross-list cs.SC + cs.NE). Establishes
   priority and lets the source paper's author (A. Odrzywołek —
   whose stated open problem this work resolves) see and cite it.
   The paper explicitly answers the call in arXiv:2603.21852 §4.3;
   a short courteous email to the author with the preprint link is
   likely to produce an engaged expert reader, and possibly an
   endorsement or collaboration.

2. **Primary venue: TMLR** (Transactions on Machine Learning
   Research). Best fit by design: rolling submission (no deadline
   pressure), no page limit (the current 10-page article format needs
   only the TMLR style file), and — decisively — its acceptance
   criterion is *"are the claims supported by correct, convincing
   evidence?"* rather than subjective impact. This work was built
   exactly for that standard: every claim is scoped (real-clamped
   semantics, grid identity, dense verification), every number has a
   seeded reproduction path, and the limitations are stated by us
   before a reviewer can. Double-blind: strip the repository link into
   an anonymised supplementary zip for review.

3. **Stretch: ICLR or NeurIPS main track.** The story (an open
   problem from a recent paper resolved by a representation change,
   with mechanism theory and a deployable system) is a strong main-
   track narrative; the risk is a reviewer discounting the niche
   (one operator grammar). If submitting there: compress to the page
   limit, move the transcendence section and certificate details to
   the appendix, and lead with Figures 1–2. The ψ-generality
   benchmark (`psi_depth_wall_results.json`) is the direct rebuttal
   to "does this only work for eml?" — keep it in the main text.

4. **Workshop fallback (fast feedback):** NeurIPS/ICML workshops on
   symbolic regression / AI-for-Math / interpretability accept 4–8
   page versions; useful for community contact if the main-track
   timing is wrong.

## Honest odds assessment (do not skip)

* **TMLR: strong position.** The claims-evidence match is the whole
  design of this project; the scope limits are stated; reproduction
  is turnkey. Main reviewer risks: (a) "enumeration with
  observational equivalence is standard program synthesis" — addressed
  head-on in Related Work (the novelty is the collapse measurement,
  the cause=cure identification, the root inversion, and the
  certificates, not the enumeration schema); (b) "niche grammar" —
  addressed by the ψ benchmark and by framing the recipe (enumerate
  the reachable semantics under the numerics actually used) as
  generic.
* **ICLR/NeurIPS: real but not safe.** Acceptance depends on drawing
  reviewers who value problem-resolution + mechanism over topical
  breadth. No one can promise acceptance anywhere; what is promised
  is that no rejection can honestly say "claims not supported."

## Venue-specific checklist

- [ ] Add author name(s), affiliation, correspondence email.
- [ ] Swap `\documentclass` to the venue style (TMLR: `tmlr.sty`;
      ICLR/NeurIPS: their class files); the content needs no change.
- [ ] For double-blind venues: remove the repository URL from the
      text; upload the repo as anonymised supplementary material
      (`git archive HEAD` minus `.git`), and de-anonymise on
      acceptance.
- [ ] Reproducibility statement: point to `REPRODUCIBILITY.md` and
      `scripts/reproduce_all.sh` (most venues have a dedicated field).
- [ ] Licenses: code MIT; UCI datasets CC-BY 4.0 (cite loaders'
      mirror provenance as in `scripts/run_uci_safe.py`).
- [ ] Optional but valuable: email A. Odrzywołek with the preprint.

## One-paragraph cover summary (paste into forms)

> Odrzywołek (2026) showed a single operator eml(x,y)=exp(x)−ln(y)
> generates all elementary functions and proposed training EML trees
> for symbolic regression, reporting a "depth wall": blind gradient
> recovery collapses beyond depth 4. We show the wall is a mirage:
> under the clamped semantics every trainable implementation uses,
> the reachable function space is 5+ orders of magnitude smaller than
> its syntax — small enough to enumerate completely to depth 4 in a
> minute on a CPU, and to invert analytically at the root for depth-5
> recovery in seconds. On depth-critical targets, verified exact
> recovery is 40/40, 40/40 and 36/40 at depths 3–5 (baselines: ≤1/8);
> the source paper's own showcase target is recovered exactly in 10
> seconds; complete enumeration yields the first minimal-depth
> certificates for the operator. A mean-field analysis shows the
> clamp saturation that defeats gradient descent is the same
> degeneracy that makes enumeration feasible, and measured gradient
> pathologies (explosion to 4×10⁷, exactly-zero dead slots reaching
> 54%) confirm it. The machinery is operator-generic (demonstrated on
> ψ = sinh − arsinh) and deploys as SafePoolGAM, an additive model
> with tail-certified extrapolation that attains the best worst-case
> R² of any compared model on nine UCI extrapolation splits while
> keeping genuine closed-form wins no tree model can produce.
