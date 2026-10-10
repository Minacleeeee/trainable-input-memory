# Critical review and claim boundary

This is an AI-assisted internal review, not independent human peer review.

## Publishable observation

One 8 KiB command input changes a previously adapted reader from zero successful held-out skill responses to substantial, imperfect success on command-untrained vocabulary. Paired old responses remain identical under input-only isolation. A tested always-active shared update loses old correct responses. Full saved records permit readers to inspect successes and failures and recompute counts.

## Material weaknesses

1. Prior related-task exposure: the reader had learned three-word rotation and reversal through a maintenance-adapter lineage. Four-word success can bind or reuse existing behavior.
2. Missing baseline: a concise natural-language permutation instruction may work as well or better without training. No comparison has been run, so superiority to document skills is unsupported.
3. Limited independence: 192 transfer queries reuse 16 words; one seed and two vocabulary worlds do not support broad statistical claims.
4. Developmental evaluation: earlier attempts informed later designs. Current panels are not an untouched benchmark across the whole research history.
5. Isolation explains retention: separate new rows and a frozen reader protect paths that do not use those rows. This is not a solution to forgetting under shared updates.
6. Confounded shared comparison: both trainable architecture and group learning rates differ. It does not prove shared adaptation is generally harmful.
7. Storage grows per entity: 8 KiB per synthetic entity is not a fixed-capacity or information-theoretic compression result. The base, adapter and metadata add cost.
8. Deployment equivalence is narrow: the successful FP16 export preserves values already cast to FP16 during the original forward path; CPU conversion previously failed equivalence.
9. Partial reproducibility: released statistics are recomputable, historical sources are inspectable, and parameter files are hash-verified. Clean-machine inference and complete training reproduction were not run for this publication.
10. No long-agent result: context compression, procedural use of factual histories, cross-backbone portability and superiority to text skills remain proposed research.

## Evidence preserved rather than hidden

- Input A still fails 37/192 transfer cases and exceeds the historical 80% gate by only one success.
- Shared A fails 47/192 transfer cases and loses 92 old correct answers.
- An earlier narrower skill attempt obtained 132/192 transfer successes.
- An earlier prospective factual wording test obtained 483/512, below a 95% gate.
- Original result records retain overall_goal_achieved=false.

## Recommendation

Release as a transparent small empirical study and reusable evidence bundle. Do not market it as a new general memory architecture, a replacement for document skills, or a demonstrated solution to long-running agent context loss. Stronger claims require new controlled experiments, which are not included in this release.
