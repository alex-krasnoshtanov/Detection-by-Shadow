# Results

Submission files, oldest first.

| File | Produced by | Usable? | Score |
| --- | --- | --- | --- |
| `submission_example.csv` | the challenge organisers | reference | — |
| `submission_v1_direct.csv` | v1, direct box regression | **no, malformed** | not recorded |
| `submission_v2_giou.csv` | v2, direct regression fully tuned | yes | not recorded |
| `submission_v5_ensemble.csv` | retrained 3-seed ensemble, 2026-09 | yes | not submitted |

`submission_example.csv` is the schema and, more importantly, the **required id
order**.

## `submission_v5_ensemble.csv`

Regenerated from scratch with the recipe in this repository — three seeds
(42/123/777) on all 1692 annotated frames, 40 epochs each, averaged in
decomposed space with flip TTA. The corresponding weights are published as
[`weights-v1`](https://github.com/alex-krasnoshtanov/Detection-by-Shadow/releases/tag/weights-v1).

It is not scored: the challenge leaderboard closed in April 2026 and the test
labels were never released. What can be said is that a sibling run holding out
254 frames scored **0.6096 mean IoU** with side classification 254/254 correct
(see [`docs/experiments.md`](../docs/experiments.md#filling-in-the-missing-comparison)),
and that this file is well-formed by the same checks that the v1 file fails:

- 414 rows in the sample submission's exact id order
- no NaN coordinates, no inverted boxes
- every box fully outside the frame, as every annotated person is
- `direction` abstains on 407 of 414 — the shadow does not carry that signal

## `submission_v1_direct.csv` is broken, and instructively so

Kept precisely because it is. Two independent faults, neither of which raises
an error anywhere:

**No id column.** It was written with `to_csv()` including the DataFrame index,
so the first column is an unnamed integer — and not even a sequential one
(`0, 330, 370, 292, ...`), because the rows came out of a shuffled loader. There
is nothing tying any row to an image. The file cannot be scored at all.

**Every box is inside-out.** All 414 rows have `xmin >= xmax` — the first row
reads `xmin=942.3, xmax=562.2`. The model's two x outputs had swapped roles and
nothing downstream noticed, because a CSV of floats looks fine.

This is the concrete reason
[`write_submission`](../src/shadow_detection/predict.py) validates rather than
trusts. It now rejects wrong columns, ids that do not match the sample
submission in order, and NaN coordinates; `to_submission` floors physically
impossible values; and [`blend`](../src/shadow_detection/blend.py) warns about
inverted and not-fully-off-frame boxes. Every one of those checks exists because
of a failure that actually happened, in a competition where the feedback loop is
a leaderboard number several hours later.

`submission_v2_giou.csv` is well-formed: proper `id` column, no inverted boxes,
and `direction = -1` throughout — validation direction accuracy was 46.3%
against a 48.3% base rate, so declining to answer beat guessing.

## The original v4 and v5 submissions are gone

They were written to the university GPU server and not retrieved before access
ended. Only their leaderboard scores survive, recorded in the last cell of
[notebook 05](../notebooks/05_v5_ensemble.ipynb) and tabulated in
[`docs/experiments.md`](../docs/experiments.md). The file above is a
reproduction, not the artifact that scored 0.626.
