# Results

The submission files that survive locally.

| File | Produced by | Usable? | Score |
| --- | --- | --- | --- |
| `submission_example.csv` | the challenge organisers | reference | — |
| `submission_v1_direct.csv` | v1, direct box regression | **no, malformed** | not recorded |
| `submission_v2_giou.csv` | v2, direct regression fully tuned | yes | not recorded |

`submission_example.csv` is the schema and, more importantly, the **required id
order**.

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
submission in order, and NaN coordinates — and
[`blend`](../src/shadow_detection/blend.py) separately warns about inverted and
not-fully-off-frame boxes. Every one of those checks exists because of a
failure that actually happened, in a competition where the feedback loop is a
leaderboard number several hours later.

`submission_v2_giou.csv` is well-formed: proper `id` column, no inverted boxes,
and `direction = -1` throughout — validation direction accuracy was 46.3%
against a 48.3% base rate, so declining to answer beat guessing.

## The good submissions are gone

The v4 and v5 files — including the 0.626 three-seed ensemble, the best result
this repository can evidence — were written to the university GPU server and
not retrieved before access ended. Only their leaderboard scores survive,
recorded in the last cell of
[notebook 05](../notebooks/05_v5_ensemble.ipynb) and tabulated in
[`docs/experiments.md`](../docs/experiments.md).

So both files here are from the superseded direct-regression architecture. They
are historical artifacts, not anything to reproduce.
