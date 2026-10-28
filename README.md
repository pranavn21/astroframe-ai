# AstroFrame AI

AstroFrame AI is a local AI-assisted astrophotography session analysis tool.

The project will analyze FITS subframes using objective astronomical image metrics & use a locally hosted LLM to explain image-quality anomalies.

## Planned Stack

- Python
- AstroPy
- Photutils
- NumPy
- Pandas
- Streamlit
- Ollama
- GPT-OSS

## Architecture

Development runs on a client machine, while local LLM inference can run on another machine with Ollama.

No cloud LLM API is required!

## Status

Early development

## Guiding logs (Day 4)

```bash
python analyze_session.py sample_data/my_session \
  --guide-log sample_data/PHD2_GuideLog_example.txt \
  --guide-timezone America/Chicago
```

Only supply a timezone after checking the logger's clock settings. Without an
offset or verified timezone, guide timestamps stay unqualified and are excluded
from exposure matching. Ambiguous DST times also stay unavailable.

The parser follows the [PHD2 log format](https://github.com/OpenPHDGuiding/phd2/wiki/PHD2GuideLog).
It reads guiding sections, skips calibration, and warns on malformed rows.It excludes 
DROP, error, AO, and paused records from mount RMS. Raw distances
are pixels; arcseconds are calculated only when a positive pixel scale is logged.
Exports with explicit `ra_error_arcsec` and `dec_error_arcsec` columns retain
those values.

The frame CSV gains exposure bounds, guide sample count, alignment status, and
per-unit RMS and peak errors. Empty windows have zero samples and missing errors.
Normalized samples and events are saved to `outputs/<session_name>_guiding.csv`.
RMS uses paired finite RA/DEC measurements and includes constant offsets.
`--buffer-seconds` extends the guiding window on both sides; the default is zero.
Each unit has its own session median and relative RMS deviation. Guiding more
than 50% above its median sets `high_guiding_rms`, a temporary heuristic.
Missing or zero baselines leave this flag unknown. The combined deviation and
flag prefer arcseconds when available, otherwise pixels.

DATE-OBS is assumed to be exposure start, with EXPTIME in seconds. The available
test frame explicitly uses UTC exposure start. Verify this convention for your
capture software. Missing TIMESYS defaults to UTC; other explicit time scales
are left unmatched. Clock synchronization still needs to be checked.

Timed dither markers flag overlapping exposures. Standard PHD2 dither markers
can be untimed; those are retained without inventing timestamps. In that case,
`dither_during_exposure` is blank when overlap cannot be established. Dither
checks use the actual exposure bounds even if a guiding buffer is requested.

No real guide log is available locally yet. Tests use small synthetic logs;
ASIAIR formatting, timezone, and alignment still need validation on a real log.

```bash
python -m unittest discover -s tests -v
python -m compileall astroframe analyze_frame.py analyze_session.py
```

## Diagnostic events (Day 5)

Session analysis now saves `outputs/<session_name>_events.json` alongside the CSV
and plots. Python groups affected frames and records measurements, possible
causes, missing evidence, and up to two normal frames of context on each side.
Cause labels are possibilities, not diagnoses. They require evidence on the
same frame; missing guiding is never treated as normal guiding.
Large star losses or background changes keep an otherwise guiding-like event
unclear rather than assigning it to guiding alone.

The frame CSV includes evidence flags for guiding, transparency, focus/seeing,
and dither-related possibilities. Background below -25% now has a matching
`low_background` flag. The image and guiding thresholds remain temporary
heuristics and are not universal scientific cutoffs.

The default grouping gap is 1.5 times the median interval between exposure
starts. Set `--event-gap-seconds` to override it. Normal frames are not event
members, though affected frames can share an event across a short normal gap.
Frames without usable dates remain separate undated events.

The events file also contains simple fitted trends for FWHM, star count,
background, and each guiding unit. A trend needs at least three valid samples,
a nonzero baseline, and more than 20% fitted change over the sampled interval.
These describe the available session, not a forecast or a cause.

## Optional local explanations

```bash
export OLLAMA_HOST="http://localhost:11434"
export REASONING_MODEL="gpt-oss:20b"
python analyze_session.py sample_data/my_session \
  --guide-log sample_data/PHD2_GuideLog_example.txt \
  --guide-timezone America/Chicago --diagnose
```

Set `OLLAMA_HOST` to your own server for LAN inference. Environment variables
remain the configuration source. Without `--diagnose`, no Ollama requests are
made, and the full deterministic analysis works offline.

With `--diagnose`, AstroFrame checks that Ollama and the selected model are
available, then sends one summarized event at a time. The model only explains
precomputed evidence. Its [JSON output](https://docs.ollama.com/capabilities/structured-outputs)
must have the required fields, valid confidence, and an allowed cause label.
Numeric claims must occur in the event evidence, allowing simple rounding.
These checks cannot prove that prose uses the right metric or explains a cause
correctly; explanations still need comparison with the original evidence.

Results are saved incrementally to `outputs/<session_name>_diagnoses.json`.
Each result retains its event evidence. Failed calls keep an error entry and
do not discard earlier results or deterministic outputs. Only final model
content is used; hidden reasoning is not saved. No automatic retries are made.

Tests use generated FITS images, synthetic logs, and mocked Ollama responses.
Live GPT-OSS and a real multi-frame session with a matching guide log have not
been validated yet. Those remain manual, opt-in checks before trusting diagnoses.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m unittest discover -s tests -v
python -m compileall astroframe analyze_frame.py analyze_session.py
python analyze_frame.py sample_data/test_light.fit
```

To check your configured Ollama server manually:

```bash
python -c 'from astroframe.llm import check_ollama_connection; print(check_ollama_connection())'
```

## Day 7 testing

Tested on macOS with Python 3.14.4 using 106 Veil Nebula
(NGC 6960) light frames. Each exposure was 300 seconds.
The two nights were analyzed separately.

| Session    | Frames analyzed | Failed frames | Events |
| ---------- | --------------: | ------------: | -----: |
| 2025-06-21 |              54 |             0 |      1 |
| 2025-06-22 |              52 |             0 |      0 |

- All 48 automated tests passed. Six warnings came from test images with no detected stars.
- Dependency and compilation checks passed.
- CSV, JSON, and timeline plots were generated. Frame counts, timestamp ordering, and event membership checks passed.
- Repeating the first session produced identical metrics and event files.
- The synthetic control produced the expected event with a possible guiding cause.
- When Ollama was unavailable, the analysis outputs were preserved and the diagnosis error was saved.

The first night's flagged frame had fewer detected stars and a higher
background. Its cause remained `insufficient_evidence`.

Live Ollama testing, real guiding-log correlation, and manual checks
against the images are still pending. These results confirm that the
pipeline runs consistently, but do not establish measurement accuracy.
