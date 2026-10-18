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

Development is performed on a client machine while local LLM inference can be hosted on another machine running Ollama.

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
It reads guiding sections, skips calibration, and warns on malformed rows.
DROP, error, AO, and paused records are excluded from mount RMS. Raw distances
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
