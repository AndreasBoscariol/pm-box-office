# The Numbers Prediction Image OCR

The Numbers news articles sometimes publish prediction/projection tables as PNG
images. The fastest reliable pipeline is:

1. Fetch the home/news page once and discover candidate `/images/news/*.png` tables
   from articles whose title/context mentions prediction, projection, or
   forecast.
2. Cache every image by URL hash so reruns do not re-hit the site.
3. OCR only the cached table images.
4. Parse the OCR text into structured rows while preserving raw OCR text,
   confidence, image URL, article date, and linked chart URL for audit.
5. Review low-confidence rows before loading them into model features or a
   database table.

Useful commands:

```bash
pm-box-office-the-numbers-predictions --list-images --format json
pm-box-office-the-numbers-predictions --format csv --output data/raw/the_numbers_predictions/predictions.csv
```

Local OCR uses the external `tesseract` binary if installed:

```bash
brew install tesseract
pm-box-office-the-numbers-predictions --ocr tesseract --tesseract-psm 6
```

For development or manual QA, put precomputed OCR text files in a directory and
key them by either the cached image stem or original image filename stem:

```bash
pm-box-office-the-numbers-predictions \
  --ocr-text-dir data/raw/the_numbers_predictions/ocr_text \
  --format json
```

Reliability notes:

- Keep raw image, raw OCR, parsed row, confidence, and source URLs together.
- Prefer the linked HTML chart for actuals when available; use the PNG for the
  forecast/predicted columns that are not exposed as HTML.
- Treat aggregate rows such as `Top 10 projected vs. predicted` as QA checks,
  not movie-level predictions.
- If Tesseract struggles on a new layout, generate text with a stronger vision
  OCR service and feed it through `--ocr-text-dir`; the parser and cache remain
  the same.
