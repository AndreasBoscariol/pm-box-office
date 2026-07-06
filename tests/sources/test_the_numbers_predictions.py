#!/usr/bin/env python3
from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

from pm_box_office.sources.the_numbers import predictions
from tests.postgres_test_utils import drop_isolated_postgres_schema, make_isolated_postgres_schema


NEWS_HTML = """
<html>
  <body>
    <section id="news">
      <article id="20260705-projections" class="news-item">
        <h1>Weekend projections: Minions off to slow $61.4-million five-day opening</h1>
        <p class="news-date">July 5, 2026</p>
        <img src="/images/movie-stills/Minions-and-Monsters-(2026)-still-3.png" width="720" height="300" alt="Minions & Monsters">
        <p>Here are the official studio projections for the weekend.</p>
        <a href="/box-office-chart/weekend/2026/07/03">
          <img src="/images/news/20260705-weekend-projections.png" width="720" height="520" alt="Weekend top 10 movies">
        </a>
      </article>
      <article id="20260703-predictions" class="news-item">
        <h1>Weekend prediction: Minions struggling going into the weekend</h1>
        <p class="news-date">July 3, 2026</p>
        <p>As the table below shows, the multiplier typically settles lower.</p>
        <img src="/images/news/20260703-Minions.png" width="720" height="190" alt="Minions & Monsters Wednesday opening comparisons">
      </article>
    </section>
  </body>
</html>
"""


PROJECTION_OCR = """
Studio reported weekend box office July 3-5, 2026
Movie Distributor Actual Predicted % Change vs. Prediction
Minions & Monsters Universal $36,400,000 $52,100,000 70%
Toy Story 5 Walt Disney $31,000,000 $44,300,000 -56% 70%
Young Washington Angel Studios $20,847,688 $21,000,000 99%
Top 10 projected vs. predicted $117,086,783 $152,870,708 77%
"""


COMPARISON_OCR = """
Minions & Monsters Wednesday opening comparison
Release Date Wednesday Thursday Weekend Multiplier
Minions & Monsters 7/1/2026 $14,231,110 $10,810,285
Jurassic World Rebirth 7/2/2025 $30,503,855 $25,300,775 $92,016,065 1.65
Despicable Me 4 7/3/2024 $27,202,050 $20,398,275 $75,009,210 1.58
Medians $28,852,953 $22,849,525 $100,026,972 2.08
Predicted Fri-Sun $52,095,281
Predicted Opening $53,000,000
Final opening prediction $54,000,000
"""


class TheNumbersPredictionTests(unittest.TestCase):
    def test_discovers_news_prediction_images_only(self) -> None:
        images = predictions.discover_prediction_images(
            NEWS_HTML,
            page_url="https://www.the-numbers.com/news/",
        )

        self.assertEqual(2, len(images))
        self.assertEqual(
            "https://www.the-numbers.com/images/news/20260705-weekend-projections.png",
            images[0].image_url,
        )
        self.assertEqual(
            "https://www.the-numbers.com/box-office-chart/weekend/2026/07/03",
            images[0].linked_url,
        )
        self.assertEqual("2026-07-05", images[0].article_date)
        self.assertEqual(
            "https://www.the-numbers.com/images/news/20260703-Minions.png",
            images[1].image_url,
        )

    def test_parses_weekend_projection_ocr_rows(self) -> None:
        image = predictions.discover_prediction_images(
            NEWS_HTML,
            page_url="https://www.the-numbers.com/news/",
        )[0]

        rows = predictions.parse_prediction_rows(
            image,
            predictions.ocr_result_from_text(PROJECTION_OCR, confidence=91.5),
        )

        self.assertEqual(3, len(rows))
        self.assertEqual("weekend_projection", rows[0].table_kind)
        self.assertEqual("Minions & Monsters", rows[0].source_movie_title)
        self.assertEqual("Universal", rows[0].distributor)
        self.assertEqual(36400000, rows[0].actual_usd)
        self.assertEqual(52100000, rows[0].predicted_usd)
        self.assertEqual(70.0, rows[0].pct_vs_prediction)
        self.assertEqual(-56.0, rows[1].pct_change)
        self.assertEqual(70.0, rows[1].pct_vs_prediction)
        self.assertEqual(91.5, rows[0].confidence)

    def test_parses_comparison_prediction_ocr_rows(self) -> None:
        image = predictions.discover_prediction_images(
            NEWS_HTML,
            page_url="https://www.the-numbers.com/news/",
        )[1]

        rows = predictions.parse_prediction_rows(
            image,
            predictions.ocr_result_from_text(COMPARISON_OCR),
        )

        summary = [row for row in rows if row.row_label == "Predicted Fri-Sun"][0]
        predicted_opening = [row for row in rows if row.row_label == "Predicted Opening"][0]
        final_prediction = [row for row in rows if row.row_label == "Final Opening Prediction"][0]
        jurassic = [row for row in rows if row.source_movie_title == "Jurassic World Rebirth"][0]

        self.assertEqual("comparison_prediction", summary.table_kind)
        self.assertEqual("predicted_opening", summary.metric)
        self.assertEqual(52095281, summary.predicted_usd)
        self.assertEqual(53000000, predicted_opening.predicted_usd)
        self.assertEqual(54000000, final_prediction.predicted_usd)
        self.assertEqual("opening_comparison", jurassic.table_kind)
        self.assertEqual("2025-07-02", jurassic.release_date)
        self.assertEqual(30503855, jurassic.actual_usd)
        self.assertEqual(92016065, jurassic.weekend_usd)
        self.assertEqual(1.65, jurassic.multiplier)
        self.assertEqual("opening_comparison:4", jurassic.source_row_key)
        self.assertEqual("jurassic world rebirth", jurassic.normalized_movie_title)


class TheNumbersPredictionDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn, self.schema = make_isolated_postgres_schema()
        predictions.initialize_database(self.conn)
        self.conn.execute(
            """
            INSERT INTO movies (movie_url, title, release_year, release_date, updated_at)
            VALUES
                ('https://www.the-numbers.com/movie/Minions-and-Monsters-(2026)', 'Minions & Monsters (2026)', 2026, '2026-07-01', CURRENT_TIMESTAMP),
                ('https://www.the-numbers.com/movie/Jurassic-World-Rebirth-(2025)', 'Jurassic World Rebirth (2025)', 2025, '2025-07-02', CURRENT_TIMESTAMP)
            """
        )
        self.conn.commit()

    def tearDown(self) -> None:
        drop_isolated_postgres_schema(self.conn, self.schema)

    def test_import_prediction_results_persists_images_rows_and_matches_movies(self) -> None:
        image = predictions.discover_prediction_images(
            NEWS_HTML,
            page_url="https://www.the-numbers.com/news/",
        )[0]
        ocr = predictions.ocr_result_from_text(PROJECTION_OCR, confidence=91.5)
        rows = predictions.parse_prediction_rows(image, ocr)

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "fixture.png"
            image_path.write_bytes(b"fake image bytes")
            result = predictions.PredictionImageResult(
                image=image,
                image_cache_path=image_path,
                ocr=ocr,
                rows=rows,
                fetched_at="2026-07-06T00:00:00+00:00",
            )
            for _ in range(2):
                counts = predictions.import_prediction_results(self.conn, [result])
                self.conn.commit()

        self.assertEqual((1, 1, 3), counts)
        stored_rows = self.conn.execute(
            """
            SELECT source_movie_title, predicted_usd, match_status, movie_id IS NOT NULL
            FROM the_numbers_prediction_rows
            ORDER BY row_ordinal
            """
        ).fetchall()
        self.assertEqual(3, len(stored_rows))
        self.assertEqual(("Minions & Monsters", 52_100_000, "matched", True), tuple(stored_rows[0]))
        self.assertEqual(("Young Washington", 21_000_000, "unmatched", False), tuple(stored_rows[2]))

        source_row = self.conn.execute(
            """
            SELECT source, source_title, match_status
            FROM movie_source_ids
            WHERE source = 'the_numbers_predictions'
            """
        ).fetchone()
        self.assertEqual(("the_numbers_predictions", "Minions & Monsters", "matched"), tuple(source_row))


if __name__ == "__main__":
    unittest.main()
