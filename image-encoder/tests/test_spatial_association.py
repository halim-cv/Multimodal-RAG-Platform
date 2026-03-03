"""
Unit tests for spatial association logic in document_understanding_engine.py

Tests cover:
- calculate_distance()  →  vertical/horizontal proximity
- associate_figures_with_captions()  →  greedy one-to-one matching
- crop_and_combine()  →  output shape and dtype
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make sure the image-encoder root is on the path so imports work when running
# from the repo root with:  pytest image-encoder/tests/
sys.path.insert(0, str(Path(__file__).parent.parent))

from document_understanding.document_understanding_engine import (
    calculate_distance,
    crop_and_combine,
)


# ─────────────────────────────────────────────────────────────────────────────
# calculate_distance
# ─────────────────────────────────────────────────────────────────────────────

class TestCalculateDistance:
    """Tests for the spatial proximity function."""

    def test_caption_directly_below(self):
        """Caption directly below figure → small vertical distance, direction='below'."""
        figure_bbox  = [10, 10, 200, 100]   # figure  y: 10→100
        caption_bbox = [10, 105, 200, 130]  # caption y: 105→130  (gap = 5 px)
        v_dist, h_prox, direction = calculate_distance(figure_bbox, caption_bbox)
        assert direction == "below"
        assert v_dist == pytest.approx(5.0)
        assert h_prox <= 0  # full horizontal overlap → negative value

    def test_caption_directly_above(self):
        """Caption above figure → direction='above'."""
        figure_bbox  = [10, 110, 200, 200]  # figure  y: 110→200
        caption_bbox = [10,  10, 200, 100]  # caption y: 10→100  (gap = 10 px)
        v_dist, h_prox, direction = calculate_distance(figure_bbox, caption_bbox)
        assert direction == "above"
        assert v_dist == pytest.approx(10.0)

    def test_overlapping_vertically_returns_inf(self):
        """Vertically overlapping boxes → infinity (not a valid caption)."""
        figure_bbox  = [10, 10, 200, 200]
        caption_bbox = [10, 50, 200, 150]   # overlaps vertically
        v_dist, h_prox, direction = calculate_distance(figure_bbox, caption_bbox)
        assert v_dist == float("inf")
        assert direction == "overlap"

    def test_horizontal_gap_no_overlap(self):
        """Caption is below but horizontally to the right — large horizontal distance."""
        figure_bbox  = [10,  10, 100, 80]   # x: 10→100
        caption_bbox = [200, 90, 300, 110]  # x: 200→300  (gap = 100 px)
        v_dist, h_prox, direction = calculate_distance(figure_bbox, caption_bbox)
        assert direction == "below"
        assert h_prox == pytest.approx(100.0)  # distance between x=100 and x=200

    def test_partial_horizontal_overlap(self):
        """Partial horizontal overlap → negative proximity."""
        figure_bbox  = [10,  10, 200, 80]   # x: 10→200
        caption_bbox = [150, 90, 300, 110]  # x: 150→300  overlap = 200-150 = 50 px
        v_dist, h_prox, direction = calculate_distance(figure_bbox, caption_bbox)
        assert direction == "below"
        assert h_prox == pytest.approx(-50.0)


# ─────────────────────────────────────────────────────────────────────────────
# crop_and_combine
# ─────────────────────────────────────────────────────────────────────────────

class TestCropAndCombine:
    """Tests for the crop-and-stack utility."""

    def _make_page(self, h=500, w=400):
        """Create a blank white page image."""
        return np.full((h, w, 3), 255, dtype=np.uint8)

    def test_output_dtype_is_uint8(self):
        """Combined image must be uint8 to avoid PIL save errors."""
        page = self._make_page()
        main_bbox    = [10, 10, 200, 150]
        caption_bbox = [10, 160, 200, 200]
        result = crop_and_combine(main_bbox, caption_bbox, page, "figure-caption", 0)
        assert result["combined_image"].dtype == np.uint8

    def test_output_shape_three_channels(self):
        """Output must be RGB (3-channel)."""
        page = self._make_page()
        result = crop_and_combine([10, 10, 200, 150], [10, 160, 200, 200], page, "figure-caption", 0)
        assert result["combined_image"].ndim == 3
        assert result["combined_image"].shape[2] == 3

    def test_combined_height_equals_sum_plus_separator(self):
        """Height = main_height + 20 (separator) + caption_height."""
        page = self._make_page()
        main_h    = 140   # 150 - 10
        caption_h =  40   # 200 - 160
        result = crop_and_combine([10, 10, 200, 150], [10, 160, 200, 200], page, "figure-caption", 0)
        expected_h = main_h + 20 + caption_h
        assert result["combined_image"].shape[0] == expected_h

    def test_width_is_max_of_both(self):
        """Output width = max(main_width, caption_width)."""
        page = self._make_page(h=500, w=600)
        main_bbox    = [10,  10, 300, 100]   # width = 290
        caption_bbox = [10, 110, 500, 150]   # width = 490  (wider)
        result = crop_and_combine(main_bbox, caption_bbox, page, "figure-caption", 0)
        assert result["combined_image"].shape[1] == 490

    def test_metadata_fields_present(self):
        """Return dict must contain all required keys."""
        page = self._make_page()
        result = crop_and_combine([10, 10, 200, 150], [10, 160, 200, 200], page, "table-caption", 2)
        for key in ["combined_image", "pair_type", "page_number", "main_bbox", "caption_bbox"]:
            assert key in result
        assert result["pair_type"] == "table-caption"
        assert result["page_number"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# Integration: associate_figures_with_captions (without model — pure logic)
# ─────────────────────────────────────────────────────────────────────────────

class TestFigureCaptionAssociation:
    """
    Tests for the greedy one-to-one figure→caption matching.
    Uses the engine's association method with a mock detections DataFrame —
    no model loading required.
    """

    def _make_detections(self, rows):
        """Build a DataFrame matching the engine's expected schema."""
        return pd.DataFrame(rows)

    def _make_engine(self):
        """Import the engine without loading the model."""
        from document_understanding.document_understanding_engine import DocumentUnderstandingEngine
        return DocumentUnderstandingEngine()   # _is_loaded = False → no model download

    def test_single_figure_single_caption(self):
        engine = self._make_engine()
        df = self._make_detections([
            {"page_number": 0, "bbox": [10,  10, 200, 100], "class_id": 0, "class_name": "figure"},
            {"page_number": 0, "bbox": [10, 110, 200, 140], "class_id": 1, "class_name": "figure_caption"},
        ])
        pairs = engine.associate_figures_with_captions(df, prefer_below=True)
        assert len(pairs) == 1
        assert pairs[0]["figure_bbox"] == [10, 10, 200, 100]

    def test_no_matching_caption_returns_empty(self):
        """Figure with no caption on the same page → no pairs."""
        engine = self._make_engine()
        df = self._make_detections([
            {"page_number": 0, "bbox": [10, 10, 200, 100], "class_id": 0, "class_name": "figure"},
        ])
        pairs = engine.associate_figures_with_captions(df, prefer_below=True)
        assert pairs == []

    def test_caption_above_ignored_when_prefer_below(self):
        """Caption above the figure is ignored when prefer_below=True."""
        engine = self._make_engine()
        df = self._make_detections([
            {"page_number": 0, "bbox": [10, 50,  200, 150], "class_id": 0, "class_name": "figure"},
            {"page_number": 0, "bbox": [10,  5,  200,  40], "class_id": 1, "class_name": "figure_caption"},
        ])
        pairs = engine.associate_figures_with_captions(df, prefer_below=True)
        assert pairs == []

    def test_each_caption_used_at_most_once(self):
        """Two figures compete for the same caption — only the closer one wins."""
        engine = self._make_engine()
        df = self._make_detections([
            # figure A is 5 px above the caption  (closer)
            {"page_number": 0, "bbox": [10,  10, 200, 95],  "class_id": 0, "class_name": "figure"},
            # figure B is 50 px above the caption (farther)
            {"page_number": 0, "bbox": [10,  10, 200, 50],  "class_id": 0, "class_name": "figure"},
            {"page_number": 0, "bbox": [10, 100, 200, 130], "class_id": 1, "class_name": "figure_caption"},
        ])
        pairs = engine.associate_figures_with_captions(df, prefer_below=True)
        # Only 1 pair can exist (caption is exclusive)
        assert len(pairs) == 1

    def test_two_figures_two_captions_correct_pairing(self):
        """Two figures each have their own caption directly below — both paired correctly."""
        engine = self._make_engine()
        df = self._make_detections([
            # left figure + its caption
            {"page_number": 0, "bbox": [10,  10, 200, 100], "class_id": 0, "class_name": "figure"},
            {"page_number": 0, "bbox": [10, 105, 200, 130], "class_id": 1, "class_name": "figure_caption"},
            # right figure + its caption (different x range)
            {"page_number": 0, "bbox": [250,  10, 450, 100], "class_id": 0, "class_name": "figure"},
            {"page_number": 0, "bbox": [250, 105, 450, 130], "class_id": 1, "class_name": "figure_caption"},
        ])
        pairs = engine.associate_figures_with_captions(df, prefer_below=True)
        assert len(pairs) == 2
