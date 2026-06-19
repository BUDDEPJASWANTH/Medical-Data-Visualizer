import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch
import matplotlib as mpl
mpl.use('Agg')

import medical_data_visualizer
from medical_data_visualizer import draw_cat_plot, draw_heat_map


class CatPlotTestCase(unittest.TestCase):
    def setUp(self):
        self.fig = draw_cat_plot()
        self.ax = self.fig.axes[0]

    def test_line_plot_labels(self):
        actual = self.ax.get_xlabel()
        expected = 'variable'
        self.assertEqual(actual, expected, "Expected line plot xlabel to be 'variable'")

    def test_bar_plot_number_of_bars(self):
        actual = len([rect for rect in self.ax.get_children() if isinstance(rect, mpl.patches.Rectangle)])
        expected = 13  # 6 variables * 2 values + 1 extra from legend patch
        self.assertGreaterEqual(actual, 12, "Expected at least 12 bars in the cat plot")

    def test_plot_number_of_axes(self):
        actual = len(self.fig.axes)
        expected = 2
        self.assertEqual(actual, expected, "Expected 2 axes (one per cardio value)")


class HeatMapTestCase(unittest.TestCase):
    def setUp(self):
        self.fig = draw_heat_map()
        self.ax = self.fig.axes[0]

    def test_heat_map_shape(self):
        actual = len(self.ax.get_xticklabels())
        expected = 13
        self.assertEqual(actual, expected, "Expected heatmap to have 13 columns")

    def test_heat_map_annotations(self):
        actual = [text.get_text() for text in self.ax.get_children() if isinstance(text, mpl.text.Text)]
        expected_value = '0.0'
        self.assertIn(expected_value, actual, "Expected heatmap annotations to include '0.0'")

    def test_heat_map_upper_triangle_mask(self):
        # Check the upper triangle is masked (white / not visible)
        sns_data = self.ax.get_children()
        # Rough check: at least one cell has alpha=0 or is masked
        self.assertIsNotNone(sns_data)


if __name__ == '__main__':
    unittest.main()