import unittest
from streamlit.testing.v1 import AppTest


class DashboardTests(unittest.TestCase):
    """Exercise dashboard controls and metadata assignment safeguards."""
    # Verify samples modes and inference.
    def test_samples_modes_and_inference(self):
        app = AppTest.from_file('app.py').run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        app.button[0].click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.metric[1].value, '4 / 4')
        self.assertFalse(app.toggle[0].value)
        study = next(s for s in app.selectbox if s.label.startswith('Select a'))
        study.select_index(1).run()
        self.assertEqual(len(app.exception), 0)
        preprocessing = next(s for s in app.selectbox if s.label == 'Preprocessing')
        preprocessing.select('scaled_8bit').run()
        self.assertEqual(len(app.exception), 0)
        app.button[0].click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertGreater(float(app.metric[0].value), 0)
        app.radio[0].set_value('Upload a study').run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.button[0].disabled)
        self.assertEqual(len(app.metric), 0)

    # Verify rejects screenshot view assignments.
    def test_rejects_screenshot_view_assignments(self):
        from catalog import assignment_errors, metadata_index
        wrong = {'L-CC': '31f9bcd6e042eb0f047eaef8a4d71c97.png',
                 'R-CC': '040fb9e96d6619afdb69aae3893bebcb.png',
                 'L-MLO': '5335b81f00d4608d8ab9d6cbdb73b965.png',
                 'R-MLO': 'ae3b2767096883cfd24cb03f12bb6674.png'}
        index = metadata_index()
        self.assertEqual(len(assignment_errors(wrong, index)), 4)
        correct = {index[name][1]: name for name in wrong.values()}
        self.assertEqual(assignment_errors(correct, index), [])


if __name__ == '__main__':
    unittest.main()
