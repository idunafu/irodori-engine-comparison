"""Check that partial or inconsistent screen evidence cannot trigger expansion."""
import unittest
import runtime_ab as ab


def measurements(ratios):
    result = []
    for version in ('old', 'new'):
        for round_id, ratio in enumerate(ratios):
            for kind in ('wall', 'profile', 'idle'):
                for rep in range(1 if kind == 'idle' else 5):
                    value = 100 * (ratio if version == 'new' else 1)
                    result.append(dict(phase='screen', case=dict(base_case='test', version=version),
                        cache='reencode', mode='screen', prompt=dict(id='p01', tokens=15), seconds=2.,
                        kind=kind, round=round_id,
                        result=dict(ms=value, samples=96000, rtf=value/2000,
                                    peak_vram=1024, resident_vram=512, events=[])))
    return result


class DecisionTests(unittest.TestCase):
    def test_consistent_change(self):
        row, = ab.summarize(measurements([.9, .9, .9]))
        self.assertTrue(row['complete'])
        self.assertTrue(row['speed_difference'])
        self.assertAlmostEqual(row['delta_pct'], -10)

    def test_inconsistent_rounds(self):
        row, = ab.summarize(measurements([.9, 1.1, .9]))
        self.assertFalse(row['speed_difference'])

    def test_partial_data(self):
        row, = ab.summarize(measurements([.9, .9, .9])[:-2])
        self.assertFalse(row['complete'])
        self.assertNotIn('speed_difference', row)

    def test_full_matrix_matches_original(self):
        self.assertEqual(len(ab.groups('expanded')), 130)
        self.assertEqual(len(ab.CASES), 8)
        self.assertEqual(len(ab.groups('screen')), 8)


if __name__ == '__main__':
    unittest.main()
