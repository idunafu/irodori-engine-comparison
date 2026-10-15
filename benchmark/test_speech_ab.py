"""Check that the expansion gate distinguishes a reproducible change from round noise."""
import unittest
from speech_ab import summarize

def measurements(factors):
    records=[]
    for version in ('old','new'):
        for round_id in range(3):
            common=dict(status='ok',phase='screen',case=dict(variant='rf',backend='cuda',version=version),
                cache='native_cache',prompt=dict(id='p01'),seconds=8.,round=round_id)
            records.append(dict(common,kind='idle',result=dict(resident_vram=1024*2**20)))
            for rep in range(5):
                ms=(100+round_id)* (1 if version=='old' else factors[round_id])
                records.append(dict(common,kind='wall',result=dict(ms=ms,peak_vram=1024*2**20)))
                records.append(dict(common,kind='profile',result=dict(events=[])))
    return records

class ExpansionTest(unittest.TestCase):
    def test_reproducible_speedup_expands(self):
        r=summarize(measurements([.9,.9,.9]))[0]
        self.assertTrue(r['complete']);self.assertTrue(r['speed_difference'])
        self.assertLess(r['ci_high_pct'],0)

    def test_equal_times_do_not_expand(self):
        self.assertFalse(summarize(measurements([1,1,1]))[0]['difference'])

    def test_round_direction_reversal_does_not_expand(self):
        self.assertFalse(summarize(measurements([.8,1.2,.8]))[0]['speed_difference'])

    def test_small_reproducible_change_is_still_expanded(self):
        r=summarize(measurements([1.005,1.005,1.005]))[0]
        self.assertTrue(r['speed_difference']);self.assertFalse(r['practical_speed_difference'])

if __name__=='__main__':unittest.main()
