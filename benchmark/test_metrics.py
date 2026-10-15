"""Regression checks for comparisons that could otherwise silently give wrong rankings."""
import tempfile
import json
import unittest
from pathlib import Path
from report import components,latest_records,aggregate

class MetricsTest(unittest.TestCase):
    def test_batch_forward_is_not_divided_by_steps(self):
        result={'events':[{'stage':'sampling','ms':100},
            {'stage':'dit_forward','ms':10,'shapes':[[4,50,32]]},
            {'stage':'dit_forward','ms':8,'shapes':[[4,50,32]]},
            {'stage':'dit_forward','ms':3,'shape':[1,50,32]}]}
        c=components(result)
        self.assertEqual(c['sampling_ms'],100)
        self.assertEqual(c['forward_b4_ms'],8)
        self.assertEqual(c['forward_b1_ms'],3)
        self.assertEqual(c['forward_calls'],3)
    def test_reference_parent_is_not_double_counted(self):
        c=components({'events':[{'stage':'reference_total','ms':30},
                    {'stage':'codec_encode','ms':20},{'stage':'speaker_encode','ms':5}]})
        self.assertEqual(c['reference_encode_ms'],30)
    def test_old_python_profile_does_not_replace_current_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'data.jsonl'
            base={'key':'same','kind':'profile','status':'ok','case':{'engine':'Irodori-TTS'}}
            path.write_text('\n'.join(json.dumps(dict(base,instrumentation_version=v)) for v in (3,2)))
            self.assertEqual(latest_records(path)[0]['instrumentation_version'],3)
    def test_decode_includes_host_transfer(self):
        c=components({'events':[{'stage':'codec_decode','ms':10}], 'stages':{'decode_latent':.013}})
        self.assertEqual(c['decode_ms'],13)
    def test_codec_preprocessing_is_not_counted_twice(self):
        c=components({'events':[{'stage':'reference_total','ms':30},
            {'stage':'codec_encode','ms':20},{'stage':'reference_preprocess_in_codec','ms':5},
            {'stage':'speaker_encode','ms':4}]})
        self.assertEqual(c['reference_encode_ms'],30)
        self.assertEqual(c['reference_codec_ms'],15)
        self.assertEqual(c['reference_preprocess_ms'],5)
    def test_wall_profile_and_idle_are_separate(self):
        case=dict(case='test',engine='test',variant='rf',backend='cuda',precision='fp32',precision_profile='fp32')
        common=dict(status='ok',case=case,cache='reencode',mode='matched',prompt=dict(id='p01',tokens=10,characters=20,bucket='8-15'))
        records=[]
        for i in range(5):
            records.append(dict(common,key=f'g/wall/{i}',kind='wall',seconds=2,
                result=dict(ms=10+i,samples=96000,sample_rate=48000,frames=50,rtf=(10+i)/2000,peak_vram=200*2**20,vram_method='nvml_process')))
            records.append(dict(common,key=f'g/profile/{i}',kind='profile',seconds=2,result=dict(ms=9999,events=[dict(stage='sampling',ms=7)])))
        records.append(dict(common,key='g/idle',kind='idle',result=dict(resident_vram=100*2**20)))
        row=aggregate(records)[0]
        self.assertEqual(row['warm_ms'],12)
        self.assertEqual(row['sampling_ms'],7)
        self.assertEqual(row['resident_mib'],100)
        self.assertTrue(row['complete'])

if __name__=='__main__':unittest.main()
