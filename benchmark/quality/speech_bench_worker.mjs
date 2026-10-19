// Import the pinned upstream definitions, so Japanese normalization is not reimplemented.
import readline from 'node:readline';
import fs from 'node:fs';
import { readWav } from '../../artifacts/quality/assets/speech-bench/src/core/wav.ts';
import { countHeardErrors, heardAsSaid } from '../../artifacts/quality/assets/speech-bench/src/measure/heard.ts';
import { medianPitch } from '../../artifacts/quality/assets/speech-bench/src/analysis/pitch.ts';
import { voicedSeconds } from '../../artifacts/quality/assets/speech-bench/src/analysis/speaker.ts';

console.log('QUALITY '+JSON.stringify({type:'ready'}));
for await (const line of readline.createInterface({input:process.stdin})) {
  const r=JSON.parse(line);
  if(r.type==='stop')break;
  try {
    const result={type:'result',id:r.id};
    if(r.transcript!==undefined){
      const count=countHeardErrors(r.text,r.transcript,'ja-JP');
      result.heard_errors=count.errors;
      result.heard_reference_length=count.referenceLength;
      result.heard_as_said=heardAsSaid(r.text,r.transcript,'ja-JP');
      const {editDistance}=await import('../../artifacts/quality/assets/speech-bench/src/measure/scoring.ts');
      const {heardUnits}=await import('../../artifacts/quality/assets/speech-bench/src/measure/quality-heard.ts');
      result.heard_raw_errors=editDistance(heardUnits(r.text,'ja-JP'),heardUnits(r.transcript,'ja-JP'));
    }
    if(r.wav){
      const pcm=readWav(fs.readFileSync(r.wav));
      result.median_pitch_hz=medianPitch(pcm);result.voiced_seconds=voicedSeconds(pcm);
    }
    console.log('QUALITY '+JSON.stringify(result));
  } catch(e){console.log('QUALITY '+JSON.stringify({type:'error',id:r.id,error:String(e)}));}
}
