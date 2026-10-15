#!/usr/bin/env python3
"""Encode the generated reference WAV using the pinned official codec."""
from pathlib import Path
import soundfile as sf
import torch
from irodori_tts.codec import DACVAECodec

root = Path(__file__).resolve().parents[1]
out = root / "artifacts/voices"
out.mkdir(parents=True, exist_ok=True)
audio, rate = sf.read(root / "artifacts/smoke/original-rf.wav", dtype="float32")
sf.write(out / "ref1.wav", audio, rate, subtype="PCM_16")
torch.set_num_threads(8)
codec = DACVAECodec.load(repo_id=str(root / "models/codec/weights.pth"), device="cpu", dtype=torch.float32)
wave = torch.from_numpy(audio).view(1, 1, -1)
with torch.inference_mode():
    latent = codec.encode_waveform(wave, sample_rate=rate, normalize_db=-16.0, ensure_max=True)
torch.save(latent.cpu().squeeze(0).contiguous(), out / "ref1_lat.pt")
print("Reference", rate, len(audio), "latent", tuple(latent.shape))
