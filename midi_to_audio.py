import os
import subprocess
from pathlib import Path

MIDI_DIR  = './outputs/midi'
AUDIO_DIR = './outputs/audio'
SOUNDFONT = './data/soundfont.sf3'

os.makedirs(AUDIO_DIR, exist_ok=True)

midi_files = list(Path(MIDI_DIR).rglob('*.mid'))
print(f"Converting {len(midi_files)} MIDI files...")

env = os.environ.copy()
env['SDL_AUDIODRIVER'] = 'dummy'
env['PULSE_SERVER']    = '/dev/null'

for midi_path in midi_files:
    rel      = midi_path.relative_to(MIDI_DIR)
    wav_path = Path(AUDIO_DIR) / rel.with_suffix('.wav')
    mp3_path = Path(AUDIO_DIR) / rel.with_suffix('.mp3')
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    # MIDI -> WAV: midi file before -F flag
    cmd_wav = [
        'fluidsynth', '-ni',
        '-T', 'wav',
        '-r', '44100',
        SOUNDFONT,
        str(midi_path),
        '-F', str(wav_path),
    ]
    r1 = subprocess.run(cmd_wav, capture_output=True, env=env)

    if not wav_path.exists() or wav_path.stat().st_size == 0:
        print(f"  FAIL (wav): {midi_path.name}")
        print(f"    stderr: {r1.stderr.decode()[:120]}")
        continue

    # WAV -> MP3
    cmd_mp3 = ['ffmpeg', '-y', '-i', str(wav_path),
                '-codec:a', 'libmp3lame', '-q:a', '2',
                str(mp3_path)]
    r2 = subprocess.run(cmd_mp3, capture_output=True, env=env)

    if r2.returncode == 0:
        os.remove(wav_path)
        print(f"  OK: {mp3_path}")
    else:
        print(f"  FAIL (mp3): {midi_path.name}")

print("Done.")
