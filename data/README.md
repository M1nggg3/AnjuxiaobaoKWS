# AnJuXiaoBaoKWS training data layout

This directory stores audio data and generated WeKWS training manifests for the
"安居小宝" keyword spotting project.

## Directory layout

```text
data/
  raw/
    rk3566_positive/      # Real positive samples recorded from RK3566.
    rk3566_negative/      # Real negative samples recorded from RK3566.
    tts_positive/         # TTS positive samples for smoke tests.
    tts_negative/         # TTS or synthetic negative samples, optional.
    noise/                # Background noise clips for augmentation/evaluation.
  metadata/
    speakers.csv          # Speaker registry.
    recordings.csv        # Recording-level metadata.
  manifests/
    rk3566_pull_manifest.csv
  prepared/
    train/                # WeKWS train split.
    dev/                  # WeKWS validation split.
    test/                 # WeKWS test split.
  templates/
    wav.scp.example
    text.example
    wav.dur.example
    data.list.example
```

## Raw data rules

- Audio format: WAV PCM, 16 kHz, mono, 16-bit is preferred.
- Positive sample: contains the keyword "安居小宝".
- Negative sample: must not contain "安居小宝".
- Keep raw files immutable after collection. If audio needs conversion, write the
  converted files into a generated directory or directly into `prepared/`.
- File names should be stable and unique.

Recommended naming:

```text
rkpos_spk001_normal_0001.wav
rkpos_spk001_fast_0001.wav
rkneg_spk001_office_0001.wav
ttspos_voice001_normal_0001.wav
```

## WeKWS prepared split files

Each split under `prepared/train`, `prepared/dev`, and `prepared/test` should be
generated with these files:

```text
wav.scp     sample_id absolute_or_project_relative_wav_path
text        sample_id transcript
wav.dur     sample_id duration_seconds
data.list   JSONL file consumed by WeKWS
```

Do not put example/comment lines into real `wav.scp`, `text`, `wav.dur`, or
`data.list` files. Keep examples under `templates/`.

## Initial small-scale validation target

For the first real-data validation round:

- Positive samples: 20 colleagues x 50-80 clips = about 1,000-1,600 clips.
- Negative samples: 3-6 hours minimum, 5-10 hours preferred.
- Split suggestion: 80% train, 10% dev, 10% test.
- Keep dev/test speakers as independent as possible from train speakers.
