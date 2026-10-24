# Ballroom music classifier

Tell which ballroom dance a song fits from its audio, so you can find new music to dance to.
Point it at a song (or a list of titles) and it returns the dances ranked by probability, the
tempo in MPM (bars per minute), and whether that tempo meets each style's competition rule.

```
$ ballroom-classify waltz_177bpm.wav --model tempo
waltz_177bpm.wav  (tracker: ~89 BPM, sounds like 3/4; model: tempo)
  1. Waltz           0.68   29.7 MPM (89 BPM)
     International Standard (Waltz 28-30): fits; American Smooth (Waltz 28-30): fits
  2. Viennese waltz  0.27   59.4 MPM (178 BPM)
     International Standard (Viennese waltz 58-60): fits; American Smooth (Viennese waltz 53-54): fast
  3. Jive            0.05   44.6 MPM (178 BPM)
     International Latin (Jive 42-44): close
```

(A synthetic 177 BPM track through the tempo-only model. The beat tracker halved the tempo,
which is why Waltz edges out Viennese waltz: exactly the octave confusion the MERT model fixes.)

## Dances and styles

The model predicts the **dance**; the **style** comes from the tempo rules. Smooth and Standard
Waltz, Tango and Foxtrot are danced to the same recordings, so audio can't tell them apart, but
the tempo can say which rule a track meets (Viennese waltz is the clear case: Standard 58-60
MPM, Smooth 53-54).

| Style | Dances |
|---|---|
| International Standard | Waltz, Tango, Viennese waltz, (Slow) Foxtrot, Quickstep |
| International Latin | Cha-cha, Samba, Rumba, Paso doble, Jive |
| American Smooth | Waltz, Tango, Foxtrot, Viennese waltz |

Rumba also reports American Rhythm Rumba (32-36 MPM) and Bolero (24-26), since the same slow
ballad often suits more than one. Tempo rules are in [`ballroom/dances.py`](ballroom/dances.py):
WDSF for International, USA Dance ranges for American. NDCA reference tempos come from a
secondary chart and still need checking against the current rulebook.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[model]"     # MERT model (torch + transformers)
pip install -e .              # or: tempo-only model, no torch
```

## Use

```bash
ballroom-classify song.mp3 ~/Music/some_album/          # local files and folders
ballroom-classify --songs songs.csv --out tagged.csv     # CSV with title,artist -> Deezer 30 s previews
ballroom-classify song.mp3 --top-k 5 --json              # more dances, JSON output
ballroom-classify song.mp3 --offset 60                   # take the 30 s clip from 1:00 into the song
ballroom-classify song.mp3 --model tempo                 # tempo + meter rules only
```

`--model auto` (the default) uses the MERT probe at `runs/mert_probe.pt` when it exists and
otherwise falls back to the tempo-only model with a warning. Point `--probe` at a checkpoint
elsewhere, e.g. `--probe ~/code/ballroom/runs/mert_probe.pt`. `--out` writes `.csv` (one row per
song, top-k dances with probability, MPM and the styles they fit) or `.json` (everything).

## Results

Held-out test set of 618 Extended Ballroom clips (duplicate-aware split, see below). Metrics
files and the confusion matrix are in [`results/`](results/).

| Model | Accuracy | Top-2 | Macro-F1 |
|---|---|---|---|
| CLAP zero-shot (no training) | 14.9% | 32.4% | 0.13 |
| Estimated tempo only | 57.1% | 80.6% | 0.48 |
| CLAP embeddings + probe | 83.8% | 95.3% | 0.82 |
| Annotated tempo only | 87.4% | 97.9% | 0.86 |
| Tempogram + MFCC, logistic regression | 88.8% | 95.6% | 0.86 |
| **MERT + probe** | **95.6%** | **98.5%** | **0.947** |

Published reference: 93.9% (Pavlin et al., ICPR 2020), on a different split.

- Weakest classes: Rumba (F1 0.90, most often predicted as Waltz) and Paso doble (0.84, only 8
  test clips).
- The rhythm information sits in MERT's middle layers (3 to 7), not the top layer: the probe's
  learned layer weights peak at layers 3 and 7, and the per-layer sweep peaks at layer 4.
- The tempo-only CLI model is hand-written rules, not the learned tempo baseline above; it has
  not been scored on the test set. Use it as a rough guide.

## Approach

No usable pre-trained ballroom classifier exists (the one public Hugging Face Space is broken),
so a pre-trained music foundation model is kept frozen with a small classifier on top.

- **MERT-v1-95M** (self-supervised on 160k hours of music; CC BY-NC 4.0). Each 30 s clip is cut
  into 5 s windows; every layer's hidden states are mean+std pooled and averaged over windows.
- **Probe:** a learned softmax weighting over MERT's 13 layers, then LayerNorm and a small MLP.
  Model selection on validation macro-F1 only; test is scored once.
- **Tempo and meter** come from librosa: the beat tracker's BPM, and a 3-vs-4 meter estimate
  from whether onset strength, bass energy and chord changes repeat every 3 or every 4 beats.
  The MPM shown for each dance picks the half/double-tempo reading closest to that dance.

## Data

**Extended Ballroom** (Marchand & Peeters, ISMIR 2016): 4,180 30-second clips with annotated
tempo. 10 dances are used: Slow waltz is merged into Waltz, and Salsa (47 clips) and West Coast
Swing (23) are dropped as too small to evaluate.

Audio is not redistributed and is never committed. The official IRCAM host currently returns
404; the GitHub mirror [`Pliploop/AllTempo`](https://github.com/Pliploop/AllTempo) has the audio,
which was checked against the official MD5s. Put the XML and the per-dance audio folders in
`data/`.

**Leakage:** the dataset repeats songs (exact copies, karaoke, re-recordings).
`ballroom.make_split` groups clips by normalised title+artist, by the XML's duplicate links, and
by near-identical audio embeddings, then makes a 70/15/15 split stratified by dance (seed 42).

## Reproduce

```bash
pip install -e ".[model]"
bash run_all.sh     # manifest -> MERT embeddings -> split -> baselines -> probe -> CLAP
```

About 3 GB free disk; embedding 4.1k clips with MERT is the slow step (it checkpoints and
resumes). Set `BALLROOM_HOME` to keep `data/`, `features/` and `runs/` outside the checkout.

| Module | Does |
|---|---|
| `ballroom.classify` | the CLI (`ballroom-classify`) |
| `ballroom.dances` | styles, dances, tempo rules, MPM and style fit |
| `ballroom.rhythm` | tempo and meter estimates, tempo-only classifier |
| `ballroom.manifest` | dataset XML + audio -> `data/manifest.csv` |
| `ballroom.embed` | MERT / CLAP embeddings -> `features/*.npz` |
| `ballroom.make_split` | duplicate-aware train/val/test split |
| `ballroom.baselines` | tempo-only and classic-feature baselines |
| `ballroom.train_probe` | layer-mixing probe on frozen embeddings (+ per-layer sweep) |
| `ballroom.zeroshot_clap` | CLAP with text prompts, no training |

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The tests need neither the dataset nor MERT: they use synthetic dance-like audio with a known
tempo and meter, a stand-in embedder, and tiny synthetic feature sets for the split and probe.

## Known limits

- Extended Ballroom is competition ballroom music. Pop songs that dancers actually use are out
  of domain, so accuracy on them is unknown. A hand-labelled set of 20 to 40 real songs is the
  honest test of the discovery use case, and the next step.
- Many songs fit several dances (a 120 BPM song in 4 can be a foxtrot or a cha-cha), which is
  why the output is ranked, not a single label.
- The meter estimate is a heuristic: reliable on clear accents, weaker on legato ballads. It is
  reported and used by the tempo-only model, but not yet fed into the MERT model; testing it on
  the Rumba/Waltz confusion is an open experiment.
- 30-second clips, not whole songs.
