# Ballroom style classifier

Tell which ballroom dance a song fits (waltz, tango, cha-cha, …) from its audio, so you can find
new music to dance to. Point it at any song, or a list of titles, and it returns the top three
styles with probabilities and the tempo.

Status: trained and evaluated on Jason's Mac (Oct 2026). **MERT + probe: 95.6 % accuracy, 0.947
macro-F1** on 618 held-out clips. Metrics and confusion matrices are in [`results/`](results/)
(copied from `runs/`; the trained checkpoints stay local).

## Results (618 held-out test clips)

| Model | Accuracy | Top-2 | Macro-F1 |
|---|---|---|---|
| CLAP zero-shot (no training) | 14.9 % | 32.4 % | 0.126 |
| Estimated tempo only | 57.1 % | 80.6 % | 0.479 |
| CLAP embeddings + probe | 83.8 % | 95.3 % | 0.819 |
| Annotated tempo only | 87.4 % | 97.9 % | 0.864 |
| Tempogram + MFCC, logistic regression | 88.8 % | 95.6 % | 0.859 |
| **MERT + probe** | **95.6 %** | **98.5 %** | **0.947** |

- Weakest classes: Rumba (F1 0.90, often predicted as Waltz) and Paso doble (F1 0.84, only 8 test
  clips).
- The rhythm information sits in MERT's middle layers: the single-layer sweep peaks at layer 4
  (val macro-F1 0.968) and the learned layer mix puts most weight on layers 3 and 7
  ([`results/mert_layer_sweep.csv`](results/mert_layer_sweep.csv)).
- Published reference: 93.9 % (Pavlin et al. 2020), on a different split.

## Approach (and why)

**No usable pre-trained ballroom classifier exists.** The only public one found
([waidhoferj/dance-classifier](https://huggingface.co/spaces/waidhoferj/dance-classifier)) is a
broken Space with no published weights. So: a pre-trained music foundation model, frozen, with a
small classifier trained on top.

| Option | Verdict |
|---|---|
| **MERT-v1-95M** (music self-supervised, trained on 160k h of music, predicts CQT + EnCodec codes) | **Main model.** Built for music, strong on rhythm/beat tasks in the MARBLE benchmark, runs on a Mac (95M params). Licence CC BY-NC 4.0, fine for a non-commercial portfolio. |
| **CLAP** (laion/larger_clap_music_and_speech; the music-only checkpoint on the Hub is broken) | Second model and a **zero-shot** baseline ("music for a rumba"), i.e. a pre-existing model with no training at all. |
| AST / BEATs | Trained on AudioSet (general sound events). Weaker on musical rhythm than MERT; skipped unless MERT disappoints. |
| MERT-330M, full fine-tuning | Only if the frozen probe plateaus: 3.5× slower and a 4 k-clip dataset overfits easily. |

Frozen embeddings + a probe means the expensive step (one pass of MERT over ~4 k clips) runs once,
and every classifier experiment after that takes seconds.

## Data

**Extended Ballroom** (Marchand & Peeters, ISMIR 2016 LBD): 4,180 30-second previews from
ballroomdancers.com in 13 styles, with annotated tempo. Audio is not redistributed: the official
script downloads it from [IRCAM](http://anasynth.ircam.fr/home/media/ExtendedBallroom); that host
now returns 404, so the audio here came from the `Pliploop/AllTempo` GitHub mirror, checked against
the official MD5s. It is kept local and never committed.

| Used here | Clips |
|---|---|
| Cha-cha 455, Foxtrot 507, Jive 350, Quickstep 497, Rumba 470, Samba 468, Tango 464, Viennese waltz 252 | |
| Waltz 529 + Slow waltz 65 (same dance, merged as in Pavlin et al. 2020) | 594 |
| Paso doble | 53 |
| Dropped: Salsa (47), West Coast Swing (23), too few to evaluate | |

The original 698-clip **Ballroom** set (Dixon et al. 2004) is a possible cross-dataset test later;
the authors advise against merging the two because audio quality differs.

**Leakage:** Extended Ballroom repeats songs (exact copies, karaoke, other recordings).
`make_split.py` groups clips by normalised title+artist and by near-identical audio embeddings and
keeps each group on one side of a 70/15/15 split stratified by style (seed 42).

## Baselines the model must beat

Ballroom styles are largely defined by tempo and meter, and
[Sturm (2016)](https://arxiv.org/abs/1606.03044) showed classic Ballroom systems mostly learn
tempo. So `baselines.py` reports tempo-only (annotated and estimated BPM), and classic rhythm +
timbre features. Published reference on Extended Ballroom: 93.9 % top-1 (Pavlin et al., ICPR
2020, CNN with segment aggregation; 10 classes, close to ours but a different split).

## Run (on a Mac with ~3 GB free)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 1. download extendedballroom_v1.1.zip, unzip into data/, run its download script
# 2.
bash run_all.sh                # manifest -> MERT embeddings -> split -> baselines -> probe -> CLAP
# 3. tag new music
python classify.py --songs songs.csv --out tagged.csv     # title,artist -> Deezer previews
python classify.py ~/Music/some_album/
```

| File | Does |
|---|---|
| `manifest.py` | XML + audio → `data/manifest.csv` |
| `embed.py` | MERT / CLAP embeddings → `features/*.npz` |
| `make_split.py` | duplicate-aware train/val/test split |
| `baselines.py` | tempo-only and classic-feature baselines |
| `train_probe.py` | layer-mixing MLP probe on frozen embeddings (+ per-layer sweep) |
| `zeroshot_clap.py` | CLAP with text prompts, no training |
| `classify.py` | tag local files or Deezer previews |

## Known limits

- Extended Ballroom is competition ballroom music. Pop songs that dancers use (the point of
  discovery) are out of domain, and Pavlin et al. saw a large drop on YouTube recordings. A small
  hand-labelled set of real songs is the honest test of the discovery use case.
- Many songs fit several dances (a 120 BPM 4/4 song can be a foxtrot or a cha-cha), which is why
  output is the top three, not one label.
- 30-second previews, not whole songs.
