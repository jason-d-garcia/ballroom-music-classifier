# Labelled dataset (`labels.csv`)

One row per clip or song: dance, ballroom style family, tempo (BPM and MPM) and meter, checked
against the competition tempo rules from the project's tempo atlas (`dance_rules.py`). Metadata
only; audio stays local. Rebuild with `build_dataset.py` (sources and command in its docstring).

| Source | Rows | Labels |
|---|---|---|
| Extended Ballroom v1.1 XML | 4,180 | dance + annotated BPM (dataset authors) |
| Ballroom 2004 beat annotations | 253 of those clips | second, beat-derived BPM and measured beats per bar |
| `ballroom-test/candidate_songs.csv` | 33 | dance only, `unconfirmed_guess` until Jason checks it |

## Columns

| Column | Meaning |
|---|---|
| `dance` | Class label, same names as `common.STYLES` (Slow waltz merged into Waltz). Salsa and West Coast Swing kept but `in_10_class_set=False` |
| `style_families` | Families the dance is competed in: International Standard, American Smooth, International Latin, American Rhythm |
| `meter`, `beats_per_bar` | Written time signature, and the beats per bar used for MPM (as in the atlas) |
| `bpm`, `mpm` | Annotated tempo; `mpm = bpm / beats_per_bar` |
| `tempo_fits` | Families whose competition range for this dance contains the clip's MPM (±2 MPM). Separates e.g. International Viennese waltz (58–60) from American (53–54) |
| `tempo_check` | `in_range`, `slow`, `fast`, or `half_time?` / `double_time?` (likely tempo-octave annotation errors) |
| `dup_group` | Same song (title+artist or XML repeat links); keep a group on one side of any split |
| `beats_bpm`, `beats_per_bar_measured` | From the Ballroom 2004 beat files, where the clip overlaps |
| `label_status` | `dataset_annotation`, `unconfirmed_guess`, or `jason_confirmed` |

## What the checks found

- 3,956 of 4,180 clips (95 %) sit inside the competition tempo range for their dance; 127 are a
  little slow (mostly Rumba and Jive, social-dance tempos) and 22 look like half/double-time
  annotation errors. Worth dropping or relabelling the 22 before retraining.
- Where Ballroom 2004 beats overlap (253 clips), the beat-derived tempo agrees with the Extended
  Ballroom BPM within 4 % for 97.6 % of clips, and the measured beats per bar matches the dance
  in every case.
- Only 1,768 clips fall in a single family by tempo; most Standard dances also fit American
  Smooth, so style family is a property of the tempo, not a separate label the audio can supply.

Not included: competition playlists (no reachable source from the cloud container; WDSF / NDCA
music lists or DJ playlists could be added as another `source` later).
