# Data Directory

Large data files are not committed to git. Copy or regenerate them after cloning.

## Required Files

```
data/processed/
├── metadata.json           # Battle metadata (~4MB) - copy from parent repo
├── perspective_map.json    # Player perspective mapping (~6MB) - copy from parent repo
├── priors.json             # Metagame priors (~220KB) - copy from parent repo
├── vocabs/gen3ou/          # Vocabularies (included in repo)
└── battles/                # .npz tensor files (15GB+, download with scripts)
```

## Setup

```bash
# Option 1: Copy from existing Pokemon-Battle-Model repo
cp /path/to/Pokemon-Battle-Model/data/processed/{metadata,perspective_map,priors}.json data/processed/

# Option 2: Download and process from scratch
python scripts/download_replays_stratified.py --format gen3ou --num-battles 100000
python scripts/process_dataset.py --input data/raw --output data/processed
```
