# dev-gym-slm

**Subject:** Fine-tuning a Small Language Model.
**Researcher:** mjohnson.
**Status:** Training complete. API pending.

---

This is the working repository from Day 1 of Raj's SLM workshop. The goal: fine-tune a pre-trained language model to classify text. The outcome: a model that can tell the difference between Brian Eno and Slayer. Both goals were achieved. One was more useful than the other.

The events documented here are real. The frustrations were real. The disk error was very real. Some details have been rendered with creative license in the interest of narrative clarity — but this is not fiction. It happened. All of it.

Names were not changed. No one was innocent.

---

## Read the Field Notes

**[→ NOTES-DAY1.md](./NOTES-DAY1.md)**

For full effect, read as if narrated by Detective Joe Friday from Dragnet. Measured. Deadpan. Just the facts. You'll know you've got the right voice when the vending machine line hits and you don't laugh — you simply nod.

If you want to skip straight to the technical bits, the tables and code blocks are your friends. But you'll be missing out.

---

## What's In Here

```
dev-gym-slm/
├── NOTES-DAY1.md                         # Field notes. Start here.
├── requirements.txt                      # Dependencies (Python 3.14 compatible)
├── commands.sh                           # Shell helpers — source this
└── training/
    ├── music-energy-classifier.ipynb     # Jupyter notebook — the whole thing
    └── data/
        └── feedback-classifier.json      # Training data (music, 0–5 energy scale)
```

The trained model is not included. It is 1.5GB. It lives on a machine that briefly ran out of disk space. See field notes for details.

---

## Quick Start

```bash
git clone https://github.com/matt-hsd/dev-gym-slm.git
cd dev-gym-slm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Open `training/music-energy-classifier.ipynb` in VS Code. Select the `.venv` kernel. Run all cells.

Read the field notes first. They will save you time.

---

*Day 2: inference API. Details to follow.*
