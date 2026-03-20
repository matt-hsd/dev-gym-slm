# Field Notes — Day 1
## Subject: Fine-tuning a Small Language Model
## Researcher: mjohnson
## Status: Training complete. API pending.

---

> **TLDR:** What follows is a factual account of one afternoon spent fine-tuning a small language model. The events are real. The frustrations are real. The disk error was very real. Some details have been rendered with creative license in the interest of narrative clarity — but this is not fiction. It happened. All of it.
>
> Names were not changed. No one was innocent.
>
> For full effect, read this document as if narrated by Detective Joe Friday from Dragnet. Measured. Deadpan. Just the facts. You'll know you've got the right voice when the vending machine line hits and you don't laugh — you simply nod.
>
> If you want to skip straight to the technical bits, the tables and code blocks are your friends. But you'll be missing out.

---

**13:00 — The client**

The client's name was Raj. Short. Confident. The kind of name that arrives in a room before the person does. He seemed well-meaning — and he was. He'd put together a clean workshop repo, written up the steps, laid out the path. Raj had done the work.

What Raj had not done was test it on Python 3.14.

This is not unusual. Most people haven't tested anything on Python 3.14. Python 3.14 is new. Python 3.14 has opinions. Raj had gotten himself into a bit of a jam, and to his credit, he knew it. He flagged the environment setup as a potential rough patch, offered some guidance, and then — in the tradition of instructors everywhere — pointed at the materials and wished us luck.

Some tried to help him sort it out. Raj thanked them and kept moving. You had to respect that.

We had a secret weapon. Not a search engine. Not a forum post from 2019. Claude.

---

**13:10 — The repo**

Raj distributed the workshop repo. Cloned it. Opened it with Claude. Read the tutorial steps. Steps were clear. The part where I type everything manually — that part wasn't going to happen. Clearly.

---

**13:20 — The requirements**

Opened `requirements.txt`. Five packages pinned to specific versions. Ran `pip install`. The installer considered the request and declined.

`torch==2.5.1`? Doesn't exist for 3.14. `pydantic==2.9.2`? Its Rust internals were compiled for a Python that has since been surpassed. The installer returned errors the way a waiter returns to tell you the kitchen is out of everything you ordered. Politely. Repeatedly. Without solutions.

The culprit: several packages ship compiled Rust extensions via PyO3. Older PyO3 hard-caps at Python 3.13. No wheel, no install, no explanation beyond `ERROR: No matching distribution found`. This is technically accurate. It is not technically helpful.

Turned to Claude. Claude identified the problem, bumped the versions, noted two missing packages the tutorial hadn't mentioned, and updated the file. The install completed. We moved on.

Fix: stop pinning old versions. Use what exists.

| Package | Pinned | Working |
|---|---|---|
| `torch` | 2.5.1 | 2.10.0 |
| `pydantic` | 2.9.2 | 2.12.5 |
| `transformers` | 4.46.3 | 5.3.0 |
| `tokenizers` | 0.20.3 | 0.22.2 |
| `safetensors` | 0.4.5 | 0.7.0 |

Two additional packages the tutorial doesn't mention but silently requires:
- `accelerate` — `Trainer` in transformers 5.x won't initialize without it. The error eventually tells you this. Eventually.
- `ipykernel` — needed for Jupyter. Not in the requirements file. This one was our idea. Any self-respecting data scientist works in a notebook. We're here to learn, not to run scripts into the void and hope for the best.

Lesson: pinned dependencies are a promise that expires.

---

**13:35 — The notebook**

The tutorial is designed to be followed step by step into a `.py` file. Functional. Instructional. Not how I wanted to work.

Asked Claude to convert the whole thing into a Jupyter notebook — each section its own cell, outputs visible inline, every step inspectable before moving to the next. Claude built it without complaint. Claude doesn't complain.

This was the right call. Watching tokenization happen in a single cell — raw text in, token IDs out, decoded back to subwords right below it — made the concept land in a way a completed script never would have.

---

**13:50 — The kernel situation**

Two repos, one machine. `dev-gym-slm-workshop` — Raj's. `dev-gym-slm` — mine. VS Code, given the choice between the two venvs, chose the wrong one. Confidently. Without hesitation. It was nearly 2pm. The researcher was hungry. Nobody was at their best.

In fairness — the steps were new. Jupyter kernels, venv registration, VS Code's kernel selector. It's possible VS Code knew exactly what it was doing and the confusion was on this end. The record will reflect that no definitive determination was made.

Errors followed either way. The kind that look like code problems but are actually environment problems. Took several rounds to diagnose. The fix:

```bash
cd ~/dev/dev-gym-slm
source .venv/bin/activate
python -m ipykernel install --user --name=dev-gym-slm --display-name="dev-gym-slm"
```

In VS Code, the kernel selector shows the Python path. The correct one shows `.venv/bin/python`. Any path pointing somewhere else is the wrong one. When switching kernels mid-session, restart and run all cells. The kernel has no memory. Unlike the researcher, who will remember this for some time.

---

**14:15 — Transformers 5.x API drift**

Two breaking changes from the tutorial code. Neither was announced. Both were discovered the usual way.

`use_mps_device` no longer exists in `TrainingArguments`. MPS is auto-detected now. Remove the line.

`tokenizer` renamed to `processing_class` in `Trainer`. The old name is gone. The new name is longer.

```python
# Dead
trainer = Trainer(..., tokenizer=tokenizer)

# Alive
trainer = Trainer(..., processing_class=tokenizer)
```

---

**14:30 — Specimen substitution**

The tutorial classifies ride-share safety feedback on a 0–5 severity scale. Functional example. Correct example. An example I would be staring at for the rest of the day.

Substituted: **music energy classification**. Same model architecture. Same label schema. Different domain. The model doesn't know the difference. The model doesn't know anything yet. That's the point.

| Label | Observed Behavior |
|---|---|
| 0 | Chill / ambient / sleep music |
| 1 | Mellow / acoustic / lo-fi |
| 2 | Mid-tempo / background |
| 3 | Upbeat / feel-good / danceable |
| 4 | High energy / hype / workout |
| 5 | Absolute banger / chaos |

Initial dataset: 36 examples, 6 per label. 80s, 90s, 2000s, 2010s, and classics. Descriptions of sonic character and energy — not lyrics, not metadata. Just vibes. The model will learn to recognize vibes. This is what we've built.

Key finding: the tutorial data is a placeholder. The architecture is the thing. Swap the domain freely.

---

**15:30 — First training run**

3 epochs. Speed mode. 36 examples, 28 training / 8 test. ModernBERT-large: 395 million parameters, now being asked to learn the difference between Boards of Canada and Slayer.

| Epoch | Val Loss | Accuracy |
|---|---|---|
| 1 | 2.12 | 0% |
| 2 | 1.69 | 12.5% |
| 3 | 1.59 | **50%** |

Loss trending down. Model learning. 50% on a 6-class problem where random baseline is ~17% is meaningful. The dataset is too small to go further. We noted this and moved on.

---

**16:00 — Incident report: disk exhaustion**

`SafetensorError: No space left on device`

It didn't ask. It didn't warn. It didn't negotiate. It stopped. Like a vending machine that takes your dollar and then goes dark. No refund. No explanation. No snack.

The `Trainer` saves a full model checkpoint after every epoch. ModernBERT-large checkpoints at ~500MB each. Three epochs in speed mode means the trainer quietly wrote 1.5GB to disk *before* attempting the final save — another 1.5GB. On top of the base model download. The venv. Two repos. Whatever else had been accumulating on this machine since the last time anyone looked.

The drive ran out of room mid-save. The process died. Nothing was recoverable.

Training halted. Researcher closed the laptop, accepted the situation, and went to dinner. It should be noted that the researcher was hungry. This may have influenced the speed of acceptance.

---

**19:00 — Field tool developed during downtime**

Sitting at dinner. Thinking about disk space. This is what the job does to you.

The problem isn't unique to this situation. It's Xcode derived data. iOS simulators that haven't been booted in two years but are still 8GB each. Old Docker images. npm caches. Model checkpoints from experiments you don't remember running, for models you never deployed, sitting in directories you forgot existed. It accumulates. Quietly. Without asking. Until one day you're mid-training and your computer informs you, via a Rust error, that it's full.

Built a storage assistant. An AI agent that hunts down disk space you can safely reclaim on macOS. You tell it how much space you want. It finds the culprits — simulators, derived data, caches, orphaned downloads. Presents the evidence. Then asks before deleting anything. Every time. No exceptions. You cannot accidentally nuke something important. That behavior is not available.

Tuned for iOS developers. Works for everyone.

**If you want a copy, ask me.**

---

**19:30 — Dataset expansion**

Cleared space. Restarted kernel. Before retraining, expanded the dataset — 36 examples wasn't enough. Added more entries across eras, more range within each energy level. Metallica. Miles Davis. Daft Punk. Norah Jones. 79 examples total. The model would have opinions about all of them. We would measure those opinions with F1 score.

---

**19:45 — Second training run**

Ran all cells. Watched the loss go down again. It went down again.

Result: **37.5% accuracy, loss 1.50**.

This looks worse than the first run. It isn't. Test set doubled from 8 to 16 samples — each wrong answer now costs 6.25% instead of 12.5%. Loss continued decreasing. The model improved. The number went down. Both statements are accurate. Only one caused concern.

Model saved to `models/music-energy-classifier/final/`. Size on disk: **1.5GB**. Predominantly base model weights — ModernBERT-large is 395 million parameters that already knew how to read before we got here. The classification head we actually trained is negligible by comparison. We taught a rocket scientist to sort music. The rocket science was already in there.

---

**End of Day 1**

Environment configured. Data prepared. Model trained. Disk survived. Raj vindicated.

Day 2: inference API. FastAPI server, load the model, serve predictions over HTTP.

Open question: ModernBERT-large may be overkill for this task. ModernBERT-base is ~150MB. Worth evaluating if you want something you can actually ship without a loading dock.

---

*Notebook: `training/music-energy-classifier.ipynb`*
*Training data: `training/data/feedback-classifier.json`*
*Model: `models/music-energy-classifier/final/`*
