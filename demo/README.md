# Demo (Week 2)

A Gradio app for Hugging Face Spaces (free):
- Upload a face → **real / synthetic** + confidence bar + a robustness note.
- Optional **attack-then-defend** panel: a mock "enrollment" that accepts a pre-generated fake face,
  then this auditor flagging it. (Uses existing free generated faces — NOT an adversarial-attack pipeline.)

`app.py` will load the saved CLIP probe (the small classifier head only — the repo ships the probe,
not the datasets) and CLIP features at inference time.

Keep the UX interpretable: show the side-by-side real/fake example and one sentence of explanation,
not just a raw score.
