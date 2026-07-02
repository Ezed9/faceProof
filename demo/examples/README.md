# Demo examples

Drop 2–3 face images here and the demo shows them as **click-to-try** examples (the app auto-detects
any `.jpg/.jpeg/.png` in this folder — an empty folder is fine, no examples just don't appear).

For a demo that tells the whole story, use one of each (copy compression-matched crops from Drive
`MyDrive/faceproof/data/crops/...`, or any face image — the app preprocesses on the fly):

| File (suggested name) | Where to get it | What it shows |
|---|---|---|
| `1_real.jpg` | a real FFHQ crop (`crops/real/`) | passes → "real (passed)" |
| `2_stylegan_fake.jpg` | a StyleGAN crop (`crops/stylegan/`) | flagged → "SYNTHETIC" (detector works in-distribution) |
| `3_modern_t2i_fake.jpg` | an SDXL/Flux/DALL·E 3 face (from SFHQ-T2I) | **slips through as "real"** — the project's headline failure |

Naming them `1_…`, `2_…`, `3_…` keeps them in order (the app sorts by filename). Keep them small
(the 224×224 crops are ideal). These images are examples only — they are not committed with the repo
unless you add them here.
