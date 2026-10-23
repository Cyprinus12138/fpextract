# Samples

Every image in this directory is **synthetic**: it was drawn with Pillow by
[`scripts/make_samples.py`](../scripts/make_samples.py) and is released
under CC0. No third-party floor plan is included in this repository.

Regenerate with:

```bash
python scripts/make_samples.py
```

## Layout

```
samples/
├── manifest.jsonl                 # input contract, manifest form
└── harbour-view/                  # input contract, directory-tree form
    ├── harbour-view-type-a1.png   # unit plan, 2 bedroom
    ├── harbour-view-type-b2.png   # unit plan, 3 bedroom
    └── harbour-view-site-map.png  # site map / unit distribution chart
```

## Pointing the CLI at your own images

Either drop images into `<root>/<project>/` folders and pass the root:

```bash
fpextract extract --input /data/plans --output out/extractions.jsonl
```

or write a manifest (relative paths resolve against the manifest's folder):

```jsonl
{"project": "my-project", "title": "My Project", "images": ["a.jpg", "b.jpg"],
 "description": "3 Bedroom, Penthouse", "url": "https://example.com/my-project"}
```

```bash
fpextract extract --input /data/manifest.jsonl --output out/extractions.jsonl
```
