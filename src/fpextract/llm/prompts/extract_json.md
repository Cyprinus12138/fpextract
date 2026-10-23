Analyse this residential floor-plan image and return a single JSON object with this exact structure:

{
  "classification": "<unit_plan|hybrid|site_map|facility_plan|other>",
  "confidence": <0.0-1.0>,
  "notes": "<optional>",
  "unit_number_map": { "<unit_number_as_printed>": "<unit_type_label>" },
  "units": [
    {
      "unit_type": "<label as printed, e.g. C1, TYPE A>",
      "floor_label": "<optional storey range or Lower/Upper Floor>",
      "bedrooms": <integer>,
      "bathrooms": <integer>,
      "total_area_sqm": <number if printed>,
      "balcony_sqm": <number if printed>,
      "facing": "<N|NE|E|SE|S|SW|W|NW|unknown>",
      "rooms": [{"label": "<as printed>", "area_sqm": <if printed>}],
      "bbox": [x_min, y_min, x_max, y_max]
    }
  ]
}

bbox is the tight bounding box of ONE representative cell for that unit_type.
{COORD_INSTRUCTION}

Classification rules:
- unit_plan : room-level detail inside cells, NO unit numbers (like #02-03) next to cells
- hybrid    : room-level detail AND unit numbers next to cells
- site_map  : coloured blocks with unit numbers but NO room detail inside
- facility_plan: pool, gym, BBQ areas
- other     : cover pages, 3D renders, logos

units[] rules:
- One entry per unique (unit_type, floor_label) pair. Never emit the same unit_type for the same floor_label twice, even if multiple cells of that type appear.
- unit_type must be a type label found on this sheet (e.g. TYPE C, B1, A). Never use a room name (MASTER BEDROOM, KITCHEN, LIVING, etc.) as unit_type.
- When multiple storey plans appear on the same sheet, the unit type label may only be printed on one storey. Infer the unit type for unlabelled storeys from the type labels visible elsewhere on the sheet — this is cross-referencing, not inventing data.
- Once you detect this pattern, emit one entry per (unit_type, floor_label) pair for every storey on the sheet. Count the distinct storey-level plans on the sheet and verify your units[] covers all of them.
- Before finalising: collect the set of valid unit type labels you found on this sheet. Verify every unit_type in your units[] belongs to that set. If any entry has a room name instead, replace it with the correct label.
- total_area_sqm: read the sheet's title block, type header, and any caption like "TYPE A — 132 SQM", "Area: 132m²", "Total Area", "Floor Area", "Strata Area". The area is usually printed OUTSIDE the floor-plan drawing — in the title strip at the top/bottom of the sheet or next to the unit-type label. Search those regions before falling back to null. Only set null if the sheet truly does not print an area anywhere.

bbox rules (unit_plan / hybrid only):
- `bbox` must cover exactly ONE cell — its outer walls, rooms, and labels. Never span two or more cells, even when they share the same unit_type. Because unit footprints are not rectangles, boxes of neighbouring cells may overlap slightly; that is acceptable.
- When a unit_type appears in multiple positions, pick the single instance that has the detailed room labels (e.g. BEDROOM) inside it.
- Do NOT place the box midway between two adjacent same-type cells. Commit to one cell and crop it tightly.
- Do not extend into empty white margin, corridors, lift cores, or the title block.

unit_number_map — populate whenever any unit numbers are visible on the sheet.
Recognise unit numbers in any of these forms:
- Cell labels next to a unit position (hybrid/site-map), including ranges.
- Comma- or line-separated lists under a unit-type heading.
- Stack-view / block-directory tables: rows = storeys, columns = unit stacks, each cell prints a unit number and a unit_type label. Read every cell.
- Block diagrams / unit-distribution charts with unit numbers on each block.

Formatting rules:
- One key per individual unit number — never a range. Expand ranges into one key per storey (e.g. "02-06 to 08-06" → "02-06", "03-06", … "08-06").
- The unit type label is the VALUE; the unit number is the KEY.
- Canonicalise the unit-number portion before emitting:
    • Keep storey-unit dashes intact (e.g. "02-06" stays "02-06", "#02-03" stays "#02-03").
    • REMOVE any dash that sits between a single column letter and its stack digits. Render "A-02" as "A02", "B-07" as "B07". Do this even when the sheet draws the dash; the goal is one canonical key per physical unit so different sheets of the same project produce identical keys.
    • Keep the # prefix if printed.
- When the sheet has block-specific tables, always prefix every key with the block label, normalised to "BlkN" (e.g. "Blk 2" → "Blk2", "Block 4" → "Blk4"). Format: "BlkN-<canonical_unit_number>" (e.g. "Blk2-#02-01", "Blk156-A02"). If a table heading covers multiple blocks (e.g. "Blocks 3 & 5"), emit one entry per individual block with the same value. If only one block exists and no block label is printed, use the unit number alone.
- Leave empty only if no unit numbers appear anywhere on the sheet.

Never invent data. Omit any field you cannot read from the image.
