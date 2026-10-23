-- fpextract Postgres sink: one parent row per project, one child row per
-- physical unit. Apply once; `fpextract sync-pg` only upserts.

CREATE TABLE IF NOT EXISTS floorplan_projects (
  slug                    text PRIMARY KEY,
  title                   text,
  url                     text,
  n_source_images         integer NOT NULL DEFAULT 0,
  n_physical_units        integer,
  n_multi_storey_units    integer,
  dominant_property_type  text,
  property_type_source    text,
  property_types_seen     text[],
  confidence              double precision,
  unit_number_map         jsonb NOT NULL DEFAULT '{}'::jsonb,
  unit_number_evidence    jsonb NOT NULL DEFAULT '{}'::jsonb,
  conflicts               jsonb NOT NULL DEFAULT '[]'::jsonb,
  aggregated_at           timestamptz,
  synced_at               timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS floorplan_units (
  id                      bigserial PRIMARY KEY,
  slug                    text NOT NULL REFERENCES floorplan_projects(slug) ON DELETE CASCADE,
  unit_type               text NOT NULL,
  classification          text,
  property_type           text,
  n_storeys               integer NOT NULL DEFAULT 1,
  bedrooms                integer,
  bathrooms               integer,
  total_area_sqm          numeric(8,2),
  balcony_sqm             numeric(8,2),
  facing                  text,
  rooms                   jsonb NOT NULL DEFAULT '[]'::jsonb,
  windows                 jsonb NOT NULL DEFAULT '[]'::jsonb,
  storeys                 jsonb,
  confidence              double precision,
  n_source_records        integer,
  source_image_sha256s    text[],
  source_image_basenames  text[],
  match_methods           text[],
  UNIQUE (slug, unit_type)
);

CREATE INDEX IF NOT EXISTS floorplan_units_bedrooms_idx ON floorplan_units (bedrooms);
CREATE INDEX IF NOT EXISTS floorplan_units_property_type_idx ON floorplan_units (property_type);
