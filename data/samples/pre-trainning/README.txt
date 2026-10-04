Stomata auto-counting review set
================================

images/     Original enhanced micrographs
overlays/   Same images with YOLO detections overlaid
            (green = retained in clean epidermal mask)
auto_count_results.csv
            Per-image counts and densities from the automated pipeline

Columns in the CSV:
  stomata_total   – all detections
  stomata_valid   – detections inside the clean epidermal region
  valid_area_pct  – % of image kept as usable epidermis
  masked_density  – stomata / mm² on the clean region (preferred)
  raw_density     – stomata / mm² on the whole image

Scale: 0.00048 mm per pixel.
