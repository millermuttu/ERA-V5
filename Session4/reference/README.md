# Session 4 — Reference material (crawled)

Source lesson: **Session 4: Data Cleaning and Deduplication**
URL: https://axiom.theschoolofai.in/courses/cmq97i5kn032208o8xu5dab4q/sessions/cmroc28350a8409qv5c8pm2a0/lesson

## Contents
- `session4-lesson.md` — full lesson text (14 sections, incl. the §13 assignment).
- `widgets/` — the 10 original reference widgets (self-contained HTML), pulled from `/widgets/` on the course site.

## Widget → pipeline stage
| File | Stage |
|------|-------|
| widget_1_cleaning_pipeline_map | whole pipeline + surviving-token bar |
| widget_2_clean_text_live | Unicode/normalization (keeps Indic joiners) |
| widget_3_special_tokens_ghost_tags | unify 4 conversation formats |
| widget_4_quality_filter_cascade | heuristics + classifier gate |
| widget_5_minhash_lsh_simulator | shingles → MinHash → LSH |
| widget_6_global_dedup_scale | local vs global dedup |
| widget_7_language_id_validation | detect language vs trust-the-folder |
| widget_8_pii_scrubber | regex + NER, precision/recall |
| widget_9_lineage_manifest_builder | provenance JSON + content hash |
| widget_16_contamination_heldout | eval/train firewall + canary strings |

## Video recordings (YouTube)
- https://www.youtube.com/watch?v=GpS-oisqkqA
- https://www.youtube.com/watch?v=LGQK548XxB4

## Assignment (§13)
1. Find how many "strategies" are listed in the session; describe them.
2. Pick a 10–100M dataset (ideally one seen in Session 3) and apply these cleanups.
3. Build one widget covering: the strategies (count + descriptions), the dataset picked,
   what was cleaned / why / how, any other concern addressed, and final statistics.
4. Deploy to Netlify and share the link.
