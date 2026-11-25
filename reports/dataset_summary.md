# Dataset summary

Written by `scripts/02_prepare_data.py`.

## Posts

- Raw posts downloaded: 4705
- Dropped because none of their raw tags maps to an approved tag: 0
- Posts kept: 4705

| Split | Posts | Share | First post | Last post |
|---|---|---|---|---|
| train | 3294 | 70.0% | 2025-11-14 | 2026-06-09 |
| val | 706 | 15.0% | 2026-06-09 | 2026-09-07 |
| test | 705 | 15.0% | 2026-09-04 | 2026-11-14 |

## Near-duplicates

Posts are joined into one group if they have the same normalized title, the same canonical URL, or bodies with TF-IDF cosine similarity >= 0.9. A group is never split across train/val/test.

- Duplicate groups (2 or more posts): 12
- Posts inside those groups: 29
- Links found: same title 16, same canonical URL 0, similar body 12

Example pairs:

- (same title) "useMemo y useCallback: cuándo sí sirven… y cuándo solo estorban"  /  "useMemo y useCallback: cuándo sí sirven… y cuándo solo estorban"
- (same title) "Decoding the Beautiful Game: AI's Play-by-Play Revolution by Arvind Sundararajan"  /  "Decoding the Beautiful Game: AI's Play-by-Play Revolution by Arvind Sundararajan"
- (same title) "Built an AI Agent That Actually Runs Agile Sprints End-to-End (Not Just Ticket Generation)"  /  "Built an AI Agent That Actually Runs Agile Sprints End-to-End (Not Just Ticket Generation)"

## Approved labels per post

| Labels | Posts |
|---|---|
| 1 | 1501 |
| 2 | 2256 |
| 3 | 831 |
| 4 | 117 |

Average labels per post: 1.91

## Generic tags dropped

62.4% of raw posts (2936 of 4705) used at least one of their 4 tag slots for a generic tag.

| Generic tag | Posts using it |
|---|---|
| webdev | 828 |
| programming | 688 |
| opensource | 500 |
| beginners | 311 |
| discuss | 286 |
| tutorial | 267 |
| productivity | 237 |
| showdev | 185 |
| devchallenge | 176 |
| performance | 170 |
| learning | 131 |
| linux | 83 |
| coding | 75 |
| softwaredevelopment | 61 |
| tooling | 56 |
| cli | 53 |
| softwareengineering | 47 |
| bugsmash | 41 |
| development | 40 |
| news | 38 |
| devjournal | 36 |
| buildinpublic | 30 |
| community | 29 |
| statemanagement | 25 |
| watercooler | 17 |
| weekendchallenge | 15 |
| developer | 11 |
| developers | 10 |
| hacktoberfest | 6 |

## Tag counts

See `reports/tag_counts.csv` and `reports/figures/tag_counts.png`.
Tags flagged too_few (test < 20 or total < 80): Data Science
