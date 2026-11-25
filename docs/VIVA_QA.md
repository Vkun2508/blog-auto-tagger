# Likely evaluator questions, with answers

Short answers you can say out loud. Numbers are in `reports/test_results.md` and the report.

## The problem

**1. What does your system do, in one sentence?**
It reads a blog post's title and body and suggests up to five tags from an approved list, with a score for
each, and sends the post to a human editor when it isn't confident or the text is too short.

**2. Why is this multi-label and not multi-class?**
A post can have several correct tags at once: a tutorial on securing a Python API is Python, API Development
and Security. Multi-class would force exactly one answer.

**3. Where does your data come from, and is it allowed?**
The public DEV.to API, which is free and needs no key. We don't redistribute the article text: the repo
holds the download script and the list of post IDs, so anyone can rebuild the same dataset. That fits the
brief's "reproducible data preparation script".

**4. How many posts, and from when?**
4,705 posts published between November 2025 and November 2026.

## Taxonomy and cleaning

**5. How did you build the taxonomy?**
22 tags chosen from what the posts actually cover, each with a definition, example titles, aliases (raw
author tags that merge in) and keywords. Generic tags like `programming`, `beginners` and `webdev` were
dropped because they appear on posts about everything.

**6. What is an alias map? Give an example.**
A dictionary from messy author tags to one approved tag: `ml`, `machinelearning` and `deeplearning` all
become Machine Learning. A test makes sure no alias maps to two tags.

**7. What did you remove during cleaning, and why?**
Navigation, headers, footers, sidebars, scripts, embeds, images, URLs, "follow me / subscribe" lines and
tag-widget lines like `#python #ai`. Template text teaches the model the wrong thing, and tag widgets
would leak the answer into the input.

**8. Why keep code but cut it to 30 lines?**
Code contains strong clues (`import pandas`, `kubectl`), but a 500-line dump would drown out the prose.

## Splitting

**9. Why split by date instead of randomly?**
The system will tag new drafts. Training on older posts and testing on the newest ones imitates real use;
a random split would mix future posts into training and look better than reality.

**10. Why group near-duplicates before splitting?**
If one copy of a republished post is in training and another in the test set, the model has effectively
seen the exam question, and the score is inflated. Duplicate groups always stay in one split.

**11. What are train, validation and test each used for?**
Train: fit the models. Validation: choose settings (title weight, pooling, C, thresholds, max tags).
Test: one final, honest measurement on unseen posts.

## Models

**12. What is your baseline and why have one?**
A keyword matcher using the taxonomy's keywords (title matches count three times). If a learned model
can't beat it, the learned model isn't worth its complexity.

**13. Explain TF-IDF simply.**
Each post becomes a list of word weights: high when a word is frequent in this post but rare across all
posts. "the" gets almost nothing; "kubectl" gets a lot.

**14. What does one-vs-rest logistic regression mean?**
We train one yes/no classifier per tag, 22 in total. Each gives a probability, so one post can get several
tags.

**15. Did giving the title more weight help?**
No. Weights 0 and 1 tied (0.688 vs 0.682 validation micro-F1) and 2-3 were worse. DEV.to titles are often
catchy rather than descriptive ("I got rejected 2 minutes after applying"), so the body carries the topic.
Each setting was compared at its own best threshold, so the comparison was fair.

**16. What is a sentence embedding?**
A pretrained neural network turns text into 384 numbers that capture meaning, so texts about the same
topic land close together even when they use different words.

**17. How do you handle long articles in the semantic model?**
Split the body into ~150-word chunks (at most 8, spread evenly across the post), embed each chunk and
combine them. We compared title-only, average of chunks, title + average, and title + maximum on
validation; the average of the body chunks won (0.667).

**18. Why might TF-IDF beat a neural model here?**
Tags are largely about named technologies (Docker, React, Postgres). Exact words are strong signals, and
TF-IDF captures them directly; a small general-purpose embedding model blurs specific names together.

## Thresholds and review

**19. What does the threshold control?**
How confident a score must be before a tag is suggested. Lower means more suggestions and more wrong ones;
higher means fewer wrong suggestions but more missed tags, and more posts sent to review.

**20. How did you choose it?**
A grid of thresholds (0.05 to 0.95) and maximum tags (1 to 5) on the validation set, keeping the best
micro-F1. Tags with at least 20 validation examples may get their own threshold, kept only if it helps.

**21. When does a post go to an editor?**
When it is empty, has fewer than 30 words, or no tag passes its threshold.

**22. Can the system ever return a tag outside the approved list?**
No. Models only have columns for the 22 approved tags, and editor additions are restricted to the same
list. Tests check both.

## Evaluation

**23. Micro-F1 vs macro-F1?**
Micro pools all decisions, so common tags dominate. Macro averages per-tag F1, so rare tags count
equally. Reporting both shows whether rare tags are handled.

**24. What are P@3 and R@3?**
Precision and recall looking only at the top three ranked tags, which matches what an editor sees first.

**25. What is review coverage?**
The share of posts the system tags without sending them to an editor.

**26. Which model did you pick and why?**
The hybrid (70% TF-IDF + 30% semantic): best on validation and on test (micro-F1 0.719, 97% of posts
tagged without review, 173 ms per post). TF-IDF alone is almost as good (0.714) and 16x faster (11 ms), so
for bulk tagging an archive I'd use TF-IDF; for the editor app the hybrid, because it handles short drafts
better and 0.2 seconds is unnoticeable.

**27. What kinds of mistakes does it make?**
Similar tags (AI & LLMs vs Machine Learning, Docker & Kubernetes vs DevOps), very short drafts, rare
topics with few training examples, and long posts that cover several subjects. Examples are in
`reports/error_examples.md`.

**28. How do incomplete historical labels affect your results?**
Authors could pick only 4 tags and often used generic ones, so correct topics are missing from the labels.
When the model suggests such a topic it's scored as wrong. Our 40-post audit measured this: precision
rose from 0.704 (authors' tags) to 0.765 (audited tags), and 21% of the "wrong" suggestions were correct.

## App and engineering

**29. How does an editor use it?**
Paste a draft, see ranked tags with score bars and a threshold tick, tick or untick, add missing tags
from the approved list, save. In batch mode, edit a table of final tags and export CSV or JSON.

**30. Where is feedback stored and what is it for?**
`feedback/feedback.jsonl`, one record per decision with suggested tags, scores and final tags. It can be
used to measure the model on real editor decisions and to retrain later. On the free host it resets on
restart, so the app offers downloads.

**31. How can someone reproduce your results?**
Follow the README's commands in order: download, prepare, train, tune thresholds, evaluate. Seeds are
fixed and versions are pinned.

**32. What would you improve with more time?**
More editor-checked labels, a fine-tuned embedding model, active learning from saved feedback, and
storing feedback in a database.
