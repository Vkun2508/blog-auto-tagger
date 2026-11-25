# Error examples

Model: Hybrid (TF-IDF + Semantic) (`hybrid`), test split, tuned settings. Titles and URLs only.

## Similar tags confused

The model suggested one tag of a similar pair while the post's true tag was the other one. Found in 118 test posts. First examples:

- **Dear Coder: Open This If You're Feeling AI FOMO** (https://dev.to/canro91/dear-coder-open-this-if-youre-feeling-ai-fomo-58d4)
  - true tags: AI & LLMs, Career
  - suggested: AI & LLMs (0.95), Career (0.26), Machine Learning (0.18)
  - false positives: Machine Learning; missed: none
- **AI Agent vs Agentic AI: The Distinction That Changes Your Architecture** (https://dev.to/aws-builders/ai-agent-vs-agentic-ai-the-distinction-that-changes-your-architecture-3o8f)
  - true tags: AI & LLMs, Cloud
  - suggested: AI & LLMs (0.99), Cloud (0.69), DevOps (0.38), Software Architecture (0.37)
  - false positives: DevOps, Software Architecture; missed: none
- **How to Size Java Database Connection Pools in Kubernetes** (https://dev.to/apoorvtyagi/how-to-size-java-database-connection-pools-in-kubernetes-161p)
  - true tags: Databases, Docker & Kubernetes, Java
  - suggested: Docker & Kubernetes (0.41), Databases (0.34), DevOps (0.27), Java (0.14)
  - false positives: DevOps; missed: none

## Short drafts

Exactly right on the full post, but wrong when given only the title and the first 40 words. Found in 145 test posts. First examples:

- **I trained my AI agent to burn less money. Here's what actually worked.** (https://dev.to/jenatechio/i-trained-my-ai-agent-to-burn-less-money-heres-what-actually-worked-cjn)
  - true tags: AI & LLMs, DevOps
  - suggested: AI & LLMs (0.71), DevOps (0.23)
  - false positives: none; missed: none
  - suggested from title + first 40 words: AI & LLMs, Career, Machine Learning
- **Dealing with sensitive permissions on Android** (https://dev.to/tkuenneth/dealing-with-sensitive-permissions-on-android-3icg)
  - true tags: Mobile, Security
  - suggested: Mobile (0.69), Security (0.46)
  - false positives: none; missed: none
  - suggested from title + first 40 words: AI & LLMs
- **The Dealership Charges $180/hr for This, I Do It With an ESP32** (https://dev.to/numbpill3d/the-dealership-charges-180hr-for-this-i-do-it-with-an-esp32-off)
  - true tags: Security
  - suggested: Security (0.22)
  - false positives: none; missed: none
  - suggested from title + first 40 words: Machine Learning

## Niche / rare tags

A tag from the rare third (fewest training posts) was missed. Found in 100 test posts. First examples:

- **I sent 17 pull requests to one repository and the maintainer locked it for 24 hours** (https://dev.to/mahirhir/i-sent-17-pull-requests-to-one-repository-and-the-maintainer-locked-it-for-24-hours-2ko5)
  - true tags: Career, Git
  - suggested: Git (0.81), AI & LLMs (0.34)
  - false positives: AI & LLMs; missed: Career
- **A red I expected hid an instrument that was reading an empty directory** (https://dev.to/mahirhir/a-red-i-expected-hid-an-instrument-that-was-reading-an-empty-directory-2ph8)
  - true tags: DevOps, Rust, Testing
  - suggested: (none)
  - false positives: none; missed: DevOps, Rust, Testing
- **Learn Trapping Rain Water, Top K Frequent and Selection Sort with Step-by-Step Visualization in DSA View View 👀👀** (https://dev.to/nyaomaru/learn-trapping-rain-water-top-k-frequent-and-selection-sort-with-step-by-step-visualization-in-dsa-1flg)
  - true tags: TypeScript
  - suggested: JavaScript (0.21), Databases (0.13)
  - false positives: Databases, JavaScript; missed: TypeScript

## Long multi-topic posts

Among the longest 10% of posts (>= 3220 words) with 3 or more true tags. Found in 18 test posts. First examples:

- **An AI agent is just a while loop. I built one in 70 lines of Python, then tricked it into leaking my .env** (https://dev.to/alisterbaroi/an-ai-agent-is-just-a-while-loop-i-built-one-in-70-lines-of-python-then-tricked-it-into-leaking-4ehf)
  - true tags: AI & LLMs, Python, Security
  - suggested: AI & LLMs (0.95), Python (0.65), Security (0.27)
  - false positives: none; missed: none
- **Stop rebuilding from scratch: cache Docker layers on Cloud Build** (https://dev.to/gde/stop-rebuilding-from-scratch-cache-docker-layers-on-cloud-build-41m0)
  - true tags: Cloud, DevOps, Docker & Kubernetes, Python
  - suggested: Docker & Kubernetes (0.47), DevOps (0.41), Python (0.36), Cloud (0.31)
  - false positives: none; missed: none
- **What Nobody Is Using in Your Google Cloud Projects, and What It Costs** (https://dev.to/gde/what-nobody-is-using-in-your-google-cloud-projects-and-what-it-costs-1k0)
  - true tags: Cloud, DevOps, Python
  - suggested: Cloud (0.88), DevOps (0.32)
  - false positives: none; missed: Python

## 5 highest-scoring false positives

| Score | Suggested tag | Post | True tags |
|---|---|---|---|
| 0.981 | Databases | [Update DB Schema in Production with Flyway like Syntax](https://dev.to/codenameone/update-db-schema-in-production-with-flyway-like-syntax-2lj6) | Java, Mobile |
| 0.963 | Data Science | [Data Engineering Basics for Developers: Pipelines, Warehouses, and Streaming](https://dev.to/apeder/data-engineering-basics-for-developers-pipelines-warehouses-and-streaming-3b1m) | Databases, Software Architecture |
| 0.946 | AI & LLMs | [I Put a Local LLM in Charge of a Colony and Asked It to Tell the Truth. It Didn't.](https://dev.to/mikachu/i-built-a-text-based-survival-game-to-test-ai-morals-the-honest-one-lost-3fan) | Machine Learning, Python |
| 0.939 | AI & LLMs | [I built a Python script that generates writing and coding prompts for my 30-day writing challenge](https://dev.to/rupaligauravaram/i-built-a-python-script-that-generates-writing-and-coding-prompts-for-my-30-day-writing-challenge-36e4) | Data Science, Python |
| 0.933 | AI & LLMs | [The Rootkit Exorcist: Summoning Local AI to Investigate Suspicious Scripts 👻](https://dev.to/data_secchronicles/the-rootkit-exorcist-summoning-local-ai-to-investigate-suspicious-scripts-an1) | Security |

## 5 most-missed true tags

| tag | missed | test_posts | miss_rate |
|---|---|---|---|
| DevOps | 55 | 131 | 0.420 |
| Cloud | 30 | 63 | 0.476 |
| Software Architecture | 28 | 40 | 0.700 |
| AI & LLMs | 26 | 314 | 0.083 |
| Testing | 25 | 72 | 0.347 |
