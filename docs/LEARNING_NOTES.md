# Learning notes

Plain-language notes on everything we build. Read these before your presentation.

## Session 0: setup

**The project in one sentence:** a program reads a blog post and suggests up to 5 tags from an approved list; when it isn't sure, a human editor decides.

**Multi-label:** one post can have several correct tags at once (Python *and* API Development *and* Application Security). A "multi-class" problem would allow only one.

**Terminal:** a window where you type commands instead of clicking. `cd folder` moves into a folder.

**Virtual environment (venv):** a private box of Python libraries for one project. It stops this project's libraries from clashing with other projects. Ours lives at `C:\Users\vinoc\.venvs\blog-tagger`, outside OneDrive, so OneDrive doesn't sync thousands of library files. You "activate" it each time you open a new terminal; the prompt then starts with `(blog-tagger)`.

**pip:** Python's app store. `python -m pip install -r requirements.txt` installs every library listed in `requirements.txt`.

**git:** a save-history for code. `git add` picks what to save, `git commit` saves a snapshot with a message, `git push` uploads it to GitHub. `.gitignore` lists files git should never save.

**Python pieces we used** (see `scripts/00_python_refresher.py`):
- *variable*: a name holding a value
- *list* `[...]`: several values in order
- *dictionary* `{key: value}`: look something up by its key (our alias map is one)
- *function* `def name(...)`: a reusable recipe
- *loop* `for x in items`: do something for each item
- *if*: do something only when a condition is true
