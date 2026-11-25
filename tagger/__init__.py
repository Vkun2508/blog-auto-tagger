"""Blog Auto-Tagger: suggests approved taxonomy tags for blog posts.

Main entry points:
    from tagger.engine import TaggingEngine      # load trained models and suggest tags
    python -m tagger --title "..." --body "..."   # try it from the command line
"""

from pathlib import Path

# The project folder (one level above this package). Every default path is built from it.
PROJECT_DIR = Path(__file__).parent.parent
