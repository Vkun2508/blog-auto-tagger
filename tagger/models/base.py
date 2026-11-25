"""The shape every tagging model shares, so the engine can use any of them the same way."""


class TagModel:
    """A model gives every post a score from 0 to 1 for every approved tag.

    Titles and bodies passed in are already cleaned (see tagger.cleaning).
    """

    name = "base"

    def fit(self, titles, bodies, Y):
        """Learn from training posts. Y is a 0/1 matrix [posts x tags]."""
        raise NotImplementedError

    def predict_scores(self, titles, bodies):
        """Return a float32 array [posts x tags] with values between 0 and 1."""
        raise NotImplementedError

    def save(self, models_dir):
        raise NotImplementedError

    @classmethod
    def load(cls, models_dir):
        raise NotImplementedError
