"""Small helper for writing tables into the Markdown reports."""


def markdown_table(df, digits=3):
    """A pandas DataFrame as a Markdown table, numbers rounded to `digits` places."""
    def cell(value):
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    lines = ["| " + " | ".join(str(c) for c in df.columns) + " |",
             "|" + "---|" * len(df.columns)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(cell(v) for v in row) + " |")
    return "\n".join(lines)
