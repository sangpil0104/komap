"""Recipe path rendering retained for the original strict model packager."""
def render(values,source,run,workers):
    return [v.format(source=str(source),run=str(run),workers=workers) for v in values]
