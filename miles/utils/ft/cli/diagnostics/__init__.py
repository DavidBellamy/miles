import typer

from miles.utils.ft.cli.diagnostics.cluster import cluster
from miles.utils.ft.cli.diagnostics.local import local

app = typer.Typer(help="Node diagnostic commands.")
app.command()(local)
app.command()(cluster)
