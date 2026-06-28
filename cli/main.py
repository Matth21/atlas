import typer
from rich.console import Console

app = typer.Typer(name="atlas", help="Universal lossless model compressor")
console = Console()


@app.command()
def compress(
    model: str = typer.Argument(help="HuggingFace model ID or local path"),
    target: str = typer.Option("auto", help="Hardware target (e.g. macbook-air-16gb, auto)"),
    quality: float = typer.Option(99.0, min=90, max=100, help="Quality target (percent of FP16)"),
    output_format: str = typer.Option("mlx", help="Output format: mlx or gguf"),
):
    console.print(f"[bold green]Atlas v0.1.0[/bold green]")
    console.print(f"Model:   {model}")
    console.print(f"Target:  {target}")
    console.print(f"Quality: {quality}%")
    console.print(f"Format:  {output_format}")


if __name__ == "__main__":
    app()
