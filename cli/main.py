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
    from atlas.core.pipeline import Pipeline

    console.print("[bold green]Atlas v0.1.0[/bold green] — Universal Lossless Model Compressor\n")

    with console.status("[bold blue]Profiling hardware..."):
        pipeline = Pipeline()
        result = pipeline.run(model, target, quality, output_format)

    hw = result.hardware
    mi = result.model_info

    console.print(f"[blue]\\[profile][/blue] {hw.chip} | {hw.ram_total_gb} GB RAM | {hw.gpu_cores}-core GPU")
    console.print(f"[blue]\\[model][/blue]   {mi.model_id} | {mi.num_params/1e9:.1f}B params | {mi.num_layers} layers")
    console.print(f"[blue]\\[plan][/blue]    Target {result.estimated_bits}-bit | {mi.size_fp16_gb:.1f} GB -> {result.estimated_size_gb:.2f} GB")

    usable_gb = hw.ram_total_gb * 0.7
    if result.fits_in_memory:
        console.print(f"\n[bold green]✓[/bold green] Model fits in memory ({result.estimated_size_gb:.2f} GB < {usable_gb:.1f} GB usable)")
    else:
        console.print(f"\n[bold red]✗[/bold red] Model too large ({result.estimated_size_gb:.2f} GB > {usable_gb:.1f} GB usable)")
        console.print("[yellow]  Consider a smaller model or machine with more RAM[/yellow]")

    console.print("\n[dim]Phase 0 stub — quantization pipeline coming in Phase 1[/dim]")


if __name__ == "__main__":
    app()
