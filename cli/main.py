import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="atlas", help="Universal lossless model compressor")
console = Console()


@app.command()
def compress(
    model: str = typer.Argument(help="HuggingFace model ID or local path"),
    target: str = typer.Option("auto", help="Hardware target (e.g. macbook-air-16gb, auto)"),
    quality: float = typer.Option(99.0, min=90, max=100, help="Quality target (percent of FP16)"),
    output_format: str = typer.Option("mlx", help="Output format: mlx or gguf"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Profile only, skip quantization"),
):
    from atlas.core.pipeline import Pipeline

    console.print("[bold green]Atlas v0.1.0[/bold green] — Universal Lossless Model Compressor\n")

    pipeline = Pipeline()

    with console.status("[bold blue]Profiling hardware + loading model metadata..."):
        hw = pipeline._profiler.detect()
        mi = pipeline._loader.load_metadata(model)

    console.print(f"[blue]\\[profile][/blue] {hw.chip} | {hw.ram_total_gb} GB RAM | {hw.gpu_cores}-core GPU")
    console.print(f"[blue]\\[model][/blue]   {mi.model_id} | {mi.num_params/1e9:.1f}B params | {mi.num_layers} layers | {mi.size_fp16_gb:.1f} GB FP16")

    result = pipeline.run(model, target, quality, output_format, dry_run=dry_run)

    usable_gb = pipeline._profiler.usable_memory_gb()
    if not result.fits_in_memory:
        console.print(f"\n[bold red]✗[/bold red] Model too large ({result.estimated_size_gb:.2f} GB > {usable_gb:.1f} GB usable)")
        console.print("[yellow]  Consider a smaller model or machine with more RAM[/yellow]")
        return

    console.print(f"[blue]\\[plan][/blue]    Target {result.estimated_bits}-bit | {mi.size_fp16_gb:.1f} GB → {result.estimated_size_gb:.2f} GB estimated")

    if dry_run:
        console.print(f"\n[bold green]✓[/bold green] Model fits ({result.estimated_size_gb:.2f} GB < {usable_gb:.1f} GB usable)")
        console.print("[dim]Dry run — skipping quantization[/dim]")
        return

    console.print()
    qr = result.quant_result
    er = result.eval_result

    console.print(f"[green]\\[quant][/green]  {qr.original_size_mb:.0f} MB → {qr.quantized_size_mb:.0f} MB ({qr.bits}-bit, group {qr.group_size})")
    console.print(f"[green]\\[eval][/green]   PPL baseline: {er.ppl_baseline:.2f} | quantized: {er.ppl_quantized:.2f} | delta: {er.ppl_delta_pct:+.2f}%")

    pi = result.package_info
    console.print(f"[green]\\[pack][/green]   Output: {pi.output_path}")
    console.print(f"\n[bold green]✓ Compression complete![/bold green] {pi.total_size_mb:.0f} MB total, PPL delta {er.ppl_delta_pct:+.2f}%")


if __name__ == "__main__":
    app()
