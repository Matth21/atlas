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
    mode: str = typer.Option("mixed", help="Quantization mode: uniform or mixed (legacy)"),
    budget_gb: float = typer.Option(None, "--budget-gb", help="SGSR-2: memory budget in GB — measures per-block cost and finds the best plan that fits (recommended)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Profile only, skip quantization"),
):
    from atlas.core.pipeline import Pipeline

    console.print("[bold green]Atlas v0.1.0[/bold green] — Universal Lossless Model Compressor\n")

    if budget_gb is not None:
        _compress_sgsr2(model, budget_gb)
        return

    if mode == "mixed":
        console.print("[yellow]⚠ 'mixed' mode uses the legacy entropy planner, which controlled "
                      "experiments showed carries no allocation signal (see paper §6). "
                      "Prefer --budget-gb (SGSR-2, measured cost).[/yellow]\n")

    pipeline = Pipeline()

    with console.status("[bold blue]Running pipeline..."):
        result = pipeline.run(model, target, quality, output_format, mode=mode, dry_run=dry_run)

    hw = result.hardware
    mi = result.model_info

    console.print(f"[blue]\\[profile][/blue] {hw.chip} | {hw.ram_total_gb} GB RAM | {hw.gpu_cores}-core GPU")
    console.print(f"[blue]\\[model][/blue]   {mi.model_id} | {mi.num_params/1e9:.1f}B params | {mi.num_layers} layers | {mi.size_fp16_gb:.1f} GB FP16")

    usable_gb = pipeline._profiler.usable_memory_gb()
    if not result.fits_in_memory:
        console.print(f"\n[bold red]✗[/bold red] Model too large ({result.estimated_size_gb:.2f} GB > {usable_gb:.1f} GB usable)")
        console.print("[yellow]  Consider a smaller model or machine with more RAM[/yellow]")
        return

    console.print(f"[blue]\\[plan][/blue]    Target {result.estimated_bits}-bit | {mi.size_fp16_gb:.1f} GB → {result.estimated_size_gb:.2f} GB estimated")

    if result.quant_plan is not None:
        qp = result.quant_plan
        bit_counts = {}
        for lp in qp.layers:
            bit_counts[lp.bits] = bit_counts.get(lp.bits, 0) + 1
        dist = ", ".join(f"{n}×{b}-bit" for b, n in sorted(bit_counts.items()))
        console.print(f"[blue]\\[mixed][/blue]   avg {qp.avg_bits:.1f}-bit | layers: {dist}")

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


def _compress_sgsr2(model: str, budget_gb: float) -> None:
    from atlas.core.model import ModelLoader
    from atlas.core.sgsr2_flow import compress_to_budget

    mi = ModelLoader().load_metadata(model)
    console.print(f"[blue]\\[model][/blue]  {mi.model_id} | {mi.num_params/1e9:.1f}B params | {mi.size_fp16_gb:.1f} GB FP16")
    console.print(f"[blue]\\[sgsr2][/blue]  budget {budget_gb:.2f} GB — profiling misurato per blocco "
                  f"(prima volta: ore; poi in cache)\n")

    try:
        r = compress_to_budget(model, budget_gb, mi.num_params)
    except ValueError as exc:
        console.print(f"[bold red]✗[/bold red] {exc}")
        raise typer.Exit(1)

    console.print(f"[green]\\[plan][/green]   {r.plan_bits:.2f} bit/w effettivi (budget {r.budget_bits:.2f})")
    console.print(f"[green]\\[plan][/green]   {r.assignment_summary}")
    console.print(f"[green]\\[quant][/green]  {r.original_size_mb:.0f} MB → {r.quantized_size_mb:.0f} MB")
    console.print(f"[green]\\[out][/green]    {r.output_path}")
    console.print(f"\n[bold green]✓ SGSR-2 compression complete[/bold green] — fits in {budget_gb:.2f} GB")


if __name__ == "__main__":
    app()
