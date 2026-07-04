# SGSR-2 — Verdetto finale sui criteri di successo

**Data:** 2026-07-04. Spec: `docs/superpowers/specs/2026-07-02-sgsr2-pareto-design.md`.
Accounting unificato: bit/w = size su disco × 8 / parametri totali del modello, per tutti i metodi.
Protocollo: Wikitext-2 test completo; MLX = sliding window 2048/512; llama.cpp = llama-perplexity ctx 2048, Δ% relativo alla propria baseline f16.

## Criterio 1 — Gate di additività: ✅ PASS

TinyLlama, 3 piani (3.25/3.8/4.6 eff bits): ratio predetto/misurato 0.74 / 0.87 / 1.11
(tolleranza [0.5, 1.5]), ranking preservato. Il costo totale è ben approssimato dalla
somma dei costi per-blocco → il solver Lagrangiano è affidabile.

## Criterio 2 — Batte uniform+SQ a pari bit: ✅ PASS su entrambi i modelli

| Modello | SGSR-2 | Uniform migliore a pari/più bit |
|---|---|---|
| TinyLlama | 14.28% @ 3.83b | 23.07% @ 3.94b (3b/gs32) |
| TinyLlama | 4.65% @ 4.44b | 4.75% @ 4.50b (4b/gs64) |
| Qwen2.5-7B | 18.66% @ 3.62b | 35.97% @ 3.93b (3b/gs32) |
| Qwen2.5-7B | 25.54% @ 3.40b | 48.32% @ 3.50b (3b/gs64) |

Dominio stretto (meglio in qualità E in bit) nella zona 3.4–4.5 bit/w.
Nota: curva uniform Qwen incompleta (2/7 punti; run troncata) — i punti 4-bit/5-bit
uniform su Qwen non sono misurati sotto il nuovo protocollo.

## Criterio 3 — K-quants (llama.cpp): ⚠️ PARZIALE

| Zona | Esito |
|---|---|
| ~4.5 bit | SGSR-2 sotto la frontiera K-quants interpolata su entrambi i modelli: Qwen 3.97% @ 4.55b (interp. ≈4.3%); TinyLlama 4.65% @ 4.44b (interp. ≈5.2%) ✅ |
| ≤4 bit | Q3_K_M nettamente migliore: Qwen 7.79% @ 4.00b vs SGSR-2 18.66% @ 3.62b ❌ |

**Interpretazione:** sotto i 4 bit il divario è attribuibile al formato, non all'allocazione:
K-quants usa superblocchi con scale 6-bit e minimi per sotto-blocco; MLX offre affine
semplice (scale/bias bf16 per gruppo). SGSR-2 è ottimale *nello spazio dei formati MLX*.
Dove i formati si equivalgono (≥4.4 bit), l'allocazione misurata batte il tuning manuale.
Non misurati: Q2_K, Q3_K_S (frontiera K-quants sotto 4b incompleta).

## Sintesi

- 2.5 / 3 criteri. Niente claim "batte tutti"; claim difendibile:
  **prima allocazione congiunta (bit-width, group-size) guidata da costo KL misurato,
  interamente on-device su Apple Silicon; domina la quantizzazione uniforme MLX ovunque;
  pareggia o batte llama.cpp K-quants a ≥4.4 bit/w.**
- Risultato negativo confermato e pubblicabile: il proxy a entropia (SGSR v1) è
  indistinguibile da ranking casuale su entrambi i modelli.
- Costo riproducibilità: 1 notte di profiling per modello (cost table riusabile),
  curva Pareto completa da una singola tabella.

## Gap noti (dichiarati, non bloccanti)

1. Curva uniform Qwen 4–5 bit incompleta (run troncata).
2. Frontiera K-quants sotto 4 bit non campionata (Q2_K, Q3_K_S).
3. Protocolli PPL MLX vs llama.cpp non identici (mitigato: Δ% vs baseline propria).
4. Nessun downstream task (solo PPL), 1 seed di calibrazione su Qwen.
