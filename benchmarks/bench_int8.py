"""Benchmark int8 vs fp16 vs fp32 across input lengths on GPU and CPU.

Measures latency and model memory footprint. Interleaves conditions within
the same process to avoid warm-cache bias (per CLAUDE.md benchmarking rules).
"""

import gc
import json
import os
import statistics
import time
from datetime import datetime

import torch
from gliner import GLiNER

MODEL_NAME = "urchade/gliner_small-v2.1"
LABELS = ["person", "organization", "location", "date", "event"]
N_REPS = 40
N_WARMUP = 5

# Inputs of increasing length
INPUTS = {
    "short (~20w)": (
        "Elon Musk founded SpaceX in Hawthorne, California in 2002."
    ),
    "medium (~80w)": (
        "The United Nations General Assembly convened in New York City on "
        "September 15, 2024, where Secretary-General Antonio Guterres "
        "addressed delegates from 193 member states. Key topics included "
        "climate change mitigation, the ongoing conflict in Eastern Europe, "
        "and global economic recovery following the pandemic. Representatives "
        "from the European Union, African Union, and ASEAN presented joint "
        "proposals for sustainable development goals. The World Health "
        "Organization also provided updates on disease surveillance programs "
        "across Sub-Saharan Africa and Southeast Asia."
    ),
    "long (~200w)": (
        "In a landmark announcement on March 15, 2024, the European Space "
        "Agency and NASA jointly revealed plans for the Artemis-Europa "
        "collaborative mission, scheduled for launch from Kennedy Space "
        "Center in late 2028. The mission, overseen by project director "
        "Dr. Maria Chen and deputy director Professor James Okafor from "
        "the University of Cambridge, aims to deploy an autonomous "
        "submarine probe beneath the ice crust of Jupiter's moon Europa. "
        "The probe, named Poseidon, was developed by a consortium including "
        "Lockheed Martin, Airbus Defence, and the Japan Aerospace "
        "Exploration Agency. Testing began at the Jet Propulsion Laboratory "
        "in Pasadena in January 2023 and continued at facilities in "
        "Toulouse, France and Tsukuba, Japan. The European Commission has "
        "allocated 2.3 billion euros to the project through the Horizon "
        "Europe framework. Meanwhile, the National Science Foundation "
        "contributed an additional 800 million dollars. Critics from the "
        "Planetary Society and the International Astronomical Union have "
        "raised concerns about contamination protocols. A review panel "
        "chaired by Dr. Sarah Williams of MIT published findings in Nature "
        "Astronomy suggesting the mission's sterilization procedures exceed "
        "those used in the Viking and Curiosity missions. President Biden "
        "praised the initiative during a ceremony at the White House, "
        "calling it a triumph of international cooperation."
    ),
    "very long (~400w)": (
        "The 2024 Global Technology Summit, hosted by the World Economic "
        "Forum in Davos, Switzerland from January 15 to January 19, brought "
        "together over 2,800 leaders from industry, government, and "
        "academia. Microsoft CEO Satya Nadella delivered the opening keynote, "
        "outlining the company's vision for artificial intelligence "
        "integration across enterprise software. Google DeepMind's CEO "
        "Demis Hassabis presented breakthroughs in protein structure "
        "prediction following their AlphaFold 3 release. Tesla and SpaceX "
        "founder Elon Musk participated in a panel discussion on autonomous "
        "systems with Waymo CEO Tekedra Mawakana and General Motors "
        "president Mark Reuss. The European Commission's Executive "
        "Vice-President Margrethe Vestager announced new regulatory "
        "frameworks for AI governance under the EU AI Act, which had been "
        "formally adopted in December 2023. China's Ministry of Science and "
        "Technology sent a delegation led by Minister Yin Hejun, who "
        "presented China's national AI development roadmap through 2030. "
        "Japan's Prime Minister Fumio Kishida announced a 5 billion dollar "
        "investment in semiconductor manufacturing, with new facilities "
        "planned in Kumamoto and Hokkaido in partnership with Taiwan "
        "Semiconductor Manufacturing Company. Samsung Electronics vice "
        "chairman Jay Y. Lee discussed the company's 230 billion dollar "
        "investment plan for chip fabrication plants in Taylor, Texas and "
        "Pyeongtaek, South Korea. The Bill and Melinda Gates Foundation "
        "unveiled a 500 million dollar initiative for AI-powered healthcare "
        "diagnostics in Sub-Saharan Africa, developed in collaboration with "
        "the World Health Organization and Doctors Without Borders. "
        "Stanford University's Institute for Human-Centered AI released "
        "their annual AI Index Report, compiled by researchers including "
        "Professor Fei-Fei Li and Dr. Erik Brynjolfsson. The report "
        "highlighted that global AI investment reached 189 billion dollars "
        "in 2023, with the United States, China, and the United Kingdom "
        "accounting for 75 percent of total spending. OpenAI CEO Sam "
        "Altman and Anthropic CEO Dario Amodei held a joint session on AI "
        "safety research, discussing alignment techniques and the need for "
        "international cooperation on frontier model evaluation. The summit "
        "concluded with the Davos AI Accord, signed by representatives "
        "from 47 nations, establishing shared principles for responsible "
        "AI development and deployment across borders."
    ),
}


def get_model_size_mb(model):
    """Estimate model parameter memory in MB."""
    total = 0
    for p in model.parameters():
        total += p.nelement() * p.element_size()
    for b in model.buffers():
        total += b.nelement() * b.element_size()
    return total / (1024 * 1024)


def get_torchao_model_size_mb(model):
    """Estimate size including torchao quantized tensors."""
    total = 0
    for name, p in model.named_parameters():
        total += p.nelement() * p.element_size()
    for name, b in model.named_buffers():
        total += b.nelement() * b.element_size()
    # torchao int8 stores weights as module attributes, not always as parameters
    for mod in model.modules():
        if hasattr(mod, "weight") and not isinstance(mod.weight, torch.nn.Parameter):
            w = mod.weight
            if hasattr(w, "nelement"):
                total += w.nelement() * w.element_size()
    return total / (1024 * 1024)


def measure_latency(model, text, labels, n_warmup, n_reps):
    """Measure inference latency with warmup, return list of times in ms."""
    for _ in range(n_warmup):
        model.predict_entities(text, labels)

    if model.device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_reps):
        if model.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.predict_entities(text, labels)
        if model.device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return times


def run_benchmark(device: str):
    print(f"\n{'='*70}")
    print(f"  DEVICE: {device.upper()}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Reps: {N_REPS} (warmup: {N_WARMUP})")
    print(f"{'='*70}")

    results = {}

    # --- Load models ---
    conditions = {}

    # fp32
    print("\nLoading fp32 model...")
    conditions["fp32"] = GLiNER.from_pretrained(MODEL_NAME, map_location=device)

    # fp16
    print("Loading fp16 model...")
    conditions["fp16"] = GLiNER.from_pretrained(
        MODEL_NAME, map_location=device, quantize="fp16"
    )

    # int8
    print("Loading int8 model...")
    conditions["int8"] = GLiNER.from_pretrained(
        MODEL_NAME, map_location=device, quantize="int8"
    )

    # --- Memory ---
    print("\n--- Model Size (parameters + buffers) ---")
    for cond_name, model in conditions.items():
        if cond_name == "int8":
            size = get_torchao_model_size_mb(model.model)
        else:
            size = get_model_size_mb(model.model)
        results.setdefault(cond_name, {})["size_mb"] = round(size, 1)
        print(f"  {cond_name:>5}: {size:>8.1f} MB")

    # --- Latency per input length ---
    for input_name, text in INPUTS.items():
        word_count = len(text.split())
        print(f"\n--- {input_name} ({word_count} words) ---")
        header = f"  {'cond':>5}  {'mean':>8}  {'median':>8}  {'stdev':>8}  {'min':>8}  {'max':>8}"
        print(header)

        for cond_name, model in conditions.items():
            times = measure_latency(model, text, LABELS, N_WARMUP, N_REPS)
            mean = statistics.mean(times)
            med = statistics.median(times)
            sd = statistics.stdev(times)
            mn = min(times)
            mx = max(times)

            results.setdefault(cond_name, {})[input_name] = {
                "mean_ms": round(mean, 2),
                "median_ms": round(med, 2),
                "stdev_ms": round(sd, 2),
                "min_ms": round(mn, 2),
                "max_ms": round(mx, 2),
                "n": N_REPS,
                "word_count": word_count,
            }
            print(
                f"  {cond_name:>5}  {mean:>7.2f}ms  {med:>7.2f}ms  "
                f"{sd:>7.2f}ms  {mn:>7.2f}ms  {mx:>7.2f}ms"
            )

    # --- Speedup summary ---
    print(f"\n--- Speedup vs fp32 (median latency) ---")
    header = f"  {'input':>20}"
    for cond_name in conditions:
        header += f"  {cond_name:>10}"
    print(header)

    for input_name in INPUTS:
        fp32_med = results["fp32"][input_name]["median_ms"]
        row = f"  {input_name:>20}"
        for cond_name in conditions:
            med = results[cond_name][input_name]["median_ms"]
            speedup = fp32_med / med
            row += f"  {speedup:>9.2f}x"
        print(row)

    # Cleanup
    for model in conditions.values():
        del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return results


def main():
    all_results = {"timestamp": datetime.now().isoformat(), "model": MODEL_NAME}

    # GPU benchmark
    if torch.cuda.is_available():
        all_results["gpu"] = run_benchmark("cuda")
        gc.collect()
        torch.cuda.empty_cache()

    # CPU benchmark
    all_results["cpu"] = run_benchmark("cpu")

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = os.path.join(os.path.dirname(__file__), f"bench_int8_{ts}.json")
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    main()
