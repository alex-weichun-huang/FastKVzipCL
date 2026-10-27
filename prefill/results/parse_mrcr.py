import glob
import json
import os


def aggregate_scores(model_name, level, root_dir="./results/mrcr", idx=0, num=None):
    # 1. Define the pattern to find the specific files in the specific folders
    # Matches: [Any Prefix]_qwen2.5... / output-adakv-layer.json
    folder_pattern = f"*_{model_name}"
    filename = f"output-{level}.json"
    search_path = os.path.join(root_dir, folder_pattern, filename)

    # Find all matching files
    files = glob.glob(search_path)

    # Keep only sample indices in [idx, idx + num); folders are named <idx>_<model>
    def sample_idx(path):
        return int(os.path.basename(os.path.dirname(path)).split("_", 1)[0])

    files = [f for f in files if sample_idx(f) >= idx]
    if num is not None:
        files = [f for f in files if sample_idx(f) < idx + num]

    if not files:
        print(f"No files found matching pattern: {search_path}")
        return

    print(f"Found {len(files)} files. Processing...")

    # Dictionary to store scores: {'1.0': [0.0376, 0.04...], '0.75': [...]}
    score_aggregator = {}

    # 2. Iterate through files and collect scores
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

                for ratio_key, content in data.items():
                    # Ensure the key contains a score before trying to access it
                    if isinstance(content, dict) and "score" in content:
                        score = content["score"]

                        if ratio_key not in score_aggregator:
                            score_aggregator[ratio_key] = []

                        score_aggregator[ratio_key].append(score)

        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading {file_path}: {e}")

    # 3. Calculate Averages
    averaged_results = {}

    for ratio, scores in score_aggregator.items():
        if scores:
            avg_score = sum(scores) / len(scores) * 100
            averaged_results[ratio] = round(avg_score, 2)  # Rounding for readability

    return averaged_results, len(files)


if __name__ == "__main__":
    import argparse

    from results.parse import get_eviction_level

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="qwen3-4b-instruct-2507_fastkvzip_chunk16k_w4096",
    )
    parser.add_argument("-l", "--level", type=str, default="")
    parser.add_argument("--idx", type=int, default=0, help="first sample index")
    parser.add_argument("-n", "--num", type=int, default=None, help="number of samples")
    parser.add_argument(
        "-o", "--out", type=str, default="", help="save scores to this JSON file"
    )
    args = parser.parse_args()

    if args.level == "":
        args.level = get_eviction_level(args.model)

    out = aggregate_scores(args.model, args.level, idx=args.idx, num=args.num)
    if out is None:
        raise SystemExit(1)
    results, n_files = out

    # Print results nicely formatted
    print("\n--- Averaged Scores per Ratio ---")
    print(json.dumps(results, indent=4))

    if args.out:
        full = results.get("1.0")
        rel = {
            r: round(v / full * 100, 2) if full else float("nan")
            for r, v in results.items()
        }
        summary = {
            "method": args.model,
            "level": args.level,
            "datasets": {
                "mrcr": {"num_samples": n_files, "score": results, "relative_score": rel}
            },
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"Scores saved at {args.out}")
