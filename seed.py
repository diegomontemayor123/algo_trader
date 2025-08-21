import os, subprocess, re, json, csv

# List of seeds you want to sweep
SEEDS = [42, 1, 2, 3, 4, 5, 7, 11, 21, 0, 77, 101, 123, 999]

def extract_metric(label, out):
    pattern = rf"{re.escape(label)}:\s*Strat:\s*([-+]?\d+(?:\.\d+)?)%"
    match = re.search(pattern, out, re.IGNORECASE)
    return float(match.group(1)) / 100 if match else None

def extract_avgoutperf(output):
    match = re.search(r"Avg Bench Outperf(?: thru Chunks)?:\s*\ncagr:\s*([-+]?\d+(?:\.\d+)?)%", output, re.MULTILINE)
    if match:
        return float(match.group(1)) / 100.0
    matches = re.findall(r"Avg Bench Outperf(?: thru Chunks)?:\s*([-+]?\d+(?:\.\d+)?)%", output)
    for val in reversed(matches):
        try:
            return float(val) / 100.0
        except:
            pass
    return 0.0

def extract_exp_delta(output):
    match = re.search(r"Total Exp Delta:\s*([-+]?\d+(?:\.\d+)?)", output)
    return float(match.group(1)) if match else None

def run_with_seed(base_config, seed):
    env = os.environ.copy()
    for k, v in base_config.items():
        env[k] = str(v)
    env["SEED"] = str(seed)

    python_exe = os.path.join(env.get("VIRTUAL_ENV", ""), "Scripts", "python.exe") if "VIRTUAL_ENV" in env else "python"

    proc = subprocess.Popen([python_exe, "model.py"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            env=env,
                            bufsize=1)

    output_lines = []
    for line in proc.stdout:
        print(line, end="")
        output_lines.append(line)
        if "KILLRUN" in line:
            print("[KILLRUN] — aborting early.")
            proc.kill()
            proc.wait()
            return None

    proc.wait()
    output = "".join(output_lines)

    sharpe = extract_metric("Sharpe Ratio", output)
    down = extract_metric("Max Down", output)
    cagr = extract_metric("CAGR", output)
    avg_outperf = extract_avgoutperf(output)
    exp_delta = extract_exp_delta(output)

    return {
        "seed": seed,
        "sharpe": sharpe,
        "down": down,
        "CAGR": cagr,
        "avg_outperf": avg_outperf,
        "exp_delta": exp_delta,
    }

def main():
    with open("hyparams.json", "r") as f:
        base_config = json.load(f)

    results = []
    for seed in SEEDS:
        print(f"\n=== Running with SEED={seed} ===")
        res = run_with_seed(base_config, seed)
        if res:
            results.append(res)

    # Save to CSV
    os.makedirs("csv", exist_ok=True)
    log_path = "csv/seed_sweep.csv"
    fieldnames = ["seed", "sharpe", "down", "CAGR", "avg_outperf", "exp_delta"]
    with open(log_path, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print("\n=== Seed sweep results ===")
    for r in results:
        print(r)

if __name__ == "__main__":
    main()
