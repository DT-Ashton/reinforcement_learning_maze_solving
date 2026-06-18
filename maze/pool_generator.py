import argparse
import json
import os

from maze.generator import MazeGenerator


def generate_pool(sizes: list = None, n_per_size: int = 3,
                   base_seed: int = 100, out_dir: str = "mazes") -> list:
    if sizes is None:
        sizes = [5, 10, 15]

    os.makedirs(out_dir, exist_ok=True)

    index_path = os.path.join(out_dir, "index.json")
    index = []
    used_seeds = set()
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        used_seeds = {entry["seed"] for entry in index}

    written = []

    for size in sizes:
        for i in range(n_per_size):
            seed = base_seed + size * 100 + i
            while seed in used_seeds:
                seed += 1
            used_seeds.add(seed)

            grid = MazeGenerator(size, seed).generate()

            filename = f"maze_{size}x{size}_seed{seed}.json"
            path = os.path.join(out_dir, filename)
            data = {
                "size": size,
                "seed": seed,
                "start": [0, 0],
                "goal": [size - 1, size - 1],
                "grid": grid.tolist(),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)

            written.append(path)
            index.append({"file": filename, "size": size, "seed": seed})

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a pool of random mazes")
    parser.add_argument("--sizes", type=int, nargs="+", default=[5, 10, 15])
    parser.add_argument("--n-per-size", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=100)
    parser.add_argument("--out-dir", type=str, default="mazes")
    args = parser.parse_args()

    files = generate_pool(args.sizes, args.n_per_size, args.base_seed, args.out_dir)
    print(f"Generated {len(files)} mazes in '{args.out_dir}/' "
          f"(sizes={args.sizes}, n_per_size={args.n_per_size})")
