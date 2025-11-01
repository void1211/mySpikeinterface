import argparse
import os
import re
from pathlib import Path

import numpy as np
from scipy.io import savemat


def sanitize_matlab_varname(name: str) -> str:
	"""Convert arbitrary filename to a safe MATLAB variable name."""
	# Remove extension and replace non-word chars with underscores
	var = re.sub(r"\W+", "_", name)
	# MATLAB variables cannot start with a digit
	if var and var[0].isdigit():
		var = f"v_{var}"
	# Avoid empty names
	return var or "var"


def load_npy_for_mat(path: Path):
	"""Load .npy safely for savemat.

	- allow_pickle=True to cover assorted metadata arrays
	- convert object arrays to Python lists so savemat handles them
	"""
	arr = np.load(str(path), allow_pickle=True)
	if isinstance(arr, np.ndarray) and arr.dtype == object:
		# Convert object arrays (ragged, lists of arrays) to list for savemat
		return arr.tolist()
	return arr


def collect_npy_files(root: Path, recursive: bool) -> list[Path]:
	pattern = "**/*.npy" if recursive else "*.npy"
	return sorted([p for p in root.glob(pattern) if p.is_file()])


def build_mat_dict(npy_files: list[Path], base_dir: Path, include_relpath: bool) -> dict:
	mat_dict: dict[str, object] = {}
	for f in npy_files:
		key_base = f.relative_to(base_dir).as_posix() if include_relpath else f.name
		key_no_ext = os.path.splitext(key_base)[0]
		varname = sanitize_matlab_varname(key_no_ext)
		# Disambiguate duplicates by appending an index
		candidate = varname
		idx = 2
		while candidate in mat_dict:
			candidate = f"{varname}_{idx}"
			idx += 1
		mat_dict[candidate] = load_npy_for_mat(f)
	return mat_dict


def main():
	parser = argparse.ArgumentParser(
		description=(
			"Convert Kilosort4 (or any) .npy outputs in a folder to a single .mat file.\n"
			"Each .npy becomes one variable named after the file (sanitized)."
		)
	)
	parser.add_argument("input_dir", type=Path, help="Directory containing .npy files")
	parser.add_argument("output_mat", type=Path, help="Output .mat filepath")
	parser.add_argument(
		"--recursive",
		action="store_true",
		help="Search .npy files recursively",
	)
	parser.add_argument(
		"--keep-relpath",
		action="store_true",
		help=(
			"Include relative subpath in variable names (useful with --recursive).\n"
			"Example: 'subdir/spike_times' becomes variable 'subdir_spike_times'"
		),
	)
	parser.add_argument(
		"--compress",
		action="store_true",
		help="Use compression when saving .mat",
	)

	args = parser.parse_args()

	input_dir: Path = args.input_dir
	output_mat: Path = args.output_mat

	if not input_dir.exists() or not input_dir.is_dir():
		raise SystemExit(f"Input directory not found: {input_dir}")

	npy_files = collect_npy_files(input_dir, recursive=args.recursive)
	if not npy_files:
		raise SystemExit("No .npy files found.")

	mat_dict = build_mat_dict(
		npy_files=npy_files,
		base_dir=input_dir,
		include_relpath=args.keep_relpath,
	)

	# Add small manifest
	mat_dict["__manifest__"] = {
		"source_dir": input_dir.as_posix(),
		"file_count": len(npy_files),
		"recursive": bool(args.recursive),
		"kept_relpath": bool(args.keep_relpath),
	}

	output_mat.parent.mkdir(parents=True, exist_ok=True)
	savemat(str(output_mat), mat_dict, do_compression=bool(args.compress))
	print(f"Saved {len(npy_files)} arrays to: {output_mat}")


if __name__ == "__main__":
	main()


