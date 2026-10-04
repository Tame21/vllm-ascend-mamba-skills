#!/usr/bin/env python3
"""Compare small, logically aligned tensor/token snapshots without torch."""

import argparse
import json
import math
from pathlib import Path
import sys


def reject_constant(value):
    raise ValueError(f"Use quoted NaN/Inf strings instead of JSON constant {value}")


def read_records(path):
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".jsonl":
        rows = [
            json.loads(line, parse_constant=reject_constant)
            for line in text.splitlines()
            if line.strip()
        ]
    else:
        rows = json.loads(text, parse_constant=reject_constant)
        if isinstance(rows, dict):
            rows = [rows]
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: expected at least one record")
    result = {}
    for index, row in enumerate(rows):
        context = f"{path}: record {index}"
        if not isinstance(row, dict):
            raise ValueError(f"{context}: record must be an object")
        key = row.get("key")
        if not isinstance(key, str) or not key or key in result:
            raise ValueError(f"{context}: key must be a unique nonempty string")
        if row.get("kind") not in ("tensor", "exact"):
            raise ValueError(f"{context}: kind must be tensor or exact")
        if not isinstance(row.get("dtype"), str) or not row["dtype"]:
            raise ValueError(f"{context}: dtype must be a nonempty string")
        shape = row.get("shape")
        if not isinstance(shape, list) or any(
            type(dim) is not int or dim < 0 for dim in shape
        ):
            raise ValueError(f"{context}: shape must contain nonnegative integers")
        values = row.get("data")
        if not isinstance(values, list) or len(values) != math.prod(shape):
            raise ValueError(f"{context}: flattened data length disagrees with shape")
        normalized = []
        for value in values:
            if type(value) in (int, float):
                # Reject exponent overflow (e.g. 1e999) as well as NaN JSON.
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError(f"{context}: nonfinite values must be quoted")
                normalized.append(value)
            elif isinstance(value, str) and value in ("NaN", "Inf", "+Inf", "-Inf"):
                normalized.append(float(value))
            else:
                raise ValueError(f"{context}: data must be numeric or NaN/Inf strings")
        row["data"] = normalized
        result[key] = row
    return result


def printable(value):
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Inf" if value > 0 else "-Inf"
    return value


def coordinates(index, shape):
    result = []
    for dimension in reversed(shape):
        result.append(index % dimension)
        index //= dimension
    return list(reversed(result))


def is_finite(value):
    return isinstance(value, int) or math.isfinite(value)


def compare_record(reference, candidate, args):
    result = {"key": reference["key"]}
    for field in ("kind", "shape", "dtype"):
        if field == "dtype" and args.ignore_dtype:
            continue
        if reference[field] != candidate[field]:
            return dict(
                result,
                reason=f"{field}_mismatch",
                reference=reference[field],
                candidate=candidate[field],
            )
    mismatch_count = 0
    nonfinite_count = 0
    max_abs_error = 0
    first = None
    for index, (ref, cand) in enumerate(zip(reference["data"], candidate["data"])):
        finite = is_finite(ref) and is_finite(cand)
        if not finite:
            nonfinite_count += 1
            matches = args.allow_matching_inf and ref == cand
        elif reference["kind"] == "exact":
            matches = ref == cand
            max_abs_error = max(max_abs_error, abs(cand - ref))
        else:
            overflow_message = (
                f"{reference['key']}: numeric comparison overflow at flat index "
                f"{index}; finite values or tolerances exceed the supported "
                "comparison range"
            )
            try:
                error = abs(cand - ref)
                tolerance = args.atol + args.rtol * abs(ref)
            except OverflowError as exc:
                raise ValueError(overflow_message) from exc
            if not is_finite(error) or not is_finite(tolerance):
                raise ValueError(overflow_message)
            max_abs_error = max(max_abs_error, error)
            matches = error <= tolerance
        if not matches:
            mismatch_count += 1
            if first is None:
                first = {
                    "flat_index": index,
                    "coordinate": coordinates(index, reference["shape"]),
                    "reference": printable(ref),
                    "candidate": printable(cand),
                }
    if not mismatch_count:
        return None
    return dict(
        result,
        reason="value_mismatch",
        mismatch_count=mismatch_count,
        nonfinite_pair_count=nonfinite_count,
        max_finite_pair_abs_error=printable(max_abs_error),
        first=first,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--ignore-dtype", action="store_true")
    parser.add_argument("--allow-matching-inf", action="store_true")
    parser.add_argument("--max-records", type=int, default=20)
    args = parser.parse_args()
    try:
        if any(not math.isfinite(v) or v < 0 for v in (args.atol, args.rtol)):
            raise ValueError("atol/rtol must be finite and nonnegative")
        if args.max_records < 1:
            raise ValueError("max-records must be positive")
        reference = read_records(args.reference)
        candidate = read_records(args.candidate)
        mismatches = []
        for key, row in reference.items():
            if key not in candidate:
                mismatches.append({"key": key, "reason": "missing_in_candidate"})
            else:
                diff = compare_record(row, candidate[key], args)
                if diff:
                    mismatches.append(diff)
        mismatches.extend(
            {"key": key, "reason": "extra_in_candidate"}
            for key in candidate
            if key not in reference
        )
        report = {
            "status": "mismatch" if mismatches else "match",
            "reference_records": len(reference),
            "candidate_records": len(candidate),
            "mismatched_records": len(mismatches),
            "atol": args.atol,
            "rtol": args.rtol,
            "ignore_dtype": args.ignore_dtype,
            "allow_matching_inf": args.allow_matching_inf,
            "differences": mismatches[: args.max_records],
            "truncated": len(mismatches) > args.max_records,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 1 if mismatches else 0
    except (OSError, ValueError, OverflowError) as error:
        print(json.dumps({"status": "input_error", "error": str(error)}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
