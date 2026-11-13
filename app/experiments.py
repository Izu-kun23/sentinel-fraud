"""Reproducible training and validation entry point; never promotes a model automatically."""

import argparse, json
from app.training import build_artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--csv")
    args = parser.parse_args()
    manifest, directory, record = build_artifact(args.output, args.csv)
    print(
        json.dumps(
            {
                "candidate_directory": str(directory),
                "experiment_record": str(record),
                "artifact_sha256": manifest["artifact_sha256"],
                "selected_model": manifest["selected_model"],
                "quality_gates": manifest["quality_gates"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
