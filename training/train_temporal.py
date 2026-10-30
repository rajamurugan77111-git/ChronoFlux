"""Train/refresh the ChronoFlux sequence forecast model from chronological recording sessions."""
from core.history import read
from core.temporal_ml import train


def main():
    rows = read(limit=2500)
    if len(rows) < 12:
        raise SystemExit("Collect more chronological live/lab recording sessions first.")
    result = train(rows, force=True)
    print("Sequence forecast training complete")
    print(result)


if __name__ == "__main__":
    main()
