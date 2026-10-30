# Using CIC-IDS-2018 and CTU-13 with ChronoFlux

ChronoFlux auto-detects and converts these open-source datasets into its dual flow/packet feature schema.

## CIC-IDS-2018 (CSE-CIC-IDS2018)

Official: https://www.unb.ca/cic/datasets/ids-2018.html  
AWS Open Data: https://registry.opendata.aws/cse-cic-ids2018/

1. Download CICFlowMeter CSV day files into `data/raw/cicids2018/`
2. Convert (optional — Offline Analysis also auto-converts):

```powershell
.\venv\Scripts\python.exe data_tools\convert_cic_ids2018.py data\raw\cicids2018 --max-rows 100000
```

Mapped features include duration, fwd/bwd packets & bytes, Flow/Fwd/Bwd IAT mean/std/max, SYN/ACK/FIN/RST/**PSH/URG** flag counts, TCP window bytes, payload length means, ports/protocol.

## CTU-13

Official: https://www.stratosphereips.org/datasets-ctu13

1. Download bidirectional NetFlow (`.binetflow`) scenarios into `data/raw/ctu13/`
2. Convert:

```powershell
.\venv\Scripts\python.exe data_tools\convert_ctu13.py data\raw\ctu13 --max-rows 100000
```

Mapped fields: Dur, Proto, Src/Dst addr+port, State, TotPkts, TotBytes, SrcBytes, optional TTL/window/RTT when present, Label → ChronoFlux attack category.

## Demo without downloading full datasets

Bundled samples:

- `data/samples/cicids2018_sample.csv`
- `data/samples/ctu13_sample.csv`

Upload either file in **Offline Analysis** — the API detects the schema and converts automatically.

## Retrain detector on converted data (optional)

```powershell
# After conversion, merge selected converted CSVs into data/training.csv / data/testing.csv
.\venv\Scripts\python.exe -m training.train_all
```

World-model forecasting still prefers chronological sessions (lab/live/PCAP). Converted CIC/CTU rows are excellent for **current-state detection benchmarks** and offline demos required by NTRO.
