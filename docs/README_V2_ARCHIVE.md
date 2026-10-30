# HISTORICAL README ARCHIVE — NOT CURRENT IMPLEMENTATION

This file is retained only for project history. Use the root `README.md` for v7 architecture, terminology, model claims and evaluation.

# ChronoFlux v2 — Detect · Forecast · Verify

ChronoFlux is a dual-mode network security intelligence console designed as a
real browser application rather than a Streamlit demo. One FastAPI server
serves both the APIs and the responsive web console.

## Core design

ChronoFlux now combines five layers:

1. **Deep flow telemetry** — source/destination IP, source/destination ports,
   protocol, service, TCP flags, packet/byte rates, SYN/RST/ICMP activity,
   destination-port fan-out and destination-IP diversity.
2. **Hybrid current-threat detection** — supervised ML + normal-baseline
   anomaly scoring + transparent behavioural signatures.
3. **T1 → T4 behaviour history** — consecutive live gateway windows.
4. **Next-stage forecasting** — a clearly labelled behavioural sequence
   baseline, with an upgrade slot for a genuine temporal ML model.
5. **Private Security Evidence Chain** — elevated security events are hashed
   with SHA-256 and linked block-to-block so later modification is detectable.

## Web pages

- Overview
- Live Traffic
- Offline Analysis
- Forecasting
- Alerts
- Evidence Chain
- System

The UI includes light/dark modes, responsive navigation, real-time charts,
connection intelligence, risk/explainability panels and tamper-evidence status.

## Blockchain / cybersecurity note

The packaged Evidence Chain is a **private local tamper-evident hash-chained
ledger**. It demonstrates blockchain-style evidence integrity without falsely
claiming decentralized consensus. Each elevated event stores a compact metadata
record, an event hash, the previous block hash and the current block hash.

For a later external/final review, the same event schema can be anchored to
Hyperledger Fabric, Ethereum or another permissioned ledger.

## Windows

Extract the ZIP, then double-click:

`START_CHRONOFLUX.bat`

First run creates the virtual environment, installs dependencies, starts the
server and opens:

`http://127.0.0.1:8000`

Manual alternative:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run.py
```

## Ubuntu gateway sensor

Copy `live/ubuntu_sensor.py` to Ubuntu.

```bash
python3 -m venv ~/chronoflux-env
source ~/chronoflux-env/bin/activate
pip install -r requirements-ubuntu-sensor.txt
```

Current VirtualBox lab:

```bash
sudo ~/chronoflux-env/bin/python ~/ubuntu_sensor.py \
  --interface enp0s8 \
  --window 10 \
  --server http://192.168.56.1:8000
```

The sensor extracts metadata only. It does not inspect application payload
contents.

## Kali

Kali is only the authorized controlled traffic/test side in the isolated lab.
Do not scan or attack a real college network unless the institution's network
or security team explicitly authorizes a defined test scope.

## Live architecture

```text
Kali / Authorized Network Traffic
              ↓
        Ubuntu Gateway
              ↓
   Packet + Flow Metadata
              ↓
        ChronoFlux API
              ↓
 ┌────────────┼──────────────┐
 │            │              │
ML        Anomaly        Signatures
 └────────────┼──────────────┘
              ↓
        Hybrid Decision
              ↓
 Current State + Risk + Why
              ↓
       T1 → T2 → T3 → T4
              ↓
      Sequence Forecast
              ↓
      Next Likely State
              ↓
        Elevated Event
              ↓
      SHA-256 Evidence Hash
              ↓
      Private Evidence Chain
              ↓
        Integrity Verified
```

## Forecasting honesty

The packaged build does not claim that the sequence baseline is a trained
temporal ML model. The UI says behavioural baseline. A genuine temporal model
requires ordered timestamped traffic with real progression labels.

## Data/model note

The packaged pre-trained current-stage models are carried forward from the
working ChronoFlux build so the application runs immediately. Replace the
training/testing files with verified real benchmark splits before reporting
final evaluation metrics. Do not report attack-detection performance from an
all-normal test split.

## First-review demo story

1. Start the Windows web console.
2. Start Ubuntu gateway sensing.
3. Show normal traffic and deep endpoint/port/protocol metadata.
4. Show T1–T4 filling with real windows.
5. Generate only authorized controlled lab activity.
6. Show hybrid detection, anomaly/risk, explanation and forecast.
7. Open **Evidence Chain** and verify that the elevated event was committed to
   the SHA-256 hash-linked ledger.
8. Use Offline Analysis for the benchmark/dataset demonstration.
