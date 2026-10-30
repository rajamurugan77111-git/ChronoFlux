# ChronoFlux v7.1 Feature Set

## Sensor / network intelligence (dual flow + packet)
- Demo seed, CSV, and PCAP ingest (live Ubuntu gateway deferred in this build)
- Source/destination IP, MAC, ports, protocol, service
- TCP flags (SYN / ACK / RST / FIN)
- Packet count / rate, inbound/outbound bytes, bidirectional ratio
- IAT mean / variance / max
- TTL mean / variance
- TCP window size
- Retransmission count, IP fragment flags
- Payload size distribution
- Port-scan signature score

## Detection
- Supervised Normal vs Attack model
- Attack-category classifier
- Isolation Forest anomaly score
- Transparent behavioural signatures
- Hybrid fusion with auditable risk formula

## World-model forecasting
- Structured state vector S_t
- LSTM learning P(S_t+1 | history)
- K-step forward simulation (T5/T6/T7)
- Infiltration probability timeline
- MITRE ATT&CK stage mapping
- Attention / SHAP-style driving features
- Logistic regression baseline comparison
- Sequence RF ablation

## Offline analysis
- CSV (UNSW-style detection)
- PCAP / PCAPNG (Scapy dual-feature windows + world-model forecast)

## Device visibility + evidence integrity
- Passive source-device registry
- SHA-256 tamper-evident evidence chain
