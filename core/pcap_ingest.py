"""PCAP → ChronoFlux flow/packet feature windows.

Parses packet captures with Scapy into timestamped windows covering both
flow-level aggregates and packet-level attributes (TTL, window, IAT, fragments,
retransmits, payload size, port-scan signals).
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import math
import tempfile

FEATURE_DEFAULTS = {
    "dur": 0.0, "proto": "other", "service": "-", "state": "INT",
    "spkts": 0, "dpkts": 0, "sbytes": 0, "dbytes": 0, "rate": 0.0,
    "sttl": 0, "dttl": 0, "sload": 0.0, "dload": 0.0, "sloss": 0, "dloss": 0,
    "sinpkt": 0.0, "dinpkt": 0.0, "sjit": 0.0, "djit": 0.0,
    "swin": 0, "dwin": 0, "tcprtt": 0.0, "synack": 0.0, "ackdat": 0.0,
    "smean": 0.0, "dmean": 0.0, "ct_srv_src": 0, "ct_state_ttl": 0,
    "ct_dst_ltm": 0, "ct_src_dport_ltm": 0, "ct_dst_sport_ltm": 0,
    "ct_dst_src_ltm": 0, "ct_src_ltm": 0, "ct_srv_dst": 0,
}


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _var(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _avg(values)
    return sum((v - m) ** 2 for v in values) / len(values)


def _service(port: int | None) -> str:
    mapping = {80: "http", 443: "https", 22: "ssh", 53: "dns", 25: "smtp", 21: "ftp", 3389: "rdp"}
    return mapping.get(int(port or 0), "-")


def parse_pcap_bytes(data: bytes, window_seconds: float = 5.0, max_packets: int = 20000) -> list[dict]:
    """Parse raw PCAP bytes into ChronoFlux window records."""
    try:
        from scapy.all import rdpcap, IP, IPv6, TCP, UDP, ICMP, Ether  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Scapy is required for PCAP analysis. Install with: pip install scapy"
        ) from exc

    with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as tmp:
        tmp.write(data)
        path = Path(tmp.name)

    try:
        packets = rdpcap(str(path))
    finally:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass

    if not packets:
        return []

    windows: dict[int, dict[str, Any]] = {}

    def bucket(ts: float) -> int:
        return int(ts // max(window_seconds, 0.1))

    def ensure(idx: int) -> dict:
        if idx not in windows:
            windows[idx] = {
                "times": [], "src_ttls": [], "dst_ttls": [], "src_wins": [], "payloads": [],
                "spkts": 0, "dpkts": 0, "sbytes": 0, "dbytes": 0,
                "syn_count": 0, "ack_count": 0, "rst_count": 0, "fin_count": 0, "icmp_count": 0,
                "retransmit_count": 0, "fragment_flag_count": 0,
                "dst_ports": set(), "src_ports": set(), "dst_ips": set(), "tcp_flags": set(),
                "services_seen": set(), "seen_seq": set(),
                "src_ip": None, "dst_ip": None, "src_mac": None, "dst_mac": None,
                "primary_src_port": None, "primary_dst_port": None, "proto": "other", "service": "-", "state": "INT",
            }
        return windows[idx]

    count = 0
    for pkt in packets:
        if count >= max_packets:
            break
        if not pkt.haslayer(IP) and not pkt.haslayer(IPv6):
            continue
        count += 1
        ts = float(getattr(pkt, "time", 0.0) or 0.0)
        w = ensure(bucket(ts))
        w["times"].append(ts)

        if pkt.haslayer(Ether):
            w["src_mac"] = w["src_mac"] or str(pkt[Ether].src)
            w["dst_mac"] = w["dst_mac"] or str(pkt[Ether].dst)

        ip = pkt[IP] if pkt.haslayer(IP) else pkt[IPv6]
        src = getattr(ip, "src", None)
        dst = getattr(ip, "dst", None)
        w["src_ip"] = w["src_ip"] or src
        w["dst_ip"] = w["dst_ip"] or dst
        if dst:
            w["dst_ips"].add(str(dst))

        # Fragment flags (IPv4)
        if pkt.haslayer(IP):
            flags = int(getattr(ip, "flags", 0) or 0)
            frag = int(getattr(ip, "frag", 0) or 0)
            if flags & 0x1 or frag > 0:  # MF or offset
                w["fragment_flag_count"] += 1
            w["src_ttls"].append(int(getattr(ip, "ttl", 0) or 0))

        payload_len = int(len(bytes(pkt.payload)) if hasattr(pkt, "payload") else 0)
        w["payloads"].append(payload_len)

        if pkt.haslayer(TCP):
            tcp = pkt[TCP]
            w["proto"] = "tcp"
            sport, dport = int(tcp.sport), int(tcp.dport)
            w["src_ports"].add(sport)
            w["dst_ports"].add(dport)
            w["primary_src_port"] = w["primary_src_port"] or sport
            w["primary_dst_port"] = w["primary_dst_port"] or dport
            w["service"] = _service(dport)
            w["services_seen"].add(w["service"])
            w["src_wins"].append(int(getattr(tcp, "window", 0) or 0))
            flags = str(tcp.flags)
            for name, bit in (("S", "syn_count"), ("A", "ack_count"), ("R", "rst_count"), ("F", "fin_count")):
                if name in flags:
                    w[bit] += 1
                    w["tcp_flags"].add(name)
            # Retransmit heuristic: repeated SEQ
            seq = int(getattr(tcp, "seq", 0) or 0)
            key = (src, dst, sport, dport, seq)
            if key in w["seen_seq"]:
                w["retransmit_count"] += 1
            else:
                w["seen_seq"].add(key)
            w["spkts"] += 1
            w["sbytes"] += max(payload_len, 0)
            w["state"] = "CON" if "A" in flags else "INT"
        elif pkt.haslayer(UDP):
            udp = pkt[UDP]
            w["proto"] = "udp"
            sport, dport = int(udp.sport), int(udp.dport)
            w["src_ports"].add(sport)
            w["dst_ports"].add(dport)
            w["primary_src_port"] = w["primary_src_port"] or sport
            w["primary_dst_port"] = w["primary_dst_port"] or dport
            w["service"] = _service(dport)
            w["spkts"] += 1
            w["sbytes"] += max(payload_len, 0)
        elif pkt.haslayer(ICMP):
            w["proto"] = "icmp"
            w["icmp_count"] += 1
            w["spkts"] += 1

    records = []
    for idx in sorted(windows):
        w = windows[idx]
        times = sorted(w["times"])
        dur = max((times[-1] - times[0]) if len(times) > 1 else window_seconds, 1e-6)
        diffs_ms = [(times[i] - times[i - 1]) * 1000 for i in range(1, len(times))]
        iat_mean = _avg(diffs_ms)
        iat_var = _var(diffs_ms)
        iat_max = max(diffs_ms) if diffs_ms else 0.0
        total = w["spkts"] + w["dpkts"]
        features = {
            **FEATURE_DEFAULTS,
            "dur": dur,
            "proto": w["proto"],
            "service": w["service"],
            "state": w["state"],
            "spkts": w["spkts"],
            "dpkts": w["dpkts"],
            "sbytes": w["sbytes"],
            "dbytes": w["dbytes"],
            "rate": total / max(window_seconds, 1e-6),
            "sttl": round(_avg(w["src_ttls"])),
            "dttl": round(_avg(w["dst_ttls"])),
            "sload": (w["sbytes"] * 8) / dur,
            "dload": (w["dbytes"] * 8) / dur,
            "sloss": w["retransmit_count"],
            "sinpkt": iat_mean,
            "dinpkt": iat_mean,
            "sjit": iat_var,
            "djit": iat_var,
            "swin": round(_avg(w["src_wins"])),
            "smean": _avg(w["payloads"]),
            "ct_srv_src": len(w["dst_ports"]),
            "ct_state_ttl": 1 if total else 0,
            "ct_dst_ltm": len(w["dst_ports"]),
            "ct_src_dport_ltm": len(w["dst_ports"]),
            "ct_dst_sport_ltm": len(w["src_ports"]),
            "ct_dst_src_ltm": len(w["dst_ips"]),
            "ct_src_ltm": 1 if total else 0,
            "ct_srv_dst": len(w["dst_ports"]),
        }
        telemetry = {
            "packet_count": total,
            "packet_rate": features["rate"],
            "unique_dst_ports": len(w["dst_ports"]),
            "unique_src_ports": len(w["src_ports"]),
            "unique_dst_ips": len(w["dst_ips"]),
            "syn_count": w["syn_count"],
            "ack_count": w["ack_count"],
            "rst_count": w["rst_count"],
            "fin_count": w["fin_count"],
            "icmp_count": w["icmp_count"],
            "outbound_bytes": w["sbytes"],
            "inbound_bytes": w["dbytes"],
            "source_ip": w["src_ip"],
            "source_port": w["primary_src_port"],
            "primary_destination": w["dst_ip"],
            "destination_port": w["primary_dst_port"],
            "protocol": w["proto"],
            "service": w["service"],
            "tcp_flags": sorted(w["tcp_flags"]),
            "services_seen": sorted(x for x in w["services_seen"] if x and x != "-"),
            "dst_ports_sample": sorted(w["dst_ports"])[:30],
            "destination_ips_sample": sorted(w["dst_ips"])[:10],
            "source_mac": w["src_mac"],
            "destination_mac": w["dst_mac"],
            "window_seconds": window_seconds,
            "iat_mean": iat_mean,
            "iat_variance": iat_var,
            "iat_max": iat_max,
            "ttl_mean": _avg(w["src_ttls"]),
            "ttl_variance": _var(w["src_ttls"]),
            "tcp_window_mean": _avg(w["src_wins"]),
            "retransmit_count": w["retransmit_count"],
            "fragment_flag_count": w["fragment_flag_count"],
            "payload_size_mean": _avg(w["payloads"]),
        }
        ts_iso = datetime.fromtimestamp(times[0] if times else 0, tz=timezone.utc).isoformat()
        records.append({"timestamp": ts_iso, "features": features, "telemetry": telemetry})
    return records


def parse_pcap_file(path: str | Path, window_seconds: float = 5.0) -> list[dict]:
    data = Path(path).read_bytes()
    return parse_pcap_bytes(data, window_seconds=window_seconds)
