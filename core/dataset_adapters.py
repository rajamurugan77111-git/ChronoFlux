"""Adapters that map CIC-IDS-2018 and CTU-13 flow CSVs into ChronoFlux schema.

Both datasets are converted to:
  - FEATURE_COLUMNS (detection / offline analysis)
  - optional telemetry dicts (world-model state vectors)
  - attack_cat / label targets

Column-name variants (spaces vs underscores, CICFlowMeter aliases) are handled.
"""
from __future__ import annotations

from typing import Any
import math
import re

import numpy as np
import pandas as pd

from core.schema import FEATURE_COLUMNS, DEFAULT_FEATURES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _colmap(df: pd.DataFrame) -> dict[str, str]:
    return {_norm_col(c): c for c in df.columns}


def _get(row: pd.Series, cmap: dict[str, str], *aliases: str, default: Any = 0.0):
    for alias in aliases:
        key = _norm_col(alias)
        if key in cmap:
            value = row[cmap[key]]
            if pd.isna(value):
                return default
            return value
    return default


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        value = float(value)
        if not math.isfinite(value):
            return default
        return value
    except Exception:
        return default


def _proto(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "other"
    text = str(value).strip().lower()
    if text in {"6", "tcp"}:
        return "tcp"
    if text in {"17", "udp"}:
        return "udp"
    if text in {"1", "icmp"}:
        return "icmp"
    if text in {"tcp", "udp", "icmp", "arp"}:
        return text
    return "other"


def _service_from_port(port: Any) -> str:
    try:
        p = int(float(port))
    except Exception:
        return "-"
    return {
        20: "ftp-data", 21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp",
        53: "dns", 80: "http", 110: "pop3", 143: "imap", 443: "https",
        445: "smb", 3389: "rdp", 3306: "mysql", 8080: "http",
    }.get(p, "-")


def detect_dataset(df: pd.DataFrame) -> str:
    """Return 'cicids2018', 'ctu13', 'chronoflux', or 'unknown'."""
    cols = {_norm_col(c) for c in df.columns}
    if {"proto", "spkts", "dpkts", "sbytes", "dbytes"}.issubset(cols) and (
        "attackcat" in cols or "label" in cols
    ):
        return "chronoflux"
    if {"flowduration", "totfwdpkts", "flowiatmean"}.issubset(cols) or {
        "flowduration",
        "totalfwdpackets",
        "flowiatmean",
    }.issubset(cols):
        return "cicids2018"
    if {"synflagcnt", "ackflagcnt", "flowiatstd"}.issubset(cols) and "label" in cols:
        return "cicids2018"
    if {"starttime", "dur", "srcaddr", "dstaddr", "totpkts", "totbytes"}.issubset(cols):
        return "ctu13"
    if {"dur", "proto", "srcaddr", "dstaddr", "totpkts"}.issubset(cols) and "label" in cols:
        return "ctu13"
    return "unknown"


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

CIC_LABEL_TO_CAT = {
    "benign": "Normal",
    "normal": "Normal",
    "bot": "Backdoor",
    "botnet": "Backdoor",
    "ddos": "DoS",
    "ddos attack-hoic": "DoS",
    "ddos attack-loic-http": "DoS",
    "ddos attacks-loic-http": "DoS",
    "ddos attack-loic-udp": "DoS",
    "dos": "DoS",
    "dos attacks-hulk": "DoS",
    "dos attacks-slowhttp": "DoS",
    "dos attacks-goldeneye": "DoS",
    "dos attacks-slowloris": "DoS",
    "ftp-bruteforce": "Exploits",
    "ssh-bruteforce": "Exploits",
    "brute force -web": "Exploits",
    "brute force -xss": "Exploits",
    "sql injection": "Exploits",
    "infilteration": "Exploits",
    "infiltration": "Exploits",
    "web attack": "Exploits",
    "web attacks": "Exploits",
    "heartbleed": "Exploits",
}


def cic_label_to_category(label: Any) -> str:
    text = str(label or "").strip().lower()
    if text in CIC_LABEL_TO_CAT:
        return CIC_LABEL_TO_CAT[text]
    for key, cat in CIC_LABEL_TO_CAT.items():
        if key in text:
            return cat
    if "benign" in text or text in {"", "nan"}:
        return "Normal"
    return "Exploits"


def ctu_label_to_category(label: Any) -> str:
    text = str(label or "").strip().lower()
    if "botnet" in text or "c&c" in text or "cnc" in text or "cc" == text:
        return "Backdoor"
    if "normal" in text or "background" in text or "legitimate" in text:
        return "Normal"
    if "malware" in text:
        return "Generic"
    return "Suspicious" if text else "Normal"


# ---------------------------------------------------------------------------
# CIC-IDS-2018 → ChronoFlux
# ---------------------------------------------------------------------------

def cic_row_to_features(row: pd.Series, cmap: dict[str, str]) -> dict:
    dur_us = _num(_get(row, cmap, "Flow Duration", "flow_duration", default=0))
    dur = dur_us / 1_000_000.0 if dur_us > 1000 else dur_us  # CIC often microseconds
    spkts = _num(_get(row, cmap, "Tot Fwd Pkts", "Total Fwd Packets", "tot_fwd_pkts", default=0))
    dpkts = _num(_get(row, cmap, "Tot Bwd Pkts", "Total Backward Packets", "tot_bwd_pkts", default=0))
    sbytes = _num(_get(row, cmap, "TotLen Fwd Pkts", "Total Length of Fwd Packets", "totlen_fwd_pkts", default=0))
    dbytes = _num(_get(row, cmap, "TotLen Bwd Pkts", "Total Length of Bwd Packets", "totlen_bwd_pkts", default=0))
    iat_mean = _num(_get(row, cmap, "Flow IAT Mean", "flow_iat_mean", default=0))
    iat_std = _num(_get(row, cmap, "Flow IAT Std", "flow_iat_std", default=0))
    iat_max = _num(_get(row, cmap, "Flow IAT Max", "flow_iat_max", default=0))
    # Convert microsecond IATs to ms-ish ChronoFlux style when huge
    if iat_mean > 10_000:
        iat_mean, iat_std, iat_max = iat_mean / 1000.0, iat_std / 1000.0, iat_max / 1000.0

    rate = _num(_get(row, cmap, "Flow Pkts/s", "Flow Packets/s", "flow_pkts_s", default=0))
    if rate <= 0 and dur > 0:
        rate = (spkts + dpkts) / max(dur, 1e-6)

    proto = _proto(_get(row, cmap, "Protocol", "protocol", default="tcp"))
    dport = _get(row, cmap, "Dst Port", "Destination Port", "dst_port", default=0)
    sport = _get(row, cmap, "Src Port", "Source Port", "src_port", default=0)
    syn = _num(_get(row, cmap, "SYN Flag Cnt", "SYN Flag Count", "syn_flag_cnt", default=0))
    ack = _num(_get(row, cmap, "ACK Flag Cnt", "ACK Flag Count", "ack_flag_cnt", default=0))
    fin = _num(_get(row, cmap, "FIN Flag Cnt", "FIN Flag Count", "fin_flag_cnt", default=0))
    rst = _num(_get(row, cmap, "RST Flag Cnt", "RST Flag Count", "rst_flag_cnt", default=0))
    psh = _num(_get(row, cmap, "PSH Flag Cnt", "PSH Flag Count", "psh_flag_cnt", default=0))
    urg = _num(_get(row, cmap, "URG Flag Cnt", "URG Flag Count", "urg_flag_cnt", default=0))
    swin = _num(_get(row, cmap, "Init Fwd Win Byts", "Init_Win_bytes_forward", "init_fwd_win_byts", default=0))
    dwin = _num(_get(row, cmap, "Init Bwd Win Byts", "Init_Win_bytes_backward", "init_bwd_win_byts", default=0))
    smean = _num(_get(row, cmap, "Fwd Pkt Len Mean", "Fwd Packet Length Mean", "fwd_pkt_len_mean", default=0))
    dmean = _num(_get(row, cmap, "Bwd Pkt Len Mean", "Bwd Packet Length Mean", "bwd_pkt_len_mean", default=0))
    fwd_iat = _num(_get(row, cmap, "Fwd IAT Mean", "fwd_iat_mean", default=iat_mean))
    bwd_iat = _num(_get(row, cmap, "Bwd IAT Mean", "bwd_iat_mean", default=iat_mean))
    retrans = _num(_get(row, cmap, "Fwd Act Data Pkts", default=0))  # weak proxy when true retrans absent

    state = "CON" if ack > 0 else ("INT" if syn > 0 else "INT")
    if rst > 0:
        state = "RST"
    elif fin > 0:
        state = "FIN"

    features = {
        **DEFAULT_FEATURES,
        "proto": proto,
        "service": _service_from_port(dport),
        "state": state,
        "dur": max(dur, 0.0),
        "spkts": spkts,
        "dpkts": dpkts,
        "sbytes": sbytes,
        "dbytes": dbytes,
        "rate": max(rate, 0.0),
        "sttl": 64.0,  # CICFlowMeter CSVs usually omit TTL; keep enterprise-default
        "dttl": 64.0,
        "sload": _num(_get(row, cmap, "Flow Byts/s", "Flow Bytes/s", default=0)),
        "dload": 0.0,
        "sloss": retrans,
        "dloss": 0.0,
        "sinpkt": fwd_iat if fwd_iat else iat_mean,
        "dinpkt": bwd_iat if bwd_iat else iat_mean,
        "sjit": iat_std,
        "djit": iat_std,
        "swin": max(swin, 0.0),
        "dwin": max(dwin, 0.0),
        "smean": smean,
        "dmean": dmean,
        "ct_srv_src": 1.0,
        "ct_state_ttl": 1.0,
        "ct_dst_ltm": 1.0,
        "ct_src_dport_ltm": 1.0,
        "ct_dst_sport_ltm": 1.0,
        "ct_dst_src_ltm": 1.0,
        "ct_src_ltm": 1.0,
        "ct_srv_dst": 1.0,
    }

    telemetry = {
        "packet_count": spkts + dpkts,
        "packet_rate": features["rate"],
        "unique_dst_ports": 1,
        "unique_src_ports": 1,
        "unique_dst_ips": 1,
        "syn_count": syn,
        "ack_count": ack,
        "rst_count": rst,
        "fin_count": fin,
        "psh_count": psh,
        "urg_count": urg,
        "icmp_count": 0,
        "outbound_bytes": sbytes,
        "inbound_bytes": dbytes,
        "source_ip": str(_get(row, cmap, "Src IP", "Source IP", "src_ip", default="")),
        "source_port": int(_num(sport)),
        "primary_destination": str(_get(row, cmap, "Dst IP", "Destination IP", "dst_ip", default="")),
        "destination_port": int(_num(dport)),
        "protocol": proto,
        "service": features["service"],
        "tcp_flags": [f for f, c in (("S", syn), ("A", ack), ("R", rst), ("F", fin), ("P", psh), ("U", urg)) if c],
        "dst_ports_sample": [int(_num(dport))] if _num(dport) else [],
        "window_seconds": max(dur, 1.0),
        "iat_mean": iat_mean,
        "iat_variance": iat_std ** 2,
        "iat_max": iat_max,
        "ttl_mean": 64.0,
        "ttl_variance": 0.0,
        "tcp_window_mean": (swin + dwin) / 2.0 if (swin or dwin) else 0.0,
        "retransmit_count": retrans,
        "fragment_flag_count": 0,
        "payload_size_mean": smean,
    }
    return features, telemetry


def convert_cic_ids2018(df: pd.DataFrame, max_rows: int | None = None) -> pd.DataFrame:
    cmap = _colmap(df)
    rows = []
    limit = len(df) if max_rows is None else min(len(df), max_rows)
    for _, row in df.head(limit).iterrows():
        features, telemetry = cic_row_to_features(row, cmap)
        label_raw = _get(row, cmap, "Label", "label", default="Benign")
        cat = cic_label_to_category(label_raw)
        out = {**features, "attack_cat": cat, "label": 0 if cat == "Normal" else 1}
        # stash telemetry as JSON-serializable extras for optional world-model builders
        out["_telemetry_syn"] = telemetry["syn_count"]
        out["_telemetry_ack"] = telemetry["ack_count"]
        out["_telemetry_rst"] = telemetry["rst_count"]
        out["_telemetry_fin"] = telemetry["fin_count"]
        out["_telemetry_psh"] = telemetry.get("psh_count", 0)
        out["_telemetry_urg"] = telemetry.get("urg_count", 0)
        out["_telemetry_iat_mean"] = telemetry["iat_mean"]
        out["_telemetry_iat_var"] = telemetry["iat_variance"]
        out["_telemetry_iat_max"] = telemetry["iat_max"]
        out["_source"] = "cicids2018"
        rows.append(out)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CTU-13 → ChronoFlux
# ---------------------------------------------------------------------------

def ctu_row_to_features(row: pd.Series, cmap: dict[str, str]) -> tuple[dict, dict]:
    dur = _num(_get(row, cmap, "Dur", "dur", default=0))
    proto = _proto(_get(row, cmap, "Proto", "proto", default="tcp"))
    tot_pkts = _num(_get(row, cmap, "TotPkts", "totpkts", "pkts", default=0))
    tot_bytes = _num(_get(row, cmap, "TotBytes", "totbytes", "bytes", default=0))
    src_bytes = _num(_get(row, cmap, "SrcBytes", "srcbytes", "sbytes", default=tot_bytes / 2))
    dst_bytes = max(tot_bytes - src_bytes, 0.0)
    # Approximate bidirectional packet split
    spkts = max(1.0, round(tot_pkts * 0.6)) if tot_pkts else 0.0
    dpkts = max(0.0, tot_pkts - spkts)
    sport = _get(row, cmap, "Sport", "sport", default=0)
    dport = _get(row, cmap, "Dport", "dport", default=0)
    state_raw = str(_get(row, cmap, "State", "state", default="INT"))
    state = "CON"
    upper = state_raw.upper()
    if "RST" in upper or "R" == upper:
        state = "RST"
    elif "FIN" in upper or "F" in upper:
        state = "FIN"
    elif "S" in upper and "A" not in upper:
        state = "INT"

    rate = (tot_pkts / max(dur, 1e-6)) if dur > 0 else tot_pkts
    iat_mean = (dur * 1000.0 / max(tot_pkts - 1, 1)) if tot_pkts > 1 else 0.0

    features = {
        **DEFAULT_FEATURES,
        "proto": proto,
        "service": _service_from_port(dport),
        "state": state,
        "dur": max(dur, 0.0),
        "spkts": spkts,
        "dpkts": dpkts,
        "sbytes": src_bytes,
        "dbytes": dst_bytes,
        "rate": max(rate, 0.0),
        "sttl": _num(_get(row, cmap, "sTtl", "sttl", default=64)),
        "dttl": _num(_get(row, cmap, "dTtl", "dttl", default=64)),
        "sload": (src_bytes * 8) / max(dur, 1e-6) if dur else 0.0,
        "dload": (dst_bytes * 8) / max(dur, 1e-6) if dur else 0.0,
        "sinpkt": iat_mean,
        "dinpkt": iat_mean,
        "sjit": iat_mean * 0.2,
        "djit": iat_mean * 0.2,
        "swin": _num(_get(row, cmap, "sWin", "swin", default=0)),
        "dwin": _num(_get(row, cmap, "dWin", "dwin", default=0)),
        "tcprtt": _num(_get(row, cmap, "TcpRtt", "tcprtt", default=0)),
        "synack": _num(_get(row, cmap, "SynAck", "synack", default=0)),
        "ackdat": _num(_get(row, cmap, "AckDat", "ackdat", default=0)),
        "smean": src_bytes / max(spkts, 1),
        "dmean": dst_bytes / max(dpkts, 1) if dpkts else 0.0,
        "ct_srv_src": 1.0,
        "ct_state_ttl": 1.0,
        "ct_dst_ltm": 1.0,
        "ct_src_dport_ltm": 1.0,
        "ct_dst_sport_ltm": 1.0,
        "ct_dst_src_ltm": 1.0,
        "ct_src_ltm": 1.0,
        "ct_srv_dst": 1.0,
    }

    syn = 1.0 if "S" in upper else 0.0
    ack = 1.0 if "A" in upper or state == "CON" else 0.0
    rst = 1.0 if state == "RST" else 0.0
    fin = 1.0 if state == "FIN" else 0.0

    telemetry = {
        "packet_count": tot_pkts,
        "packet_rate": features["rate"],
        "unique_dst_ports": 1,
        "unique_src_ports": 1,
        "unique_dst_ips": 1,
        "syn_count": syn,
        "ack_count": ack,
        "rst_count": rst,
        "fin_count": fin,
        "icmp_count": 1 if proto == "icmp" else 0,
        "outbound_bytes": src_bytes,
        "inbound_bytes": dst_bytes,
        "source_ip": str(_get(row, cmap, "SrcAddr", "srcaddr", default="")),
        "source_port": int(_num(sport)),
        "primary_destination": str(_get(row, cmap, "DstAddr", "dstaddr", default="")),
        "destination_port": int(_num(dport)),
        "protocol": proto,
        "service": features["service"],
        "tcp_flags": [f for f, c in (("S", syn), ("A", ack), ("R", rst), ("F", fin)) if c],
        "dst_ports_sample": [int(_num(dport))] if _num(dport) else [],
        "window_seconds": max(dur, 1.0),
        "iat_mean": iat_mean,
        "iat_variance": (iat_mean * 0.2) ** 2,
        "iat_max": iat_mean * 3,
        "ttl_mean": features["sttl"],
        "ttl_variance": abs(features["sttl"] - features["dttl"]),
        "tcp_window_mean": (features["swin"] + features["dwin"]) / 2.0,
        "retransmit_count": 0,
        "fragment_flag_count": 0,
        "payload_size_mean": features["smean"],
        "start_time": str(_get(row, cmap, "StartTime", "starttime", default="")),
    }
    return features, telemetry


def convert_ctu13(df: pd.DataFrame, max_rows: int | None = None) -> pd.DataFrame:
    cmap = _colmap(df)
    rows = []
    limit = len(df) if max_rows is None else min(len(df), max_rows)
    for _, row in df.head(limit).iterrows():
        features, telemetry = ctu_row_to_features(row, cmap)
        label_raw = _get(row, cmap, "Label", "label", default="Normal")
        cat = ctu_label_to_category(label_raw)
        out = {**features, "attack_cat": cat, "label": 0 if cat == "Normal" else 1}
        out["_telemetry_syn"] = telemetry["syn_count"]
        out["_telemetry_ack"] = telemetry["ack_count"]
        out["_telemetry_rst"] = telemetry["rst_count"]
        out["_telemetry_fin"] = telemetry["fin_count"]
        out["_telemetry_iat_mean"] = telemetry["iat_mean"]
        out["_telemetry_iat_var"] = telemetry["iat_variance"]
        out["_telemetry_iat_max"] = telemetry["iat_max"]
        out["_source"] = "ctu13"
        out["_start_time"] = telemetry.get("start_time", "")
        rows.append(out)
    return pd.DataFrame(rows)


def select_rows_for_analysis(
    df: pd.DataFrame,
    max_rows: int | None = None,
    mode: str = "stratified",
) -> tuple[pd.DataFrame, dict]:
    """Pick rows for offline analysis.

    mode:
      - all: every row in the CSV
      - stratified: balanced Label/attack_cat sample so BENIGN + attacks both appear
      - head: first N rows (legacy; can miss late attack rows in CIC files)
    """
    total = len(df)
    meta: dict = {"mode": mode, "total_rows": total, "selected_rows": total, "label_counts": {}}
    if total == 0:
        return df, meta

    cols = {str(c).strip().lower(): c for c in df.columns}
    label_col = cols.get("label") or cols.get("attack_cat")

    def _label_counts(frame: pd.DataFrame) -> dict:
        if label_col is None:
            return {}
        return {str(k): int(v) for k, v in frame[label_col].value_counts().items()}

    if mode == "all":
        out = df.copy()
        meta["mode"] = "all"
        meta["selected_rows"] = len(out)
        meta["label_counts"] = _label_counts(out)
        return out, meta

    if mode == "head":
        limit = total if max_rows is None else min(total, int(max_rows))
        out = df.iloc[:limit].copy()
        meta["selected_rows"] = len(out)
        meta["label_counts"] = _label_counts(out)
        return out, meta

    # stratified (default)
    if max_rows is None or int(max_rows) >= total:
        out = df.copy()
        meta["mode"] = "all"
        meta["selected_rows"] = len(out)
        meta["label_counts"] = _label_counts(out)
        return out, meta

    limit = min(total, int(max_rows))
    if label_col is None:
        out = df.sample(n=limit, random_state=42).reset_index(drop=True)
        meta["mode"] = "random"
        meta["selected_rows"] = len(out)
        return out, meta

    groups = list(df.groupby(label_col, sort=False))
    n_groups = max(1, len(groups))
    base = max(1, limit // n_groups)
    parts = []
    leftover = limit
    for _, g in groups:
        take = min(len(g), base)
        parts.append(g.sample(n=take, random_state=42) if take < len(g) else g)
        leftover -= take
    if leftover > 0:
        used_idx = set().union(*[set(p.index) for p in parts]) if parts else set()
        remain = df.loc[~df.index.isin(used_idx)]
        if len(remain) > 0:
            parts.append(remain.sample(n=min(leftover, len(remain)), random_state=42))
    out = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    if len(out) > limit:
        out = out.head(limit)
    meta["mode"] = "stratified"
    meta["selected_rows"] = len(out)
    meta["label_counts"] = _label_counts(out)
    return out, meta


def to_chronoflux(
    df: pd.DataFrame,
    max_rows: int | None = None,
    mode: str = "stratified",
) -> tuple[pd.DataFrame, str, dict]:
    """Auto-detect dataset type and convert to ChronoFlux columns."""
    kind = detect_dataset(df)
    selected, sample_meta = select_rows_for_analysis(df, max_rows=max_rows, mode=mode)
    if kind == "chronoflux":
        return selected, kind, sample_meta
    if kind == "cicids2018":
        return convert_cic_ids2018(selected, max_rows=None), kind, sample_meta
    if kind == "ctu13":
        return convert_ctu13(selected, max_rows=None), kind, sample_meta
    raise ValueError(
        "Unrecognized CSV schema. Expected ChronoFlux, CIC-IDS-2018 (CICFlowMeter), or CTU-13 (.binetflow) columns."
    )


def chronoflux_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Converted frame missing columns: {missing}")
    return df[FEATURE_COLUMNS].copy()
