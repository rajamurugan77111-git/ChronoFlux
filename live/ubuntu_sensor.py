"""ChronoFlux Ubuntu gateway sensor.

Captures only packet/flow metadata required by the prototype. It does not
inspect application payload contents.
"""
import argparse
import time
import uuid
from datetime import datetime, timezone

import requests
from scapy.all import sniff, IP, TCP, UDP, ICMP, Ether, ARP

FEATURE_COLUMNS = [
    "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes", "rate",
    "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit",
    "swin", "dwin", "tcprtt", "synack", "ackdat", "smean", "dmean", "ct_srv_src",
    "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
    "ct_dst_src_ltm", "ct_src_ltm", "ct_srv_dst",
]


def avg(values):
    return sum(values) / len(values) if values else 0.0


def new_window():
    return {
        "first": None,
        "last": None,
        "times": [],
        "spkts": 0,
        "dpkts": 0,
        "sbytes": 0,
        "dbytes": 0,
        "src_ip": None,
        "dst_ip": None,
        "proto": "other",
        "service": "-",
        "state": "INT",
        "src_ttls": [],
        "dst_ttls": [],
        "src_wins": [],
        "dst_wins": [],
        "dst_ports": set(),
        "src_ports": set(),
        "dst_ips": set(),
        "syn_count": 0,
        "ack_count": 0,
        "rst_count": 0,
        "fin_count": 0,
        "icmp_count": 0,
        "retransmit_count": 0,
        "fragment_flag_count": 0,
        "payloads": [],
        "seen_seq": set(),
        "primary_src_port": None,
        "primary_dst_port": None,
        "tcp_flags": set(),
        "services_seen": set(),
        "src_mac": None,
        "dst_mac": None,
        "arp_count": 0,
    }


def process(pkt, w):
    now = time.time()
    if w["first"] is None:
        w["first"] = now
    w["last"] = now
    w["times"].append(now)

    if Ether in pkt:
        if w["src_mac"] is None:
            w["src_mac"] = str(pkt[Ether].src)
            w["dst_mac"] = str(pkt[Ether].dst)

    if ARP in pkt:
        w["arp_count"] += 1
        if w["src_ip"] is None:
            w["src_ip"] = str(pkt[ARP].psrc)
            w["dst_ip"] = str(pkt[ARP].pdst)
        if w["src_mac"] is None:
            w["src_mac"] = str(pkt[ARP].hwsrc)
        return

    if IP not in pkt:
        return

    src = pkt[IP].src
    dst = pkt[IP].dst
    w["dst_ips"].add(dst)

    if w["src_ip"] is None:
        w["src_ip"], w["dst_ip"] = src, dst

    forward = src == w["src_ip"]
    ip_flags = int(getattr(pkt[IP], "flags", 0) or 0)
    frag = int(getattr(pkt[IP], "frag", 0) or 0)
    if (ip_flags & 0x1) or frag > 0:
        w["fragment_flag_count"] += 1

    payload_len = max(0, len(pkt) - (pkt[IP].ihl * 4 if hasattr(pkt[IP], "ihl") else 20))
    w["payloads"].append(payload_len)

    if forward:
        w["spkts"] += 1
        w["sbytes"] += len(pkt)
        w["src_ttls"].append(int(pkt[IP].ttl))
    else:
        w["dpkts"] += 1
        w["dbytes"] += len(pkt)
        w["dst_ttls"].append(int(pkt[IP].ttl))

    if TCP in pkt:
        w["proto"] = "tcp"
        dport = int(pkt[TCP].dport)
        sport = int(pkt[TCP].sport)
        w["dst_ports"].add(dport)
        w["src_ports"].add(sport)
        if w["primary_src_port"] is None:
            w["primary_src_port"] = sport
            w["primary_dst_port"] = dport
        (w["src_wins"] if forward else w["dst_wins"]).append(int(pkt[TCP].window))

        service_map = {
            20: "ftp-data", 21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp",
            53: "dns", 67: "dhcp", 68: "dhcp", 80: "http", 110: "pop3",
            143: "imap", 443: "https", 445: "smb", 3306: "mysql",
            3389: "rdp", 5432: "postgresql", 8080: "http-alt",
        }
        service = service_map.get(dport) or service_map.get(sport)
        if service:
            w["service"] = service
            w["services_seen"].add(service)

        w["tcp_flags"].add(str(pkt[TCP].flags))
        flags = int(pkt[TCP].flags)
        seq = int(getattr(pkt[TCP], "seq", 0) or 0)
        seq_key = (src, dst, sport, dport, seq)
        if seq_key in w["seen_seq"]:
            w["retransmit_count"] += 1
        else:
            w["seen_seq"].add(seq_key)
        if flags & 0x02:
            w["syn_count"] += 1
        if flags & 0x10:
            w["ack_count"] += 1
        if flags & 0x04:
            w["rst_count"] += 1
            w["state"] = "RST"
        elif flags & 0x01:
            w["fin_count"] += 1
            w["state"] = "FIN"
        else:
            w["state"] = "CON"

    elif UDP in pkt:
        w["proto"] = "udp"
        dport = int(pkt[UDP].dport)
        sport = int(pkt[UDP].sport)
        w["dst_ports"].add(dport)
        w["src_ports"].add(sport)
        if w["primary_src_port"] is None:
            w["primary_src_port"] = sport
            w["primary_dst_port"] = dport
        if dport == 53 or sport == 53:
            w["service"] = "dns"
            w["services_seen"].add("dns")

    elif ICMP in pkt:
        w["proto"] = "icmp"
        w["icmp_count"] += 1


def make_payload(w, window_seconds):
    dur = 0.0
    if w["first"] is not None and w["last"] is not None:
        dur = max(w["last"] - w["first"], 1e-6)

    total_packets = w["spkts"] + w["dpkts"]
    diffs = [(w["times"][i] - w["times"][i - 1]) * 1000 for i in range(1, len(w["times"]))]
    iat = avg(diffs)
    iat_var = (sum((x - iat) ** 2 for x in diffs) / len(diffs)) if len(diffs) > 1 else 0.0
    iat_max = max(diffs) if diffs else 0.0
    ttl_all = w["src_ttls"] + w["dst_ttls"]
    ttl_mean = avg(ttl_all)
    ttl_var = (sum((x - ttl_mean) ** 2 for x in ttl_all) / len(ttl_all)) if len(ttl_all) > 1 else 0.0
    payload_mean = avg(w["payloads"])

    features = {
        "dur": dur,
        "proto": w["proto"],
        "service": w["service"],
        "state": w["state"],
        "spkts": w["spkts"],
        "dpkts": w["dpkts"],
        "sbytes": w["sbytes"],
        "dbytes": w["dbytes"],
        "rate": total_packets / max(window_seconds, 1e-6),
        "sttl": round(avg(w["src_ttls"])),
        "dttl": round(avg(w["dst_ttls"])),
        "sload": (w["sbytes"] * 8) / max(dur, 1e-6),
        "dload": (w["dbytes"] * 8) / max(dur, 1e-6),
        "sloss": w["retransmit_count"],
        "dloss": 0,
        "sinpkt": iat,
        "dinpkt": iat,
        "sjit": iat_var,
        "djit": iat_var,
        "swin": round(avg(w["src_wins"])),
        "dwin": round(avg(w["dst_wins"])),
        "tcprtt": 0.0,
        "synack": 0.0,
        "ackdat": 0.0,
        "smean": payload_mean if payload_mean else (w["sbytes"] / w["spkts"] if w["spkts"] else 0.0),
        "dmean": w["dbytes"] / w["dpkts"] if w["dpkts"] else 0.0,
        "ct_srv_src": len(w["dst_ports"]),
        "ct_state_ttl": 1 if total_packets else 0,
        "ct_dst_ltm": len(w["dst_ports"]),
        "ct_src_dport_ltm": len(w["dst_ports"]),
        "ct_dst_sport_ltm": len(w["src_ports"]),
        "ct_dst_src_ltm": len(w["dst_ips"]),
        "ct_src_ltm": 1 if total_packets else 0,
        "ct_srv_dst": len(w["dst_ports"]),
    }

    telemetry = {
        "packet_count": total_packets,
        "packet_rate": round(features["rate"], 3),
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
        "services_seen": sorted(w["services_seen"]),
        "dst_ports_sample": sorted(w["dst_ports"])[:20],
        "destination_ips_sample": sorted(w["dst_ips"])[:10],
        "source_mac": w["src_mac"],
        "destination_mac": w["dst_mac"],
        "arp_count": w["arp_count"],
        "window_seconds": window_seconds,
        "iat_mean": iat,
        "iat_variance": iat_var,
        "iat_max": iat_max,
        "ttl_mean": ttl_mean,
        "ttl_variance": ttl_var,
        "tcp_window_mean": avg(w["src_wins"] + w["dst_wins"]),
        "retransmit_count": w["retransmit_count"],
        "fragment_flag_count": w["fragment_flag_count"],
        "payload_size_mean": payload_mean,
    }

    return {k: features[k] for k in FEATURE_COLUMNS}, telemetry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", default="enp0s8")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--server", required=True, help="Example: http://192.168.56.1:8000")
    parser.add_argument("--sensor-id", default="ubuntu-gateway")
    args = parser.parse_args()

    endpoint = args.server.rstrip("/") + "/api/sensor/live"
    session_id = f"{args.sensor_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    print("\nChronoFlux Ubuntu Gateway Sensor")
    print("Capture interface :", args.interface)
    print("Window            :", args.window, "seconds")
    print("Backend           :", endpoint)
    print("Recording session :", session_id)
    print("Press Ctrl+C to stop.\n")

    while True:
        window = new_window()
        sniff(iface=args.interface, timeout=args.window, store=False, prn=lambda p: process(p, window))
        features, telemetry = make_payload(window, args.window)
        payload = {
            "sensor_id": args.sensor_id,
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "features": features,
            "telemetry": telemetry,
        }

        try:
            response = requests.post(endpoint, json=payload, timeout=8)
            if response.ok:
                result = response.json().get("prediction", {})
                forecast = result.get("forecast", {})
                print(
                    f"HTTP 200 | packets={telemetry['packet_count']} | "
                    f"state={result.get('current_stage')} | risk={result.get('risk_level')} | "
                    f"next={forecast.get('next_stage')}"
                )
            else:
                print("HTTP", response.status_code, response.text[:400])
        except Exception as exc:
            print("Send failed:", exc)


if __name__ == "__main__":
    main()
