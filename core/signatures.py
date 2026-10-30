"""Transparent behavioural signatures for the live gateway feed."""


def evaluate_signatures(features: dict, telemetry: dict | None = None) -> list[dict]:
    t = telemetry or {}
    alerts = []

    unique_ports = int(t.get("unique_dst_ports", features.get("ct_dst_ltm", 0)) or 0)
    syn_count = int(t.get("syn_count", 0) or 0)
    rst_count = int(t.get("rst_count", 0) or 0)
    icmp_count = int(t.get("icmp_count", 0) or 0)
    unique_dst_ips = int(t.get("unique_dst_ips", 0) or 0)
    outbound_bytes = float(t.get("outbound_bytes", features.get("sbytes", 0)) or 0)
    packet_rate = float(t.get("packet_rate", features.get("rate", 0)) or 0)

    def add(name, severity, mapped_stage, reason, score):
        alerts.append({
            "name": name,
            "severity": severity,
            "mapped_stage": mapped_stage,
            "reason": reason,
            "score": score,
        })

    if unique_ports >= 12:
        add(
            "PORT_FANOUT",
            "HIGH" if unique_ports >= 25 else "MEDIUM",
            "Reconnaissance",
            f"{unique_ports} destination ports were contacted in one window.",
            min(100, 45 + unique_ports * 2),
        )

    if syn_count >= 25:
        add(
            "SYN_BURST",
            "HIGH" if syn_count >= 80 else "MEDIUM",
            "Impact" if syn_count >= 80 else "Reconnaissance",
            f"{syn_count} TCP SYN packets were observed in the window.",
            min(100, 40 + syn_count * 0.7),
        )

    if rst_count >= 12:
        add(
            "RST_BURST",
            "MEDIUM",
            "Suspicious",
            f"{rst_count} TCP reset packets indicate repeated failed connections.",
            min(90, 40 + rst_count),
        )

    if icmp_count >= 120:
        add(
            "ICMP_RATE_SPIKE",
            "HIGH",
            "Impact",
            f"{icmp_count} ICMP packets were observed in one window.",
            min(100, 55 + icmp_count * 0.2),
        )

    if packet_rate >= 450:
        add(
            "PACKET_RATE_SPIKE",
            "HIGH",
            "Impact",
            f"Packet rate rose to {packet_rate:.1f} packets/second.",
            min(100, 60 + packet_rate / 20),
        )

    if outbound_bytes >= 2_500_000:
        add(
            "OUTBOUND_VOLUME_SPIKE",
            "HIGH",
            "Exfiltration",
            f"Outbound volume reached {outbound_bytes / 1_000_000:.2f} MB in one window.",
            min(100, 60 + outbound_bytes / 500_000),
        )

    if unique_dst_ips >= 20:
        add(
            "DESTINATION_SWEEP",
            "MEDIUM",
            "Reconnaissance",
            f"Traffic reached {unique_dst_ips} unique destination IPs in one window.",
            min(95, 45 + unique_dst_ips * 1.5),
        )

    return sorted(alerts, key=lambda x: x["score"], reverse=True)
