"""Behaviour interpretation and MITRE-style context for ChronoFlux.

Important: the benchmark attack category is kept separate from the ChronoFlux
behaviour stage. MITRE mappings here are analyst interpretations used to make
alerts understandable; they are not claimed as dataset ground-truth labels.
"""

CATEGORY_TO_STAGE = {
    "Normal": "Normal",
    "Reconnaissance": "Reconnaissance",
    "Fuzzers": "Reconnaissance",
    "Analysis": "Reconnaissance",
    "Exploits": "Initial Access",
    "Generic": "Initial Access",
    "Shellcode": "Execution",
    "Backdoor": "Persistence",
    "Worms": "Lateral Movement",
    "DoS": "Impact",
    "Suspicious": "Suspicious",
    "Initial Access": "Initial Access",
    "Execution": "Execution",
    "Persistence": "Persistence",
    "Lateral Movement": "Lateral Movement",
    "Command & Control": "Command & Control",
    "Exfiltration": "Exfiltration",
    "Impact": "Impact",
}

STAGE_CONTEXT = {
    "Normal": {
        "interpretation": "Traffic is within the currently observed baseline.",
        "mitre_tactics": [],
    },
    "Suspicious": {
        "interpretation": "Unclassified behaviour deviates from the normal baseline and needs observation.",
        "mitre_tactics": [],
    },
    "Reconnaissance": {
        "interpretation": "Probe, scan, discovery or target-enumeration-like behaviour.",
        "mitre_tactics": ["Reconnaissance", "Discovery"],
    },
    "Initial Access": {
        "interpretation": "Exploit or access-attempt-like behaviour targeting an exposed service.",
        "mitre_tactics": ["Initial Access"],
    },
    "Execution": {
        "interpretation": "Execution-like behaviour associated with running attacker-controlled code.",
        "mitre_tactics": ["Execution"],
    },
    "Persistence": {
        "interpretation": "Backdoor or persistence-like behaviour that could maintain access.",
        "mitre_tactics": ["Persistence"],
    },
    "Lateral Movement": {
        "interpretation": "Propagation or movement-like behaviour across hosts or services.",
        "mitre_tactics": ["Lateral Movement"],
    },
    "Command & Control": {
        "interpretation": "Beaconing or command-channel-like communication behaviour.",
        "mitre_tactics": ["Command and Control"],
    },
    "Exfiltration": {
        "interpretation": "Unusually large or suspicious outbound transfer behaviour.",
        "mitre_tactics": ["Exfiltration"],
    },
    "Impact": {
        "interpretation": "Availability or service-impact-like behaviour such as denial-of-service activity.",
        "mitre_tactics": ["Impact"],
    },
}


def category_to_stage(category: str | None) -> str:
    if not category:
        return "Normal"
    return CATEGORY_TO_STAGE.get(str(category), str(category))


def context_for(stage: str, category: str | None = None) -> dict:
    ctx = STAGE_CONTEXT.get(stage, {
        "interpretation": f"Observed behaviour is classified as {stage}.",
        "mitre_tactics": [],
    })
    return {
        "attack_category": category,
        "behaviour_stage": stage,
        "behaviour_interpretation": ctx["interpretation"],
        "mitre_tactics": list(ctx["mitre_tactics"]),
        "mapping_note": "Behaviour/MITRE mapping is an inferred analyst interpretation, not UNSW-NB15 stage ground truth.",
    }
