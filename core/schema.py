"""Shared network feature schema for ChronoFlux."""

NUMERIC_FEATURES = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate",
    "sttl", "dttl", "sload", "dload", "sloss", "dloss",
    "sinpkt", "dinpkt", "sjit", "djit", "swin", "dwin",
    "tcprtt", "synack", "ackdat", "smean", "dmean",
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "ct_src_ltm", "ct_srv_dst",
]

CATEGORICAL_FEATURES = ["proto", "service", "state"]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET_COLUMN = "label"
ATTACK_CATEGORY_COLUMN = "attack_cat"

DEFAULT_FEATURES = {
    **{name: 0.0 for name in NUMERIC_FEATURES},
    "proto": "other",
    "service": "-",
    "state": "INT",
}
