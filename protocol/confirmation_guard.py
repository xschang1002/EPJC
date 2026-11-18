"""Refuse confirmation work until a frozen, approved protocol is present.

This command is intentionally a guard only.  It does not create data, alter
the protocol, or infer approval from a draft file.  A future confirmation
runner should call it before opening any confirmation input or output path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.environment.read_text(encoding="utf-8"))
    failures: list[str] = []
    if config.get("confirmation_enabled") is not True:
        failures.append("confirmation_enabled is not true")
    if config.get("status") != "PRODUCTION_FROZEN":
        failures.append("protocol status is not PRODUCTION_FROZEN")
    if not config.get("frozen_protocol_sha256"):
        failures.append("frozen_protocol_sha256 is missing")
    if not config.get("human_freeze_approval"):
        failures.append("human_freeze_approval is missing")
    approval = config.get("confirmation_approval")
    if not approval:
        failures.append("confirmation_approval is missing")
    elif not approval.get("approval_text"):
        failures.append("confirmation_approval.approval_text is missing")
    protocol_path = config.get("confirmation_protocol_path") or config.get("frozen_protocol_path")
    if not protocol_path:
        failures.append("confirmation protocol path is missing")
    else:
        path = Path(protocol_path)
        if not path.is_absolute():
            path = args.environment.resolve().parent.parent / path
        if not path.exists():
            failures.append("confirmation protocol file is missing")
        else:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            registered = config.get("confirmation_protocol_sha256") or config.get("frozen_protocol_sha256")
            if actual != registered:
                failures.append("confirmation protocol hash does not match registration")
    result = {
        "allowed": not failures,
        "decision": "ALLOW_CONFIRMATION" if not failures else "REFUSE_CONFIRMATION",
        "read_only": True,
        "environment": str(args.environment.resolve()),
        "reasons": failures,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
