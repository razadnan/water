"""CLI: python run_investigation.py --zone "Zone B" | --scan | --ask "Why is Zone B consumption high?" [--json]"""
import argparse
import json
from agents.orchestrator import Orchestrator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zone")
    p.add_argument("--scan", action="store_true")
    p.add_argument("--ask")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    orch = Orchestrator()
    if a.ask:
        out = orch.ask(a.ask)
    elif a.scan or not a.zone:
        out = orch.scan_all()
    else:
        out = {"reports": [orch.investigate(a.zone)], "normal_zones": []}
    if a.json:
        print(json.dumps(out, indent=2, default=str))
        return
    print(f"LLM: {orch.llm.name}\n")
    for r in out["reports"]:
        if "error" in r:
            print(r["error"]); continue
        print(r["report_markdown"], "\n")
        print("Activity log:")
        for e in r["activity_log"]:
            print(f"  {e['time']}  {e['agent']:<30} {e['event']:<28} {e['detail'][:70]}")
        print("-" * 70)
    if out["normal_zones"]:
        print("Normal zones:", ", ".join(out["normal_zones"]))


if __name__ == "__main__":
    main()
