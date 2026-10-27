"""AgenCALE coordinator: five role-specific agents and an auditable privacy gate.

Agents are deterministic software components with fixed inputs and outputs (no LLM is used).
  ThreatAgent      : builds the threat model from schema metadata (identifiers, quasi-identifiers,
                     protected attributes, free-text fields, release history).
  EncodingAgent    : proposes encoding plans on an escalation ladder (least to most transformative).
  AdversaryAgent   : runs the attack suite A1-A4 against a candidate release (in this study the suite is
                     executed by privenc_experiment.run and its outputs are read here) and, inside CALE,
                     the internal adversary A_int that drives scrubbing.
  UtilityAgent     : measures structured and text utility retention U.
  GovernanceAgent  : computes EPGI, applies the gate, and writes the evidence log.
The Coordinator iterates EncodingAgent -> AdversaryAgent -> UtilityAgent -> GovernanceAgent until a plan
passes the gate or the ladder is exhausted (release blocked).
"""
import hashlib, json, sys
import numpy as np

RISKS = ["reid", "attr_disc", "mia_adv", "linkability", "text_dx_adv", "text_loc_adv"]
import os, yaml
GATE = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")))["gate"]


class ThreatAgent:
    def run(self, schema):
        return dict(direct_identifiers=schema["direct"], quasi_identifiers=schema["qi"],
                    protected=schema["protected"], text_fields=schema["text"],
                    threats=["A1 linkage re-identification", "A1 attribute disclosure", "A2 membership inference",
                             "A3 cross-release linkability", "A4 contextual inference (diagnosis)",
                             "A4 contextual inference (exact ZIP)"])


class EncodingAgent:
    LADDER = ["Redaction", "Generalization (k=5)", "CALE (proposed)", "CALE-S (proposed)"]

    def propose(self, attempt):
        return self.LADDER[attempt] if attempt < len(self.LADDER) else None


class AdversaryAgent:
    def __init__(self, results):
        self.results = results

    def attack(self, plan):
        r = self.results[plan]
        out = {k: float(r[k]) for k in ("reid", "attr_disc", "linkability", "text_dx_adv", "text_loc_adv")}
        out["mia_adv"] = float(np.clip(2 * (r["mia_auc"] - 0.5), 0, 1))
        return out


class UtilityAgent:
    def __init__(self, results):
        self.raw = results["Raw (no encoding)"]
        self.results = results

    def measure(self, plan):
        r = self.results[plan]
        us = float(np.clip((r["utility_auc"] - 0.5) / (self.raw["utility_auc"] - 0.5), 0, 1))
        ut = float(np.clip((r["text_util"] - 0.5) / (self.raw["text_util"] - 0.5), 0, 1))
        return dict(u_struct=us, u_text=ut, U=(us + ut) / 2)


class GovernanceAgent:
    def __init__(self, gate=GATE, weights=None):
        self.gate, self.w = gate, np.ones(len(RISKS)) / len(RISKS) if weights is None else np.asarray(weights) / np.sum(weights)

    def decide(self, plan, risks, util, extra):
        R = float(np.dot(self.w, [risks[k] for k in RISKS]))
        P, U = 1 - R, util["U"]
        epgi = 1 - 2 * P * U / (P + U)
        checks = dict(max_threat=max(risks.values()) <= self.gate["max_threat"], epgi=epgi <= self.gate["max_epgi"],
                      utility=U >= self.gate["min_utility"])
        log = dict(plan=plan, residual_risk=R, per_threat=risks, utility=util, epgi=epgi, gate=self.gate,
                   checks=checks, passed=all(checks.values()), weights=self.w.tolist(),
                   scrub_list=extra.get(plan + "|scrub"), scrub_history=extra.get(plan + "|scrub_info"))
        log["log_sha256"] = hashlib.sha256(json.dumps(log, sort_keys=True, default=str).encode()).hexdigest()
        return log


def coordinate(results, extra, schema):
    threat = ThreatAgent().run(schema)
    enc, adv, uti, gov = EncodingAgent(), AdversaryAgent(results), UtilityAgent(results), GovernanceAgent()
    trace, attempt = [], 0
    while True:
        plan = enc.propose(attempt)
        if plan is None:
            return dict(threat_model=threat, trace=trace, released=None)
        log = gov.decide(plan, adv.attack(plan), uti.measure(plan), extra)
        trace.append(log)
        if log["passed"]:
            return dict(threat_model=threat, trace=trace, released=plan)
        attempt += 1


def gate_all(results, extra):
    adv, uti, gov = AdversaryAgent(results), UtilityAgent(results), GovernanceAgent()
    return {m: gov.decide(m, adv.attack(m), uti.measure(m), extra) for m in results}


SCHEMA = dict(direct=["name", "account number"], qi=["age", "sex", "ZIP", "occupation"],
              protected=["diagnosis", "exact ZIP"], text=["clinical note"])

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "results_v2"
    d = json.load(open(f"{src}/raw_results.json"))
    out = {}
    for seed, res in d["results"].items():
        ex = d["extra"][seed]
        dec = coordinate(res, ex, SCHEMA)
        allg = gate_all(res, ex)
        out[seed] = dict(released=dec["released"], ladder=[(t["plan"], t["passed"], round(t["epgi"], 3),
                                                            {k: v for k, v in t["checks"].items()}) for t in dec["trace"]],
                         gate={m: dict(passed=g["passed"], checks=g["checks"], epgi=g["epgi"],
                                       max_threat=max(g["per_threat"].values())) for m, g in allg.items()},
                         evidence_example=dec["trace"][-1])
        print(seed, "released:", dec["released"])
        for t in out[seed]["ladder"]:
            print("   ", t)
    json.dump(out, open(f"{src}/agencale_decisions.json", "w"), indent=1, default=str)
