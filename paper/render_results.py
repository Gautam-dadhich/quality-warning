"""Regenerate the reporting tables from the archived benchmark and decision queues."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper"
R = ROOT / "results"
main = pd.read_csv(R / "main_results.csv")
test = main[main.split.eq("test")].set_index("method")
valid = main[main.split.eq("val")].set_index("method")
names = {"rule_low_share":"High low-rating share", "rule_share_change":"Rising low-rating share",
         "cusum (k0.05)":"CUSUM", "logreg [S]":"Logistic, S", "logreg [S+A]":"Logistic, S+A",
         "logreg [S+A+T]":"Logistic, S+A+T", "boosting [S]":"Boosting, S",
         "boosting [S+A]":"Boosting, S+A", "boosting [S+A+T]":"Boosting, S+A+T"}
def write(name, lines):
    if name in ("benchmark.tex", "queue.tex", "capacity.tex"):
        macro_name = Path(name).stem.title() + "Rows"
        lines = [chr(92) + "newcommand{" + chr(92) + macro_name + "}{%"] + lines + ["}"]
    (OUT / name).write_text("\n".join(lines) + "\n")
def line(values):
    return " & ".join(map(str, values)) + r" \\"
write("benchmark.tex", [line([name, f"{valid.loc[key,'pr_auc']:.3f}", f"{test.loc[key,'pr_auc']:.3f}",
        f"{test.loc[key,'p_at_cap50']:.3f}"]) for key, name in names.items()])
m = pd.read_csv(R / "queue_monthly.csv", parse_dates=["decision_date"])
timing = pd.read_csv(R / "warning_days.csv")
methods = ["CUSUM", "Volume", "Low prior share", "Boosting", "Coverage-adjusted"]
s = pd.read_csv(R / "queue_summary.csv").set_index(["method", "cap"])
write("queue.tex", [line([method] + [int(s.loc[(method,50), col]) for col in
        ["issued", "event","non_event","unknown"]] + [f"{100*s.loc[(method,50),'verified_yield']:.1f}"])
        for method in methods])
write("capacity.tex", [line([cap] + [f"{100*s.loc[(method,cap),'verified_yield']:.1f}" for method in
        ["CUSUM","Boosting","Coverage-adjusted"]]) for cap in (10,50,100,200)])
w = timing[timing.method.eq("Coverage-adjusted")].days
pairs = m[m.cap.eq(50)].pivot(index="decision_date", columns="method", values="event")
gap = (pairs["Coverage-adjusted"] - pairs["CUSUM"]).to_numpy()
# Paired circular three-month blocks, exploratory interval conditional on these fitted models.
rng = np.random.default_rng(42)
starts = rng.integers(len(gap), size=(5000, (len(gap)+2)//3))
indices = ((starts[:,:,None]+np.arange(3)) % len(gap)).reshape(5000,-1)[:,:len(gap)]
lo, hi = np.quantile(gap[indices].mean(axis=1), [.025,.975])
ca = s.loc[("Coverage-adjusted",50)]
b = s.loc[("Boosting",50)]
cu = s.loc[("CUSUM",50)]
def macro(name, value):
    return chr(92) + "newcommand{" + chr(92) + name + "}{" + str(value) + "}"
write("values.tex", [macro("Gain",f"{gap.mean():.1f}"),macro("GapLo",f"{lo:.1f}"),
    macro("GapHi",f"{hi:.1f}"), macro("QDays",int(w.median()) if w.median().is_integer() else w.median()),
    macro("QLow",f"{w.quantile(.25):g}"),macro("QHigh",f"{w.quantile(.75):g}"),
    macro("CAHits",int(ca.event)),macro("CAAlerts",int(ca.issued)),macro("CUHits",int(cu.event)),
    macro("GBHits",int(b.event)),macro("CAUnknown",int(ca.unknown)),
    macro("CAObserved",f"{100*ca.precision_observed:.1f}"),
    macro("CAPercent",f"{100*ca.verified_yield:.1f}"),
    macro("Wins",int((gap>0).sum())),macro("Months",len(gap))])

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.0), sharex=True)
for method, color in [("Coverage-adjusted","#00665e"),("Boosting","#8b4189"),("CUSUM","#555555")]:
    frame = m[(m.cap == 50) & (m.method == method)].sort_values("decision_date")
    for ax, column in zip(axes, ["event", "unknown"]):
        ax.plot(frame.decision_date, frame[column], ".-", color=color, label=method)
for ax, label in zip(axes, ["Confirmed spikes", "Unknown outcomes"]):
    ax.set_ylabel(label)
    ax.grid(axis="y", alpha=.2)
axes[0].legend(fontsize=8, ncol=3, frameon=False)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(OUT / "monthly_queue.pdf", bbox_inches="tight")
plt.close(fig)
