#!/usr/bin/env python3
"""
Weight-matrix analysis for three task-specific effective DQN checkpoints.

Usage:
python analyze_weight_matrices.py \
  --task1 Task1_weight.pth --task2 Task2_weight.pth --task3 Task3_weight.pth \
  --out chapter4_weight_analysis

Outputs:
- Shannon marginal entropy (fixed layer-wise bins, plus 32/64/128-bin sensitivity)
- row, column, flattened, and 2D matrix autocorrelation
- singular values, rank/stable/effective-rank metrics, shuffled controls
- task-to-task matrix correlations and Frobenius distances
- plots and CSV files

Fisher information is intentionally not computed: checkpoints alone do not
provide the state-transition samples and per-sample gradients required for an
empirical Fisher estimate.
"""
from __future__ import annotations
import argparse, json, math, re, sys
from pathlib import Path
from typing import Mapping
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

TASKS = ("Task 1", "Task 2", "Task 3")
SENSITIVITY_BINS = (32, 64, 128)

def parse_args():
    p = argparse.ArgumentParser(description="Analyze three DQN weight checkpoints.")
    p.add_argument("--task1", type=Path, default=Path("MC1_3624.pth"))
    p.add_argument("--task2", type=Path, default=Path("Mc2_3624.pth"))
    p.add_argument("--task3", type=Path, default=Path("MC3_3624.pth"))
    p.add_argument("--out", type=Path, default=Path("chapter4_weight_analysis"))
    p.add_argument("--bins", type=int, default=64, help="Main entropy histogram bins.")
    p.add_argument("--max-lag", type=int, default=25)
    p.add_argument("--controls", type=int, default=100, help="Shuffled controls per matrix.")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--no-heatmaps", action="store_true")
    return p.parse_args()

def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")

def extract_state_dict(obj):
    if isinstance(obj, Mapping):
        if obj and all(torch.is_tensor(v) for v in obj.values()):
            return obj
        for key in ("state_dict", "model_state_dict", "q_network_state_dict"):
            cand = obj.get(key)
            if isinstance(cand, Mapping) and cand and all(torch.is_tensor(v) for v in cand.values()):
                return cand
    raise ValueError("Checkpoint does not contain a tensor-only state_dict.")

def load_weights(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    sd = extract_state_dict(torch_load(path))
    mats = {
        name: t.detach().cpu().numpy().astype(np.float64)
        for name, t in sd.items()
        if torch.is_tensor(t) and t.ndim == 2 and name.endswith("weight")
    }
    if not mats:
        raise ValueError(f"No 2D *.weight matrices found in {path}. Keys: {list(sd)}")
    return mats

def layer_key(name):
    nums = re.findall(r"\d+", name)
    return (int(nums[0]) if nums else 10**9, name)

def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")

def finite(x):
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if len(x) == 0:
        raise ValueError("No finite values.")
    return x

def common_edges(mats, bins):
    v = np.concatenate([finite(m) for m in mats])
    lo, hi = float(v.min()), float(v.max())
    if math.isclose(lo, hi):
        pad = max(1.0, abs(lo)*.05); lo -= pad; hi += pad
    pad = max((hi-lo)*1e-12, 1e-12)
    return np.linspace(lo-pad, hi+pad, bins+1)

def entropy(m, edges):
    counts, _ = np.histogram(finite(m), bins=edges)
    p = counts[counts>0] / counts.sum()
    h = float(-(p*np.log2(p)).sum())
    return h, h/math.log2(len(edges)-1)

def flat_acf(m, max_lag):
    x = finite(m); x = x-x.mean(); den = float(np.dot(x,x))
    L = min(max_lag, len(x)-1)
    lags = np.arange(L+1); acf = np.full(L+1, np.nan)
    if den <= np.finfo(float).eps: return lags, acf
    for k in lags: acf[k] = np.dot(x[:len(x)-k], x[k:]) / den
    return lags, acf

def acf2d(m, max_lag):
    w = np.asarray(m, dtype=float)
    if not np.isfinite(w).all(): raise ValueError("Non-finite weight.")
    x = w-w.mean(); var = float(np.mean(x*x))
    L = min(max_lag, w.shape[0]-1, w.shape[1]-1)
    lags = np.arange(-L, L+1); out = np.full((2*L+1, 2*L+1), np.nan)
    if var <= np.finfo(float).eps: return lags, out
    R, C = w.shape
    for ii, dr in enumerate(lags):
        for jj, dc in enumerate(lags):
            r0a, r0b = max(0,-dr), min(R,R-dr)
            c0a, c0b = max(0,-dc), min(C,C-dc)
            r1a, r1b = max(0,dr), min(R,R+dr)
            c1a, c1b = max(0,dc), min(C,C+dc)
            a = x[r0a:r0b, c0a:c0b]; b = x[r1a:r1b, c1a:c1b]
            if a.size: out[ii,jj] = np.mean(a*b)/var
    return lags, out

def svd_metrics(m):
    s = np.linalg.svd(m, compute_uv=False).astype(float)
    e = s*s; total = float(e.sum()); s0, sl = float(s[0]), float(s[-1])
    if total > 0:
        p = e/total; p = p[p>0]
        erank = float(np.exp(-(p*np.log(p)).sum()))
        srank = float(total/(s0*s0)) if s0>0 else np.nan
    else: erank=srank=np.nan
    frac = lambda k: float(e[:min(k,len(e))].sum()/total) if total>0 else np.nan
    return s, {
        "spectral_norm":s0, "smallest_singular_value":sl,
        "condition_number":s0/sl if sl>np.finfo(float).eps else np.inf,
        "frobenius_norm":float(np.sqrt(total)), "stable_rank":srank,
        "effective_rank":erank, "top_1_energy_fraction":frac(1),
        "top_5_energy_fraction":frac(5), "top_10_energy_fraction":frac(10)
    }

def shuffled_controls(m, n, rng):
    flat = finite(m); nsv = min(m.shape); out = np.empty((n,nsv))
    for i in range(n):
        out[i] = np.linalg.svd(rng.permutation(flat).reshape(m.shape), compute_uv=False)
    return out

def corr(a,b):
    x,y=finite(a),finite(b)
    if np.std(x)<=np.finfo(float).eps or np.std(y)<=np.finfo(float).eps: return np.nan
    return float(np.corrcoef(x,y)[0,1])

def ndist(a,b):
    n=float(np.linalg.norm(a-b,'fro')); d=.5*(float(np.linalg.norm(a,'fro'))+float(np.linalg.norm(b,'fro')))
    return n/d if d>np.finfo(float).eps else np.nan

def rdiff(a,b):
    d=float(np.linalg.norm(a,'fro'))
    return float(np.linalg.norm(a-b,'fro'))/d if d>np.finfo(float).eps else np.nan

def savefig(path):
    plt.tight_layout()
    plt.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

def write_readme(out, args, layers):
    (out/"README.md").write_text(f"""# Chapter 4 weight-matrix analysis

## What is included
- Histogram-based Shannon marginal entropy of effective weight values.
- Entropy sensitivity for 32, 64, and 128 common bins per layer.
- Row, column, flattened, and 2D matrix autocorrelation.
- Singular-value spectra, shuffled matrix controls, stable rank, effective rank,
  and top-k spectral-energy fractions.
- Pairwise correlation and normalized Frobenius distance between the three
  task-specific effective matrices.

## What is not included
Fisher information cannot be calculated from saved weights alone. An empirical
Fisher calculation requires a fixed dataset of state-transition samples and
per-sample gradients of a specified TD loss with respect to the parameters.

## Interpretation limits
- These checkpoints are final snapshots. They do not show how statistics evolve
  through training.
- Shuffled controls preserve the marginal weight distribution but destroy index
  organization. They are an exploratory random-control comparison, not a full
  Marchenko--Pastur fit.
- Matrix-index autocorrelation is a statistical descriptor. Hidden-neuron
  permutation invariance means it is not direct physical spatial correlation.

## Layers
{chr(10).join("- "+x for x in layers)}
""", encoding="utf-8")

def main():
    args = parse_args()
    if args.bins < 2 or args.controls < 1 or args.max_lag < 0:
        raise ValueError("Invalid bins, controls, or max-lag.")
    out=args.out; plots=out/"plots"; data=out/"data"
    plots.mkdir(parents=True,exist_ok=True); data.mkdir(parents=True,exist_ok=True)

    paths={"Task 1":args.task1,"Task 2":args.task2,"Task 3":args.task3}
    ckpt={t:load_weights(p) for t,p in paths.items()}
    common=set.intersection(*(set(x) for x in ckpt.values()))
    if not common: raise ValueError("No common 2D layers across checkpoints.")
    layers=sorted(common,key=layer_key); rng=np.random.default_rng(args.seed)

    ent_rows=[]; sens_rows=[]; ac_rows=[]; spec_rows=[]; srows=[]; compare_rows=[]; edge_rows=[]

    for layer in layers:
        mats={t:ckpt[t][layer] for t in TASKS}
        edges=common_edges(mats.values(), args.bins)
        for i,e in enumerate(edges): edge_rows.append({"layer":layer,"bins":args.bins,"edge_index":i,"edge_value":e})

        # Weight distribution
        plt.figure(figsize=(8,5))
        for t,m in mats.items():
            plt.hist(finite(m),bins=edges,density=True,histtype="step",linewidth=1.8,label=t)
        plt.xlabel("Effective weight value"); plt.ylabel("Probability density")
        plt.title(f"Weight-value distribution: {layer}"); plt.legend()
        savefig(plots/f"weight_distribution_{safe_name(layer)}")

        singular_by_task={}; control_by_task={}
        ac_layer=[]

        for t,m in mats.items():
            vals=finite(m); H,Hn=entropy(m,edges)
            ent_rows.append({"task":t,"layer":layer,"rows":m.shape[0],"columns":m.shape[1],
                             "number_of_weights":m.size,"weight_mean":float(vals.mean()),
                             "weight_std":float(vals.std()),"weight_min":float(vals.min()),
                             "weight_max":float(vals.max()),"entropy_bins":args.bins,
                             "entropy_bits":H,"normalized_entropy":Hn})
            for B in SENSITIVITY_BINS:
                h,hn=entropy(m,common_edges(mats.values(),B))
                sens_rows.append({"task":t,"layer":layer,"bins":B,"entropy_bits":h,"normalized_entropy":hn})

            d_lags,A=acf2d(m,args.max_lag); zi=int(np.where(d_lags==0)[0][0])
            fl,fa=flat_acf(m,args.max_lag)
            for lag in d_lags[d_lags>=0]:
                ix=int(np.where(d_lags==lag)[0][0])
                ac_layer.append({"task":t,"layer":layer,"lag":int(lag),
                                 "row_direction_acf":A[ix,zi],"column_direction_acf":A[zi,ix],
                                 "flattened_acf":fa[int(lag)] if int(lag)<len(fa) else np.nan})
            if not args.no_heatmaps:
                plt.figure(figsize=(6.5,5.5))
                im=plt.imshow(A,origin="lower",aspect="equal",extent=[d_lags[0],d_lags[-1],d_lags[0],d_lags[-1]])
                plt.colorbar(im,label="Normalized autocorrelation")
                plt.xlabel(r"Column lag $\Delta c$"); plt.ylabel(r"Row lag $\Delta r$")
                plt.title(f"2D weight autocorrelation: {t}, {layer}")
                savefig(plots/f"acf2d_{safe_name(t)}_{safe_name(layer)}")

            s,metrics=svd_metrics(m); singular_by_task[t]=s
            spec_rows.append({"task":t,"layer":layer,"rows":m.shape[0],"columns":m.shape[1],**metrics})
            for rank,v in enumerate(s,1):
                srows.append({"task":t,"layer":layer,"series_type":"learned","rank":rank,
                              "singular_value":v,"normalized_singular_value":v/s[0] if s[0]>0 else np.nan,
                              "control_std":np.nan})
            ctrl=shuffled_controls(m,args.controls,rng); mu=ctrl.mean(0); sd=ctrl.std(0,ddof=1) if args.controls>1 else np.zeros_like(mu)
            control_by_task[t]=(mu,sd)
            for rank,(v,e) in enumerate(zip(mu,sd),1):
                srows.append({"task":t,"layer":layer,"series_type":"shuffled_control_mean","rank":rank,
                              "singular_value":v,"normalized_singular_value":v/mu[0] if mu[0]>0 else np.nan,
                              "control_std":e})

        ac_rows.extend(ac_layer); acdf=pd.DataFrame(ac_layer)

        for col,title,suffix in [("row_direction_acf","Row-direction autocorrelation","acf_rows"),
                                 ("column_direction_acf","Column-direction autocorrelation","acf_columns"),
                                 ("flattened_acf","Flattened weight-vector autocorrelation","acf_flattened")]:
            plt.figure(figsize=(8,5))
            for t in TASKS:
                z=acdf[acdf.task==t]
                plt.plot(z.lag,z[col],marker="o",label=t)
            plt.axhline(0,linewidth=.8); plt.xlabel("Lag"); plt.ylabel("Normalized autocorrelation")
            plt.title(f"{title}: {layer}"); plt.legend()
            savefig(plots/f"{suffix}_{safe_name(layer)}")

        # Rank-ordered singular values and controls
        plt.figure(figsize=(9,5.5))
        for t in TASKS:
            s=singular_by_task[t]; mu,sd=control_by_task[t]; rank=np.arange(1,len(s)+1)
            plt.scatter(rank, s, s=18, label=f"{t}: learned")
            plt.scatter(rank, mu, s=18, marker="x", label=f"{t}: shuffled control")
            plt.fill_between(rank,mu-sd,mu+sd,alpha=.15)
        plt.xlabel("Singular-value rank"); plt.ylabel("Singular value")
        plt.title(f"Rank-ordered singular values: {layer}"); plt.legend(fontsize=8)
        savefig(plots/f"singular_value_rank_{safe_name(layer)}")

        allsv=np.concatenate([singular_by_task[t] for t in TASKS])
        h_edges=np.linspace(allsv.min(),allsv.max(),30) if not np.isclose(allsv.min(),allsv.max()) else np.linspace(allsv.min()-.5,allsv.max()+.5,30)
        plt.figure(figsize=(8,5))
        for t in TASKS:
            plt.hist(singular_by_task[t],bins=h_edges,density=True,histtype="step",linewidth=1.8,label=t)
        plt.xlabel("Singular value"); plt.ylabel("Probability density")
        plt.title(f"Singular-value distribution: {layer}"); plt.legend()
        savefig(plots/f"singular_value_distribution_{safe_name(layer)}")

        C=np.full((3,3),np.nan); D=np.full((3,3),np.nan)
        for i,ti in enumerate(TASKS):
            for j,tj in enumerate(TASKS):
                C[i,j]=corr(mats[ti],mats[tj]); D[i,j]=ndist(mats[ti],mats[tj])
                compare_rows.append({"layer":layer,"task_reference":ti,"task_comparison":tj,
                                     "pearson_correlation":C[i,j],"normalized_frobenius_distance":D[i,j],
                                     "relative_difference_wrt_reference":rdiff(mats[ti],mats[tj])})
        for Z,title,suffix in [(C,"Task-wise Pearson correlation","task_correlation"),
                               (D,"Task-wise normalized Frobenius distance","task_distance")]:
            plt.figure(figsize=(5.5,4.8)); im=plt.imshow(Z,aspect="equal"); plt.colorbar(im)
            plt.xticks(range(3),TASKS); plt.yticks(range(3),TASKS); plt.title(f"{title}: {layer}")
            for i in range(3):
                for j in range(3):
                    val=Z[i,j]; plt.text(j,i,"nan" if not np.isfinite(val) else f"{val:.3f}",ha="center",va="center")
            savefig(plots/f"{suffix}_{safe_name(layer)}")

    E=pd.DataFrame(ent_rows); ES=pd.DataFrame(sens_rows); A=pd.DataFrame(ac_rows)
    SP=pd.DataFrame(spec_rows); SV=pd.DataFrame(srows); CP=pd.DataFrame(compare_rows); BE=pd.DataFrame(edge_rows)
    E.to_csv(data/"entropy_summary.csv",index=False); ES.to_csv(data/"entropy_sensitivity.csv",index=False)
    A.to_csv(data/"autocorrelation.csv",index=False); SP.to_csv(data/"spectral_metrics.csv",index=False)
    SV.to_csv(data/"singular_values.csv",index=False); CP.to_csv(data/"task_similarity.csv",index=False)
    BE.to_csv(data/"entropy_bin_edges.csv",index=False)

    # Entropy figures
    # Scatter plots are used instead of bars because entropy is a point estimate
    # for each task/layer. The y-axis is zoomed automatically so that small
    # differences between tasks are visible.
    for col,ylabel,suf in [("entropy_bits","Shannon marginal entropy (bits)","entropy_bits"),
                           ("normalized_entropy","Normalized Shannon marginal entropy","entropy_normalized")]:
        plt.figure(figsize=(9,5))
        x=np.arange(len(layers))
        offsets=np.linspace(-0.12,0.12,len(TASKS))
        all_values=[]

        for offset,t in zip(offsets,TASKS):
            z=E[E.task==t].set_index("layer").reindex(layers)
            y=z[col].to_numpy(dtype=float)
            all_values.extend(y[np.isfinite(y)])
            plt.scatter(x+offset,y,s=70,label=t)

            # Add exact value labels above each point.
            for xi,yi in zip(x+offset,y):
                if np.isfinite(yi):
                    plt.text(xi,yi,f"{yi:.3f}",ha="center",va="bottom",fontsize=8)

        all_values=np.asarray(all_values,dtype=float)
        if all_values.size:
            ymin=float(np.nanmin(all_values))
            ymax=float(np.nanmax(all_values))
            span=ymax-ymin
            if span <= np.finfo(float).eps:
                margin=max(0.05,abs(ymin)*0.01)
            else:
                margin=max(0.05*span,0.01)
            plt.ylim(ymin-margin,ymax+margin*2.5)

        plt.axhline(float(np.nanmean(all_values)),linewidth=.8,linestyle="--",alpha=.5) if all_values.size else None
        plt.xticks(x,layers,rotation=25,ha="right")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} by task and layer")
        plt.legend()
        savefig(plots/suf)

    metadata={"task_paths":{t:str(p) for t,p in paths.items()},"analyzed_layers":layers,
              "entropy_bins":args.bins,"sensitivity_bins":list(SENSITIVITY_BINS),
              "max_lag":args.max_lag,"shuffled_controls":args.controls,"random_seed":args.seed}
    (data/"run_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    write_readme(out,args,layers)
    print(f"Completed. Results: {out.resolve()}")
    print(f"CSV files: {data.resolve()}")
    print(f"Figures: {plots.resolve()}")

if __name__=="__main__":
    try: main()
    except Exception as exc:
        print(f"ERROR: {exc}",file=sys.stderr)
        raise
