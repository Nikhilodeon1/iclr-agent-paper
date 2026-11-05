"""Regenerates the five report figures from the raw result CSVs.

    cd experiments/poisoning && python make_figures.py        (about 2 minutes, CPU only)

Reads CSVs from this folder and from ../../anotherResultsWave/; writes fig1_model.png ... fig5_decline.png
to ../../figures/report_v2/. Requires numpy, pandas, matplotlib, scipy (see requirements.txt).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); EXP = HERE
W6 = os.path.abspath(os.path.join(HERE, "..", "..", "anotherResultsWave"))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "figures", "report_v2")); os.makedirs(OUT, exist_ok=True)
os.chdir(OUT)

def P(n):
    for d in (EXP, W6):
        if os.path.exists(os.path.join(d, n)): return os.path.join(d, n)
    raise FileNotFoundError(n)

import matplotlib as mpl, matplotlib.pyplot as plt
def apply_figure_style(sizes=(8, 7, 6)):
    mpl.rcParams.update({"font.size": sizes[0], "axes.labelsize": sizes[0], "xtick.labelsize": sizes[1], "ytick.labelsize": sizes[1],
                         "legend.fontsize": sizes[2], "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100,
                         "savefig.dpi": 300, "font.family": "DejaVu Sans"})

import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, EXP); import analyse_wave7 as a7
apply_figure_style(sizes=(8,7,6))
BLUE="#1f5fa8"; GREY="#7f7f7f"; LG="#cfcfcf"; ORG="#d95f02"; GRN="#2a8f5a"; PUR="#7b3fa0"
W=6.5
def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return max(0,c-h),min(1,c+h)
def letter(ax,t,dx=-0.16): ax.text(dx,1.05,t,transform=ax.transAxes,fontsize=9,fontweight="bold",va="bottom")
def se(g): return g.std(ddof=1)/np.sqrt(len(g)) if len(g)>1 else 0.0
def per_seed_mean(d,keys,col="asr"): return d.groupby(keys+["seed"])[col].mean().reset_index()
def mean_se(d,key):
    g=d.groupby(key).asr; return g.mean(), g.apply(se)

# ================= Figure 1: model and first test
rng=np.random.default_rng(3)
fig=plt.figure(figsize=(W,3.05)); gs=fig.add_gridspec(2,2,width_ratios=[1.1,1],wspace=0.3,hspace=0.75)
for r,(ttl,gen) in enumerate([("Rare trigger: never occurs in clean examples",False),("Common trigger: also occurs in clean examples",True)]):
    ax=fig.add_subplot(gs[r,0]); ax.set_xlim(0,60); ax.set_ylim(0,1.0); ax.axis("off"); ax.text(0,1.08,ttl,fontsize=7,va="bottom")
    pois=set(rng.choice(60,5,replace=False)); trig=set(rng.choice([i for i in range(60) if i not in pois],12,replace=False)) if gen else set()
    for i in range(60): ax.add_patch(Rectangle((i,0.05),0.8,0.8,color=ORG if i in pois else (BLUE if i in trig else LG),lw=0))
ax=fig.add_subplot(gs[:,1])
th=pd.read_csv(P("rho_v2_thresholds.csv")); g=th[th.condition=="generic"].sort_values("n_clean")
xs=np.array([100,300,900.]); 
ax.plot([40,1500],[40,1500],"--",color="black",lw=0.9,label="Equation (4): number needed = clean uses")
ax.errorbar(xs,g.N50.values,yerr=[g.N50.values-g.ci_lo.values,g.ci_hi.values-g.N50.values],fmt="o",color=BLUE,ms=4.5,capsize=2,lw=1.1,label="Common trigger, measured",zorder=3)
rr=th[th.condition=="rare"].sort_values("n_clean"); xr=np.array([6,9,14.])
for x,(_,row) in zip(xr,rr.iterrows()):
    if pd.isna(row.N50) or row.N50<25.5: ax.plot([x],[25],marker="v",color=GREY,ms=5,ls="none",mfc="white",zorder=3)
    else: ax.errorbar([x],[row.N50],yerr=[[row.N50-row.ci_lo],[row.ci_hi-row.N50]],fmt="s",color=GREY,ms=4.5,capsize=2,lw=1.1,zorder=3)
ax.plot([],[],"s",color=GREY,ms=4.5,label="Rare trigger (open: below 25)")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(4,1800); ax.set_ylim(15,1800); ax.set_yticks([30,100,300,1000]); ax.set_yticklabels(["30","100","300","1,000"]); ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax.set_xticks([6,9,14,100,300,900]); ax.set_xticklabels(["","0","","100","300","900"]); ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax.set_xlabel("Expected clean uses of the trigger"); ax.set_ylabel("Poisoned examples for 50% success")
ax.legend(frameon=False,fontsize=5.4,loc="upper left"); letter(ax,"b",-0.2); fig.text(0.02,0.96,"a",fontsize=9,fontweight="bold")
fig.savefig("fig1_model.png",dpi=300,bbox_inches="tight"); plt.close(fig)

# ================= Figure 2: scalar model against simulations
s=pd.read_csv(P("sparse_trigger_check.csv")); hd=pd.read_csv(P("highdim_sparse_check.csv"))
fig,axs=plt.subplots(1,3,figsize=(W,2.35),gridspec_kw=dict(wspace=0.45))
ax=axs[0]; a_=s[~s.at_boundary]; b_=s[s.at_boundary]; lim=(min(s.z_pred.min(),s.z_sim.min())-0.4,max(s.z_pred.max(),s.z_sim.max())+0.4); ax.plot(lim,lim,color=GREY,lw=0.8)
ax.scatter(a_.z_pred,a_.z_sim,s=16,color=BLUE,label=f"Off the boundary (n = {len(a_)})",zorder=3); ax.scatter(b_.z_pred,b_.z_sim,s=18,facecolor="white",edgecolor=BLUE,lw=1,label=f"Prediction = 0 (n = {len(b_)})",zorder=3)
ax.axhline(0,color=LG,lw=0.6); ax.axvline(0,color=LG,lw=0.6); ax.set_xlabel("Predicted margin, Equation (4)"); ax.set_ylabel("Simulated margin, scalar model"); ax.legend(frameon=False,fontsize=5.6,loc="upper left"); letter(ax,"a")
ax=axs[1]; q=hd[hd.model=="qpos"].copy(); q["zstar"]=np.log(q.n_poison/(q.q*q.n_clean)); cell=q.groupby(["B","q","n_clean","n_poison"]).agg(z=("z_tail","mean"),zs=("zstar","first")).reset_index(); cell=cell[cell.zs.abs()>=0.3]
for Bv,mk,col in [(0.5,"o","#8fb4de"),(1.0,"s","#4d86c4"),(2.0,"^",BLUE)]:
    c=cell[cell.B==Bv]; ax.scatter(c.zs,c.z,s=15,marker=mk,color=col,label=f"B = {Bv:g}",zorder=3)
lim=(cell.zs.min()-0.4,cell.zs.max()+0.4); ax.plot(lim,lim,color=GREY,lw=0.8); ax.axhline(0,color=LG,lw=0.6); ax.axvline(0,color=LG,lw=0.6)
ax.set_xlabel("Predicted margin (scalar)"); ax.set_ylabel("Margin, 200-parameter model"); ax.legend(frameon=False,fontsize=5.6,loc="upper left",title="Clean weight norm",title_fontsize=5.6); letter(ax,"b")
ax=axs[2]; q0=hd[hd.model=="q0"]
for Bv,mk,col in [(0.5,"o","#8fb4de"),(1.0,"s","#4d86c4"),(2.0,"^",BLUE)]:
    c=q0[q0.B==Bv].groupby("n_clean").nstar.agg(["mean",se]); ax.errorbar(c.index,c["mean"],yerr=c["se"],fmt=mk+"-",color=col,ms=3.5,lw=1,capsize=1.5,label=f"B = {Bv:g}")
ax.axhline((2+np.exp(2)-1)/0.3,color="black",ls="--",lw=0.9,label="Scalar value, Equation (2)"); ax.set_xscale("log"); ax.set_xticks([1000,4000,16000]); ax.set_xticklabels(["1,000","4,000","16,000"])
ax.set_ylim(0,75); ax.set_yticks([0,10,20,30]); ax.set_xlabel("Clean steps"); ax.set_ylabel("Poisoned examples for margin 2"); ax.legend(frameon=False,fontsize=5.4,loc="upper left",ncol=1); letter(ax,"c")
fig.savefig("fig2_checks.png",dpi=300,bbox_inches="tight"); plt.close(fig)

# ================= Figure 3: transformers and Pythia
r70=pd.read_csv(P("results_realtext_5seed.csv")); r160=pd.read_csv(P("results_realtext_160m.csv"))
t70=a7.table(r70); t160=a7.table(r160); ms=pd.read_csv(P("th_modelsize_valid.csv"))
fig,axs=plt.subplots(1,3,figsize=(W,2.45),gridspec_kw=dict(wspace=0.5,width_ratios=[1.05,1,1]))
ax=axs[0]
for t,col,mk,lab,off in [(t70,BLUE,"o","Pythia-70M",0.93),(t160,ORG,"s","Pythia-160M",1.07)]:
    p=t[t.clean_uses>0]; ax.errorbar(p.clean_uses*off,p.N50,yerr=[p.N50-p.lo,p.hi-p.N50],fmt=mk+"-",color=col,ms=4,lw=1,capsize=2,label=lab,zorder=3)
    z=t[t.clean_uses==0].iloc[0]; ax.errorbar([6.5*off],[z.N50],yerr=[[z.N50-z.lo],[z.hi-z.N50]],fmt=mk,color=col,ms=4,capsize=2,zorder=3)
xs=np.array([22,420]); ax.plot(xs,xs,"--",color="black",lw=0.9,label="Equal to clean uses")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(4.5,500); ax.set_ylim(5,420); ax.set_xticks([6.5,30,120,300]); ax.set_xticklabels(["0","30","120","300"]); ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter()); ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter()); ax.set_yticks([10,30,100]); ax.set_yticklabels(["10","30","100"])
ax.set_xlabel("Expected clean uses of the trigger"); ax.set_ylabel("Poisoned examples for 50% success"); ax.legend(frameon=False,fontsize=5.4,loc="upper left"); letter(ax,"a")
ax=axs[1]; cm=mpl.cm.Blues
tt=t70.sort_values("clean_uses")
for i,(_,row) in enumerate(tt.iterrows()):
    g=r70[r70.trigger_id==row.trigger_id]; m=g.groupby("n_poison").asr; mu,s_=m.mean(),m.apply(se); col=cm(0.35+0.6*i/(len(tt)-1))
    ax.errorbar(mu.index,mu.values,yerr=s_.values,fmt="o-",color=col,ms=3.5,lw=1.1,capsize=1.5,label=f"{row.clean_uses:.0f} uses")
ax.axhline(0.5,color=GREY,lw=0.6,ls=":"); ax.set_xscale("log"); ax.set_xticks([4,16,64,256,1024]); ax.set_xticklabels(["4","16","64","256","1,024"]); ax.set_xlabel("Poisoned examples, Pythia-70M"); ax.set_ylabel("Attack success rate"); ax.set_ylim(-0.03,1.5); ax.set_yticks([0,0.25,0.5,0.75,1.0]); ax.legend(frameon=False,fontsize=5.2,loc="upper center",ncol=2,columnspacing=1.2,handletextpad=0.3,title="Expected clean uses of the trigger",title_fontsize=5.2); letter(ax,"b")
ax=axs[2]; rows=[]
for (e,npar),g in ms.groupby(["experiment","n_params"]):
    pt,lo,hi=a7.n50_ci(g); rows.append((npar,pt,lo,hi,g.groupby("n_poison").asr.mean().iloc[-1]))
for npar,pt,lo,hi,last in sorted(rows):
    if np.isfinite(pt): ax.errorbar([npar],[pt],yerr=[[max(pt-lo,0)],[max(hi-pt,0)]],fmt="o",color=BLUE,ms=4.5,capsize=2,zorder=3)
    else: ax.plot([npar],[64],marker="^",color=GREY,mfc="white",ms=5.5,ls="none",zorder=3); ax.annotate("50% not reached\nby 64 examples",(npar,64),xytext=(8,-3),textcoords="offset points",fontsize=5.4,ha="left",va="center",color=GREY)
ax.set_xscale("log"); ax.set_xticks([1e5,1e6,1e7]); ax.set_xticklabels(["0.1M","1M","10M"]); ax.set_ylim(0,80); ax.set_xlim(5e4,4e7)
ax.set_xlabel("Parameters (trained from scratch)"); ax.set_ylabel("Poisoned examples for 50% success"); letter(ax,"c")
fig.savefig("fig3_scale.png",dpi=300,bbox_inches="tight"); plt.close(fig)

# ================= Figure 4: optimizer mechanism
ed=pd.read_csv(P("results_embdisp.csv")); ed["opt"]=ed.experiment.str.extract(r"embdisp_(\w+?)_")[0]; ed["k"]=ed.experiment.str.extract(r"_k(\d+)$")[0].astype(int); ed["b2"]=ed.experiment.str.extract(r"_b([0-9.]+)_k")[0]
pg=pd.read_csv(P("results_paramgroup_pooled.csv")); pg["E"]=pg.experiment.str.extract(r"_E(\d+)$")[0].astype(int); pg["cond"]=pg.experiment.str.replace(r"_E\d+$","",regex=True); pg["succ"]=pg.asr>0.5
ma=pd.concat([pd.read_csv(P("results_matched_adamw_a.csv")),pd.read_csv(P("results_matched_adamw_b.csv"))]); ma["E"]=ma.experiment.str.extract(r"_E(\d+)_lr")[0].astype(int)
sg=pd.concat([pd.read_csv(P("results_events_matched.csv")),pd.read_csv(P("results_events_sgd_n64_topup.csv"))]); sg=sg[sg.experiment.str.startswith("events_sgd")&(sg.n_poison==64)&(sg.seed<20)].copy(); sg["E"]=sg.experiment.str.extract(r"_E(\d+)$")[0].astype(int)
fig,axs=plt.subplots(1,3,figsize=(W,2.55),gridspec_kw=dict(wspace=0.45))
ax=axs[0]; ks=[1,4,16,64]; shade={"0.95":"#8fb4de","0.99":"#4d86c4","0.999":BLUE}
for b2,col in shade.items():
    q=ed[(ed.opt=="adamw")&(ed.b2==b2)]; m_=q.groupby("k").disp_step.mean(); ax.plot(m_.index,m_.values,"o-",color=col,ms=3.5,lw=1.1); ax.axhline(q.pred_disp.mean(),color="black",ls="--",lw=0.6)
q=ed[ed.opt=="sgd"]; m_=q.groupby("k").disp_step.mean(); ax.plot(m_.index,m_.values,"s-",color=GREY,ms=3.5,lw=1.1,label="SGD"); 
ax.plot([],[],"o-",color=BLUE,ms=3.5,lw=1.1,label="AdamW"); ax.plot([],[],"k--",lw=0.6,label="Closed form")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks]); ax.set_xlabel("Poisoned examples per batch"); ax.set_ylabel("Movement of the trigger embedding"); ax.legend(frameon=False,fontsize=5.4,loc="center left",bbox_to_anchor=(0.0,0.45)); letter(ax,"a")
ax=axs[1]; Es=[1,4,16]
for cond,col,mk,lab,off in [("pg_adamw_trigadamw_N16",BLUE,"o","AdamW everywhere",0.95),("pg_adamw_trigsgd_N16",GREY,"s","Trigger row on SGD",1.0),("pg_sgd_trigadamw_N16",ORG,"^","SGD, row on AdamW",1.05)]:
    d=pg[pg.cond==cond]; ps=[];lo=[];hi=[]
    for E in Es: x=d[d.E==E]; k=int(x.succ.sum()); n=len(x); l,h=wilson(k,n); ps.append(k/n); lo.append(k/n-l); hi.append(h-k/n)
    ax.errorbar([E*off for E in Es],ps,yerr=[lo,hi],fmt=mk+"-",color=col,ms=4,lw=1.1,capsize=2,label=lab)
ax.set_xscale("log"); ax.set_xticks(Es); ax.set_xticklabels([str(e) for e in Es]); ax.set_ylim(-0.02,1.02); ax.set_xlabel("Poisoned batches (16 examples)"); ax.set_ylabel("Fraction of runs succeeding"); ax.legend(frameon=False,fontsize=5.4,loc="upper left"); letter(ax,"b")
ax=axs[2]; Es=[1,2,4,16]; rng=np.random.default_rng(1)
for d,col,mk,lab,off in [(ma,BLUE,"o","AdamW, learning rate 0.0003",0.93),(sg,GREY,"s","SGD",1.07)]:
    mu=d.groupby("E").asr.mean(); s_=d.groupby("E").asr.apply(se)
    for E in Es:
        y=d[d.E==E].asr.values; ax.scatter(E*off*np.exp(rng.normal(0,0.03,len(y))),y,s=4,color=col,alpha=0.22,lw=0,zorder=1)
    ax.errorbar([E*off for E in Es],mu.values,yerr=s_.values,fmt=mk+"-",color=col,ms=4.5,lw=1.3,capsize=2,label=lab,zorder=3)
ax.set_xscale("log"); ax.set_xticks(Es); ax.set_xticklabels([str(e) for e in Es]); ax.set_ylim(-0.04,1.3); ax.set_yticks([0,0.25,0.5,0.75,1.0]); ax.set_xlabel("Poisoned batches (64 examples)"); ax.set_ylabel("Attack success rate (mean, SE)"); ax.legend(frameon=False,fontsize=5.4,loc="upper left"); letter(ax,"c")
fig.savefig("fig4_mechanism.png",dpi=300,bbox_inches="tight"); plt.close(fig)

# ================= Figure 5: the decline
dd=pd.read_csv(P("results_decomp_pooled.csv")); m=dd.experiment.str.extract(r"decomp_(\w+?)_B(\d+)_A(\d+)_S(\d+)$"); dd["B"]=m[1].astype(int); dd["A"]=m[2].astype(int); dd["S"]=m[3].astype(int)
sp=pd.concat([pd.read_csv(P(f"results_spacing_s{k}.csv")) for k in (8,13,18,23)]); sp["S"]=sp.experiment.str.extract(r"_S(\d+)$")[0].astype(int)
fig,axs=plt.subplots(1,3,figsize=(W,2.55),gridspec_kw=dict(wspace=0.45))
ax=axs[0]; As=[0,8000,32000]; cols={0:"#bcd3ec",2000:"#8fb4de",8000:"#4d86c4",32000:BLUE}
for Bv,col in cols.items():
    d=dd[(dd.B==Bv)&(dd.S==700)]; mu=d.groupby("A").asr.mean().reindex(As); s_=d.groupby("A").asr.apply(se).reindex(As); ax.errorbar(range(3),mu.values,yerr=s_.values,fmt="o-",color=col,ms=4,lw=1.2,capsize=2,label=f"{Bv:,} steps before")
ax.set_xticks(range(3)); ax.set_xticklabels(["0","8,000","32,000"]); ax.set_xlabel("Clean steps after the poison"); ax.set_ylabel("Attack success rate (mean, SE)"); ax.set_ylim(-0.03,0.9); ax.legend(frameon=False,fontsize=5.4,loc="upper right"); letter(ax,"a")
ax=axs[1]; Ss=[700,2800,10000]; rng=np.random.default_rng(2)
for k,S in enumerate(Ss):
    y=sp[sp.S==S].pivot_table(index="seed",values="asr").asr.values; ax.scatter(k+rng.normal(0,0.045,len(y)),y,s=7,color=BLUE,alpha=0.45,lw=0,zorder=2)
    ax.errorbar([k+0.25],[y.mean()],yerr=[se(pd.Series(y))],fmt="o",color="black",ms=4,capsize=2,zorder=3)
    y0=dd[(dd.B==0)&(dd.A==0)&(dd.S==S)].asr.values
    if len(y0): ax.errorbar([k-0.25],[y0.mean()],yerr=[se(pd.Series(y0))],fmt="o",color=GREY,ms=4,capsize=2,zorder=3)
ax.plot([],[],"o",color=GREY,ms=4,label="First look, 8 seeds"); ax.plot([],[],"o",color="black",ms=4,label="New test, 20 seeds (mean, SE)"); ax.scatter([],[],s=7,color=BLUE,alpha=0.5,label="Individual new seeds")
ax.set_xticks(range(3)); ax.set_xticklabels(["700","2,800","10,000"]); ax.set_xlim(-0.55,2.55); ax.set_ylim(-0.04,1.62); ax.set_yticks([0,0.25,0.5,0.75,1.0]); ax.set_xlabel("Steps between batches"); ax.set_ylabel("Attack success rate"); ax.legend(frameon=False,fontsize=5.2,loc="upper center",markerscale=0.7,handletextpad=0.2,labelspacing=0.25); letter(ax,"b")
ax=axs[2]; Ts=[500,2000,8000,32000]
for f,col,mk,lab in [("results_ds_ctrl.csv",BLUE,"o","AdamW, 16 examples"),("results_ds_wd0.csv",PUR,"D","AdamW, no weight decay"),("results_ds_sgd_n64.csv",GREY,"s","SGD, 64 examples"),("results_ds_sgd_n128.csv",ORG,"^","SGD, 128 examples")]:
    d=pd.read_csv(P(f)); mu=d.groupby("n_clean").asr.mean().reindex(Ts); s_=d.groupby("n_clean").asr.apply(se).reindex(Ts); ax.errorbar(range(4),mu.values,yerr=s_.values,fmt=mk+"-",color=col,ms=3.8,lw=1.1,capsize=1.5,label=lab)
ax.set_xticks(range(4)); ax.set_xticklabels(["500","2,000","8,000","32,000"]); ax.set_xlabel("Clean steps (poison at the end)"); ax.set_ylabel("Attack success rate"); ax.set_ylim(-0.03,1.55); ax.set_yticks([0,0.25,0.5,0.75,1.0]); ax.legend(frameon=False,fontsize=5.2,loc="upper center",ncol=1,markerscale=0.7,labelspacing=0.25); letter(ax,"c")
fig.savefig("fig5_decline.png",dpi=300,bbox_inches="tight"); plt.close(fig)
