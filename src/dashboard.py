import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
r=json.load(open('out/metrics.json'));m=r['metrics'];s=r['score']
y=np.load('out/_traj_y.npy');meas=np.load('out/_traj_meas.npy')
contacts=json.load(open('out/_contacts.json'));fps=30
t=np.arange(len(y))/fps
feet=np.percentile(y,95);head=np.percentile(y,3)

plt.rcParams.update({'font.size':10,'axes.edgecolor':'#444'})
fig=plt.figure(figsize=(13,8),facecolor='#0d1117')
gs=GridSpec(3,4,figure=fig,height_ratios=[1.1,1,1],hspace=.55,wspace=.45)
ACC='#00d4ff';TXT='#e6edf3'
def style(ax):
    ax.set_facecolor('#161b22')
    for sp in ax.spines.values():sp.set_color('#30363d')
    ax.tick_params(colors=TXT);ax.title.set_color(TXT)
    ax.xaxis.label.set_color(TXT);ax.yaxis.label.set_color(TXT)

# --- trajectoire + contacts (large) ---
ax=fig.add_subplot(gs[0,:3]);style(ax)
ax.plot(t,feet-y,color='#888',lw=1)
ax.plot(t[meas],(feet-y)[meas],'.',color=ACC,ms=4)
for c in contacts:
    ax.axvline(c/fps,color='#ff5555',alpha=.35,lw=1)
ax.scatter(np.array(contacts)/fps,(feet-y)[contacts],color='#ff5555',s=40,zorder=5,label='contact')
ax.set_title('Trajectoire verticale du ballon & contacts detectes',fontweight='bold')
ax.set_xlabel('temps (s)');ax.set_ylabel('hauteur (px)');ax.legend(facecolor='#161b22',labelcolor=TXT)

# --- score gauge ---
ax=fig.add_subplot(gs[0,3]);style(ax);ax.axis('off')
sc=s['score_0_100']
ax.add_patch(plt.Circle((.5,.55),.42,color='#21262d',transform=ax.transAxes))
th=np.linspace(90,90-360*sc/100,100)
ax.plot(.5+.42*np.cos(np.radians(th)),.55+.42*np.sin(np.radians(th)),
        color=ACC,lw=10,transform=ax.transAxes,solid_capstyle='round')
ax.text(.5,.58,f"{sc}",ha='center',va='center',fontsize=40,color=TXT,fontweight='bold',transform=ax.transAxes)
ax.text(.5,.40,"/ 100",ha='center',color='#888',transform=ax.transAxes)
ax.text(.5,.04,f"GRADE  {s['grade']}",ha='center',fontsize=15,color=ACC,fontweight='bold',transform=ax.transAxes)
ax.set_title('Score global',fontweight='bold',color=TXT)

# --- repartition parties du corps ---
ax=fig.add_subplot(gs[1,0]);style(ax)
bp=m['by_body_part'];lbl=[k for k,v in bp.items() if v];val=[bp[k] for k in lbl]
ax.bar(lbl,val,color=['#ff5555','#ffa500','#00d4ff','#4ade80'][:len(lbl)])
ax.set_title('Parties du corps',fontweight='bold');ax.set_ylabel('touches')
plt.setp(ax.get_xticklabels(),rotation=20,ha='right')

# --- equilibre G/D ---
ax=fig.add_subplot(gs[1,1]);style(ax)
ax.bar(['cote G','cote D'],[m['left_foot_side'],m['right_foot_side']],color=['#a78bfa','#4ade80'])
ax.set_title(f"Equilibre G/D ({int(m['balance_LR']*100)}%)",fontweight='bold');ax.set_ylabel('touches')

# --- jauges qualite ---
ax=fig.add_subplot(gs[1,2:]);style(ax)
g=['Regularite\nrythme','Controle\nhauteur','Variete','Equilibre\nG/D']
gv=[m['rhythm_regularity'],m['control_score'],m['variety'],m['balance_LR']]
yp=np.arange(len(g))
ax.barh(yp,gv,color=ACC,height=.55);ax.barh(yp,[1]*len(g),color='#21262d',height=.55,zorder=0)
ax.set_yticks(yp);ax.set_yticklabels(g);ax.set_xlim(0,1)
for i,v in enumerate(gv):ax.text(v+.02,i,f"{v:.2f}",va='center',color=TXT)
ax.set_title('Indices de qualite (0 a 1)',fontweight='bold');ax.invert_yaxis()

# --- chiffres cles ---
ax=fig.add_subplot(gs[2,:]);style(ax);ax.axis('off')
cards=[('DUREE',f"{m['duration_s']} s"),('JONGLES',f"{m['total_juggles']}"),
       ('TEMPO',f"{m['tempo_touches_per_s']} /s"),
       ('INTERVALLE MOY',f"{m['mean_interval_s']} s"),
       ('SERIE MAX',f"{m['longest_streak']}"),('CHUTES',f"{m['drops']}")]
n=len(cards)
for i,(k,v) in enumerate(cards):
    x=i/n+.5/n
    ax.text(x,.62,v,ha='center',fontsize=22,color=ACC,fontweight='bold',transform=ax.transAxes)
    ax.text(x,.30,k,ha='center',fontsize=10,color='#888',transform=ax.transAxes)
fig.suptitle('ANALYSE DE SESSION — JONGLE FOOT   |   POC',color=TXT,fontsize=15,fontweight='bold',x=.5,y=.97)
plt.savefig('out/dashboard.png',dpi=120,facecolor='#0d1117',bbox_inches='tight')
print('dashboard ok')
