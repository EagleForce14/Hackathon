# %% [markdown]
# # EDA: corrélations ciblées (ON / OFF / délai de prise / données manquantes)
#
# Complète `data/eda.py` en allant plus loin sur les relations entre les
# attributs et la `target` (vrai OFF débiaisé).
#
# - **Raw data** en lecture seule, depuis `data/*.csv`.
# - **Sortie** : `data/eda_correlations.html`, un rapport interactif plotly
#   (survol des points pour voir les valeurs).
#
# Exécution :
#   .venv/Scripts/python.exe data/eda_correlations.py

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

EDA_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd() / "data"
OUT_HTML = EDA_DIR / "eda_correlations.html"

# Palette (slots catégoriels validés, fond clair) + divergente bleu <-> rouge.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRID, INK, INK_2 = "#e6e5e1", "#0b0b0b", "#52514e"
DIVERGING = [
    [0.0, "#9e2b2a"],
    [0.25, "#e34948"],
    [0.5, "#f0efec"],
    [0.75, "#2a78d6"],
    [1.0, "#104281"],
]
LAYOUT = dict(
    template="plotly_white",
    font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=13, color=INK),
    paper_bgcolor="#fcfcfb",
    plot_bgcolor="#fcfcfb",
    margin=dict(l=70, r=30, t=60, b=60),
    hoverlabel=dict(bgcolor="white", font_size=12),
)

# %% [markdown]
# ## Chargement et variables dérivées
#
# - `duree_maladie` = `age - age_at_diagnosis`.
# - `traite` = le patient a une dose `ledd` renseignée **ou** un score ON.
#   Les visites sans ON correspondent presque toujours à des patients non
#   traités (`ledd` manquant, début de maladie).
# - Indicateurs `*_manquant` : la présence ou l'absence d'une mesure est
#   elle-même une information.

# %%
X = pd.read_csv(EDA_DIR / "X_train.csv", index_col="Index")
y = pd.read_csv(EDA_DIR / "y_train.csv", index_col="Index")
df = X.join(y)

df["duree_maladie"] = df["age"] - df["age_at_diagnosis"]
df["cohorte_B"] = (df["cohort"] == "B").astype(int)
for g in ["LRRK2+", "GBA+", "OTHER+"]:
    df[f"gene_{g.rstrip('+')}"] = (df["gene"] == g).astype(int)
for col in ["on", "off", "ledd", "gene", "age_at_diagnosis", "time_since_intake_on", "time_since_intake_off"]:
    df[f"{col}_manquant"] = df[col].isna().astype(int)
df["motif"] = (
    np.where(df["on"].notna(), "ON", "–") + " / " + np.where(df["off"].notna(), "OFF", "–")
)
df["traite"] = np.where(df["ledd"].notna() | df["on"].notna(), "traité", "non traité")
df["target_moins_off"] = df["target"] - df["off"]
df["on_sur_target"] = df["on"] / df["target"].where(df["target"] > 0)

# Moyennes par patient (tendance longitudinale)
grp = df.groupby("patient_id")
df["off_moy_patient"] = grp["off"].transform("mean")
df["on_moy_patient"] = grp["on"].transform("mean")
df.shape

# %% [markdown]
# ## 1. Matrice de corrélation (Spearman)
#
# Spearman plutôt que Pearson : robuste aux relations monotones non
# linéaires (ex. courbe pharmacocinétique). Corrélations calculées par paires
# sur les lignes disponibles.

# %%
cols_heat = [
    "target", "off", "on", "off_moy_patient", "on_moy_patient",
    "duree_maladie", "age", "age_at_diagnosis", "ledd",
    "time_since_intake_on", "time_since_intake_off",
    "sexM", "cohorte_B", "gene_LRRK2", "gene_GBA", "gene_OTHER",
    "on_manquant", "off_manquant", "ledd_manquant", "gene_manquant",
]
corr = df[cols_heat].corr(method="spearman")
fig_heat = go.Figure(
    go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        zmin=-1,
        zmax=1,
        colorscale=DIVERGING,
        xgap=2,
        ygap=2,
        text=corr.round(2).values,
        texttemplate="%{text}",
        textfont=dict(size=9),
        hovertemplate="%{y} × %{x}<br>ρ = %{z:.3f}<extra></extra>",
        colorbar=dict(title="ρ", thickness=12),
    )
)
fig_heat.update_layout(**LAYOUT, height=720, title="Corrélations de Spearman entre attributs")
fig_heat.update_yaxes(autorange="reversed")

# %% [markdown]
# ## 2. Corrélation de chaque attribut avec la target

# %%
tcorr = corr["target"].drop("target").sort_values()
fig_bar = go.Figure(
    go.Bar(
        x=tcorr.values,
        y=tcorr.index,
        orientation="h",
        marker=dict(color=np.where(tcorr.values >= 0, BLUE, "#e34948")),
        hovertemplate="%{y}<br>ρ avec target = %{x:.3f}<extra></extra>",
    )
)
fig_bar.update_layout(
    **LAYOUT, height=560, title="Corrélation (Spearman) avec la target", bargap=0.25
)
fig_bar.update_xaxes(range=[-1, 1], zeroline=True, zerolinecolor=INK_2, gridcolor=GRID)

# %% [markdown]
# ## 3. OFF mesuré vs vrai OFF
#
# La diagonale `y = x` correspond à « aucun biais ». Chez les patients
# traités, le OFF mesuré **sous-estime** le vrai OFF (effet résiduel du
# médicament) ; chez les non traités, il en est proche.

# %%
rng = np.random.default_rng(0)
sub = df[df["off"].notna()]
sub = sub.iloc[rng.choice(len(sub), size=min(6000, len(sub)), replace=False)]
fig_off = go.Figure()
for name, color in [("non traité", ORANGE), ("traité", BLUE)]:
    s = sub[sub["traite"] == name]
    fig_off.add_trace(
        go.Scattergl(
            x=s["off"], y=s["target"], mode="markers", name=name,
            marker=dict(size=5, color=color, opacity=0.45),
            hovertemplate="OFF mesuré %{x}<br>vrai OFF %{y}<extra>" + name + "</extra>",
        )
    )
fig_off.add_trace(
    go.Scatter(x=[0, 110], y=[0, 110], mode="lines", name="y = x",
               line=dict(color=INK_2, width=1.5, dash="dash"), hoverinfo="skip")
)
fig_off.update_layout(**LAYOUT, height=520, title="Vrai OFF (target) en fonction du OFF mesuré")
fig_off.update_xaxes(title="off (mesuré)", gridcolor=GRID)
fig_off.update_yaxes(title="target (vrai OFF)", gridcolor=GRID)

# %% [markdown]
# ## 4. Effet du délai depuis la prise sur le score ON
#
# Ratio `on / target` en fonction de `time_since_intake_on` (médiane et
# intervalle interquartile par tranche). Juste après la prise, le médicament
# n'est pas encore absorbé : ON reste proche du OFF. Après ~1,5 h l'effet
# plafonne et ON ≈ 45 % du vrai OFF.

# %%
d = df[df["on_sur_target"].notna() & df["time_since_intake_on"].notna()].copy()
bins = np.arange(0, 6.51, 0.25)
d["tranche"] = pd.cut(d["time_since_intake_on"], bins)
agg = d.groupby("tranche", observed=True)["on_sur_target"].agg(
    med="median", q1=lambda s: s.quantile(0.25), q3=lambda s: s.quantile(0.75), n="size"
)
agg = agg[agg["n"] >= 30]
mid = [iv.mid for iv in agg.index]
fig_on = go.Figure()
fig_on.add_trace(go.Scatter(x=mid, y=agg["q3"], mode="lines", line=dict(width=0),
                            showlegend=False, hoverinfo="skip"))
fig_on.add_trace(go.Scatter(x=mid, y=agg["q1"], mode="lines", line=dict(width=0),
                            fill="tonexty", fillcolor="rgba(42,120,214,0.15)",
                            name="intervalle interquartile", hoverinfo="skip"))
fig_on.add_trace(go.Scatter(
    x=mid, y=agg["med"], mode="lines+markers", name="médiane on / target",
    line=dict(color=BLUE, width=2), marker=dict(size=8),
    customdata=agg["n"],
    hovertemplate="%{x:.2f} h<br>médiane = %{y:.2f}<br>n = %{customdata}<extra></extra>",
))
fig_on.update_layout(**LAYOUT, height=460,
                     title="Ratio ON / vrai OFF selon le délai depuis la prise")
fig_on.update_xaxes(title="time_since_intake_on (heures)", gridcolor=GRID)
fig_on.update_yaxes(title="on / target", gridcolor=GRID, rangemode="tozero")

# %% [markdown]
# ## 5. Écart vrai OFF − OFF mesuré selon le délai depuis la prise
#
# Chez les traités, l'écart reste d'environ +8 points quel que soit le délai
# (6 à 25 h) : `time_since_intake_off` apporte peu, contrairement au côté ON.

# %%
t = df[df["target_moins_off"].notna() & df["time_since_intake_off"].notna()].copy()
t["tranche"] = pd.cut(t["time_since_intake_off"], np.arange(6, 26, 1))
agg_off = t.groupby("tranche", observed=True)["target_moins_off"].agg(
    moy="mean", n="size", sd="std"
)
agg_off = agg_off[agg_off["n"] >= 30]
fig_toff = go.Figure(go.Scatter(
    x=[iv.mid for iv in agg_off.index], y=agg_off["moy"], mode="lines+markers",
    line=dict(color=ORANGE, width=2), marker=dict(size=8),
    error_y=dict(type="data", array=1.96 * agg_off["sd"] / np.sqrt(agg_off["n"]),
                 color=ORANGE, thickness=1.5),
    customdata=agg_off["n"],
    hovertemplate="%{x:.1f} h<br>écart moyen = %{y:.2f}<br>n = %{customdata}<extra></extra>",
    name="target − off",
))
fig_toff.update_layout(**LAYOUT, height=420,
                       title="Écart moyen vrai OFF − OFF mesuré selon time_since_intake_off (IC 95 %)")
fig_toff.update_xaxes(title="time_since_intake_off (heures)", gridcolor=GRID)
fig_toff.update_yaxes(title="target − off", gridcolor=GRID, rangemode="tozero")

# %% [markdown]
# ## 6. Les données manquantes comme signal
#
# Distribution de la target selon les mesures disponibles à la visite.

# %%
order = ["– / OFF", "ON / –", "ON / OFF"]
colors = {"– / OFF": ORANGE, "ON / –": BLUE, "ON / OFF": AQUA}
fig_miss = go.Figure()
for m in order:
    s = df.loc[df["motif"] == m, "target"]
    fig_miss.add_trace(go.Box(
        y=s, name=f"{m} (n={len(s):,})".replace(",", " "), marker_color=colors[m],
        boxpoints=False, line=dict(width=1.5),
    ))
fig_miss.update_layout(**LAYOUT, height=440, showlegend=False,
                       title="Target selon les scores mesurés à la visite (ON / OFF)")
fig_miss.update_yaxes(title="target", gridcolor=GRID)

# %% [markdown]
# ## 7. Progression avec la durée de la maladie
#
# Target moyenne et part de mesures manquantes par année depuis le
# diagnostic. Deux graphiques séparés (échelles différentes, pas de double axe).

# %%
p = df[df["duree_maladie"].notna()].copy()
p["annee"] = np.floor(p["duree_maladie"]).clip(upper=18)
agg_p = p.groupby("annee").agg(
    target=("target", "mean"), sd=("target", "std"), n=("target", "size"),
    on_manq=("on_manquant", "mean"), off_manq=("off_manquant", "mean"),
    ledd_manq=("ledd_manquant", "mean"),
)
fig_prog = go.Figure(go.Scatter(
    x=agg_p.index, y=agg_p["target"], mode="lines+markers", name="target moyenne",
    line=dict(color=BLUE, width=2), marker=dict(size=8),
    error_y=dict(type="data", array=agg_p["sd"], color="rgba(42,120,214,0.35)", thickness=1),
    customdata=agg_p["n"],
    hovertemplate="année %{x}<br>target moy = %{y:.1f}<br>n = %{customdata}<extra></extra>",
))
fig_prog.update_layout(**LAYOUT, height=420, showlegend=False,
                       title="Target moyenne (± 1 écart-type) selon la durée de la maladie")
fig_prog.update_xaxes(title="années depuis le diagnostic (18 = 18+)", gridcolor=GRID)
fig_prog.update_yaxes(title="target", gridcolor=GRID, rangemode="tozero")

fig_manq = go.Figure()
for col, name, color in [("on_manq", "ON manquant", BLUE),
                         ("off_manq", "OFF manquant", ORANGE),
                         ("ledd_manq", "LEDD manquant", AQUA)]:
    fig_manq.add_trace(go.Scatter(
        x=agg_p.index, y=agg_p[col], mode="lines+markers", name=name,
        line=dict(color=color, width=2), marker=dict(size=8),
        hovertemplate="année %{x}<br>" + name + " = %{y:.0%}<extra></extra>",
    ))
fig_manq.update_layout(**LAYOUT, height=420,
                       title="Part de mesures manquantes selon la durée de la maladie",
                       legend=dict(orientation="h", y=1.1))
fig_manq.update_xaxes(title="années depuis le diagnostic (18 = 18+)", gridcolor=GRID)
fig_manq.update_yaxes(title="part manquante", tickformat=".0%", range=[0, 1.05], gridcolor=GRID)

# %% [markdown]
# ## 8. Chiffres clés

# %%
m_off = df["off"].notna()
between = grp["target"].transform("mean").var() / df["target"].var()
slopes = grp.apply(
    lambda s: np.polyfit(s["age"], s["target"], 1)[0] if s["age"].nunique() > 1 else np.nan,
    include_groups=False,
)
key_facts = {
    "Visites (train)": f"{len(df):,}".replace(",", " "),
    "Patients (train)": f"{df['patient_id'].nunique():,}".replace(",", " "),
    "ρ(off, target) quand OFF mesuré": f"{df.loc[m_off, ['off', 'target']].corr('spearman').iloc[0, 1]:.2f}",
    "Écart moyen target − off, traités": f"{df.loc[m_off & (df['traite'] == 'traité'), 'target_moins_off'].mean():+.1f}",
    "Écart moyen target − off, non traités": f"{df.loc[m_off & (df['traite'] == 'non traité'), 'target_moins_off'].mean():+.1f}",
    "Part de variance de la target entre patients": f"{between:.0%}",
    "Progression médiane par patient": f"{slopes.median():.1f} pt/an",
}
key_facts

# %% [markdown]
# ## Export HTML

# %%
SECTIONS = [
    ("Matrice de corrélation", fig_heat,
     "Les blocs à retenir : <b>off</b> (ρ≈0,88) et <b>on</b> (ρ≈0,69) sont les plus liés à la target, "
     "puis <b>duree_maladie</b> (0,55) et <b>ledd</b>, eux-mêmes très corrélés entre eux (0,71). "
     "<b>age_at_diagnosis</b> et <b>age</b> sont presque redondants (0,94) : leur différence est plus informative. "
     "Sexe, cohorte et gènes pèsent très peu."),
    ("Corrélation avec la target", fig_bar,
     "Les moyennes par patient (<b>off_moy_patient</b>, <b>on_moy_patient</b>) sont presque aussi fortes que les "
     "mesures de la visite et existent même quand la visite n'a pas de mesure. "
     "<b>on_manquant</b> est fortement négatif : pas de ON signifie pas de traitement, donc un début de maladie."),
    ("OFF mesuré vs vrai OFF", fig_off,
     "Le OFF mesuré sous-estime systématiquement le vrai OFF chez les patients traités "
     "(effet résiduel de la lévodopa), beaucoup moins chez les non traités."),
    ("Délai depuis la prise → score ON", fig_on,
     "La courbe pharmacocinétique est très nette : juste après la prise, ON ≈ 70 % du vrai OFF ; "
     "après ~1,5 h, ON ≈ 45 %. Une feature du type <code>on / f(time_since_intake_on)</code> devrait permettre de "
     "reconstruire le OFF à partir du ON."),
    ("Délai depuis la prise → écart OFF", fig_toff,
     "Côté OFF, l'écart est d'environ +8 points et dépend peu du délai (6 à 25 h). "
     "Attention : time_since_intake_off manque dans 63 % des visites où OFF est mesuré."),
    ("Données manquantes = signal", fig_miss,
     "Les visites « OFF seul » ont une target moyenne de ~21 contre ~44 pour les visites avec ON : "
     "ce sont des patients non traités, en début de maladie. Le motif de valeurs manquantes est une feature à part entière."),
    ("Progression", fig_prog,
     "La target monte d'environ 2,5 points par an et par patient. "
     "79 % de sa variance est entre patients : l'information du patient sur ses autres visites est décisive."),
    ("Mesures manquantes dans le temps", fig_manq,
     "Au diagnostic : pas de traitement (LEDD et ON manquants), seul OFF est mesuré. "
     "Avec la progression, le traitement démarre, ON apparaît, et OFF est de plus en plus souvent sauté."),
]

facts_html = "".join(f"<div class='fact'><div class='v'>{v}</div><div class='k'>{k}</div></div>"
                     for k, v in key_facts.items())
body = []
for i, (title, fig, text) in enumerate(SECTIONS):
    body.append(
        f"<section><h2>{i + 1}. {title}</h2><p>{text}</p>"
        + fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False,
                      config={"displaylogo": False, "responsive": True})
        + "</section>"
    )

html = f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Corrélations Parkinson</title>
<style>
:root {{ color-scheme: light; --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e; --line:#e6e5e1; }}
body {{ margin:0; background:var(--surface); color:var(--ink);
       font-family:system-ui,-apple-system,"Segoe UI",sans-serif; line-height:1.5; }}
main {{ max-width:1100px; margin:0 auto; padding:24px 16px 64px; }}
h1 {{ font-size:1.6rem; margin:0 0 4px; }}
h2 {{ font-size:1.15rem; margin:0 0 6px; }}
.sub {{ color:var(--ink-2); margin:0 0 20px; }}
section {{ border-top:1px solid var(--line); padding:24px 0 8px; }}
section p {{ color:var(--ink-2); max-width:80ch; margin:0 0 8px; }}
.facts {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin-bottom:12px; }}
.fact {{ border:1px solid var(--line); border-radius:8px; padding:12px; background:#fff; }}
.fact .v {{ font-size:1.4rem; font-weight:600; }}
.fact .k {{ font-size:.82rem; color:var(--ink-2); }}
</style></head>
<body><main>
<h1>Corrélations : ON, OFF, délai de prise, données manquantes</h1>
<p class="sub">Jeu d'entraînement Parkinson, target = vrai OFF (MDS-UPDRS partie III). Graphiques interactifs : survolez pour voir les valeurs.</p>
<div class="facts">{facts_html}</div>
{''.join(body)}
</main></body></html>"""
OUT_HTML.write_text(html, encoding="utf-8")
print(f"Rapport écrit : {OUT_HTML}")
