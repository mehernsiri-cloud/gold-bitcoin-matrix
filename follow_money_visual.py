"""Independent editorial infographic version of Follow the Money."""
import plotly.graph_objects as go
import streamlit as st

SIGNALS = [
 {"sector":"Énergie & réseaux","icon":"⚡","amount":3400,"label":"3 400 Md$","period":"2026","kind":"Prévision annuelle","summary":"Investissement énergétique mondial attendu","detail":"Environ 2 200 Md$ vers les énergies propres, les réseaux, le stockage et l’électrification ; environ 1 200 Md$ vers les combustibles fossiles.","impact":"Les réseaux, transformateurs, batteries et capacités de production deviennent des infrastructures critiques.","watch":"Dépenses de réseau, raccordements, stockage, nucléaire et retards de projets.","source":"IEA — World Energy Investment 2026","url":"https://www.iea.org/topics/investment","color":"#E9B85C"},
 {"sector":"Défense & sécurité","icon":"🛡️","amount":2887,"label":"2 887 Md$","period":"2025","kind":"Dépense observée","summary":"Dépenses militaires mondiales","detail":"Hausse réelle de 2,9 %. L’Europe représente 864 Md$ (+14 %) et l’Asie-Océanie 681 Md$ (+8,1 %).","impact":"Les budgets pluriannuels peuvent soutenir les carnets de commandes, mais les annonces doivent devenir des contrats et des livraisons.","watch":"Budgets votés, commandes fermes, capacités industrielles et reconstitution des stocks.","source":"SIPRI — 27 avril 2026","url":"https://www.sipri.org/media/press-release/2026/global-military-spending-rise-continues-european-and-asian-expenditures-surge","color":"#B7C2CE"},
 {"sector":"Investissements directs & industrie","icon":"🏭","amount":1600,"label":"1 600 Md$","period":"2025","kind":"Estimation annuelle","summary":"Investissements directs étrangers mondiaux","detail":"Les projets greenfield annoncés dans cinq secteurs stratégiques ont atteint 576 Md$ en 2025, contre 109 Md$ en 2020.","impact":"Montre où les entreprises construisent des capacités productives : usines, semi-conducteurs, data centers et énergie.","watch":"Décisions finales d’investissement, chantiers lancés, annulations et pays hôtes.","source":"UNCTAD — World Investment Report 2026","url":"https://unctad.org/publication/world-investment-report-2026-international-investment-turbulent-era","color":"#86BDA7"},
 {"sector":"IA, data centers & semi-conducteurs","icon":"🧠","amount":900,"label":"≈ 900 Md$","period":"2026","kind":"Estimation annuelle","summary":"Investissement mondial lié à l’IA estimé","detail":"Projection de 5 500 Md$ cumulés d’ici 2030. Les dépenses des grands hyperscalers sont le moteur principal.","impact":"La dépense logicielle entraîne des besoins physiques en GPU, mémoire, centres de données, refroidissement et électricité.","watch":"Capex annoncé vs réalisé, commandes GPU/mémoire, raccordements électriques et cash-flow des hyperscalers.","source":"J.P. Morgan — 19 octobre 2026","url":"https://www.jpmorgan.com/insights/global-research/artificial-intelligence/ai-capex","color":"#8BB8E4"},
 {"sector":"Agroalimentaire & sécurité alimentaire","icon":"🌾","amount":630,"label":"≈ 630 Md$","period":"Estimation 2026","kind":"Capacité potentielle, non flux réalisé","summary":"Capacité potentielle de financement privé","detail":"Le financement climatique nécessaire aux systèmes agroalimentaires est estimé à 1 100 Md$ par an.","impact":"L’irrigation, le stockage, les engrais, les semences et la logistique déterminent la résilience alimentaire.","watch":"Financement déployé, eau, stocks céréaliers, engrais et capacités de stockage.","source":"Food and Land Use Coalition — rapport 2026","url":"https://www.foodandlandusecoalition.org/knowledge-hub/re-wire-agricultural-resilience/","color":"#B7C987"},
 {"sector":"Capital-risque & startups","icon":"🚀","amount":427.1,"label":"427,1 Md$","period":"2025","kind":"Investissement observé","summary":"Investissement mondial en venture capital","detail":"L’IA a capté 258,7 Md$, soit 61 % du total. Les financements se concentrent dans les méga-tours.","impact":"Le capital-risque finance les futurs acteurs technologiques, mais une levée n’est pas encore une capacité rentable.","watch":"Part de l’IA, financement early-stage, sorties/IPO et taux de survie des startups.","source":"OCDE — 17 février 2026","url":"https://www.oecd.org/en/about/news/announcements/2026/02/ai-firms-capture-61-percent-of-global-venture-capital-in-2025.html","color":"#B9A5D8"},
 {"sector":"Santé, biotech & pharmacie","icon":"🧬","amount":60,"label":"60 Md$","period":"2025","kind":"Investissement observé","summary":"Capital-risque mondial dans la santé","detail":"En hausse par rapport à 45,4 Md$ en 2024. HSBC anticipait 65–70 Md$ en 2026.","impact":"Les capitaux ciblent biotechs, dispositifs médicaux et entreprises proches de la commercialisation ; les cycles restent longs.","watch":"Essais cliniques, autorisations, financement late-stage et capacités de production.","source":"HSBC Innovation Banking — 13 janvier 2026","url":"https://www.about.us.hsbc.com/newsroom/press-releases/healthcare-venture-markets-regain-momentum-as-capital-returns","color":"#D9A8BC"},
]

def render_financial_intelligence():
 st.markdown("""
 <style>
 .fm-hero{background:linear-gradient(115deg,#13283e,#244a63 60%,#416f78);padding:28px 30px;border-radius:16px;color:white;margin:8px 0 20px}
 .fm-hero .kicker{font-size:12px;letter-spacing:2px;text-transform:uppercase;color:#c5d8e4;font-weight:700}
 .fm-hero h2{color:white;font-size:30px;line-height:1.15;margin:8px 0}
 .fm-hero p{color:#e3edf2;font-size:15px;margin:0;max-width:850px}
 .fm-card{background:#f6f8fa;border:1px solid #e4e9ee;border-left:5px solid var(--accent);border-radius:12px;padding:16px;min-height:165px;margin-bottom:10px}
 .fm-card-top{display:flex;justify-content:space-between;gap:8px;color:#526274;font-size:12px;font-weight:700}
 .fm-number{font-size:29px;line-height:1.2;font-weight:800;color:#182b3d;margin:13px 0 5px}
 .fm-desc{font-size:14px;color:#34485a;min-height:36px}.fm-meta{font-size:11px;color:#738292;text-transform:uppercase;letter-spacing:.6px;margin-top:9px}
 .fm-insight{background:#f2f6f8;border:1px solid #e3eaf0;border-radius:11px;padding:15px;margin-bottom:10px;color:#26394b}
 </style>
 <div class="fm-hero"><div class="kicker">Follow the Money · Financial Intelligence</div>
 <h2>Où les capitaux construisent-ils l’économie de demain ?</h2>
 <p>Une lecture visuelle des grandes masses financières : énergie, IA, défense, industrie, santé et alimentation.
 Le chiffre seul ne suffit pas : sa nature, son horizon et sa transmission à l’économie réelle comptent autant.</p></div>
 """,unsafe_allow_html=True)
 st.markdown("### Les chiffres qui structurent le paysage")
 st.caption("Des indicateurs de nature différente : dépenses, investissements observés, estimations et capacités potentielles.")
 for start in range(0,len(SIGNALS),2):
  cols=st.columns(2,gap="medium")
  for col,item in zip(cols,SIGNALS[start:start+2]):
   with col:
    st.markdown(f"""<div class="fm-card" style="--accent:{item['color']}">
    <div class="fm-card-top"><span>{item['icon']} {item['sector']}</span><span>{item['period']}</span></div>
    <div class="fm-number">{item['label']}</div><div class="fm-desc">{item['summary']}</div>
    <div class="fm-meta">{item['kind']}</div></div>""",unsafe_allow_html=True)
 st.markdown("### La carte des flux de capitaux")
 st.caption("Classement indicatif par montant. Les montants ne sont pas additionnables ni strictement comparables.")
 ordered=sorted(SIGNALS,key=lambda x:x["amount"],reverse=True)
 fig=go.Figure(go.Bar(x=[x["amount"] for x in ordered],y=[x["sector"] for x in ordered],orientation="h",
  marker=dict(color=[x["color"] for x in ordered]),text=[x["label"] for x in ordered],textposition="outside",
  customdata=[[x["kind"],x["period"]] for x in ordered],
  hovertemplate="<b>%{y}</b><br>%{text}<br>%{customdata[0]} · %{customdata[1]}<extra></extra>"))
 fig.update_layout(height=400,margin=dict(l=5,r=85,t=8,b=8),xaxis=dict(visible=False,showgrid=False),
  yaxis=dict(autorange="reversed",tickfont=dict(size=12)),plot_bgcolor="rgba(0,0,0,0)",
  paper_bgcolor="rgba(0,0,0,0)",showlegend=False)
 st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False},key="fm_visual_capital_map")
 st.markdown("### Ce que ces capitaux peuvent changer")
 a,b=st.columns(2,gap="medium")
 with a:
  st.markdown('<div class="fm-insight"><b>⚡ De l’IA à l’électricité</b><br>Les data centers tirent les besoins en GPU, refroidissement, réseaux, transformateurs et production électrique.</div>',unsafe_allow_html=True)
  st.markdown('<div class="fm-insight"><b>🛡️ Des budgets aux commandes</b><br>Les dépenses de défense ne deviennent des revenus industriels qu’après contrats, livraisons et montée en cadence.</div>',unsafe_allow_html=True)
 with b:
  st.markdown('<div class="fm-insight"><b>🏭 Des annonces aux actifs physiques</b><br>Les investissements directs prennent leur sens lorsqu’ils deviennent sites, équipements et capacités opérationnelles.</div>',unsafe_allow_html=True)
  st.markdown('<div class="fm-insight"><b>🧬 Des cycles longs aux besoins essentiels</b><br>Santé et alimentation répondent à des besoins structurels, avec des risques scientifiques, réglementaires et climatiques.</div>',unsafe_allow_html=True)
 st.markdown("### Décryptage : du chiffre à la thèse d’investissement")
 selected=st.selectbox("Choisir un thème",options=[x["sector"] for x in SIGNALS],key="fm_visual_selected_sector")
 item=next(x for x in SIGNALS if x["sector"]==selected)
 st.markdown(f"#### {item['icon']} {item['sector']} — {item['label']}")
 st.write(item["detail"])
 c1,c2=st.columns(2)
 with c1:
  st.markdown("**Pourquoi c’est important**");st.write(item["impact"])
 with c2:
  st.markdown("**Indicateurs à surveiller**");st.write(item["watch"])
 st.caption(f"Source : {item['source']} · Période : {item['period']} · Nature : {item['kind']}")
 st.markdown(f"[Consulter la source originale ↗]({item['url']})")
 st.warning("Précaution : dépenses publiques, investissements d’entreprises, capital-risque et capacités potentielles ne sont pas comparables ni additionnables. Des chevauchements existent. Cette vue est une analyse éditoriale, pas un signal automatique d’achat ou de vente.")
 st.caption("Version expérimentale indépendante. Les références documentaires et chiffres sont repris de la base de la version actuelle ; leur actualisation doit être vérifiée à chaque publication.")
