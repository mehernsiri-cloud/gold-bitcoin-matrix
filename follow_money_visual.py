"""Financial Intelligence — capital allocation radar, distinct from the legacy flow catalogue."""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

SIGNALS = [
    {"sector":"Énergie & réseaux","icon":"⚡","amount":3400,"label":"3 400 Md$","period":"2026","kind":"Prévision annuelle","summary":"Investissement énergétique mondial attendu","detail":"Environ 2 200 Md$ vers les énergies propres, les réseaux, le stockage et l’électrification ; environ 1 200 Md$ vers les combustibles fossiles.","impact":"Les réseaux, transformateurs, batteries et capacités de production deviennent des infrastructures critiques.","watch":"Dépenses de réseau, raccordements, stockage, nucléaire et retards de projets.","source":"IEA — World Energy Investment 2026","url":"https://www.iea.org/topics/investment","color":"#E9B85C","horizon":"2026–2035","maturity":"Capex / infrastructures","signal":"Très élevé","risk":"Risque d’exécution : raccordements, délais, coûts et rentabilité des capacités."},
    {"sector":"Défense & sécurité","icon":"🛡️","amount":2887,"label":"2 887 Md$","period":"2025","kind":"Dépense observée","summary":"Dépenses militaires mondiales","detail":"Hausse réelle de 2,9 %. L’Europe représente 864 Md$ (+14 %) et l’Asie-Océanie 681 Md$ (+8,1 %).","impact":"Les budgets pluriannuels peuvent soutenir les carnets de commandes, mais les annonces doivent devenir des contrats et des livraisons.","watch":"Budgets votés, commandes fermes, capacités industrielles et reconstitution des stocks.","source":"SIPRI — 27 avril 2026","url":"https://www.sipri.org/media/press-release/2026/global-military-spending-rise-continues-european-and-asian-expenditures-surge","color":"#B7C2CE","horizon":"2025–2030","maturity":"Budgets publics / commandes","signal":"Élevé","risk":"Le budget global ne correspond pas au chiffre d’affaires immédiatement accessible aux industriels."},
    {"sector":"Investissements directs & industrie","icon":"🏭","amount":1600,"label":"1 600 Md$","period":"2025","kind":"Estimation annuelle","summary":"Investissements directs étrangers mondiaux","detail":"Les projets greenfield annoncés dans cinq secteurs stratégiques ont atteint 576 Md$ en 2025, contre 109 Md$ en 2020.","impact":"Montre où les entreprises construisent des capacités productives : usines, semi-conducteurs, data centers et énergie.","watch":"Décisions finales d’investissement, chantiers lancés, annulations et pays hôtes.","source":"UNCTAD — World Investment Report 2026","url":"https://unctad.org/publication/world-investment-report-2026-international-investment-turbulent-era","color":"#86BDA7","horizon":"2025–2030","maturity":"Investissement productif","signal":"Élevé","risk":"Les annonces de projets et les flux effectivement déployés peuvent diverger."},
    {"sector":"IA, data centers & semi-conducteurs","icon":"🧠","amount":900,"label":"≈ 900 Md$","period":"2026","kind":"Estimation annuelle","summary":"Investissement mondial lié à l’IA estimé","detail":"Projection de 5 500 Md$ cumulés d’ici 2030. Les dépenses des grands hyperscalers sont le moteur principal.","impact":"La dépense logicielle entraîne des besoins physiques en GPU, mémoire, centres de données, refroidissement et électricité.","watch":"Capex annoncé vs réalisé, commandes GPU/mémoire, raccordements électriques et cash-flow des hyperscalers.","source":"J.P. Morgan — 19 octobre 2026","url":"https://www.jpmorgan.com/insights/global-research/artificial-intelligence/ai-capex","color":"#8BB8E4","horizon":"2026–2030","maturity":"Capex privé / infrastructures","signal":"Très élevé","risk":"Risque de surcapacité, de financement et de monétisation des investissements."},
    {"sector":"Agroalimentaire & sécurité alimentaire","icon":"🌾","amount":630,"label":"≈ 630 Md$","period":"Estimation 2026","kind":"Capacité potentielle, non flux réalisé","summary":"Capacité potentielle de financement privé","detail":"Le financement climatique nécessaire aux systèmes agroalimentaires est estimé à 1 100 Md$ par an.","impact":"L’irrigation, le stockage, les engrais, les semences et la logistique déterminent la résilience alimentaire.","watch":"Financement déployé, eau, stocks céréaliers, engrais et capacités de stockage.","source":"Food and Land Use Coalition — rapport 2026","url":"https://www.foodandlandusecoalition.org/knowledge-hub/re-wire-agricultural-resilience/","color":"#B7C987","horizon":"2026–2035","maturity":"Capacité / besoin de financement","signal":"Structurel","risk":"Ne pas confondre capacité de financement et investissement réellement engagé."},
    {"sector":"Capital-risque & startups","icon":"🚀","amount":427.1,"label":"427,1 Md$","period":"2025","kind":"Investissement observé","summary":"Investissement mondial en venture capital","detail":"L’IA a capté 258,7 Md$, soit 61 % du total. Les financements se concentrent dans les méga-tours.","impact":"Le capital-risque finance les futurs acteurs technologiques, mais une levée n’est pas encore une capacité rentable.","watch":"Part de l’IA, financement early-stage, sorties/IPO et taux de survie des startups.","source":"OCDE — 17 février 2026","url":"https://www.oecd.org/en/about/news/announcements/2026/02/ai-firms-capture-61-percent-of-global-venture-capital-in-2025.html","color":"#B9A5D8","horizon":"2025–2028","maturity":"Capital privé / VC","signal":"Concentré","risk":"Concentration dans l’IA et les méga-tours ; valorisations sensibles aux taux et à la liquidité."},
    {"sector":"Santé, biotech & pharmacie","icon":"🧬","amount":60,"label":"60 Md$","period":"2025","kind":"Investissement observé","summary":"Capital-risque mondial dans la santé","detail":"En hausse par rapport à 45,4 Md$ en 2024. HSBC anticipait 65–70 Md$ en 2026.","impact":"Les capitaux ciblent biotechs, dispositifs médicaux et entreprises proches de la commercialisation ; les cycles restent longs.","watch":"Essais cliniques, autorisations, financement late-stage et capacités de production.","source":"HSBC Innovation Banking — 13 janvier 2026","url":"https://www.about.us.hsbc.com/newsroom/press-releases/healthcare-venture-markets-regain-momentum-as-capital-returns","color":"#D9A8BC","horizon":"2025–2035+","maturity":"VC / innovation médicale","signal":"En reprise","risk":"Risque clinique, réglementaire et de refinancement sur des cycles longs."},
]

def _metric_card(label, value, sub, accent):
    st.markdown(f"""
    <div style="background:#14283B;border:1px solid #2D465D;border-top:4px solid {accent};border-radius:13px;padding:15px 16px;min-height:124px">
      <div style="font-size:11px;letter-spacing:1.1px;text-transform:uppercase;color:#AFC1D1;font-weight:800">{label}</div>
      <div style="font-size:29px;font-weight:850;color:#FFFFFF;margin:7px 0 2px">{value}</div>
      <div style="font-size:12px;color:#C5D2DD">{sub}</div>
    </div>""", unsafe_allow_html=True)

def _chart_theme(fig, height=300):
    fig.update_layout(
        height=height, margin=dict(l=12,r=12,t=34,b=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#26384A"), hoverlabel=dict(bgcolor="#14283B",font_color="white"),
        showlegend=False
    )
    return fig

def render_financial_intelligence():
    st.markdown("""
    <style>
    .fi-hero{background:linear-gradient(120deg,#101E30 0%,#183B52 58%,#27645F 100%);padding:27px 29px;border-radius:17px;color:#fff;margin:4px 0 18px}
    .fi-kicker{font-size:11px;letter-spacing:2px;text-transform:uppercase;color:#B7D8D1;font-weight:800}
    .fi-hero h2{color:#fff;font-size:30px;line-height:1.12;margin:9px 0}
    .fi-hero p{color:#DFEAF0;font-size:14px;max-width:900px;margin:0}
    .fi-section{font-size:20px;font-weight:850;color:#1D3448;margin:20px 0 5px}
    .fi-sub{font-size:12px;color:#697B8B;margin-bottom:12px}
    .fi-panel{background:#F4F7FA;border:1px solid #E0E7EE;border-radius:14px;padding:15px 16px;margin-bottom:10px}
    .fi-chip{display:inline-block;border-radius:20px;padding:4px 9px;background:#E2EBF1;color:#30485D;font-size:10px;font-weight:800;letter-spacing:.4px}
    .fi-callout{background:#EAF2F1;border-left:4px solid #39877A;border-radius:8px;padding:13px 15px;color:#203A42;font-size:13px;margin:8px 0}
    .fi-risk{background:#FFF4E6;border-left:4px solid #D69B3C;border-radius:8px;padding:12px 14px;color:#5A4324;font-size:12px}
    .fi-sector{background:#fff;border:1px solid #E3E9EF;border-radius:12px;padding:13px;min-height:130px}
    .fi-sector .name{font-size:13px;font-weight:800;color:#24384B}
    .fi-sector .value{font-size:23px;font-weight:850;color:#1B3448;margin:8px 0 3px}
    .fi-sector .meta{font-size:10px;color:#738292;text-transform:uppercase;letter-spacing:.5px}
    </style>
    <div class="fi-hero">
      <div class="fi-kicker">Follow the Money / Financial Intelligence · Capital Allocation Radar</div>
      <h2>Où le capital se déploie — et où il peut créer de la valeur</h2>
      <p>Une lecture d’allocation du capital, pas un simple catalogue de montants : distinguer les dépenses réellement observées,
      les budgets, les capex projetés et les capacités théoriques ; identifier les goulets d’étranglement, les bénéficiaires potentiels
      et les risques de mauvaise allocation.</p>
    </div>
    """, unsafe_allow_html=True)

    # Executive dashboard: values intentionally labelled by their nature, never summed.
    cols=st.columns(4,gap="medium")
    metrics=[
        ("ÉNERGIE","3 400 Md$","Investissement attendu · 2026","#E9B85C"),
        ("DÉFENSE","2 887 Md$","Dépenses observées · 2025","#AAB8C8"),
        ("VC MONDIAL","427,1 Md$","Investissements observés · 2025","#B9A5D8"),
        ("FDI GREENFIELD","576 Md$","Projets stratégiques annoncés · 2025","#86BDA7"),
    ]
    for col,m in zip(cols,metrics):
        with col: _metric_card(*m)
    st.markdown('<div class="fi-callout"><b>Lecture du moment :</b> le thème transversal est la conversion du capital financier en capacités physiques — électricité, réseaux, data centers, semi-conducteurs, défense et chaînes d’approvisionnement. Le point critique n’est plus seulement le montant annoncé, mais la vitesse d’exécution et le rendement économique final.</div>',unsafe_allow_html=True)

    st.markdown('<div class="fi-section">01 · Les flux que l’on peut réellement comparer</div><div class="fi-sub">Chaque graphique reste dans son propre périmètre statistique. Aucune addition entre secteurs hétérogènes.</div>',unsafe_allow_html=True)
    c1,c2=st.columns(2,gap="large")
    with c1:
        st.markdown("**⚡ Énergie — allocation de l’investissement 2026**")
        energy=go.Figure(go.Bar(
            x=[2200,1200],y=["Énergies propres, réseaux & électrification","Combustibles fossiles"],
            orientation="h",marker_color=["#49A58D","#D9A04B"],
            text=["2 200 Md$","1 200 Md$"],textposition="inside",
            hovertemplate="%{y}<br>%{x:,.0f} Md$<extra></extra>"
        ))
        energy.update_layout(barmode="stack",xaxis_title="Md$ estimés",yaxis=dict(autorange="reversed"))
        st.plotly_chart(_chart_theme(energy,270),use_container_width=True,key="fi_energy_split")
        st.caption("Source : IEA · prévision annuelle 2026. Les 2 200 Md$ regroupent plusieurs catégories d’énergie propre et d’infrastructure.")
    with c2:
        st.markdown("**🚀 Venture capital — concentration de l’IA en 2025**")
        vc=go.Figure(go.Pie(labels=["IA","Autres secteurs"],values=[258.7,168.4],hole=.68,
            marker=dict(colors=["#8B78B9","#DCE3EA"]),textinfo="label+percent",
            hovertemplate="%{label}<br>%{value:.1f} Md$<extra></extra>"))
        vc.add_annotation(text="<b>61 %</b><br><span style='font-size:10px'>du VC</span>",x=.5,y=.5,showarrow=False,font=dict(size=21,color="#20384C"))
        st.plotly_chart(_chart_theme(vc,270),use_container_width=True,key="fi_vc_ai_share")
        st.caption("Source : OCDE · 427,1 Md$ au total ; 258,7 Md$ attribués aux entreprises IA.")
    c3,c4=st.columns(2,gap="large")
    with c3:
        st.markdown("**🛡️ Dépenses militaires — répartition géographique 2025**")
        defence=go.Figure(go.Bar(x=["Europe","Asie-Océanie","Autres régions"],y=[864,681,1342],
            marker_color=["#7C9BB5","#AAB8C8","#D5DCE3"],text=["864","681","1 342"],textposition="outside",
            hovertemplate="%{x}<br>%{y:,.0f} Md$<extra></extra>"))
        defence.update_layout(yaxis_title="Md$ dépensés",xaxis_title="")
        st.plotly_chart(_chart_theme(defence,270),use_container_width=True,key="fi_defence_regions")
        st.caption("Source : SIPRI · total mondial 2 887 Md$. « Autres régions » est le solde calculé.")
    with c4:
        st.markdown("**🏭 Projets greenfield stratégiques — évolution**")
        green=go.Figure(go.Bar(x=["2020","2025"],y=[109,576],marker_color=["#C8D3DC","#398F7C"],
            text=["109 Md$","576 Md$"],textposition="outside",
            hovertemplate="%{x}<br>%{y:,.0f} Md$<extra></extra>"))
        green.update_layout(yaxis_title="Md$ annoncés",xaxis_title="")
        st.plotly_chart(_chart_theme(green,270),use_container_width=True,key="fi_greenfield_growth")
        st.caption("Source : UNCTAD · projets annoncés dans cinq secteurs stratégiques, pas investissements tous réalisés.")

    st.markdown('<div class="fi-section">02 · Radar sectoriel — intensité, maturité et risque</div><div class="fi-sub">Les montants servent de repères de taille ; le statut décrit la nature de la donnée, pas une recommandation d’achat.</div>',unsafe_allow_html=True)
    for start in range(0,len(SIGNALS),4):
        cols=st.columns(4,gap="small")
        for col,item in zip(cols,SIGNALS[start:start+4]):
            with col:
                st.markdown(f"""<div class="fi-sector" style="border-top:4px solid {item['color']}">
                <div class="name">{item['icon']} {item['sector']}</div>
                <div class="value">{item['label']}</div>
                <div class="meta">{item['kind']} · {item['period']}</div></div>""",unsafe_allow_html=True)

    st.markdown('<div class="fi-section">03 · Analyse d’un secteur : de l’allocation au bénéficiaire économique</div>',unsafe_allow_html=True)
    selected=st.selectbox("Choisir un secteur à examiner",options=[x["sector"] for x in SIGNALS],key="fi_selected_sector")
    item=next(x for x in SIGNALS if x["sector"]==selected)
    left,right=st.columns([1.05,1.4],gap="large")
    with left:
        st.markdown(f"""<div style="background:linear-gradient(145deg,#172C40,#28516A);padding:20px;border-radius:14px;color:white">
        <div style="font-size:12px;color:#B8D0DD;font-weight:800;letter-spacing:1px">{item['horizon']} · {item['signal']}</div>
        <div style="font-size:22px;font-weight:850;margin:10px 0 3px">{item['icon']} {item['sector']}</div>
        <div style="font-size:31px;font-weight:900">{item['label']}</div>
        <div style="font-size:12px;color:#D6E2E9">{item['maturity']} · {item['period']}</div>
        </div>""",unsafe_allow_html=True)
        st.markdown("**Nature exacte du chiffre**")
        st.write(item["detail"])
        st.markdown(f"[Source primaire / institutionnelle ↗]({item['url']})")
    with right:
        st.markdown("**Pourquoi les capitaux se dirigent ici**")
        st.write(item["impact"])
        st.markdown("**Indicateurs avancés à suivre**")
        st.write(item["watch"])
        st.markdown(f'<div class="fi-risk"><b>Risque de conversion :</b> {item["risk"]}</div>',unsafe_allow_html=True)


    # New drill-down: where capital goes by sub-sector and destination country.
    # Keep reported allocations separate from country context and unallocated totals.
    ALLOCATION_MAP = {
        "Énergie & réseaux": {
            "status": "Ventilation mondiale par grandes catégories disponible ; ventilation pays/sous-secteur détaillée à compléter depuis l’IEA Data Explorer.",
            "rows": [
                ["Énergies propres", "Renouvelables, nucléaire, efficacité énergétique", "Monde — ventilation pays non publiée dans le chiffre synthétique", "≈ 2 200 Md$ (agrégat incluant réseaux, stockage et électrification)", "Prévision 2026"],
                ["Réseaux & électrification", "Réseaux de transport/distribution, transformateurs, câbles, raccordements", "Monde — pays non ventilés dans l’agrégat cité", "Inclus dans les 2 200 Md$ ; montant isolé non fourni", "Prévision 2026"],
                ["Stockage", "Batteries et stockage stationnaire", "Monde — pays non ventilés dans l’agrégat cité", "Inclus dans les 2 200 Md$ ; montant isolé non fourni", "Prévision 2026"],
                ["Combustibles fossiles", "Pétrole, gaz, charbon et chaînes associées", "Monde — pays non ventilés dans l’agrégat cité", "≈ 1 200 Md$", "Prévision 2026"],
            ],
            "sources": [("IEA — World Energy Investment 2026", "https://www.iea.org/reports/world-energy-investment-2026")],
            "note": "Ne pas répartir artificiellement les 2 200 Md$ entre pays ou sous-secteurs : l’agrégat de la page d’accueil IEA les regroupe. Le Data Explorer IEA doit être la source du futur détail pays/technologie."
        },
        "Défense & sécurité": {
            "status": "Dépenses nationales observées, mais pas une ventilation mondiale fiable par type d’équipement dans le total SIPRI.",
            "rows": [
                ["Forces armées & fonctionnement", "Personnel, opérations, maintenance, infrastructures", "États-Unis", "954 Md$", "Dépense militaire nationale 2025"],
                ["Modernisation & capacités", "Équipements, R&D, plateformes et systèmes", "Chine", "336 Md$ (estimation SIPRI)", "Dépense militaire nationale 2025"],
                ["Modernisation & capacités", "Équipements, R&D, plateformes et systèmes", "Russie", "190 Md$ (estimation SIPRI)", "Dépense militaire nationale 2025"],
                ["Forces armées & équipements", "Personnel, équipements, infrastructures", "Allemagne", "114 Md$", "Dépense militaire nationale 2025"],
                ["Forces armées & équipements", "Personnel, équipements, infrastructures", "Inde", "92,1 Md$", "Dépense militaire nationale 2025"],
                ["Agrégat régional", "Toutes catégories de dépenses militaires", "Europe", "864 Md$", "Dépense régionale 2025"],
                ["Agrégat régional", "Toutes catégories de dépenses militaires", "Asie & Océanie", "681 Md$", "Dépense régionale 2025"],
            ],
            "sources": [("SIPRI — Trends in World Military Expenditure 2025", "https://www.sipri.org/publications/2026/sipri-fact-sheets/trends-world-military-expenditure-2025")],
            "note": "Le budget national n’est pas un montant investi dans un sous-secteur industriel précis. Il finance aussi salaires, opérations et entretien. Les montants pays ne sont donc pas des commandes adressables aux entreprises de défense."
        },
        "Investissements directs & industrie": {
            "status": "Pays destinataires connus ; la ventilation sectorielle exacte doit être lue dans les tables UNCTAD/OECD, pas déduite des totaux nationaux.",
            "rows": [
                ["Projets greenfield stratégiques", "IA, semi-conducteurs, minéraux critiques, technologies de transition énergétique", "Monde", "576 Md$ de valeur de projets annoncés dans cinq secteurs stratégiques", "Annonces 2025 — pas tous réalisés"],
                ["Investissement direct étranger", "Tous secteurs confondus", "États-Unis", "Plus grand pays destinataire en 2025 ; montant pays à extraire de la base UNCTAD", "Flux FDI 2025"],
                ["Investissement direct étranger", "Tous secteurs confondus", "Singapour", "2e destination mentionnée par UNCTAD", "Flux FDI 2025"],
                ["Investissement direct étranger", "Tous secteurs confondus", "Hong Kong (Chine)", "3e destination mentionnée par UNCTAD", "Flux FDI 2025"],
                ["Investissement direct étranger", "Tous secteurs confondus", "Chine", "4e destination mentionnée par UNCTAD", "Flux FDI 2025"],
                ["Investissement direct étranger", "Tous secteurs confondus", "Brésil", "5e destination mentionnée par UNCTAD", "Flux FDI 2025"],
            ],
            "sources": [("UNCTAD — World Investment Report 2026", "https://unctad.org/topic/investment/foreign-direct-investment-explorer"), ("UNCTAD — World Investment Report 2026 overview", "https://investmentpolicy.unctad.org/publications/1324/world-investment-report-2026-international-investment-in-a-turbulent-era")],
            "note": "Important : les cinq pays sont un classement des destinations FDI tous secteurs confondus. Il ne faut pas les présenter comme les cinq premiers pays de chacun des secteurs stratégiques."
        },
        "IA, data centers & semi-conducteurs": {
            "status": "Le chiffre de capex IA est une estimation agrégée ; le détail géographique et sous-sectoriel doit être relié aux dépenses hyperscalers et aux projets greenfield.",
            "rows": [
                ["Infrastructure de calcul", "GPU, accélérateurs IA, serveurs et mémoire", "États-Unis, Taïwan, Corée du Sud — pôles industriels suivis, pas allocation chiffrée du total", "Non ventilé dans l’estimation agrégée", "Capex / chaîne d’approvisionnement"],
                ["Data centers", "Bâtiments, terrains, alimentation électrique, refroidissement", "États-Unis en tête des projets annoncés ; autres pays non ventilés dans le chiffre agrégé", "Non ventilé dans l’estimation agrégée", "Investissement physique"],
                ["Semi-conducteurs", "Fonderies, packaging avancé, équipements de fabrication", "États-Unis, Taïwan, Corée du Sud, Japon et Europe — destinations/écosystèmes à suivre", "Non ventilé dans l’estimation agrégée", "Capacité industrielle"],
                ["Électricité & réseau", "Production, transformateurs, raccordements, stockage", "Pays d’implantation des data centers ; pas de ventilation nationale consolidée dans le chiffre cité", "Non ventilé dans l’estimation agrégée", "Capex induit"],
            ],
            "sources": [("UNCTAD — World Investment Report 2026", "https://investmentpolicy.unctad.org/publications/1324/world-investment-report-2026-international-investment-in-a-turbulent-era"), ("IEA — World Energy Investment 2026", "https://www.iea.org/reports/world-energy-investment-2026")],
            "note": "Les pays cités identifient des pôles de production ou d’accueil de projets, pas une ventilation monétaire auditée des 900 Md$ estimés. Le chiffre IA ne doit pas être additionné aux capex énergie ou data centers s’il les recouvre déjà."
        },
        "Agroalimentaire & sécurité alimentaire": {
            "status": "Le montant de la carte initiale est une capacité/besoin potentiel, pas une dépense observée ventilée par pays.",
            "rows": [
                ["Irrigation & gestion de l’eau", "Irrigation efficiente, pompage, réutilisation de l’eau", "Pays à stress hydrique ; ventilation financière non disponible dans la source synthétique", "Pas de montant réalisé vérifié", "Besoin/capacité de financement"],
                ["Intrants agricoles", "Engrais, semences, protection des cultures", "Principaux bassins agricoles mondiaux ; pas d’allocation chiffrée disponible", "Pas de montant réalisé vérifié", "Besoin/capacité de financement"],
                ["Stockage & logistique", "Silos, chaîne du froid, transport et transformation", "Pays importateurs nets et grands producteurs ; pas d’allocation chiffrée disponible", "Pas de montant réalisé vérifié", "Besoin/capacité de financement"],
                ["Financement des systèmes alimentaires", "Résilience des chaînes alimentaires et agriculture durable", "Monde", "≈ 1 100 Md$/an de besoin de financement cité ; ≠ investissement réalisé", "Besoin annuel estimé"],
            ],
            "sources": [("Food and Land Use Coalition — Agricultural resilience", "https://www.foodandlandusecoalition.org/knowledge-hub/re-wire-agricultural-resilience/")],
            "note": "Ne pas afficher les 630 Md$ de la carte comme du capital effectivement investi : la donnée initiale est une capacité potentielle et la source mentionne un besoin de financement."
        },
        "Capital-risque & startups": {
            "status": "Ventilation par secteur disponible pour l’IA ; ventilation pays homogène non fournie par le chiffre global OCDE cité.",
            "rows": [
                ["IA", "Modèles, logiciels, infrastructure IA et applications", "Monde", "258,7 Md$ — 61 % du VC mondial", "Investissement VC observé 2025"],
                ["Autres secteurs", "Santé, climat, fintech, industrie, mobilité et autres", "Monde", "168,4 Md$ (solde calculé : 427,1 - 258,7)", "Investissement VC observé 2025"],
                ["VC — géographie", "Tous secteurs confondus", "États-Unis, Europe, Asie", "Pas de ventilation pays attachée au chiffre OCDE de 427,1 Md$", "Donnée géographique à compléter"],
            ],
            "sources": [("OCDE — AI firms capture 61% of global VC in 2025", "https://www.oecd.org/en/about/news/announcements/2026/02/ai-firms-capture-61-percent-of-global-venture-capital-in-2025.html")],
            "note": "Le solde « autres secteurs » est un calcul arithmétique. Ne pas lui attribuer une ventilation pays sans base VC géographique distincte."
        },
        "Santé, biotech & pharmacie": {
            "status": "Le montant de 60 Md$ concerne le capital-risque santé ; il ne représente pas l’ensemble des investissements mondiaux en santé.",
            "rows": [
                ["Biotechnologie thérapeutique", "Découverte de médicaments, plateformes biologiques, essais cliniques", "États-Unis et Europe comme grands écosystèmes VC ; allocation pays du total non publiée dans le chiffre cité", "Non ventilé", "VC santé 2025"],
                ["Dispositifs médicaux", "Imagerie, implants, robotique et équipements", "Amérique du Nord, Europe et Asie développée — géographie indicative, non chiffrée", "Non ventilé", "VC santé 2025"],
                ["Diagnostics & services", "Diagnostic, outils de laboratoire, santé numérique et services", "Monde — allocation pays non publiée dans le chiffre cité", "Non ventilé", "VC santé 2025"],
                ["Capital-risque santé", "Toutes sous-catégories santé confondues", "Monde", "60 Md$ en 2025 ; contre 45,4 Md$ en 2024", "VC observé"],
            ],
            "sources": [("HSBC Innovation Banking — Healthcare venture markets", "https://www.about.us.hsbc.com/newsroom/press-releases/healthcare-venture-markets-regain-momentum-as-capital-returns")],
            "note": "La géographie affichée est une cartographie des écosystèmes, pas une répartition chiffrée des 60 Md$. Les données VC ne couvrent ni les dépenses publiques de santé ni les capex des grands laboratoires."
        },
    }

    st.markdown('<div class="fi-section">04 · Où va réellement le capital ? — sous-secteurs & pays</div><div class="fi-sub">Pour chaque famille : destination économique, pays bénéficiaires et montant traçable. Les données non publiées sont explicitement signalées : aucune ventilation n’est inventée.</div>',unsafe_allow_html=True)
    allocation_sector = st.selectbox("Famille d’investissement à détailler", options=list(ALLOCATION_MAP.keys()), key="fi_allocation_sector")
    allocation = ALLOCATION_MAP[allocation_sector]
    st.info(allocation["status"])
    allocation_df = pd.DataFrame(allocation["rows"], columns=["Sous-secteur / flux", "Activité financée exactement", "Pays / géographie", "Capital / montant", "Nature de la donnée"])
    st.dataframe(allocation_df, use_container_width=True, hide_index=True, column_config={
        "Sous-secteur / flux": st.column_config.TextColumn(width="medium"),
        "Activité financée exactement": st.column_config.TextColumn(width="large"),
        "Pays / géographie": st.column_config.TextColumn(width="medium"),
        "Capital / montant": st.column_config.TextColumn(width="medium"),
        "Nature de la donnée": st.column_config.TextColumn(width="medium"),
    })
    st.markdown("**Sources institutionnelles**")
    st.markdown(" · ".join([f"[{label} ↗]({url})" for label, url in allocation["sources"]]))
    st.caption("Méthode : distinguer flux réalisés, dépenses observées, annonces greenfield, estimations, besoins et capacités potentielles. Une destination géographique indicative n’est jamais présentée comme une allocation monétaire. Les montants de familles différentes peuvent se chevaucher.")
    st.warning(allocation["note"])

    st.markdown('<div class="fi-section">05 · La grille de lecture investisseur</div>',unsafe_allow_html=True)
    matrix=pd.DataFrame([
        ["IA & data centers","GPU / mémoire, refroidissement, réseaux électriques, fournisseurs d’énergie","Capex réellement exécuté, taux d’utilisation, cash-flow, financement","Surinvestissement, obsolescence, valorisations"],
        ["Énergie & réseaux","Équipements électriques, stockage, production pilotable, ingénierie","Commandes fermes, raccordements, délais de livraison, régulation","Retards, coûts de financement, dépendance aux subventions"],
        ["Défense & sécurité","Électronique, munitions, défense aérienne, cyber, maintenance","Budgets votés, contrats attribués, carnet de commandes, cadence","Annonces non contractualisées, dépendance politique"],
        ["Industrie & FDI","Construction, automatisation, semi-conducteurs, logistique","Décisions finales, permis, travaux commencés, taux d’annulation","Projets reportés, risque pays, fragmentation commerciale"],
        ["Santé & biotech","Bioproduction, dispositifs, diagnostics, fournisseurs spécialisés","Résultats d’essais, autorisations, financement late-stage","Échec clinique, réglementation, dilution"],
        ["Agroalimentaire","Irrigation, stockage, engrais, semences, chaîne du froid","Capacité financée, stress hydrique, stocks, prix des intrants","Risque climatique, politique commerciale, accès à l’eau"],
    ],columns=["Thème","Bénéficiaires potentiels (exposition indirecte)","Preuves à rechercher","Risque principal"])
    st.dataframe(matrix,use_container_width=True,hide_index=True)
    st.markdown('<div class="fi-callout"><b>Règle de décision :</b> ne pas investir sur la seule taille d’un marché. Chercher la preuve que l’argent est engagé, que les commandes existent, que les capacités sont livrées et que les marges restent positives. Les entreprises situées sur les goulets d’étranglement peuvent capter davantage de valeur que les acteurs les plus visibles.</div>',unsafe_allow_html=True)
    st.warning("Limites méthodologiques : les chiffres proviennent de publications de nature et de dates différentes. Les dépenses militaires, capex énergétiques, investissements directs, VC et capacités potentielles ne sont ni additionnables ni directement comparables. Les données de cette page sont des repères documentaires, pas des flux temps réel ni des signaux automatiques d’achat/vente.")
    st.caption("Financial Intelligence · version renforcée · chiffres et sources hérités de la base documentaire existante ; à actualiser à chaque nouvelle publication des organismes cités.")
