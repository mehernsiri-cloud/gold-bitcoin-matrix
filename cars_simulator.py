"""Toyota RAV4 HEV/PHEV decision dashboard.

Commercial listing data and all ownership assumptions are editable.
Resale, maintenance and battery reserves are planning scenarios, not forecasts.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

LACENTRALE_URL = "https://www.lacentrale.fr/listing?autoviza=true&customerFamilyCodes=PROFESSIONNEL%2CCOURTIER_AUTOMOBILE%2CINTERMEDIAIRE&dptCp=78140&energies=plug_hyb%2Chyb&firstHand=true&gearbox=AUTO&goodDealBadges=VERY_GOOD_DEAL%2CGOOD_DEAL&makesModelsCommercialNames=TOYOTA%3A%3ARAV%204%3A%3ARAV%204%20(5E%20GENERATION)&mileageMax=30000&yearMin=2024"
LACENTRALE_HEV_AWD_URL = "https://www.lacentrale.fr/listing?autoviza=true&customerFamilyCodes=PROFESSIONNEL%2CCOURTIER_AUTOMOBILE%2CINTERMEDIAIRE&dptCp=78140&energies=hyb&firstHand=true&gearbox=AUTO&goodDealBadges=VERY_GOOD_DEAL%2CGOOD_DEAL&makesModelsCommercialNames=TOYOTA%3A%3ARAV%204&mileageMax=30000&versions=awd-i&yearMin=2024"


def money(value):
    return f"{value:,.0f} €".replace(",", " ")


def render_rav4_simulator():
    st.markdown("---")
    st.header("🚙 Toyota RAV4 — comparateur HEV / PHEV")
    st.caption(
        "Comparaison des annonces réelles, simulation du coût total de possession et recommandation "
        "selon tes priorités. Les hypothèses sont modifiables ; le résultat n'est pas une cote ni une prévision."
    )

    st.link_button("🔎 Ouvrir directement ma recherche La Centrale (Yvelines · RAV4 HEV/PHEV)", LACENTRALE_URL, use_container_width=True)
    st.link_button("🔎 Recherche ciblée RAV4 HEV AWD-i (Yvelines · 2024+ · ≤30 000 km)", LACENTRALE_HEV_AWD_URL, use_container_width=True)
    st.caption("Filtres du lien : professionnels/courtiers, 78 (78140), HEV ou PHEV, première main, automatique, "
               "bonnes affaires, RAV4 5e génération, moins de 30 000 km, année 2024 ou plus.")

    # Reserve the first screen position for the decision summary; populate it once all filters are calculated.
    summary_placeholder = st.container()

    st.subheader("1. Les deux annonces de référence")
    left, right = st.columns(2)
    with left:
        st.markdown("### HEV · 218 ch · 2WD")
        st.markdown("**Dynamic Business**")
        hev_price = st.number_input("Prix affiché HEV (€)", 20000, 80000, 39000, 500, key="sim_hev_price_v2")
        st.caption("39 000 € · 20 km · 04/09/2025 · 9 CV · 126 g CO₂/km")
        st.caption("Hybride non rechargeable · traction avant")
    with right:
        st.markdown("### PHEV · 306 ch · AWD-i")
        st.markdown("**Collection**")
        phev_price = st.number_input("Prix affiché PHEV (€)", 20000, 80000, 50000, 500, key="sim_phev_price_v2")
        st.caption("50 000 € · 14 326 km · 31/12/2025 · 10 CV · 22 g CO₂/km")
        st.caption("Rechargeable · batterie 18,1 kWh · autonomie annoncée 75 km")
    premium = phev_price - hev_price
    st.info(
        f"Écart de prix actuel : **{money(premium)}** ({premium / max(1, hev_price) * 100:.1f} %). "
        "Le PHEV est aussi mieux équipé (Collection) et AWD-i, alors que le HEV est Dynamic Business et 2WD. "
        "Les 14 306 km supplémentaires du PHEV sont intégrés dans l'appréciation de l'annonce, "
        "mais pas convertis artificiellement en coût."
    )

    st.subheader("2. Filtres liés à ton usage")
    c1, c2, c3 = st.columns(3)
    with c1:
        annual_km = st.number_input("Kilométrage annuel total", 1000, 30000, 4500, 250, key="sim_annual_km_v2")
        local_km = st.number_input("Dont trajets locaux / an", 0, 20000, 1500, 100, key="sim_local_km_v2")
    with c2:
        years = st.select_slider("Durée de détention", options=[3, 4, 5, 6, 7, 8, 9, 10, 12, 15], value=8, key="sim_years_v2")
        home_charging = st.checkbox("Recharge à domicile disponible ou installation prévue", value=False, key="sim_home_charge_v3")
    with c3:
        priority = st.select_slider("Priorité de décision", options=["Budget", "Équilibrée", "Confort / polyvalence"], value="Équilibrée", key="sim_priority_v2")
        awd_need = st.checkbox("Transmission intégrale importante", value=False, key="sim_awd_v2")

    ev_share = st.slider(
        "Part des trajets locaux réalisables en électrique (%)", 0, 100, 80, 5,
        key="sim_ev_local_v2",
        help="Appliquée uniquement aux kilomètres locaux et seulement si une recharge régulière est disponible."
    )
    phev_ev_km = min(local_km, annual_km) * ev_share / 100 if home_charging else 0
    phev_ev_km = min(phev_ev_km, annual_km)
    long_km = max(0, annual_km - min(local_km, annual_km))
    st.metric("Kilomètres électriques PHEV estimés / an", f"{phev_ev_km:,.0f} km".replace(",", " "))
    st.caption(
        f"Le modèle considère {long_km:,.0f} km/an de grands trajets en mode hybride essence, "
        "sans recharge en route (voyages Europe et Tunisie).".replace(",", " ")
    )
    if not home_charging:
        st.warning("Sans recharge régulière à domicile, le simulateur ne crédite actuellement aucune économie électrique au PHEV.")

    st.subheader("3. Énergie — hypothèses modifiables")
    charger_installation_quote = st.number_input(
        "Coût de la borne et de son installation (€)", 0, 10000, 1500, 100,
        key="sim_charger_installation_v1",
        help="Coût ponctuel à saisir selon le devis : borne, protections électriques, câblage et pose. Comptabilisé uniquement si la recharge à domicile est activée."
    )
    charger_installation_cost = charger_installation_quote if home_charging else 0
    if home_charging:
        st.caption(f"Le simulateur ajoute {money(charger_installation_cost)} au coût initial du PHEV.")
    else:
        st.caption("La recharge à domicile n'étant pas activée, aucun coût de borne ni économie électrique n'est intégré.")
    c1, c2, c3 = st.columns(3)
    with c1:
        fuel = st.number_input("Essence (€/L)", 0.80, 4.00, 1.95, 0.05, key="sim_fuel_v2")
        power = st.number_input("Électricité domicile (€/kWh)", 0.05, 1.00, 0.25, 0.01, key="sim_power_v2")
    with c2:
        hev_l100 = st.number_input("HEV (L/100 km)", 3.0, 12.0, 5.5, 0.1, key="sim_hev_l100_v2")
        phev_hv_l100 = st.number_input("PHEV batterie déchargée / mode hybride (L/100 km)", 3.0, 14.0, 6.2, 0.1, key="sim_phev_l100_v2")
    with c3:
        phev_kwh100 = st.number_input("PHEV électrique (kWh/100 km)", 12.0, 35.0, 20.0, 0.5, key="sim_phev_kwh_v2")
        losses = st.slider("Pertes à la recharge (%)", 0, 25, 10, 1, key="sim_losses_v2")

    hev_energy_year = annual_km * hev_l100 / 100 * fuel
    phev_energy_year = (annual_km - phev_ev_km) * phev_hv_l100 / 100 * fuel + phev_ev_km * phev_kwh100 / 100 * power * (1 + losses / 100)

    st.subheader("4. Coûts annexes et revente")
    st.caption("Valeurs de travail à remplacer par tes devis et les cotes de revente observées.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**HEV**")
        hev_maint = st.number_input("Entretien annuel HEV (€)", 0, 3000, 500, 50, key="sim_maint_h_v2")
        hev_ins = st.number_input("Assurance annuelle HEV (€)", 0, 3000, 750, 50, key="sim_ins_h_v2")
        hev_resale_pct = st.slider("Valeur de revente HEV (% du prix)", 5, 90, 48, key="sim_resale_h_v2")
        hev_battery = st.number_input("Provision risque batterie HEV sur la période (€)", 0, 15000, 500, 100, key="sim_batt_h_v2")
    with c2:
        st.markdown("**PHEV**")
        phev_maint = st.number_input("Entretien annuel PHEV (€)", 0, 3000, 550, 50, key="sim_maint_p_v2")
        phev_ins = st.number_input("Assurance annuelle PHEV (€)", 0, 3000, 850, 50, key="sim_ins_p_v2")
        phev_resale_pct = st.slider("Valeur de revente PHEV (% du prix)", 5, 90, 43, key="sim_resale_p_v2")
        phev_battery = st.number_input("Provision risque batterie PHEV sur la période (€)", 0, 25000, 1200, 100, key="sim_batt_p_v2")

    hev_resale = hev_price * hev_resale_pct / 100
    phev_resale = phev_price * phev_resale_pct / 100
    hev_total = hev_price + years * (hev_energy_year + hev_maint + hev_ins) + hev_battery - hev_resale
    phev_total = phev_price + charger_installation_cost + years * (phev_energy_year + phev_maint + phev_ins) + phev_battery - phev_resale
    distance_total = max(1, annual_km * years)
    hev_per_km = hev_total / distance_total
    phev_per_km = phev_total / distance_total
    tco_gap = phev_total - hev_total


    # Compute the financial winner and weighted recommendation BEFORE rendering the summary.
    cheapest = "HEV" if hev_total <= phev_total else "PHEV"
    financial_hev = 100 if cheapest == "HEV" else 0
    financial_phev = 100 - financial_hev
    ev_score = 85 if home_charging and phev_ev_km >= 1000 else 65 if home_charging and phev_ev_km > 0 else 20
    polyvalence_phev = 90 if awd_need else 78
    polyvalence_hev = 55 if awd_need else 70
    if priority == "Budget":
        weight_cost, weight_use = 0.80, 0.20
    elif priority == "Confort / polyvalence":
        weight_cost, weight_use = 0.30, 0.70
    else:
        weight_cost, weight_use = 0.60, 0.40
    hev_use = polyvalence_hev
    phev_use = (ev_score + polyvalence_phev) / 2
    hev_score = weight_cost * financial_hev + weight_use * hev_use
    phev_score = weight_cost * financial_phev + weight_use * phev_use
    overall = "PHEV" if phev_score > hev_score else "HEV"

    # Render the decision summary in the reserved top position, above all detailed filters.
    with summary_placeholder:
        st.markdown("---")
        st.subheader("🧾 Synthèse de l'achat — décision en un coup d'œil")
        s1, s2 = st.columns(2)
        with s1:
            st.markdown("**RAV4 HEV · 218 ch · 2WD Dynamic Business**")
            st.metric("Prix d'achat", money(hev_price))
            st.caption("2025 · 20 km · 126 g CO₂/km · non rechargeable")
        with s2:
            st.markdown("**RAV4 PHEV · 306 ch · AWD-i Collection**")
            st.metric("Prix d'achat", money(phev_price), f"{money(premium)} vs HEV")
            st.caption("2025 · 14 326 km · 18,1 kWh · 75 km annoncés")
        st.markdown(f"**Écart à l'achat : {money(premium)}** · soit {premium / max(1, hev_price) * 100:.1f} % de plus pour le PHEV.")
        st.markdown(f"**Borne + installation intégrées au PHEV : {money(charger_installation_cost)}**" if home_charging else "**Borne : non comptabilisée (recharge domicile désactivée).**")
        st.markdown("#### Coût total sur la durée de détention")
        r1, r2 = st.columns(2)
        r1.metric(f"TCO HEV · {years} ans", money(hev_total), f"{hev_per_km:.2f} €/km")
        r2.metric(f"TCO PHEV · {years} ans", money(phev_total), f"{phev_per_km:.2f} €/km")
        delta_label = f"{money(abs(tco_gap))} de moins pour le {cheapest}" if abs(tco_gap) >= 1 else "Coûts totaux quasiment identiques"
        st.markdown(f"**Écart de TCO : {delta_label}.**")
        st.markdown("#### Verdict")
        d1, d2 = st.columns(2)
        d1.metric("Moins cher financièrement", cheapest)
        d2.metric("Recommandation selon tes priorités", overall, f"Score {max(hev_score, phev_score):.0f}/100")
        st.info(f"**Décision du simulateur : {overall}.** Le {cheapest} reste le meilleur choix sur le seul coût total. Le verdict pondéré ajoute tes critères d'usage et de polyvalence.")
        st.caption(
            f"Priorité : {priority} · Recharge domicile : {'oui' if home_charging else 'non'} · "
            f"Kilométrage : {annual_km:,} km/an · Électrique PHEV estimé : {phev_ev_km:,.0f} km/an. "
            "Le verdict est calculé à partir des hypothèses visibles plus bas.".replace(",", " ")
        )
        st.warning(
            "Comparaison non strictement équivalente : HEV en 2WD Dynamic Business contre PHEV AWD-i Collection. "
            "Le PHEV a aussi 14 306 km de plus. Vérifier l'historique et l'état de santé de sa batterie avant achat."
        )

    st.subheader("6. Détail du coût total de possession")
    costs = pd.DataFrame({
        "Poste": ["Achat", "Borne + installation", "Énergie", "Entretien", "Assurance", "Provision batterie", "Revente déduite"],
        "HEV": [hev_price, 0, years * hev_energy_year, years * hev_maint, years * hev_ins, hev_battery, -hev_resale],
        "PHEV": [phev_price, charger_installation_cost, years * phev_energy_year, years * phev_maint, years * phev_ins, phev_battery, -phev_resale],
    })
    fig = px.bar(costs.melt(id_vars="Poste", var_name="Motorisation", value_name="Coût (€)"),
                 x="Poste", y="Coût (€)", color="Motorisation", barmode="group",
                 title="Décomposition du coût total de possession")
    fig.update_layout(height=380, legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("6. Verdict selon tes priorités")
    # Transparent 0-100 suitability indicators, separate from the financial TCO.
    financial_hev = 100 if cheapest == "HEV" else 0
    financial_phev = 100 - financial_hev
    ev_score = 85 if home_charging and phev_ev_km >= 1000 else 65 if home_charging and phev_ev_km > 0 else 20
    polyvalence_phev = 90 if awd_need else 78
    polyvalence_hev = 55 if awd_need else 70
    if priority == "Budget":
        weight_cost, weight_use = 0.80, 0.20
    elif priority == "Confort / polyvalence":
        weight_cost, weight_use = 0.30, 0.70
    else:
        weight_cost, weight_use = 0.60, 0.40
    hev_use = polyvalence_hev
    phev_use = (ev_score + polyvalence_phev) / 2
    hev_score = weight_cost * financial_hev + weight_use * hev_use
    phev_score = weight_cost * financial_phev + weight_use * phev_use
    overall = "PHEV" if phev_score > hev_score else "HEV"
    v1, v2 = st.columns(2)
    v1.metric("Score HEV", f"{hev_score:.0f}/100")
    v2.metric("Score PHEV", f"{phev_score:.0f}/100")
    st.markdown(f"### Recommandation pondérée : **{overall}**")
    st.caption(
        f"Pondération choisie : {weight_cost:.0%} coût total / {weight_use:.0%} usage et polyvalence. "
        "Le score d'usage est un indicateur d'aide à la décision, pas un test constructeur."
    )
    if overall == "HEV":
        st.write("Le HEV ressort devant avec cette pondération : prix d'achat nettement inférieur et simplicité d'utilisation. "
                 "Le PHEV ne devient plus intéressant que si tu valorises davantage l'AWD-i, la finition Collection et la conduite électrique.")
    else:
        st.write("Le PHEV ressort devant avec cette pondération : AWD-i, puissance, finition Collection et possibilité de rouler en électrique si tu recharges régulièrement. "
                 "Vérifie impérativement l'état de santé de la batterie et l'historique des 14 326 km.")
    if awd_need:
        st.info("Tu as indiqué que l'AWD-i est important : le HEV de l'annonce est 2WD. Il faudrait aussi rechercher un HEV AWD-i pour une comparaison technique plus équitable.")

    st.subheader("7. Sensibilité au prix de l'essence")
    rows = []
    for price in [1.50, 1.75, 2.00, 2.25, 2.50, 3.00]:
        h_energy = annual_km * hev_l100 / 100 * price
        p_energy = (annual_km - phev_ev_km) * phev_hv_l100 / 100 * price + phev_ev_km * phev_kwh100 / 100 * power * (1 + losses / 100)
        h_tco = hev_price + years * (h_energy + hev_maint + hev_ins) + hev_battery - hev_resale
        p_tco = phev_price + charger_installation_cost + years * (p_energy + phev_maint + phev_ins) + phev_battery - phev_resale
        rows.append({"Essence €/L": price, "TCO HEV (€)": h_tco, "TCO PHEV (€)": p_tco, "Avantage HEV (€)": p_tco - h_tco})
    st.dataframe(pd.DataFrame(rows).style.format({"Essence €/L": "{:.2f}", "TCO HEV (€)": "{:,.0f}", "TCO PHEV (€)": "{:,.0f}", "Avantage HEV (€)": "{:+,.0f}"}),
                 use_container_width=True, hide_index=True)

    st.subheader("8. Contrôles avant décision")
    st.markdown(
        "- **Comparabilité :** génération RAV4 V (2) / XA50 restylée ; les annonces diffèrent en finition et transmission (2WD vs AWD-i).\n"
        "- **PHEV :** demander un rapport de santé batterie, vérifier le câble et la recharge, l'historique d'entretien et la garantie applicable au VIN.\n"
        "- **HEV :** contrôler l'état de la batterie hybride et les conditions de garantie.\n"
        "- **Annonce PHEV :** expliquer les 14 326 km malgré la date de première mise en circulation affichée au 31/12/2025.\n"
        "- **Revente :** tester plusieurs valeurs résiduelles ; la batterie ne doit pas être considérée automatiquement comme à remplacer à 10 ou 15 ans."
    )
