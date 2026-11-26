"""Toyota RAV4 HEV/PHEV decision dashboard.

Commercial listing data and all ownership assumptions are editable.
Resale, maintenance and battery reserves are planning scenarios, not forecasts.
"""
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import date

LACENTRALE_URL = "https://www.lacentrale.fr/listing?autoviza=true&customerFamilyCodes=PROFESSIONNEL%2CCOURTIER_AUTOMOBILE%2CINTERMEDIAIRE&energies=not_plug_hyb%2Cplug_hyb&firstHand=true&gearbox=AUTO&goodDealBadges=VERY_GOOD_DEAL%2CGOOD_DEAL&makesModelsCommercialNames=TOYOTA%3A%3ARAV%204&mileageMax=55000&sortBy=priceAsc&versions=awd-i&yearMax=2025&yearMin=2022"


def money(value):
    return f"{value:,.0f} €".replace(",", " ")


def render_rav4_simulator():
    st.markdown("---")
    st.header("🚙 Toyota RAV4 — comparateur HEV / PHEV")
    st.caption(
        "Comparaison des annonces réelles, simulation du coût total de possession et recommandation "
        "selon tes priorités. Les hypothèses sont modifiables ; le résultat n'est pas une cote ni une prévision."
    )

    st.link_button("🔎 Ma recherche principale La Centrale — RAV4 AWD-i HEV/PHEV", LACENTRALE_URL, use_container_width=True)
    st.caption("Filtres du lien : professionnels/courtiers, HEV ou PHEV, première main, automatique, bonnes affaires, "
               "RAV4 AWD-i, jusqu'à 55 000 km, années 2022 à 2025, tri par prix croissant.")

    # Reserve the first screen position for the decision summary; populate it once all filters are calculated.
    summary_placeholder = st.container()

    st.subheader("1. Les deux annonces de référence")
    left, right = st.columns(2)
    with left:
        st.markdown("### HEV · 222 ch · AWD-i")
        st.markdown("**GR SPORT · RAV4 V 2.5 Hybrid**")
        hev_price = st.number_input("Prix affiché HEV (€)", 20000, 80000, 40990, 250, key="sim_hev_price_v2")
        st.caption("40 990 € · 35 407 km · 26/12/2023 · 9 CV · 132 g CO₂/km")
        st.caption("Hybride non rechargeable · AWD-i · première main · rouge / toit noir")
    with right:
        st.markdown("### PHEV · 306 ch · AWD-i")
        st.markdown("**Collection · RAV4 V 2.5 Hybrid Rechargeable**")
        phev_price = st.number_input("Prix affiché PHEV (€)", 20000, 80000, 49990, 250, key="sim_phev_price_v2")
        st.caption("49 990 € · 14 326 km · 31/12/2025 · 10 CV · 22 g CO₂/km")
        st.caption("Hybride rechargeable · batterie 18,1 kWh · autonomie annoncée 75 km · première main · bleu de Prusse / toit noir")
    premium = phev_price - hev_price
    st.info(
        f"Écart de prix actuel : **{money(premium)}** ({premium / max(1, hev_price) * 100:.1f} %). "
        "Les deux annonces sont en AWD-i ; le HEV est en finition GR SPORT et le PHEV en finition Collection. "
        "Le PHEV est plus récent et affiche 21 081 km de moins. Cette différence est visible dans les fiches, "
        "mais n'est pas convertie artificiellement en coût."
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
        hev_battery = st.number_input("Provision risque batterie HEV sur la période (€)", 0, 15000, 500, 100, key="sim_batt_h_v2")
    with c2:
        st.markdown("**PHEV**")
        phev_maint = st.number_input("Entretien annuel PHEV (€)", 0, 3000, 550, 50, key="sim_maint_p_v2")
        phev_ins = st.number_input("Assurance annuelle PHEV (€)", 0, 3000, 850, 50, key="sim_ins_p_v2")
        phev_battery = st.number_input("Provision risque batterie PHEV sur la période (€)", 0, 25000, 1200, 100, key="sim_batt_p_v2")

    # Vehicle age and starting odometer are both carried into the exit-value estimate.
    today = date.today()
    hev_first_reg = date(2023, 12, 26)
    phev_first_reg = date(2025, 12, 31)
    def age_years(first_reg, on_date):
        return max(0.0, (on_date - first_reg).days / 365.2425)
    exit_date = date(today.year + years, today.month, min(today.day, 28 if today.month == 2 else today.day))
    hev_age_now = age_years(hev_first_reg, today)
    phev_age_now = age_years(phev_first_reg, today)
    hev_age_exit = age_years(hev_first_reg, exit_date)
    phev_age_exit = age_years(phev_first_reg, exit_date)
    st.markdown("**Âge et kilométrage intégrés à la valeur de revente**")
    st.caption(f"Âge aujourd'hui : HEV {hev_age_now:.1f} ans · PHEV {phev_age_now:.1f} ans. Âge à la revente dans {years} ans : HEV {hev_age_exit:.1f} ans · PHEV {phev_age_exit:.1f} ans.")
    mileage_c1, mileage_c2, mileage_c3 = st.columns(3)
    with mileage_c1:
        hev_start_km = st.number_input("Kilométrage initial HEV", 0, 300000, 35407, 1000, key="sim_hev_start_km_v1")
    with mileage_c2:
        phev_start_km = st.number_input("Kilométrage initial PHEV", 0, 300000, 14326, 1000, key="sim_phev_start_km_v1")
    with mileage_c3:
        reference_km_year = st.number_input("Kilométrage de référence du marché (km/an)", 5000, 30000, 12000, 1000, key="sim_reference_km_year_v1")
    age_c1, age_c2, age_c3 = st.columns(3)
    with age_c1:
        depreciation_first5 = st.slider("Décote annuelle jusqu'à 5 ans d'âge (%)", 0.0, 20.0, 7.0, 0.5, key="sim_age_dep_first5")
    with age_c2:
        depreciation_after5 = st.slider("Décote annuelle après 5 ans d'âge (%)", 0.0, 15.0, 4.0, 0.5, key="sim_age_dep_after5")
    with age_c3:
        mileage_adjustment_per_10k = st.slider("Impact sur la revente par écart de 10 000 km (%)", 0.0, 10.0, 3.0, 0.5, key="sim_mileage_adjustment_v1")
    st.caption("Le kilométrage de sortie = kilométrage initial de l'annonce + kilométrage annuel × durée. La valeur est ensuite ajustée par rapport à un véhicule de même âge roulant au kilométrage annuel de référence. Hypothèses modifiables, non assimilables à une cote professionnelle.")
    def age_based_resale(price, current_age, holding_years):
        value = float(price)
        for step in range(holding_years):
            age_at_step = current_age + step
            rate = depreciation_first5 if age_at_step < 5 else depreciation_after5
            value *= (1 - rate / 100)
        return value
    def mileage_adjusted_resale(age_value, starting_km, annual_distance, exit_age):
        projected_km = starting_km + annual_distance * years
        reference_exit_km = reference_km_year * exit_age
        difference_10k = (projected_km - reference_exit_km) / 10000
        adjustment_pct = max(-20.0, min(20.0, -difference_10k * mileage_adjustment_per_10k))
        adjusted_value = age_value * (1 + adjustment_pct / 100)
        return adjusted_value, projected_km, reference_exit_km, adjustment_pct
    hev_age_resale = age_based_resale(hev_price, hev_age_now, years)
    phev_age_resale = age_based_resale(phev_price, phev_age_now, years)
    hev_resale, hev_exit_km, hev_reference_exit_km, hev_mileage_adjustment = mileage_adjusted_resale(hev_age_resale, hev_start_km, annual_km, hev_age_exit)
    phev_resale, phev_exit_km, phev_reference_exit_km, phev_mileage_adjustment = mileage_adjusted_resale(phev_age_resale, phev_start_km, annual_km, phev_age_exit)
    resale_c1, resale_c2 = st.columns(2)
    resale_c1.metric("Revente HEV estimée", money(hev_resale), f"{hev_exit_km:,.0f} km à la sortie · {hev_mileage_adjustment:+.1f} % lié au km".replace(",", " "))
    resale_c2.metric("Revente PHEV estimée", money(phev_resale), f"{phev_exit_km:,.0f} km à la sortie · {phev_mileage_adjustment:+.1f} % lié au km".replace(",", " "))
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
    polyvalence_phev = 85 if awd_need else 70
    polyvalence_hev = 85 if awd_need else 70
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
            st.markdown("**RAV4 HEV · 222 ch · AWD-i GR SPORT**")
            st.metric("Prix d'achat", money(hev_price))
            st.caption("26/12/2023 · 35 407 km · 132 g CO₂/km · non rechargeable")
        with s2:
            st.markdown("**RAV4 PHEV · 306 ch · AWD-i Collection**")
            st.metric("Prix d'achat", money(phev_price), f"{money(premium)} vs HEV")
            st.caption("31/12/2025 · 14 326 km · 18,1 kWh · 75 km annoncés · 22 g CO₂/km")
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
            "Comparaison entre deux RAV4 V AWD-i, mais de finitions et d'années différentes : HEV GR SPORT de 2023 "
            "contre PHEV Collection de 2025. Le PHEV est plus récent et moins kilométré ; vérifier son historique "
            "et l'état de santé de sa batterie avant achat."
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
    # Reuse the exact same scores calculated above for the top decision summary.
    # Both reference vehicles are AWD-i: AWD must not create a scoring advantage for either.
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
                 "Le PHEV ne devient plus intéressant que si tu valorises davantage sa finition Collection, sa puissance et la conduite électrique (si tu peux le recharger). L'AWD-i est commun aux deux annonces.")
    else:
        st.write("Le PHEV ressort devant avec cette pondération : puissance, finition Collection et possibilité de rouler en électrique si tu recharges régulièrement. L'AWD-i est commun aux deux annonces. "
                 "Vérifie impérativement l'état de santé de la batterie et l'historique des 14 326 km.")
    if awd_need:
        st.info("Les deux annonces de référence sont bien en AWD-i : le HEV GR SPORT comme le PHEV Collection. La transmission intégrale est donc un critère commun, pas un avantage propre au PHEV ; le score de polyvalence ne favorise aucun des deux sur ce point.")

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
        "- **Comparabilité :** les deux annonces sont en AWD-i ; elles diffèrent en finition (GR SPORT / Collection), année (2023 / 2025) et kilométrage (35 407 / 14 326 km).\n"
        "- **PHEV :** demander un rapport de santé batterie, vérifier le câble et la recharge, l'historique d'entretien et la garantie applicable au VIN.\n"
        "- **HEV :** contrôler l'état de la batterie hybride et les conditions de garantie.\n"
        "- **Annonce PHEV :** expliquer les 14 326 km malgré la date de première mise en circulation affichée au 31/12/2025.\n"
        "- **Revente :** tester plusieurs valeurs résiduelles ; la batterie ne doit pas être considérée automatiquement comme à remplacer à 10 ou 15 ans."
    )


    # C4 Cactus: trade-in versus keeping it for peer-to-peer daily rental.
    st.markdown("---")
    st.header("🚘 Arbitrage C4 Cactus — reprise ou conservation en location")
    st.caption(
        "Comparateur indépendant du choix HEV/PHEV : il mesure l'effet de la reprise sur le financement du RAV4 "
        "et le revenu potentiel d'une conservation avec location journalière (Getaround). "
        "Toutes les hypothèses sont modifiables ; les revenus de location ne sont pas garantis."
    )

    st.subheader("9. Valeur et situation actuelle du C4 Cactus")
    c1, c2, c3 = st.columns(3)
    with c1:
        cactus_year = st.number_input(
            "Année du C4 Cactus", 2008, 2026, 2016, 1, key="cactus_year_v1",
            help="Tu as indiqué « année 2026 » : vérifie et corrige l'année réelle."
        )
        cactus_km = st.number_input("Kilométrage actuel", 0, 300000, 77000, 1000, key="cactus_km_v1")
    with c2:
        cactus_trade_value = st.number_input(
            "Estimation de reprise concessionnaire (€)", 0, 30000, 6000, 250, key="cactus_trade_value_v1",
            help="Valeur provisoire illustrative : remplace-la par une offre écrite du concessionnaire."
        )
        cactus_private_value = st.number_input(
            "Valeur de vente entre particuliers (€)", 0, 35000, 7500, 250, key="cactus_private_value_v1"
        )
    with c3:
        cactus_loan_balance = st.number_input(
            "Capital restant dû sur le C4 (€)", 0, 30000, 0, 250, key="cactus_loan_balance_v1"
        )
        cactus_monthly_cost = st.number_input(
            "Coût fixe mensuel C4 hors surcoût assurance location (€)", 0, 2000, 100, 10,
            key="cactus_monthly_cost_v1"
        )
    st.caption(
        "Les valeurs de reprise et de vente sont des champs de simulation, pas une cote de marché. "
        "La reprise nette tient compte du remboursement éventuel du crédit C4."
    )
    trade_net = cactus_trade_value - cactus_loan_balance
    private_net = cactus_private_value - cactus_loan_balance
    k1, k2, k3 = st.columns(3)
    k1.metric("Reprise nette estimée", money(trade_net))
    k2.metric("Vente particulier nette", money(private_net), f"{money(private_net-trade_net)} vs reprise")
    k3.metric("Kilométrage C4", f"{cactus_km:,} km".replace(",", " "))

    st.subheader("10. Financer le RAV4 : reprise + crédit classique ou crédit ballon")
    c1, c2, c3 = st.columns(3)
    with c1:
        rav4_cash_price = st.number_input(
            "Prix du RAV4 retenu (€)", 20000, 80000, 40990, 250, key="rav4_cash_price_v1"
        )
        cash_down = st.number_input(
            "Apport personnel complémentaire (€)", 0, 30000, 0, 250, key="rav4_cash_down_v1"
        )
    with c2:
        loan_apr = st.number_input(
            "TAEG crédit classique (%)", 0.0, 20.0, 6.5, 0.1, key="rav4_loan_apr_v1",
            help="Hypothèse modifiable : saisir le TAEG réel du devis bancaire."
        )
        loan_months = st.select_slider(
            "Durée crédit classique (mois)", options=[24, 36, 48, 60, 72, 84], value=60,
            key="rav4_loan_months_v1"
        )
    with c3:
        balloon_apr = st.number_input(
            "TAEG crédit ballon Toyota (%)", 0.0, 20.0, 8.56, 0.01, key="rav4_balloon_apr_v1"
        )
        balloon_monthly = st.number_input(
            "Mensualité ballon annoncée (€)", 0, 3000, 454, 10, key="rav4_balloon_monthly_v1"
        )
        balloon_final = st.number_input(
            "Ballon final (€)", 0, 60000, 23936, 100, key="rav4_balloon_final_v1"
        )

    use_trade = st.checkbox(
        "Utiliser la reprise du C4 comme apport sur le RAV4", value=True, key="use_cactus_trade_v1"
    )
    trade_contribution = max(0, trade_net) if use_trade else 0
    loan_principal = max(0, rav4_cash_price - trade_contribution - cash_down)
    monthly_rate = (1 + loan_apr / 100) ** (1 / 12) - 1
    if loan_principal <= 0:
        classic_monthly = 0.0
    elif monthly_rate == 0:
        classic_monthly = loan_principal / loan_months
    else:
        classic_monthly = loan_principal * monthly_rate / (1 - (1 + monthly_rate) ** (-loan_months))
    classic_total = classic_monthly * loan_months + trade_contribution + cash_down
    balloon_principal = loan_principal
    balloon_n = 35
    balloon_monthly_rate = (1 + balloon_apr / 100) ** (1 / 12) - 1
    if balloon_principal <= 0:
        estimated_balloon_monthly = 0.0
    elif balloon_monthly_rate == 0:
        estimated_balloon_monthly = max(0.0, (balloon_principal - balloon_final) / balloon_n)
    else:
        estimated_balloon_monthly = max(0.0, (balloon_principal - balloon_final / (1 + balloon_monthly_rate) ** balloon_n) * balloon_monthly_rate / (1 - (1 + balloon_monthly_rate) ** (-balloon_n)))
    balloon_total = trade_contribution + cash_down + estimated_balloon_monthly * balloon_n + balloon_final
    p1, p2, p3 = st.columns(3)
    p1.metric("Capital du crédit classique", money(loan_principal))
    p2.metric(f"Mensualité classique · {loan_months} mois", money(classic_monthly))
    p3.metric("Total classique (apport inclus)", money(classic_total))
    st.caption(
        f"Crédit ballon estimé : {money(estimated_balloon_monthly)} × 35 + ballon {money(balloon_final)} "
        f"+ reprise/apport {money(trade_contribution + cash_down)} = {money(balloon_total)} au total. "
        f"Référence de l'offre initiale Toyota : {money(balloon_monthly)}/mois avec 8 198 € d'apport et 32 792 € financés. "
        "(hors assurances et frais non indiqués)."
    )
    st.warning(
        "Le calcul du crédit classique est une simulation d'amortissement à taux fixe à partir du TAEG saisi ; "
        "le TAEG n'est pas strictement équivalent à un taux nominal mensuel. Demande une offre normalisée au prêteur. "
        "Le crédit ballon Toyota cité prévoit aussi une limite de restitution de 36 mois et 45 000 km au compteur : "
        "vérifier le kilométrage contractuel, les frais de restitution et les conditions de remise en état."
    )

    st.subheader("11. Garder le C4 et le louer sur Getaround : simulation annuelle")
    st.caption(
        "Saisis un nombre réaliste de jours réellement loués (pas le nombre de jours où l'annonce est publiée). "
        "Les commissions, assurances et règles fiscales Getaround doivent être confirmées dans ton contrat et ton espace hôte."
    )
    r1, r2, r3 = st.columns(3)
    with r1:
        rental_daily_price = st.number_input(
            "Prix moyen facturé par jour (€)", 15, 200, 35, 5, key="cactus_daily_price_v1"
        )
        rental_days = st.number_input(
            "Jours loués par an", 0, 250, 45, 5, key="cactus_rental_days_v1"
        )
        platform_fee_pct = st.slider(
            "Commission / frais plateforme (%)", 0, 50, 25, 1, key="cactus_platform_fee_v1",
            help="Hypothèse de simulation à remplacer par le taux réellement applicable à ton compte."
        )
    with r2:
        rental_insurance_month = st.number_input(
            "Surcoût assurance annuel / mois (€)", 0, 500, 0, 10, key="cactus_rental_insurance_v1",
            help="Saisir uniquement le surcoût d'assurance non déjà inclus dans les coûts fixes C4 ni dans la couverture plateforme."
        )
        cleaning_per_rental = st.number_input(
            "Nettoyage / préparation par location (€)", 0, 100, 8, 1, key="cactus_cleaning_v1"
        )
        avg_rental_days = st.number_input(
            "Durée moyenne d'une réservation (jours)", 1, 30, 3, 1, key="cactus_avg_booking_v1"
        )
    with r3:
        maintenance_per_km = st.number_input(
            "Provision entretien/usure par km loué (€)", 0.00, 1.00, 0.12, 0.01,
            key="cactus_maintenance_km_v1"
        )
        rental_km_per_day = st.number_input(
            "Kilomètres moyens parcourus par jour loué", 20, 300, 100, 10, key="cactus_rental_km_day_v1"
        )
        annual_fixed_cost = st.number_input(
            "Autres frais fixes annuels (contrôle, parking...) (€)", 0, 5000, 250, 50,
            key="cactus_fixed_cost_v1"
        )
    rental_gross = rental_daily_price * rental_days
    rental_bookings = rental_days / max(1, avg_rental_days)
    platform_cost = rental_gross * platform_fee_pct / 100
    cleaning_cost = rental_bookings * cleaning_per_rental
    rental_km = rental_days * rental_km_per_day
    wear_cost = rental_km * maintenance_per_km
    rental_net_before_tax = (
        rental_gross - platform_cost - cleaning_cost - wear_cost
        - rental_insurance_month * 12 - annual_fixed_cost - cactus_monthly_cost * 12
    )

    st.markdown("#### Fiscalité française — micro-BIC ou régime réel")
    st.caption(
        "Le choix du régime change la base imposable : micro-BIC = recettes brutes moins abattement forfaitaire "
        "(50 % dans cette simulation, minimum 305 €), sans déduction séparée des frais réels ; régime réel = "
        "recettes moins charges réellement déductibles et justifiées. Le régime réel peut être choisi sur option "
        "sous les seuils du micro-BIC. Vérifie ton éligibilité et les règles applicables à ta situation."
    )
    tax_c1, tax_c2 = st.columns(2)
    with tax_c1:
        tax_regime = st.radio(
            "Régime fiscal simulé",
            ["Micro-BIC", "Régime réel"],
            horizontal=True,
            key="cactus_tax_regime_v2",
            help="Compare les deux régimes avec les mêmes recettes. Le résultat réel dépend des charges fiscalement admises."
        )
        marginal_tax_rate = st.number_input(
            "Ta tranche marginale d'imposition (%)", 0.0, 45.0, 30.0, 1.0,
            key="cactus_marginal_tax_rate_v2"
        )
        micro_bic_allowance = st.number_input(
            "Abattement micro-BIC (%)", 0.0, 100.0, 50.0, 1.0,
            key="cactus_micro_bic_allowance_v2"
        )
    with tax_c2:
        fixed_cost_deductible_share = st.slider(
            "Part des coûts fixes C4 attribuable à la location (%)",
            0, 100, 0, 5, key="cactus_fixed_cost_share_v2",
            help="À utiliser uniquement pour la part réellement justifiable et déductible au régime réel. 0 % par défaut pour ne pas déduire automatiquement les coûts d'usage privé."
        )
        other_deductible_charges = st.number_input(
            "Autres charges réelles déductibles annuelles (€)", 0, 20000, 0, 50,
            key="cactus_other_deductible_v2",
            help="Saisir uniquement des dépenses justifiées, attribuables à l'activité de location et fiscalement déductibles. Ne pas inclure un amortissement sans validation comptable."
        )
        social_charges_estimate = st.number_input(
            "Cotisations / prélèvements sociaux estimés annuels (€)", 0, 20000, 0, 50,
            key="cactus_social_charges_manual_v2",
            help="Saisie manuelle à partir de ta situation et, si nécessaire, du simulateur officiel URSSAF. L'application ne suppose pas un taux forfaitaire universel."
        )

    micro_bic_abatement_eur = max(305.0, rental_gross * micro_bic_allowance / 100) if rental_gross > 0 else 0.0
    micro_bic_taxable_base = max(0.0, rental_gross - micro_bic_abatement_eur)
    real_deductible_expenses = (
        platform_cost + cleaning_cost + wear_cost
        + rental_insurance_month * 12 + annual_fixed_cost
        + cactus_monthly_cost * 12 * fixed_cost_deductible_share / 100
        + other_deductible_charges
    )
    real_taxable_profit = max(0.0, rental_gross - real_deductible_expenses)
    taxable_bic = micro_bic_taxable_base if tax_regime == "Micro-BIC" else real_taxable_profit
    estimated_income_tax = taxable_bic * marginal_tax_rate / 100
    estimated_social_charges = social_charges_estimate
    rental_net_after_tax = rental_net_before_tax - estimated_income_tax - estimated_social_charges

    st.caption(
        f"Base imposable retenue ({tax_regime}) : {money(taxable_bic)}. "
        + (
            f"Micro-BIC : recettes {money(rental_gross)} − abattement forfaitaire {money(micro_bic_abatement_eur)}. "
            "Les frais réels ne sont pas déduits séparément de la base fiscale."
            if tax_regime == "Micro-BIC"
            else f"Régime réel : recettes {money(rental_gross)} − charges saisies et attribuées à la location {money(real_deductible_expenses)}."
        )
    )
    tax_kpis = st.columns(4)
    tax_kpis[0].metric("Impôt sur le revenu estimé", money(estimated_income_tax))
    tax_kpis[1].metric("Cotisations / prélèvements sociaux", money(estimated_social_charges))
    tax_kpis[2].metric("Net après fiscalité", money(rental_net_after_tax))
    tax_kpis[3].metric(
        "Fiscalité / chiffre d'affaires",
        f"{(estimated_income_tax + estimated_social_charges) / rental_gross * 100:.1f} %" if rental_gross > 0 else "—"
    )
    st.caption(
        "Les cotisations ou prélèvements sociaux sont saisis séparément : leur régime dépend notamment du niveau "
        "de recettes, du caractère professionnel ou non de l'activité et de ta situation. Le seuil URSSAF de "
        "9 612 € pour la location de biens en 2026 est un repère à vérifier, pas un taux de cotisation. "
        "Ne saisis pas deux fois les mêmes prélèvements."
    )
    if rental_gross > 0 and social_charges_estimate == 0:
        st.warning(
            "Aucun montant de cotisations / prélèvements sociaux n'est renseigné : le net après fiscalité "
            "n'intègre donc que l'impôt sur le revenu estimé et reste provisoire. Vérifie le traitement social "
            "applicable à ton activité, même si les recettes restent sous le seuil URSSAF indicatif."
        )
    if rental_gross > 9612:
        st.caption(
            "Repère 2026 : les recettes dépassent 9 612 €, seuil indiqué par l'URSSAF pour la location de biens. "
            "Cela justifie une vérification spécifique avec le simulateur URSSAF ; ce seuil ne permet pas, à lui seul, "
            "de calculer le montant dû."
        )
    st.markdown(
        "Références à vérifier : [impots.gouv.fr — location de biens meubles et régimes BIC](https://www.impots.gouv.fr/particulier/questions/je-loue-des-biens-comment-declarer-les-revenus-tires-de-cette-location) · "
        "[URSSAF — simulateur de revenus de location de biens](https://mon-entreprise.urssaf.fr/simulateurs/location-de-biens)."
    )
    st.caption(
        "Le chiffre d'affaires utilisé est le montant brut facturé avant commission. Les frais réellement payés "
        "restent déduits une seule fois dans le calcul de trésorerie ; au régime réel, ils peuvent aussi réduire "
        "le bénéfice imposable dans la limite de leur déductibilité et de leur quote-part professionnelle."
    )

    st.markdown("#### Résultat annuel estimé de la location")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Chiffre d'affaires brut", money(rental_gross))
    a2.metric("Frais plateforme", money(platform_cost))
    a3.metric("Usure / entretien variable", money(wear_cost))
    a4.metric("Net annuel après fiscalité", money(rental_net_after_tax), f"avant fiscalité : {money(rental_net_before_tax)}")
    rental_costs = pd.DataFrame({
        "Poste": ["Revenus location", "Plateforme", "Nettoyage", "Usure / entretien", "Assurance supplémentaire",
                  "Frais fixes", "Coûts fixes C4 actuels"],
        "Montant (€)": [rental_gross, -platform_cost, -cleaning_cost, -wear_cost,
                        -rental_insurance_month * 12, -annual_fixed_cost, -cactus_monthly_cost * 12]
    })
    fig_rental = px.bar(rental_costs, x="Poste", y="Montant (€)", title="Économie annuelle estimée du C4 loué")
    fig_rental.update_layout(height=350, xaxis_tickangle=-20)
    st.plotly_chart(fig_rental, use_container_width=True)

    rental_days_break_even = None
    net_per_rental_day = rental_daily_price * (1 - platform_fee_pct / 100) - (
        cleaning_per_rental / max(1, avg_rental_days)
    ) - rental_km_per_day * maintenance_per_km
    annual_fixed_rental_costs = rental_insurance_month * 12 + annual_fixed_cost + cactus_monthly_cost * 12
    if net_per_rental_day > 0:
        rental_days_break_even = annual_fixed_rental_costs / net_per_rental_day
    if rental_days_break_even is not None:
        st.metric("Jours loués/an pour couvrir les coûts saisis", f"{rental_days_break_even:.0f} jours")
    st.info(
        "Le net affiché est après estimation de l’impôt et des cotisations/prélèvements saisis, avant dépréciation supplémentaire liée au kilométrage, "
        "et suppose que les frais déjà inclus par Getaround ne sont pas comptés une seconde fois. "
        "Vérifier l'éligibilité du véhicule, les conditions d'assurance pendant et hors location, la franchise, "
        "les sinistres, l'assistance, les périodes d'indisponibilité et les obligations fiscales."
    )

    st.subheader("12. Décision économique : vendre/reprendre ou garder pour louer")
    rental_horizon = st.select_slider(
        "Horizon de comparaison (années)", options=[1, 2, 3, 4, 5], value=3,
        key="cactus_rental_horizon_v1"
    )
    cactus_future_value = st.number_input(
        "Valeur estimée du C4 à la fin de l'horizon (€)", 0, 25000,
        max(0, int(cactus_private_value * 0.65)), 250, key="cactus_future_value_v1"
    )
    keep_net = rental_net_after_tax * rental_horizon + cactus_future_value - cactus_private_value
    sell_net = private_net - trade_net
    # The private-sale premium is shown separately; compare rental cash flow with the net value forgone.
    keep_advantage_vs_trade = rental_net_after_tax * rental_horizon + cactus_future_value - trade_net
    trade_value_net_of_debt = trade_net
    d1, d2, d3 = st.columns(3)
    d1.metric(f"Flux location cumulé net d’impôts · {rental_horizon} ans", money(rental_net_after_tax * rental_horizon))
    d2.metric("Valeur C4 estimée à terme", money(cactus_future_value))
    d3.metric("Avantage théorique garder vs reprise", money(keep_advantage_vs_trade))
    st.caption(
        "Comparaison patrimoniale simplifiée : location nette après fiscalité cumulée + valeur future du C4, comparées à la valeur nette "
        "de reprise aujourd'hui. Elle n'intègre pas l'impôt, le risque de sinistre, la dépréciation additionnelle incertaine "
        "ni le coût d'opportunité du capital. Une valeur positive ne garantit pas que la location soit préférable."
    )
    if keep_advantage_vs_trade > 0:
        st.success("Avec les hypothèses saisies, garder et louer le C4 ressort financièrement devant la reprise. Vérifie la faisabilité opérationnelle et fiscale.")
    else:
        st.warning("Avec les hypothèses saisies, la reprise ressort financièrement plus intéressante que garder et louer le C4.")
