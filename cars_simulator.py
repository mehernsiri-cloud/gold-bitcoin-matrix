"""Interactive Toyota RAV4 HEV vs PHEV total-cost simulator.

All resale, maintenance and battery-risk values are editable planning assumptions,
not market forecasts or manufacturer guarantees.
"""
import math
import pandas as pd
import plotly.express as px
import streamlit as st


def _money(value):
    return f"{value:,.0f} €".replace(",", " ")


def render_rav4_simulator():
    st.markdown("---")
    st.header("🧮 Simulateur de choix Toyota RAV4 · HEV vs PHEV")
    st.caption(
        "Compare le coût total de possession selon ton kilométrage, les prix de l'énergie, "
        "la durée de détention, la revente et une provision de risque batterie. "
        "Les hypothèses sont modifiables et ne constituent pas des prévisions."
    )

    with st.expander("🚘 Véhicules de référence — annonces fournies", expanded=True):
        a, b = st.columns(2)
        with a:
            st.markdown("**RAV4 HEV 2.5 218 ch · 2WD Dynamic Business**")
            hev_price = st.number_input("Prix HEV (€)", 20000, 80000, 39000, 500, key="sim_hev_price")
            st.caption("2025 · 20 km · 218 ch · 2 roues motrices · 9 CV")
        with b:
            st.markdown("**RAV4 PHEV 2.5 306 ch · AWD-i Collection**")
            phev_price = st.number_input("Prix PHEV (€)", 20000, 80000, 49000, 500, key="sim_phev_price")
            st.caption("2025 · 14 326 km · 306 ch · AWD-i · 10 CV · batterie 18,1 kWh · 75 km annoncés")
        st.info(
            f"Écart de prix affiché : {_money(phev_price-hev_price)}. "
            "Attention : les finitions et transmissions diffèrent (2WD Business contre AWD-i Collection). "
            "Le simulateur compare les annonces telles quelles ; il ne considère donc pas les 10 000 € "
            "comme un pur surcoût de batterie."
        )

    st.subheader("1. Ton utilisation")
    c1, c2, c3 = st.columns(3)
    with c1:
        annual_km = st.number_input("Kilométrage annuel total", 1000, 30000, 4500, 500, key="sim_annual_km")
    with c2:
        years = st.select_slider("Durée de détention", options=[3, 4, 5, 6, 7, 8, 9, 10, 12], value=8, key="sim_years")
    with c3:
        local_km = st.number_input("Km/an de trajets locaux", 0, 20000, 1500, 250, key="sim_local_km")
    st.caption("Le reste du kilométrage est considéré comme effectué sur les grands trajets (Tunisie et Europe).")
    ev_share = st.slider(
        "Part des kilomètres annuels du PHEV effectuée en électrique (%)",
        min_value=0, max_value=100, value=55, step=5, key="sim_ev_share",
        help="Part théorique électrique, plafonnée au kilométrage local annuel saisi juste au-dessus : les grands voyages sans recharge sont donc comptés en thermique."
    )
    phev_ev_km = min(annual_km * ev_share / 100, local_km)
    st.caption(f"Kilométrage électrique PHEV simulé : {phev_ev_km:,.0f} km/an · thermique : {annual_km-phev_ev_km:,.0f} km/an".replace(",", " "))

    st.subheader("2. Prix de l'énergie et consommations")
    c1, c2, c3 = st.columns(3)
    with c1:
        fuel_price = st.number_input("Essence (€ / litre)", 0.80, 4.00, 1.95, 0.05, key="sim_fuel")
        home_power = st.number_input("Électricité domicile (€ / kWh)", 0.05, 1.00, 0.25, 0.01, key="sim_power")
    with c2:
        hev_l100 = st.number_input("Consommation HEV (L/100 km)", 3.0, 12.0, 5.5, 0.1, key="sim_hev_l100")
        phev_hv_l100 = st.number_input("PHEV en mode hybride (L/100 km)", 3.0, 14.0, 6.2, 0.1, key="sim_phev_l100")
    with c3:
        phev_kwh100 = st.number_input("PHEV en électrique (kWh/100 km)", 12.0, 35.0, 20.0, 0.5, key="sim_phev_kwh")
        charging_loss = st.slider("Pertes de recharge (%)", 0, 25, 10, 1, key="sim_loss")

    hev_energy_y = annual_km * hev_l100 / 100 * fuel_price
    phev_fuel_y = (annual_km-phev_ev_km) * phev_hv_l100 / 100 * fuel_price
    phev_elec_y = phev_ev_km * phev_kwh100 / 100 * home_power * (1+charging_loss/100)
    phev_energy_y = phev_fuel_y + phev_elec_y

    st.subheader("3. Entretien, assurance et risque batterie")
    st.caption("Saisis les différences que tu anticipes ou que tu connais. Les montants de provision sont des hypothèses prudentes, pas des tarifs Toyota.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**HEV**")
        hev_maint = st.number_input("Entretien annuel HEV (€)", 0, 3000, 500, 50, key="sim_hev_maint")
        hev_ins = st.number_input("Assurance annuelle HEV (€)", 0, 3000, 750, 50, key="sim_hev_ins")
        hev_batt_reserve = st.number_input("Provision batterie HEV sur toute la période (€)", 0, 15000, 500, 100, key="sim_hev_batt")
    with c2:
        st.markdown("**PHEV**")
        phev_maint = st.number_input("Entretien annuel PHEV (€)", 0, 3000, 550, 50, key="sim_phev_maint")
        phev_ins = st.number_input("Assurance annuelle PHEV (€)", 0, 3000, 850, 50, key="sim_phev_ins")
        phev_batt_reserve = st.number_input("Provision batterie PHEV sur toute la période (€)", 0, 25000, 1200, 100, key="sim_phev_batt")
    st.caption("La provision batterie peut représenter une réserve de risque, pas nécessairement un remplacement complet. Elle est ajoutée au TCO et peut être mise à zéro.")

    st.subheader("4. Valeur de revente estimée")
    st.caption("Pourcentage du prix d'achat conservé au moment de la revente. Modifie ces valeurs selon les annonces observées ; elles ne sont pas des cotes garanties.")
    r1, r2 = st.columns(2)
    with r1:
        hev_resale_pct = st.slider("Valeur résiduelle HEV (%)", 5, 90, 48, 1, key="sim_hev_resale")
    with r2:
        phev_resale_pct = st.slider("Valeur résiduelle PHEV (%)", 5, 90, 43, 1, key="sim_phev_resale")
    hev_resale = hev_price * hev_resale_pct / 100
    phev_resale = phev_price * phev_resale_pct / 100

    # Ownership cost: purchase + energy + maintenance + insurance + battery risk - resale.
    hev_total = hev_price + years*(hev_energy_y+hev_maint+hev_ins) + hev_batt_reserve - hev_resale
    phev_total = phev_price + years*(phev_energy_y+phev_maint+phev_ins) + phev_batt_reserve - phev_resale
    hev_per_km = hev_total / max(1, annual_km*years)
    phev_per_km = phev_total / max(1, annual_km*years)
    winner = "HEV" if hev_total <= phev_total else "PHEV"
    diff = abs(hev_total-phev_total)

    st.subheader(f"5. Résultat sur {years} ans")
    k1, k2, k3 = st.columns(3)
    k1.metric("TCO HEV", _money(hev_total), f"{hev_per_km:.2f} €/km")
    k2.metric("TCO PHEV", _money(phev_total), f"{phev_per_km:.2f} €/km")
    k3.metric("Écart", _money(diff), f"{winner} moins cher")
    chart = pd.DataFrame({
        "Poste": ["Achat", "Énergie", "Entretien", "Assurance", "Risque batterie", "Revente déduite"],
        "HEV": [hev_price, years*hev_energy_y, years*hev_maint, years*hev_ins, hev_batt_reserve, -hev_resale],
        "PHEV": [phev_price, years*phev_energy_y, years*phev_maint, years*phev_ins, phev_batt_reserve, -phev_resale],
    })
    chart_long = chart.melt(id_vars="Poste", var_name="Motorisation", value_name="Coût (€)")
    fig = px.bar(chart_long, x="Poste", y="Coût (€)", color="Motorisation", barmode="group",
                 title="Décomposition du coût total de possession")
    fig.update_layout(height=390, legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    breakdown = pd.DataFrame([
        {"Poste": "Prix d'achat", "HEV": hev_price, "PHEV": phev_price},
        {"Poste": "Énergie sur la période", "HEV": years*hev_energy_y, "PHEV": years*phev_energy_y},
        {"Poste": "Entretien sur la période", "HEV": years*hev_maint, "PHEV": years*phev_maint},
        {"Poste": "Assurance sur la période", "HEV": years*hev_ins, "PHEV": years*phev_ins},
        {"Poste": "Provision batterie", "HEV": hev_batt_reserve, "PHEV": phev_batt_reserve},
        {"Poste": "Revente estimée (déduite)", "HEV": -hev_resale, "PHEV": -phev_resale},
        {"Poste": "COÛT TOTAL", "HEV": hev_total, "PHEV": phev_total},
    ])
    st.dataframe(breakdown.style.format({"HEV": "{:,.0f} €", "PHEV": "{:,.0f} €"}), use_container_width=True, hide_index=True)

    st.subheader("6. Sensibilité au prix de l'essence")
    rows = []
    for price in [1.50, 1.75, 2.00, 2.25, 2.50, 3.00]:
        h_energy = annual_km*hev_l100/100*price
        p_energy = (annual_km-phev_ev_km)*phev_hv_l100/100*price + phev_ev_km*phev_kwh100/100*home_power*(1+charging_loss/100)
        h_tco = hev_price+years*(h_energy+hev_maint+hev_ins)+hev_batt_reserve-hev_resale
        p_tco = phev_price+years*(p_energy+phev_maint+phev_ins)+phev_batt_reserve-phev_resale
        rows.append({"Essence (€/L)": price, "TCO HEV (€)": h_tco, "TCO PHEV (€)": p_tco, "Avantage (€)": h_tco-p_tco})
    sens = pd.DataFrame(rows)
    st.dataframe(sens.style.format({"Essence (€/L)": "{:.2f}", "TCO HEV (€)": "{:,.0f}", "TCO PHEV (€)": "{:,.0f}", "Avantage (€)": "{:+,.0f}"}), use_container_width=True, hide_index=True)

    st.subheader("7. Points à vérifier avant achat")
    st.markdown(
        "- **Comparabilité :** le PHEV est AWD-i Collection (306 ch), le HEV est 2WD Dynamic Business (218 ch). Pour isoler l'effet de la motorisation, comparer idéalement deux finitions et transmissions proches.\n"
        "- **Batterie :** demander un diagnostic/état de santé documenté, l'historique de charge et les conditions exactes de garantie applicables au VIN.\n"
        "- **Revente :** vérifier les annonces comparables par année, kilométrage, finition et motorisation au moment de la décision.\n"
        "- **Usage Tunisie :** le PHEV peut continuer en mode hybride sans borne ; la consommation réelle batterie vide doit être prise en compte dans les longs trajets.\n"
        "- **Données à confirmer :** prix, kilométrage, première mise en circulation, garantie restante, historique et état des pneus/freins."
    )
