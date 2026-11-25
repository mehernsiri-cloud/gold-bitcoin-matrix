"""Expanded family SUV total-cost comparator: RAV4 HEV/PHEV and Audi Q5 quattro."""
from datetime import date
import pandas as pd
import plotly.express as px
import streamlit as st


VEHICLES = [
    {"id":"rav4_hev_2wd","name":"Toyota RAV4 HEV 2WD","group":"RAV4 HEV 2WD","fuel":"HEV","drive":"2WD","price":33990,"km":45499,"year":2023,"fuel_l100":5.5,"ev_km":0,"ev_kwh100":0,"maint":500,"insurance":750,"battery":500,"resale5":0.58,"resale8":0.40,"resale10":0.30,"space_score":8.5,"traction_score":4.5,"comfort_score":7.5,"note":"Hybride auto-rechargeable, traction avant"},
    {"id":"rav4_hev_awd","name":"Toyota RAV4 HEV AWD-i","group":"RAV4 HEV AWD-i","fuel":"HEV","drive":"AWD-i","price":39999,"km":19163,"year":2024,"fuel_l100":5.8,"ev_km":0,"ev_kwh100":0,"maint":525,"insurance":800,"battery":600,"resale5":0.60,"resale8":0.42,"resale10":0.31,"space_score":8.5,"traction_score":8.0,"comfort_score":7.5,"note":"Hybride auto-rechargeable, transmission intégrale AWD-i"},
    {"id":"rav4_phev_awd","name":"Toyota RAV4 PHEV AWD-i","group":"RAV4 PHEV AWD-i","fuel":"PHEV","drive":"AWD-i","price":49990,"km":14326,"year":2025,"fuel_l100":6.3,"ev_km":60,"ev_kwh100":19.0,"maint":575,"insurance":900,"battery":1200,"resale5":0.56,"resale8":0.37,"resale10":0.26,"space_score":8.0,"traction_score":8.5,"comfort_score":7.8,"note":"Rechargeable, 306 ch, AWD-i, autonomie électrique dépendante de l'usage"},
    {"id":"q5_phev","name":"Audi Q5 50 TFSI e quattro","group":"Audi Q5 PHEV quattro","fuel":"PHEV","drive":"quattro","price":36900,"km":44252,"year":2022,"fuel_l100":7.0,"ev_km":40,"ev_kwh100":22.0,"maint":850,"insurance":1050,"battery":2000,"resale5":0.48,"resale8":0.31,"resale10":0.21,"space_score":7.8,"traction_score":9.0,"comfort_score":8.8,"note":"PHEV 299 ch quattro, annonce Business Executive à 36 900 €"},
]


def render_family_suv_comparator():
    st.header("🚙 Comparateur SUV familial — RAV4 & Audi Q5 quattro")
    st.caption("Comparaison multi-versions du coût total de possession. Toutes les valeurs de départ sont modifiables : les résultats sont des scénarios, pas des devis ni des prévisions de cote.")
    st.info("Périmètre : RAV4 HEV 2WD, RAV4 HEV AWD-i, RAV4 PHEV AWD-i, Audi Q5 50 TFSI e quattro PHEV. Les hypothèses initiales sont indicatives ; remplace-les par les annonces et devis réels.")

    st.subheader("1. Véhicules et annonces de référence")
    edited = []
    cols = st.columns(2)
    for i, v in enumerate(VEHICLES):
        with cols[i % 2]:
            with st.expander(v["name"], expanded=(i < 3)):
                st.caption(v["note"])
                price = st.number_input("Prix d'achat (€)", 10000, 100000, int(v["price"]), 250, key=f"fam_{v['id']}_price")
                km = st.number_input("Kilométrage à l'achat", 0, 200000, int(v["km"]), 1000, key=f"fam_{v['id']}_km")
                year = st.number_input("Année de 1re mise en circulation", 2018, 2026, int(v["year"]), 1, key=f"fam_{v['id']}_year")
                fuel_l100 = st.number_input("Consommation essence réelle (L/100 km)", 3.0, 16.0, float(v["fuel_l100"]), 0.1, key=f"fam_{v['id']}_fuel")
                ev_km = st.number_input("Autonomie électrique utile (km)", 0, 120, int(v["ev_km"]), 5, key=f"fam_{v['id']}_evkm")
                ev_kwh100 = st.number_input("Consommation électrique (kWh/100 km)", 0.0, 40.0, float(v["ev_kwh100"]), 0.5, key=f"fam_{v['id']}_evkwh")
                maint = st.number_input("Entretien annuel (€)", 0, 4000, int(v["maint"]), 50, key=f"fam_{v['id']}_maint")
                insurance = st.number_input("Assurance annuelle (€)", 0, 4000, int(v["insurance"]), 50, key=f"fam_{v['id']}_insurance")
                battery = st.number_input("Provision risque batterie / électronique sur la période (€)", 0, 20000, int(v["battery"]), 100, key=f"fam_{v['id']}_battery")
                resale5 = st.number_input("Valeur de revente à 5 ans (% du prix d’achat)", 0.0, 1.0, float(v["resale5"]), 0.01, format="%.2f", key=f"fam_{v['id']}_resale5")
                resale8 = st.number_input("Valeur de revente à 8 ans (% du prix d’achat)", 0.0, 1.0, float(v["resale8"]), 0.01, format="%.2f", key=f"fam_{v['id']}_resale8")
                resale10 = st.number_input("Valeur de revente à 10 ans (% du prix d’achat)", 0.0, 1.0, float(v["resale10"]), 0.01, format="%.2f", key=f"fam_{v['id']}_resale10")
                edited.append({**v, "price":price, "km":km, "year":year, "fuel_l100":fuel_l100, "ev_km":ev_km, "ev_kwh100":ev_kwh100, "maint":maint, "insurance":insurance, "battery":battery, "resale5":resale5, "resale8":resale8, "resale10":resale10})

    st.subheader("2. Ton usage et tes filtres")
    a,b,c = st.columns(3)
    with a:
        annual_km = st.number_input("Kilométrage annuel total", 1000, 40000, 15000, 500, key="family_annual_km")
        local_share = st.slider("Part des km locaux (%)", 0, 100, 65, 5, key="family_local_share")
    with b:
        fuel_price = st.number_input("Prix essence (€/L)", 0.80, 4.00, 1.90, 0.05, key="family_fuel_price")
        electricity_price = st.number_input("Électricité domicile (€/kWh)", 0.05, 1.00, 0.25, 0.01, key="family_elec_price")
    with c:
        home_charge = st.checkbox("Installation de recharge prévue / disponible", value=True, key="family_home_charge")
        charger_cost = st.number_input("Borne + raccordement + pose (€)", 0, 12000, 2000, 100, key="family_charger_cost")
        charge_frequency = st.selectbox("Fréquence de recharge", ["Quotidienne","3 à 4 fois/semaine","1 à 2 fois/semaine"], index=0, key="family_charge_freq")
        ev_coverage = st.slider("Part des trajets locaux couverts par l'autonomie électrique (%)", 0, 100, 75, 5, key="family_ev_coverage")
    charge_factor = {"Quotidienne":1.0,"3 à 4 fois/semaine":0.75,"1 à 2 fois/semaine":0.45}[charge_frequency] if home_charge else 0.0
    local_km = annual_km * local_share / 100
    st.caption("Les kilomètres électriques sont limités par la part des trajets locaux, la fréquence de recharge et l'autonomie utile saisie (40/60 km par recharge). La fréquence n'est appliquée qu'une fois. C'est une approximation : les distances quotidiennes et les possibilités de recharge au travail ne sont pas modélisées. Les longs trajets restent calculés en essence/hybride.")
    st.subheader("3. Notes pondérées — espace, motricité, confort et âge")
    st.caption("Notes initiales indicatives sur 10, modifiables. L’âge est calculé automatiquement. Les valeurs de revente sont également modifiables dans chaque fiche : elles ne sont pas des cotes garanties.")
    w1, w2, w3, w4 = st.columns(4)
    with w1: weight_space = st.slider("Poids espace global (%)", 0, 100, 30, 5, key="fam_w_space")
    with w2: weight_traction = st.slider("Poids motricité 4 roues (%)", 0, 100, 20, 5, key="fam_w_traction")
    with w3: weight_comfort = st.slider("Poids confort conduite (%)", 0, 100, 30, 5, key="fam_w_comfort")
    with w4: weight_age = st.slider("Poids âge véhicule (%)", 0, 100, 20, 5, key="fam_w_age")
    total_weight = max(1, weight_space + weight_traction + weight_comfort + weight_age)
    score_rows = []
    for v in edited:
        v["space_score"] = st.slider(f"{v['name']} — espace global /10", 0.0, 10.0, float(v["space_score"]), 0.5, key=f"score_space_{v['id']}")
        v["traction_score"] = st.slider(f"{v['name']} — motricité /10", 0.0, 10.0, float(v["traction_score"]), 0.5, key=f"score_traction_{v['id']}")
        v["comfort_score"] = st.slider(f"{v['name']} — confort de conduite /10", 0.0, 10.0, float(v["comfort_score"]), 0.5, key=f"score_comfort_{v['id']}")
        age_purchase = max(0, date.today().year - int(v["year"]))
        age_score = max(0.0, 10.0 - age_purchase * 1.25)
        score = (v["space_score"]*weight_space + v["traction_score"]*weight_traction + v["comfort_score"]*weight_comfort + age_score*weight_age) / total_weight
        score_rows.append({"Véhicule":v["name"],"Espace /10":v["space_score"],"Motricité /10":v["traction_score"],"Confort /10":v["comfort_score"],"Âge à l’achat":f"{age_purchase} an(s)","Note âge /10":age_score,"Note pondérée /10":score})
    score_df = pd.DataFrame(score_rows).sort_values("Note pondérée /10", ascending=False)
    score_by_name = {row["Véhicule"]: row["Note pondérée /10"] for row in score_rows}
    st.dataframe(score_df.style.format({"Espace /10":"{:.1f}","Motricité /10":"{:.1f}","Confort /10":"{:.1f}","Note âge /10":"{:.1f}","Note pondérée /10":"{:.2f}"}), use_container_width=True, hide_index=True)

    st.subheader("4. Horizons de détention")
    horizons = [5,8,10]
    st.caption("Les trois horizons sont calculés simultanément afin de visualiser le moment où le classement économique change.")

    results = []
    details = []
    for v in edited:
        is_phev = v["fuel"] == "PHEV"
        # Electric mileage cannot exceed either the share of local trips
        # the user expects to cover electrically or one usable battery range
        # per effective charging day. This prevents autonomy from being ignored.
        requested_local_ev_km = local_km * ev_coverage / 100 if is_phev else 0
        annual_charge_capacity = v["ev_km"] * 365 * charge_factor if is_phev else 0
        ev_km_year = min(requested_local_ev_km * charge_factor, annual_charge_capacity, annual_km)
        electric_cost_year = ev_km_year * v["ev_kwh100"] / 100 * electricity_price * 1.10
        petrol_km_year = annual_km - ev_km_year
        energy_year = petrol_km_year * v["fuel_l100"] / 100 * fuel_price + electric_cost_year
        for years in horizons:
            exit_km = v["km"] + annual_km * years
            age_exit = max(0, date.today().year - int(v["year"]) + years)
            resale_rate = v["resale5"] if years == 5 else v["resale8"] if years == 8 else v["resale10"]
            # The editable resale rate already represents the assumed residual
            # value at this horizon. Do not apply a second age/mileage discount.
            # Reflect projected age and mileage when choosing the resale rate.
            resale = v["price"] * resale_rate
            charging = charger_cost if (is_phev and home_charge) else 0
            usage_cost = years * (energy_year + v["maint"] + v["insurance"]) + v["battery"]
            net_depreciation = v["price"] - resale
            tco = net_depreciation + charging + usage_cost
            results.append({"Véhicule":v["name"],"Horizon":f"{years} ans","Âge à l’achat":max(0, date.today().year-int(v["year"])),"Âge à la revente":age_exit,"TCO (€)":tco,"€/km":tco/(annual_km*years),"Dépréciation nette (€)":net_depreciation,"Coût d’usage hors achat/revente (€)":usage_cost,"Borne (€)":charging,"Note pondérée /10":score_by_name.get(v["name"], 0),"Revente estimée (€)":resale,"Km électriques/an":ev_km_year,"Km à la sortie":exit_km})
            details.append({"Véhicule":v["name"],"Horizon":f"{years} ans","Achat":v["price"],"Borne":charging,"Énergie":years*energy_year,"Entretien":years*v["maint"],"Assurance":years*v["insurance"],"Provision batterie":v["battery"],"Revente déduite":-resale})

    df = pd.DataFrame(results)
    st.subheader("5. Résultat — coût total à 5, 8 et 10 ans")
    for years in horizons:
        sub = df[df["Horizon"] == f"{years} ans"].copy()
        baseline_rows = sub[sub["Véhicule"] == "Toyota RAV4 HEV 2WD"]
        baseline_tco = float(baseline_rows.iloc[0]["TCO (€)"]) if not baseline_rows.empty else float(sub["TCO (€)"].min())
        sub["Surcoût vs HEV 2WD (€)"] = sub["TCO (€)"] - baseline_tco
        sub["Surcoût annuel vs HEV 2WD (€)"] = sub["Surcoût vs HEV 2WD (€)"] / years
        sub = sub.sort_values("TCO (€)")
        winner = sub.iloc[0]
        st.markdown(f"**À {years} ans : {winner['Véhicule']} ressort le moins cher** — {winner['TCO (€)']:,.0f} € ({winner['€/km']:.3f} €/km).".replace(",", " "))
        st.caption("Le surcoût vs RAV4 HEV 2WD chiffre ce que tu paies en plus pour passer à une autre motorisation ou transmission. Il ne signifie pas automatiquement que cette version est moins intéressante.")
        display_cols = ["Véhicule","Note pondérée /10","Âge à l’achat","Âge à la revente","TCO (€)","Surcoût vs HEV 2WD (€)","Surcoût annuel vs HEV 2WD (€)","Dépréciation nette (€)","Coût d’usage hors achat/revente (€)","€/km"]
        st.dataframe(sub[display_cols].style.format({"Note pondérée /10":"{:.2f}","TCO (€)":"{:,.0f} €","Surcoût vs HEV 2WD (€)":"{:+,.0f} €","Surcoût annuel vs HEV 2WD (€)":"{:+,.0f} €/an","Dépréciation nette (€)":"{:,.0f} €","Coût d’usage hors achat/revente (€)":"{:,.0f} €","€/km":"{:.3f} €"}), use_container_width=True, hide_index=True)

    chart = px.bar(df, x="Véhicule", y="TCO (€)", color="Horizon", barmode="group", title="Coût total estimé selon la durée de détention")
    chart.update_layout(height=460, xaxis_tickangle=-20)
    st.plotly_chart(chart, use_container_width=True)

    st.subheader("6. Décomposition du coût")
    detail_df = pd.DataFrame(details)
    horizon_choice = st.selectbox("Horizon à détailler", horizons, index=1, key="family_detail_horizon")
    detail = detail_df[detail_df["Horizon"] == f"{horizon_choice} ans"].set_index("Véhicule")
    # Apply currency formatting only to numeric columns (never to vehicle-name strings).
    detail_format = {col: "{:,.0f} €" for col in detail.select_dtypes(include="number").columns}
    st.dataframe(detail.style.format(detail_format), use_container_width=True)

    st.subheader("7. Sensibilité et lecture des résultats")
    st.caption("Teste notamment le prix de l'essence, le tarif électrique, la fréquence de recharge, les frais d'entretien et la valeur de revente. Les kilomètres électriques sont désormais plafonnés par l'autonomie utile × le nombre de jours de recharge estimé. Les provisions batterie sont des réserves de risque paramétrables, pas une panne supposée.")
    st.warning("Les valeurs de revente, coûts d'entretien et provisions batterie sont des hypothèses de travail à personnaliser. Le modèle ne prétend pas connaître la valeur future exacte ni garantir le coût réel des réparations. Les valeurs initiales des Audi et Toyota doivent être remplacées par les annonces que tu retiendras.")
    st.markdown("**À contrôler avant achat :** historique d'entretien, garantie écrite et ses exclusions, état de santé batterie HT pour PHEV, boîte automatique, système AWD, pneus (usure homogène), et devis réel de recharge en copropriété.")
