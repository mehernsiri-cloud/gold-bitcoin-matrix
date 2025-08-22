# Save actual data
def save_actual_data():
    today = datetime.today().strftime("%Y-%m-%d")  # <-- fixed here
    indicators = fetch_indicators()
    prices = fetch_prices()

    row = {
        "date": today,
        "gold_actual": prices.get("Gold"),
        "bitcoin_actual": prices.get("Bitcoin"),
        **indicators
    }

    if os.path.exists(ACTUAL_FILE):
        df = pd.read_csv(ACTUAL_FILE)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(ACTUAL_FILE, index=False)
    print("✅ Actual data updated:", row)
