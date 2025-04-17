
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run_optimization(df):
    df["Used Volume"] = df["Manual Load"] * df["Volume (cbm)"]
    df["Used Weight"] = df["Manual Load"] * df["Weight (kg)"]
    used_volume_total = df["Used Volume"].sum()
    used_weight_total = df["Used Weight"].sum()

    # Simulate Training Data
    sim_data = []
    np.random.seed(42)
    for _ in range(1000):
        for _, row in df.iterrows():
            rem_vol = np.random.uniform(5, 76)
            rem_wt = np.random.uniform(5000, 28000)
            usage = row["Weekly Usage"] + np.random.uniform(-10, 10)
            stock = row["Stock on Hand"] + np.random.randint(-100, 100)
            cover = stock / (usage + 1e-5)
            priority = ((usage / (cover + 1e-2)) * 10 +
                        (1 / (cover + 1.1)) * 100 +
                        (1 / (row["Lead Time"] + 1e-2)) * 50)
            max_by_vol = int(rem_vol // row["Volume (cbm)"])
            max_by_wt = int(rem_wt // row["Weight (kg)"])
            realistic_max = min(max_by_vol, max_by_wt, stock)
            cartons = max(0, int(realistic_max * np.random.uniform(0.5, 1.0)))

            sim_data.append([
                rem_vol, rem_wt, row["Volume (cbm)"], row["Weight (kg)"], usage, stock,
                cover, row["Stock on Order"], row["Lead Time"],
                row["MOQ"], row["Safety Stock"], row["ROP"], priority, cartons
            ])

    sim_df = pd.DataFrame(sim_data, columns=[
        "Remaining Volume", "Remaining Weight", "SKU Volume", "SKU Weight", "Weekly Usage",
        "Stock on Hand", "Weeks of Cover", "Stock on Order", "Lead Time", "MOQ",
        "Safety Stock", "ROP", "Priority Score", "Cartons"
    ])

    # Train ML Model
    X = sim_df.drop("Cartons", axis=1)
    y = sim_df["Cartons"]
    X_train, _, y_train, _ = train_test_split(X, y, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    def hybrid_fill_full(container_name, max_volume, max_weight):
        remaining_volume = max_volume - used_volume_total
        remaining_weight = max_weight - used_weight_total
        predictions = []

        for _, row in df[df["Manual Load"] == 0].iterrows():
            weeks_of_cover = row["Stock on Hand"] / (row["Weekly Usage"] + 1e-5)
            input_data = pd.DataFrame([{
                "Remaining Volume": remaining_volume,
                "Remaining Weight": remaining_weight,
                "SKU Volume": row["Volume (cbm)"],
                "SKU Weight": row["Weight (kg)"],
                "Weekly Usage": row["Weekly Usage"],
                "Stock on Hand": row["Stock on Hand"],
                "Weeks of Cover": weeks_of_cover,
                "Stock on Order": row["Stock on Order"],
                "Lead Time": row["Lead Time"],
                "MOQ": row["MOQ"],
                "Safety Stock": row["Safety Stock"],
                "ROP": row["ROP"],
                "Priority Score": (row["Weekly Usage"] / (weeks_of_cover + 1e-2)) * 10 +
                                  (1 / (weeks_of_cover + 1.1)) * 100 +
                                  (1 / (row["Lead Time"] + 1e-2)) * 50
            }])
            input_scaled = scaler.transform(input_data)
            pred_qty = int(model.predict(input_scaled)[0] * 1.8)
            max_qty_by_vol = int(remaining_volume // row["Volume (cbm)"])
            max_qty_by_wt = int(remaining_weight // row["Weight (kg)"])
            max_possible = min(pred_qty, max_qty_by_vol, max_qty_by_wt,
                               row["Stock on Hand"] + row["Stock on Order"])

            if max_possible >= row["MOQ"]:
                predictions.append({
                    "Container Type": container_name,
                    "SKU": row["SKU Code"],
                    "Predicted Qty": max_possible,
                    "Used Volume": round(max_possible * row["Volume (cbm)"], 3),
                    "Used Weight": round(max_possible * row["Weight (kg)"], 2)
                })

        if not predictions:
            return container_name, 0, 0, pd.DataFrame(columns=["Container Type", "SKU", "Predicted Qty", "Used Volume", "Used Weight"])

        pred_df = pd.DataFrame(predictions).sort_values(by="Used Volume").reset_index(drop=True)
        final_selection = []
        vol_accum, wt_accum = 0, 0

        for _, row in pred_df.iterrows():
            if vol_accum + row["Used Volume"] <= remaining_volume and wt_accum + row["Used Weight"] <= remaining_weight:
                final_selection.append(row)
                vol_accum += row["Used Volume"]
                wt_accum += row["Used Weight"]
            else:
                rem_vol = remaining_volume - vol_accum
                rem_wt = remaining_weight - wt_accum
                max_qty_by_vol = int(rem_vol // (row["Used Volume"] / row["Predicted Qty"]))
                max_qty_by_wt = int(rem_wt // (row["Used Weight"] / row["Predicted Qty"]))
                trim_qty = min(max_qty_by_vol, max_qty_by_wt)
                if trim_qty >= df[df["SKU Code"] == row["SKU"]]["MOQ"].values[0]:
                    trimmed_row = row.copy()
                    trimmed_row["Predicted Qty"] = trim_qty
                    trimmed_row["Used Volume"] = round(trim_qty * (row["Used Volume"] / row["Predicted Qty"]), 3)
                    trimmed_row["Used Weight"] = round(trim_qty * (row["Used Weight"] / row["Predicted Qty"]), 2)
                    final_selection.append(trimmed_row)
                    vol_accum += trimmed_row["Used Volume"]
                    wt_accum += trimmed_row["Used Weight"]
                break

        return container_name, vol_accum, wt_accum, pd.DataFrame(final_selection)

    # Container setup
    containers = {
        "20ft": (33, 28000),
        "40ft": (67, 26000),
        "20ft HC": (37.5, 28000),
        "40ft HC": (76, 26000)
    }

    results = {name: hybrid_fill_full(name, vol, wt) for name, (vol, wt) in containers.items()}
    best_container = max(results.values(), key=lambda x: max(x[1] / containers[x[0]][0], x[2] / containers[x[0]][1]))

    manual_df = df[df["Manual Load"] > 0][["SKU Code", "Manual Load"]].copy()
    manual_df.columns = ["SKU", "Predicted Qty"]
    manual_df["Used Volume"] = manual_df["SKU"].map(df.set_index("SKU Code")["Volume (cbm)"]) * manual_df["Predicted Qty"]
    manual_df["Used Weight"] = manual_df["SKU"].map(df.set_index("SKU Code")["Weight (kg)"]) * manual_df["Predicted Qty"]
    manual_df["Container Type"] = best_container[0]

    final_df = pd.concat([manual_df, best_container[3]], ignore_index=True)

    summary_df = pd.DataFrame({
        "Metric": ["Used Volume (cbm)", "Used Weight (kg)", "Max Volume", "Max Weight"],
        "Value": [final_df["Used Volume"].sum(), final_df["Used Weight"].sum(),
                  containers[best_container[0]][0], containers[best_container[0]][1]]
    })

    return best_container[0], final_df, summary_df
