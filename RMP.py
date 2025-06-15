#This code picks a combination of distinct item copies that will give you the maximum total packed volume, under the rule that only one copy per item is allowed.†
import pulp
import pandas as pd

# Load the dataset
df = pd.read_csv("cleaned_picklist_dataset_with_config.csv")

# Expand rows by quantity so that each item copy is treated individually
expanded_rows = []
for _, row in df.iterrows():
    for q in range(int(row["Qty"])):
        expanded_rows.append({
            "config_id": f"{row['Material Barcode']}_{q}",
            "item_id": row["Material Barcode"],
            "volume": row["Length"] * row["Width"] * row["Height"]
        })
expanded_df = pd.DataFrame(expanded_rows)

# Initialize the RMP
rmp = pulp.LpProblem("3D_Bin_Packing_RMP", pulp.LpMaximize)

# Create binary decision variables
x = {
    row["config_id"]: pulp.LpVariable(f"x_{row['config_id']}", cat='Binary')
    for _, row in expanded_df.iterrows()
}

# Objective: Maximize total packed volume
rmp += pulp.lpSum(
    x[row["config_id"]] * row["volume"] for _, row in expanded_df.iterrows()
), "TotalPackedVolume"

# Constraint: Each item copy (config_id) can be used at most once — automatically enforced by binary vars
# But if we want to restrict item ID to appear only once (ignoring copies), use below:
for item in expanded_df["item_id"].unique():
    rmp += pulp.lpSum(
        x[row["config_id"]] for _, row in expanded_df.iterrows() if row["item_id"] == item
    ) <= 1, f"Item_{item}_Once"

# Solve the RMP
rmp.solve()

# Output the results
print("Status:", pulp.LpStatus[rmp.status])
print("Total Packed Volume:", pulp.value(rmp.objective))
for cid, var in x.items():
    if var.varValue > 0.5:
        print(f"Selected Configuration: {cid}")
