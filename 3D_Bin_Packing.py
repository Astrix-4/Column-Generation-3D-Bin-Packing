import pandas as pd

# Step 1: Load the CSV file
file_path = "/Users/manav/Desktop/3D Bin Packing/cleaned_picklist_dataset_with_config.csv" # Adjust path if needed
df = pd.read_csv(file_path)

# Step 2: Define the Item class with rotation support
class Item:
    def __init__(self, id, length, width, height, quantity):
        self.id = id
        self.length = length
        self.width = width
        self.height = height
        self.quantity = quantity

    def get_orientations(self):
        l, w, h = self.length, self.width, self.height
        return [
            (l, w, h),
            (l, h, w),
            (w, l, h),
            (w, h, l),
            (h, l, w),
            (h, w, l)
        ]

# Step 3: Filter required columns
items_df = df[['Picklist Barcode', 'Qty', 'Length', 'Width', 'Height']].copy()

# Step 4: Remove rows with missing dimension values
items_df.dropna(subset=['Length', 'Width', 'Height'], inplace=True)

# Step 5: Create list of individual Item instances (one per unit)
items = []
for _, row in items_df.iterrows():
    for q in range(int(row['Qty'])):
        items.append(Item(
            id=f"{row['Picklist Barcode']}_{q+1}",
            length=float(row['Length']),
            width=float(row['Width']),
            height=float(row['Height']),
            quantity=1
        ))

# Step 6: Print orientation info for first 3 items as a check
print("Sample item orientations:\n")
for item in items[:3]:
    print(f"Item ID: {item.id}")
    for i, o in enumerate(item.get_orientations()):
        print(f"  Orientation {i+1}: {o}")
    print()
