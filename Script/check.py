import pickle
import os

target_id = 4691
year = 2013
folder = f"orders/partial/{year}"

current_id = 0
found = False

for fname in sorted(os.listdir(folder)):
    if not fname.endswith(".pkl") or not fname.startswith(f"orders_{year}_block_"):
        continue

    with open(os.path.join(folder, fname), "rb") as f:
        block = pickle.load(f)

    for result in block:
        if current_id == target_id:
            final_cash, orders = result
            print(f"✅ Trovata combinazione {target_id} in file: {fname}")
            print(f"💰 Capitale: €{final_cash:,.2f}")
            print(f"📊 Numero ordini: {len(orders)}")
            if orders:
                print("📌 Parametri:", {k: orders[0][k] for k in ['SL', 'TP', 'RSI Entry', 'RSI Exit', 'BB Std', 'Exposure']})
            found = True
            break
        current_id += 1

    if found:
        break

if not found:
    print(f"❌ Combinazione {target_id} non trovata.")
