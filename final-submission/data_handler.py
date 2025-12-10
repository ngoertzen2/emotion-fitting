import pickle, os, numpy as np

while True :
    action = input("What would you like to do?\n1. read file\n2. combine files\n3. remove duplicates\n4. exit\n")
    match action:
        case "1":
            file = input("File name: ")
            if os.path.exists(file):
                with open(file, "rb") as f:
                    data = pickle.load(f)
                points = data.get("points", [])
                values = data.get("values", [])
                print(f"Loaded {len(values)} samples from {file}")
                for i in range(len(values)) :
                    print(f"Value: {values[i]}, {len(points[i])} points.")
            else:
                print(f"File not found. Exiting.")
                break

        case "2":
            f1 = input("File 1 name: ")
            f2 = input("File 2 name: ")
            f3 = input("Output file title: ")
            if os.path.exists(f1) and os.path.exists(f2) and not os.path.exists(f3):
                with open(f1, "rb") as f:
                    data = pickle.load(f)
                f1_points = data.get("points", [])
                f1_values = data.get("values", [])
                print(f"Loaded {len(f1_values)} samples from {f1}")

                with open(f2, "rb") as f:
                    data = pickle.load(f)
                f2_points = data.get("points", [])
                f2_values = data.get("values", [])
                print(f"Loaded {len(f2_values)} samples from {f2}")

                f3_points = f1_points + f2_points
                f3_values = f1_values + f2_values
                with open(f3, 'wb') as f:
                    pickle.dump({'points': f3_points, 'values': f3_values}, f)
                    print(f"Loaded {len(f3_values)} samples into {f3}")
            else:
                print(f"File(s) not found. Exiting.")
                break
        case "3":
            file = input("File name: ")
            if os.path.exists(file):
                with open(file, "rb") as f:
                    data = pickle.load(f)
                points = data.get("points", [])
                values = data.get("values", [])
                print(f"Loaded {len(values)} samples from {file}")

                points = np.array(points)
                values = np.array(values)
                new_points = []
                new_values = []
                for i in range(len(values)) :
                    p = tuple(points[i])
                    if p not in [tuple(x) for x in new_points]:
                        new_points.append(points[i])
                        new_values.append(values[i])
                
                with open(file, 'wb') as f:
                    pickle.dump({'points': new_points, 'values': new_values}, f)

                print(f"Removed {len(values) - len(new_values)} duplicate samples in {file}, {len(new_values)} samples remaining.")

            else:
                print(f"File not found. Exiting.")
                break
        case _:
            break
    