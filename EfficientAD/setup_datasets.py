import os
import shutil

DATA_DIR = r"W:\Papers\patatnik\data"
DIST_DIR = r"W:\Papers\patatnik\EfficientAD\dist"
MVTEC_FORMAT_DIR = os.path.join(DATA_DIR, "mvtec_plants")

def extract_and_format():
    print(f"🧹 Clearing old data from {MVTEC_FORMAT_DIR}...")
    if os.path.exists(MVTEC_FORMAT_DIR):
        shutil.rmtree(MVTEC_FORMAT_DIR)
    os.makedirs(MVTEC_FORMAT_DIR, exist_ok=True)
    
    print(f"⚡ Processing PlantVillage into MVTec Anomaly Detection Format...")
    
    # Restructure PlantVillage into MVTec AD format (train/good, test/defect)
    base_pv = os.path.join(DIST_DIR, "PlantVillage")
    if os.path.exists(os.path.join(base_pv, "PlantVillage")):
        base_pv = os.path.join(base_pv, "PlantVillage")
    
    if not os.path.exists(base_pv):
        print(f"❌ Error: Could not find PlantVillage folder at {base_pv}")
        return
        
    # Discover all plants
    plants = {}
    known_plants = ['Tomato', 'Potato', 'Pepper__bell']
    
    for foldername in os.listdir(base_pv):
        if foldername == "PlantVillage" or not os.path.isdir(os.path.join(base_pv, foldername)): 
            continue
            
        plant_name = None
        for kp in known_plants:
            if foldername.startswith(kp):
                plant_name = kp
                break
                
        if not plant_name:
            continue
            
        if plant_name not in plants:
            plants[plant_name] = {'healthy': None, 'defects': []}
        
        if "healthy" in foldername.lower():
            plants[plant_name]['healthy'] = foldername
        else:
            plants[plant_name]['defects'].append(foldername)
            
    # Master "all_plants" directory for unified training
    all_plants_dir = os.path.join(MVTEC_FORMAT_DIR, "all_plants")
    all_train_good = os.path.join(all_plants_dir, "train", "good")
    all_test_good = os.path.join(all_plants_dir, "test", "good")
    os.makedirs(all_train_good, exist_ok=True)
    os.makedirs(all_test_good, exist_ok=True)

    for plant, conditions in plants.items():
        if not conditions['healthy']: 
            continue # Can't do anomaly detection without a healthy baseline
            
        print(f"🍅 Structuring MVTec dataset for {plant} (and merging into all_plants)...")
        plant_dir = os.path.join(MVTEC_FORMAT_DIR, plant)
        
        # Train / Good
        train_good = os.path.join(plant_dir, "train", "good")
        os.makedirs(train_good, exist_ok=True)
        healthy_path = os.path.join(base_pv, conditions['healthy'])
        
        # Take 80% for training good, 20% for testing good
        healthy_imgs = [f for f in os.listdir(healthy_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        split_idx = int(len(healthy_imgs) * 0.8)
        
        for img in healthy_imgs[:split_idx]:
            # Copy to individual plant folder
            shutil.copy(os.path.join(healthy_path, img), os.path.join(train_good, img))
            # Copy to master all_plants folder
            shutil.copy(os.path.join(healthy_path, img), os.path.join(all_train_good, f"{plant}_{img}"))
            
        # Test / Good
        test_good = os.path.join(plant_dir, "test", "good")
        os.makedirs(test_good, exist_ok=True)
        for img in healthy_imgs[split_idx:]:
            shutil.copy(os.path.join(healthy_path, img), os.path.join(test_good, img))
            shutil.copy(os.path.join(healthy_path, img), os.path.join(all_test_good, f"{plant}_{img}"))
            
        # Test / Defects (Anomalies)
        for defect in conditions['defects']:
            test_defect = os.path.join(plant_dir, "test", defect)
            os.makedirs(test_defect, exist_ok=True)
            
            all_test_defect = os.path.join(all_plants_dir, "test", f"{plant}_{defect}")
            os.makedirs(all_test_defect, exist_ok=True)
            
            defect_path = os.path.join(base_pv, defect)
            for img in os.listdir(defect_path):
                if img.lower().endswith(('.jpg', '.png', '.jpeg')):
                    shutil.copy(os.path.join(defect_path, img), os.path.join(test_defect, img))
                    shutil.copy(os.path.join(defect_path, img), os.path.join(all_test_defect, f"{plant}_{img}"))
                    
    print(f"✅ MVTec Anomaly Detection dataset successfully structured at {MVTEC_FORMAT_DIR}!")
    print(f"✅ You can now run: python efficientad.py -d plants -s all_plants")

if __name__ == "__main__":
    extract_and_format()
