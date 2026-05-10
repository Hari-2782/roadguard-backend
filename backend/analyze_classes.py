import os
import xml.etree.ElementTree as ET
from collections import Counter

dataset_path = r"C:\Users\mural\OneDrive\Desktop\RESEARCH\road_signboard_detection_dataset\train"

def analyze_classes():
    classes = Counter()
    xml_files = [f for f in os.listdir(dataset_path) if f.endswith('.xml')]
    
    print(f"Scanning {len(xml_files)} XML files...")
    
    for xml_file in xml_files:
        tree = ET.parse(os.path.join(dataset_path, xml_file))
        root = tree.getroot()
        
        for obj in root.findall('object'):
            name = obj.find('name').text
            classes[name] += 1
            
    print("\n--- Class Distribution (All) ---")
    keys = sorted(classes.keys())
    for name in keys:
        print(f"'{name}': {classes[name]}")
        
    print("\n--- Targeted Search ---")
    for name in keys:
        lower = name.lower()
        if "speed" in lower or "limit" in lower or "junction" in lower or "cross" in lower or "merge" in lower:
            print(f"FOUND RELEVANT: '{name}'")

if __name__ == "__main__":
    analyze_classes()
