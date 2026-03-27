#!/bin/bash

# Define paths
DATA_DIR="/Users/vasilvasilev/Desktop/Projects/patatnik/data"
DOWNLOADS_DIR="/Users/vasilvasilev/Downloads"

echo "📂 Creating data directory at $DATA_DIR..."
mkdir -p "$DATA_DIR/plantdoc"
mkdir -p "$DATA_DIR/plantvillage"

echo "⚡ Unzipping PlantDoc dataset..."
unzip -q "$DOWNLOADS_DIR/plantdoc-dataset.zip" -d "$DATA_DIR/plantdoc"

echo "⚡ Unzipping PlantVillage dataset..."
unzip -q "$DOWNLOADS_DIR/plantvillage.zip" -d "$DATA_DIR/plantvillage"

echo "✅ Datasets successfully unzipped into $DATA_DIR"

# List contents for verification
echo "📊 Dataset Preview:"
ls -F "$DATA_DIR/plantdoc" | head -n 5
ls -F "$DATA_DIR/plantvillage" | head -n 5
