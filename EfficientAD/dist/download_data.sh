#!/bin/bash
# 📂 Path configuration
DATA_DIR="/Users/vasilvasilev/Desktop/Projects/patatnik/data"
DOWNLOADS_DIR="/Users/vasilvasilev/Downloads"

# Clear any previous partial unzips to prevent corruption
echo "🧹 Cleaning previous partial data..."
rm -rf "$DATA_DIR"

echo "📂 Creating fresh data directories..."
mkdir -p "$DATA_DIR/plantdoc"
mkdir -p "$DATA_DIR/plantvillage"

echo "⚡ Unzipping PlantDoc dataset..."
unzip -q "$DOWNLOADS_DIR/plantdoc-dataset.zip" -d "$DATA_DIR/plantdoc"

echo "⚡ Unzipping PlantVillage dataset..."
unzip -q "$DOWNLOADS_DIR/plantvillage.zip" -d "$DATA_DIR/plantvillage"

echo "✅ All datasets have been unzipped into $DATA_DIR"
