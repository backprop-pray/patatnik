$DataDir = "W:\Papers\patatnik\data"
$DownloadsDir = "C:\Users\Vasil\Downloads"

Write-Host "Cleaning previous partial data..." -ForegroundColor Cyan
if (Test-Path $DataDir) {
    Remove-Item -Path $DataDir -Recurse -Force
}

Write-Host "Creating fresh data directories..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path "$DataDir\plantdoc" | Out-Null
New-Item -ItemType Directory -Force -Path "$DataDir\plantvillage" | Out-Null

Write-Host "Unzipping PlantDoc dataset..." -ForegroundColor Yellow
if (Test-Path "$DownloadsDir\plantdoc-dataset.zip") {
    Expand-Archive -LiteralPath "$DownloadsDir\plantdoc-dataset.zip" -DestinationPath "$DataDir\plantdoc" -Force
} else {
    Write-Host "plantdoc-dataset.zip not found in $DownloadsDir" -ForegroundColor Red
}

Write-Host "Unzipping PlantVillage dataset..." -ForegroundColor Yellow
if (Test-Path "$DownloadsDir\plantvillage.zip") {
    Expand-Archive -LiteralPath "$DownloadsDir\plantvillage.zip" -DestinationPath "$DataDir\plantvillage" -Force
} else {
    Write-Host "plantvillage.zip not found in $DownloadsDir" -ForegroundColor Red
}

Write-Host "All datasets have been unzipped into $DataDir" -ForegroundColor Green
