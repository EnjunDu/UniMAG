#!/bin/bash
# Git LFS automatic directory-by-directory commit script

folders=(
  "books-lp"
  "books-nc"
  "cloth-copurchase"
  "ele-fashion"
  "Grocery"
  "mm-code"
  "Movies"
  "Reddit-M"
  "Reddit-S"
  "sports-copurchase"
  "Toys"
)

git lfs install

git lfs track "*.pt"
git lfs track "*.tar"
git lfs track "*.tar.gz"
git lfs track "*.json"
git lfs track "*.jsonl"
git lfs track "*.csv"
git lfs track "*.pkl"
git lfs track "*.pickle"

git add .gitattributes
git commit -m "Add .gitattributes for Git LFS tracking" || echo "No changes in .gitattributes"
git push origin main --force

for folder in "${folders[@]}"; do
  if [ -d "$folder" ]; then
    echo "🟢 Adding and committing: $folder"
    git add "$folder"
    git commit -m "Add $folder dataset" || {
      echo "⚠️  Nothing new to commit in $folder"
      continue
    }

    echo "🚀 Pushing $folder to Hugging Face..."
    git push origin main --force || {
      echo "❌ Failed to push $folder. Aborting."
      exit 1
    }

  else
    echo "⚠️  Skipped: $folder (not found)"
  fi
done
