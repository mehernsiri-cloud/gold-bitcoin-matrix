# roi_safe_update.py
import os
import json
import shutil

ROI_PATH = 'data/roi_data.json'
BACKUP_PATH = 'data/roi_data_backup.json'

# Backup previous ROI
if os.path.exists(ROI_PATH):
    shutil.copy2(ROI_PATH, BACKUP_PATH)

# Run the main scraper
try:
    from update_roi_data import main
    main()
except Exception as e:
    print(f"⚠️ update_roi_data.py failed: {e}")

# Restore backup if new ROI is empty
try:
    with open(ROI_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not data:
        raise ValueError("ROI data empty")
except Exception:
    print("⚠️ ROI data empty, restoring previous backup")
    if os.path.exists(BACKUP_PATH):
        shutil.copy2(BACKUP_PATH, ROI_PATH)
