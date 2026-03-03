# Persisting Annotation Data

## Background
All annotations from all datasets are saved to a single file `exported_annotation_data.json` inside the container at `/app/exported_annotation_data.json`. This file is **not** mounted to the host by default, so it will be lost if the container is removed.

## Steps to Persist the Annotation File

### 1. Check the container and file
```bash
docker ps
docker exec -it mlex_tomo_framework-mlex_segmentation-1 ls -la /app
```

The file exists and it's 27MB:
```
-rw-r--r-- 1 root root 27228573 Feb 18 00:51 exported_annotation_data.json
```

### 2. Copy the file out of the container before adding the volume mount
```bash
docker cp mlex_tomo_framework-mlex_segmentation-1:/app/exported_annotation_data.json ./exported_annotation_data.json
```

### 3. Place it in the host directory you plan to mount
```bash
mkdir -p ./data/annotations
mv ./exported_annotation_data.json ./data/annotations/
```

### 4. Update `docker-compose.yml` to add a volume mount
```yaml
mlex_segmentation:
    volumes:
      - ./data/annotations:/app/data
    environment:
      - EXPORT_FILE_PATH=/app/data/exported_annotation_data.json
      # ... other env vars
```

### 5. Restart the containers
```bash
docker-compose down
docker-compose up -d
```

## Notes
- All annotations from all datasets are stored in the **same file**, differentiated by the `source` field (image URI)
- Without the volume mount, the file only exists inside the container and will be lost on `docker-compose down`
- With the volume mount, the file persists on the host machine across container restarts and recreations
