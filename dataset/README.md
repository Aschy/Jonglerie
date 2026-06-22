# Dataset — ballons annotés (auto-généré)

47 images extraites de la vidéo « parking », annotées automatiquement à partir des
détections YOLO sûres (conf ≥ 0,4). Format **YOLO** :

```
images/frame_XXXX.png
labels/frame_XXXX.txt    # "0 cx cy w h" (normalisé), classe 0 = ball
data.yaml
```

**Usage — fine-tuning** (recommandé d'ajouter des frames floues annotées à la main) :
```bash
yolo detect train data=dataset/data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

⚠️ Annotations auto à **vérifier/compléter** avant entraînement sérieux.
