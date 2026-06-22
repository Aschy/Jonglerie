import cv2, numpy as np

class YOLOv8ONNX:
    """Detecteur YOLOv8 via OpenCV DNN (aucune dependance torch/ultralytics)."""
    BALL_CLASS = 32  # 'sports ball' dans COCO
    def __init__(self, onnx='yolov8m.onnx', conf=0.25, iou=0.45, imgsz=640):
        self.net = cv2.dnn.readNetFromONNX(onnx)
        self.conf, self.iou, self.sz = conf, iou, imgsz

    def _forward(self, bgr):
        h, w = bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(bgr, 1/255., (self.sz, self.sz), swapRB=True, crop=False)
        self.net.setInput(blob)
        return self.net.forward()[0].T, w, h     # (8400, 84)

    def detect_ball(self, bgr):
        """Meilleur ballon (x,y,r,conf) apres NMS, ou None."""
        out, w, h = self._forward(bgr)
        ball_conf = out[:, 4 + self.BALL_CLASS]
        keep = ball_conf > self.conf
        boxes, scores = [], []
        for (cx, cy, bw, bh, *_), sc in zip(out[keep], ball_conf[keep]):
            x = (cx - bw/2)/self.sz*w; y = (cy - bh/2)/self.sz*h
            boxes.append([int(x), int(y), int(bw/self.sz*w), int(bh/self.sz*h)]); scores.append(float(sc))
        if not boxes:
            return None
        idx = cv2.dnn.NMSBoxes(boxes, scores, self.conf, self.iou)
        if len(idx) == 0:
            return None
        flat = np.array(idx).flatten()
        i = int(flat[np.argmax([scores[int(j)] for j in flat])])
        bx, by, bw, bh = boxes[i]
        return (bx + bw/2, by + bh/2, max(bw, bh)/2, scores[i])

    def detect_ball_candidates(self, bgr, conf=0.05):
        """TOUS les candidats ballon (x,y,r,score) au-dessus d'un seuil bas,
        pour gating par position predite (rattrape le ballon flou score faiblement)."""
        out, w, h = self._forward(bgr)
        bc = out[:, 4 + self.BALL_CLASS]
        keep = bc > conf
        return [(float(cx/self.sz*w), float(cy/self.sz*h),
                 float(max(bw, bh)/self.sz*w/2), float(sc))
                for (cx, cy, bw, bh, *_), sc in zip(out[keep], bc[keep])]


if __name__ == "__main__":
    import glob
    det = YOLOv8ONNX()
    files = sorted(glob.glob('v3/f_*.png'))
    hits = 0
    for i in range(0, len(files), 47):
        if det.detect_ball(cv2.imread(files[i])): hits += 1
    print("ball found in", hits, "sampled frames")
