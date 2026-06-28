import os, cv2, numpy as np

class YOLOv8ONNX:
    """Detecteur YOLOv8 via ONNX. Backend onnxruntime si dispo (plus rapide sur CPU),
    sinon repli sur OpenCV DNN. Aucune dependance torch/ultralytics au runtime."""
    BALL_CLASS = 32   # 'sports ball' dans COCO
    PERSON_CLASS = 0  # 'person' -> le bas de sa boite ~ pieds au sol

    def __init__(self, onnx='yolov8m.onnx', conf=0.25, iou=0.45, imgsz=640):
        self.conf, self.iou, self.sz = conf, iou, imgsz
        self.backend = None
        try:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = os.cpu_count() or 4
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.sess = ort.InferenceSession(onnx, so, providers=["CPUExecutionProvider"])
            self.inp = self.sess.get_inputs()[0].name
            self.backend = "onnxruntime"
        except Exception:
            self.net = cv2.dnn.readNetFromONNX(onnx)
            self.backend = "cv2.dnn"

    def _forward(self, bgr):
        h, w = bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(bgr, 1/255., (self.sz, self.sz), swapRB=True, crop=False)
        if self.backend == "onnxruntime":
            raw = self.sess.run(None, {self.inp: blob})[0]      # (1, 84, 8400)
        else:
            self.net.setInput(blob); raw = self.net.forward()    # (1, 84, 8400)
        return raw[0].T, w, h                                    # (8400, 84)

    def _best_box(self, out, w, h, cls, conf):
        """Meilleure boite (x, y, bw, bh, score) de la classe cls apres NMS, ou None."""
        cc = out[:, 4 + cls]
        keep = cc > conf
        boxes, scores = [], []
        for (cx, cy, bw, bh, *_), sc in zip(out[keep], cc[keep]):
            x = (cx - bw/2)/self.sz*w; y = (cy - bh/2)/self.sz*h
            boxes.append([int(x), int(y), int(bw/self.sz*w), int(bh/self.sz*h)])
            scores.append(float(sc))
        if not boxes:
            return None
        idx = cv2.dnn.NMSBoxes(boxes, scores, conf, self.iou)
        if len(idx) == 0:
            return None
        flat = np.array(idx).flatten()
        i = int(flat[np.argmax([scores[int(j)] for j in flat])])
        bx, by, bw, bh = boxes[i]
        return (bx, by, bw, bh, scores[i])

    # --- UNE seule inference -> ballon + candidats + boite personne --------------
    def detect_ball_and_candidates(self, bgr, low_conf=0.05):
        """Retourne (best, candidates, person) en UN forward pass :
          best       = (x, y, r, conf) du ballon apres NMS, ou None
          candidates = [(x, y, r, score), ...] ballon au-dessus de low_conf (gapfill)
          person     = (x, y, w, h) de la meilleure boite 'personne', ou None
                       (le bas y+h ~ pieds au sol, pour la hauteur sol->ballon)."""
        out, w, h = self._forward(bgr)
        bc = out[:, 4 + self.BALL_CLASS]

        kc = bc > low_conf
        cands = [(float(cx/self.sz*w), float(cy/self.sz*h),
                  float(max(bw, bh)/self.sz*w/2), float(sc))
                 for (cx, cy, bw, bh, *_), sc in zip(out[kc], bc[kc])]

        bb = self._best_box(out, w, h, self.BALL_CLASS, self.conf)
        best = (bb[0] + bb[2]/2, bb[1] + bb[3]/2, max(bb[2], bb[3])/2, bb[4]) if bb else None

        pb = self._best_box(out, w, h, self.PERSON_CLASS, 0.35)
        person = (pb[0], pb[1], pb[2], pb[3]) if pb else None
        return best, cands, person

    # --- API historiques (compat) : reutilisent le forward unique ---------------
    def detect_ball(self, bgr):
        return self.detect_ball_and_candidates(bgr)[0]

    def detect_ball_candidates(self, bgr, conf=0.05):
        return self.detect_ball_and_candidates(bgr, low_conf=conf)[1]


# --- cache module : charge le modele UNE SEULE FOIS (evite ~3.7s par analyse) -- #
_CACHE = {}
def get_detector(onnx='yolov8m.onnx', conf=0.10, iou=0.45, imgsz=640):
    key = (os.path.abspath(onnx), conf, iou, imgsz)
    det = _CACHE.get(key)
    if det is None:
        det = YOLOv8ONNX(onnx, conf=conf, iou=iou, imgsz=imgsz)
        _CACHE[key] = det
    return det


if __name__ == "__main__":
    import glob
    det = get_detector()
    print("backend:", det.backend)
    files = sorted(glob.glob('v3/f_*.png'))
    hits = sum(1 for i in range(0, len(files), 47) if det.detect_ball(cv2.imread(files[i])))
    print("ball found in", hits, "sampled frames")
