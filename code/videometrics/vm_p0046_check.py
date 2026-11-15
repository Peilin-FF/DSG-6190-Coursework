"""Side-by-side check image: switch frame, GT frames 0 and 40, CorrFlow frame 0 for case p0046 (and p0019 as a control)."""
import os, sys
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vm_common as C
out = []
for cid in ["physics-iq-veried-p0046", "physics-iq-veried-p0133"]:
    c = [x for x in C.load_cases() if x["id"] == cid][0]
    sw = cv2.imread(os.path.join(C.MEDIA, "public", "media", "video", "physics-iq-veried", "01_input", c["stem"] + "_switch-frame.jpg"))
    sw = cv2.resize(sw, (320, 180), interpolation=cv2.INTER_AREA)
    gt = [v for v in c["videos"] if v["method"] == "Ground truth"][0]
    g = C.read_all_frames(gt["path"])
    ours = [v for v in c["videos"] if v["method"] == "CorrFlow"][0]
    o = C.read_all_frames(ours["path"])
    tiles = [sw] + [cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2BGR), (320, 180), interpolation=cv2.INTER_AREA) for f in (g[0], g[75], o[0], o[40])]
    out.append(np.concatenate(tiles, axis=1))
img = np.concatenate(out, axis=0)
p = os.path.join(C.VM, "logs", "p0046_check.jpg")
cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
print(p, img.shape)
