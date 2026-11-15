"""Final listing: figure sizes (inches at 600 dpi), PDF sanity (header, trailer, all referenced objects exist), key medians."""
import csv, os, re, glob
from PIL import Image
VM = "/mnt/data_3T/peilin/ce6190_p2/videometrics"
for png in sorted(glob.glob(VM + "/figures/*.png") + glob.glob(VM + "/figures/masks/*.png")):
    w, h = Image.open(png).size
    pdf = png[:-4] + ".pdf"
    b = open(pdf, "rb").read()
    objs = set(int(x) for x in re.findall(rb"(\d+) 0 obj", b))
    refs = set(int(x) for x in re.findall(rb"(\d+) 0 R", b))
    ok = b.startswith(b"%PDF") and b.rstrip().endswith(b"%%EOF") and refs <= objs
    print("%-60s %.2f x %.2f in  pdf_ok=%s" % (os.path.relpath(png, VM), w / 600, h / 600, ok))
print()
for r in csv.DictReader(open(VM + "/results/physiq_means.csv")):
    print("physiq", r["method"], r["metric"], "mean %.5f median %.5f" % (float(r["mean"]), float(r["median"])))
for r in csv.DictReader(open(VM + "/results/noref_means.csv")):
    if r["metric"] == "flow_px_s" and r["group"] in ("VideoPhy2", "VBench", "Physics-IQ", "FinePhyEval"):
        print("flow", r["group"], r["method"], "mean %.2f median %.2f" % (float(r["mean"]), float(r["median"])))
print(open(VM + "/figures/FONT_USED.txt").read())
