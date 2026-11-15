"""Check every figure PDF: header, %%EOF, every 'N 0 R' reference has an 'N 0 obj', page size,
fonts embedded (FontFile2). Also try pypdf / fitz if present.
Usage: python q_pdfcheck.py DIR"""
import glob, os, re, sys
d = sys.argv[1]
for f in sorted(glob.glob(os.path.join(d, '*.pdf'))):
    b = open(f, 'rb').read()
    objs = set(int(m) for m in re.findall(rb'(\d+) 0 obj', b))
    refs = set(int(m) for m in re.findall(rb'(\d+) 0 R', b))
    missing = sorted(refs - objs)
    mb = re.search(rb'/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]', b)
    size = (float(mb.group(3)) / 72, float(mb.group(4)) / 72) if mb else None
    fonts = sorted(set(m.decode() for m in re.findall(rb'/BaseFont\s*/([A-Za-z0-9+\-_]+)', b)))
    ok = b.startswith(b'%PDF') and b.rstrip().endswith(b'%%EOF') and not missing
    print(f"{os.path.basename(f):48s} ok={ok} bytes={len(b)} objs={len(objs)} missing_refs={missing[:5]} "
          f"size_in={tuple(round(x, 3) for x in size) if size else None} fonts={fonts} embedded={b.count(b'/FontFile2')}")
try:
    import pypdf
    for f in sorted(glob.glob(os.path.join(d, '*.pdf'))):
        r = pypdf.PdfReader(f)
        print('pypdf', os.path.basename(f), 'pages', len(r.pages))
except Exception as e:
    print('pypdf not usable:', type(e).__name__, e)
