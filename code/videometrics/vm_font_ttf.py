"""F7: convert the TeX Gyre Termes OpenType-CFF faces to TrueType (quadratic) outlines.

Why: matplotlib 3.11 writes a CFF-OpenType program into a FontFile2 stream (declared TrueType) when
pdf.fonttype = 42, which is not a valid PDF font embedding. TrueType outlines embed correctly (Type 42,
FontFile2 with a glyf table), like the DejaVu Serif fonts before.
Method: the fontTools otf2ttf recipe (Cu2QuPen, max error 1 font unit, reversed contour direction).
Usage: python vm_font_ttf.py SRC_DIR_WITH_OTF DEST_DIR (called by vm_font_setup.sh)
"""
import glob
import os
import sys

from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont, newTable


def otf_to_ttf(font, max_err=1.0):
    assert font.sfntVersion == "OTTO" and "CFF " in font
    order = font.getGlyphOrder()
    gs = font.getGlyphSet()
    font["loca"] = newTable("loca")
    font["glyf"] = glyf = newTable("glyf")
    glyf.glyphOrder = order
    glyf.glyphs = {}
    for name in order:
        pen = TTGlyphPen(gs)
        gs[name].draw(Cu2QuPen(pen, max_err, reverse_direction=True))
        glyf.glyphs[name] = pen.glyph()
    del font["CFF "]
    if "VORG" in font:
        del font["VORG"]
    glyf.compile(font)
    hmtx = font["hmtx"]
    for name, g in glyf.glyphs.items():
        if hasattr(g, "xMin"):
            hmtx[name] = (hmtx[name][0], g.xMin)
    font["maxp"] = maxp = newTable("maxp")
    maxp.tableVersion = 0x00010000
    maxp.maxZones = 1
    maxp.maxTwilightPoints = 0
    maxp.maxStorage = 0
    maxp.maxFunctionDefs = 0
    maxp.maxInstructionDefs = 0
    maxp.maxStackElements = 0
    maxp.maxSizeOfInstructions = 0
    maxp.maxComponentElements = max(len(getattr(g, "components", []) or []) for g in glyf.glyphs.values())
    maxp.compile(font)
    post = font["post"]
    post.formatType = 2.0
    post.extraNames = []
    post.mapping = {}
    post.glyphOrder = order
    font.sfntVersion = "\000\001\000\000"
    return font


def main():
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(src, "*.otf"))):
        out = os.path.join(dst, os.path.basename(f)[:-4] + ".ttf")
        font = otf_to_ttf(TTFont(f))
        font.save(out)
        chk = TTFont(out)
        print("%s -> %s glyphs %d glyf %s CFF %s" % (os.path.basename(f), out, len(chk.getGlyphOrder()),
                                                    "glyf" in chk, "CFF " in chk))


if __name__ == "__main__":
    main()
