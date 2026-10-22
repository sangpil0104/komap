#!/bin/bash
# sheet.sh PDF OUTPNG
rm -rf _pg && mkdir _pg && pdftoppm -png -r 45 "$1" _pg/p && python3 -c "
from PIL import Image; import glob,sys
fs=sorted(glob.glob('_pg/p-*.png')); ims=[Image.open(f) for f in fs]; w,h=ims[0].size
sheet=Image.new('RGB',(w*6,h*2),'white')
for i,im in enumerate(ims): sheet.paste(im,((i%6)*w,(i//6)*h))
sheet.save('$2')"
