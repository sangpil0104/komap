#!/bin/bash
# wrender.sh IN.docx OUT.pdf : render with Microsoft Word via a temp copy in ~/Downloads
n=zz_sp_$RANDOM; cp "$1" ~/Downloads/$n.docx
timeout 150 osascript -e "tell application \"Microsoft Word\"
set d to open file name (POSIX path of (path to downloads folder) & \"$n.docx\")
delay 3
save as d file name (POSIX path of (path to downloads folder) & \"$n.pdf\") file format format PDF
end tell" >/dev/null 2>&1
mv ~/Downloads/$n.pdf "$2"; rm -f ~/Downloads/$n.docx; pdfinfo "$2" | grep Pages
