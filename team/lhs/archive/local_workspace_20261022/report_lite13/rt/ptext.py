import zipfile,re,sys
x=zipfile.ZipFile(sys.argv[1]).read("word/document.xml").decode()
for i,p in enumerate(re.findall(r"<w:p[ >].*?</w:p>",x,re.S)):
    t="".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>",p))
    if t.strip(): print(i,t)
