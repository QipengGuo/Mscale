import numpy as NP
import sys, getopt

try:
    opts, args = getopt.getopt(sys.argv[1:], "h", ["in_fname=", "out_fname="])
except getopt.GetoptError:
    print 'Usage, please type --ptb, --bbc, --imdb, --wiki, to determine which dataset, --train or --test, and the model can be --baseline, --context_free, --context_dependent'
    sys.exit(2)

in_fname = None
out_fname = None
for opt, arg in opts:
    if opt == '--in_fname':
        in_fname = arg
    if opt == '--out_fname':
        out_fname = arg

print in_fname, out_fname
f = open(in_fname, 'r')
sl = f.readlines()
f.close()
wf = open(out_fname, 'w')
for s in sl:
    st = 0
    ed = len(s)-1
    while st<len(s) and s[st]==' ':
        st+=1
    while ed>0 and s[ed]==' ' or s[ed]=='\n':
        ed-=1
    if (s[st]=='<' or s[st]==u'<') and (s[ed]=='>' or s[ed]==u'>'):
        while st<len(s) and not (s[st]=='>' or s[st]==u'>'):
            st+=1
        while ed>0 and not (s[ed]=='<' or s[ed]==u'<'):
            ed-=1
        if st<ed:
            wf.write(s[st+1:ed]+'\n')
wf.close()
