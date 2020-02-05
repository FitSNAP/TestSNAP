#!/usr/bin/python 
import glob

sentinelline = "// ----------------------------------------------------------------------\n"
headerfilename = "HEADER"
headerfile = open(headerfilename,'r')
headerlines = headerfile.readlines()
headerfile.close()

targetfiles = glob.glob("src/*.h") + glob.glob("src/*.cpp")
for targetfilename in targetfiles:
    print targetfilename
    targetfile = open(targetfilename,'r')
    targetlines = targetfile.readlines()
    targetfile.close()
    targetfile = open(targetfilename,'w')

    for line in headerlines:
        print >>targetfile, line,

    toggle = 1
    for line in targetlines:
        if line == sentinelline:
            toggle = -toggle
        elif toggle == 1:
            print >>targetfile, line,
    targetfile.close()
