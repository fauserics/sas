import os, pathlib
print("hello from node")
print("CWD:", os.getcwd())
pathlib.Path("ms_test.txt").write_text("ok")
