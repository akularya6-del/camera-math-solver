import json

with open("exam_source.html", "r", encoding="utf-8") as f:
    html = f.read()

import re
# Extract the JS array
m = re.search(r"const questions = \[(.*?)\];", html, re.DOTALL)
if m:
    pass # we can parse it if needed
else:
    print("Could not find questions in HTML")
