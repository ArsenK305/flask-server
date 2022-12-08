#!/usr/bin/env python
"""Show messages in two new console windows simultaneously."""
import subprocess
import sys
import platform
from subprocess import Popen, call
import os
import json
from time import sleep

messages = 'This is Console1'

# define a command that starts new terminal
if platform.system() == "Windows":
    new_window_command = "cmd.exe /c start".split()
else:  # XXX this can be made more portable
    new_window_command = "x-terminal-emulator -e".split()

with open('json_data.json', encoding='utf-8') as json_file:
    data = json.load(json_file)
    print(data)

print_text = "print({})".format(data)
print_text_encoded = print_text.encode()

# open new consoles, display messages
def create_process():
    proc = Popen("cmd.exe /c start python.exe print{}".format(print_text_encoded).split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, creationflags=subprocess.CREATE_NEW_CONSOLE)
    proc.stdin.write(print_text_encoded)
    proc.wait()
    input('Press Enter to Exit')
    # os.system("start cmd /c {}\\python.exe ".format(os.getcwd))
    # call(['python', os.getcwd() + '\\Mask_RCNN-master\\mrcnn\\' + 'output_to_cmd.py'])
    # echo = [sys.executable, "-c",
    #         "import sys; print(str(sys.argv[1:])); input('Press Enter..')"]
    # processes = Popen(new_window_command + echo + [json_dictionary])

# wait for the windows to be closed
#     processes.wait()

# cmd.exe /c start sys.executable, "-c", "import sys; print(sys.argv[1]); input('Press Enter..')"
