from body import BodyThread
import time
from sys import exit

thread = BodyThread()
thread.start()

i = input()
print("Exiting…")
thread.stop()
time.sleep(0.5)
exit()
