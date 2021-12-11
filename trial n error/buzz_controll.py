import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setup(25, GPIO.OUT)

try:
    while True:
        GPIO.output(25, GPIO.HIGH)
        time.sleep(1)
        GPIO.output(25, GPIO.LOW)
        time.sleep(1)

finally:
    GPIO.cleanup()
