import RPi.GPIO as GPIO
from time import sleep
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
buzzer=18
GPIO.setup(buzzer,GPIO.OUT)
while True:
    GPIO.output(buzzer, GPIO.HIGH)
    print("Buzzer On")
    sleep(1)
    GPIO.output(buzzer, GPIO.LOW)
    print("Buzzer OFF")
    sleep(1)
