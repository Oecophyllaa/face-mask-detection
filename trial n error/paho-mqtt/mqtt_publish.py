import paho.mqtt.publish as publish

publish.single("CoreElectronics/test", "On", hostname="test.mosquitto.org")
# publish.single("CoreElectronics/topic", "Off", hostname="test.mosquitto.org")
print("Done")