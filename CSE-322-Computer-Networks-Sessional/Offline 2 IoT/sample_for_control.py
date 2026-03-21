import paho.mqtt.client as mqtt

broker = "broker.hivemq.com"
topic = "buet/cse/2105084/led" # TODO: Put the same topic you used in the ESP code

client = mqtt.Client()
client.connect(broker)

# TODO: the following is an example of publishing a message. You have to modify it so that the python code will run infinitely and wait for input from keyboard. If user presses 'y', it will send "ON"; it will send "OFF" if 'n' is pressed. The program will terminate if user presses 'q'.
while True:
    cmd = input("Enter your Command:y for turning LED ON,n for turning LED OFF,q for quitting: ")
    if cmd == 'y':
        client.publish(topic,"ON")
    elif cmd == 'n':
        client.publish(topic,"OFF")
    elif cmd == 'q':
        break
    else:
        print("Try Again from commands: y/n/q.")

