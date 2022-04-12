#final
import pymongo
from pymongo import MongoClient
import pyshark
import time

#CONNECT TO MONGODB
#URI (Uniform Resource Identifier) to connect to MongoDB database 
client_URI = "mongodb+srv://coline:clondonc@cluster0.4olg7.mongodb.net/myFirstDatabase?retryWrites=true&w=majority" 
client = MongoClient(client_URI)
db = client.Tshark 
test_collection = db.test #use or create a collection named 

#DEFINE THE INTERFACE 
networkInterface = "eth0"

#DEFINE CAPTURE OBJECT 
capture = pyshark.LiveCapture(networkInterface)

#CAPTURE 
while True: 

    for packet in capture.sniff_continuously():
                
        try:
            # get timestamp
            
                        
            protocol = packet.transport_layer
            source_address = packet.ip.src
            source_port = packet[packet.transport_layer].srcport
            destination_address = packet.ip.dst
            destination_port = packet[packet.transport_layer].dstport
            packet_time = packet.sniff_time
            packet_length = packet.length

            netpacket = { 
                                        'Packet Timestamp': packet_time, 
                                        'Protocol type': protocol,
                                        'Source address': source_address,
                                        'Source port': source_port,
                                        'Packet length': packet_length,
                                        'Destination address': destination_address,
                                        'Destination port': destination_port 
                                        }
            #capture_packets.append(netpacket)                
            print("netpacket",netpacket)

            if bool(netpacket) == False:
                print("not working")

            if bool(netpacket) == True:
                    
                test_collection.insert_one(netpacket)
                test_collection.find({}, {'_id': False})
                print("working")
                           
                            
                

        except AttributeError as e:
                    #  ignore packets other than TCP, UDP and IPv4
            pass
                
    #         #print (" ")

    # print (capture_packets)
