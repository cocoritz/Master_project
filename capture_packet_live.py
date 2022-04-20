import pyshark
import datetime
import time 


while True:
    
    now = datetime.datetime.now()
    time_string = now. strftime("%Y-%m-%d_%H-%M-%S")
    current_date_and_time_string = str(time_string)
    extension = ".pcap"
    path = "/Users/colineritz/Desktop/Master_project/Data/pcap/"
    name= "packets_"
    file =  path + name + current_date_and_time_string + extension
    

    #file_name =  current_date_and_time_string + extensionfile = f"/Users/colineritz/Desktop/Master_project/Data/pcap/packets:{datetime.datetime.now():%Y-%m-%d %H-%m-%d}.pcap"
    #file = "/Users/colineritz/Desktop/Master_project/Data/pcap/packets" + str(i) + ".pcap"
    #i +=1
    capture = pyshark.LiveCapture(interface="en0", output_file=file, bpf_filter='host 192.168.1.6')
    capture.sniff(timeout=300)
