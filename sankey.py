import pyshark
import pandas as pd
import numpy as np

import holoviews as hv
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as pex


df = pd.read_csv('/Users/colineritz/Desktop/Year 4/Master project/petiteconversation.csv')
df['Time']=pd.to_datetime(df['Time'])
df.set_index('Time', inplace=True)
df.reset_index(drop=True, inplace=True)
df.drop(columns = ['No.'])
df.drop(columns = ['Protocol'])
df.drop(columns = ['Info'])
df_grouped = df.groupby(by=["Source","Destination"]).sum()[["Length"]]
df_grouped = df_grouped.reset_index()
df_grouped.head()

IP = ["192.168.1.113", "143.204.65.104","192.168.1.113"]
IP_test = df_grouped[df_grouped.Destination.isin(IP)]

hv.Sankey(IP_test)


# fig = go.Figure(data=[go.Sankey(
#     node = dict(
#       pad = 15,
#       thickness = 20,
#       line = dict(color = "black", width = 0.5),
#       label = ["A1", "A2", "B1", "B2", "C1", "C2"],
#       customdata = ["Long name A1", "Long name A2", "Long name B1", "Long name B2",
#                     "Long name C1", "Long name C2"],
#       hovertemplate='Node %{customdata} has total value %{value}<extra></extra>',
#       color = "blue"
#     ),
#     link = dict(
#       source = ['01:80:c2:ef:03:fe', '01:80:c2:ef:03:fe','01:80:c2:ef:03:fe','01:80:c2:ef:03:fe','01:80:c2:ef:03:fe'], # indices correspond to labels, eg A1, A2, A2, B1, ...
#       target = ['d4:86:60:14:fb:9e','d4:86:60:14:fb:9e','d4:86:60:14:fb:9e','d4:86:60:14:fb:9e','d4:86:60:14:fb:9e'],
#       value = [111,111,111,8,10],
#       customdata = ["q","r","s","t","u","v"],
#       hovertemplate='Link from node %{source.customdata}<br />'+
#         'to node%{target.customdata}<br />has value %{value}'+
#         '<br />and data %{customdata}<extra></extra>',
#   ))])

# fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)
# fig.show()


# import plotly.graph_objects as go

# fig = go.Figure(data=[go.Sankey(
#     node = dict(
#       pad = 15,
#       thickness = 20,
#       line = dict(color = "black", width = 0.5),
#       label = ["A1", "A2", "B1", "B2", "C1", "C2"],
#       customdata = ["Long name A1", "Long name A2", "Long name B1", "Long name B2",
#                     "Long name C1", "Long name C2"],
#       hovertemplate='Node %{customdata} has total value %{value}<extra></extra>',
#       color = "blue"
#     ),
#     link = dict(
#       source = [0, 1, 0, 2, 3, 3], # indices correspond to labels, eg A1, A2, A2, B1, ...
#       target = [2, 3, 3, 4, 4, 5],
#       value = [8, 4, 2, 8, 4, 2],
#       customdata = ["q","r","s","t","u","v"],
#       hovertemplate='Link from node %{source.customdata}<br />'+
#         'to node%{target.customdata}<br />has value %{value}'+
#         '<br />and data %{customdata}<extra></extra>',
#   ))])

# fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)
# fig.show()

