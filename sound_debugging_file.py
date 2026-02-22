# this code for the testing and verigying the audio as a debugger.

import matplotlib.pyplot as plt
import networkx as nx

# Create directed graph
G = nx.DiGraph()

nodes = [
    "Audio (16 kHz)", 
    "Preprocessing\n(normalize + trim)", 
    "5 s Window\n(pad/crop)", 
    "Feature Extraction\n(MFCC + prosody)", 
    "CNN", 
    "BiLSTM", 
    "Attention", 
    "Dense (2-way)", 
    "Post-hoc\n(temperature + threshold τ)"
]

# Add edges in order
for i in range(len(nodes)-1):
    G.add_edge(nodes[i], nodes[i+1])

pos = nx.spring_layout(G, seed=42)  # layout

plt.figure(figsize=(12, 5))
nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=2500, font_size=8, arrows=True)
plt.title("SER Pipeline Diagram")
plt.show()
