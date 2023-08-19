import networkx as nx
import matplotlib.pyplot as plt

# 创建一个空的有向图
G = nx.DiGraph()

# 添加节点
G.add_node("A", node_type="source")
G.add_node("B", node_type="intermediate")
G.add_node("C", node_type="intermediate")
G.add_node("D", node_type="sink")

# 添加边（连接节点）并指定边权重
G.add_edge("A", "B", weight=0.6)
G.add_edge("A", "C", weight=0.2)
G.add_edge("B", "C", weight=0.4)
G.add_edge("C", "D", weight=0.7)

# 根据节点类型设置节点颜色
node_colors = {"source": "blue", "intermediate": "green", "sink": "red"}
colors = [node_colors[G.nodes[node]["node_type"]] for node in G.nodes()]

# 根据边权重设置边的粗细
edge_widths = [G.edges[edge]["weight"] * 5 for edge in G.edges()]

# 绘制图形
pos = nx.spring_layout(G)  # 使用Spring布局算法进行节点布局
nx.draw(G, pos, with_labels=True, node_color=colors, edge_color="gray", node_size=1000, width=edge_widths, arrows=True)

# 显示图形
plt.show()
