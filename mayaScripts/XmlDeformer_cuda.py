import csv
from collections import defaultdict
import xml.etree.ElementTree as ET
import maya.cmds as cmds
import maya.api.OpenMaya as om

# Set file path
csv_path = "C:/Users/user/Desktop/boyun/codes/backup_251208/GeodesicVoxelBinding/VertexWeight.csv"
xml_output_path = "C:/Users/user/Desktop/boyun/maya/cgxr/scripts/NewVertexWeight251215_cuda_65_1.xml"

with open(csv_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    rows = list(reader)
weights_data = rows[:] 
joint_weights = defaultdict(list)
for row in weights_data:
    vert_index = int(row[0])
    weights = list(map(float, row[1:]))
    total = sum(weights)
    for i, w in enumerate(row[1:]):
        weight = float(w)/total
        if weight > 0.0:  # Only include non-zero weights
            joint_weights[i].append((vert_index, weight))

# Confirm mesh selection
selection = cmds.ls(selection=True)
if not selection:
    cmds.error("스킨 메시를 먼저 선택하세요.")

mesh = selection[0]

# Finding Skin Clusters
def get_skin_cluster(mesh):
    history = cmds.listHistory(mesh)
    skin_clusters = cmds.ls(history, type="skinCluster")
    if skin_clusters:
        return skin_clusters[0]
    return None

skin_cluster = get_skin_cluster(mesh)
if not skin_cluster:
    cmds.error(f"{mesh}에 스킨 클러스터가 없습니다.")

# Extract joint list
joint_list = cmds.skinCluster(skin_cluster, q=True, inf=True)

# Extract shape names
shape_node = cmds.listRelatives(mesh, shapes=True)[0]

# Creating XML structure
root = ET.Element("deformerWeight")

header = ET.SubElement(root, "headerInfo")
header.set("fileName", "D:/maya/projects/cgxr/scripts/NewVertexWeight0725_64_4.xml")
header.set("worldMatrix", "1.000000 0.000000 0.000000 0.000000 0.000000 1.000000 0.000000 0.000000 0.000000 0.000000 1.000000 0.000000 0.000000 0.000000 0.000000 1.000000")

# Get Vertex Coordinates
def get_vertex_positions(mesh_name):
    sel_list = om.MSelectionList()
    sel_list.add(mesh_name)
    dag_path = sel_list.getDagPath(0)
    mfn_mesh = om.MFnMesh(dag_path)
    return mfn_mesh.getPoints(om.MSpace.kWorld)

vertex_positions = get_vertex_positions(mesh)

# Add <shape> tag
shape_elem = ET.SubElement(root, "shape")
shape_elem.set("name", shape_node)
shape_elem.set("group", "0")
shape_elem.set("stride", "3")
shape_elem.set("size", str(len(vertex_positions)))
shape_elem.set("max", str(len(vertex_positions) - 1))

for idx, pt in enumerate(vertex_positions):
    pt_elem = ET.SubElement(shape_elem, "point")
    pt_elem.set("index", str(idx))
    pt_elem.set("value", f"{pt.x:.6f} {pt.y:.6f} {pt.z:.6f}")

# Add the <weights> tag
layer = 0
for j, p in joint_weights.items():
    w_elem = ET.SubElement(root, "weights")
    w_elem.set("deformer", skin_cluster)
    if j >= len(joint_list):
        print(f"[ERROR] joint_list[{j}] 접근 불가. joint_list 길이: {len(joint_list)}")
        continue
    w_elem.set("source", joint_list[j])
    w_elem.set("shape", shape_node)
    w_elem.set("layer", str(layer))
    w_elem.set("defaultValue", "0.000")
    w_elem.set("size", str(len(p)))
    max_index = max(i for i, _ in p)
    w_elem.set("max", str(max_index))
    for index, value in p:
        point = ET.SubElement(w_elem, "point")
        point.set("index", str(index))
        point.set("value", f"{value:.3f}")
    layer += 1

# indent
def indent(elem, level=0):
    i = "\n" + level * "  " 
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i

# save XML 
indent(root)
tree = ET.ElementTree(root)
tree.write(xml_output_path, encoding='utf-8', xml_declaration=True)

print("XML 저장 완료:", xml_output_path)
