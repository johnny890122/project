import sys, os, json
from FINDER import FINDER
from io import BytesIO
import networkx as nx
from networkx.readwrite import json_graph

def converter(dct):
    tmp_dct = dict()
    cnt = 0
    for node in dct["nodes"]:
        node["label"] = node["id"]
        tmp_dct["node_"+str(cnt)]= node
        cnt += 1

    for edge in dct["links"]:
        tmp_dct["edge_"+str(cnt)] = edge
        cnt += 1

    str_ = str(json.dumps(tmp_dct, indent=2, ensure_ascii=False))

    for idx in range(cnt, -1, -1):
        str_ = str_.replace("_"+str(idx), "")
    str_ = "graph [" + str_[1:]
    str_ = str_.replace('\"', '').replace(',', '').replace(':', '').replace('{', '[').replace('}', ']')

    return str_

def convert_to_FINDER_format(G):
    data = json_graph.node_link_data(G)
    str_ = converter(data)

    return str_

def data_model_pairs(name:str):
    data = f"../data/empirical/{name}.gml"
    G = nx.read_gml(data)
    map_dct = {node: int(idx) for idx, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, map_dct, copy=True)
    content = BytesIO(convert_to_FINDER_format(G).encode('utf-8'))
    model = f'../models/Model_EMPIRICAL/{name}.ckpt'
    return content, model

dqn = FINDER()
for name in ["HAMBURG_TIE_YEAR", "HEROIN_DEALING",  "911",
             "SWINGERS_club", "MAIL"]:
    #TODO: "suicide", "DOMESTICTERRORWEB"
    content, model = data_model_pairs(name)
    val, sol = dqn.Evaluate(content, model)
    print(name, sol)
    print("done")
    print("="*10)
