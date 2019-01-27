import os

def tree_to_graphviz_str(root_node, prefix='0'):
    root_node_label = f"feat={root_node.feature_ix},value={root_node.value}"
    ostr = f'{prefix} [label="{root_node_label}"];\n'

    if isinstance(root_node.left, float):
        ostr += f'{prefix}0 [label="{root_node.left}"];\n {prefix}->{prefix}0;\n'
    else:
        ostr += tree_to_graphviz_str(root_node.left, f'{prefix}0') + f'{prefix}->{prefix}0;\n'

    if isinstance(root_node.right, float):
        ostr += f'{prefix}1 [label="{root_node.right}"];\n {prefix}->{prefix}1;\n'
    else:
        ostr += tree_to_graphviz_str(root_node.right, f'{prefix}1') + f'{prefix}->{prefix}1;\n'

    return ostr


def tree_to_pdf(root_node, out_fn_base):
    with open(out_fn_base + '.dot', 'w') as fp:
        fp.write('digraph G {\n' + tree_to_graphviz_str(root_node) + '\n}')
    os.system(f'dot -Tpdf -o {out_fn_base + ".pdf"} {out_fn_base + ".dot"}')
