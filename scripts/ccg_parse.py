from ccg_class import tree
import re


def parse_data(ccg_data: list[str]) -> list[list[tuple[int, str]]]:
    """
    parse_data Parses a prolog tree into a python structure

    converts the structure into a list of CCG elements.
    These CCG elements contain of a list of each specific row.
    Each row has information stored in a tuple format.
    The tuple contains the depth and the raw string

    Args:
        ccg_file (IO[Any]): a file with CCG info

    Returns:
        list[list[tuple[int, str]]]: Python format specified above
    """
    # replace amount of indents with a number and set split tokens
    split_token: str = "+SPLIT+"
    ccg_data_indent_num: list[tuple[int, str] | str] = [(len(x) - len(x.lstrip()), x.strip())
                                                        if x != "\n"
                                                        else split_token
                                                        for x in ccg_data]

    # split into chunk based on split_token (not included)
    cgg_parse: list[list[tuple[int, str]]] = [[]]

    counter: int = 0

    for line in ccg_data_indent_num:
        if line == split_token:
            counter += 1
            cgg_parse.append([])
        else:
            assert type(line) is tuple
            cgg_parse[counter].append(line)

    # remove empty lists
    return [x for x in cgg_parse if x]


def parse_class(ccg_data: list[list[tuple[int, str]]]) -> dict[int, tree]:
    """
    parse_class Parses the python structure into a tree and leaf class

    Converts the lists of lists with tuples from the parse_data function into a tree structure.
    Exists out of tree and leaf classes.

    Args:
        ccg_data (list[list[tuple[int, str]]]): Python structure after prolog convert

    Returns:
        dict[int, tree]: A dict containing trees where ID is the ccg num.
    """
    # ready spacy and set param info
    ccg_dict: dict[int, tree] = {}

    for ccg_tree in ccg_data:
        current: tree
        num = int(re.findall(r"([\d]+)", ccg_tree[0][1])[0])
        depth, raw_str = ccg_tree[1]

        current = tree(raw_str=raw_str,
                       depth=depth,
                       parent=None,
                       left=None,
                       right=None,
                       tree_ID=num)

        for line in ccg_tree[2:]:
            depth, raw_str = line

            # move current to appropiate level first
            # lazy method, looking up in parents is better
            if depth <= current.depth:
                current = current.find_parent(depth - 1)

            if raw_str[0:2] == "t(":
                current.add_leaf(raw_str)
            else:
                current = current.add_tree(raw_str=raw_str)

        ccg_dict[num] = current.root()
    return ccg_dict
