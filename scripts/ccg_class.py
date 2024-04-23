from __future__ import annotations
from typing import Any, NoReturn, Callable
from collections.abc import Iterator
import re


#  reworked to proper terms and better  documentation
class leaf:
    def __init__(self, raw_str: str, depth: int, tree_ID: int, parent: tree) -> None:
        """
        __init__ Leaf class used to store CCG leafs

        Args:
            raw_str (str): the entire string of the leaf, parsed later into linguistical args
            depth (int): depth of the leaf compared to the root
            tree_ID (int): ID of the entire tree
            parent (tree): Parent node of the leaf
        """
        self.raw_str: str = raw_str
        self.depth: int = depth
        self.tree_ID: int = tree_ID
        self.parent: tree = parent

        # seperate function to split string
        self.filter_raw_str(raw_str)

    def filter_raw_str(self, inp_str: str) -> None:
        """
        filter_str Filter a string from CandC into linguistical parts

        Args:
            inp_str (str): raw input string of CCG tree
        """
        # filters leafs by removing the T and splitting on ,; keeping all inside the quoatation.
        re_string = r"(?<=t\()([^,]*), '((?:[^'\\]|\\.)*)', '((?:[^'\\]|\\.)*)', '((?:[^'\\]|\\.)*)', '((?:[^'\\]|\\.)*)', '((?:[^'\\]|\\.)*)'"
        all_info = re.findall(re_string, inp_str)[0]

        # failsafe for the regex
        assert len(all_info) == 6

        # tags from candc parser
        self.ccg: str = all_info[0]
        self.word: str = all_info[1]
        self.lemma: str = all_info[2]
        self.POS: str = all_info[3]
        self.BIO_pos: str = all_info[4]
        self.BIO_ner: str = all_info[5]

    def find_child(self, *args: Any) -> NoReturn:
        """
        find_child Impossible function, should not be called
        """
        raise RuntimeError("Entered leaf during child search. Child can't have child.")

    def get_sent(self, sent: str = "", lemma: bool = False) -> str:
        """
        Function to parse a subtree to a sentence.
        """
        if lemma:
            return sent + " " + self.lemma
        else:
            return sent + " " + self.word

    def pprint(self, *args: Any) -> str:
        """
        pprint Prints the tree's down in the linux LS format. Info in Tree.
        """
        return f"Leaf: {self.word} ({self.POS})"

    def __repr__(self) -> str:
        return self.pprint()


class tree:
    """Tree structure"""

    def __init__(
        self,
        raw_str: str,
        depth: int,
        parent: "tree | None",
        left: leaf | tree | None,
        right: leaf | tree | None,
        tree_ID: int,
    ) -> None:
        """
        Args:
            raw_str (str): raw string from CCG
            depth (int): Depth of the binary tree, used for print and tree building.
            parent (tree | None): current parent, none if root.
            left (leaf | tree): Left node, never None.
            right (leaf | tree | None): Right node, None if LX.
            tree_ID (int): senID from sen.pl file.
        """
        self.name: str = raw_str
        self.depth: int = depth
        self.parent: tree | None = parent
        self.left: leaf | tree | None = left
        self.right: leaf | tree | None = right
        self.tree_ID = tree_ID
        self.filter_str(raw_str)

    def filter_str(self, raw_str: str) -> None:
        """
        filter_str extracts the action, ccg and lx from the raw string

        finds the action of a CCG line, stored in action
        finds the ccg phrase itself, stored in ccg
        and if the term is LX, stores that in lx

        Args:
            raw_str (str): the raw string from the prolog file
        """
        self.combinator: str
        self.syn_type: str
        self.lx: str = ""

        self.combinator = raw_str.split("(")[0]

        # TODO: replace with regex?
        self.syn_type = raw_str.split(",")[0]
        # remove combinator
        self.syn_type = self.syn_type[self.syn_type.find("(") + 1:]
        # remove :dlc, :ng etc
        self.syn_type_clean = re.sub(":(\w)*", "", self.syn_type)

        # information if LX -> used for child tree searches.
        if raw_str.count(",") == 2:
            self.lx = raw_str.split(",")[-2].strip()

    def add_tree(self, raw_str: str) -> tree:
        """
        add_tree Adds a tree to children, and returns child.
        This function is used to create and go into the child tree,
            assumes depth first binary tree ccg format.
            (which is almost always the case)
        Used for tree building.

        Args:
            raw_str (str): raw_str of the tree to be added.
        """
        if self.left is None:
            self.left = tree(
                raw_str=raw_str,
                parent=self,
                depth=self.depth + 1,
                left=None,
                right=None,
                tree_ID=self.tree_ID,
            )
            return self.left

        elif self.right is None:
            self.right = tree(
                raw_str=raw_str,
                parent=self,
                depth=self.depth + 1,
                left=None,
                right=None,
                tree_ID=self.tree_ID,
            )
            return self.right
        else:
            raise TypeError("Three children found")

    def add_leaf(self, raw_str: str) -> None:
        """
        add_leaf adds a leaf to the current tree
        Used for tree building.

        Args:
            raw_str (str): raw_str of the leaf
        """
        if self.left is None:
            self.left = leaf(raw_str=raw_str, depth=self.depth + 1, tree_ID=self.tree_ID, parent=self)
        elif self.right is None:
            self.right = leaf(raw_str=raw_str, depth=self.depth + 1, tree_ID=self.tree_ID, parent=self)
        else:
            raise TypeError("Error: Three children found")

    def parent_check(self) -> bool:
        """
        parent_check returns true if a parent exists.
        Used for tree building.

        Returns:
            bool: if parent is true
        """
        return self.parent is not None

    def get_parent_tree(self) -> tree:
        """
        get_parent_tree get the tree of the parent, if it doesn't exist, return error.
        Used for tree building.

        Returns:
            tree : pointer
        """
        if not self.parent_check():
            raise NotImplementedError("Calling a parent which is None, use Root to find top tree.")
        else:
            return self.parent

    def find_parent(self, depth: int) -> tree:
        """
        find_parent Find the most recent added tree child node at the appropiate depth.
        Used for tree building.

        Args:
            depth (int): at what depth the tree should find the most recent child

        Returns:
            tree: returns a tree class
        """
        if depth == self.depth:
            return self
        else:
            if self.parent is None:
                raise NotImplementedError("Mismatch between requested depth and search depth")
            return self.parent.find_parent(depth)

    def root(self) -> tree:
        """
        root Returns a pointer to the root of the tree.
        Used for tree building.

        Returns:
            tree: Highest possible parent
        """
        if not self.parent_check():
            return self
        else:
            return self.parent.root()

    def get_sent(self, sent: str = "", lemma: bool = False) -> str:
        """
        Seperate function to get sentence from subtree.
        Lemma is used to indicate if lemmatised sentence should be returned.
        """
        for child in [self.left, self.right]:
            if isinstance(child, tree):
                sent = child.get_sent(sent, lemma=lemma)

            elif isinstance(child, leaf):
                sent = child.get_sent(sent, lemma=lemma)

            else:
                # cases such as LX
                pass
        return sent

    def get_leaves(self, leaf_list: list[leaf] = []) -> list[leaf]:
        """
        get_leaves Function used for subtree generation.


        Args:
            leaf_list (list[leaf], optional): Recursive argument. Defaults to [].

        Returns:
            list[leaf]: Returns a list of child leaf nodes.
        """
        # FIXME: function keeps prior leaf_lists??? Manually needs to assign [] at start???
        for child in [self.left, self.right]:
            if isinstance(child, tree):
                leaf_list = child.get_leaves(leaf_list)
            elif isinstance(child, leaf):
                leaf_list.append(child)
            else:
                # cases such as LX
                pass
        return leaf_list

    def length_check(self, val: int) -> bool:
        """Checks if the length of the subtree sentence split on spaces is above val argument."""
        sent = self.get_sent()
        word_count = sent.count(" ")
        return word_count > val

    def tree_recursive(self, func: Callable[[Any], bool]) -> bool:
        """
        parent_recursive Returns True is function is true once.

        This function iterates through every TREE using a function.
        Acts as an ANY, if true once, returns immediately.
        If not found, return false.

        Args:
            func (Callable[[Any], bool]): function to test on parents

        Raises:
            NotImplementedError: If tree structure is faulty

        Returns:
            bool: If function is once true, returns true. Else false
        """
        if func(self):
            return True

        for child in [self.left, self.right]:
            if isinstance(child, tree):
                return child.tree_recursive(func)

            elif isinstance(child, leaf):
                pass

            elif child is None:
                if self.combinator not in ["lx", "tr"]:
                    raise NotImplementedError("Impossible to reach, children mismatch.")
            else:
                raise NotImplementedError("Impossible to reach, children mismatch.")

        return False

    def leaf_recursive(self, func: Callable[[Any], bool]) -> bool:
        """
        leaf_recursive Returns True is function is true once.

        This function iterates through every leaf using a function.
        Acts as an ANY, if true once, returns immediately.
        If not found, return false.

        Args:
            func (Callable[[Any], bool]): function to test on leaves

        Raises:
            NotImplementedError: If tree structure is faulty

        Returns:
            bool: If function is once true, returns true. Else false
        """
        for child in [self.left, self.right]:
            if isinstance(child, tree):
                if child.leaf_recursive(func):
                    return True

            elif isinstance(child, leaf):
                if func(child):
                    return True

            elif child is None:
                if self.combinator not in ["lx", "tr"]:
                    raise NotImplementedError("Impossible to reach, children mismatch.")
            else:
                raise NotImplementedError("Impossible to reach, children mismatch.")

        return False

    def leaf_clean(self, replace_lst: list[str]):
        """
        leaf_clean Replaces strings in replace list with empty string in leafs
        """
        for child in [self.left, self.right]:
            if isinstance(child, tree):
                child.leaf_clean(replace_lst)

            elif isinstance(child, leaf):
                if child.word in replace_lst:
                    child.word = ""
        return

    def pprint(self, depth: int = 1) -> str:
        """
        print Prints the tree's down in the linux LS format

        prints the trees down using depth in tabs where trees give info about:
            label, left tree and right tree
        and nodes give info about:
            word and semantics

        Args:
            depth (int, optional): Used to print the depth of the lower trees/nodes. Defaults to 0.

        Returns:
            str: returns a string of all the information
        """
        left_str: str
        right_str: str
        if self.left is None:
            left_str = "None"
        else:
            left_str = self.left.pprint(depth + 1)

        if self.right is None:
            right_str = "None"
        else:
            right_str = self.right.pprint(depth + 1)

        tab_str_1: str = "|  " * (depth - 1)
        tab_str: str = "|  " * depth
        return f"Tree: {self.combinator} ({self.syn_type})\n{tab_str}\n{tab_str_1}---{left_str}\n{tab_str}\n{tab_str_1}---{right_str}"

    def gen_subtrees(self) -> Iterator[tree]:
        """
        Generator function that returns all subtrees.
        Preferably used on root.
        Filter functionality is added, filter has to be a callable..
        """
        yield self

        for child in [self.left, self.right]:
            if isinstance(child, tree):
                yield from child.gen_subtrees()

    def __repr__(self) -> str:
        return self.pprint()
