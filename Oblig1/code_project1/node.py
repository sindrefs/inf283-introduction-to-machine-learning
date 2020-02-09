class Tree(object):
    """
    Basic tree structure found on stackowerflow, later modified
    URL: https://stackoverflow.com/questions/41760856/most-simple-tree-data-structure-in-python-that-can-be-easily-traversed-in-both
    """
    def __init__(self, data, split_on_attribute=None,  children=None, parent=None, error=0, label=None):
        self.data = data
        self.children = children or []
        self.parent = parent
        self.split_on_attribute = split_on_attribute
        self.error = error
        self.label = label

    def add_child(self, new_child):
        """
        Adds a child to the children list
        :param new_child: child as Tree (node representation)
        :return: the new child as Tree (node representation)
        """
        self.children.append(new_child)
        return new_child

    def get_children(self):
        """
        Returns the children list
        :return: the children list as list of Trees (node representation)
        """
        return self.children

    def set_label(self, label):
        """
        Sets label for node
        :param label: label as string
        :return: void
        """
        self.label = label

    def set_was_split_on(self, was_split_on):
        """
        Sets the was_split_on attribute to a given attribute value
        :param was_split_on: the attribute value this node was split on (branch) as string
        :return: void
        """
        self.split_on_attribute = was_split_on

    def inc_error(self):
        """
        Increments the error count by one
        :return: void
        """
        self.error = self.error + 1

    def is_root(self):
        """
        Checks if this node is a root node
        :return: if this node is a root or not as boolean
        """
        return self.parent is None

    def is_leaf(self):
        """
        Checks if this node is a leaf node
        :return: if this node is a leaf node or not as boolean
        """
        return not self.children

    def __str__(self):
        """
        String representation of node
        :return: the representation of the object as a string
        """
        if self.is_leaf():
            return '{data}(E: {error} L: {label})'.format(data=self.data, error=self.error, label=self.label)
        return '{data}(E: {error} L: {label}) [{children}]'.format(data=self.data, error=self.error, label=self.label, children=', '.join(map(str, self.children)))