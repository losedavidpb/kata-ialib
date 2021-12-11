class GraphGNG(object):

    def __init__(self, identifier, neighborhood=None, age_neighborhood=None):
        if neighborhood is None: neighborhood = []
        if age_neighborhood is None: age_neighborhood = []
        self.__identifier = identifier
        self.__neighborhood = neighborhood
        self.__age_neighborhood = age_neighborhood

    def add_neighbour(self, neighbour, age_neighbour):
        self.__neighborhood.append(neighbour)
        self.__age_neighborhood.append(age_neighbour)

    def remove_neighbour(self, neighbour):
        self.__age_neighborhood.pop(self.__neighborhood.index(neighbour))
        self.__neighborhood.remove(neighbour)

    def increment_age_neighborhood(self, increment):
        self.__age_neighborhood = [age_neighbor + increment for age_neighbor in self.__age_neighborhood]

    def increment_age_neighbor(self, neighbor, increment):
        self.__age_neighborhood[self.__neighborhood.index(neighbor)] += increment

    def set_age(self, neighbour, age):
        self.__age_neighborhood[self.__neighborhood.index(neighbour)] = age

    def prune_graph(self, a_max):
        for neighbour in [n for n, a in zip(self.__neighborhood, self.__age_neighborhood) if a >= a_max]:
            self.remove_neighbour(neighbour)

    @property
    def identifier(self):
        return self.__identifier

    @property
    def neighborhood(self):
        return self.__neighborhood

    @property
    def age_neighborhood(self):
        return self.__age_neighborhood

    @identifier.setter
    def identifier(self, identifier):
        self.__identifier = identifier

    @neighborhood.setter
    def neighborhood(self, neighborhood):
        self.__neighborhood = neighborhood

    @age_neighborhood.setter
    def age_neighborhood(self, age_neighborhood):
        self.__age_neighborhood = age_neighborhood

    def __copy__(self):
        copy_graph = GraphGNG(int(self.identifier))

        for neighborhood, age_neighborhood in zip(self.neighborhood, self.age_neighborhood):
            copy_graph.__neighborhood.append(neighborhood)
            copy_graph.__age_neighborhood.append(age_neighborhood)

        return copy_graph

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if other.identifier != self.__identifier: return False
            if len(other.neighborhood) != len(self.neighborhood): return False

            _other_neighborhood = other.neighborhood.copy()
            _self_neighborhood = self.neighborhood.copy()
            _other_neighborhood.sort()
            _self_neighborhood.sort()

            for elem_other, elem_self in zip(_other_neighborhood, _self_neighborhood):
                if elem_other != elem_self: return False

            if len(other.age_neighborhood) != len(self.age_neighborhood): return False

            _other_age_neighborhood = other.age_neighborhood.copy()
            _self_age_neighborhood = self.age_neighborhood.copy()
            _other_age_neighborhood.sort()
            _self_age_neighborhood.sort()

            for elem_other, elem_self in zip(_other_age_neighborhood, _self_age_neighborhood):
                if elem_other != elem_self: return False

            return True

        return False

    def __ne__(self, other):
        return not self.__eq__(other)
