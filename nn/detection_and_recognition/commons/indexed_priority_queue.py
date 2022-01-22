class IndexedMinPQ:
    def __init__(self, N):
        self.N = N
        self.keys_to_idx = {}  # str key -> int index in queue
        self.idxs_to_key = [None for i in range(self.N+1)]  # int index in queue -> str key
        self.pq = [None for i in range(self.N+1)]  # datetimes
        self.total = 0

    def __sink(self, i):
        child_i = i * 2
        while child_i <= self.total:
            value = self.pq[i]
            child_value = self.pq[child_i]
            other_child = child_i + 1
            if other_child <= self.total:
                other_child_value = self.pq[other_child]
                if other_child_value < child_value:
                    child_i = other_child
                    child_value = other_child_value
            if child_value > value:
                break
            self.__exch(i, child_i)
            i = child_i
            child_i = i * 2

    def __swim(self, i):
        parent_i = i//2

        while parent_i > 0:
            value = self.pq[i]
            parent_value = self.pq[parent_i]
            if parent_value < value:
                break

            self.__exch(parent_i, i)

            i = parent_i
            parent_i = i // 2

    def __exch(self, parent_i, i):
        self.pq[i], self.pq[parent_i] = self.pq[parent_i], self.pq[i]
        key, parent_key = self.idxs_to_key[i], self.idxs_to_key[parent_i]
        self.keys_to_idx[key], self.keys_to_idx[parent_key] = self.keys_to_idx[parent_key], self.keys_to_idx[key]
        self.idxs_to_key[i], self.idxs_to_key[parent_i] = self.idxs_to_key[parent_i], self.idxs_to_key[i]

    def peek(self):
        if not self.is_empty():
            out = self.pq[1]
            out_key = self.idxs_to_key[1]

            return out_key, out
        raise IndexError('IndexedMinPQ is Empty')

    def contains(self, key):
        assert type(key) is str
        # assert type(value) is datetime
        return key in self.keys_to_idx

    def insert(self, key, value):
        assert type(key) is str
        # assert type(value) is datetime

        if key in self.keys_to_idx:
            raise IndexError('key is already in the IndexedMinPQ.')
        self.total += 1
        self.keys_to_idx[key] = self.total
        self.idxs_to_key[self.total] = key
        self.pq[self.total] = value
        self.__swim(self.total)

    def del_min(self):
        if not self.is_empty():
            out = self.pq[1]
            out_key = self.idxs_to_key[1]

            self.__exch(1, self.total)

            del self.keys_to_idx[out_key]
            self.idxs_to_key[self.total] = None
            self.pq[self.total] = None
            self.total -= 1

            self.__sink(1)
            return out_key, out
        raise IndexError('IndexedMinPQ is Empty')

    def is_empty(self):
        return self.total == 0

    def is_full(self):
        return self.total == self.N

    def change(self, key, value):
        if key not in self.keys_to_idx:
            raise IndexError('key is not in the IndexedMinPQ')
        assert type(key) is str
        # assert type(value) is datetime
        i = self.keys_to_idx[key]
        self.pq[i] = value
        self.__swim(i)
        self.__sink(i)


# usage sketch
# import time
#
# timestamp0 = datetime.datetime.now()
#
# ipq = IndexedMinPQ(5)
#
# timestamp1 = datetime.datetime.now()
# ipq.insert('a', timestamp1)
# time.sleep(2)
#
# timestamp2 = datetime.datetime.now()
# ipq.insert('b', timestamp2)
# time.sleep(2)
#
# timestamp3 = datetime.datetime.now()
# ipq.insert('c', timestamp3)
#
# ipq.change('c', timestamp0)
# while not ipq.is_empty():
#     key, val = ipq.del_min()
#     print(key, val)

