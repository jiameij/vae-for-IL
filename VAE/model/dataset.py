import numpy as np
import random
import sys

class Dset(object):
    def __init__(self, inputs):
        self.inputs = inputs
        self.num_files = len(inputs)
        self.init_file_id()
        self.init_pointer()

    def init_file_id(self): ##标记第几组数据
        self.file_id = 0

    def init_pointer(self):
        self.pointer = 0 ##指向下一个需要从中读取数据的一个文件

    def shuffle(self):
        perm = np.arange(self.num_files)
        np.random.shuffle(perm)
        # shuffle file list
        random.shuffle(self.inputs)
        # self.inputs = self.inputs[perm] ##这个地方这样处理不行
        self._next_id = 0
        self.pointer = 0

    def get_next_batch(self, batch_size, include_final_partial_batch=True): ## 有滑动窗口
        self.init_file_id()
        self.shuffle() ##还是qos整体shuffle???这样做有没有意义呢
        qpos_now = []
        while (self.file_id < self.num_files):
            qpos, qvel = self.inputs[self.file_id]["qpos"][:, self.pointer:self.pointer+batch_size], self.inputs[self.file_id]["qvel"][:, self.pointer:self.pointer + batch_size]
            qpos_now.append((qpos, self.inputs[self.file_id]["qpos"][:,self.pointer + batch_size]))
            self.pointer += 1
            if(self.pointer + batch_size >= self.inputs[self.file_id]["qpos"].shape[1] - 1): ### 留一个做 ob_next
                self.init_pointer()
                self.file_id += 1
        return qpos_now ##只取前1000

        # if batch_size is negative -> return all
        # if batch_size < 0 or self.file_id >= self.num_files:
        #     self.shuffle()
        #     #sys.exit(0)
        #     return False
        # if self.pointer + batch_size >= self.inputs[self.file_id]["qpos"].shape[1] and self.pointer < self.inputs[self.file_id]["qpos"].shape[1] and include_final_partial_batch:
        #     end = self.inputs[self.file_id]["qpos"].shape[1]
        #     qpos, qvel = self.inputs[self.file_id]["qpos"][:, end - batch_size : end], self.inputs[self.file_id]["qvel"][:,
        #                                                                          end - batch_size : end]
        #     self.init_pointer()
        #     self.file_id +=1
        # else:
        #     # end = self.pointer + batch_size
        #     end = self.pointer + 1
        #     # qpos, qvel = self.inputs[self.file_id]["qpos"][:, self.pointer:end], self.inputs[self.file_id]["qvel"][:,
        #     #                                                                      self.pointer:end]  ##这里也要好好改改
        #     qpos, qvel = self.inputs[self.file_id]["qpos"][:, self.pointer:self.pointer + batch_size], self.inputs[self.file_id]["qvel"][:,
        #                                                                          self.pointer:self.pointer + batch_size]  ##这里也要好好改改
        #     self.pointer = end
        # return qpos, qvel


# def iterbatches(arrays, *, num_batches=None, batch_size=None, shuffle=True, include_final_partial_batch=True):
#     assert (num_batches is None) != (batch_size is None), 'Provide num_batches or batch_size, but not both'
#     arrays = tuple(map(np.asarray, arrays))
#     n = arrays[0].shape[0]
#     assert all(a.shape[0] == n for a in arrays[1:])
#     inds = np.arange(n)
#     if shuffle: np.random.shuffle(inds)
#     sections = np.arange(0, n, batch_size)[1:] if num_batches is None else num_batches
#     for batch_inds in np.array_split(inds, sections):
#         if include_final_partial_batch or len(batch_inds) == batch_size:
#             yield tuple(a[batch_inds] for a in arrays)
