import random
from torch.utils.data import BatchSampler

class BalancedBatchSampler(BatchSampler):
    """
    保證每個 batch 內，每個 writer 出現 samples_per_writer 次
    """
    def __init__(self, dataset, samples_per_writer=2, writers_per_batch=8):
        self.dataset = dataset
        self.samples_per_writer = samples_per_writer
        self.writers_per_batch = writers_per_batch

        # 1) 紀錄每個 writer 擁有的資料索引
        self.indices_by_writer = {w: [] for w in range(dataset.writer_count)}
        for idx, (w, _) in enumerate(dataset.remap_list):      # remap_list 定義在 dataset.py
            self.indices_by_writer[w].append(idx)

        # 2) 為每位 writer 預先 shuffle 索引，之後不斷 pop
        self._refresh_pools()

    def _refresh_pools(self):
        self.pools = {w: random.sample(v, k=len(v))            # 深拷貝並打亂
                      for w, v in self.indices_by_writer.items()}

    def __iter__(self):
        while True:
            # a) 隨機挑 writers_per_batch 個 writer
            writers = random.sample(list(self.indices_by_writer),
                          int(self.writers_per_batch))
            batch = []
            for w in writers:
                # b) 若 pool 不足就重建並再次 shuffle
                if len(self.pools[w]) < self.samples_per_writer:
                    self.pools[w].extend(random.sample(
                        self.indices_by_writer[w],
                        k=len(self.indices_by_writer[w])))
                    random.shuffle(self.pools[w])
                # c) 取 samples_per_writer 個索引
                for _ in range(self.samples_per_writer):
                    batch.append(self.pools[w].pop())
            yield batch

    def __len__(self):
        # 粗略估計：總樣本數 / batch_size
        return len(self.dataset) // (self.samples_per_writer *
                                     self.writers_per_batch)
