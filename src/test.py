# import torch.multiprocessing as tmp
# import torch
import multiprocessing as tmp
import numpy as np

def method(pid, q=None):
    np.random.seed()
    # result = np.random.randint(-1, 2)
    result = 1
    if q != None:
        q.put(result)
        return
    return result


if __name__ == '__main__':
    use_mp = True
    cpu_count = tmp.cpu_count()

    # torch.set_num_threads(cpu_count)
    # torch.set_num_interop_threads(cpu_count)
    summing = 0
    queue = tmp.Manager().Queue()
    processes = []

    pool = tmp.Pool(processes=cpu_count)
    for g in range(500):
        if use_mp:
            # for i in range(cpu_count):
            #     p = tmp.Process(target=method, args=(i, queue))
            #     processes.append(p)
            # for p in processes:
            #     p.start()
            # for p in processes:
            #     p.join()
            #     num = queue.get()
            #     summing += num
            #     # del num
            # processes = []
            res = pool.map(method, [i for i in range(cpu_count)])
            pool.map()
            summing += sum(res)
        else:
            z = method(pid=0)
            summing += z

        print(g) 
    print(f'Sum is {summing}')