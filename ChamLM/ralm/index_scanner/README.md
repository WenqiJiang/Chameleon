# GPU Index Scanner

Using the Flat index, with number of vectors = nlist, and the number of results to return = nprobe.

## Performance

Platform: RTX 3090

* dim = 1024, nlist = 32768, nprobe = 32

Typically < 1 ms per search

batch size = 128:
Time: 0.81 ms
QPS: 158790.5684708666

batch size = 64:
Time: 0.76 ms
QPS: 83807.51045894474

batch size = 32:
Time: 0.48 ms
QPS: 66974.91417165668

batch size = 16:
Time: 0.45 ms
QPS: 35772.31556503198