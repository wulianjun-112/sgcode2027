# import time
# from multiprocessing import Process
 
# def f(name):
#     print('hello', name)
#     print('我是子进程')
 
# if __name__ == '__main__':
#     p = Process(target=f, args=('bob',))
#     p.start()
#     time.sleep(1)
#     print('执行主进程的内容了')
# import time
# from multiprocessing import Process
 
# def f(name):
#     print('hello', name)
#     time.sleep(1)
#     print('我是子进程')
 
 
# if __name__ == '__main__':
#     p = Process(target=f, args=('bob',))
#     p.start()
#     p.join()
#     print('我是父进程')

import time
from multiprocessing import Process
 
 
def f(name):
    print('hello', name)
    time.sleep(1)
 
 
if __name__ == '__main__':
    p_lst = []
    for i in range(5):
        p = Process(target=f, args=('bob',))
        p.start()
        p_lst.append(p)