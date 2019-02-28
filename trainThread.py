import threading
class trainThread(threading.Thread):
    def __init__(self,jsondata):
        super(trainThread,self).__init__()
        self.jsondata=jsondata
    def run(self):
        jsondata=self.jsondata
        #train code

        #获取loss，存数据库
        pass