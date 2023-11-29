import datetime


class Logger:
    # 定义颜色常量
    COLOR_RESET = '\033[0m'
    COLOR_BLUE = '\033[94m'
    COLOR_RED = '\033[91m'
    COLOR_GREEN = '\033[92m'

    def __init__(self, log_file_path):
        self.res_path = log_file_path
        self.log_buffer = []
        self.fd = None
        self.open()

    def log_info(self, context):
        current_time = datetime.datetime.now().strftime("%y::%m::%d::%H::%M")
        log_record = "[TIME]: " + current_time + " LOG[" + self.COLOR_BLUE + "INFO" + self.COLOR_RESET + "]: " + context
        # self.log_buffer.append(log_record)
        print(log_record)

    def log_warn(self, context):
        log_record = "LOG[" + self.COLOR_RED + "CLI" + self.COLOR_RESET + "]: " + context
        # current_time = datetime.datetime.now().strftime("%y::%m::%d::%H::%M")
        # rc_in = "[TIME]: " + current_time + " LOG[" + self.COLOR_RED + "WARN" + self.COLOR_RESET + "]: " + context
        # self.log_buffer.append(log_record)
        print(log_record)

    def log_cli(self, context):
        log_record = "LOG[" + self.COLOR_GREEN + "CLI" + self.COLOR_RESET + "]: " + context
        print(log_record)

    def flush_buffer(self):
        if self.fd.closed:
            raise ValueError("FileDescriptor Not Open")
        for log in self.log_buffer:
            self.fd.write(log)

    def open(self):
        self.fd = open(self.res_path, 'a')
        self.log_cli("Log Has Opened")

    def close(self):
        self.fd.close()
        self.log_cli("Log Has Closed")
