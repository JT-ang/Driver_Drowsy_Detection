import datetime


class Logger:
    # 定义颜色常量
    C_RESET = '\033[0m'
    C_BLUE = '\033[94m'
    C_RED = '\033[91m'
    C_GREEN = '\033[92m'

    def __init__(self, log_file_path):
        self.res_path = log_file_path
        self.log_buffer = []
        self.fd = None
        self.open()

    def log_info(self, context):
        log_record = "LOG[" + self.C_BLUE + "CLI" + self.C_RESET + "]: " + context
        # current_time = datetime.datetime.now().strftime("%y::%m::%d::%H::%M")
        # rc_in = "[TIME]: " + current_time + " LOG[" + self.COLOR_RED + "WARN" + self.COLOR_RESET + "]: " + context
        # self.log_buffer.append(log_record)
        print(log_record)

    def log_warn(self, context):
        log_record = "LOG[" + self.C_RED + "CLI" + self.C_RESET + "]: " + self.C_RED + context + self.C_RESET
        # current_time = datetime.datetime.now().strftime("%y::%m::%d::%H::%M")
        # rc_in = "[TIME]: " + current_time + " LOG[" + self.COLOR_RED + "WARN" + self.COLOR_RESET + "]: " + context
        # self.log_buffer.append(log_record)
        print(log_record)

    def log_cli(self, context):
        log_record = "LOG[" + self.C_GREEN + "CLI" + self.C_RESET + "]: " + self.C_GREEN + context + self.C_RESET
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
