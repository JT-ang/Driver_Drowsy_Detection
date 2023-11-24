import datetime
class Logger:
    def __init__(self, log_file_path):
        self.res_path = log_file_path
        self.log_buffer = []
        self.fd = None
        self.open()

    def log_info(self, context):
        current_time = datetime.datetime.now().strftime("%y::%m::%d::%H::%M")
        log_record = "[TIME]: " + current_time + " LOG[INFO]: " + context + '\n'
        self.log_buffer.append(log_record)

    def log_warn(self, context):
        current_time = datetime.datetime.now().strftime("%y::%m::%d::%H::%M")
        log_record = "[TIME]: " + current_time + " LOG[WARN]: " + context + '\n'
        self.log_buffer.append(log_record)

    def log_cli(self, context):
        log_record = "LOG[CLI]: " + context
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
