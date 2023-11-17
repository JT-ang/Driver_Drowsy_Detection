class Logger:
    def __init__(self, log_file_path):
        self.res_path = log_file_path
        self.log_buffer = []
        self.fd = None

    def log_info(self, context):
        log_record = "LOG[INFO]: " + context
        self.log_buffer.append(log_record)

    def log_warn(self, context):
        log_record = "LOG[WARN]: " + context
        self.log_buffer.append(log_record)

    def log_cli(self, context):
        log_record = "LOG[CLI]: " + context
        print(log_record)

    def flush_buffer(self):
        for log in self.log_buffer:
            self.fd.write(log)

    def open(self):
        self.fd = open(self.res_path,'a')
        self.log_cli("Log Has Opened")

    def close(self):
        self.fd.close()
        self.log_cli("Log Has Closed")
