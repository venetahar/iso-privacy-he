class ClientParameters:

    def __init__(self, batch_size=1, hostname="localhost", port=34000, encrypt_data_str="encrypt",
                 tensor_name="import/input"):
        self.batch_size = batch_size
        self.hostname = hostname
        self.port = port
        self.encrypt_data_str = encrypt_data_str
        self.tensor_name = tensor_name
