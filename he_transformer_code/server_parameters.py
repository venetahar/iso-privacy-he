class ServerParameters:

    def __init__(self, batch_size=1, enable_client=False, enable_gc=False, mask_gc_inputs=False, mask_gc_outputs=False,
                 num_gc_threads=1, backend="HE_SEAL", encryption_parameters="", encrypt_server_data=False,
                 pack_data=True, model_file="", input_node="import/input:0", output_node="import/output/BiasAdd:0"):
        self.batch_size = batch_size
        self.enable_client = enable_client
        self.enable_gc = enable_gc
        self.mask_gc_inputs = mask_gc_inputs
        self.mask_gc_outputs = mask_gc_outputs
        self.num_gc_threads = num_gc_threads
        self.backend = backend
        self.encryption_parameters = encryption_parameters
        self.encrypt_server_data = encrypt_server_data
        self.pack_data = pack_data
        self.model_file = model_file
        self.input_node = input_node
        self.output_node = output_node
