# iso-privacy-he

Hosts the code necessary to run experiments on HE-Transformer. 

First you need to have build the HE-Transformer: https://github.com/IntelAI/he-transformer/

To run the HE-Transformer experiments do the following steps:
1. Copy the iso-privacy-he directory to $HE_TRANSFORMER_INSTALLATION_DIR$/examples/
```cp -R iso-privacy-he $HE_TRANSFORMER_INSTALLATION_DIR$/examples/```
2. In one tab execute the following: 
```cd $HE_TRANSFORMER_INSTALLATION_DIR$```
```export HE_TRANSFORMER=`pwd` ```
```cd build```
```source external/venv-tf-py3/bin/activate```
```cd $HE_TRANSFORMER/examples/iso-privacy-he```
3. Open another tab and do:
```cd $HE_TRANSFORMER_INSTALLATION_DIR$```
```cd build```
```source external/venv-tf-py3/bin/activate```
```cd $HE_TRANSFORMER/examples/iso-privacy-he```
4. To run MNIST in tab one execute: 
```python3 mnist_server.py --model_file=mnist/models/alice_pool_model.pb --enable_client=true --encryption_parameters=$HE_TRANSFORMER/configs/he_seal_ckks_config_N13_L8.json ```
5. In tab two execute: 
```python3 mnist_client.py --batch_size=1 --encrypt_data_str=encrypt```

You can do the same with the malaria experiments just replace mnist_server with malaria_server, 
mnist_client with malaria_client and put the alice_conv_pool_model.pb as the model name.
