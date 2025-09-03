import os
import json
import subprocess
from web3 import Web3

# Compile the smart contract
def compile_contract():
    print("Compiling Smart Contract...")
    compile_command = "solc --combined-json abi,bin --optimize --overwrite -o . SimpleStorage.sol"
    subprocess.run(compile_command, shell=True, check=True)
    with open("combined.json") as f:
        compiled_contract = json.load(f)
    return compiled_contract

# Deploy the smart contract to each node
def deploy_contract(web3, contract_abi, contract_bytecode, address):
    SimpleStorage = web3.eth.contract(abi=contract_abi, bytecode=contract_bytecode)
    tx_hash = SimpleStorage.constructor().transact({'from': address})
    tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    return tx_receipt.contractAddress

# Connect to Ganache CLI nodes
nodes = [
    {"address": "http://127.0.0.1:8545", "port": 8545},
    {"address": "http://127.0.0.1:8547", "port": 8547},
    {"address": "http://127.0.0.1:8549", "port": 8549}
]

# Get available addresses
addresses = []
for node in nodes:
    web3 = Web3(Web3.HTTPProvider(node["address"]))
    if not web3.is_connected():
        raise Exception(f"Failed to connect to node at {node['address']}")
    accounts = web3.eth.accounts
    addresses.append(accounts[0])

print(f"Generated addresses: {addresses}")

# Compile the contract
compiled_contract = compile_contract()
contract_abi = json.loads(compiled_contract['contracts']['SimpleStorage.sol:SimpleStorage']['abi'])
contract_bytecode = compiled_contract['contracts']['SimpleStorage.sol:SimpleStorage']['bin']

# Deploy the contract on each node
contract_addresses = []
for i, node in enumerate(nodes):
    web3 = Web3(Web3.HTTPProvider(node["address"]))
    if not web3.is_connected():
        raise Exception(f"Failed to connect to node at {node['address']}")
    contract_address = deploy_contract(web3, contract_abi, contract_bytecode, addresses[i])
    contract_addresses.append(contract_address)
    print(f"Contract deployed at {contract_address} on node {i}")

# Save the contract addresses to a file
with open("contract_addresses.json", "w") as f:
    json.dump(contract_addresses, f)

print("Deployed contract addresses saved.")
