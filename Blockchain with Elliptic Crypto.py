from fastecdsa import keys, curve,ecdsa

class Transaction:

    def __init__(self, sender_address, sender_private_key, recipient_address, value):
        self.sender_address = sender_address
        self.sender_private_key = sender_private_key
        self.recipient_address = recipient_address
        self.value = value

    def __getattr__(self, attr):
        return self.data[attr]

    def to_dict(self):
        return OrderedDict({'sender_address': self.sender_address,
                            'recipient_address': self.recipient_address,
                            'value': self.value})

    def sign_transaction(self):
        """
        Sign transaction with private key
        """
        private_key = keys.gen_keypair(curve.P256)
        signer = PKCS1_v1_5.new(private_key)
        h = SHA.new(str(self.to_dict()).encode('utf8'))
        return binascii.hexlify(signer.sign(h)).decode('ascii')


app = Flask(__name__)

@app.route('/')
def index():
  return render_template('./index.html')

@app.route('/make/transaction')
def make_transaction():
    return render_template('./make_transaction.html')

@app.route('/view/transactions')
def view_transaction():
    return render_template('./view_transactions.html')

    @app.route('/wallet/new', methods=['GET'])
def new_wallet():
  random_gen = Crypto.Random.new().read
  private_key = keys.gen_keypair(curve.P256)
  public_key = private_key.publickey()
  response = {
    'private_key': binascii.hexlify(private_key.exportKey(format='DER')).decode('ascii'),
    'public_key': binascii.hexlify(public_key.exportKey(format='DER')).decode('ascii')
  }

  return jsonify(response), 200

  @app.route('/generate/transaction', methods=['POST'])
def generate_transaction():

  sender_address = request.form['sender_address']
  sender_private_key = request.form['sender_private_key']
  recipient_address = request.form['recipient_address']
  value = request.form['amount']

  transaction = Transaction(sender_address, sender_private_key, recipient_address, value)

  response = {'transaction': transaction.to_dict(), 'signature': transaction.sign_transaction()}

  return jsonify(response), 200

  class Blockchain:

    def __init__(self):

        self.transactions = []
        self.chain = []
        self.nodes = set()
        #Generate random number to be used as node_id
        self.node_id = str(uuid4()).replace('-', '')
        #Create genesis block
        self.create_block(0, '00')

    def register_node(self, node_url):
        """
        Add a new node to the list of nodes
        """
        ...

    def verify_transaction_signature(self, sender_address, signature, transaction):
        """
        Check that the provided signature corresponds to transaction
        signed by the public key (sender_address)
        """
        ...

    def submit_transaction(self, sender_address, recipient_address, value, signature):
        """
        Add a transaction to transactions array if the signature verified
        """
        ...

    def create_block(self, nonce, previous_hash):
        """
        Add a block of transactions to the blockchain
        """
        ...

    def hash(self, block):
        """
        Create a SHA-256 hash of a block
        """
        ...

    def proof_of_work(self):
        """
        Proof of work algorithm
        """
        ...

    def valid_proof(self, transactions, last_hash, nonce, difficulty=MINING_DIFFICULTY):
        """
        Check if a hash value satisfies the mining conditions. This function is used within the proof_of_work function.
        """
        ...

    def valid_chain(self, chain):
        """
        check if a bockchain is valid
        """
        ...

    def resolve_conflicts(self):
        """
        Resolve conflicts between blockchain's nodes
        by replacing our chain with the longest one in the network.
        """
app = Flask(__name__)
CORS(app)
blockchain = Blockchain()
@app.route('/')
def index():
    return render_template('./index.html')

@app.route('/configure')
def configure():
    return render_template('./configure.html')

    @app.route('/transactions/new', methods=['POST'])
    def new_transaction():
        values = request.form

        # Check that the required fields are in the POST'ed data
        required = ['sender_address', 'recipient_address', 'amount', 'signature']
        if not all(k in values for k in required):
            return 'Missing values', 400
        # Create a new Transaction
        transaction_result = blockchain.submit_transaction(values['sender_address'], values['recipient_address'], values['amount'], values['signature'])

        if transaction_result == False:
            response = {'message': 'Invalid Transaction!'}
            return jsonify(response), 406
        else:
            response = {'message': 'Transaction will be added to Block '+ str(transaction_result)}
            return jsonify(response), 201

    @app.route('/transactions/get', methods=['GET'])
    def get_transactions():
        #Get transactions from transactions pool
        transactions = blockchain.transactions

        response = {'transactions': transactions}
        return jsonify(response), 200

    @app.route('/chain', methods=['GET'])
    def full_chain():
        response = {
            'chain': blockchain.chain,
            'length': len(blockchain.chain),
        }
        return jsonify(response), 200

    @app.route('/mine', methods=['GET'])
    def mine():
        # We run the proof of work algorithm to get the next proof...
        last_block = blockchain.chain[-1]
        nonce = blockchain.proof_of_work()

        # We must receive a reward for finding the proof.
        blockchain.submit_transaction(sender_address=MINING_SENDER, recipient_address=blockchain.node_id, value=MINING_REWARD, signature="")

        # Forge the new Block by adding it to the chain
        previous_hash = blockchain.hash(last_block)
        block = blockchain.create_block(nonce, previous_hash)

        response = {
            'message': "New Block Forged",
            'block_number': block['block_number'],
            'transactions': block['transactions'],
            'nonce': block['nonce'],
            'previous_hash': block['previous_hash'],
        }
        return jsonify(response), 200

        @app.route('/nodes/register', methods=['POST'])
        def register_nodes():
            values = request.form
            nodes = values.get('nodes').replace(" ", "").split(',')

            if nodes is None:
                return "Error: Please supply a valid list of nodes", 400

            for node in nodes:
                blockchain.register_node(node)

            response = {
                'message': 'New nodes have been added',
                'total_nodes': [node for node in blockchain.nodes],
            }
            return jsonify(response), 201


        @app.route('/nodes/resolve', methods=['GET'])
        def consensus():
            replaced = blockchain.resolve_conflicts()

            if replaced:
                response = {
                    'message': 'Our chain was replaced',
                    'new_chain': blockchain.chain
                }
            else:
                response = {
                    'message': 'Our chain is authoritative',
                    'chain': blockchain.chain
                }
            return jsonify(response), 200


        @app.route('/nodes/get', methods=['GET'])
        def get_nodes():
            nodes = list(blockchain.nodes)
            response = {'nodes': nodes}
            return jsonify(response), 200
