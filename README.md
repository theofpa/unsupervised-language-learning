# unsupervised-language-learning

Request a couple of instances with Tesla V100:

	aws ec2 run-instances --image-id ami-5bf34b22 --count 2 --instance-type p3.2xlarge --key-name theofpa --security-group-ids sg-xxx --subnet-id subnet-xxx

Enable the environment:

	source activate pytorch_p36
	pip install funcy zarr whoosh scipy sklearn
	git clone git@github.com:theofpa/unsupervised-language-learning.git

Get the data and run the training:

	cd unsupervised-language-learning
	wget https://surfdrive.surf.nl/files/index.php/s/Bliv4tIwd7NLAxP/download -O hansards.tgz
	wget https://surfdrive.surf.nl/files/index.php/s/C4QRRulMMX4bdhn/download -O europarl.tgz
	wget https://surfdrive.surf.nl/files/index.php/s/71bLDwNbeTOX1IA/download -O lst.tgz
	tar zxvf hansards.tgz
	tar zxvf europarl.tgz
	tar zxvf lst.tgz
	python SkipGram.py 
