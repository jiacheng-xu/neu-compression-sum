
# Environment Configuration Guide
# Author: Jiacheng Xu Oct. 2019
# Description: setting up the environment for



sudo apt-get -y update
sudo apt-get install -y build-essential libssl-dev libffi-dev python-dev default-jdk htop zsh python-tk

# Create .bash_profile in your home directory and add these lines:
#export SHELL=/bin/zsh
#exec /bin/zsh -l


wget https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh
sh Anaconda3-5.3.0-Linux-x86_64.sh
rm Anaconda3-5.3.0-Linux-x86_64.sh

#########
# NOT USED. ILP related.
#########
conda install -c conda-forge glpk
conda config --add channels conda-forge
conda install -c cvxgrp cvxpy
conda install nose
wget http://ftp.gnu.org/gnu/glpk/glpk-4.65.tar.gz
tar -xzf glpk-4.65.tar.gz
#CVXOPT_GLPK_LIB_DIR=/home/cc/glpk-4.65/lib
#CVXOPT_GLPK_INC_DIR=/home/cc/glpk-4.65/include
pip install -U cvxopt numpy pip bpython pyrouge cvxpy


conda install -c conda-forge lapack
conda install -c cvxgrp cvxpy


wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip
unzip stanford-corenlp-full-2018-02-27.zip

# My customed version of PYTHONROUGE with some fix of the original version
git clone https://github.com/jiacheng-xu/pythonrouge.git
cd pythonrouge/
python setup.py install
cd pythonrouge/RELEASE-1.5.5/data/
rm WordNet-2.0.exc.db # only if exist
cd WordNet-2.0-Exceptions
rm WordNet-2.0.exc.db # only if exist
./buildExeptionDB.pl . exc WordNet-2.0.exc.db
cd ../
ln -s WordNet-2.0-Exceptions/WordNet-2.0.exc.db WordNet-2.0.exc.db
cd ../../../..

# For running DPLP by Yangfeng Ji
pip2 install -U pip
pip2 install sklearn numpy scipy nltk coloredlogs

pyrouge_set_rouge_path /home/cc/pythonrouge/pythonrouge/RELEASE-1.5.5

# pip install allennlp
git clone https://github.com/abisee/cnn-dailymail.git

#mkdir exComp
#cd exComp
#scp -r jcxu@128.83.143.215:/backup3/jcxu/data/cnndm_vocab ./
#mkdir data
#cd data
#scp -r jcxu@128.83.143.215:/backup3/jcxu/data/1-billion-word-language-modeling-benchmark-r13output.word2vec.vec ./
#cd ../..
#
#mkdir exp
#mkdir log
#mkdir data
#cd data
## scp  jcxu@128.83.143.215:/backup3/jcxu/data/1-billion-word-language-modeling-benchmark-r13output.word2vec.vec ./
#scp -r jcxu@128.83.143.215:/backup3/jcxu/data/2merge-cnndm ./
#
#scp -r jcxu@128.83.143.215:/backup3/jcxu/data/2merge-dm ./
#scp -r jcxu@128.83.143.215:/backup3/jcxu/data/2merge-cnn ./
#
#for file in *.pkl.* ; do mv "$file" "`echo $file | sed 's/\.loaded//'`" ; done
#
#scp -r jcxu@128.83.143.215:/backup3/jcxu/data/cnn-parse.tar.gz ./
#scp -r jcxu@128.83.143.215:/backup3/jcxu/data/dm-parse.tar.gz ./
#
#scp -r jcxu@titan-14.cs.utexas.edu:/scratch/cluster/jcxu/data
#mkdir 2merge-cnndm
#
#cd data/cnn/read_ready-grammarTrue-miniFalse-maxsent30-beam8
#for i in `ls *.pkl.*`; do mv $i ${i}.111; done
#
#cd 2merge-cnn
#cp *.pkl.* ../2merge-cnndm/
#cd ..
#
#cd 2merge-dm
#cp *.pkl.* ../2merge-cnndm/
#cd ../../

#
#source .bashrc
#git clone https://github.com/jiacheng-xu/allennlp.git
#cd allennlp/
#INSTALL_TEST_REQUIREMENTS=true scripts/install_requirements.sh
#pip install --editable .
#cd ..
#
#
#cd lib
#tar -xzvf thesaurus-0.2.3.tar.gz
#cd thesaurus-0.2.3
#python setup.py install
#cd ../../